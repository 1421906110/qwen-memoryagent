"""
TurnEngine — CogniMem Agent 事件驱动主循环

对标 OpenWorker 的 `coworker/engine.py` TurnEngine。

🔥 CogniMem 相对优化（四更）：
  1. LRU 工具缓存：相同参数命中 → 0 Token
  2. 提前退出：LLM 无工具调用 → 立即返回，不跑满 max_iterations
  3. 简单路径：无工具时一次 LLM 调用直接返回，不进循环
  4. Narration：代码级进度回调（模板映射 <5tok/次，不依赖 LLM）
  5. 可中断：cancel() 让用户随时停，不卡死

架构：
```
用户输入
    │
    ▼
TurnEngine.turn()
    ├─ 简单路径（无工具）→ LLM 1次调用 → 直接返回（省循环）
    │
    └─ 复杂路径（有工具）→ 循环:
          ├─ LLM.complete()
          ├─ 有 tool_calls?
          │   ├─ 权限检查
          │   ├─ 缓存命中? → 0 Token
          │   ├─ 执行 → Narration → 追加结果
          │   └─ 回 LLM 下一轮
          └─ 无 → 提前退出（省后续轮次）
```
"""

from __future__ import annotations

import asyncio
import functools
import json
import re
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("agent.engine")


# ── XML 清洗工具（v0.27: 根治 LLM 输出中的 <tool_calls> 泄漏）──
def _clean_xml(text: str) -> str:
    """清理 LLM 输出中模拟 Claude Code 的 XML tool call 格式"""
    if not text or ('<tool_calls>' not in text and '<invoke' not in text):
        return text
    cleaned = re.sub(r'<tool_calls>.*?</tool_calls>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<invoke name=".*?>.*?</invoke>', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()


class Mode(str, Enum):
    """权限模式"""
    DISCUSS = "discuss"       # 只读：不能写/执行
    INTERACTIVE = "interactive"  # 默认：读自动，写审批
    AUTO = "auto"             # 全放权


class ApprovalOutcome(str, Enum):
    ONCE = "once"
    DENY = "deny"


# 审批回调签名：参数 (tool_name, args) → 返回 ApprovalOutcome
Approver = Callable[[str, dict], Awaitable[ApprovalOutcome]]

# Narration 回调签名：参数 (text) → 异步通知前端
Narrator = Callable[[str], Awaitable[None]]


@dataclass
class TurnResult:
    """一次 turn() 的返回结果"""
    reply: str = ""
    iterations: int = 0
    tools_called: int = 0
    cancelled: bool = False
    truncated: bool = False
    error: Optional[str] = None


# █████████████████████████████████████████████████████████████████████
#  ToolCache — LRU 工具结果缓存（省Token核心）
# █████████████████████████████████████████████████████████████████████

class ToolCache:
    """🔥 LRU 工具结果缓存

    相同工具 + 相同参数 → 命中缓存 → 0 Token
    """

    def __init__(self, maxsize: int = 128):
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0

    def _key(self, name: str, args: dict) -> str:
        """生成缓存键"""
        return f"{name}:{json.dumps(args, sort_keys=True, default=str)}"

    def get(self, name: str, args: dict) -> Optional[Any]:
        """获取缓存，命中返回结果，未命中返回 None"""
        key = self._key(name, args)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, name: str, args: dict, result: Any):
        """写入缓存"""
        key = self._key(name, args)
        self._cache[key] = result
        self._cache.move_to_end(key)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(total, 1), 3),
            "size": len(self._cache),
        }


# █████████████████████████████████████████████████████████████████████
#  Narration — 超轻量进度汇报（<5tok/次）
# █████████████████████████████████████████████████████████████████████

# 🔥 模板映射：代码控制，不依赖 LLM
_NARRATION_MAP = {
    "web_search":  "🔍 正在搜索相关信息...",
    "web_fetch":   "📄 正在抓取网页...",
    "read_file":   "📖 正在读取文件...",
    "write_file":  "📝 正在写入文件...",
    "edit_file":   "✏️ 正在编辑文件...",
    "list_dir":    "📂 正在浏览目录...",
    "shell":       "⚡ 正在执行命令...",
    "memory_recall":    "🧠 正在查找记忆...",
    "memory_remember":  "💾 正在保存记忆...",
    "memory_diagnose":  "🔍 正在检查记忆状态...",
    "memory_status":    "📊 正在查看记忆统计...",
    "memory_forget":    "🗑️ 正在清理记忆...",
}


def _get_narration(tool_name: str, args: dict) -> Optional[str]:
    """🔥 超轻量 Narration：模板映射 + lambda，不依赖 LLM

    Returns:
        narration 文本，或 None（不认识的工具不汇报）
    """
    tpl = _NARRATION_MAP.get(tool_name)
    return tpl


# ═══════════════════════════════════════════════════════════════════════
#  消息压缩块（v0.31，对标 OpenWorker Auto-compaction）
#  ── 机械状态块（零 LLM 零幻觉）+ 结构化摘要（4 段）──
# ═══════════════════════════════════════════════════════════════════════
_MECH_USER_CLIP = 200       # 机械块：单条用户消息截断字符
_MECH_USER_MAX = 10         # 机械块：用户消息上限
_MECH_TOOL_CLIP = 120       # 机械块：单条工具参数截断字符
_MECH_TOOL_MAX = 10         # 机械块：工具操作上限
_SUMMARY_MSG_MAX = 40       # 摘要源：最多取的消息条数（最新优先）
_SUMMARY_MSG_CLIP = 150     # 摘要源：单条消息截断字符
_SUMMARY_MAX_TOKENS = 200   # 摘要输出上限


def _text_of(msg: dict) -> str:
    """消息文本内容（兼容 str / content-parts 列表）"""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def _extract_user_lines(messages: list[dict]) -> list[str]:
    """被裁段里的用户消息原文——意图的 ground truth，不依赖 LLM 记得"""
    out = []
    for m in messages:
        if m.get("role") != "user":
            continue
        text = " ".join(_text_of(m).split())
        if not text:
            continue
        out.append(text[:_MECH_USER_CLIP - 1] + "…" if len(text) > _MECH_USER_CLIP else text)
        if len(out) >= _MECH_USER_MAX:
            break
    return out


def _extract_tool_ops(messages: list[dict]) -> list[str]:
    """被裁段里的工具操作记录（名字 + 关键参数，参数截断防膨胀）"""
    out = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = str(fn.get("name") or "")
            args = fn.get("arguments") or "{}"
            try:
                parsed = json.loads(args)
            except (ValueError, TypeError):
                parsed = {}
            if not isinstance(parsed, dict):  # 标量 JSON（如 '"x"'）防御
                parsed = {}
            key = (parsed.get("path") or parsed.get("file_path")
                   or parsed.get("command") or parsed.get("url") or "")
            line = name + (f" {str(key)[:_MECH_TOOL_CLIP]}" if key else "")
            out.append(line)
            if len(out) >= _MECH_TOOL_MAX:
                break
    return out


def _extract_mechanical_block(messages: list[dict]) -> str:
    """机械状态块：用户原话清单 + 工具操作记录（纯代码提取，零 LLM 零幻觉）"""
    lines = []
    users = _extract_user_lines(messages)
    if users:
        lines.append("【被压缩段的用户原话】")
        lines += [f"- {u}" for u in users]
    ops = _extract_tool_ops(messages)
    if ops:
        lines.append("【被压缩段的工具操作】")
        lines += [f"- {o}" for o in ops]
    return "\n".join(lines)


_SUMMARY_SYSTEM = (
    "你是对话压缩器。把下面的旧对话压缩成结构化摘要，它是模型对该段对话的唯一记忆。\n"
    "只输出以下 4 段，每段一个 markdown 标题：\n"
    "1. 主要请求与约束 — 用户想完成什么，包括任何时候提出的约束"
    "（如「未经批准不要发」），约束比当时更重要\n"
    "2. 关键决策 — 已确定的结论和理由（含 WHY）\n"
    "3. 文件与命令 — 涉及的文件路径和执行的命令（只需路径和用途，"
    "内容让模型需要时重读）\n"
    "4. 待办与下一步 — 未完成事项和紧接着的下一步动作\n"
    "不要复制文件内容，不要复述工具返回的大段结果，不要输出其他内容。"
)


def _summarize_dropped(llm, messages: list[dict]) -> str:
    """LLM 结构化摘要（4 段），失败返回空串（调用方降级为纯机械块）"""
    span = []
    for m in reversed(messages):  # 最新优先收集，保证最新的一定在内
        if m.get("role") not in ("user", "assistant"):
            continue
        text = " ".join(_text_of(m).split())
        if not text:
            continue
        span.append(f"[{m.get('role')}] {text[:_SUMMARY_MSG_CLIP]}")
        if len(span) >= _SUMMARY_MSG_MAX:
            break
    span.reverse()  # 还原时间顺序
    if not span:
        return ""
    try:
        summary = llm.chat(
            messages=[{"role": "system", "content": _SUMMARY_SYSTEM},
                      {"role": "user", "content": "\n".join(span)}],
            system_prompt=None, temperature=0.3, max_tokens=_SUMMARY_MAX_TOKENS,
        )
    except Exception:  # 任何失败 → 空串，调用方降级为纯机械块
        return ""
    return (summary or "").strip()


# █████████████████████████████████████████████████████████████████████
#  TurnEngine — 事件驱动 Agent 主循环
# █████████████████████████████████████████████████████████████████████

class TurnEngine:
    """事件驱动 Agent 主循环

    对标 OpenWorker 的 TurnEngine，但加了 CogniMem 四更优化：
    - LRU 工具缓存（省Token）
    - 提前退出（省Token）
    - 简单路径直接返回（省Token）
    - Narration 进度（<5tok/次）

    用法：
        engine = TurnEngine(llm_client=llm, tool_registry=registry)
        result = await engine.turn(messages, user_message="帮我搜一下...")
    """

    def __init__(
        self,
        *,
        llm_client,
        tool_registry,
        mode: Mode = Mode.INTERACTIVE,
        approver: Optional[Approver] = None,
        narrator: Optional[Narrator] = None,
        max_iterations: int = 5,  # 🔥 5轮上限：平衡质量与速度
        tool_cache: Optional[ToolCache] = None,
    ):
        """
        Args:
            llm_client: LLMClient 实例
            tool_registry: ToolRegistry 实例
            mode: 权限模式
            approver: 审批回调（对话式审批）
            narrator: Narration 进度回调
            max_iterations: 最大循环轮次
            tool_cache: 工具结果缓存（省Token）
        """
        self.llm = llm_client
        self.registry = tool_registry
        self.mode = mode
        self.approver = approver
        self.narrator = narrator
        self.max_iterations = max_iterations
        self.cache = tool_cache or ToolCache()
        # 🆕 v0.25: CogniMem context for memory tools
        self._cogni = None
        self._agent_id = "default"

    def set_cogni_context(self, cogni, agent_id: str = "default"):
        """设置 CogniMem 上下文，供工具调用时传递给 AgentContext"""
        self._cogni = cogni
        self._agent_id = agent_id

        # 中断信号
        self._cancel = asyncio.Event()

    def cancel(self):
        """从任何线程调用 —— 停止当前 turn

        🔥 可打断：用户随时能停，不卡死
        """
        self._cancel.set()
        logger.info("🛑 TurnEngine cancelled")

    # ── 主入口 ──

    async def turn(
        self,
        messages: list[dict],
        user_message: str = "",
        *,
        has_tools: bool = True,
        temperature: float = 0.5,
    ) -> TurnResult:
        """执行一轮 Agent 对话

        🔥 省 Token 五件套:
          1. 无工具 → 直接单次 LLM 调用（省循环开销）
          2. 工具缓存 → 相同参数命中直接返回（0 Token）
          3. 提前退出 → LLM 不调工具立即退出（省后续轮次）
          4. 消息裁剪 → 超预算时自动裁旧工具记录
          5. 并发只读工具 → READ 风险工具并行执行（省等待时间）

        Args:
            messages: 当前消息列表（第一条应为 system prompt）
            user_message: 用户最新消息（日志用）
            has_tools: 是否有工具可用
            temperature: LLM 温度参数

        Returns:
            TurnResult
        """
        self._cancel.clear()
        messages = list(messages)

        # 🔥 简单路径：无工具 → 一次 LLM 调用直接返回
        if not has_tools or not self.registry.names:
            return await self._simple_turn(messages, temperature)

        iterations = 0
        tools_called = 0
        search_count = 0  # web_search 计数
        fetch_count = 0   # web_fetch 计数

        while iterations < self.max_iterations:
            # 中断检查
            if self._cancel.is_set():
                return TurnResult(
                    reply="已中断",
                    iterations=iterations,
                    tools_called=tools_called,
                    cancelled=True,
                )

            iterations += 1

            # 🔥 消息裁剪：超预算时自动裁旧工具记录
            messages = self._prune_messages(messages)

            # ── LLM 调用 ──
            try:
                turn = await self._llm_complete(messages, temperature)
            except Exception as e:
                logger.exception("LLM call failed at iteration %d", iterations)
                return TurnResult(
                    reply=f"抱歉，处理时出错: {str(e)[:100]}",
                    iterations=iterations,
                    tools_called=tools_called,
                    error=str(e),
                )

            # 🔥 提前退出：没有 tool_calls → 直接输出回复
            if not turn.get("tool_calls"):
                _reply = _clean_xml(turn.get("text", ""))
                return TurnResult(
                    reply=_reply or "",
                    iterations=iterations,
                    tools_called=tools_called,
                )
            # 低风险工具（READ）并发跑，高风险（EXEC/WRITE）顺序跑
            tool_calls = turn["tool_calls"]
            cleared = []

            for tc in tool_calls:
                if self._cancel.is_set():
                    break
                name = tc.get("function", {}).get("name", "")
                try:
                    args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                allowed = await self._check_permission(name, args)
                if allowed:
                    cleared.append((tc, name, args))

            # ── 拆分为并发组+顺序组 ──
            if not cleared:
                # 所有工具都被权限拒绝 → 告知 LLM 避免死循环
                messages.append({
                    "role": "user",
                    "content": "【工具被拒绝】你调用的工具在当前模式下不允许，或含危险操作符。请换用其他方式或改用安全命令。",
                })
                continue
            concurrent_group = [x for x in cleared if self._parallel_safe(x[1])]
            serial_group = [x for x in cleared if x not in concurrent_group]

            # ── 追加 assistant tool_calls 消息（DeepSeek 必须配对）──
            _assistant_msg = {
                "role": "assistant",
                "content": turn.get("text", ""),
                "tool_calls": [
                    {"id": tc[0]["id"], "type": "function",
                     "function": {"name": tc[0]["function"]["name"],
                                  "arguments": tc[0]["function"]["arguments"]}}
                    for tc in cleared
                ],
            }
            # ⭐ DeepSeek thinking: 保留 reasoning_content 回传
            if turn.get("reasoning_content"):
                _assistant_msg["reasoning_content"] = turn["reasoning_content"]
            messages.append(_assistant_msg)

            # ── 并发执行只读工具 ──
            if concurrent_group:
                for _, name, args in concurrent_group:
                    if self.narrator:
                        narration = _get_narration(name, args)
                        if narration:
                            await self.narrator(narration)
                loop = asyncio.get_running_loop()
                outcomes = await asyncio.gather(*[
                    loop.run_in_executor(
                        None,
                        functools.partial(self._execute_tool, name, args),
                    )
                    for tc, name, args in concurrent_group
                ])
                for (tc, name, args), result in zip(concurrent_group, outcomes):
                    tools_called += 1
                    self._append_tool_result(messages, tc, name, args, result)
                    # 跟踪搜索次数
                    if name == "web_search":
                        search_count += 1
                    elif name == "web_fetch":
                        fetch_count += 1

            # ── 顺序执行高风险工具 ──
            for tc, name, args in serial_group:
                if self._cancel.is_set():
                    break
                if self.narrator:
                    narration = _get_narration(name, args)
                    if narration:
                        await self.narrator(narration)

                cached = self.cache.get(name, args)
                if cached is not None:
                    result = cached
                    logger.debug("💥 Cache HIT: %s (rate=%.0f%%)",
                                 name, self.cache.stats["hit_rate"] * 100)
                else:
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(
                        None,
                        functools.partial(self._execute_tool, name, args),
                    )
                    self.cache.put(name, args, result)

                tools_called += 1
                self._append_tool_result(messages, tc, name, args, result)
                if name == "web_search":
                    search_count += 1
                elif name == "web_fetch":
                    fetch_count += 1

            # 🔥 搜索超限 → 通知 LLM 停止搜索
            if search_count > 3 or fetch_count > 2:
                messages.append({
                    "role": "user",
                    "content": "【停止搜索】你已经搜索多次了，请根据已有结果直接回复用户。",
                })

        # 超轮次保护
        _last_text = turn.get("text", "") if iterations > 0 else ""
        _last_text = _clean_xml(_last_text) if _last_text else _last_text
        # 🧠 L4 反思：超轮次 → 记录教训（防止下次同样问题）
        if iterations >= self.max_iterations and self._cogni:
            try:
                if hasattr(self._cogni, '_store_lesson'):
                    self._cogni._store_lesson(
                        agent_id=self._agent_id,
                        category="工具错误",
                        summary=f"Agent循环{iterations}轮未完成(超轮次)",
                        details=f"tools_called={tools_called} max_iterations={self.max_iterations}",
                        source="self_reflection",
                    )
            except Exception:
                pass
        return TurnResult(
            reply=_last_text or ("已尽力，但未完成..." if tools_called else ""),
            iterations=iterations,
            tools_called=tools_called,
            truncated=True,
        )

    # ── 内部方法 ──

    async def _simple_turn(self, messages: list[dict],
                           temperature: float) -> TurnResult:
        """🔥 简单路径：不出循环，一次 LLM 调用"""
        try:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(
                None,
                functools.partial(
                    self.llm.chat,
                    messages=messages,
                    temperature=temperature,
                ),
            )
            # 清理 _simple_turn 返回中的 XML
            _text = _clean_xml(text or "")
            return TurnResult(reply=_text or text or "", iterations=1)
        except Exception as e:
            logger.exception("Simple turn failed")
            return TurnResult(reply="抱歉，出错了", iterations=0, error=str(e))

    async def _llm_complete(self, messages: list[dict],
                            temperature: float) -> dict:
        """异步包装 LLM 调用，返回归一化 dict（兼容 ChatCompletion SDK 对象）"""
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            functools.partial(
                self.llm.chat_completion,
                messages=messages,
                tools=self.registry.schemas() if self.registry.names else None,
                temperature=temperature,
            ),
        )
        # ⭐ 归一化：ChatCompletion SDK 对象 → 简单 dict
        msg = response.choices[0].message
        # DeepSeek thinking mode 会返回 reasoning_content，必须保留并回传
        raw_msg = msg.to_dict() if hasattr(msg, 'to_dict') else msg.model_dump() if hasattr(msg, 'model_dump') else {}
        reasoning = raw_msg.get("reasoning_content") or getattr(msg, "reasoning_content", None)
        # 🆕 v0.26: 清理LLM输出中的XML tool_call格式（避免自激循环）
        _text = _clean_xml(msg.content or "")
        return {
            "text": _text,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in (msg.tool_calls or [])
            ],
            "reasoning_content": reasoning,
        }

    async def _check_permission(self, tool_name: str, args: dict) -> bool:
        """权限检查 — 基于 RiskClass 风险分级

        对标 OpenWorker 的 `coworker/permissions.py: PermissionEngine.evaluate()`：

        ┌───────────────┬──────────┬──────────┬──────────┐
        │ RiskClass     │ DISCUSS  │ INTERACT │ AUTO     │
        ├───────────────┼──────────┼──────────┼──────────┤
        │ READ          │ ✅ 允许  │ ✅ 允许  │ ✅ 允许  │
        │ WRITE_LOCAL   │ ❌ 拒绝  │ 需审批   │ ✅ 允许  │
        │ EXEC          │ ❌ 拒绝  │ 需审批   │ ✅ 允许  │
        │ EXTERNAL      │ ❌ 拒绝  │ 需审批   │ ✅ 允许  │
        └───────────────┴──────────┴──────────┴──────────┘

        🔥 对话式审批（vs OpenWorker 前端弹窗）：
        - 返回 DENY 时，调用者追加询问消息到对话
        - 不需要前端弹窗，利用已有 chat 接口做审批
        """
        from .risk import classify as _classify_risk, is_consequential, RiskClass

        risk = _classify_risk(tool_name)

        # AUTO 模式：全部放权
        if self.mode == Mode.AUTO:
            return True

        # DISCUSS 模式：只允许 READ 风险的工具
        if self.mode == Mode.DISCUSS:
            return risk == RiskClass.READ

        # INTERACTIVE 模式：READ 自动允许，其余需审批
        if not is_consequential(risk):
            return True

        # EXEC 工具额外检测 shell 操作符（安全加固）
        if risk == RiskClass.EXEC and tool_name == "shell":
            command = str(args.get("command", ""))
            from .risk import has_shell_operators as _has_ops
            if _has_ops(command):
                logger.warning("⛔ Shell操作符拦截: %s", command[:80])
                return False  # 含 ; & | > 等操作符 → 拒绝

        # 调审批回调
        if self.approver:
            outcome = await self.approver(tool_name, args)
            return outcome == ApprovalOutcome.ONCE

        # 无审批回调 → 默认拒绝
        return False

    # ── 🔥 v0.23 新增方法：消息裁剪 / 并发安全 / 工具执行 ──

    def _prune_messages(self, messages: list[dict],
                         max_tokens: int = 24000,
                         keep_recent: int = 8) -> list[dict]:
        """消息裁剪：超预算时压缩旧消息，保留最近 + system + 首条 user。

        🔥 v0.31 优化（对标 OpenWorker Auto-compaction）：
          1. 机械状态块（零 LLM 零幻觉）：被裁段里的用户消息原文 + 工具操作记录，
             确定性状态不依赖摘要模型
          2. 结构化摘要（4 段）：主要请求与约束 / 关键决策 / 文件与命令 / 待办与下一步
          3. 续跑契约：压缩块末尾明确"继续当前工作"，防止模型复述/重问
          4. 三层降级链：LLM 摘要失败 → 只用机械块 → 机械块为空 → 纯裁剪

        DeepSeek 要求 tool 消息必须跟在对应 tool_calls 消息后，
        裁剪时确保 tool_calls 配对不被破坏。
        """
        if len(messages) <= 2:
            return messages

        # 估算 token（中英文混合）
        def _est(s):
            en = sum(1 for c in str(s) if ord(c) < 128)
            cn = len(str(s)) - en
            return int(en / 4 + cn / 1.5)

        total = sum(_est(m.get("content", "")) for m in messages)
        if total < max_tokens:
            return messages

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        if len(non_system) <= keep_recent:
            return messages

        # 保留：第一条 user + 最近 keep_recent 条
        first_user = None
        for m in non_system:
            if m.get("role") == "user":
                first_user = m
                break

        keep = list(system_msgs)
        if first_user is not None:
            keep.append(first_user)
        for m in non_system[-keep_recent:]:
            if m not in keep:
                keep.append(m)

        # 补全 tool_calls 配对（DeepSeek 要求）
        _tool_ids = {m["tool_call_id"] for m in keep
                     if m.get("role") == "tool" and m.get("tool_call_id")}
        if _tool_ids:
            for m in non_system:
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        if tc.get("id") in _tool_ids and m not in keep:
                            keep.append(m)
                            break

        dropped = len(messages) - len(keep)
        if dropped:
            logger.info("✂️ TurnEngine 裁剪: %d→%d (丢%d条)", len(messages), len(keep), dropped)
            # ── 机械状态块：零 LLM 零幻觉，永远尝试提取 ──
            _dropped_msgs = [m for m in non_system if m not in keep]
            _mech_block = _extract_mechanical_block(_dropped_msgs)
            _block_parts = []
            if _mech_block:
                _block_parts.append(_mech_block)
            # ── 结构化摘要（LLM，失败自动降级为纯机械块）──
            if self.llm:
                try:
                    _summary = _summarize_dropped(self.llm, _dropped_msgs)
                    if _summary:
                        _block_parts.insert(0, _summary)
                except Exception as e:
                    logger.warning("⚠️ 裁剪摘要失败，仅用机械块: %s", str(e)[:100])
            if _block_parts:
                _block_parts.append("继续当前工作，不要复述或重复提问已答内容。")
                keep.insert(1, {"role": "user", "content": "\n\n".join(_block_parts)})
                logger.info("📦 压缩块: 摘要%s + 机械块%s (%d条源消息)",
                            "✅" if len(_block_parts) > 2 else "❌",
                            "✅" if _mech_block else "❌",
                            len(_dropped_msgs))
        return keep

    def _parallel_safe(self, tool_name: str) -> bool:
        """工具是否能并发执行？只有 READ 风险工具可以。

        对标 OpenWorker `_parallel_safe()`：
        低风险（read_file/list_dir/memory_recall/web_search）并行，
        write_file/shell 等顺序执行。
        """
        from .risk import classify as _classify_risk, RiskClass
        return _classify_risk(tool_name) == RiskClass.READ

    def _execute_tool(self, name: str, args: dict) -> dict:
        """同步执行工具，带缓存"""
        cached = self.cache.get(name, args)
        if cached is not None:
            logger.debug("💥 Cache HIT: %s", name)
            return cached
        # 🆕 v0.25: 构建 AgentContext 供 memory tools 使用
        ctx = None
        if self._cogni:
            from . import AgentContext
            ctx = AgentContext(
                agent_id=self._agent_id,
                cogni=self._cogni,
            )
        result = self.registry.execute("", name, args, ctx=ctx)
        self.cache.put(name, args, result)
        return result

    def _append_tool_result(self, messages: list[dict], tc: dict,
                             name: str, args: dict, result: dict):
        """追加工具结果到对话历史（确保不超过上下文窗口）"""
        msg = {
            "role": "tool",
            "tool_call_id": tc.get("id", ""),
            "content": json.dumps(result, ensure_ascii=False, default=str)[:2000],
        }
        messages.append(msg)

    @property
    def cache_stats(self) -> dict:
        """工具缓存统计"""
        return self.cache.stats
