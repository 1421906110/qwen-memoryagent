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
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("agent.engine")


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

        🔥 省 Token 三件套:
          1. 无工具 → 直接单次 LLM 调用（省循环开销）
          2. 工具缓存 → 相同参数命中直接返回（0 Token）
          3. 提前退出 → LLM 不调工具立即退出（省后续轮次）

        Args:
            messages: 当前消息列表
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
                return TurnResult(
                    reply=turn.get("text", ""),
                    iterations=iterations,
                    tools_called=tools_called,
                )

            # ── 处理工具调用 ──
            for tc in turn["tool_calls"]:
                if self._cancel.is_set():
                    break

                name = tc.get("function", {}).get("name", "")
                try:
                    args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}

                # 权限检查
                allowed = await self._check_permission(name, args)
                if not allowed:
                    continue

                # 🔥 Narration 进度汇报
                if self.narrator:
                    narration = _get_narration(name, args)
                    if narration:
                        await self.narrator(narration)

                # 🔥 缓存检查
                cached = self.cache.get(name, args)
                if cached is not None:
                    result = cached
                    logger.debug("💥 Cache HIT: %s (rate=%.0f%%)",
                                 name, self.cache.stats["hit_rate"] * 100)
                else:
                    result = self.registry.execute("", name, args)
                    self.cache.put(name, args, result)

                tools_called += 1

                # 追加工具结果到消息列表
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": json.dumps(result, ensure_ascii=False, default=str)[:2000],
                })

        # 超轮次保护
        return TurnResult(
            reply="已尽力，但未完成..." if tools_called else "",
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
            return TurnResult(reply=text or "", iterations=1)
        except Exception as e:
            logger.exception("Simple turn failed")
            return TurnResult(reply="抱歉，出错了", iterations=0, error=str(e))

    async def _llm_complete(self, messages: list[dict],
                            temperature: float) -> dict:
        """异步包装 LLM 调用（不阻塞事件循环）"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(
                self.llm.chat_completion,
                messages=messages,
                tools=self.registry.schemas() if self.registry.names else None,
                temperature=temperature,
            ),
        )

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

    @property
    def cache_stats(self) -> dict:
        """工具缓存统计"""
        return self.cache.stats
