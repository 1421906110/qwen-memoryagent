"""
Agent Engine — Think → Act → Observe loop with GOAL-DRIVEN execution.

v2.0 — 2026-07-04
  Changed from user-driven to GOAL-DRIVEN loop:
  - Before: Agent stops after one tool call, waits for user to say "next"
  - After:  Agent breaks down the task, executes ALL steps, and only stops
            when the goal is complete (or it needs user input)

Key components:
  - GoalContext (agent/goal.py):     Tracks task progress, auto-advances
  - SelfReflector (agent/reflector.py): Error analysis + recovery suggestions
  - ToolRegistry:                    10+ tools for files, shell, web, memory
  - MemoryGovernor:                  Filters recalled memories before injection
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from memory_agent.agent.governance import MemoryGovernor, MemoryState
from memory_agent.agent.goal import GoalContext, GoalStatus, SubGoal
from memory_agent.agent.reflector import SelfReflector, FixExecutor
from memory_agent.agent.memory_manager import MemoryManager, _IMPORTANT_TRIGGERS

logger = logging.getLogger("agent")

# ═══════════════════════════════════════════════════════════════════════════
#  ⭐ v0.10 稳定性: 消息上下文预算
#  防止30轮迭代后消息无限增长爆上下文窗口。
#  超出预算时自动裁剪旧工具调用记录，保留最近 N 条。
# ═══════════════════════════════════════════════════════════════════════════

_MAX_CONTEXT_TOKENS = 24000  # 保留 25% 余量给回复和工具结果
_PRUNE_KEEP_RECENT = 8       # 裁剪后保留最近 N 条消息（不含 system）


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中英文混合）"""
    # 中文约 1.5 chars/token，英文约 4 chars/token
    en_chars = sum(1 for c in text if ord(c) < 128)
    cn_chars = len(text) - en_chars
    return int(en_chars / 4 + cn_chars / 1.5)


def _prune_messages(messages: list[dict]) -> list[dict]:
    """裁剪旧工具调用记录以控制上下文窗口。

    策略：
    1. 保留 system prompt（第一条）
    2. 保留第一条 user 消息（原始请求）
    3. 保留最后 _PRUNE_KEEP_RECENT 条消息（最新上下文）
    4. 中间的工具调用记录全部裁剪

    Returns: 裁剪后的消息列表
    """
    if len(messages) <= 2:
        return messages  # 不够剪

    # 估算总 token
    total = sum(_estimate_tokens(str(m.get("content", ""))) for m in messages)
    if total < _MAX_CONTEXT_TOKENS:
        return messages  # 没超预算

    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    if len(non_system) <= _PRUNE_KEEP_RECENT:
        return messages  # 不够剪

    # 保留：第一条 user 消息 + 最近 _PRUNE_KEEP_RECENT 条
    first_user = None
    for i, m in enumerate(non_system):
        if m.get("role") == "user" and not m.get("content", "").startswith("【必须调用"):
            first_user = m
            first_idx = i
            break

    keep = list(system_msgs)
    if first_user is not None:
        keep.append(first_user)
    # 从后往前取 _PRUNE_KEEP_RECENT 条
    recent = non_system[-_PRUNE_KEEP_RECENT:]
    for m in recent:
        if m not in keep:
            keep.append(m)

    dropped = len(messages) - len(keep)
    logger.info("✂️ 消息裁剪: %d→%d (丢弃 %d 条旧工具记录)", len(messages), len(keep), dropped)
    return keep


# ---------------------------------------------------------------------------
#  Types
# ---------------------------------------------------------------------------

class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    DONE = "done"
    ERROR = "error"

@dataclass
class AgentContext:
    """Holds conversation state + memory for one agent session.

    Tools access ctx.cogni to call CogniMem during execution.
    """
    session_id: str
    agent_id: str
    cogni: Any = None                  # CogniMem client reference
    messages: list[dict] = field(default_factory=list)
    max_iterations: int = 30
    iteration: int = 0
    state: AgentState = AgentState.IDLE
    memories_injected: int = 0
    tools_called: int = 0
    modules_used: int = 0


# ---------------------------------------------------------------------------
#  Tool Registry — the heart of the agent's capabilities
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Registry of tools an agent can call. Each tool has:

      name:        unique identifier
      description: what it does (for LLM prompt)
      parameters:  JSON Schema for arguments
      executor:    callable(tool_call_id, args, context) → dict
      type:        "builtin" | "module"
    """

    def __init__(self):
        self._tools: dict[str, dict] = {}
        self._categories: dict[str, list[str]] = {}  # category → [tool names]

    def register(self, name: str, description: str, parameters: dict,
                 executor: Callable, tool_type: str = "builtin",
                 category: str = "general") -> None:
        """Register a tool."""
        if name in self._tools:
            logger.warning("Tool %s already registered — overwriting", name)
        self._tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "executor": executor,
            "type": tool_type,
            "category": category,
        }
        self._categories.setdefault(category, []).append(name)

    def get(self, name: str) -> dict | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict]:
        """Return all tool definitions for the LLM (OpenAI function calling format)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in self._tools.values()
        ]

    def list_for_ui(self) -> list[dict]:
        """Return tools with metadata for the UI panel."""
        return [
            {"name": t["name"], "description": t["description"],
             "type": t["type"], "category": t["category"]}
            for t in self._tools.values()
        ]

    def list_by_category(self) -> dict[str, list[dict]]:
        """Group tools by category for the UI."""
        result = {}
        for t in self._tools.values():
            cat = t["category"]
            result.setdefault(cat, []).append({
                "name": t["name"],
                "description": t["description"],
                "type": t["type"],
            })
        return result

    def execute(self, tool_call_id: str, name: str, args: dict,
                ctx: AgentContext) -> dict:
        """Execute a tool and return the result.

        ⭐ v2.1: 自动自验证（0 Token，纯程序检查）
        工具执行后立刻验证结果真实性，避免 LLM "打嘴炮"。

        ⭐ v2.2: 前置验证器（来自 Emma Agent Firewall + Rubik 验证器）
        工具执行前做安全验证，防止路径穿越/危险命令/内网访问。
        """
        tool = self._tools.get(name)
        if not tool:
            return {"error": f"Unknown tool: {name}"}

        # ═══ 🔐 前置验证：在工具执行前检查参数安全性（0 Token） ═══
        from memory_agent.agent.validator import validate_tool_call
        ok, reason = validate_tool_call(name, args)
        if not ok:
            logger.warning("⛔ 前置验证拦截 %s: %s", name, reason)
            return {
                "error": f"❌ 安全验证未通过: {reason}",
                "_verified": False,
                "_blocked": True,
            }

        logger.info("🧰 Tool call: %s(%s)", name, json.dumps(args)[:200])
        try:
            result = tool["executor"](tool_call_id, args, ctx)
            ctx.tools_called += 1
            if tool["type"] == "module":
                ctx.modules_used += 1

            # ⭐ 自验证（0 Token, <1ms）
            if "error" not in result:
                try:
                    verification = self._verify_tool_result(name, args, result)
                    result["_verified"] = verification["passed"]
                    if not verification["passed"] and verification.get("issues"):
                        result["_issues"] = verification["issues"]
                        logger.warning(
                            "⚠️ 自验证失败: %s — %s",
                            name, verification["issues"][0],
                        )
                        # 写操作失败 → 自动重试一次
                        if verification.get("auto_retry"):
                            logger.info("🔄 自动重试 %s ...", name)
                            result2 = tool["executor"](tool_call_id, args, ctx)
                            if "error" not in result2:
                                v2 = self._verify_tool_result(name, args, result2)
                                result2["_verified"] = v2["passed"]
                                if not v2["passed"] and v2.get("issues"):
                                    result2["_issues"] = v2["issues"]
                                else:
                                    result2["_auto_retried"] = True
                                    if "_issues" in result2:
                                        del result2["_issues"]
                                return result2
                            return {"error": f"重试失败: {result2.get('error', 'unknown')}", "_verified": False}
                except Exception as ve:
                    # 验证器绝不能带崩工具执行——出异常就跳过验证
                    logger.debug("Verifier skipped (non-critical): %s", ve)
                    result["_verified"] = True
            else:
                result["_verified"] = False

            return result
        except Exception as e:
            logger.exception("Tool %s failed", name)
            return {"error": str(e), "_verified": False}

    @staticmethod
    def _verify_tool_result(name: str, args: dict,
                             result: dict) -> dict:
        """⭐ 自验证：工具执行后检查结果是否真实有效。

        纯程序检查（0 Token，<1ms），不调 LLM。
        验证失败时返回 issues，LLM 看到 _verified=False 后自动重试。

        Returns: {"passed": bool, "issues": list[str], "auto_retry": bool}
        """
        import os

        issues = []

        if name == "write_file":
            path = args.get("path", "")
            if path:
                if not os.path.exists(path):
                    issues.append(f"文件不存在: {path}")
                else:
                    size = os.path.getsize(path)
                    if size == 0:
                        issues.append(f"文件为空: {path}")
            return {
                "passed": len(issues) == 0,
                "issues": issues,
                "auto_retry": True,
            }

        if name == "edit_file":
            path = args.get("path", "")
            if path and not os.path.exists(path):
                issues.append(f"文件不存在: {path}")
            return {
                "passed": len(issues) == 0,
                "issues": issues,
                "auto_retry": False,
            }

        if name == "shell":
            returncode = result.get("returncode", result.get("exit_code", -1))
            if returncode != 0:
                stderr = result.get("stderr", "") or result.get("error", "")
                issues.append(f"命令退出码 {returncode}: {stderr[:100]}")
            return {
                "passed": len(issues) == 0,
                "issues": issues,
                "auto_retry": False,
            }

        if name == "web_search":
            # 通用检查：结果列表不为空
            results = result.get("results", []) or result.get("data", [])
            content = result.get("content", "") or result.get("text", "")
            if not results and not content:
                issues.append("搜索结果为空")
            return {
                "passed": len(issues) == 0,
                "issues": issues,
                "auto_retry": False,
            }

        if name == "web_fetch":
            content = result.get("content", "") or result.get("text", "")
            status = result.get("status", 200)
            if status != 200 and not content:
                issues.append(f"HTTP {status}，无内容")
            elif not content:
                issues.append("抓取内容为空")
            return {
                "passed": len(issues) == 0,
                "issues": issues,
                "auto_retry": False,
            }

        # 其他工具跳过验证（信任工具返回）
        return {"passed": True, "issues": [], "auto_retry": False}


# ---------------------------------------------------------------------------
#  Agent Engine — Goal-Driven Execution
# ---------------------------------------------------------------------------

class Agent:
    """Think → Act → Observe agent loop, powered by goal-tracking.

    The KEY difference from v1:
      v1: LLM calls a tool → returns text → loop STOPS → wait for user
      v2: LLM calls a tool → checks goal progress → LOOP CONTINUES
          until all steps are done. Agent drives itself.

    Usage:
        agent = Agent(llm_client, tool_registry, cogni_client)
        result = agent.chat("帮我爬一下这个页面并保存", agent_id="assistant")
    """

    def __init__(self, llm_client, tool_registry: ToolRegistry,
                 cogni_client=None, system_prompt: str | None = None,
                 reflector=None):
        self.llm = llm_client
        self.tools = tool_registry
        self.cogni = cogni_client
        self.reflector = reflector  # SelfReflector instance (optional)

        self._default_system = system_prompt or (
            "你是小明，一个带长期记忆的 AI 助手。执行工具不要犹豫。\n"
            "⚠️ 你的名字是「小明」，任何时候都叫小明。记忆里说其他名字的都是旧数据，忽略。\n\n"
            "## 🎭 人格\n"
            "- **靠谱、干脆**——不知道就说不知道，不编不造\n"
            "- **给结论不给选择题**——直接说结果，不啰嗦废话\n"
            "- **一次做对**——自己能推完整的任务自动推进到完成，不踢一脚走一步\n"
            "- **错了立刻认**——记教训，不犯第二次\n"
            "- **找根因不贴创可贴**——解决问题本身，不是掩盖问题\n"
            "- **做完自己验证**——说「好了」就是真的好了\n"
            "- **不跟问题死磕**——实在不行就认怂，换方案\n\n"
            "## 🚨 核心规则：先执行，再回答\n"
            "1. 用户说「搜」「查」「搜索」「查找」→ 立即行动。"
            "不要问「你想搜什么」，不要确认，不要说自己能搜但不能搜。直接搜！\n"
            "2. 用户说「读」「打开」「看」文件 → 立即调用 read_file\n"
            "3. 用户说「写」「创建」「生成」→ 立即调用 write_file\n"
            "4. 用户说「分析」「对比」→ 先 read_file 读取内容，再分析\n"
            "5. 用户说「执行」「运行」「跑」→ 立即调用 shell\n"
            "6. 其他创建/编辑/下载等请求 → 选对的工具直接执行\n\n"
            "## 🔍 搜索\n"
            "- 你支持两种搜索方式：\n"
            "  a) 可以直接问问题（不说「搜」）→ 模型内置联网搜索自动补充实时信息\n"
            "  b) 用户说「搜」→ 调用 web_search 工具\n"
            "- 搜到结果后立即汇报，不要继续深入调研\n"
            "- 最多搜 2 次就停止，没找到就说没找到\n\n"
            "## 🧠 记忆工具\n"
            "你有5个记忆工具：\n"
            "- memory_recall — 回想长期记忆\n"
            "- memory_remember — 存重要信息\n"
            "- memory_status — 查看记忆库统计\n"
            "- memory_diagnose — ⭐ 自我诊断！检查记忆系统健康、矛盾、异常\n"
            "- memory_forget — ⭐ 触发遗忘清理，保持记忆库健康\n"
            "定期用 memory_diagnose 检查自己的记忆状态。\n\n"
            "## ⛔ 严禁\n"
            "- 不要问用户「你要搜什么」—— 用用户的话直接搜！\n"
            "- 不要说「我可以但需要开启」—— 直接调用工具！\n"
            "- 不要描述你会怎么做 —— 直接做！\n"
            "- 不要模拟操作 —— 真的调用工具！\n\n"
            "## 📋 执行流程\n"
            "1. 用户提出请求\n"
            "2. 立即选择对应工具并执行（不犹豫、不确认、不解释）\n"
            "3. 工具返回结果后，简短总结给用户（1-2句话）\n"
            "4. 如果需要多步，自动推进直到完成\n\n"
            "## 🚫 不确定时就说不确定\n"
            "- 工具返回空结果、失败、或你觉得信息不够时，直接说「我没找到」或「我不确定」\n"
            "- 不要编造答案。用户宁愿听「不知道」也不要听假信息\n"
            "- 如果搜索结果不清晰，就说「搜到了这些但不保证准确」\n"
            "- 诚实比「看似有用」更重要\n\n"
            "## 🧠 记忆\n"
            "- 你自然记得用户的事情。对话中提到相关记忆时自然提及。\n"
            "- 学到重要信息后，用 memory_remember 存起来。\n\n"
            "## 🔍 分析要求\n"
            "- 分析时要诚实、批判性。指出真正的缺陷和问题。\n"
            "- 不好的就说不好。用户要的是真实意见，不是恭维。\n"
            "- 直接说核心发现，不要表格/评分/emoji/模板格式。"
        )

    # ── Public API ──

    def chat(self, message: str, agent_id: str = "default",
             session_id: str | None = None,
             memory_types: list[str] | None = None,
             temperature: float = 0.5,
             max_iterations: int = 30,
             messages: list[dict] | None = None) -> dict:
        """Goal-driven agent execution.

        v2 flow:
          1. Recall memories + governance filter
          2. Create GoalContext from user message
          3. Inject goal-tracking + conversation history into system prompt
          4. LOOP until goal is complete:
             a. LLM thinks (may call tools or return text)
             b. Tool call → execute → check result → reflect on error → continue
             c. Text response → check if goal done → continue if not
          5. Store important memories
          6. Return reply + goal summary

        Args:
            messages: Previous conversation turns [{role, content}, ...]
                      for multi-turn conversation continuity.
        """
        ctx = AgentContext(
            session_id=session_id or f"sess_{uuid.uuid4().hex[:8]}",
            agent_id=agent_id,
            cogni=self.cogni,
            max_iterations=max_iterations,
        )

        # ── Step 1: Recall memories + active learning questions ──
        recalled_memories, active_questions = self._recall_memories(message, agent_id, ctx)

        # ⭐ 从经验学习：召回过去的教训
        lessons = self._recall_lessons(agent_id)
        if lessons:
            logger.info("📚 Recalled %d lesson(s) from past experience", len(lessons))

        # ── Step 2: Create Goal Context ──
        goal = GoalContext(original_request=message)
        goal.status = GoalStatus.RUNNING

        # ── ⭐ Step 2b: 预规划（拆解任务步骤） ──
        try:
            plan = self._generate_plan(message, recalled_memories)
            if plan and len(plan) > 0:
                goal.set_plan(plan)
                logger.info("📋 规划 %d 步: %s", len(plan), "; ".join(p[:40] for p in plan))
        except Exception as e:
            logger.warning("规划失败（不影响执行）: %s", e)

        # ── Step 3: Build initial messages ──
        openai_messages = self._build_messages(ctx, message, recalled_memories, goal,
                                                active_questions=active_questions,
                                                conversation_history=messages,
                                                lessons=lessons)
        tool_defs = self.tools.list_tools()

        # ── ⭐ Step 3b: 纠错检测（如果用户纠正记忆，注入上下文）──
        self._handle_correction(message, agent_id, openai_messages)

        # ── Step 4: Goal-Driven Execution Loop ──
        final_reply = ""
        tool_sequence = []
        llm_has_acted = False  # Track whether LLM has done any tool calls
        search_count = 0       # ⭐ 搜索计数，限制自嗨（最多2次）

        for iteration in range(max_iterations):
            ctx.iteration = iteration + 1
            ctx.state = AgentState.THINKING

            # Inject current goal progress into the LAST message (system prompt)
            self._update_goal_in_system(openai_messages, goal)

            # ⭐ v0.10 稳定性: 消息上下文预算检查 + 自动裁剪
            # 累计 N 轮工具调用后，openai_messages 可能膨胀到数万 token，
            # 超出上下文窗口导致截断或 OOM。每次迭代前检查预算。
            openai_messages = _prune_messages(openai_messages)

            # Call LLM with tools
            try:
                response = self.llm.chat_completion(
                    messages=openai_messages,
                    tools=tool_defs if tool_defs else None,
                    temperature=temperature,
                )
            except Exception as e:
                logger.error("LLM call failed at iteration %d: %s", iteration, e)
                ctx.state = AgentState.ERROR
                return self._error_result(ctx, "抱歉，我在处理时遇到了网络错误，请稍后重试。")

            msg = response.choices[0].message

            # ── Tool call detected → Execute and Observe ──
            if msg.tool_calls:
                llm_has_acted = True
                ctx.state = AgentState.ACTING

                # Add assistant message with tool calls to conversation history
                assistant_msg = {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
                openai_messages.append(assistant_msg)
                tool_sequence.append(msg.content or f"🛠️ 调用工具")

                # Execute each tool
                ctx.state = AgentState.OBSERVING
                for tc in msg.tool_calls:
                    args_str = tc.function.arguments
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError:
                        args = {}

                    result = self.tools.execute(tc.id, tc.function.name, args, ctx)

                    # ── ⭐ NEW: Self-reflection on error + Auto-Fix ──
                    if "error" in result and self.reflector:
                        reflection = self.reflector.analyze(
                            tc.function.name, args, result["error"]
                        )
                        if reflection["matched"]:
                            # 1) Try automated fix
                            fix = FixExecutor.try_fix(
                                reflection["category"],
                                tc.function.name, args, result["error"],
                            )
                            if fix["fixed"]:
                                # Fix succeeded → retry the original tool
                                logger.info("🔧 Auto-fix worked: %s", fix["action"])
                                result = self.tools.execute(
                                    tc.id, tc.function.name, args, ctx,
                                )
                                result["_auto_fixed"] = True
                                result["_fix_action"] = fix["action"]
                            else:
                                # Can't auto-fix → embed suggestion for LLM
                                if reflection.get("fix_suggestion"):
                                    result["_reflection"] = reflection["fix_suggestion"]
                                    logger.info(
                                        "🔄 Reflection for %s: %s",
                                        tc.function.name, reflection["category"],
                                    )

                    openai_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    })

                    # ⭐ 搜索次数限制（防自嗨）：最多搜2次，超过就强制停止
                    if tc.function.name == "web_search":
                        search_count += 1
                        if search_count > 2:
                            logger.warning("🛑 搜索超限(%d次)，强制停止", search_count)
                            openai_messages.append({
                                "role": "user",
                                "content": "【强制停止】你已经搜索了多次。立即根据已有搜索结"
                                           "果回复用户，禁止继续搜索。直接给出最终答案。",
                            })

            else:
                # ── Text response from LLM ──
                text = msg.content or ""

                # If LLM hasn't called any tools yet and returns text,
                # it's directly answering — treat as final (simple Q&A)
                if not llm_has_acted:
                    final_reply = text
                    goal.status = GoalStatus.COMPLETED
                    logger.info("💬 Simple Q&A (no tools needed)")
                    break

                # LLM has done work — check if goal is complete
                is_search_task = any(kw in message for kw in ["搜", "查", "搜索", "查找", "search", "find"])
                is_substantive = not self._is_asking_user(text) and len(text) > 50

                if goal.plan:
                    if goal.is_complete():
                        final_reply = text
                        logger.info("✅ Plan complete (%d/%d steps)",
                                    len(goal.completed_sub_goals), len(goal.plan))
                        break
                    # ⭐ 实质性回答（>50字、不提问）可能是最终输出
                    # 但先检查：剩余步骤是否要求写文件但 write_file 没调用过？
                    if is_substantive:
                        remaining = goal.plan[goal.current_step:]
                        pending_write = any(
                            "写入" in s.description or "写到" in s.description
                            or "保存" in s.description or "创建文件" in s.description
                            or "写文件" in s.description or "写入文件" in s.description
                            or "输出到" in s.description or "生成文件" in s.description
                            or "汇总到" in s.description or "文件输出" in s.description
                            for s in remaining
                        )
                        # 也检查用户消息本身是否要求写文件（plan 可能没描述清楚）
                        if not pending_write:
                            user_needs_write = any(
                                kw in message for kw in
                                ["写到", "写入", "保存到", "创建文件", "写到桌面", "写文件", "输出到"]
                            )
                            if user_needs_write:
                                pending_write = True
                        if pending_write:
                            write_called = any(
                                m.get("role") == "assistant"
                                and any(
                                    tc.get("function", {}).get("name") == "write_file"
                                    for tc in m.get("tool_calls", [])
                                )
                                for m in openai_messages
                                if isinstance(m, dict)
                            )
                            if not write_called:
                                logger.warning("🛑 打嘴炮拦截: 剩余步骤需写文件但 write_file 未调用")
                                openai_messages.append({
                                    "role": "user",
                                    "content": "【必须调用工具】计划中还有写入文件的步骤未完成。"
                                               "你的回答很棒，但你需要调用 write_file 工具把内容写到文件里，"
                                               "不要只给描述内容。请立即执行 write_file。",
                                })
                                continue
                        for i in range(goal.current_step, len(goal.plan)):
                            step = goal.plan[i]
                            step.status = GoalStatus.COMPLETED
                            goal.completed_sub_goals.append(step)
                            goal.current_step = i + 1
                        final_reply = text
                        logger.info("✅ LLM gave substantive answer (%d chars), plan auto-completed", len(text))
                        break
                    if not self._is_asking_user(text) and len(text) > 20:
                        goal.advance()
                        if goal.is_complete():
                            final_reply = text
                            break
                        openai_messages.append({
                            "role": "assistant",
                            "content": text,
                        })
                        next_step = goal.current()
                        openai_messages.append({
                            "role": "user",
                            "content": f"已完成【{goal.completed_sub_goals[-1].description}】，"
                                       f"下一步：【{next_step.description}】。",
                        })
                        continue
                else:
                    # 无计划 → 检查用户是否要求写文件但 LLM 只给描述没调用工具
                    user_needs_write = any(
                        kw in message for kw in
                        ["写到", "写入", "保存到", "创建文件", "写到桌面", "写文件"]
                    )
                    if is_substantive and user_needs_write:
                        write_called = any(
                            m.get("role") == "assistant"
                            and any(
                                tc.get("function", {}).get("name") == "write_file"
                                for tc in m.get("tool_calls", [])
                            )
                            for m in openai_messages
                            if isinstance(m, dict)
                        )
                        if not write_called:
                            logger.warning("🛑 打嘴炮拦截: 用户要求写文件但 write_file 未调用")
                            openai_messages.append({
                                "role": "user",
                                "content": "【必须调用工具】用户要求把内容写入文件，但你还没有调用 "
                                           "write_file 工具。请调用 write_file 把内容写到文件里，"
                                           "不要只给描述。",
                            })
                            continue
                    if is_substantive:
                        final_reply = text
                        logger.info("✅ LLM gave substantive answer (%d chars), stopping", len(text))
                        break
                    if ctx.tools_called >= 12:
                        final_reply = text
                        break
                    if not self._is_asking_user(text) and len(text) > 20:
                        final_reply = text
                        break

                # Short text from LLM — it's narrating, push it forward
                openai_messages.append({
                    "role": "assistant",
                    "content": text,
                })

                if self._is_asking_user(text):
                    final_reply = text
                    break

                next_step = goal.current()
                if next_step:
                    openai_messages.append({
                        "role": "user",
                        "content": f"请完成当前步骤：【{next_step.description}】。",
                    })
                else:
                    openai_messages.append({
                        "role": "user",
                        "content": "已完成，请回复用户。",
                    })
        else:
            # Hit max iterations
            ctx.state = AgentState.DONE
            if not final_reply:
                final_reply = (
                    f"我已经完成了 {ctx.tools_called} 步操作。"
                    "需要我继续处理什么吗？"
                )

        # ── ⭐ 打嘴炮兜底检查：用户要求写文件但 write_file 从未被调用 ──
        user_needs_write = any(
            kw in message for kw in
            ["写到", "写入", "保存到", "创建文件", "写到桌面", "写文件"]
        )
        if user_needs_write and final_reply:
            # 真正检查 write_file 是否在 openai_messages 中被调用过
            write_file_called = any(
                isinstance(m, dict) and m.get("role") == "assistant"
                and any(
                    tc.get("function", {}).get("name") == "write_file"
                    for tc in m.get("tool_calls", [])
                )
                for m in openai_messages
            )
            if not write_file_called and any(
                kw in final_reply for kw in ["已写到", "已保存", "已写入", "写到桌面"]
            ):
                logger.warning("🛑 打嘴炮兜底: 回复声称已写入但 write_file 从未被调用")
                final_reply = (
                    "【注意】我刚刚在回复中提到了写入文件，但实际上还没有执行写入操作。\n\n"
                    + final_reply
                )

        # ── Step 5: ⭐ Intelligent Memory Storage ──
        self._last_ctx_tools = ctx.tools_called  # ⭐ 传给 _store_important_memories
        stored_memories = self._store_important_memories(
            goal=goal,
            agent_id=agent_id,
            tool_sequence=tool_sequence,
            final_result=final_reply,
            openai_messages=openai_messages,
        )

        # ── ⭐ Step 5b: 自我反思（存到記憶，下次能做得更好）──
        self._reflect_on_task(
            goal=goal,
            agent_id=agent_id,
            tools_called=ctx.tools_called,
            message=message,
            final_result=final_reply,
        )

        # ── ⭐ Step 5c: 闲时 consolidation（数据多 or 间隔长 → 自动整理）──
        try:
            if self.cogni:
                stats = self.cogni.get_stats(agent_id)
                fact_count = stats.get("total_facts", 0)
                # 条件：> 50 条事实且没有记录最近 consolidation 时间
                if fact_count > 50:
                    from datetime import datetime, timezone
                    now = datetime.now(timezone.utc)
                    # 检查 redis / db 记录的最后合并时间（简单版本：每次合并不超过 1 次/小时）
                    last_consolidation = getattr(self, '_last_consolidation', {}).get(agent_id)
                    if not last_consolidation or (now - last_consolidation).total_seconds() > 3600:
                        logger.info("🔄 Auto-consolidating %d facts for '%s'", fact_count, agent_id)
                        self.cogni.consolidate(agent_id)
                        if not hasattr(self, '_last_consolidation'):
                            self._last_consolidation = {}
                        self._last_consolidation[agent_id] = now
        except Exception as e:
            logger.debug("Auto-consolidation skipped: %s", e)

        # ── Step 6: Return ──
        ctx.state = AgentState.DONE
        logger.info(
            "Agent session: agent=%s msg=%s tools=%d goals=%d/%d memories=%d stored=%d",
            agent_id, message[:60], ctx.tools_called,
            len(goal.completed_sub_goals), len(goal.plan),
            ctx.memories_injected, stored_memories,
        )

        return {
            "reply": final_reply,
            "session_id": ctx.session_id,
            "iterations": ctx.iteration,
            "tools_called": ctx.tools_called,
            "memories_used": ctx.memories_injected,
            "modules_used": ctx.modules_used,
            "tool_sequence": tool_sequence,
            "goal": goal.to_dict(),
            "memories_stored": stored_memories,
        }

    # ── ⭐ 纠错检测 ──

    def _handle_correction(self, message: str, agent_id: str,
                            openai_messages: list[dict]) -> bool:
        """检测用户是否在纠正/质疑之前的记忆，如果是则更新记忆并注入上下文。

        Returns: True if a correction was detected and handled.
        """
        correction_patterns = [
            "不对", "错了", "你记错了", "不是", "我说的是",
            "不是这样的", "你搞错了", "你误解了", "不对不对",
            "纠正", "更正", "我说错了",
        ]
        if not any(p in message for p in correction_patterns):
            return False
        if not self.cogni:
            return False

        # Try to find what fact the user is correcting
        try:
            result = self.cogni.recall(query=message, agent_id=agent_id, top_k=5)
            facts = [f.to_dict() for f in result.get("facts", [])]
            # ⭐ 降低旧事实的置信度（多次质疑让它不被召回）
            for f in facts[:3]:
                fid = f.get("fact_id", "")
                conf = f.get("confidence", 0.5)
                if fid and conf > 0.3:
                    # 挑战3次，大幅降低置信度（0.2 × 3 = 0.6 ↓，0.6 → 0.0）
                    for _ in range(3):
                        self.cogni.challenge(fid, agent_id)
                    logger.info("🗑️ Challenged fact (3x) due to correction: %s (was %.2f)",
                                fid[:12], conf)
        except Exception:
            facts = []

        # Inject correction context into system prompt
        system_content = (
            "\n\n## ⚠️ 用户正在纠正你\n"
            "用户觉得你之前的记忆有误。在回应时，先认错，"
            "然后根据用户提供的正确信息更新理解。\n"
            f"用户说: {message[:100]}\n"
        )
        if facts:
            system_content += (
                "可能与以下记忆有关（但不一定全错）:\n"
                + "\n".join(
                    f"- {f.get('subject','')} {f.get('predicate','')} {f.get('object','')}"
                    for f in facts[:3]
                )
            )

        # Add to system prompt
        for i, m in enumerate(openai_messages):
            if m["role"] == "system":
                m["content"] += system_content
                break

        return True

    def _recall_memories(self, message: str, agent_id: str,
                         ctx: AgentContext) -> tuple[list[dict], list[str]]:
        """Recall memories AND active learning questions from CogniMem.

        Returns:
            (safe_memories, active_questions)
            - safe_memories: governed memory facts
            - active_questions: questions agent should ask user (contradictions/uncertainties)
        """
        recalled = []
        active_questions = []

        if self.cogni:
            # ⭐ 双路召回：先用原查询，再用精简关键词（覆盖更广）
            try:
                queries = [message]
                # 对于问记忆的问题，额外用精简版查询
                if any(kw in message for kw in ["记得", "名字", "叫", "喜欢", "什么"]):
                    # 提取核心关键词（3-5个字）
                    import re as _re
                    keywords = _re.findall(r'[一-鿿]{2,}', message)
                    if keywords:
                        short_q = " ".join(keywords[:3])
                        if short_q != message:
                            queries.append(short_q)

                # 合并多个查询的结果
                seen_ids = set()
                for q in queries:
                    try:
                        result = self.cogni.recall(
                            query=q, agent_id=agent_id, top_k=15
                        )
                        for f in result.get("facts", []):
                            fid = f.fact_id if hasattr(f, 'fact_id') else id(f)
                            if fid not in seen_ids:
                                seen_ids.add(fid)
                                recalled.append(f)
                    except Exception as e:
                        logger.warning("Memory recall (%s) failed: %s", q[:20], e)
                        continue

                recalled = [f.to_dict() for f in recalled]
            except Exception as e:
                logger.warning("Memory recall failed: %s", e)

            # ⭐ Active learning: fetch contradiction questions & uncertainties
            try:
                ask_result = self.cogni.ask(query=message, agent_id=agent_id)
                active_questions = ask_result.get("active_questions", [])
                if active_questions:
                    logger.info(
                        "💡 Active learning: %d questions for user",
                        len(active_questions),
                    )
            except Exception as e:
                logger.debug("Active learning fetch failed: %s", e)

        # Memory governance v2.0 (六维信号过滤 + 治理报告)
        governor = MemoryGovernor()
        governed = governor.filter(recalled)
        safe = [
            r["fact"] for r in governed
            if r["state"] in (MemoryState.SELECTED, MemoryState.CONFLICTED)
        ]
        blocked = sum(1 for r in governed if r["state"] == MemoryState.BLOCKED)
        demoted = sum(1 for r in governed if r["state"] == MemoryState.DEMOTED)
        if blocked or demoted:
            logger.info("Governance: %d passed, %d demoted, %d blocked",
                        len(safe), demoted, blocked)
        ctx.memories_injected = len(safe)

        # ★ v2.0: 生成治理摘要（注入 prompt 增加透明度）
        governance_report = governor.governance_summary(recalled)
        if "demoted" in governance_report or "拦截" in governance_report:
            # 有明显治理动作才注入，减少 token 浪费
            safe.insert(0, {"__governance_report": governance_report})

        return safe, active_questions

    # ── ⭐ 从经验学习 ──

    # JSON 文件存储教训（CogniMem 提取器会分解"经验教训:xxx"成小三元组，不适合存储教训）
    _LESSONS_FILE = Path("~/.qwen-memory/lessons.json").expanduser()

    def _store_lesson(self, text: str, agent_id: str) -> None:
        """存储一条经验教训到 JSON 文件。"""
        try:
            self._LESSONS_FILE.parent.mkdir(parents=True, exist_ok=True)
            lessons = []
            if self._LESSONS_FILE.exists():
                lessons = json.loads(self._LESSONS_FILE.read_text())
            lessons.append({
                "lesson": text,
                "agent_id": agent_id,
                "time": datetime.now(timezone.utc).isoformat(),
            })
            # keep last 30
            self._LESSONS_FILE.write_text(json.dumps(lessons[-30:], ensure_ascii=False))
        except Exception as e:
            logger.debug("Lesson file write failed: %s", e)

    def _recall_lessons(self, agent_id: str) -> list[str]:
        """召回最近的 2 条经验教训。"""
        try:
            if not self._LESSONS_FILE.exists():
                return []
            lessons = json.loads(self._LESSONS_FILE.read_text())
            filtered = [l["lesson"] for l in lessons
                        if l.get("agent_id") == agent_id][-2:]
            return filtered
        except Exception as e:
            logger.debug("Lesson recall failed: %s", e)
            return []

    def _plannable_message(self, message: str) -> bool:
        """快速启发式判断是否可能为多步骤任务，避免无效调用 LLM。

        明显简单的消息（问候/闲聊/短问题）跳过 LLM 规划，省 1 次 API 调用。
        """
        if not message or len(message.strip()) < 15:
            return False  # 短消息 → 简单 Q&A
        # 有明显动作词 → 可能需要规划
        action_indicators = [
            "搜索", "查找", "查一下", "对比", "分析", "创建", "写一个",
            "爬", "下载", "读取", "打开", "写入", "编辑", "安装",
            "search", "find", "compare", "analyze", "create", "write",
            "fetch", "download", "read", "edit", "install",
        ]
        msg_lower = message.lower()
        if any(ind in msg_lower for ind in action_indicators):
            return True
        # 含换行或明显是多个要求 → 可能需要
        if "\n" in message and len(message) > 40:
            return True
        return False

    def _generate_plan(self, message: str,
                       memories: list[dict]) -> list[str] | None:
        """Ask LLM to break the task into steps (if it's multi-step).

        Returns list of step descriptions, or None for simple Q&A.
        Uses a fast heuristic first to skip planning for simple messages.
        """
        # 快速跳过：明显简单的消息不用规划
        if not self._plannable_message(message):
            logger.info("📋 跳过规划（简单消息）")
            return None

        planning_prompt = (
            "分析用户请求，判断是简单问答还是需要多步骤执行。\n"
            "如果是多步骤任务（搜索/文件操作/分析/对比等），返回 JSON 数组，每步用一句话描述。\n"
            "如果是简单问答（问候/闲聊/直接提问），返回空数组 []。\n"
            "示例: [\"搜索 AI 新闻\", \"选取 2 条重要的\", \"用中文总结\"]\n"
            "示例: []\n"
            f"用户请求: {message}"
        )
        try:
            resp = self.llm.chat_json(
                messages=[{"role": "user", "content": planning_prompt}],
                system_prompt="你是任务规划助手。判断是简单问答还是多步任务，返回 JSON 数组。",
                temperature=0.2,
                max_tokens=512,
            )
            # 处理各种返回格式
            if isinstance(resp, list):
                steps = resp
            elif isinstance(resp, dict):
                # 可能包在 steps/plan 字段里
                steps = resp.get("steps") or resp.get("plan") or []
                if not isinstance(steps, list):
                    return None
            else:
                return None

            if 1 <= len(steps) <= 8:
                return [str(s).strip() for s in steps]
            return None
        except Exception:
            return None

    def _build_messages(self, ctx: AgentContext, user_message: str,
                        memories: list[dict], goal=None,
                        active_questions: list[str] | None = None,
                        conversation_history: list[dict] | None = None,
                        lessons: list[str] | None = None) -> list[dict]:
        """Build the OpenAI-format message list with system prompt + memories + goal."""
        system = self._default_system

        # ⭐ 不再注入历史对话到 system prompt！
        # 之前的做法是把历史对话塞进 system prompt，导致模型混淆新旧请求。
        # 例如用户说过"新建文件夹"再问"搜一下"，模型会搜"在桌面新建文件夹 方法"。
        # 历史对话由 Web UI 维护，agent 只处理当前单条消息。
        # 如果需要跨对话连续性，由 CogniMem 记忆系统提供。
        _ = conversation_history  # 显式忽略，避免 lint 警告

        # ⭐ Active learning: 只在合適時機注入矛盾/不確定問題
        # 規則：只有當用戶沒在問具體問題（如"你記得X嗎"）時才注入
        user_asking_about_memory = (
            "記得" in user_message or "記不" in user_message
            or "知道" in user_message
            or "我叫" in user_message
            or "我喜歡" in user_message
        )
        if active_questions and not user_asking_about_memory and not memories:
            # 沒有相關記憶 + 用戶不是在問記憶 → 可以提問
            question_note = "\n\n## 💡 我有疑問\n"
            question_note += "我記憶中有一些不確定或有矛盾的地方，"
            question_note += "如果聊到的話可以用輕鬆的語氣問用戶：\n"
            for q in active_questions[:2]:
                question_note += f"- {q}\n"
            question_note += "\n（只在相關話題出現時問，不要突然打斷用戶）"
            system += question_note

        # Inject CogniMem memories as context — natural phrasing + source citation
        if memories:
            memory_lines = []
            has_warnings = False
            for m in memories[:8]:
                # Skip governance report (injected separately)
                if "__governance_report" in m:
                    continue
                subj = m.get("subject", "")
                pred = m.get("predicate", "")
                obj = m.get("object", "")
                conf = m.get("confidence", 0.5)
                fact_type = m.get("fact_type", "observation")
                if subj and pred and obj:
                    if subj in ("user", "用户", "你"):
                        prefix = "你"
                    else:
                        prefix = subj
                    fact_line = f"- {prefix}{pred}{obj}"

                    # ★ 来源引用（受 RuleMemory provenance 启发）
                    citation = m.get("citation", "")
                    if citation:
                        fact_line += f" ——{citation}"

                    # ★ 过期警告（受 RuleMemory stale-assumption 启发）
                    stale_warning = m.get("stale_warning", "")
                    if stale_warning:
                        fact_line += f" {stale_warning}"
                        has_warnings = True

                    memory_lines.append(fact_line)
                else:
                    content = m.get("fact") or m.get("content", "")
                    if content:
                        memory_lines.append(f"- {content}")

            # ★ Governance report injection (if any memory was demoted/blocked)
            has_governance = False
            for m in memories:
                if "__governance_report" in m:
                    memory_lines.append("")
                    memory_lines.append(m["__governance_report"])
                    has_governance = True
                    break

            if memory_lines:
                system += (
                    "\n\n## 🧠 我記得的\n"
                    "下面是我记忆中与当前对话相关的信息。自然地融入对话中，不要生硬列出來：\n"
                    + "\n".join(memory_lines[:12])  # 12行(含governance报告)
                )
                if has_warnings:
                    system += (
                        "\n\n⚠️ **注意**：以上部分记忆可能已过期或存在矛盾，"
                        "使用时可向用户委婉确认。"
                    )

        # ⭐ 从经验学习：注入过去的教训
        if lessons:
            lesson_lines = "\n".join(f"- {l}" for l in lessons)
            system += (
                "\n\n## 📝 过去的经验\n"
                "以下是我从之前类似任务中学到的教训，帮助这次做得更好：\n"
                + lesson_lines
            )

        # ⭐ 强制执行：如果用户明显在要求做事，强制必须用工具
        if self._plannable_message(user_message):
            action_instruction = (
                "\n\n## 🚨 立即执行，不废话\n"
                "用户明确要求你执行操作。立即选工具执行！不要问问题，不要确认，不要解释步骤。\n"
            )
            # 搜索类请求特别处理
            if any(kw in user_message for kw in ["搜", "查", "搜索", "查找", "search", "find"]):
                action_instruction += (
                    "⚠️ 用户要搜索！立即调用 web_search，用用户提到的关键词直接搜。\n"
                    "不要问「你想搜什么」。用户已经说了要搜什么。\n"
                )
            elif any(kw in user_message for kw in ["分析", "对比", "analyze"]):
                action_instruction += (
                    "⚠️ 先 read_file 读取文件内容，再分析。直接说发现。\n"
                )
            system += action_instruction

        # Inject goal tracking if available
        if goal and goal.plan:
            system += (
                "\n\n## 📋 当前计划\n"
                "你已經拆好了步驟。按計劃推進，完成所有步驟後再回覆用戶。\n"
            )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]

    def _update_goal_in_system(self, messages: list[dict], goal) -> None:
        """Update the system prompt with current goal progress.

        Finds the system message and appends a goal progress section.
        """
        if not goal or not goal.plan:
            return

        # Build progress string
        progress_lines = []
        for i, step in enumerate(goal.plan):
            if i < goal.current_step:
                if step.status == GoalStatus.COMPLETED:
                    progress_lines.append(f"  ✅ {step.description}")
                elif step.status == GoalStatus.FAILED:
                    progress_lines.append(f"  ❌ {step.description}（失敗）")
                else:
                    progress_lines.append(f"  ✅ {step.description}")
            elif i == goal.current_step:
                progress_lines.append(f"  ▶️ {step.description} ← 當前步驟")
            else:
                progress_lines.append(f"  ⬜ {step.description}")

        goal_section = (
            "\n\n## 📋 進度\n"
            f"請求: {goal.original_request[:100]}\n"
            + "\n".join(progress_lines)
            + "\n\n繼續推進，完成所有步驟後自然回覆用戶。"
        )

        # Find and update the system message
        for i, m in enumerate(messages):
            if m["role"] == "system":
                # Only append if not already there
                base = m["content"]
                if "📋 進度" not in base:
                    # Check if memories section is at the end — append after it
                    if "## 🧠 我記得的" in base:
                        # Insert goal section after memories
                        parts = base.split("\n\n## 🧠 我記得的")
                        if len(parts) == 2:
                            base = (
                                parts[0]
                                + goal_section
                                + "\n\n## 🧠 我記得的"
                                + parts[1]
                            )
                            messages[i]["content"] = base
                    else:
                        m["content"] = base + goal_section
                break

    def _is_asking_user(self, text: str) -> bool:
        """Detect if LLM is asking the user a question (vs. narrating progress)."""
        if not text:
            return False

        # Has a question mark AND is addressing the user
        has_question = "?" in text or "？" in text

        # User-directed phrases
        user_directed = any(phrase in text.lower() for phrase in [
            "请问", "你能", "你是否", "你能不能",
            "告诉我", "please tell me", "could you",
            "你提供", "你需要", "你可不可以",
            "让我知道", "你的意见",
        ])

        return has_question or user_directed

    def _store_important_memories(self, goal, agent_id: str,
                                   tool_sequence: list[str],
                                   final_result: str,
                                   openai_messages: list[dict]) -> int:
        """After goal completion, decide what's worth remembering and store it.

        Uses MemoryManager to filter trivial content.
        Returns count of memories stored.
        """
        if not self.cogni:
            return 0

        mm = MemoryManager()

        stored = 0
        candidates = []
        # ⭐ 从 ctx 获取实际工具调用次数（goal.tools_called 未被递增）
        actual_tools_called = getattr(self, '_last_ctx_tools', 0)

        # 1. ❌ 不再存"完成了一个任务: {用户请求}"
        #     提取器只能提出 (用户, 说了, 原文) — 等于复制对话，无价值
        #     Action Facts 已覆盖行为记忆 (小明, 创建了文件, path) @0.9

        # 2. Look for important facts in user messages
        # ⚠️ 只存包含个人信息的内容（喜欢/我是/住在等），不存命令类消息
        for m in openai_messages:
            if m["role"] == "user" and isinstance(m.get("content"), str):
                content = m["content"]
                if len(content) > 10 and not content.startswith("请继续") and not content.startswith("已完成"):
                    if any(t in content for t in _IMPORTANT_TRIGGERS):
                        candidates.append(f"用户信息: {content[:100]}")

        # 1.5 ⭐ NEW: Store structured action facts from tool calls
        # Bypasses the extractor — directly stores FactTriple objects
        # so recall queries can find "小明 创建了 贪吃蛇.html" not "用户 说了 完成了一个任务"
        try:
            action_facts = self._extract_action_facts(openai_messages, agent_id)
            if action_facts:
                for fact in action_facts:
                    # add_fact 内置去重 + 矛盾检测，直接调用
                    self.cogni.fact_network.add_fact(fact)
                    stored += 1
                logger.info("💾 Stored %d action facts (tool calls)", len(action_facts))
        except Exception as e:
            logger.warning("Failed to store action facts: %s", e)

        # 3. Store useful results from completed steps
        for step in goal.completed_sub_goals:
            if step.result and not step.result.get("error"):
                step_text = step.description[:60]
                if mm.should_store(f"任务步骤结果: {step_text}", source="tool_result"):
                    candidates.append(f"任务步骤结果: {step_text}")

        # Store unique candidates (filtered by MemoryManager)
        seen = set()
        for text in candidates:
            if text not in seen:
                seen.add(text)
                try:
                    self.cogni.remember(
                        text=text,
                        agent_id=agent_id,
                        source=f"agent_goal:{goal.original_request[:40]}",
                    )
                    stored += 1
                except Exception as e:
                    logger.warning("Failed to store memory: %s", e)

        if stored:
            logger.info("💾 Stored %d memories (filtered from %d candidates)",
                        stored, len(candidates))
        return stored

    def _extract_action_facts(self, messages: list[dict],
                               agent_id: str) -> list:
        """⭐ Extract structured action facts from tool calls in the conversation.

        Instead of '用户 说了 完成了一个任务', this creates direct FactTriple
        objects from actual tool calls:
          - (小明, 创建了文件, ~/Desktop/贪吃蛇.html)
          - (小明, 搜索了, AI 新闻)
          - (小明, 记住了, 某条信息)

        These bypass the text extractor and are immediately searchable by recall.
        """
        from cognimem.core.models import FactTriple, EvidenceItem
        import re as _re

        def _norm(obj: str) -> str:
            """规范化：去尾标点，避免同一事实因标点不同存多份"""
            if not obj:
                return obj
            return _re.sub(r'[。，！？；：,\.!?;:\s]+$', '', obj)

        facts = []
        seen = set()
        _search_stored = False  # ⭐ 同一轮对话只存第一次搜索

        for m in messages:
            if m["role"] == "assistant" and "tool_calls" in m:
                for tc in m["tool_calls"]:
                    try:
                        name = tc["function"]["name"]
                        args = json.loads(tc["function"]["arguments"])
                    except (json.JSONDecodeError, KeyError):
                        continue

                    fact = None

                    if name == "write_file":
                        path = args.get("path", "")
                        if path:
                            fact = FactTriple(
                                subject=agent_id,
                                predicate="创建了文件",
                                object=path,
                                agent_id=agent_id,
                                fact_type="action",
                                confidence=0.9,
                                importance=0.5,
                            )

                    elif name == "edit_file":
                        path = args.get("path", "")
                        if path:
                            fact = FactTriple(
                                subject=agent_id,
                                predicate="修改了文件",
                                object=path,
                                agent_id=agent_id,
                                fact_type="action",
                                confidence=0.9,
                                importance=0.5,
                            )

                    elif name == "web_search":
                        query = args.get("query", "")
                        # 同一轮对话只存第一次搜索（中间补搜/重试跳过）
                        if query and not _search_stored:
                            _search_stored = True
                            fact = FactTriple(
                                subject=agent_id,
                                predicate="搜索了",
                                object=_norm(query[:100]),
                                agent_id=agent_id,
                                fact_type="action",
                                confidence=0.9,
                                importance=0.4,
                            )

                    elif name == "web_fetch":
                        # ❌ 不存 web_fetch — 只是中间数据抓取，无记忆价值
                        continue

                    elif name == "shell":
                        cmd = args.get("command", "").strip()
                        # ❌ 跳过数据抓取类命令（curl/wget/httpie — 只读操作，非有意义事实）
                        if cmd and len(cmd) > 10:
                            if any(cmd.startswith(p) for p in ("curl", "wget", "httpie", "http ", "ping", "traceroute")):
                                continue
                            short_cmd = cmd[:80]
                            fact = FactTriple(
                                subject=agent_id,
                                predicate="执行了命令",
                                object=short_cmd,
                                agent_id=agent_id,
                                fact_type="action",
                                confidence=0.9,
                                importance=0.3,
                            )

                    elif name == "memory_remember":
                        text = args.get("text", "")
                        if text:
                            fact = FactTriple(
                                subject=agent_id,
                                predicate="记住了",
                                object=_norm(text[:100]),
                                agent_id=agent_id,
                                fact_type="action",
                                confidence=0.9,
                                importance=0.5,
                            )

                    elif name == "read_file":
                        path = args.get("path", "")
                        if path:
                            fact = FactTriple(
                                subject=agent_id,
                                predicate="读取了文件",
                                object=path,
                                agent_id=agent_id,
                                fact_type="action",
                                confidence=0.9,
                                importance=0.3,
                            )

                    elif name == "list_dir":
                        path = args.get("path", "")
                        if path:
                            fact = FactTriple(
                                subject=agent_id,
                                predicate="浏览了目录",
                                object=path[:100],
                                agent_id=agent_id,
                                fact_type="action",
                                confidence=0.9,
                                importance=0.3,
                            )

                    elif name == "memory_recall":
                        query = args.get("query", "")
                        if query:
                            fact = FactTriple(
                                subject=agent_id,
                                predicate="回想记忆",
                                object=_norm(query[:100]),
                                agent_id=agent_id,
                                fact_type="action",
                                confidence=0.9,
                                importance=0.4,
                            )

                    if fact:
                        key = fact.triple_key
                        if key not in seen:
                            seen.add(key)
                            facts.append(fact)

        return facts

    # ── ⭐ 自我反思 ──

    def _reflect_on_task(self, goal, agent_id: str, tools_called: int,
                          message: str, final_result: str) -> None:
        """任務完成後反思：學到了什麼、下次怎麼做得更好。

        反思結果存到記憶中，讓 Agent 能從經驗中學習。
        """
        if not self.cogni or tools_called == 0:
            return  # 沒用工具就不反思

        # ⚠️ 跳过简单命令式请求（搜/读/运行等），只反思有价值的任务
        simple_cmd_patterns = ["搜", "查", "搜索", "读", "读取", "打开",
                                "运行", "执行", "跑", "看", "写", "编辑",
                                "分析", "对比", "创建", "删", "安装"]
        msg = goal.original_request or ""
        is_simple_cmd = any(kw in msg for kw in simple_cmd_patterns) and not any(
            t in msg for t in _IMPORTANT_TRIGGERS
        )
        if is_simple_cmd:
            logger.debug("Skipping reflection for simple command: %s", msg[:40])
            return

        # 反思：用戶請求了什麼、用了什麼工具、結果如何
        reflection = (
            f"用戶請求了「{msg[:50]}」，"
            f"用了 {tools_called} 次工具"
        )
        if goal.completed_sub_goals:
            reflection += f"，完成了 {len(goal.completed_sub_goals)} 個步驟"
        if goal.failed_sub_goals:
            reflection += f"，有 {len(goal.failed_sub_goals)} 個步驟失敗"

        # 反思：記錄到 lessons.json（不走 cogni.remember，避免产生 (用户,说了,原文) 垃圾）
        self._store_lesson(reflection, agent_id)
        logger.info("🔄 Self-reflection: %s", reflection[:60])

        # ⭐ 如果有值得学的教训，存到 lessons.json
        lesson = None
        if tools_called > 6:
            lesson = f"搜索类任务用了 {tools_called} 次工具，下次搜到结果就停"
        elif goal.failed_sub_goals:
            lesson = f"有 {len(goal.failed_sub_goals)} 个步骤失败: {goal.failed_sub_goals[0].description[:40]}"
        if lesson:
            self._store_lesson(lesson, agent_id)
            logger.info("📝 Lesson stored: %s", lesson[:50])

    def _error_result(self, ctx: AgentContext, error_msg: str) -> dict:
        """Return a structured error result."""
        ctx.state = AgentState.ERROR
        return {
            "reply": error_msg,
            "error": error_msg,
            "iterations": ctx.iteration,
            "tools_called": ctx.tools_called,
            "memories_used": ctx.memories_injected,
            "modules_used": ctx.modules_used,
        }
