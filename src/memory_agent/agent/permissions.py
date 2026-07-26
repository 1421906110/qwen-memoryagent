"""
CogniMem 对话式权限引擎

🔥 相对优化（vs OpenWorker 前端弹窗审批）：
  - 不弹窗 → Agent 通过对话自然询问用户
  - 不写 Standing Rule → 维持简单
  - 超时自动允许（不阻塞）

用法：
    perm = PermissionEngine(mode="interactive")
    decision = perm.check("write_file", {"path": "/tmp/x"})
    # "allow" | "deny" | "ask"
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("agent.permissions")


# ── 工具风险分类 ──
# ✅ 读工具：自动放行
READ_TOOLS = frozenset({
    "web_search", "web_fetch", "read_file", "list_dir",
    "memory_recall", "memory_status", "memory_diagnose",
})

# 🔶 写工具：对话式审批
WRITE_TOOLS = frozenset({
    "write_file", "edit_file", "shell",
    "memory_remember", "memory_forget",
})

# ⚠️ 高风险工具：总是审批
HIGH_RISK_TOOLS = frozenset({
    "shell",
})


class PermissionEngine:
    """对话式权限引擎

    3 种模式:
      discuss      → 只允许读工具
      interactive  → 读自动放行，写对话审批（默认）
      auto         → 全放权

    审批不弹窗，通过 chat 接口自然询问用户。
    """

    MODES = ("discuss", "interactive", "auto")

    def __init__(self, mode: str = "interactive"):
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode: {mode}, expected {self.MODES}")
        self.mode = mode
        # session 级授权缓存（key: "tool_name:target"）
        self._grants: dict[str, set[str]] = {}

    def check(self, tool_name: str, args: dict = None,
              session_id: str = "") -> str:
        """检查工具是否允许执行

        Returns:
            "allow" — 自动放行
            "deny"  — 拒绝
            "ask"   — 需要对话审批（TurnEngine 应追加询问消息到对话）
        """
        if self.mode == "auto":
            return "allow"

        if tool_name in READ_TOOLS:
            return "allow"

        if self.mode == "discuss":
            return "deny"

        if tool_name in HIGH_RISK_TOOLS:
            return "ask"  # 高风险始终 ask

        if tool_name in WRITE_TOOLS:
            return "ask"

        # 未知工具：默认放行
        return "allow"

    def grant(self, tool_name: str, session_id: str = "",
              target: str = "") -> None:
        """授权（session内不再询问）"""
        if session_id:
            self._grants.setdefault(session_id, set()).add(tool_name)

    def revoke(self, tool_name: str, session_id: str = "") -> None:
        """撤销授权"""
        if session_id and tool_name in self._grants.get(session_id, set()):
            self._grants[session_id].discard(tool_name)

    def approval_prompt(self, tool_name: str, args: dict = None) -> str:
        """生成对话式审批的询问文本

        TurnEngine 检测到 "ask" 时调用此方法，
        生成一条自然语言消息追加到对话中。
        """
        prompts = {
            "shell": f"需要执行命令: {args.get('command','')[:60]}，允许吗？",
            "write_file": f"需要写入文件: {args.get('path','')}，允许吗？",
            "edit_file": f"需要编辑文件: {args.get('path','')}，允许吗？",
            "memory_remember": "需要保存一条记忆，允许吗？",
            "memory_forget": "需要清理记忆，允许吗？",
        }
        return prompts.get(tool_name,
                           f"需要调工具 {tool_name}，允许吗？")
