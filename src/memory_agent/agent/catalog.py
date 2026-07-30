"""
Capability 能力目录 — 工具按能力分组 + 声明依赖/风险

参考 OpenWorker `catalog.py` 设计：
每组工具声明 requires（需要什么上下文依赖）和 risk（风险等级）。
expand(ids, context) 自动跳过不满足条件的能力。

使用方式（不破坏现有注册）：
    from .catalog import Capability, expand_capabilities

    ctx = AgentContext(workspace="/tmp", executor=True)
    tools = expand_capabilities(["filesystem", "web"], ctx, registry)
    # → 返回 registry.schemas() 的子集
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger("agent.catalog")

# Context prerequisites — maps requirement name to predicate over AgentContext
_REQUIREMENTS: dict[str, Callable[[Any], bool]] = {
    "workspace": lambda ctx: hasattr(ctx, "workspace") and ctx.workspace is not None,
    "executor": lambda ctx: hasattr(ctx, "executor") and ctx.executor is not None,
    "memory": lambda ctx: hasattr(ctx, "cogni") and ctx.cogni is not None,
}


@dataclass(frozen=True)
class Capability:
    """能力定义 — 一组工具的集合，带依赖和风险声明。

    Attributes:
        id: 唯一标识（如 "filesystem"）
        name: 人类可读名称（如 "文件操作"）
        description: 能力描述
        tool_names: 包含的工具名列表
        requires: 需要的上下文依赖元组（"workspace"/"executor"/"memory"）
        risk: 风险等级元组
    """
    id: str
    name: str
    description: str
    tool_names: tuple[str, ...] = field(default_factory=tuple)
    requires: tuple[str, ...] = field(default_factory=tuple)
    risk: tuple[str, ...] = field(default_factory=lambda: ("read",))

    def available(self, context: Any) -> bool:
        """检查当前上下文是否满足能力依赖"""
        return all(_REQUIREMENTS[r](context) for r in self.requires)


# ═══════════════════════════════════════════════════════════════════
#  能力目录定义（不注册工具，只按能力名引用已有工具）
# ═══════════════════════════════════════════════════════════════════

CATALOG: dict[str, Capability] = {
    "memory": Capability(
        id="memory",
        name="记忆管理",
        description="读取/写入/管理长期记忆",
        tool_names=("memory_recall", "memory_remember", "memory_status",
                    "memory_diagnose", "memory_forget"),
        requires=("memory",),
        risk=("read", "write_local"),
    ),
    "filesystem": Capability(
        id="filesystem",
        name="文件操作",
        description="读取/写入/编辑文件",
        tool_names=("read_file", "write_file", "edit_file", "list_dir",
                    "glob", "grep", "file_search"),
        requires=("workspace",),
        risk=("read", "write_local"),
    ),
    "shell": Capability(
        id="shell",
        name="Shell 执行",
        description="运行 shell 命令",
        tool_names=("shell",),
        requires=("executor",),
        risk=("exec",),
    ),
    "web": Capability(
        id="web",
        name="网络访问",
        description="搜索网页/获取 URL 内容",
        tool_names=("web_search", "web_fetch"),
        risk=("read", "external"),
    ),
    "todo": Capability(
        id="todo",
        name="任务管理",
        description="管理待办事项列表",
        tool_names=("todo",),
        requires=("workspace",),
        risk=("read", "write_local"),
    ),
    "code": Capability(
        id="code",
        name="代码",
        description="搜索/阅读/修改代码",
        tool_names=("read_file", "write_file", "edit_file", "grep",
                    "glob", "list_dir", "file_search"),
        requires=("workspace",),
        risk=("read", "write_local"),
    ),
}


def register_capability(cap: Capability) -> None:
    """注册/覆盖一个能力定义"""
    CATALOG[cap.id] = cap


def expand(ids: list[str], context: Any, registry=None) -> list[str]:
    """根据能力 ID 列表和当前上下文，返回可用工具名列表。

    Args:
        ids: 能力 ID 列表（如 ["memory", "filesystem"]）
        context: AgentContext 对象
        registry: ToolRegistry 实例（可选，如果传入会校验工具是否已注册）

    Returns:
        可用工具名列表
    """
    available = []
    missing = []
    for cap_id in ids:
        cap = CATALOG.get(cap_id)
        if cap is None:
            missing.append(cap_id)
            continue
        if not cap.available(context):
            logger.debug("⏭️ Capability '%s' skipped: unmet requires %s",
                         cap_id, cap.requires)
            continue
        for name in cap.tool_names:
            if registry is None or registry.get(name) is not None:
                available.append(name)
    if missing:
        logger.warning("⚠️ Unknown capabilities: %s", missing)
    return available


def expand_schemas(ids: list[str], context: Any, registry) -> list[dict]:
    """根据能力 ID 列表返回工具的 JSON schema 列表（直接给 LLM 用）"""
    names = expand(ids, context, registry)
    return [registry.get(n)["schema"] for n in names if registry.get(n)]
