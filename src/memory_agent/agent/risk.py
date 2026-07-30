"""
工具风险分级系统 — 对标 OpenWorker 的 `coworker/risk.py`

OpenWorker 有 4 级风险：READ / WRITE_LOCAL / EXEC / EXTERNAL
每个工具的 intrinsic side-effect 决定它在哪种模式下可以自动执行。

CogniMem 采用同样的 4 级分类，并增加 shell 操作符检测（防止 allowlist 逃逸）。

## 风险等级

| 等级 | 含义 | 示例 | 权限策略 |
|------|------|------|---------|
| READ | 无副作用 | read_file, memory_recall | 所有模式允许 |
| WRITE_LOCAL | 修改本地文件 | write_file, memory_remember | INTERACTIVE需审批 |
| EXEC | 执行命令 | shell, launch_app | 默认需审批；shell操作符检测 |
| EXTERNAL | 外部网络 | web_search, web_fetch | INTERACTIVE需审批，AUTO允许 |
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class RiskClass(str, Enum):
    """工具风险等级

    从低到高：
    - READ: 无副作用，always allowed
    - WRITE_LOCAL: 修改本地文件/数据，path-scoped + mode-gated
    - EXEC: 执行命令，mode-gated + shell operator 检测
    - EXTERNAL: 外部网络操作，mode-gated
    """
    READ = "read"
    WRITE_LOCAL = "write_local"
    EXEC = "exec"
    EXTERNAL = "external"


# ── 风险分级表 ──
# 按分类编排，方便维护
_RISK_TABLE: dict[str, RiskClass] = {
    # ── 文件工具（READ） ──
    "read_file": RiskClass.READ,
    "list_dir": RiskClass.READ,
    "search_files": RiskClass.READ,

    # ── 文件工具（WRITE_LOCAL） ──
    "write_file": RiskClass.WRITE_LOCAL,
    "edit_file": RiskClass.WRITE_LOCAL,

    # ── Shell ──
    "shell": RiskClass.EXEC,
    "launch_app": RiskClass.EXEC,

    # ── Web ──
    "web_search": RiskClass.EXTERNAL,
    "web_fetch": RiskClass.EXTERNAL,
    "browser_open": RiskClass.EXTERNAL,

    # ── 记忆（READ） ──
    "memory_recall": RiskClass.READ,
    "memory_status": RiskClass.READ,
    "memory_diagnose": RiskClass.READ,

    # ── 记忆（WRITE_LOCAL — 修改记忆库） ──
    "memory_remember": RiskClass.WRITE_LOCAL,
    "memory_forget": RiskClass.WRITE_LOCAL,

    # ── 工具（READ） ──
    "get_current_time": RiskClass.READ,
    "ask_user": RiskClass.READ,
    "think": RiskClass.READ,
    "todo": RiskClass.READ,
}


# ── Shell 操作符检测 ──
# 防止 allowlist 逃逸：`git status` → 允许，`git status && rm -rf ~` → 拒绝
# ⚡ v0.22: 改用正则，区分 &&(安全) vs &(危险)、||(安全) vs |(管道)
import re as _re
_SHELL_OPERATORS_RE = _re.compile(
    r'(?<![&])&(?!&)'       # 单个 &（后台进程），排除 &&
    r'|(?<!\|)\|(?!\|)'     # 单个 |（管道），排除 ||
    r'|;'                    # 分号分隔
    r'|>'                    # 重定向写入
    r'|<'                    # 重定向读取（一般安全，但保留）
    r'|`'                    # 反引号执行
    r'|\$[({\[]'            # 命令/变量替换 $( {... $[
    r'|\n'                   # 换行 - 多命令
    r'|\r'                   # 回车
)


def has_shell_operators(command: str) -> bool:
    """检测 shell 命令中是否包含链式/重定向/替换操作符。

    🔥 v0.22: 改用正则，不误判 &&(安全链式) 和 ||(安全OR)。
    例：
      `cd dir && ls`        → False (&& 安全，放行)
      `cd dir && ls > /tmp` → True  (> 写入危险)
      `ls | grep foo`       → True  (| 管道，需要审查)
      `ls &`                 → True  (& 后台)
      `echo $(whoami)`       → True  ($( 命令替换)

    对标 OpenWorker 的 `coworker/permissions.py: _has_shell_operators()`。
    """
    return bool(_SHELL_OPERATORS_RE.search(command))


# ── 核心函数 ──


def classify(tool_name: str, category: str = "") -> RiskClass:
    """获取工具的风险等级。

    先在风险表中查找，找不到用 category 降级猜测：
    - file/shell → 保守视为 EXEC
    - 其余 → READ

    Args:
        tool_name: 工具名称
        category: 工具分类（兜底用）

    Returns:
        RiskClass
    """
    result = _RISK_TABLE.get(tool_name)
    if result is not None:
        return result

    # 按 category 降级猜测
    cat_guess = {
        "shell": RiskClass.EXEC,
        "file": RiskClass.WRITE_LOCAL,
        "web": RiskClass.EXTERNAL,
    }
    return cat_guess.get(category, RiskClass.READ)


def is_consequential(risk: RiskClass) -> bool:
    """是否为高风险操作（非 READ 类）。

    对标 OpenWorker 的 `is_consequential()`。
    """
    return risk is not RiskClass.READ
