"""
Self-Reflector — error analysis and automatic recovery for the Agent.

When a tool fails, the Reflector:
  1. Matches the error against known patterns
  2. Suggests a fix strategy
  3. The Agent loop uses the suggestion to retry or swap tools

This is what makes the agent self-healing instead of just error-reporting.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("agent.reflector")


# ── Error patterns → fix strategies ──
# Each entry: (regex, category, fix_prompt_template)
FIX_PATTERNS: list[tuple[str, str, str]] = [
    # Network / connection
    (r"connect(ion)?\s+refused", "service_down",
     "服务连接被拒绝，可能是目标服务没启动。建议先检查服务状态或换个端口重试。"),
    (r"timeout", "network_slow",
     "请求超时了。建议加长 timeout 参数，或者换一个更快的源。"),
    (r"resolve", "dns_failure",
     "DNS 解析失败。检查域名是否正确，或者用 IP 地址代替。"),
    (r"SSL|certificate|TLS", "ssl_error",
     "SSL 证书错误。尝试忽略证书验证，或用 http 代替 https。"),
    (r"ConnectionError|conn(ection)?\s+reset", "connection_lost",
     "连接中断。可能是网络不稳定，建议重试一次。"),

    # File system
    (r"not found|No such file|No such directory", "path_missing",
     "路径不存在。先创建目录再写入，或者确认文件路径正确。"),
    (r"Permission denied|denied", "no_permission",
     "权限不够。尝试用 sudo 或换个可写目录。"),
    (r"File exists|already exists", "file_exists",
     "文件已存在。检查是否需要覆盖，或用不同的文件名。"),
    (r"Is a directory", "is_directory",
     "路径是一个目录，不是文件。需要提供完整文件路径。"),

    # Shell / dependencies
    (r"command not found", "missing_dependency",
     "缺少命令。需要先安装：pip install / apt install / brew install。"),
    (r"ModuleNotFoundError|ImportError|No module named", "missing_python_package",
     "缺少 Python 包。需要先 pip install 对应包。"),
    (r"SyntaxError|invalid syntax", "syntax_error",
     "Python 语法错误。检查代码缩进和语法。"),
    (r"json\.decoder\.JSONDecodeError|parse error", "parse_error",
     "解析 JSON 失败。返回的不是有效 JSON 格式。"),

    # Rate limits / 429
    (r"429|rate limit|too many requests", "rate_limited",
     "请求太频繁被限流了。建议等几秒再重试。"),

    # Write file specific
    (r"缺少 path 参数|content_empty", "write_file_missing_args",
     "write_file 缺少 path 参数或 content 为空。检查参数：path 用绝对路径，content 必须有非空内容。"),
    (r"写入内容为空", "write_file_empty_content",
     "write_file 的 content 参数是空字符串。需要把完整的文件内容放到 content 参数里再调用。"),
    (r"Read-only file system|read.?only", "write_file_wrong_path",
     "写入路径是只读文件系统。用户 Mac 路径是 /Users/baikai/Desktop/，不是 /root/Desktop/。"),

    # Generic
    (r"empty|no content|404", "not_found",
     "内容不存在或为空。检查 URL 或查询条件是否正确。"),
    (r"HTTP Error|status code", "http_error",
     "HTTP 请求返回错误状态码。检查 URL 和请求参数。"),
]

# ── Tool-specific fallback suggestions ──
TOOL_FALLBACKS: dict[str, list[dict]] = {
    "web_fetch": [
        {"tool": "shell",
         "reason": "web_fetch 内置 curl 可能受限",
         "args_hint": "用 shell 执行 curl -sSL '<url>'"},
        {"tool": "web_search",
         "reason": "直接获取失败，换个方式查"},
    ],
    "shell": [
        {"tool": "shell",
         "reason": "尝试 apt/pip 安装后重试",
         "args_hint": "在命令前加安装依赖"},
    ],
    "write_file": [
        {"tool": "shell",
         "reason": "write_file 工具调用失败，改用 shell 的 heredoc 写入",
         "args_hint": "用 shell：cat <<'MAINEOF' > /Users/baikai/Desktop/文件名.html（写入完整内容）"},
    ],
    "read_file": [
        {"tool": "shell",
         "reason": "文件读取失败，改用 shell 的 cat",
         "args_hint": "用 cat <path> 代替"},
    ],
}


class SelfReflector:
    """
    Analyzes tool execution failures and generates recovery suggestions.

    Used by Agent.chat() when a tool returns an error.
    The suggestion is fed back to the LLM so it can decide the next action.
    """

    def analyze(self, tool_name: str, args: dict, error: str) -> dict:
        """
        Analyze a tool failure and return a structured recovery suggestion.

        Returns:
            {
                "matched": True/False,
                "category": "service_down" | "network_slow" | ...,
                "fix_suggestion": "natural language suggestion for LLM",
                "tool_fallbacks": [{"tool": "...", "reason": "...", "args_hint": "..."}],
                "should_retry": True/False,
            }
        """
        if not error:
            return {"matched": False, "fix_suggestion": "", "should_retry": False}

        # 1. Try to match known patterns
        for pattern, category, fix_template in FIX_PATTERNS:
            if re.search(pattern, error, re.IGNORECASE):
                logger.info("🔍 Error pattern matched: %s (%s)", category, tool_name)

                # Build suggestion
                suggestion = (
                    f"[Agent Self-Reflection] 工具 {tool_name} 遇到了 {category} 类型的错误。\n"
                    f"错误信息: {error[:200]}\n"
                    f"建议: {fix_template}\n"
                )

                # 2. Look for tool-specific fallbacks
                fallbacks = TOOL_FALLBACKS.get(tool_name, [])
                if fallbacks:
                    suggestion += "\n可选的替代方案:\n"
                    for fb in fallbacks:
                        suggestion += f"- 用 {fb['tool']}: {fb['reason']}"
                        if fb.get("args_hint"):
                            suggestion += f" ({fb['args_hint']})"
                        suggestion += "\n"

                return {
                    "matched": True,
                    "category": category,
                    "fix_suggestion": suggestion,
                    "tool_fallbacks": fallbacks,
                    "should_retry": category not in (
                        "permission_denied", "syntax_error",
                        "path_missing", "file_exists",
                    ),
                }

        # 3. No pattern matched — generic fallback
        logger.info("🤷 No pattern matched for error in %s: %s", tool_name, error[:80])
        return {
            "matched": False,
            "category": "unknown",
            "fix_suggestion": (
                f"[Agent Self-Reflection] 工具 {tool_name} 遇到了一个未知错误。\n"
                f"错误信息: {error[:300]}\n"
                f"建议: 尝试换个方法来完成这个任务。\n"
            ),
            "tool_fallbacks": TOOL_FALLBACKS.get(tool_name, []),
            "should_retry": True,
        }

    def check_tool_result(self, result: dict) -> bool:
        """
        Quick check if a tool result indicates success or failure.
        Returns True if the result looks successful.
        """
        if not result:
            return False
        if "error" in result:
            return False
        if isinstance(result, dict) and result.get("success") is False:
            return False
        return True


# ── Auto Fix Executor ──

class FixExecutor:
    """
    Executes automated fixes for common tool errors.

    Usage:
        fixer = FixExecutor()
        fix_result = fixer.try_fix("missing_python_package", "shell", args, error)
        if fix_result["fixed"]:
            # retry the original tool
    """

    @staticmethod
    def try_fix(category: str, tool_name: str, args: dict,
                error: str) -> dict:
        """
        Try to auto-fix a tool error.

        Returns:
            {"fixed": True/False, "action": "description", "result": ...}
        """
        # ── 1. Missing Python package ──
        if category == "missing_python_package":
            pkg = FixExecutor._extract_package_name(error)
            if pkg:
                logger.info("🔧 Auto-fix: pip install %s", pkg)
                import subprocess
                r = subprocess.run(
                    ["pip", "install", pkg],
                    capture_output=True, text=True, timeout=30,
                )
                if r.returncode == 0:
                    return {"fixed": True, "action": f"pip install {pkg}",
                            "detail": r.stdout[-200:]}
                return {"fixed": False, "action": f"pip install {pkg}",
                        "error": r.stderr[-200:]}

        # ── 2. Missing command (apt) ──
        if category == "missing_dependency":
            cmd_name = FixExecutor._extract_command_name(error)
            if cmd_name:
                logger.info("🔧 Auto-fix: trying to install %s", cmd_name)
                import subprocess
                for installer in [
                    f"brew install {cmd_name}",
                    f"apt-get install -y {cmd_name}",
                    f"pip install {cmd_name}",
                ]:
                    try:
                        r = subprocess.run(
                            installer.split(),
                            capture_output=True, text=True, timeout=30,
                        )
                        if r.returncode == 0:
                            return {"fixed": True, "action": installer,
                                    "detail": r.stdout[-200:]}
                    except Exception:
                        continue
                return {"fixed": False, "action": f"tried install {cmd_name}"}

        # ── 3. Path missing — create parent directories ──
        if category == "path_missing":
            path_str = args.get("path", "")
            if path_str:
                from pathlib import Path
                p = Path(path_str).expanduser()
                try:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    logger.info("🔧 Auto-fix: created %s", p.parent)
                    return {"fixed": True, "action": f"mkdir -p {p.parent}"}
                except Exception as e:
                    return {"fixed": False, "action": "mkdir", "error": str(e)}

        # ── 4. File exists — back up old file then overwrite ──
        if category == "file_exists":
            path_str = args.get("path", "")
            if path_str:
                from pathlib import Path
                import shutil
                p = Path(path_str).expanduser()
                if p.exists():
                    backup = p.with_suffix(p.suffix + ".bak")
                    shutil.copy2(p, backup)
                    logger.info("🔧 Auto-fix: backed up %s → %s", p.name, backup.name)
                    return {"fixed": True, "action": f"backup to {backup.name}"}

        # ── 5. write_file empty content → no auto-fix (need LLM to provide content) ──
        if category in ("write_file_empty_content", "write_file_missing_args"):
            logger.info("🔧 Auto-fix 不可用: write_file 需要 LLM 提供正确参数")
            return {"fixed": False, "action": "write_file_needs_llm_retry"}

        return {"fixed": False, "action": "no_auto_fix"}

    @staticmethod
    def _extract_package_name(error: str) -> str | None:
        """Extract Python package name from ModuleNotFoundError."""
        m = re.search(r"ModuleNotFoundError.*?['\"]([^'\"]+)['\"]", error)
        if m:
            return m.group(1)
        m = re.search(r"No module named ['\"]([^'\"]+)['\"]", error)
        if m:
            return m.group(1)
        m = re.search(r"import\s+([a-zA-Z_][a-zA-Z0-9_]*)", error)
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _extract_command_name(error: str) -> str | None:
        """Extract command name from 'command not found' error."""
        m = re.search(r"(['\"])?([a-zA-Z0-9_-]+)(['\"])?\s*:\s*(command not found|未找到)", error)
        if m:
            return m.group(2)
        m = re.search(r"command not found:\s*['\"]?([a-zA-Z0-9_-]+)", error)
        if m:
            return m.group(1)
        return None
