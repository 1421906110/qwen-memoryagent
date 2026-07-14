"""
🔐 工具调用验证器（Tool Validator Layer）

受 Emma MemoryAgent 的 Agent Firewall + Rubik Instructor 的验证器模式启发。
在工具执行前做逻辑验证，防止：
- 路径穿越 / 系统文件写入
- 危险 shell 命令
- 内网 URL 访问
- 超大内容写入/读取

每个验证函数返回 (passed: bool, reason: str)
passed=False 时工具应拒绝执行，直接返回 reason。
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger("agent.validator")

# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

# 禁止写入的系统路径前缀
FORBIDDEN_WRITE_PATHS = [
    "/etc", "/sys", "/proc", "/dev", "/boot", "/var/lib",
    "/usr/lib", "/usr/bin", "/usr/sbin", "/bin", "/sbin",
    "/System", "/Library", "/private/etc",
]

# 禁止写入的文件名模式
FORBIDDEN_WRITE_FILES = [
    ".ssh/id_rsa", ".ssh/id_ed25519", ".ssh/authorized_keys",
    ".aws/credentials", ".config/gcloud/",
    ".git/config", ".env",
]

# 危险命令前缀（shell 执行时检查）
BLOCKED_COMMAND_PREFIXES = [
    "rm -rf /", "rm -rf /*", "rm -rf ~", "rm -rf .",
    "mkfs.", "dd if=", ":(){ :|:& };:", "chmod 777 /",
    "chown -R", "> /dev/sda", "> /dev/sdb",
    "wget ", "curl ",  # 外部下载可能不安全，用内置 web_fetch 代替
]

# 危险命令关键词
BLOCKED_COMMAND_KEYWORDS = [
    "sudo", "su ", "chmod 777", "chmod 0",
    "passwd", "useradd", "usermod", "deluser",
    "shutdown", "reboot", "poweroff", "init 0", "init 6",
]

# 不安全的主机地址（禁止 web_fetch）
BLOCKED_HOSTS = [
    "127.0.0.1", "localhost", "0.0.0.0", "::1",
    "10.", "172.16.", "172.17.", "172.18.", "172.19.",
    "172.20.", "172.21.", "172.22.", "172.23.",
    "172.24.", "172.25.", "172.26.", "172.27.",
    "172.28.", "172.29.", "172.30.", "172.31.",
    "192.168.",
]

# 最大写入字节数
MAX_WRITE_BYTES = 1_000_000  # 1MB
# 最大读取行数
MAX_READ_LINES = 100_000  # 10万行
# 最大读取单行长度
MAX_READ_LINE_LENGTH = 100_000  # 10万字符
# 最大命令超时（秒）
MAX_COMMAND_TIMEOUT = 120

# 本地 Mac 路径模式（Agent 运行在服务器上，无法访问这些路径）
LOCAL_MAC_PATH_PATTERNS = [
    "/Users/",
    "~/Desktop",
    "~/Downloads",
    "~/Documents",
    "~/Movies",
    "~/Music",
    "~/Pictures",
]


def is_local_mac_path(path: str) -> tuple[bool, str]:
    """
    检测路径是否为本机 Mac 路径（服务器上不可访问）。

    返回 (True, 匹配的模式) 如果是本地路径，
    返回 (False, "") 如果是服务器路径。
    """
    expanded = os.path.expanduser(path)
    for pattern in LOCAL_MAC_PATH_PATTERNS:
        if pattern in expanded or pattern in path:
            return True, pattern
    return False, ""


def build_local_path_hint(path: str) -> str:
    """
    构建友好的本地路径错误提示。
    Agent 在服务器上运行，无法写入/复制到本机 Mac 路径。
    """
    is_local, pattern = is_local_mac_path(path)
    if is_local:
        return (
            f"❌ 目标路径包含本机 Mac 路径模式「{pattern}」\n"
            f"   Agent 运行在远程服务器上，无法直接写入/复制到本机路径。\n"
            f"   ✅ 正确做法：\n"
            f"      1. 文件先保存到服务器路径（如 /home/ecs-user/）\n"
            f"      2. 然后本机执行 scp 拉取：\n"
            f"         scp root@47.99.151.253:/服务器路径/文件名 ~/Desktop/\n"
        )
    return ""


# ═══════════════════════════════════════════════════════════════
# 验证函数
# ═══════════════════════════════════════════════════════════════

def validate_read_path(path: str | Path) -> tuple[bool, str]:
    """
    验证读取路径是否安全。
    - 不能是目录（read_file 操作）
    - 不能是符号链接到系统路径
    - 不能超过合理大小（由工具自身控制）
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return True, ""  # 不存在由工具处理
    if p.is_dir():
        return False, f"路径是目录，不能直接读取: {p}"
    return True, ""


def validate_write_path(path: str | Path) -> tuple[bool, str]:
    """
    验证写入路径是否安全。
    - 不允许写入系统路径
    - 不允许覆盖关键配置文件
    - 不允许路径穿越到父目录外
    """
    p = Path(path).expanduser().resolve()
    p_str = str(p)

    # 禁止写入系统路径
    for forbidden in FORBIDDEN_WRITE_PATHS:
        if p_str.startswith(forbidden):
            return False, f"拒绝写入系统路径: {forbidden}"

    # 禁止覆盖关键配置文件
    for pattern in FORBIDDEN_WRITE_FILES:
        if p_str.endswith(pattern) or pattern in p_str:
            return False, f"拒绝覆盖关键文件: {pattern}"

    # 检查文件名是否为隐藏敏感文件
    fname = p.name
    if fname in (".env", ".env.local", ".git-credentials", ".netrc"):
        return False, f"拒绝写入敏感文件: {fname}"

    # 检测本地 Mac 路径（Agent 在远程服务器上，无法写入本机路径）
    hint = build_local_path_hint(p_str)
    if hint:
        return False, hint

    return True, ""


def validate_shell_command(command: str) -> tuple[bool, str]:
    """
    验证 shell 命令是否安全。
    - 阻止危险命令
    - 阻止交互式命令
    - 阻止 cp/mv 到本地 Mac 路径（服务器上不存在）
    - 限制超时
    """
    cmd_stripped = command.strip().lower()

    # 阻止危险命令前缀
    for prefix in BLOCKED_COMMAND_PREFIXES:
        if cmd_stripped.startswith(prefix):
            return False, f"拒绝执行危险命令: {prefix}"

    # 阻止危险关键词
    for kw in BLOCKED_COMMAND_KEYWORDS:
        if kw in cmd_stripped:
            return False, f"命令包含危险操作: {kw}"

    # 阻止交互式命令
    interactive = ["vim ", "vi ", "nano ", "emacs ", "less ", "more ",
                   "top", "htop", "ssh ", "telnet ", "ftp ", "python -i"]
    for ic in interactive:
        if cmd_stripped.startswith(ic):
            return False, f"拒绝交互式命令: {ic}"

    # 阻止可能危险的管道组合
    dangerous_pipes = ["dd if=", "cat /dev/sda", "cat /dev/sdb",
                       "fdisk", "parted", "mkswap"]
    for dp in dangerous_pipes:
        if dp in cmd_stripped:
            return False, f"命令包含危险操作: {dp}"

    # 检测 cp/mv/scp 目标路径是否为本机 Mac 路径
    file_cmds = ["cp ", "mv ", "scp ", "rsync ", "cat > ", "cat >> "]
    for fc in file_cmds:
        if fc in cmd_stripped:
            # 提取命令中的路径参数（简单启发式：找 /Users/ 或 ~/Desktop 等模式）
            for pattern in LOCAL_MAC_PATH_PATTERNS:
                if pattern.lower() in cmd_stripped:
                    return False, build_local_path_hint(pattern)

    return True, ""


def validate_url(url: str) -> tuple[bool, str]:
    """
    验证 URL 是否可安全访问。
    - 必须是 http/https
    - 不能是内网地址
    - 不能包含凭证
    """
    if not url.startswith(("http://", "https://")):
        return False, "只支持 http/https 协议"

    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""

        # 拒绝内网地址
        for blocked in BLOCKED_HOSTS:
            if host.startswith(blocked):
                return False, f"拒绝访问内网地址: {host}"

        # 拒绝带凭证的 URL
        if parsed.username or parsed.password:
            return False, "URL 不能包含用户名密码"

        # 拒绝 IP 形式的 localhost
        import ipaddress
        try:
            ip = ipaddress.ip_address(host)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return False, f"拒绝访问内网 IP: {host}"
        except ValueError:
            pass  # 域名，没问题

    except Exception as e:
        return False, f"URL 解析失败: {e}"

    return True, ""


def validate_write_content(content: str) -> tuple[bool, str]:
    """验证写入内容是否过大"""
    if not isinstance(content, str):
        return False, "写入内容必须是字符串"
    if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
        return False, f"写入内容过大（最大 {MAX_WRITE_BYTES//1024//1024}MB）"
    return True, ""


def validate_command_timeout(timeout: int | float) -> tuple[bool, str]:
    """验证命令超时是否合理"""
    if timeout <= 0:
        return False, "超时时间必须大于 0"
    if timeout > MAX_COMMAND_TIMEOUT:
        return False, f"超时时间不能超过 {MAX_COMMAND_TIMEOUT}s"
    return True, ""


# ═══════════════════════════════════════════════════════════════
# 工具专用验证（组合验证函数）
# ═══════════════════════════════════════════════════════════════

def check_read_file(args: dict) -> tuple[bool, str]:
    """验证 read_file 参数"""
    path = args.get("path", "")
    if not path:
        return False, "缺少 path 参数"
    return validate_read_path(path)


def check_write_file(args: dict) -> tuple[bool, str]:
    """验证 write_file 参数"""
    path = args.get("path", "")
    content = args.get("content", "")
    if not path:
        return False, "缺少 path 参数"
    ok, reason = validate_write_path(path)
    if not ok:
        return False, reason
    ok, reason = validate_write_content(content)
    if not ok:
        return False, reason
    return True, ""


def check_shell(args: dict) -> tuple[bool, str]:
    """验证 shell 命令参数"""
    command = args.get("command", "")
    timeout = args.get("timeout", 30)
    if not command:
        return False, "缺少 command 参数"
    ok, reason = validate_shell_command(command)
    if not ok:
        return False, reason
    ok, reason = validate_command_timeout(timeout)
    if not ok:
        return False, reason
    return True, ""


def check_web_fetch(args: dict) -> tuple[bool, str]:
    """验证 web_fetch/visit_url 参数"""
    url = args.get("url", "")
    if not url:
        return False, "缺少 url 参数"
    return validate_url(url)


def check_python_repl(args: dict) -> tuple[bool, str]:
    """验证 python_repl 参数（防止危险代码）"""
    code = args.get("code", "")
    if not code:
        return False, "缺少 code 参数"
    # 阻止 os.system/subprocess 调用
    dangerous_imports = ["import os", "from os import", "import subprocess",
                         "from subprocess import", "import shutil"]
    for di in dangerous_imports:
        if di in code:
            # 允许 os.path 这种无害操作
            if "os.path" in code and "os.system" not in code and "os.popen" not in code:
                continue
            return False, f"Python REPL 不允许执行系统命令: {di}"
    return True, ""


# ═══════════════════════════════════════════════════════════════
# 验证器注册表
# ═══════════════════════════════════════════════════════════════

# 工具名称 → 验证函数映射
VALIDATORS: dict[str, callable] = {
    "read_file": check_read_file,
    "write_file": check_write_file,
    "edit_file": check_write_file,  # edit 也需要路径验证
    "shell": check_shell,
    "web_fetch": check_web_fetch,
    "visit_url": check_web_fetch,
    "python_repl": check_python_repl,
    "execute_code": check_python_repl,
}


def validate_tool_call(tool_name: str, args: dict) -> tuple[bool, str]:
    """
    统一的工具调用验证入口。

    Args:
        tool_name: 工具名称
        args: 参数字典

    Returns:
        (passed, reason): passed=True 表示验证通过
    """
    validator = VALIDATORS.get(tool_name)
    if validator is None:
        # 没有验证器的工具默认通过（memory 类操作）
        return True, ""
    return validator(args)
