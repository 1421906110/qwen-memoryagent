"""
Built-in tools for the Agent system.

Each tool function signature: (tool_call_id: str, args: dict, ctx: AgentContext) -> dict

Tools access ctx.cogni for CogniMem integration, where ctx is the AgentContext
that holds the cogni_client reference passed at agent creation time.

Categories match the research report's vision:
  - file:     Read/write/edit files (like Claude Code)
  - shell:    Execute commands
  - web:      Fetch URLs and search
  - memory:   CogniMem integration
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory_agent.agent import AgentContext

logger = logging.getLogger("agent.tools")

# ── Web 结果缓存（避免同 URL 反复抓取） ──
_WEB_CACHE: OrderedDict[str, dict] = OrderedDict()
_WEB_CACHE_MAX = 30
_WEB_TIMEOUT = 8

# ── 失败域名追踪（连续失败 2 次后自动跳过） ──
_FAILED_DOMAINS: dict[str, int] = {}


def _is_domain_blocked(url: str) -> bool:
    from urllib.parse import urlparse
    try:
        return _FAILED_DOMAINS.get(urlparse(url).netloc, 0) >= 2
    except Exception:
        return False


def _mark_domain_failed(url: str) -> None:
    from urllib.parse import urlparse
    try:
        domain = urlparse(url).netloc
        _FAILED_DOMAINS[domain] = _FAILED_DOMAINS.get(domain, 0) + 1
        if _FAILED_DOMAINS[domain] >= 2:
            logger.info("⛔ Domain blocked: %s", domain)
    except Exception as e:
        logger.debug("Failed to track domain failure: %s", e)


def _cache_get(url: str) -> dict | None:
    """Get cached web result."""
    return _WEB_CACHE.get(url)


def _cache_set(url: str, result: dict) -> None:
    """Cache web result with LRU eviction."""
    if url in _WEB_CACHE:
        _WEB_CACHE.move_to_end(url)
    else:
        _WEB_CACHE[url] = result
        if len(_WEB_CACHE) > _WEB_CACHE_MAX:
            _WEB_CACHE.popitem(last=False)  # 淘汰最旧的

# ===================================================================
#  FILE TOOLS
# ===================================================================

def tool_read_file(tool_call_id: str, args: dict,
                   ctx: "AgentContext") -> dict:
    """Read a file from the filesystem."""
    path = Path(args["path"]).expanduser().resolve()
    offset = args.get("offset", 0)
    limit = args.get("limit", 2000)

    if not path.exists():
        return {"error": f"File not found: {path}", "found": False}
    if path.is_dir():
        return {"error": f"Path is a directory: {path}", "found": False}

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        total = len(lines)
        selected = lines[offset:offset + limit]

        return {
            "found": True,
            "path": str(path),
            "total_lines": total,
            "offset": offset,
            "lines_read": len(selected),
            "content": "".join(selected),
        }
    except Exception as e:
        return {"error": str(e), "found": False}


def tool_write_file(tool_call_id: str, args: dict,
                    ctx: "AgentContext") -> dict:
    """Write content to a file (overwrite)."""
    path = Path(args["path"]).expanduser().resolve()
    content = args["content"]

    # Safety check
    for forbidden in ["/etc", "/sys", "/proc", "/dev"]:
        if str(path).startswith(forbidden):
            return {"error": f"Refusing to write to system path: {forbidden}"}

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.info("📝 Wrote %d bytes to %s", len(content), path)
        return {
            "success": True,
            "path": str(path),
            "bytes_written": len(content),
            "lines": content.count("\n") + 1,
        }
    except Exception as e:
        return {"error": str(e)}


def tool_edit_file(tool_call_id: str, args: dict,
                   ctx: "AgentContext") -> dict:
    """Surgical edit — replace exact old_string with new_string."""
    path = Path(args["path"]).expanduser().resolve()

    if not path.exists():
        return {"error": f"File not found: {path}"}

    try:
        content = path.read_text(encoding="utf-8")
        old = args["old_string"]
        new = args.get("new_string", "")

        count = content.count(old)
        if count == 0:
            return {"error": f"old_string not found in {path}", "matches": 0}
        if count > 1:
            return {
                "error": f"old_string found {count} times — must be unique",
                "matches": count,
            }

        content = content.replace(old, new)
        path.write_text(content, encoding="utf-8")
        logger.info("✏️  Edited %s: replaced 1 occurrence", path)
        return {"success": True, "path": str(path), "replacements": 1}
    except Exception as e:
        return {"error": str(e)}


def tool_list_dir(tool_call_id: str, args: dict,
                  ctx: "AgentContext") -> dict:
    """List files/directories at a path."""
    path = Path(args["path"]).expanduser().resolve()
    pattern = args.get("pattern", "*")

    if not path.exists():
        return {"error": f"Path not found: {path}"}
    if not path.is_dir():
        return {"error": f"Not a directory: {path}"}

    try:
        items = []
        for p in path.glob(pattern):
            items.append({
                "name": p.name,
                "path": str(p),
                "type": "dir" if p.is_dir() else "file",
                "size": p.stat().st_size if p.is_file() else 0,
            })

        items.sort(key=lambda x: (x["type"] != "dir", x["name"]))
        return {"path": str(path), "total": len(items), "items": items}
    except Exception as e:
        return {"error": str(e)}


# ===================================================================
#  SHELL TOOL
# ===================================================================

def tool_shell(tool_call_id: str, args: dict,
               ctx: "AgentContext") -> dict:
    """Execute a shell command."""
    command = args["command"]
    timeout = min(args.get("timeout", 30), 120)

    logger.info("⚡ Shell: %s", command[:200])

    try:
        result = subprocess.run(
            command, shell=True, capture_output=True,
            text=True, timeout=timeout,
        )
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout[-5000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
            "truncated_stdout": len(result.stdout or "") > 5000,
            "truncated_stderr": len(result.stderr or "") > 2000,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout}s"}
    except Exception as e:
        return {"error": str(e)}


# ===================================================================
#  WEB TOOLS
# ===================================================================

def tool_web_fetch(tool_call_id: str, args: dict,
                   ctx: "AgentContext") -> dict:
    """Fetch a URL and return content as text.

    Uses httpx (with proxy if SEARCH_PROXY is set) instead of curl subprocess.
    Results are cached for the duration of the agent session (LRU, max 30 URLs).
    Domains that fail twice are auto-blocked for the session.
    """
    url = args["url"]

    # 跳过已知失败的域名
    if _is_domain_blocked(url):
        return {"error": f"Skipped (domain previously failed): {url}"}

    # 检查缓存
    cached = _cache_get(url)
    if cached:
        cached["from_cache"] = True
        return cached

    proxy = os.environ.get("SEARCH_PROXY", "")
    try:
        import httpx
        client_kwargs = {
            "timeout": httpx.Timeout(_WEB_TIMEOUT),
            "follow_redirects": True,
            "headers": {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
        }
        if proxy:
            if proxy.startswith("socks"):
                # httpx 原生不支持 socks，需用 socksio 包
                # 回退：用 SEARCH_PROXY 作为 HTTP 代理发送 socks 请求
                # 大多数梯子同时支持 HTTP 和 SOCKS，用 HTTP 代理模式兼容性更好
                client_kwargs["proxy"] = proxy.replace("socks5://", "http://").replace("socks5h://", "http://")
            else:
                client_kwargs["proxy"] = proxy

        with httpx.Client(**client_kwargs) as client:
            resp = client.get(url)
            resp.raise_for_status()
            content = resp.text[:8000]
            result = {
                "url": url, "content": content, "length": len(content),
                "truncated": len(resp.text) > 8000,
                "status_code": resp.status_code,
            }
            _cache_set(url, result)
            return result
    except Exception as e:
        _mark_domain_failed(url)
        result = {"error": f"Failed: {str(e)[:80]}"}
        _cache_set(url, result)
        return result


def tool_web_search(tool_call_id: str, args: dict,
                    ctx: "AgentContext") -> dict:
    """Search the web.

    Uses Qwen's built-in search (enable_search=true) via LLM client.
    Falls back to Bing scraping if SEARCH_PROXY is set.
    """
    query = args["query"]

    # 方式 1：用 Qwen 内置搜索（走 DashScope API，国内可用，无需代理）
    try:
        from memory_agent.services.llm_client import LLMClient
        from openai import OpenAI
        import os

        # 直接用 OpenAI 客户端调 Qwen，走 enable_search，设长超时
        api_key = os.getenv("QWEN_API_KEY", "")
        base_url = os.getenv("QWEN_BASE_URL", "")
        model = os.getenv("QWEN_MODEL", "qwen3.7-plus")
        if api_key and base_url:
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"请搜索：{query}，用中文总结搜索结果，列出具体信息"}],
                temperature=0.3,
                extra_body={"enable_search": True},
            )
            reply = resp.choices[0].message.content or ""
            if reply and len(reply) > 20:
                return {"result": reply, "source": "qwen_search", "query": query}
    except Exception as e:
        logger.info("Qwen 内置搜索不可用: %s", e)

    # 方式 2：Bing 直搜（需代理）
    proxy = os.environ.get("SEARCH_PROXY", "")
    if proxy:
        try:
            import re as _re, urllib.parse, httpx
            encoded = urllib.parse.quote(query)
            p = proxy
            if p.startswith("socks"):
                p = p.replace("socks5://", "http://").replace("socks5h://", "http://")
            with httpx.Client(proxy=p, timeout=15, follow_redirects=True) as client:
                resp = client.get(f"https://www.bing.com/search?q={encoded}&setlang=zh-CN",
                    headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "zh-CN"})
                if resp.status_code == 200:
                    texts = []
                    for m in _re.findall(r'<p[^>]*class="b_lineclamp[^"]*"[^>]*>(.*?)</p>', resp.text):
                        clean = _re.sub(r'<[^>]+>', '', m).strip()
                        if clean and len(clean) > 10:
                            texts.append(clean)
                    for m in _re.findall(r'<h2>.*?<a[^>]*>(.*?)</a>', resp.text):
                        clean = _re.sub(r'<[^>]+>', '', m).strip()
                        if clean:
                            texts.append(clean)
                    if texts:
                        return {"result": "\n".join(texts[:10]), "source": "bing", "query": query}
        except Exception as e:
            logger.info("Bing 搜索失败: %s", e)

    return {"error": "搜索服务暂不可用"}


# ===================================================================
#  MEMORY TOOLS (CogniMem integration — ctx.cogni must be set)
# ===================================================================

def tool_memory_recall(tool_call_id: str, args: dict,
                       ctx: "AgentContext") -> dict:
    """Recall relevant memories from CogniMem."""
    if not ctx.cogni:
        return {"error": "CogniMem not connected", "memories": []}
    try:
        result = ctx.cogni.recall(
            query=args["query"],
            agent_id=ctx.agent_id,
            top_k=args.get("limit", 10),
        )
        facts = result.get("facts", [])
        return {
            "found": len(facts) > 0,
            "count": len(facts),
            "memories": [
                {
                    "id": f.get("fact_id", ""),
                    "content": f"{f.get('subject','')} {f.get('predicate','')} {f.get('object','')}",
                    "type": f.get("fact_type", "observation"),
                    "confidence": f.get("confidence", 0.5),
                }
                for f in facts
            ],
        }
    except Exception as e:
        return {"error": str(e), "memories": []}


def tool_memory_remember(tool_call_id: str, args: dict,
                         ctx: "AgentContext") -> dict:
    """Store new information in CogniMem."""
    if not ctx.cogni:
        return {"error": "CogniMem not connected"}
    try:
        result = ctx.cogni.remember(
            text=args["text"],
            agent_id=ctx.agent_id,
            source=f"agent_tool:{ctx.session_id}",
        )
        return {
            "stored": True,
            "facts_added": result.get("facts_added", 0),
            "status": result.get("status", "unknown"),
        }
    except Exception as e:
        return {"error": str(e)}


def tool_memory_status(tool_call_id: str, args: dict,
                       ctx: "AgentContext") -> dict:
    """Get CogniMem status."""
    if not ctx.cogni:
        return {"error": "CogniMem not connected"}
    try:
        stats = ctx.cogni.get_status(ctx.agent_id)
        return {
            "total_facts": stats.get("total_facts", 0),
            "core_beliefs": stats.get("core_beliefs", 0),
            "by_type": stats.get("by_type", {}),
        }
    except Exception as e:
        return {"error": str(e)}


def tool_memory_diagnose(tool_call_id: str, args: dict,
                         ctx: "AgentContext") -> dict:
    """自我诊断记忆系统健康。检查记忆总量、矛盾数、置信度分布、路由命中率。"""
    if not ctx.cogni:
        return {"error": "CogniMem not connected"}
    try:
        stats = ctx.cogni.get_stats(ctx.agent_id)
        total = stats.get("total_facts", 0)
        contradictions = stats.get("contradictions", 0)
        core = stats.get("core_beliefs", 0)
        unreliable = stats.get("unreliable", 0)
        by_type = stats.get("by_type", {})
        router_stats = stats.get("router_stats", {})

        # 计算健康指标
        conflict_rate = contradictions / max(total, 1)
        unreliable_rate = unreliable / max(total, 1)
        core_rate = core / max(total, 1)

        issues = []
        if conflict_rate > 0.2:
            issues.append(f"⚠️ 矛盾率 {conflict_rate:.0%}（{contradictions}/{total}），建议 consolidation")
        if unreliable_rate > 0.5:
            issues.append(f"⚠️ 不可靠事实占比 {unreliable_rate:.0%}，需要用户确认")
        if total < 10:
            issues.append(f"ℹ️ 记忆总量较少（{total}条），多对话后会积累更多")
        if core_rate < 0.1 and total > 20:
            issues.append(f"💡 核心信念较少（{core}条），高频记忆可提升为信念")

        return {
            "total_facts": total,
            "core_beliefs": core,
            "contradictions": contradictions,
            "unreliable": unreliable,
            "by_type": by_type,
            "conflict_rate": f"{conflict_rate:.1%}",
            "core_rate": f"{core_rate:.1%}",
            "router_stats": router_stats,
            "issues": issues,
            "healthy": len(issues) == 0,
        }
    except Exception as e:
        return {"error": str(e)}


def tool_memory_forget(tool_call_id: str, args: dict,
                       ctx: "AgentContext") -> dict:
    """主动遗忘低置信度记忆（手动触发记忆衰减清理）。"""
    if not ctx.cogni:
        return {"error": "CogniMem not connected"}
    try:
        result = ctx.cogni.fact_network.forget(ctx.agent_id)
        return {
            "decayed": result.get("decayed", 0),
            "deleted": result.get("deleted", 0),
            "core_preserved": result.get("core_preserved", 0),
            "message": f"衰减 {result.get('decayed',0)} 条，"
                       f"删除 {result.get('deleted',0)} 条，"
                       f"保留核心 {result.get('core_preserved',0)} 条",
        }
    except Exception as e:
        return {"error": str(e)}


# ===================================================================
#  TOOL REGISTRATION
# ===================================================================

def register_all_tools(registry, cogni_client=None):
    """Register all built-in tools with the registry.

    Call once during app initialization.
    Memory tools are only registered if cogni_client is available.
    """

    # ── File tools ──
    registry.register(
        "read_file",
        "读取文件内容。适合查看源码、日志、配置文件等。返回带行号的内容。"
        "如果文件很大可以用 offset/limit 分段读取。",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file"},
                "offset": {"type": "integer", "description": "Line number to start from (0-indexed)", "default": 0},
                "limit": {"type": "integer", "description": "Max lines to read (max 5000)", "default": 2000},
            },
            "required": ["path"],
        },
        tool_read_file, category="file",
    )
    registry.register(
        "write_file",
        "写入文件（覆蓋模式）。自動創建父目錄。"
        "適合創建新文件、保存修改後的內容。注意：會完全覆蓋原文件！",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file"},
                "content": {"type": "string", "description": "Full file content to write"},
            },
            "required": ["path", "content"],
        },
        tool_write_file, category="file",
    )
    registry.register(
        "edit_file",
        "精準編輯文件——替換指定的 old_string 為 new_string。"
        "適合小幅修改（改一行、改一個詞），比 write_file 安全。"
        "注意：old_string 必須在文件中唯一。",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file"},
                "old_string": {"type": "string", "description": "The exact text to replace (must be unique)"},
                "new_string": {"type": "string", "description": "The replacement text"},
            },
            "required": ["path", "old_string"],
        },
        tool_edit_file, category="file",
    )
    registry.register(
        "list_dir",
        "列出目錄內容。支持 glob 模式過濾（如 *.py）。"
        "適合先查看目錄結構再決定讀哪個文件。",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to list"},
                "pattern": {"type": "string", "description": "Glob filter (e.g. '*.py')", "default": "*"},
            },
            "required": ["path"],
        },
        tool_list_dir, category="file",
    )

    # ── Shell tool ──
    registry.register(
        "shell",
        "執行 shell 命令。適合運行腳本、git、pip、npm、python 等。"
        "注意：命令在服務器上直接執行，不要用 rm -rf 等危險操作。"
        "如果只是查看文件用 read_file 更安全。",
        {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (max 120)", "default": 30},
            },
            "required": ["command"],
        },
        tool_shell, category="shell",
    )

    # ── Web tools ──
    registry.register(
        "web_fetch",
        "抓取指定 URL 的內容。適合讀取網頁、REST API 等。"
        "返回純文本（自動截斷到 8000 字符）。URL 會緩存，同一 URL 不重複抓。",
        {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch (http/https)"},
            },
            "required": ["url"],
        },
        tool_web_fetch, category="web",
    )
    registry.register(
        "web_search",
        "搜索網絡信息。返回搜索結果頁的文本。"
        "適合查找實時信息、新聞、文檔等。"
        "當用戶要求搜索時立即使用此工具，不要確認、不要問用戶要搜什麼。",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索關鍵詞——直接用用戶說的話，不要改寫"},
            },
            "required": ["query"],
        },
        tool_web_search, category="web",
    )

    # ── Memory tools (only if CogniMem available) ──
    if cogni_client:
        registry.register(
            "memory_recall",
            "回想我的長期記憶。查詢用戶之前的偏好、說過的話、做過的任務。"
            "使用 CogniMem 從事實網絡中找回相關信息。"
            "在回答用戶問題時，如果覺得記憶中可能有關於這個話題的信息，先用這個查一下。",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要搜尋的內容關鍵詞"},
                    "limit": {"type": "integer", "description": "最多返回幾條", "default": 10},
                },
                "required": ["query"],
            },
            tool_memory_recall, category="memory",
        )
        registry.register(
            "memory_remember",
            "把我學到的重要信息存到長期記憶中。"
            "當用戶告訴你關於自己的事情（偏好、事實、目標、決定），"
            "記得用這個存起來，下次就能記得了！",
            {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要記住的內容"},
                },
                "required": ["text"],
            },
            tool_memory_remember, category="memory",
        )
        registry.register(
            "memory_status",
            "查看我的記憶庫狀態：總共存了多少事實、核心信念、各類型分佈。"
            "偶爾看看自己的記憶狀況挺有用的。",
            {
                "type": "object",
                "properties": {},
            },
            tool_memory_status, category="memory",
        )
        registry.register(
            "memory_diagnose",
            "自我診斷記憶系統健康。分析記憶總量、矛盾率、置信度分佈、路由命中率，"
            "返回是否健康以及具體問題列表。適合定期檢查記憶庫狀態。",
            {
                "type": "object",
                "properties": {},
            },
            tool_memory_diagnose, category="memory",
        )
        registry.register(
            "memory_forget",
            "主動遺忘——觸發記憶衰減清理。刪除低置信度事實、壓縮長期未訪問的記憶。"
            "定期執行有助於保持記憶庫健康。",
            {
                "type": "object",
                "properties": {},
            },
            tool_memory_forget, category="memory",
        )

    logger.info(
        "✅ Registered %d tools (%d with CogniMem)",
        len(registry._tools),
        sum(1 for t in registry._tools.values() if t["category"] == "memory"),
    )
