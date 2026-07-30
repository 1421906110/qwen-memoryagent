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

🔥 v0.17: 改用 @tool 装饰器自动生成 Schema（零依赖）
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory_agent.agent import AgentContext

from memory_agent.agent.registry import tool, ToolRegistry
from memory_agent.agent.risk import RiskClass

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


def _mark_domain_failed(url: str, is_network_error: bool = False) -> None:
    """记录域名失败次数。网络级错误（超时/连接拒绝等）不会屏蔽域名。"""
    if is_network_error:
        return  # 网络级错误不屏蔽域名（可能是代理/网络问题，不是域名问题）
    from urllib.parse import urlparse
    try:
        domain = urlparse(url).netloc
        _FAILED_DOMAINS[domain] = _FAILED_DOMAINS.get(domain, 0) + 1
        if _FAILED_DOMAINS[domain] >= 2:
            logger.info("⛔ Domain blocked (2+ HTTP errors): %s", domain)
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
    content = args.get("content", "")

    # Safety check
    for forbidden in ["/etc", "/sys", "/proc", "/dev"]:
        if str(path).startswith(forbidden):
            return {"error": f"Refusing to write to system path: {forbidden}"}

    # ⭐ 拒绝空内容：防止 LLM 打嘴炮说"已写入"但实际写空文件
    if not content or not content.strip():
        return {
            "error": "写入内容为空。请提供完整的文件内容再调用 write_file。",
            "path": str(path),
            "content_empty": True,
        }

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

    使用 httpx 直连抓取（优先），如果 SEARCH_PROXY 配置且直连失败则尝试代理。
    代理也失败时不屏蔽域名（可能是代理问题不是域名问题）。
    结果会缓存当前 session（LRU, max 30 URLs）。
    连续 2 次 HTTP 4xx/5xx 错误阻塞域名。
    """
    url = args["url"]

    # 跳过已知失败的域名（仅限 HTTP 级错误，非网络级）
    if _is_domain_blocked(url):
        return {"error": f"Skipped (domain previously failed 2+ times): {url}"}

    # 检查缓存
    cached = _cache_get(url)
    if cached:
        cached["from_cache"] = True
        return cached

    proxy = os.environ.get("SEARCH_PROXY", "")
    import httpx

    client_kwargs = {
        "timeout": httpx.Timeout(_WEB_TIMEOUT),
        "follow_redirects": True,
        "headers": {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
    }

    # ── 策略 1: 有代理时先试代理 ──
    last_error = ""
    if proxy:
        _proxy_url = proxy
        if proxy.startswith("socks"):
            _proxy_url = proxy.replace("socks5://", "http://").replace("socks5h://", "http://")
        try:
            _pk = dict(client_kwargs)
            _pk["proxy"] = _proxy_url
            with httpx.Client(**_pk) as client:
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
        except httpx.HTTPStatusError as e:
            # HTTP 级错误（4xx/5xx）— 可能是域名的问题，标记 + 直连兜底
            last_error = f"HTTP {e.response.status_code}"
            _mark_domain_failed(url)
            logger.info("📡 代理返回 %s for %s, 尝试直连", e.response.status_code, url)
        except Exception as e:
            # 网络级错误（连接超时/拒绝等）— 不屏蔽域名，直连兜底
            last_error = str(e)[:60]
            logger.info("📡 代理抓取 %s 失败(%s)，直连重试", url, last_error)

    # ── 策略 2: 直连（无代理 / 代理失败后的兜底） ──
    try:
        _dk = dict(client_kwargs)
        _dk.pop("proxy", None)  # 确保不用代理
        with httpx.Client(**_dk) as client:
            resp = client.get(url)
        resp.raise_for_status()
        content = resp.text[:8000]
        result = {
            "url": url, "content": content, "length": len(content),
            "truncated": len(resp.text) > 8000,
            "status_code": resp.status_code,
            "via_proxy": bool(proxy),
        }
        if last_error:
            result["_retry"] = "proxy_failed_direct_ok"
        _cache_set(url, result)
        return result
    except httpx.HTTPStatusError as e:
        _mark_domain_failed(url)
        result = {"error": f"HTTP {e.response.status_code}: {url}"}
        _cache_set(url, result)
        return result
    except Exception as e:
        # 直连也失败 → 真实网络问题
        err_msg = str(e)[:80]
        if "Name or service not known" in err_msg or "nodename nor servname" in err_msg:
            result = {"error": f"DNS 解析失败（大陆可能无法直连该站点）: {url[:40]}"}
        elif "ConnectError" in type(e).__name__ or "Connection refused" in err_msg:
            result = {"error": f"连接被拒绝（站点可能被墙）: {url[:40]}"}
        elif "timed out" in err_msg or "Timeout" in type(e).__name__:
            result = {"error": f"连接超时（站点太慢或被墙）: {url[:40]}"}
        else:
            result = {"error": f"抓取失败: {err_msg}"}
        _cache_set(url, result)
        return result


# ═══════════════════════════════════════════════════════════════════════
#  Web Search Providers — OpenWorker 风格，DuckDuckGo 免费通用搜索
#  ═══════════════════════════════════════════════════════════════════════


def _search_ddg(query: str, max_results: int = 8,
                allowed_domains: list[str] | None = None,
                timeout: float = 6.0) -> list[dict]:
    """DuckDuckGo 搜索（免费无 API Key，带超时控制）

    Return shape: [{"title": str, "url": str, "snippet": str}, ...]
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutTimeout

    q = query
    if allowed_domains:
        q = query + ' ' + ' '.join(f'site:{d}' for d in allowed_domains)

    def _run() -> list[dict]:
        from ddgs import DDGS
        results = []
        with DDGS() as client:
            for i, hit in enumerate(client.text(q, max_results=max_results)):
                if i >= max_results:
                    break
                results.append({
                    "title": str(hit.get("title", "")),
                    "url": str(hit.get("href", "") or hit.get("url", "")),
                    "snippet": str(hit.get("body", "")),
                })
        return results

    try:
        with ThreadPoolExecutor(1) as pool:
            fut = pool.submit(_run)
            results = fut.result(timeout=timeout)
        if results:
            logger.info("🦆 DDGS 搜索成功: %d 条结果", len(results))
        return results
    except _FutTimeout:
        logger.info("DDGS 超时(>%ss)，跳过", timeout)
        return []
    except Exception as e:
        logger.info("DDGS 搜索失败: %s", str(e)[:60])
        return []


def _search_baidu(query: str, max_results: int = 8) -> list[dict]:
    """百度搜索（国内 ECS 兜底）"""
    import httpx
    import re
    import urllib.parse

    q = urllib.parse.quote(query)
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/120.0.0.0",
        "Accept-Language": "zh-CN,zh;q=0.9",
    }
    try:
        with httpx.Client(timeout=8, follow_redirects=True) as client:
            resp = client.get(
                f"https://www.baidu.com/s?wd={q}&ie=utf-8&rn={max_results}",
                headers=headers,
            )
        if resp.status_code != 200:
            return []

        results = []
        blocks = re.findall(
            r'<div[^>]*class="[^"]*c-container[^"]*"[^>]*>(.*?)</div>',
            resp.text, re.DOTALL,
        )
        for block in blocks[:max_results]:
            tm = re.search(r'<h3[^>]*>.*?<a[^>]*>(.*?)</a>', block, re.DOTALL)
            title = re.sub(r'<[^>]+>', '', tm.group(1)).strip() if tm else ''
            um = re.search(r'<a[^>]*href="(https?://[^"]+)"', block)
            url = um.group(1) if um else ''
            # 摘要
            snippet = ''
            for pat in [
                r'<span[^>]*class="[^"]*content-right[^"]*"[^>]*>(.*?)</span>',
                r'<div[^>]*class="[^"]*c-abstract[^"]*"[^>]*>(.*?)</div>',
            ]:
                sm = re.search(pat, block, re.DOTALL)
                if sm:
                    snippet = re.sub(r'<[^>]+>', '', sm.group(1)).strip()
                    snippet = re.sub(r'[—\-]{2,}.*$', '', snippet).strip()
                    break
            if title:
                results.append({"title": title, "url": url, "snippet": snippet})
        if results:
            logger.info("🔍 百度搜索成功: %d 条结果", len(results))
        return results
    except Exception as e:
        logger.info("百度搜索失败: %s", str(e)[:60])
        return []


def _search_bing(query: str, max_results: int = 8) -> list[dict]:
    """Bing 搜索（最后保底）"""
    import httpx
    import re
    import urllib.parse

    q = urllib.parse.quote(query)
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/120.0.0.0",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.5",
    }
    try:
        with httpx.Client(timeout=8, follow_redirects=True) as client:
            resp = client.get(
                f"https://www.bing.com/search?q={q}&setlang=zh-CN",
                headers=headers,
            )
        if resp.status_code != 200:
            return []

        results = []
        for block in re.findall(
            r'<li[^>]*class="b_algo[^"]*"[^>]*>(.*?)</li>',
            resp.text, re.DOTALL,
        ):
            block = re.sub(r'<link[^>]*>', '', block)
            am = re.search(r'<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>', block)
            if not am:
                continue
            url = am.group(1)
            title = re.sub(r'<[^>]+>', '', am.group(2)).strip()
            # 摘要
            snippet = ''
            for pat in [
                r'class="b_caption"[^>]*>.*?<p[^>]*class="b_lineclamp2"[^>]*>(.*?)</p>',
                r'class="b_caption"[^>]*>.*?<p[^>]*>(.*?)</p>',
            ]:
                sm = re.search(pat, block, re.DOTALL)
                if sm:
                    snippet = re.sub(r'<[^>]+>', '', sm.group(1)).strip()
                    break
            if title and len(title) > 2:
                results.append({"title": title, "url": url, "snippet": snippet})
            if len(results) >= max_results:
                break
        if results:
            logger.info("🌐 Bing 搜索成功: %d 条结果", len(results))
        return results
    except Exception as e:
        logger.info("Bing 搜索失败: %s", str(e)[:60])
        return []


def _try_lottery_api(query: str) -> dict | None:
    """彩票直连 API（双色球官方数据源）"""
    if not any(kw in query for kw in ["双色球", "ssq", "福彩"]):
        return None
    try:
        import httpx
        with httpx.Client(timeout=8, follow_redirects=True) as client:
            resp = client.get(
                "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/"
                "findDrawNotice?name=ssq&issueCount=10",
                headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "zh-CN"},
            )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get("state") != 0:
            return None
        results = data.get("result", [])
        if not results:
            return None
        lines = []
        for draw in results[:10]:
            lines.append(
                f"第{draw['code']}期（{draw['date']}）"
                f"红球:{draw['red']} 蓝球:{draw['blue']}"
            )
        logger.info("🎯 彩票 API 直连: %d 期", len(lines))
        return {"result": "\n".join(lines), "source": "lottery_api", "query": query}
    except Exception as e:
        logger.info("彩票 API 不可用: %s", e)
        return None


def tool_web_search(tool_call_id: str, args: dict,
                    ctx: "AgentContext") -> dict:
    """Search the web and return structured results (title + URL + snippet).

    搜索策略：
      1️⃣  DuckDuckGo（免费无 Key，6s 超时，行就行不行拉倒）
      2️⃣  Bing（国内 ECS 实测最快，主战引擎）
      3️⃣  百度（中文内容兜底）
      特殊：双色球走福彩官方 API 直连
    """
    query = args.get("query", "").strip()
    if not query:
        return {"error": "搜索关键词不能为空"}

    max_results = min(args.get("max_results", 8), 15)
    allowed_domains = args.get("allowed_domains")

    # 0. 彩票直连（双色球专用路径）
    lottery = _try_lottery_api(query)
    if lottery:
        return lottery

    # 1. DuckDuckGo 快速尝试（6s 超时，网络好时用）
    ddg = _search_ddg(query, max_results, allowed_domains)
    if ddg:
        return {"results": ddg, "source": "duckduckgo", "query": query, "count": len(ddg)}

    # 2. Bing（国内 ECS 实测最快，主战引擎）
    bing = _search_bing(query, max_results)
    if bing:
        return {"results": bing, "source": "bing", "query": query, "count": len(bing)}

    # 3. 百度兜底（中文内容）
    baidu = _search_baidu(query, max_results)
    if baidu:
        return {"results": baidu, "source": "baidu", "query": query, "count": len(baidu)}

    return {"error": "搜索暂不可用，请稍后重试", "query": query}


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
        # 🐛 v0.27 修复：Agent 工具存记忆后刷新快照
        ctx.cogni.refresh_snapshot(ctx.agent_id, session_id=ctx.session_id)
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
#  LAUNCH APP TOOL — 打开本地应用（macOS）
# ===================================================================

def tool_launch_app(tool_call_id: str, args: dict,
                     ctx: "AgentContext") -> dict:
    """打开 macOS 本地应用。适合用户说「打开微信」「打开浏览器」「打开计算器」等。

    使用 macOS 的 `open -a` 命令启动已安装的应用程序。
    如果应用已在运行，会切换到该应用。
    """
    app_name = args.get("app_name", "").strip()
    if not app_name:
        return {"error": "请指定要打开的 App 名称，如 微信、Safari、Chrome"}

    try:
        r = subprocess.run(
            ["open", "-a", app_name],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            logger.info("🚀 已打开应用: %s", app_name)
            return {"success": True, "app": app_name, "message": f"已打开 {app_name}"}
        else:
            err = r.stderr or r.stdout or "未知错误"
            return {"error": f"打开 {app_name} 失败: {err[:100]}"}
    except FileNotFoundError:
        return {"error": f"未找到应用 {app_name}，请确认是否已安装"}
    except subprocess.TimeoutExpired:
        return {"error": f"打开 {app_name} 超时"}
    except Exception as e:
        return {"error": str(e)[:100]}


# ===================================================================
#  BROWSER TOOL — ego-browser 集成（Chromium AI 浏览器）
# ===================================================================

def tool_browser_open(tool_call_id: str, args: dict,
                       ctx: "AgentContext") -> dict:
    """在 ego-browser 中打开网页并返回页面内容。

    ego-browser 是一个 Chromium 浏览器，能渲染 JavaScript、
    处理登录态、截图。适合抓取需要 JS 渲染的页面。

    如未安装 ego-browser，会降级使用普通 web_fetch。
    """
    url = args.get("url", "").strip()
    if not url:
        return {"error": "请提供 URL"}
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # 检查 ego-browser 是否可用
    import shutil
    if not shutil.which("ego-browser"):
        # 降级到普通 web_fetch
        logger.info("ego-browser 未安装，降级到 web_fetch")
        return tool_web_fetch(tool_call_id, {"url": url}, ctx)

    try:
        script = f"""
const task = await useOrCreateTaskSpace('browser_open')
await openOrReuseTab('{url}', {{ wait: true, timeout: 15 }})
const info = await pageInfo()
const text = await snapshotText()
await completeTaskSpace(task.name, {{ keep: false }})
cliLog(JSON.stringify({{ url: info.url, title: info.title, text }}))
"""
        r = subprocess.run(
            ["ego-browser", "nodejs"],
            input=script, text=True, capture_output=True,
            timeout=30,
        )
        if r.returncode != 0:
            err = r.stderr[:200]
            logger.warning("ego-browser 失败: %s，降级到 web_fetch", err)
            return tool_web_fetch(tool_call_id, {"url": url}, ctx)

        import json as _json
        try:
            result = _json.loads(r.stdout.strip())
            content = result.get("text", "")
            return {
                "url": result.get("url", url),
                "title": result.get("title", ""),
                "content": content[:8000],
                "truncated": len(content) > 8000,
                "source": "ego-browser",
            }
        except (_json.JSONDecodeError, ValueError):
            return {
                "url": url,
                "content": r.stdout[:8000],
                "source": "ego-browser_raw",
            }
    except subprocess.TimeoutExpired:
        logger.warning("ego-browser 超时，降级到 web_fetch")
        return tool_web_fetch(tool_call_id, {"url": url}, ctx)
    except Exception as e:
        logger.warning("ego-browser 异常: %s，降级到 web_fetch", str(e)[:60])
        return tool_web_fetch(tool_call_id, {"url": url}, ctx)


# ===================================================================
#  SEARCH FILES TOOL — grep 文件内容搜索
# ===================================================================

def tool_search_files(tool_call_id: str, args: dict,
                      ctx: "AgentContext") -> dict:
    """Search file contents with regex or plain text (grep)."""
    pattern = args["pattern"]
    path = args.get("path", ".")
    max_results = min(args.get("max_results", 20), 50)
    use_regex = args.get("use_regex", False)

    logger.info("🔍 Search files: %r in %s", pattern[:100], path)

    try:
        import subprocess
        cmd = ["grep", "-rn"]
        if not use_regex:
            cmd.append("-F")  # fixed string
        cmd.extend(["--include=*", "-l"])  # list files only first
        cmd.extend([pattern, path])

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15,
        )
        files = [f for f in result.stdout.strip().split("\n") if f.strip()][:max_results]

        # Get actual matches for found files
        matches = []
        for filepath in files:
            cmd2 = ["grep", "-n"]
            if not use_regex:
                cmd2.append("-F")
            cmd2.extend([pattern, filepath])
            r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=10)
            for line in r2.stdout.strip().split("\n"):
                if line.strip():
                    matches.append(line[:200])
                    if len(matches) >= max_results:
                        break
            if len(matches) >= max_results:
                break

        return {
            "success": True,
            "files_found": len(files),
            "matches": matches,
            "truncated": len(matches) >= max_results,
        }
    except subprocess.TimeoutExpired:
        return {"error": "Search timed out", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


# ===================================================================
#  GET TIME TOOL — 获取当前日期时间
# ===================================================================

def tool_get_current_time(tool_call_id: str, args: dict,
                           ctx: "AgentContext") -> dict:
    """Get the current date and time on this system."""
    try:
        import subprocess
        r = subprocess.run(
            ["date", "+%Y-%m-%d %H:%M:%S %A %Z"],
            capture_output=True, text=True, timeout=5,
        )
        date_str = r.stdout.strip()
        import locale
        locale.setlocale(locale.LC_TIME, "zh_CN.UTF-8")
        try:
            r_zh = subprocess.run(
                ["date", "+%Y年%m月%d日 星期%w %H:%M"],
                capture_output=True, text=True, timeout=5,
            )
            weekday_map = {"0": "日", "1": "一", "2": "二", "3": "三",
                           "4": "四", "5": "五", "6": "六"}
            date_zh = r_zh.stdout.strip()
            for k, v in weekday_map.items():
                date_zh = date_zh.replace(f"星期{k}", f"星期{v}")
        except Exception:
            date_zh = date_str

        return {
            "datetime": date_str,
            "datetime_zh": date_zh,
        }
    except Exception as e:
        return {"error": str(e)}


# ===================================================================
#  ASK USER TOOL — 向用户提问（重要：自主 Agent 需要）
# ===================================================================

def tool_ask_user(tool_call_id: str, args: dict,
                   ctx: "AgentContext") -> dict:
    """Ask the user a question when you need clarification or input to proceed.

    只有当真正卡住了、需要用户决策时才用。不要滥用——能自己推的不要问。
    """
    question = args.get("question", "")
    if not question:
        return {"error": "Question is required"}

    # 把问题存到会话上下文，前端轮询时返回
    ask_ctx = getattr(ctx, "_pending_question", None)
    ctx._pending_question = {
        "question": question,
        "timestamp": datetime.now().isoformat(),
    }
    logger.info("❓ Ask user: %s", question[:200])

    # ⭐ 返回格式化的等待响应
    return {
        "asked": True,
        "question": question,
        "instruction": "等待用户回答后再继续",
        "answer": None,
    }


# ===================================================================
#  PRIVATE THINKING TOOL — 在内部推理，不输出给用户
#  比「禁止输出思考过程」更优雅：给模型一个私密通道
# ===================================================================

# 全局思维暂存区（不持久化，仅当前会话有效）
_THINK_BUFFER: dict[str, list[str]] = {}

def tool_think(tool_call_id: str, args: dict,
               ctx: "AgentContext") -> dict:
    """Think privately. Use this to reason about complex tasks step by step.

    这是你的私人思维空间——在这里推理、分析、权衡，用户不会看到。
    用完后直接输出最终回复即可，不用把这里的思考过程复述出来。
    适合：拆解复杂问题、分析利弊、计划步骤、检查答案。
    """
    thought = args.get("thought", "")
    session_id = getattr(ctx, "session_id", "default")
    if session_id not in _THINK_BUFFER:
        _THINK_BUFFER[session_id] = []
    _THINK_BUFFER[session_id].append(thought)
    # 只保留最近 20 条思考
    _THINK_BUFFER[session_id] = _THINK_BUFFER[session_id][-20:]

    logger.debug("💭 Think (%s): %s", session_id, thought[:100])
    return {"status": "noted", "thoughts_so_far": len(_THINK_BUFFER[session_id])}


# ===================================================================
#  TASK / TODO MANAGEMENT — 多步骤任务规划和进度跟踪
#  每个 agent 必备。没有它 agent 记不住自己做到哪了。
# ===================================================================

_TASK_STORE: dict[str, list[dict]] = {}

def tool_todo(tool_call_id: str, args: dict,
              ctx: "AgentContext") -> dict:
    """管理任务列表。多步骤复杂任务开始前先创建任务列表，跟踪进度。

    用法：
    - action=list → 查看当前任务
    - action=create → 创建一组任务（items 是任务数组）
    - action=update → 更新某个任务状态（id + status）
    - action=done → 标记所有任务完成
    """
    action = args.get("action", "list")
    session_id = getattr(ctx, "session_id", "default")

    if session_id not in _TASK_STORE:
        _TASK_STORE[session_id] = []

    tasks = _TASK_STORE[session_id]

    if action == "create":
        items = args.get("items", [])
        if not items:
            return {"error": "items required for create"}
        new_tasks = []
        for i, item in enumerate(items):
            task = {
                "id": f"t{len(tasks) + i + 1}",
                "title": item.get("title", item if isinstance(item, str) else ""),
                "status": item.get("status", "pending"),
            }
            new_tasks.append(task)
        tasks.extend(new_tasks)
        _TASK_STORE[session_id] = tasks
        return {
            "status": "created",
            "total": len(tasks),
            "pending": sum(1 for t in tasks if t["status"] == "pending"),
            "in_progress": sum(1 for t in tasks if t["status"] == "in_progress"),
            "done": sum(1 for t in tasks if t["status"] == "done"),
        }

    elif action == "update":
        task_id = args.get("id", "")
        new_status = args.get("status", "")
        if not task_id or not new_status:
            return {"error": "id and status required"}
        if new_status not in ("pending", "in_progress", "done"):
            return {"error": "status must be: pending, in_progress, done"}
        for t in tasks:
            if t["id"] == task_id:
                t["status"] = new_status
                break
        else:
            return {"error": f"Task {task_id} not found"}
        return {
            "status": f"updated {task_id} → {new_status}",
            "tasks": tasks,
        }

    elif action == "done":
        for t in tasks:
            t["status"] = "done"
        return {"status": "all done", "total": len(tasks)}

    else:  # list
        if not tasks:
            return {"tasks": [], "total": 0, "message": "暂无任务"}
        return {
            "tasks": tasks,
            "total": len(tasks),
            "pending": sum(1 for t in tasks if t["status"] == "pending"),
            "in_progress": sum(1 for t in tasks if t["status"] == "in_progress"),
            "done": sum(1 for t in tasks if t["status"] == "done"),
        }


# ===================================================================
#  TOOL REGISTRATION
# ===================================================================

def register_all_tools(registry: ToolRegistry = None, cogni_client=None):
    """Register all built-in tools with the registry.

    🔥 v0.17: 改用新的 ToolRegistry.register() 接口
    函数签名自动生成 schema（零依赖）

    Args:
        registry: ToolRegistry 实例（默认使用全局 _registry）
        cogni_client: CogniMem 客户端（可选，记忆工具需要）

    Call once during app initialization.
    """
    r = registry or ToolRegistry()

    # ── File tools ──
    r.register(
        name="read_file",
        description="读取文件内容。适合查看源码、日志、配置文件等。返回带行号的内容。"
                    "如果文件很大可以用 offset/limit 分段读取。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file"},
                "offset": {"type": "integer", "description": "Line number to start from (0-indexed)", "default": 0},
                "limit": {"type": "integer", "description": "Max lines to read (max 5000)", "default": 2000},
            },
            "required": ["path"],
        },
        executor=tool_read_file, category="file",
    )
    r.register(
        name="write_file",
        description="写入文件（覆蓋模式）。自動創建父目錄。"
                    "適合創建新文件、保存修改後的內容。注意：會完全覆蓋原文件！",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file"},
                "content": {"type": "string", "description": "Full file content to write"},
            },
            "required": ["path", "content"],
        },
        executor=tool_write_file, category="file",
    )
    r.register(
        name="edit_file",
        description="精準編輯文件——替換指定的 old_string 為 new_string。"
                    "適合小幅修改（改一行、改一個詞），比 write_file 安全。"
                    "注意：old_string 必須在文件中唯一。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file"},
                "old_string": {"type": "string", "description": "The exact text to replace (must be unique)"},
                "new_string": {"type": "string", "description": "The replacement text"},
            },
            "required": ["path", "old_string"],
        },
        executor=tool_edit_file, category="file",
    )
    r.register(
        name="list_dir",
        description="列出目錄內容。支持 glob 模式過濾（如 *.py）。"
                    "適合先查看目錄結構再決定讀哪個文件。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to list"},
                "pattern": {"type": "string", "description": "Glob filter (e.g. '*.py')", "default": "*"},
            },
            "required": ["path"],
        },
        executor=tool_list_dir, category="file",
    )

    # ── Shell tool ──
    r.register(
        name="shell",
        description="執行 shell 命令。適合運行腳本、git、pip、npm、python 等。"
                    "注意：命令在服務器上直接執行，不要用 rm -rf 等危險操作。"
                    "如果只是查看文件用 read_file 更安全。",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (max 120)", "default": 30},
            },
            "required": ["command"],
        },
        executor=tool_shell, category="shell",
    )

    # ── Web tools ──
    r.register(
        name="web_fetch",
        description="抓取指定 URL 的內容。適合讀取網頁、REST API 等。"
                    "返回純文本（自動截斷到 8000 字符）。URL 會緩存，同一 URL 不重複抓。",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch (http/https)"},
            },
            "required": ["url"],
        },
        executor=tool_web_fetch, category="web",
    )
    r.register(
        name="web_search",
        description="搜索網絡信息，返回結構化結果（標題+URL+摘要）。"
                    "適合查找實時信息、新聞、文檔、代碼等。"
                    "當用戶要求搜索時立即使用此工具，不要確認、不要問用戶要搜什麼。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索關鍵詞——直接用用戶說的話，不要改寫"},
                "max_results": {"type": "integer", "description": "最多返回幾條結果（默認8，最多15）", "default": 8},
                "allowed_domains": {
                    "type": "array", "items": {"type": "string"},
                    "description": "限定搜索域名，如 ['github.com', 'stackoverflow.com']",
                },
            },
            "required": ["query"],
        },
        executor=tool_web_search, category="web",
    )

    # ── App tools ──
    r.register(
        name="launch_app",
        description="打開 macOS 本地應用。如「打開微信」「打開瀏覽器」「打開計算器」等。"
                    "使用 open -a 命令啟動已安裝的程序，已在運行的會切換到前台。",
        parameters={
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "App 名稱（如 微信、Safari、Chrome、Calculator）",
                },
            },
            "required": ["app_name"],
        },
        executor=tool_launch_app, category="utility",
    )
    r.register(
        name="browser_open",
        description="在 ego-browser（Chromium AI 瀏覽器）中打開網頁並查看內容。"
                    "能渲染 JavaScript、顯示登錄后的頁面。"
                    "適合抓取需要 JS 渲染、需要登錄、或 web_fetch 無法正常抓取的網站。"
                    "如果 ego-browser 未安裝，會自動降級到普通 web_fetch。",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "要打開的網址（http/https）"},
            },
            "required": ["url"],
        },
        executor=tool_browser_open, category="web",
    )

    # ── Search tool ──
    r.register(
        name="search_files",
        description="搜索文件內容（grep）。在指定路径中用关键词或正則搜索文件內容。"
                    "適合找代碼片段、配置項、日誌中的關鍵字。",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "搜索关键词或正则"},
                "path": {"type": "string", "description": "搜索路径（默认当前目录）", "default": "."},
                "max_results": {"type": "integer", "description": "最多返回几条结果", "default": 20},
                "use_regex": {"type": "boolean", "description": "是否用正则表达式", "default": False},
            },
            "required": ["pattern"],
        },
        executor=tool_search_files, category="file",
    )

    # ── Utility tools ──
    r.register(
        name="get_current_time",
        description="獲取當前系統的日期和時間。回答「今天幾號」「現在幾點」等問題時用。"
                    "比 shell date 更安全、更直接。",
        parameters={
            "type": "object",
            "properties": {},
        },
        executor=tool_get_current_time, category="utility",
    )
    r.register(
        name="ask_user",
        description="向用戶提問。當你卡住了、需要用戶做決策或提供額外信息時用。"
                    "注意：能自己推斷的就不要問，濫用這個工具會讓用戶覺得你很笨。"
                    "只有真正不確定、無法推進時才問。",
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "你要問用戶的問題——清晰具體，附上選項更好"},
            },
            "required": ["question"],
        },
        executor=tool_ask_user, category="utility",
    )
    r.register(
        name="think",
        description="💭 私人思考空間。在處理複雜任務時，用這個工具在內部逐步推理、"
                    "分析利弊、檢查答案。用戶看不到你的思考內容。"
                    "想完直接輸出最終回復就行，不用把思考過程復述出來。"
                    "適合：拆解問題、計劃步驟、檢查答案是否完整。",
        parameters={
            "type": "object",
            "properties": {
                "thought": {"type": "string", "description": "你的思考內容——推理過程、分析、計劃"},
            },
            "required": ["thought"],
        },
        executor=tool_think, category="utility",
    )
    r.register(
        name="todo",
        description="📋 任務管理。多步驟任務開始前，先用 todo action=create 創建任務列表。"
                    "做完一步用 todo action=update id=xxx status=done 更新進度。"
                    "重要：多步驟工作必須先創建 todo 再執行，讓用戶看到進度。",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "update", "done"],
                    "description": "list=查看 / create=創建任務 / update=更新狀態 / done=全部完成",
                },
                "items": {
                    "type": "array",
                    "description": "action=create 時必填：任務數組 [{\"title\": \"第一步\", \"status\": \"pending\"}]",
                    "items": {"type": "object"},
                    "default": [],
                },
                "id": {
                    "type": "string",
                    "description": "action=update 時必填：任務ID（如 t1, t2）",
                    "default": "",
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "done"],
                    "description": "action=update 時必填：新狀態",
                    "default": "",
                },
            },
            "required": ["action"],
        },
        executor=tool_todo, category="utility",
    )

    # ── Memory tools (only if CogniMem available) ──
    if cogni_client:
        r.register(
            name="memory_recall",
            description="回想我的長期記憶。查詢用戶之前的偏好、說過的話、做過的任務。"
                        "使用 CogniMem 從事實網絡中找回相關信息。"
                        "在回答用戶問題時，如果覺得記憶中可能有關於這個話題的信息，先用這個查一下。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要搜尋的內容關鍵詞"},
                    "limit": {"type": "integer", "description": "最多返回幾條", "default": 10},
                },
                "required": ["query"],
            },
            executor=tool_memory_recall, category="memory",
        )
        r.register(
            name="memory_remember",
            description="把我學到的重要信息存到長期記憶中。"
                        "當用戶告訴你關於自己的事情（偏好、事實、目標、決定），"
                        "記得用這個存起來，下次就能記得了！",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要記住的內容"},
                },
                "required": ["text"],
            },
            executor=tool_memory_remember, category="memory",
        )
        r.register(
            name="memory_status",
            description="查看我的記憶庫狀態：總共存了多少事實、核心信念、各類型分佈。"
                        "偶爾看看自己的記憶狀況挺有用的。",
            parameters={
                "type": "object",
                "properties": {},
            },
            executor=tool_memory_status, category="memory",
        )
        r.register(
            name="memory_diagnose",
            description="自我診斷記憶系統健康。分析記憶總量、矛盾率、置信度分佈、路由命中率，"
                        "返回是否健康以及具體問題列表。適合定期檢查記憶庫狀態。",
            parameters={
                "type": "object",
                "properties": {},
            },
            executor=tool_memory_diagnose, category="memory",
        )
        r.register(
            name="memory_forget",
            description="主動遺忘——觸發記憶衰減清理。刪除低置信度事實、壓縮長期未訪問的記憶。"
                        "定期執行有助於保持記憶庫健康。",
            parameters={
                "type": "object",
                "properties": {},
            },
            executor=tool_memory_forget, category="memory",
        )

    logger.info(
        "✅ Registered %d tools (%d with CogniMem)",
        len(r._tools),
        sum(1 for t in r._tools.values() if t["category"] == "memory"),
    )
