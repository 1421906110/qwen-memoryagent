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


def _parse_bing_results(html: str) -> list[str]:
    """从 Bing HTML 中提取搜索结果标题+摘要+URL"""
    import re as _re
    texts = []

    # 主结果：b_algo 块（标准搜索结果）
    for block in _re.findall(r'<li[^>]*class="b_algo[^"]*"[^>]*>(.*?)</li>',
                              html, _re.DOTALL):
        # 先去掉 <link> CSS 标签（Bing 页面在 b_algo 里插了多个 link）
        block = _re.sub(r'<link[^>]*>', '', block)

        # 标题 + URL（只找 <a>，跳过 <link>）
        title = ""
        url = ""
        a_m = _re.search(r'<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>', block)
        if a_m:
            url = a_m.group(1)
            title = _re.sub(r'<[^>]+>', '', a_m.group(2)).strip()
        if not title:
            title_m = _re.search(r'<a[^>]*>(.*?)</a>', block)
            if title_m:
                title = _re.sub(r'<[^>]+>', '', title_m.group(1)).strip()

        # 摘要（优先 b_caption > b_lineclamp2 > 普通 p）
        body = ""
        for pat in [
            r'class="b_caption"[^>]*>.*?<p[^>]*class="b_lineclamp2"[^>]*>(.*?)</p>',
            r'class="b_caption"[^>]*>.*?<p[^>]*>(.*?)</p>',
            r'<p[^>]*>(.*?)</p>',
        ]:
            body_m = _re.search(pat, block, _re.DOTALL)
            if body_m:
                body = _re.sub(r'<[^>]+>', '', body_m.group(1)).strip()
                if len(body) > 10:
                    break

        if title and len(title) > 2:
            if body:
                texts.append(f"【{title}】{body} ——{url}" if url else f"【{title}】{body}")
            else:
                texts.append(f"【{title}】 ——{url}" if url else f"【{title}】")

    # 兜底：如果 b_algo 没抓到，用旧方式
    if not texts:
        for m in _re.findall(r'<p[^>]*class="b_lineclamp[^"]*"[^>]*>(.*?)</p>', html):
            clean = _re.sub(r'<[^>]+>', '', m).strip()
            if clean and len(clean) > 10:
                texts.append(clean)
        for m in _re.findall(r'<h2>.*?<a[^>]*>(.*?)</a>', html):
            clean = _re.sub(r'<[^>]+>', '', m).strip()
            if clean:
                texts.append(clean)

    return texts[:10]


def _parse_baidu_results(html: str) -> list[str]:
    """从百度搜索结果 HTML 中提取标题+摘要+URL"""
    import re
    texts = []
    # 百度结果容器: <div class="result c-container"> 或 <div class="c-container">
    # 标题: <h3 class="t"> 内 <a> 的文本
    # 摘要: <span class="content-right_..."> 或 <div class="c-abstract">
    # URL: <a> 的 href

    # 提取所有结果块
    blocks = re.findall(
        r'<div[^>]*class="[^"]*(?:result\s+c-container|c-container)[^"]*"[^>]*>'
        r'(.*?)'
        r'</div>\s*(?=<div[^>]*class="[^"]*(?:result\s+c-container|c-container)|<div[^>]*id="page")',
        html, re.DOTALL
    )
    if not blocks:
        # fallback: 更宽松的匹配
        blocks = re.findall(
            r'<div[^>]*class="[^"]*c-container[^"]*"[^>]*>(.*?)</div>',
            html, re.DOTALL
        )

    for block in blocks[:10]:
        # 标题
        title_match = re.search(r'<h3[^>]*>.*?<a[^>]*>(.*?)</a>', block, re.DOTALL)
        title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip() if title_match else ''
        # URL
        url_match = re.search(r'<a[^>]*href="(https?://[^"]+)"', block)
        url = url_match.group(1) if url_match else ''
        # 摘要
        abstract = ''
        abs_match = re.search(r'<span[^>]*class="[^"]*content-right[^"]*"[^>]*>(.*?)</span>', block, re.DOTALL)
        if not abs_match:
            abs_match = re.search(r'<div[^>]*class="[^"]*c-abstract[^"]*"[^>]*>(.*?)</div>', block, re.DOTALL)
        if not abs_match:
            abs_match = re.search(r'<div[^>]*class="[^"]*c-span-last[^"]*"[^>]*>(.*?)</div>', block, re.DOTALL)
        if abs_match:
            abstract = re.sub(r'<[^>]+>', '', abs_match.group(1)).strip()
            # 去掉末尾的 ... 和百度快照等
            abstract = re.sub(r'[—\-]{2,}.*$', '', abstract).strip()

        if title:
            line = title
            if url:
                line += f' [{url}]'
            if abstract:
                line += f'：{abstract}'
            texts.append(line)

    return texts[:10]


def _extract_urls(texts: list[str]) -> list[str]:
    """从搜索结果文本中提取 URL 链接"""
    import re
    urls = []
    for line in texts:
        # 匹配 ——URL 后缀（Bing 结果格式: 【标题】摘要 ——https://...）
        m = re.search(r'——(https?://[^\s）)\]]+)', line)
        if m:
            url = m.group(1).rstrip(')').rstrip('）')
            if url not in urls:
                urls.append(url)
    return urls


def _auto_fetch_first_url(urls: list[str], timeout: int = 12) -> str:
    """尝试抓取最相关的结果页面内容（优先 gov/彩票类站点）"""
    import httpx
    blocked_hosts = [
        "127.0.0.1", "localhost", "0.0.0.0", "::1",
        "10.", "172.16.", "172.17.", "192.168.",
    ]
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0"
        ),
        "Accept-Language": "zh-CN,zh;q=0.9",
    }

    # 优先关键词（这类URL先抓）
    priority_keywords = [
        "cwl.gov.cn", "zhcw.com", "cjcp.cn", "lottery",
        "开奖", "ssq", "双色球", "福彩",
    ]
    # 跳过关键词（字典页/百科不抓）
    skip_keywords = [
        "baike.baidu", "hanyu", "zidian", "gushici",
        "shufazidian", "chagushici", "hgcha", "ufanv",
        "shidianguji", "newdu", "ced.",
    ]

    # 先按优先级排序
    def priority(url: str) -> int:
        url_lower = url.lower()
        for i, kw in enumerate(priority_keywords):
            if kw in url_lower:
                return i  # 越小越优先
        return 999

    sorted_urls = sorted(urls, key=priority)

    for url in sorted_urls[:3]:  # 最多试3个
        url_lower = url.lower()
        # 跳过黑名单
        blocked = any(host.startswith(url_lower.split('/')[2].split(':')[0])
                      if '//' in url else False
                      for host in blocked_hosts)
        if blocked:
            continue
        # 跳过字典页
        if any(sk in url_lower for sk in skip_keywords):
            continue
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                resp = client.get(url, headers=headers)
            if resp.status_code == 200:
                import re
                text = resp.text
                text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
                text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
                text = re.sub(r'<[^>]+>', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                if len(text) > 8000:
                    text = text[:8000] + "\n...(已截断)"
                if len(text) > 200:
                    return text
        except Exception:
            continue
    return ""


_LOTTERY_API_URL = (
    "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/"
    "findDrawNotice?name=ssq&issueCount=20"
)
_SAVE_DIR = Path("/home/ecs-user/search_results")

def _get_save_dir() -> Path:
    """获取保存目录（ECS 用 /home/ecs-user, 本地用 /tmp）"""
    d = _SAVE_DIR
    try:
        d.mkdir(parents=True, exist_ok=True)
        return d
    except (OSError, PermissionError):
        # 本地 Mac 没有 /home/ecs-user，用 /tmp
        tmp = Path("/tmp/cognimem_search")
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp


def _save_result(query: str, content: str, source: str) -> dict:
    """保存搜索结果到服务器文件，返回文件信息"""
    try:
        save_dir = _get_save_dir()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in '_' else '_' for c in query[:20])
        fname = f"{ts}_{safe_name}.txt"
        fpath = save_dir / fname
        fpath.write_text(content, encoding="utf-8")
        logger.info("💾 搜索结果已保存: %s (%d bytes)", fpath, len(content))
        return {
            "file_path": str(fpath),
            "file_name": fname,
            "file_size": len(content),
            "scp_command": f"scp root@47.99.151.253:{fpath} ~/Desktop/",
        }
    except Exception as e:
        logger.warning("保存搜索结果失败: %s", e)
        return {}


def _try_lottery_api(query: str) -> dict | None:
    """尝试从官方 API 直接获取彩票数据

    支持：双色球（ssq）— 中国福彩官网 JSON API
    返回：标准格式的搜索结果字典，或 None
    """
    if "双色球" not in query and "ssq" not in query.lower() and "福彩" not in query:
        return None

    try:
        import httpx
        headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "zh-CN"}
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            resp = client.get(_LOTTERY_API_URL, headers=headers)

        if resp.status_code != 200:
            return None

        data = resp.json()
        if data.get("state") != 0:
            return None

        results = data.get("result", [])
        if not results:
            return None

        # 格式化开奖数据
        lines = []
        for draw in results[:20]:
            code = draw.get("code", "")
            date = draw.get("date", "")
            red = draw.get("red", "")
            blue = draw.get("blue", "")
            lines.append(f"第{code}期（{date}）")
            lines.append(f"  红球: {red}")
            lines.append(f"  蓝球: {blue}")
            lines.append("")

        result_text = "\n".join(lines)
        logger.info("🎯 彩票 API 直连成功: %d 期数据", len(results))

        # 自动保存到服务器文件
        save_info = _save_result(query, result_text, "lottery_api")

        return {
            "result": result_text,
            "source": "lottery_api",
            "query": query,
            "_data_source": "cwl.gov.cn 中国福彩官网",
            "_saved": save_info,
        }
    except Exception as e:
        logger.info("彩票 API 不可用: %s", e)
        return None


def tool_web_search(tool_call_id: str, args: dict,
                    ctx: "AgentContext") -> dict:
    """Search the web.

    搜索策略（按优先级）：
      0. 官方 API 直连（双色球等彩票数据从 cwl.gov.cn JSON API 直接获取）
      1. Bing 直搜（httpx 直连，无需代理，国内可用）+ 自动抓取首个结果页面内容
      2. Qwen DashScope 内置搜索 (enable_search，仅当真正用 Qwen 时)
      3. Bing 代理搜索（SEARCH_PROXY 兜底）
    """
    query = args["query"]
    import urllib.parse
    _is_deepseek = "deepseek" in os.environ.get("QWEN_BASE_URL", "").lower()

    # ===================================================================
    #  方式 0：官方 API 直连（特定数据源）
    #  双色球 → cwl.gov.cn JSON API（中国福彩官网）
    # ===================================================================
    _lottery_api_result = _try_lottery_api(query)
    if _lottery_api_result:
        return _lottery_api_result

    # ⭐ Bing 中文搜索优化：双色球搜索被拆成单字「双」
    _enhanced_query = query
    if "双色球" in query and "开奖" not in query and "号码" not in query:
        _enhanced_query = query + " 开奖结果"
    # ⭐ 排除低质量站点（Bing 支持 `-site:` 排除）
    if "ai-bot.cn" not in _enhanced_query:
        _enhanced_query += " -site:ai-bot.cn"
    encoded = urllib.parse.quote(_enhanced_query)

    # ===================================================================
    #  方式 0.5：新闻专用提取（查询含「新闻」类关键词时）
    #  直接从国内可信来源获取 AI 新闻，绕过 Bing 低质量结果
    # ===================================================================
    _is_news_query = any(kw in query for kw in ["新闻", "news", "News", "最新", "热点", "动态"])
    if _is_news_query:
        try:
            import httpx
            _news_headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Accept-Language": "zh-CN,zh;q=0.9",
            }
            with httpx.Client(timeout=10, follow_redirects=True) as client:
                # 从多个可靠新闻源提取 AI 相关内容
                _news_sources = [
                    "https://www.jiqizhixin.com/",
                    "https://www.36kr.com/information/AI/",
                    "https://www.qbitai.com/",
                ]
                _news_content = []
                for _url in _news_sources:
                    try:
                        _resp = client.get(_url, headers=_news_headers, timeout=5)
                        if _resp.status_code == 200:
                            import re as _re
                            # 取标题（h2/h3 标签）
                            _titles = _re.findall(r'<h[23][^>]*>([^<]{10,80})</h[23]>', _resp.text)
                            if _titles:
                                _news_content.append(f"=== {_url} ===\n" + "\n".join(_titles[:15]))
                    except Exception:
                        continue
                if _news_content:
                    _combined = "\n\n".join(_news_content)
                    logger.info("📰 新闻专用提取: %d 来源, %d chars", len(_news_content), len(_combined))
                    # 内容太少时不提前返回，继续走 Baidu/Bing 搜索
                    if len(_combined) > 500:
                        # 也拼上 Bing 结果作为补充
                        try:
                            _resp2 = client.get(
                                f"https://www.bing.com/search?q={encoded}&setlang=zh-CN",
                                headers=_news_headers, timeout=8,
                            )
                            if _resp2.status_code == 200:
                                _bing_texts = _parse_bing_results(_resp2.text)
                                if _bing_texts:
                                    _combined += "\n\n=== Bing 补充 ===\n" + "\n".join(_bing_texts[:5])
                        except Exception:
                            pass
                        return {
                            "result": _combined,
                            "source": "news_aggregator",
                            "query": query,
                            "_saved": _save_result(query, _combined, "news_agg"),
                        }
        except ImportError:
            pass
        except Exception as e:
            logger.info("新闻源提取失败: %s", e)

    # ===================================================================
    #  方式 1：百度搜索（中国 ECS 首选，结果质量远好于 Bing.cn）
    # ===================================================================
    _baidu_ok = False
    try:
        import httpx
        _q_baidu = urllib.parse.quote(_enhanced_query)
        baidu_headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
        with httpx.Client(timeout=10, follow_redirects=True) as client:
            bd_resp = client.get(f"https://www.baidu.com/s?wd={_q_baidu}&ie=utf-8&rn=10", headers=baidu_headers)
        if bd_resp.status_code == 200:
            bd_texts = _parse_baidu_results(bd_resp.text)
            if bd_texts:
                _baidu_ok = True
                result_text = "\n".join(bd_texts)
                urls = _extract_urls(bd_texts)
                fetched = ""
                if urls:
                    fetched = _auto_fetch_first_url(urls)
                    if fetched:
                        logger.info("📄 百度自动抓取: %s (%d chars)", urls[0][:60], len(fetched))
                logger.info("🔍 百度搜索成功: %d 条结果", len(bd_texts))
                return {
                    "result": result_text,
                    "source": "baidu",
                    "query": query,
                    "fetched_page": fetched or None,
                    "fetched_url": urls[0] if fetched else None,
                    "_saved": _save_result(query, result_text + "\n\n" + (fetched or ""), "baidu"),
                }
    except Exception as e:
        logger.info("百度搜索失败: %s", e)

    # ===================================================================
    #  方式 2：Bing 直搜（百度失败时兜底）
    # ===================================================================
    try:
        import httpx
        bing_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.5",
        }
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            resp = client.get(
                f"https://www.bing.com/search?q={encoded}&setlang=zh-CN",
                headers=bing_headers,
            )
        if resp.status_code == 200:
            texts = _parse_bing_results(resp.text)
            if texts:
                result_text = "\n".join(texts)

                # ⭐ 自动抓取：从结果中提取 URL，抓取第一个可用页面
                urls = _extract_urls(texts)
                fetched = ""
                if urls:
                    fetched = _auto_fetch_first_url(urls)
                    if fetched:
                        logger.info(
                            "📄 自动抓取首个结果页: %s (%d chars)",
                            urls[0][:60], len(fetched),
                        )

                return {
                    "result": result_text,
                    "source": "bing",
                    "query": query,
                    "fetched_page": fetched or None,
                    "fetched_url": urls[0] if fetched else None,
                    "_saved": _save_result(query, result_text + "\n\n" + (fetched or ""), "bing"),
                }
    except Exception as e:
        logger.info("Bing 直搜失败: %s", e)

    # ===================================================================
    #  方式 2：Qwen DashScope 内置搜索
    # ===================================================================
    if not _is_deepseek:
        try:
            from openai import OpenAI
            api_key = os.getenv("QWEN_API_KEY", "")
            base_url = os.getenv("QWEN_BASE_URL", "")
            model = os.getenv("QWEN_MODEL", "qwen3.7-plus")
            if api_key and base_url and "dashscope" in base_url.lower():
                client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content":
                         f"请搜索：{query}，用中文总结搜索结果，列出具体信息"}
                    ],
                    temperature=0.3,
                    extra_body={"enable_search": True},
                )
                reply = resp.choices[0].message.content or ""
                if reply and len(reply) > 20:
                    return {"result": reply, "source": "qwen_search", "query": query}
        except Exception as e:
            logger.info("Qwen 内置搜索不可用: %s", e)

    # ===================================================================
    #  方式 3：Bing 代理搜索（SEARCH_PROXY 兜底）
    # ===================================================================
    proxy = os.environ.get("SEARCH_PROXY", "")
    if proxy:
        try:
            import httpx
            p = proxy
            if p.startswith("socks"):
                p = p.replace("socks5://", "http://").replace("socks5h://", "http://")
            with httpx.Client(proxy=p, timeout=15, follow_redirects=True) as client:
                resp = client.get(
                    f"https://www.bing.com/search?q={encoded}&setlang=zh-CN",
                    headers=bing_headers,
                )
            if resp.status_code == 200:
                texts = _parse_bing_results(resp.text)
                if texts:
                    return {"result": "\n".join(texts), "source": "bing_proxy", "query": query}
        except Exception as e:
            logger.info("Bing 代理搜索失败: %s", e)

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
        description="搜索網絡信息。返回搜索結果頁的文本。"
                    "適合查找實時信息、新聞、文檔等。"
                    "當用戶要求搜索時立即使用此工具，不要確認、不要問用戶要搜什麼。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索關鍵詞——直接用用戶說的話，不要改寫"},
            },
            "required": ["query"],
        },
        executor=tool_web_search, category="web",
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
