"""
MemoryAgent — FastAPI Application

Provides:
  POST /remember         — Store a memory
  POST /recall           — Retrieve relevant memories
  POST /chat             — Chat with memory-augmented Qwen
  POST /groom            — Run memory maintenance
  GET  /status           — Agent status & memory count
"""

from __future__ import annotations

import logging
import os
import time
import asyncio
import functools
from contextlib import asynccontextmanager
from pathlib import Path

# 自动加载 .env 文件（避免手动 export）
_env_path = Path(__file__).resolve().parents[2] / ".env"  # project root
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                k, v = _line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import json
import re

from memory_agent.models import MemoryRecord
from memory_agent.services.llm_client import LLMClient
from memory_agent.services.memory_service import MemoryService
from memory_agent.storage import SQLiteStore
from memory_agent.agent import Agent, SelfReflector, ToolRegistry, _BASE_SYSTEM_PROMPT
from memory_agent.agent.tools import register_all_tools
from cognimem.core.brain import CogniMem
from cognimem.core.db import DatabaseAdapter

# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("memory_agent")

# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

from datetime import datetime, timezone


def _ts_to_epoch(ts_val) -> int:
    """将 ISO 时间字符串或 datetime 转为 Unix 时间戳（秒）

    FactTriple.to_dict() 返回 '2026-07-06T12:30:00+00:00' 格式，
    而前端用 new Date(m.created_at * 1000) 解析，需要秒级时间戳。

    兼容 Python 3.10：datetime.fromisoformat 不支持带时区的 ISO 字符串，
    需要手动剥离时区后缀。
    """
    if not ts_val:
        return 0
    if isinstance(ts_val, (int, float)):
        return int(ts_val)
    if isinstance(ts_val, datetime):
        return int(ts_val.timestamp())
    if isinstance(ts_val, str):
        s = ts_val.strip()
        # Q: 解析时区偏移量（避免 Python 3.10 fromisoformat 报错）
        offset_hours = 0
        try:
            if s.endswith("Z"):
                s = s[:-1]
                offset_hours = 0
            elif "+" in s:
                # 找末尾的 +HH:MM
                head, _, tz = s.rpartition("+")
                if tz and ":" in tz and len(tz.strip()) in (5, 6):
                    s = head
                    parts = tz.split(":")
                    offset_hours = int(parts[0]) + int(parts[1]) / 60
            elif "-" in s[10:]:
                # 找末尾的 -HH:MM
                head, _, tz = s.rpartition("-")
                if tz and ":" in tz and len(tz.strip()) in (5, 6):
                    s = head
                    parts = tz.split(":")
                    offset_hours = -(int(parts[0]) + int(parts[1]) / 60)
            dt = datetime.fromisoformat(s)
            # 把 naive datetime 当作 UTC，再减去偏移量得到实际 UTC 时间戳
            utc_dt = dt.replace(tzinfo=timezone.utc)
            return int(utc_dt.timestamp()) - int(offset_hours * 3600)
        except (ValueError, TypeError):
            return 0
    return 0


def _strip_thinking_text(text: str) -> str:
    """去掉 LLM 回复开头的内心独白/思考过程，只保留实际回复。

    不再用关键词列表（打地鼠），改用**结构检测**：
    - 自我指涉规则分析型：我/我们 + 应该/需要/必须 + 做某事
    - 假设调用型：假设/假如 + 调用了/运行了/执行了 + 工具
    - 引述规则型：引用用户说/系统提示/铁律/规则
    - 日期规则分析型：关于日期/时间的内部对话

    核心思路：句子内容是关于"该怎么做"而不是"做了什么" → 跳过
    """
    text = text.strip()
    if not text:
        return text

    # 按句末标点分割
    segments = re.split(r'(?<=[。！？!?\n])', text)
    segments = [s.strip() for s in segments if s.strip()]
    if len(segments) < 2:
        return text

    # ── 思考文本匹配规则（按优先级）──
    # 每条规则是一个 (regex_pattern, description) 元组
    _think_patterns = [
        # 1. 自我指涉 + 规则/义务分析：我/我们 + 应该/需要/必须/可以/要 + ...做某事...
        (r'(我|我们)\s*(应该|需要|必须|可以|要|得)\s*(先|调|执行|运行|查|搜索|确认|读|写|回答|回复|直接|按照|列|使用|假装)',
         "self-referential obligation"),
        # 2. 假设调用型：假设/假如 + 调用了/执行了/运行了
        (r'(假设|假如|如果)\s*(我|我们)?\s*(调用了|运行了|执行了|使用了|尝试)',
         "hypothetical invocation"),
        # 3. 工具调用规则分析：必须调 + 工具名/shell/date
        (r'(必须调|需要调|应该调|要调)\s*(shell|date|工具|web_search|write_file)',
         "must-call-tool analysis"),
        # 4. 引述规则/系统提示：铁律/规则/系统提示/注意事项
        (r'(铁律|规则要求|系统提示|根据提示|按照规则|按照要求|注意事项)',
         "citing rules"),
        # 5. 用户话语分析：用户说/用户问/用户打招呼 + 分析性质
        (r'用户\s*(说|问|打招呼|当前|已经|正在|想要|要求|在问|的问题|提到的)',
         "analyzing user input"),
        # 6. 会话开始型：我们开始对话/我们被问到/这是新对话
        (r'(我们开始对话|我们被问|这是新对话|这个对话的|当前对话)',
         "session start analysis"),
        # 7. 日期/时间规则分析
        (r'关于\s*(日期|时间|当前)\s*(问题|铁律|规则|要求|必须|需要|应该)',
         "date/time rule analysis"),
        # 8. 已提供/已确认 + 但规则要求（矛盾型思考）
        (r'(已经提供了|已经确认|已用\s*date\s*命令确认).{0,30}(但是|不过|但|然而)',
         "conflict analysis"),
        # 9. 然后类规划：然后(我|我们)应该/需要/要...
        (r'然后\s*(我|我们)?\s*(应该|需要|可以|必须|要|得)',
         "sequential planning"),
        # 10. 所以/因此 + 规则结论
        (r'(所以|因此|那么)\s*(我|我们)?\s*(应该|需要|直接|先|要)',
         "rule conclusion"),
        # 11. 直接回答/直接回复/不需要思考 - 这种本身是思考指令但混合了思考
        (r'(直接回答|直接回复|直接返回|输出不要思考|不需要思考过程|不回忆不推理|不需要加)',
         "directive about thinking"),
        # 12. 用户已XXX + 所以我应该（混合型思考）
        (r'用户已.{0,20}(。|，).{0,30}(所以|因此|那么|不过|但)', "user-did + conclusion"),
        # 13. 作为小明/AI/助手 + 应该如何
        (r'作为\s*(小明|AI|助手|一个)\s*(，|,)\s*(我)?\s*(应该|需要|要)',
         "role-based obligation"),
        # 14. 但系统/规则/铁律（转折型规则思考）
        (r'但\s*(系统|提示|规则|铁律|要求|由于|根据|按照)',
         "but-system rule thinking"),
        # 14. 先(确认|查|看|执行|运行)一下 + 工具类内容
        (r'先\s*(确认|查一查|查一下|看一下|执行一下|运行一下).{0,20}(工具|shell|date|命令|搜索)',
         "pre-check planning"),
        # 15. 由于/因为 + 规则/要求 + 所以（因果型规则思考）
        (r'(由于|因为)\s*(系统|规则|铁律|要求|提示|用户).{0,30}(所以|因此|那么)',
         "causal rule thinking"),
    ]

    def _is_thinking_segment(seg: str) -> bool:
        """判断一个句段是否为思考内容"""
        s = seg[:80]  # 看前80字就够了
        # 如果句子太短（<8字）且不含实际内容 → 不判断为思考
        if len(s) < 6:
            return False
        for pattern, _ in _think_patterns:
            if re.search(pattern, s):
                return True
        return False

    # 从第一段开始检查，跳过所有思考段
    for i, seg in enumerate(segments):
        if not _is_thinking_segment(seg):
            remaining = "".join(segments[i:])
            # 如果剩下的内容太短（<4字）可能只是思考段的尾巴，继续用更后一段
            if len(remaining.strip()) < 4 and i + 1 < len(segments):
                continue
            return remaining

    # 全部都是思考内容 → 取最后一段（至少有点内容）
    return segments[-1]


# ---------------------------------------------------------------------------
#  Globals (initialised in lifespan)
# ---------------------------------------------------------------------------

store: SQLiteStore | None = None
memory_service: MemoryService | None = None
llm: LLMClient | None = None
cogni: CogniMem | None = None  # 直接集成，非 HTTP 客户端
agent: Agent | None = None
tool_registry: ToolRegistry | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global store, memory_service, llm, cogni

    # ── ⭐ v0.10 稳定性: 启动前置检查 ──
    _startup_ok = True
    _startup_checks = []

    # ── CogniMem 引擎（直接集成，无需 8001 端口）──
    try:
        _db_dsn = os.environ.get("COGNIMEM_DB", "")
        _db = DatabaseAdapter(dsn=_db_dsn) if _db_dsn else None
        if _db:
            _db.connect()
            # ⭐ 启动检查：验证数据库连通性
            conn = _db._get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'facts')")
                    has_table = cur.fetchone()[0]
                    if has_table:
                        _startup_checks.append("✅ DB facts 表存在")
                        # 检查 pgvector
                        try:
                            cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
                            has_vector = cur.fetchone()
                            if has_vector:
                                _startup_checks.append("✅ pgvector 扩展已安装")
                            else:
                                _startup_checks.append("⚠️ pgvector 未安装（向量搜索降级为纯文本）")
                                # ⭐ 纯 Python 向量搜索回填
                                try:
                                    n = _db.backfill_embeddings()
                                    if n > 0:
                                        _startup_checks.append(f"✅ 回填 {n} 条 embedding")
                                except Exception:
                                    pass
                        except Exception:
                            _startup_checks.append("⚠️ pgvector 检查失败")
                    else:
                        _startup_checks.append("⚠️ facts 表不存在（首次启动将自动创建）")
                        _db.create_tables()
                        _startup_checks.append("✅ 已自动创建表结构")
            except Exception as e:
                _startup_checks.append(f"❌ DB 连接验证失败: {e}")
                _startup_ok = False
            finally:
                _db._put_conn(conn)
        _use_llm = os.environ.get("COGNIMEM_LLM", "") in ("1", "true", "yes")
        cogni = CogniMem(db_adapter=_db, use_llm=_use_llm)
        logger.info("🧠 CogniMem engine initialized (direct integration)")
    except Exception as e:
        logger.warning("⚠️ CogniMem init failed: %s", e)
        _startup_checks.append(f"⚠️ CogniMem 初始化失败: {e}")
        cogni = None

    db_path = os.getenv("MEMORY_DB_PATH", "~/.qwen-memory/memory.db")
    store = SQLiteStore(db_path)

    # LLM — may be None if no API key configured yet
    api_key = os.getenv("QWEN_API_KEY", "")
    if api_key:
        llm = LLMClient(api_key=api_key)

    # Agent engine — only if LLM + CogniMem both available
    global agent, tool_registry
    if llm and cogni:
        tool_registry = ToolRegistry()
        register_all_tools(tool_registry, cogni)
        reflector = SelfReflector()
        agent = Agent(llm_client=llm, tool_registry=tool_registry,
                      cogni_client=cogni, reflector=reflector)
        logger.info("🤖 Agent engine initialized with %d tools + SelfReflector",
                    len(tool_registry._tools))
    elif llm:
        logger.warning("⚠️ CogniMem not connected — agent engine disabled")

        # Test if embedding actually works (some providers don't support it)
        embed_fn = None
        try:
            test_emb = llm.embed("test")
            if test_emb and len(test_emb) > 0:
                embed_fn = llm.embed
                logger.info("Embedding model available: %s", llm.embedding_model)
        except Exception:
            logger.info("Embedding not available — using FTS5 fallback")

        memory_service = MemoryService(store=store, llm_embed_fn=embed_fn, llm_client=llm)
        logger.info("LLM client initialised with model=%s", llm.model)
    else:
        memory_service = MemoryService(store=store)
        logger.warning("QWEN_API_KEY not set — running without embeddings/LLM")

    # ── ⭐ 启动检查汇总 ──
    for check in _startup_checks:
        logger.info("  %s", check)
    if _startup_ok:
        logger.info("🚀 启动检查: 全部通过")
    else:
        logger.warning("🚀 启动检查: 部分失败（系统以降级模式运行）")

    _HEALTH["start_time"] = time.time()
    yield


app = FastAPI(
    title="Qwen MemoryAgent",
    description="Persistent AI Memory Layer — Global AI Hackathon with QwenCloud",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow cross-origin requests from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════
#  ⭐ v0.10 稳定性: 全局异常处理器
#  确保任何未捕获的异常都返回 HTTP 500 JSON，不给客户端裸 traceback。
# ═══════════════════════════════════════════════════════════════════════════

from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局兜底异常处理器 — 任何未捕获的异常都返回结构化错误"""
    logger.exception("❌ 未处理异常: %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": str(exc)[:200] if str(exc) else "Internal server error",
            "path": request.url.path,
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """HTTP 异常（如 404、405）保持原始状态码"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "detail": exc.detail, "path": request.url.path},
    )


# ---------------------------------------------------------------------------
#  Web UI — static HTML pages
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
TEMPLATES_DIR = HERE / "templates"
STATIC_DIR = HERE / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


import functools
from pathlib import Path as _Path  # noqa: F811 — 避免与 import 冲突


def _read_html(name: str) -> str:
    """读取 HTML 模板，返回内容 + mtime（用于缓存键）"""
    path = TEMPLATES_DIR / name
    if not path.exists():
        raise HTTPException(404, f"Page {name} not found")
    return path.read_text(encoding="utf-8")


# HTML 模板缓存：按 (name, mtime) 缓存，文件修改后自动失效
# ⭐ v0.10 稳定性修复: 之前用 lru_cache 只按 name 缓存，
# 修改模板文件后必须重启才能看到变化。现在按 mtime 缓存，
# 改了文件即时生效，不用重启。
@functools.lru_cache(maxsize=32)
def _read_html_cached(name: str, mtime: float) -> str:
    return _read_html(name)


def read_html(name: str) -> str:
    path = TEMPLATES_DIR / name
    if not path.exists():
        raise HTTPException(404, f"Page {name} not found")
    mtime = path.stat().st_mtime
    return _read_html_cached(name, mtime)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    return read_html("chat.html")


@app.get("/chat", response_class=HTMLResponse, include_in_schema=False)
async def chat_page():
    return read_html("chat.html")


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard_page():
    return read_html("dashboard.html")


# ---------------------------------------------------------------------------
#  Request / Response models
# ---------------------------------------------------------------------------

class RememberRequest(BaseModel):
    agent_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1, max_length=10000)
    memory_type: str = "observation"
    confidence: float = 0.8
    tags: list[str] = Field(default_factory=list)


class RememberResponse(BaseModel):
    memory_id: str
    agent_id: str
    memory_type: str
    confidence: float
    status: str = "stored"


class RecallRequest(BaseModel):
    agent_id: str = Field(..., min_length=1)
    query: str | None = None
    memory_types: list[str] | None = None
    limit: int = 10
    min_confidence: float = 0.0


class ChatRequest(BaseModel):
    agent_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    messages: list[dict] | None = Field(None, description="conversation history for context")
    memory_types: list[str] | None = None


class ProcessTranscriptRequest(BaseModel):
    agent_id: str = Field(..., min_length=1)
    session_id: str | None = None
    transcript: str = Field(..., min_length=10)
    instruction: str = "Extract all key information from this transcript."
    auto_store: bool = True


class ChatLongRequest(BaseModel):
    agent_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    long_context: str | None = Field(None, description="Optional long document to include as context")
    memory_types: list[str] | None = None


class GroomResponse(BaseModel):
    status: str
    stats: dict


# ---------------------------------------------------------------------------
#  Routes
# ---------------------------------------------------------------------------

@app.post("/remember", response_model=RememberResponse)
async def remember(req: RememberRequest):
    """Store a new memory for an agent."""
    # CogniMem mode (direct integration)
    if cogni is not None:
        try:
            result = cogni.remember(
                text=req.content,
                agent_id=req.agent_id,
                source=f"api_remember:{req.session_id}",
            )
            facts_added = result.get("facts_added", 0)
            return RememberResponse(
                memory_id=str(facts_added),
                agent_id=req.agent_id,
                memory_type=req.memory_type,
                confidence=req.confidence,
                status="stored" if facts_added > 0 else "error",
            )
        except Exception as e:
            logger.error("CogniMem remember failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    # Fallback: SQLiteStore
    assert memory_service is not None
    mem = memory_service.remember(
        agent_id=req.agent_id,
        session_id=req.session_id,
        content=req.content,
        memory_type=req.memory_type,
        confidence=req.confidence,
        tags=req.tags,
    )
    return RememberResponse(
        memory_id=mem.id,
        agent_id=mem.agent_id,
        memory_type=mem.memory_type,
        confidence=mem.confidence,
    )


@app.post("/recall")
async def recall(req: RecallRequest):
    """Retrieve relevant memories — backed by CogniMem if available."""
    # Use CogniMem if available (direct integration)
    if cogni is not None:
        agent_id = _resolve_agent(req.agent_id)
        result = cogni.recall(
            query=req.query or "",
            agent_id=agent_id,
            top_k=req.limit,
        )
        facts = [f.to_dict() for f in result.get("facts", [])]
        return {
            "agent_id": req.agent_id,
            "memories": [
                {
                    "id": f.get("fact_id", ""),
                    "content": f"{f.get('subject', '')} {f.get('predicate', '')} {f.get('object', '')}",
                    "fact_type": f.get("fact_type", "observation"),
                    "memory_type": f.get("fact_type", "observation"),
                    "confidence": round(f.get("confidence", 0.5), 3),
                    "tags": f.get("context_tags", []),
                    "created_at": _ts_to_epoch(f.get("created_at")),
                    "encoding_level": f.get("encoding_level", "raw"),
                }
                for f in facts
            ],
            "count": len(facts),
            "total_found": len(facts),
        }

    # Fallback: local SQLiteStore
    assert memory_service is not None
    query_embedding = None
    if req.query and llm:
        try:
            query_embedding = llm.embed(req.query)
        except Exception as e:
            logger.warning("Embedding failed: %s", e)

    result = memory_service.recall(
        agent_id=req.agent_id,
        query_text=req.query,
        query_embedding=query_embedding,
        memory_types=req.memory_types,
        limit=req.limit,
        min_confidence=req.min_confidence,
    )
    return {
        "agent_id": req.agent_id,
        "memories": [
            {
                "id": m.id,
                "content": m.content,
                "memory_type": m.memory_type,
                "confidence": round(m.confidence, 3),
                "tags": m.tags,
                "created_at": m.created_at,
                "session_id": m.session_id,
            }
            for m in result.memories
        ],
        "count": len(result.memories),
        "total_found": result.total_found,
        "query_time_ms": result.query_time_ms,
    }


@app.post("/ask")
async def api_ask(query: str = "", agent_id: str = "default"):
    """问答式召回（Agent 友好）

    返回相关记忆、核心信念、不确定项、矛盾提醒、主动学习问题。
    与 /recall 的区别：返回结构化信息（含信念/矛盾/不确定）+ 主动学习引导。
    """
    if cogni is None:
        return {
            "agent_id": agent_id, "query": query,
            "relevant_memories": [], "core_beliefs": [],
            "uncertainties": [], "active_questions": [],
            "contradictions_warning": False,
        }
    try:
        agent_id = _resolve_agent(agent_id)
        result = cogni.ask(query=query, agent_id=agent_id)
        # 序列化 FactTriple 对象
        if "relevant_memories" in result:
            for m in result["relevant_memories"]:
                for k in ("citation", "stale_warning", "source_label"):
                    v = m.get(k)
                    if hasattr(v, 'to_dict'):
                        m[k] = v.to_dict()
        return result
    except Exception as e:
        logger.warning("ask failed: %s", e)
        return {
            "agent_id": agent_id, "query": query,
            "relevant_memories": [], "core_beliefs": [],
            "uncertainties": [], "active_questions": [],
            "contradictions_warning": False,
            "error": str(e)[:200],
        }


# ══════════════════════════════════════════════════════════
#  上下文引擎 — 不用压缩，CogniMem 就是压缩器
# ══════════════════════════════════════════════════════════
#  4 更方案：
#  - 不额外调用 LLM 做摘要（更省资源）
#  - 不存冗余的蒸馏液（更省 token）
#  - CogniMem 图谱本身承担了跨轮记忆（更智能）
#  - 只需最近 1 轮原文保连贯 + 图谱召回保记忆（更创新）
#
#  每轮成本：~250t(最近1轮) + ~300t(图谱5条) = ~550t 恒定


def _build_context(
    user_message: str,
    agent_id: str,
    session_id: str,
    conversation_history: list[dict] | None,
) -> tuple[str, list[dict]]:
    """构建上下文 = 最近 1 轮原文 + CogniMem 图谱召回。

    L1 — 回闪：最近 1 轮 user+assistant，保持对话连贯
    L3 — 图谱：CogniMem 按语义召回，跨会话+本会话都在里面
    """
    # ── L3: CogniMem 图谱召回（跨会话持久 + 本会话事实）──
    recalled = []
    if cogni:
        try:
            result = cogni.recall(query=user_message, agent_id=agent_id, top_k=8, session_id=session_id)
            recalled = [f.to_dict() for f in result.get("facts", [])]
        except Exception as e:
            logger.warning("Memory recall failed: %s", e)

    # ── L1: 最近多轮原文（保连贯）──
    recent = []
    if conversation_history:
        # 最多保留最近 10 条消息（5轮完整对话）
        # ⚠️ 前端存 role='agent'，但 LLM API 只认 role='assistant'，必须映射
        recent = [
            {"role": ("assistant" if m["role"] == "agent" else m["role"]), "content": m["content"][:500]}
            for m in conversation_history[-10:]
            if m.get("content")
        ]

    # ── 系统提示词 + 图谱注入 ──
    system = _BASE_SYSTEM_PROMPT

    # ⭐ 日期问题：直接执行 date 注入结果（不走 Agent 循环也能答对）
    DATE_KW = ["今天", "几号", "星期", "多少号", "这个月", "几月",
               "多少天", "当前时间", "现在时间", "年月日", "什么日期"]
    if any(kw in user_message for kw in DATE_KW):
        import subprocess as _sp
        try:
            _r = _sp.run(["date", "+%Y年%m月%d日 星期%w %H:%M"], capture_output=True, text=True, timeout=5)
            _ds = _r.stdout.strip()
            for k, v in {"0":"日","1":"一","2":"二","3":"三","4":"四","5":"五","6":"六"}.items():
                _ds = _ds.replace(f"星期{k}", f"星期{v}")
            system = f"## ⏰ 当前时间（已用 date 命令确认）\n{_ds}\n直接回答日期问题，不要输出任何思考过程或「我们被问到」之类的废话。\n\n" + system
            logger.info("📅 [build_context] 预注入 date=%s", _ds[:30])
        except Exception:
            pass

    if recalled:
        # ⭐ 记忆治理评分 + 类型多样化排序（对标 _score_memory）
        _type_priority = {"preference": 3, "goal": 2, "fact": 2, "decision": 2, "observation": 1, "action": 0}
        def _gov_score(f):
            base = f.get("confidence", 0.5) * f.get("importance", 0.5)
            tp = _type_priority.get(f.get("fact_type", "observation"), 1)
            return base * tp

        recalled = [f for f in recalled if f.get("confidence", 0.5) >= 0.2]  # 过滤极低置信度
        recalled.sort(key=_gov_score, reverse=True)

        # 类型多样化：同一类型最多2条
        lines = []
        seen_types = {}
        for f in recalled:
            ft = f.get("fact_type", "observation")
            seen_types.setdefault(ft, 0)
            if seen_types[ft] >= 2:
                continue
            s = f.get("subject", "")
            p = f.get("predicate", "")
            o = f.get("object", "")
            if s in ("user", "用户", "你"):
                s = "你"
            if any(kw in (p + o) for kw in ["小七", "小智", "小可爱"]):
                continue
            conf = f.get("confidence", 0.5)
            if conf < 0.3:
                continue
            lines.append(f"- {s}{p}{o}")
            seen_types[ft] = seen_types.get(ft, 0) + 1
            if len(lines) >= 4:
                break
        if lines:
            system += "\U0001f9e0 我记得\n" + "\n".join(lines) + "\n"

    # ⭐ v0.15: 跨会话记忆桥
    try:
        summary_file = Path("~/.qwen-memory/session_summaries.json").expanduser()
        if summary_file.exists():
            all_summ = json.loads(summary_file.read_text())
            agent_summ = [s for s in all_summ if s.get("agent_id") == agent_id]
            if agent_summ:
                last = agent_summ[-1]
                from datetime import datetime, timezone
                try:
                    last_ts = datetime.fromisoformat(last.get("timestamp", ""))
                    hours = (datetime.now(timezone.utc) - last_ts).total_seconds() / 3600
                except Exception:
                    hours = 999
                # 打招呼不自动回忆
                _is_greeting = user_message.strip() in ("你好", "hi", "hello", "hey", "在吗", "在不在", "哈喽")
                if hours < 72 and not _is_greeting:
                    topics = last.get("topics", [])
                    if topics:
                        system += f"\U0001f4ac 上次聊过 {'、'.join(topics[:3])}\n"
    except Exception:
        pass

    # ⭐ 简单路径最后强指令：禁止思考过程
    system += (
        "\n\n## ⛔ 只输出最终回复\n"
        "你的回复会被直接展示给用户，不能包含任何思考、分析、内心独白。\n"
        "✅ 用户问日期 → 你回「2026年7月19日」\n"
        "❌ 用户问日期 → 你不能写「根据系统时间，当前是2026年7月…」"
    )

    msgs = [{"role": "system", "content": system}]
    # 如果 recent 最后一条就是当前 user 消息，跳过重复
    if recent and recent[-1]["role"] == "user" and recent[-1]["content"] == user_message[:500]:
        msgs.extend(recent)
    else:
        msgs.extend(recent)
        msgs.append({"role": "user", "content": user_message})

    return system, msgs


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming chat with memory-augmented Agent.

    上下文 = L1回闪(最近2轮) + L2蒸馏液(会话压缩) + L3图谱(CogniMem recall)
    不用前端传来的全部历史，避免上下文膨胀。
    """
    if not agent or not llm:
        raise HTTPException(status_code=503, detail="Agent not available")

    async def event_stream():
        # ── 构建上下文（CogniMem 图谱 + 最近 1 轮原文）──
        system, llm_messages = _build_context(
            user_message=req.message,
            agent_id=req.agent_id,
            session_id=req.session_id,
            conversation_history=req.messages,
        )

        # ── 判断简单问答还是复杂任务 ──
        msg = req.message.strip()
        ACTION_WORDS = [
            "搜索", "搜一下", "查找", "查一下", "查一查", "先查", "爬", "下载", "读取",
            "写入", "编辑", "分析", "对比", "创建", "写", "生成",
            "改", "删", "跑", "试", "调用", "执行", "运行",
            "总结", "翻译", "推荐", "画", "整理", "记住", "计算",
            "search", "find", "fetch", "read", "write",
            "edit", "create", "analyze", "install",
        ]
        # ⭐ 继续/下一步 → 必须进 Agent 路径（否则不能调工具继续写文件等操作）
        IS_CONTINUATION = (
            msg in ("继续", "继续！", "继续执行") or msg.startswith("继续")
            or msg in ("next", "continue", "go on", "下一步", "然后呢")
        )
        has_action = any(v in msg.lower() for v in ACTION_WORDS)
        # 简单问答 = 没有动作关键词 + 非继续 + 非长文本
        is_simple = (len(msg) < 60) and not has_action and not IS_CONTINUATION

        try:
            if is_simple:
                # ⭐ 纯打招呼 → 硬编码回复，不走 LLM（避免思考过程泄漏）
                if msg.strip() in ("你好", "hi", "hello", "hey", "在吗", "哈喽", "您好"):
                    yield f"data: {json.dumps({'type': 'token', 'content': '你好！我是小明，有什么事吗？'})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"
                    _record_api_call(success=True)
                    return

                # ⭐ 简单路径：先收集完整回复再过滤思考过程
                # ⭐ 追加最终强制指令（离LLM最近，效果最好）
                llm_messages.append({
                    "role": "user",
                    "content": (
                        "直接输出给用户的回复，不要输出任何思考/分析/内心独白/规则引述。"
                        "不要以「我们被问」「用户说」「根据规则」开头。直接说内容。"
                    ),
                })
                full_text = ""
                stream = llm.chat_stream(
                    messages=llm_messages,
                    system_prompt=None,
                    temperature=0.5,
                    max_tokens=2048,
                )
                for token in stream:
                    full_text += token

                # ⭐ 去除 LLM 输出中的内心独白/思考过程
                cleaned = _strip_thinking_text(full_text)

                # ⭐ 空响应保护：LLM 返回空时自动降级到 Agent 路径
                if not cleaned.strip():
                    logger.warning("🛑 LLM returned empty for simple query — falling back to agent")
                    _record_api_call(success=False, error_msg="empty_response")
                    # ⭐ 子线程执行，不阻塞事件循环
                    _loop = asyncio.get_running_loop()
                    _agent_fn = functools.partial(
                        agent.chat,
                        message=req.message,
                        agent_id=req.agent_id,
                        session_id=req.session_id,
                        temperature=0.5,
                        messages=req.messages,
                    )
                    _agent_future = _loop.run_in_executor(None, _agent_fn)
                    while not _agent_future.done():
                        _done, _ = await asyncio.wait([_agent_future], timeout=15.0)
                        if not _agent_future.done():
                            yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
                    result = _agent_future.result()
                    reply = result.get("reply", "")
                    if reply:
                        yield f"data: {json.dumps({'type': 'token', 'content': reply})}\n\n"
                    else:
                        reply = f"抱歉没处理好。你说「{req.message[:30]}」，我再试一次。"
                        yield f"data: {json.dumps({'type': 'token', 'content': reply})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"
                    return

                # ⭐ 整个输出清理后的回复（快速响应比逐句更可靠）
                yield f"data: {json.dumps({'type': 'token', 'content': cleaned})}\n\n"
                _record_api_call(success=True)
                yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"
            else:
                logger.info("Complex task detected — using agent loop")
                # ⭐ 在子线程运行 agent.chat()，不阻塞事件循环
                # SSE 连接 15s 无数据会超时 → 每 15s 发心跳保活
                _loop = asyncio.get_running_loop()
                _agent_fn = functools.partial(
                    agent.chat,
                    message=req.message,
                    agent_id=req.agent_id,
                    session_id=req.session_id,
                    temperature=0.5,
                    messages=req.messages,
                )
                _agent_future = _loop.run_in_executor(None, _agent_fn)
                # 心跳循环：等 agent 完成，每 15s 发 keepalive 防超时
                while not _agent_future.done():
                    _done, _ = await asyncio.wait([_agent_future], timeout=15.0)
                    if not _agent_future.done():
                        yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
                result = _agent_future.result()
                reply = result.get("reply", "")
                memories_stored = result.get("memories_stored", 0)

                # ⭐ Agent 空响应保护
                if not reply.strip():
                    logger.warning("🛑 Agent returned empty reply — using fallback")
                    reply = (
                        f"抱歉，我刚才没正确处理。你说「{req.message[:40]}」，"
                        "能再说一次吗？我一定直接执行，不废话。"
                    )

                yield f"data: {json.dumps({'type': 'token', 'content': reply})}\n\n"
                tools = result.get("tools_called", 0)
                if tools > 0:
                    yield f"data: {json.dumps({'type': 'meta', 'content': f'🛠️ {tools} 次工具调用'})}\n\n"
                if memories_stored > 0:
                    yield f"data: {json.dumps({'type': 'meta', 'content': f'🧠 +{memories_stored} 条记忆'})}\n\n"
                _record_api_call(success=True)
                yield f"data: {json.dumps({'type': 'done', 'content': '', 'tools_called': tools, 'memories_stored': memories_stored})}\n\n"
        except Exception as e:
            _record_api_call(success=False, error_msg=str(e))
            # ⭐ 异常时也尝试降级到 Agent 路径
            logger.warning("🛑 chat_stream error — trying agent fallback: %s", e)
            try:
                # ⭐ 子线程执行，不阻塞事件循环
                _loop = asyncio.get_running_loop()
                _agent_fn = functools.partial(
                    agent.chat,
                    message=req.message,
                    agent_id=req.agent_id,
                    session_id=req.session_id,
                    temperature=0.5,
                    messages=req.messages,
                )
                _agent_future = _loop.run_in_executor(None, _agent_fn)
                while not _agent_future.done():
                    _done, _ = await asyncio.wait([_agent_future], timeout=15.0)
                    if not _agent_future.done():
                        yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
                result = _agent_future.result()
                reply = result.get("reply", "") or "抱歉，我遇到了一个暂时的问题，请再试一次。"
                yield f"data: {json.dumps({'type': 'token', 'content': reply})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"
            except Exception as e2:
                logger.exception("Agent fallback also failed: %s", e2)
                yield f"data: {json.dumps({'type': 'error', 'content': str(e2)[:200]})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/chat")
async def chat(req: ChatRequest):
    """Chat with memory-augmented Qwen Agent.

    用 三层压缩（L1回闪+L2蒸馏液+L3图谱）替代原始历史拼接。
    """
    if agent:
        # Agent 模式：传压缩后的最近消息
        recent = []
        if req.messages:
            recent = [{"role": ("assistant" if m["role"] == "agent" else m["role"]), "content": m["content"][:500]}
                      for m in req.messages[-10:] if m.get("content")]
        result = agent.chat(
            message=req.message,
            agent_id=req.agent_id,
            session_id=req.session_id,
            temperature=0.5,
            messages=recent,
        )
        reply = result.get("reply", "")
        if not reply.strip():
            reply = (
                f"我是小明！你刚刚说「{req.message[:40]}」，"
                "我不太确定需要做什么具体操作。"
                "我可以搜索信息、看网页、读写文件、或者记住事情。"
                "直接告诉我想干什么就行！"
            )
        return {
            "agent_id": req.agent_id,
            "reply": reply,
            "memories_used": result.get("memories_used", 0),
            "tools_called": result.get("tools_called", 0),
            "iterations": result.get("iterations", 0),
            "tool_sequence": result.get("tool_sequence", []),
        }

    # Fallback: simple LLM (no agent)
    assert memory_service is not None
    if not llm:
        raise HTTPException(status_code=503, detail="LLM not configured")

    system, _ = _build_context(
        user_message=req.message,
        agent_id=req.agent_id,
        session_id=req.session_id,
        conversation_history=req.messages,
    )

    answer = llm.chat(
        messages=[{"role": "user", "content": req.message}],
        system_prompt=system,
    )

    conv = req.messages[-6:] if req.messages and len(req.messages) > 2 else [
        {"role": "user", "content": req.message},
        {"role": "assistant", "content": answer},
    ]
    candidates = llm.extract_memories(conv)
    stored = sum(1 for c in candidates if memory_service.remember(
        agent_id=req.agent_id, session_id=req.session_id,
        content=c.get("content", ""), memory_type=c.get("type", "observation"),
        confidence=float(c.get("confidence", 0.6)), tags=c.get("tags", []),
        source="chat_extraction"))

    return {
        "agent_id": req.agent_id, "reply": answer,
        "memories_used": 0, "new_memories_stored": stored,
    }


@app.get("/decay-trace/{memory_id}")
async def decay_trace(memory_id: str, agent_id: str = "default", days: int = 30, points: int = 50):
    """Visualize confidence decay curve for a specific memory."""
    # CogniMem mode (direct integration)
    if cogni is not None:
        import math
        # 在指定 agent 中查找事实
        facts = cogni.fact_network._get_agent_facts(agent_id)
        target = None
        for f in facts:
            if f.fact_id == memory_id:
                target = f
                break
        if not target:
            raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")

        half_life = 30.0
        lam = half_life / (math.log(2) ** (1.0 / 1.5))
        step = max(1, days // points)
        trace = []
        for d in range(0, days + 1, step):
            cdf = 1.0 - math.exp(-((d / lam) ** 1.5))
            conf = target.confidence * (1.0 - cdf)
            trace.append({"day": d, "confidence": round(max(0.0, conf), 4)})

        return {
            "memory_id": memory_id,
            "initial_confidence": target.confidence,
            "half_life_days": half_life,
            "decay_model": "weibull(k=1.5)",
            "trace": trace,
            "days": days,
        }

    # Fallback: MemoryService
    assert memory_service is not None
    trace = memory_service.compute_decay_trace(memory_id, days=days, points=points)
    if "error" in trace:
        raise HTTPException(status_code=404, detail=trace["error"])
    return trace


@app.get("/decay-analysis")
async def decay_analysis(agent_id: str, min_confidence: float = 0.0):
    """记忆分析 — 基于 CogniMem 数据"""
    # Use CogniMem if available (direct integration)
    if cogni is not None:
        agent_id = _resolve_agent(agent_id)
        stats = cogni.get_stats(agent_id)
        facts = stats.get("total_facts", 0)
        core = stats.get("core_beliefs", 0)
        by_type = stats.get("by_type", {})
        # Fetch actual facts directly
        try:
            all_facts = cogni.fact_network._get_agent_facts(agent_id)
            all_facts.sort(key=lambda f: f.confidence, reverse=True)
            raw_facts = [f.to_dict() for f in all_facts[:50]]
        except Exception:
            raw_facts = []

        memories = []
        for t, c in by_type.items():
            memories.append({
                "id": f"type_{t}",
                "content": f"{t}: {c}条",
                "memory_type": t,
                "confidence": 1.0,
                "needs_refresh": False,
                "created_at": int(time.time()),
                "is_summary": True,
            })
        for f in raw_facts:
            _full = f"{f.get('subject','')} {f.get('predicate','')} {f.get('object','')}"
            memories.append({
                "id": f.get("fact_id", ""),
                "content": _full if len(_full) <= 19 else _full[:16] + "…",
                "memory_type": f.get("fact_type", "observation"),
                "confidence": f.get("confidence", 0.5),
                "tags": f.get("context_tags", []),
                "needs_refresh": False,
                "created_at": _ts_to_epoch(f.get("created_at")),
            })
        return {"agent_id": agent_id, "memories": memories, "total": facts}

    # Fallback: SQLiteStore decay analysis
    assert memory_service is not None
    return {
        "agent_id": agent_id,
        "memories": memory_service.get_memory_decay_analysis(agent_id, min_confidence),
        "total": store.count(agent_id) if store else 0,
    }


@app.post("/chat/long")
async def chat_long(req: ChatLongRequest):
    """Chat with memory-augmented Qwen + optional long document context.

    Uses qwen-max-longcontext when a long_context document is provided.
    """
    assert memory_service is not None

    if not llm:
        raise HTTPException(
            status_code=503,
            detail="LLM not configured — set QWEN_API_KEY",
        )

    # 1. Recall relevant memories
    query_embedding = None
    try:
        query_embedding = llm.embed(req.message)
    except Exception as e:
        logger.warning("Embedding failed (non-critical): %s", e)

    recall_result = memory_service.recall(
        agent_id=req.agent_id,
        query_text=req.message,
        query_embedding=query_embedding,
        memory_types=req.memory_types,
        limit=8,
    )

    # 2. Get active preferences
    prefs = store.get_active_preferences(req.agent_id) if store else {}

    # 3. Build memory context
    memories_dict = [
        {
            "content": m.content,
            "memory_type": m.memory_type,
            "confidence": m.confidence,
            "created_at": m.created_at,
        }
        for m in recall_result.memories
    ]

    context_parts = []
    for m in memories_dict:
        context_parts.append(
            f"[{m.get('memory_type', 'unknown')} | conf:{m.get('confidence', 0):.2f}] "
            f"{m.get('content', '')}"
        )

    system = (
        "You are a helpful AI assistant with persistent memory. "
        "Use the provided memory context to answer accurately."
    )
    if context_parts:
        system += "\n\n## Retrieved Memories\n" + "\n".join(context_parts)
    if prefs.get("preferences"):
        pref_text = "\n".join(f"- {p.get('content', '')}" for p in prefs["preferences"])
        system += f"\n\n## Known Preferences\n{pref_text}"

    # 4. If long_context provided, use qwen-max-longcontext
    use_model = None
    if req.long_context:
        use_model = "qwen-max-longcontext"
        user_content = (
            f"{req.message}\n\n"
            f"## Additional Context Document\n{req.long_context}"
        )
        logger.info("Using long-context model for chat with document")
    else:
        user_content = req.message

    # 5. Generate answer
    answer = llm.chat(
        messages=[{"role": "user", "content": user_content}],
        system_prompt=system,
        temperature=0.5,
        model=use_model,
    )

    # 6. Extract and store new memories
    conversation = [
        {"role": "user", "content": req.message},
        {"role": "assistant", "content": answer},
    ]
    candidates = llm.extract_memories(conversation)
    stored_count = 0
    for c in candidates:
        memory_service.remember(
            agent_id=req.agent_id,
            session_id=req.session_id,
            content=c.get("content", ""),
            memory_type=c.get("type", "observation"),
            confidence=float(c.get("confidence", 0.6)),
            tags=c.get("tags", []),
            source="chat_extraction",
        )
        stored_count += 1

    return {
        "agent_id": req.agent_id,
        "reply": answer,
        "memories_used": len(memories_dict),
        "new_memories_stored": stored_count,
        "long_context_used": req.long_context is not None,
        "model": use_model or llm.model,
    }


@app.post("/process-transcript")
async def process_transcript(req: ProcessTranscriptRequest):
    """Process a long transcript with qwen-max-longcontext (1M tokens).

    Extracts structured memories and optionally stores them.
    """
    assert memory_service is not None

    if not llm:
        raise HTTPException(
            status_code=503,
            detail="LLM not configured — set QWEN_API_KEY",
        )

    # Log estimated size
    char_count = len(req.transcript)
    est_tokens = char_count // 2
    logger.info(
        "Processing transcript: %d chars (~%d tokens, session=%s)",
        char_count, est_tokens, req.session_id or "none",
    )

    # Extract memories using long-context model
    candidates = llm.extract_memories_from_long_transcript(req.transcript)

    # Optionally store extracted memories
    stored_count = 0
    if req.auto_store and candidates:
        session = req.session_id or f"transcript_{int(time.time())}"
        for c in candidates:
            memory_service.remember(
                agent_id=req.agent_id,
                session_id=session,
                content=c.get("content", ""),
                memory_type=c.get("type", "observation"),
                confidence=float(c.get("confidence", 0.6)),
                tags=c.get("tags", []),
                source="transcript_extraction",
            )
            stored_count += 1
        logger.info("Stored %d memories from transcript", stored_count)

    return {
        "agent_id": req.agent_id,
        "session_id": req.session_id,
        "transcript_length": char_count,
        "estimated_tokens": est_tokens,
        "memories_extracted": len(candidates),
        "memories_stored": stored_count,
        "memories": candidates,
    }


@app.post("/groom", response_model=GroomResponse)
async def groom(agent_id: str):
    """Run memory maintenance (decay + prune)."""
    # CogniMem mode (direct integration) — consolidate handles decay+prune
    if cogni is not None:
        result = cogni.consolidate(agent_id)
        return GroomResponse(status="ok", stats=result)

    # Fallback: MemoryService
    assert memory_service is not None
    stats = memory_service.groom(agent_id)
    return GroomResponse(status="ok", stats=stats)


@app.get("/preferences")
async def get_preferences(agent_id: str):
    """Get active preferences for an agent."""
    assert store is not None
    prefs = store.get_active_preferences(agent_id)
    return {"agent_id": agent_id, "preferences": prefs}


@app.get("/preferences/history")
async def preference_history(agent_id: str):
    """Get preference evolution history (including superseded ones)."""
    # CogniMem mode (direct integration)
    if cogni is not None:
        agent_id = _resolve_agent(agent_id)
        facts = cogni.fact_network._get_agent_facts(agent_id)
        prefs = [f.to_dict() for f in facts if f.fact_type == "preference"]
        prefs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return {"agent_id": agent_id, "history": prefs, "total": len(prefs)}

    # Fallback: MemoryService
    assert memory_service is not None
    history = memory_service.get_preference_history(agent_id)
    return {"agent_id": agent_id, "history": history, "total": len(history)}


@app.get("/status")
async def status(agent_id: str):
    """Get agent memory status."""
    assert store is not None
    count = store.count(agent_id)
    prefs = store.get_active_preferences(agent_id)
    return {
        "agent_id": agent_id,
        "memory_count": count,
        "preferences": prefs,
    }


@app.post("/merge")
async def merge_memories(
    agent_id: str,
    sim_threshold: float = 0.65,
    min_cluster_size: int = 2,
    limit: int = 100,
):
    """Find and merge similar memories for an agent.

    Clusters memories by semantic similarity, then merges each cluster
    into a single aggregated memory using LLM summarization.
    """
    # CogniMem mode (direct integration) — consolidate handles merge
    if cogni is not None:
        result = cogni.consolidate(agent_id)
        return {"status": "ok", "merged": result.get("merged", 0),
                "abstracted": result.get("abstracted", 0),
                "message": "已通过 consolidate 完成合并"}

    # Fallback: MemoryService
    assert memory_service is not None
    stats = memory_service.merge_all(
        agent_id=agent_id,
        sim_threshold=sim_threshold,
        min_cluster_size=min_cluster_size,
        limit=limit,
    )
    return stats


@app.get("/graph", response_class=HTMLResponse, include_in_schema=False)
async def graph_page():
    """Memory relationship graph page."""
    return read_html("graph.html")


@app.get("/memory-graph")
async def memory_graph(
    agent_id: str,
    limit: int = 50,
    min_confidence: float = 0.1,
    threshold: float = 0.2,
):
    """知识图谱 — 基于 CogniMem 三元组"""
    agent_id = _resolve_agent(agent_id)
    # Use CogniMem if available (direct integration)
    if cogni is not None:
        try:
            all_facts = cogni.fact_network._get_agent_facts(agent_id)
            all_facts.sort(key=lambda f: f.confidence, reverse=True)
            facts = [f.to_dict() for f in all_facts[:limit]]
        except Exception as e:
            logger.error("memory-graph recall failed: %s", e)
            return {"agent_id": agent_id, "nodes": [], "edges": [],
                    "stats": {"node_count": 0, "edge_count": 0, "density": 0, "abstractions": 0}}


        # Build nodes and edges from triples
        node_map = {}
        edges = []
        node_id_counter = 0

        def get_node(name: str, type_hint: str = "entity") -> str:
            nonlocal node_id_counter
            key = name.lower().strip()
            if not key:
                return ""
            if key not in node_map:
                node_id_counter += 1
                nid = f"n{node_id_counter}"
                node_map[key] = {
                    "id": nid, "content": name, "memory_type": type_hint,
                    "confidence": 1.0, "tags": [], "node_type": "entity",
                }
            return node_map[key]["id"]

        for f in facts:
            subj = f.get("subject", "")
            pred = f.get("predicate", "")
            obj = f.get("object", "")
            ftype = f.get("fact_type", "observation")
            level = f.get("encoding_level", "raw")
            if not subj or not pred or not obj:
                continue

            sid = get_node(subj)
            oid = get_node(obj, ftype)
            if not sid or not oid:
                continue

            edges.append({
                "source": sid, "target": oid,
                "strength": round(f.get("confidence", 0.5), 3),
                "predicate": pred,
                "fact_id": f.get("fact_id", ""),
                "memory_type": ftype,
                "encoding_level": level,
            })

        nodes = list(node_map.values())
        return {
            "agent_id": agent_id,
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "density": round(
                    len(edges) / max(1, len(nodes) * (len(nodes) - 1) / 2), 4
                ),
                "abstractions": sum(1 for f in facts if f.get("encoding_level") == "abstraction"),
            },
        }

    # Fallback: local SQLiteStore
    assert store is not None
    assert memory_service is not None

    memories = store.search(
        agent_id=agent_id,
        limit=limit,
        min_confidence=min_confidence,
    )

    nodes = []
    for m in memories:
        nodes.append({
            "id": m.id,
            "content": m.content,
            "memory_type": m.memory_type,
            "confidence": round(m.confidence, 3),
            "tags": m.tags,
            "created_at": m.created_at,
            "session_id": m.session_id,
        })

    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            sim = memory_service._compute_pairwise_sim(
                nodes[i]["content"], nodes[j]["content"]
            )
            if sim >= threshold:
                edges.append({
                    "source": nodes[i]["id"],
                    "target": nodes[j]["id"],
                    "strength": round(sim, 3),
                })

    return {
        "agent_id": agent_id,
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "density": round(
                len(edges) / max(1, len(nodes) * (len(nodes) - 1) / 2), 4
            ),
        },
    }


# ══════════════════════════════════════════════════════════
#  CogniMem 直接调用（引擎已集成在进程中）
# ══════════════════════════════════════════════════════════

def _resolve_agent(agent_id: str) -> str:
    """agent_id=default 时，仅在 default 真的没有数据时才 fallback 到最后活跃 agent"""
    if agent_id == "default" and cogni and cogni.fact_network and cogni.fact_network.db:
        db = cogni.fact_network.db
        try:
            # 先查 default 自己有没有数据
            with db._plain_cursor_ctx() as cur:
                cur.execute("SELECT COUNT(*) FROM facts WHERE agent_id = 'default'")
                row = cur.fetchone()
                if row and row[0] > 0:
                    return "default"  # default 有数据 → 不跳转
            # default 没数据 → 退而求其次
            with db._plain_cursor_ctx() as cur:
                cur.execute("SELECT agent_id FROM facts GROUP BY agent_id ORDER BY MAX(created_at) DESC LIMIT 1")
                row = cur.fetchone()
                if row:
                    agent_id = row[0]
        except Exception:
            pass
    return agent_id


@app.get("/agents")
async def list_agents():
    """列出所有有数据的 Agent"""
    agents = []
    # 从 CogniMem PostgreSQL 获取所有 agent_id
    if cogni and cogni.fact_network and cogni.fact_network.db:
        try:
            db = cogni.fact_network.db
            with db._cursor_ctx() as cur:
                cur.execute("""
                    SELECT agent_id, COUNT(*) AS fact_count,
                           MAX(created_at) AS last_active
                    FROM facts GROUP BY agent_id
                    ORDER BY last_active DESC NULLS LAST
                """)
                for row in cur.fetchall():
                    agents.append({
                        "id": row[0],
                        "fact_count": row[1],
                    })
        except Exception as e:
            logger.warning("Agent list query failed: %s", e)
    if not agents:
        agents.append({"id": "default", "fact_count": 0})
    return {"agents": agents}


@app.get("/stats")
async def stats(agent_id: str):
    """统计 — CogniMem 引擎数据（v0.13 含STM+路由+知识库）"""
    if cogni is None:
        return {"agent_id": agent_id, "total_facts": 0, "core_beliefs": 0,
                "unreliable": 0, "contradictions": 0, "by_type": {},
                "router_stats": {}, "abstractions": 0, "stm_buffer": 0}
    agent_id = _resolve_agent(agent_id)
    data = cogni.get_stats(agent_id)
    # 补充知识库统计
    try:
        creds = cogni.list_credentials(agent_id)
        data["credential_count"] = len(creds)
    except Exception:
        data["credential_count"] = 0
    return data


# ═══════════════════════════════════════════════
#  记忆管理 API（Dashboard 使用）
# ═══════════════════════════════════════════════

@app.get("/memories")
async def list_memories(agent_id: str = "default", limit: int = 50, offset: int = 0):
    """列出 Agent 的所有记忆（分页），附带矛盾标记"""
    if cogni is None:
        return {"agent_id": agent_id, "memories": [], "total": 0}
    agent_id = _resolve_agent(agent_id)
    try:
        facts = cogni.fact_network._get_agent_facts(agent_id)
        # 矛盾列表（查 ID）
        contradictions = cogni.fact_network.get_contradictions(agent_id)
        contradiction_ids = set()
        for c in contradictions:
            contradiction_ids.add(c.fact_a_id)
            contradiction_ids.add(c.fact_b_id)

        total = len(facts)
        facts.sort(key=lambda f: f.confidence, reverse=True)
        page = facts[offset:offset + limit]

        memories = []
        for f in page:
            d = f.to_dict()
            d["has_contradiction"] = f.fact_id in contradiction_ids
            d["evidence_count"] = len(f.evidence)
            memories.append(d)

        return {
            "agent_id": agent_id,
            "memories": memories,
            "total": total,
            "offset": offset,
            "limit": limit,
        }
    except Exception as e:
        logger.error("list_memories error: %s", e)
        return {"agent_id": agent_id, "memories": [], "total": 0, "error": str(e)}


@app.delete("/memories/{fact_id}")
async def delete_memory(fact_id: str, agent_id: str = "default"):
    """删除一条特定记忆"""
    if cogni is None:
        return {"status": "error", "detail": "CogniMem 未初始化"}
    try:
        cogni.fact_network._delete_fact(fact_id)
        return {"status": "deleted", "fact_id": fact_id}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/memories/search")
async def search_memories(q: str = "", agent_id: str = "default", limit: int = 20):
    """搜索记忆"""
    if cogni is None:
        return {"agent_id": agent_id, "memories": [], "total": 0}
    try:
        result = cogni.recall(query=q, agent_id=agent_id, top_k=limit)
        facts = [f.to_dict() for f in result.get("facts", [])]
        return {
            "agent_id": agent_id,
            "memories": facts,
            "total": len(facts),
        }
    except Exception as e:
        return {"agent_id": agent_id, "memories": [], "total": 0, "error": str(e)}


@app.post("/consolidate")
async def consolidate(agent_id: str = "default"):
    """触发记忆归纳整合（抽象化 + 衰减 + 去重）"""
    if cogni is None:
        return {"agent_id": agent_id, "result": {}, "message": "CogniMem 未初始化"}
    result = cogni.consolidate(agent_id)
    return {
        "agent_id": agent_id,
        "result": result,
        "message": f"合并 {result.get('merged',0)} 条，"
                   f"抽象化 {result.get('abstracted',0)} 组，"
                   f"衰减 {result.get('decayed',0)} 条",
    }


@app.delete("/clear")
async def clear_memories(agent_id: str = "default"):
    """🗑️ 清除某个 Agent 的所有记忆"""
    if cogni is None:
        return {"agent_id": agent_id, "deleted": 0, "message": "CogniMem 未初始化"}
    try:
        result = cogni.reset_agent(agent_id)
        return {"agent_id": agent_id, **result}
    except Exception as e:
        return {"agent_id": agent_id, "deleted": 0, "message": str(e)}


class ConfirmRequest(BaseModel):
    fact_id: str
    agent_id: str = "default"


@app.post("/confirm")
async def confirm_fact(req: ConfirmRequest):
    """确认事实 → 提升置信度"""
    if cogni is None:
        return {"status": "error", "detail": "CogniMem 未初始化"}
    try:
        result = cogni.confirm(req.fact_id, req.agent_id)
        return result
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/challenge")
async def challenge_fact(req: ConfirmRequest):
    """质疑事实 → 降低置信度"""
    if cogni is None:
        return {"status": "error", "detail": "CogniMem 未初始化"}
    try:
        result = cogni.challenge(req.fact_id, req.agent_id)
        return result
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/versions/{fact_id}")
async def get_versions(fact_id: str):
    """获取事实版本历史"""
    if cogni is None:
        return {"fact_id": fact_id, "versions": [], "count": 0}
    try:
        versions = cogni.fact_network.get_versions(fact_id)
        return {"fact_id": fact_id, "versions": versions, "count": len(versions)}
    except Exception as e:
        return {"fact_id": fact_id, "versions": [], "count": 0, "error": str(e)}


# ══════════════════════════════════════════════════════════
#  系统健康检测 — 真正的智能诊断
# ══════════════════════════════════════════════════════════
#  检测项：
#    1. 记忆冲突率（矛盾/总事实）
#    2. API 健康状况（LLM 调用是否正常）
#    3. 配置完整性（关键环境变量是否缺失）
#    4. 记忆衰减率（低置信度事实占比）
#    5. 检索质量（是否有有效召回）
#    6. 工具可用性（Agent 能否正常执行）

import time as _time_module
import threading

# 简单健康追踪器（内存中）
# ⭐ v0.8: 使用滑动窗口计算错误率
_HEALTH = {
    "start_time": 0,
    "api_errors": 0,
    "api_calls": 0,
    "last_error_time": 0,
    "last_error_msg": "",
    # 滑动窗口 — 记录最近 100 次调用的结果
    "_error_window": [],       # list[float] — 每次调用的时间戳，失败为正数，成功为负数
    "_window_max": 100,
}
_health_lock = threading.Lock()


def _record_api_call(success: bool = True, error_msg: str = ""):
    """记录 API 调用结果到滑动窗口和全局计数器（线程安全）"""
    with _health_lock:
        _HEALTH["api_calls"] += 1
        now = time.time()
        if not success:
            _HEALTH["api_errors"] += 1
            _HEALTH["last_error_time"] = now
            _HEALTH["last_error_msg"] = str(error_msg)[:100]
            _HEALTH["_error_window"].append(now)  # 正数 = 失败
        else:
            _HEALTH["_error_window"].append(-now)  # 负数 = 成功

        # 裁剪窗口
        if len(_HEALTH["_error_window"]) > _HEALTH["_window_max"]:
            _HEALTH["_error_window"] = _HEALTH["_error_window"][-_HEALTH["_window_max"]:]


def _get_windowed_error_rate() -> float:
    """计算滑动窗口内的错误率（最近 100 次调用）"""
    window = _HEALTH.get("_error_window", [])
    if not window:
        return 0.0
    errors = sum(1 for v in window if v > 0)
    return errors / len(window)


@app.get("/health")
async def system_health(agent_id: str = "default"):
    """真正的系统健康检测。"""
    checks = {}
    issues = []
    score = 100

    # ── 1. 运行时间 ──
    uptime = time.time() - _HEALTH["start_time"] if _HEALTH["start_time"] > 0 else 0
    checks["uptime"] = f"{uptime / 3600:.1f}h"

    # ── 2. 配置完整性 ──
    config_issues = []
    if not os.environ.get("QWEN_API_KEY", ""):
        config_issues.append("QWEN_API_KEY 未配置")
    if not os.environ.get("DEEPSEEK_API_KEY", "") and not os.environ.get("QWEN_API_KEY", ""):
        config_issues.append("无可用 LLM API Key")
    if not llm:
        config_issues.append("LLM 客户端未初始化")
    checks["config"] = "✅" if not config_issues else f"⚠️ {'; '.join(config_issues)}"
    if config_issues:
        issues.append({"type": "config", "detail": "; ".join(config_issues), "severity": "high"})
        score -= len(config_issues) * 15

    # ── 2b. DB 连接状态 ──
    if cogni and cogni.fact_network and cogni.fact_network.db:
        try:
            conn = cogni.fact_network.db._get_conn()
            conn.cursor().execute("SELECT 1")
            cogni.fact_network.db._put_conn(conn)
            checks["db"] = "✅"
        except Exception as e:
            checks["db"] = f"⚠️ {e}"
            issues.append({"type": "db", "detail": str(e)[:60], "severity": "high"})
            score -= 20
    elif cogni:
        checks["db"] = "⚠️ 无 DB 适配器"
        score -= 10
    else:
        checks["db"] = "⚠️ CogniMem 未初始化"
        score -= 20

    # ── 2c. LLM 连接状态 ──
    if llm and llm.api_key:
        # 不实际调 API（太慢），只检查客户端初始化状态和预热记录
        ping_ok = getattr(llm, '_warmed_up', False)
        if ping_ok:
            checks["llm"] = "✅"
        else:
            # 尝试快速预热（只做一次，后面不走这路径）
            try:
                llm.chat(messages=[{"role": "user", "content": "ping"}], max_tokens=3, temperature=0.1)
                llm._warmed_up = True
                checks["llm"] = "✅"
            except Exception as e:
                checks["llm"] = f"⚠️ {str(e)[:40]}"
                issues.append({"type": "llm", "detail": str(e)[:60], "severity": "high"})
                score -= 20
    else:
        checks["llm"] = "⚠️ LLM 未初始化"
        score -= 20

    # ── 3. CogniMem 引擎状态 ──
    mem_issues = []
    if cogni is None:
        mem_issues.append("CogniMem 未初始化")
        score -= 30
    else:
        try:
            # agent_id=default 时汇总所有 agent 的数据
            if agent_id == "default":
                total = 0; contradictions = 0; core = 0; unreliable = 0; by_type = {}
                if cogni.fact_network.db:
                    conn = cogni.fact_network.db._get_conn()
                    cur = conn.cursor()
                    cur.execute("SELECT COUNT(*) FROM facts")
                    total = cur.fetchone()[0] or 0
                    cur.execute("SELECT COUNT(*) FROM facts WHERE confidence >= 0.9")
                    core = cur.fetchone()[0] or 0
                    cur.execute("SELECT COUNT(*) FROM facts WHERE confidence < 0.2")
                    unreliable = cur.fetchone()[0] or 0
                    cur.execute("SELECT COUNT(*) FROM facts WHERE contradictions != '[]'::JSONB AND contradictions IS NOT NULL")
                    contradictions = cur.fetchone()[0] or 0
                    cur.execute("SELECT fact_type, COUNT(*) FROM facts GROUP BY fact_type")
                    for r in cur.fetchall():
                        by_type[r[0]] = r[1]
                    cogni.fact_network.db._put_conn(conn)
            else:
                stats = cogni.get_stats(agent_id)
                total = stats.get("total_facts", 0)
                contradictions = stats.get("contradictions", 0)
                core = stats.get("core_beliefs", 0)
                unreliable = stats.get("unreliable", 0)
                by_type = stats.get("by_type", {})

            # 矛盾率
            if total > 0:
                conflict_rate = contradictions / total
                if conflict_rate > 0.3:
                    mem_issues.append(f"矛盾率 {conflict_rate:.0%} 偏高（{contradictions}/{total}）")
                    score -= 15
                elif conflict_rate > 0.1:
                    score -= 5

            # 不可靠事实率
            if total > 0 and unreliable > 0:
                unreliable_rate = unreliable / total
                if unreliable_rate > 0.7:
                    mem_issues.append(f"不可靠事实占比偏高 {unreliable_rate:.0%}")
                    score -= 5
                elif unreliable_rate > 0.5:
                    score -= 2

            # 事实总量
            if total == 0:
                mem_issues.append("暂无记忆数据")
                score -= 5
            elif total < 5:
                score -= 3

            # ★ v0.13: STM + 路由统计
            stm_count = 0
            router_stats = {}
            if cogni and cogni.fact_network:
                try:
                    stm_count = cogni.fact_network._stm_count(agent_id)
                except Exception:
                    pass
                try:
                    router_stats = cogni.recall_router.get_stats()
                except Exception:
                    pass

            checks["memory"] = {
                "total": total, "contradictions": contradictions,
                "core_beliefs": core, "unreliable": unreliable,
                "by_type": by_type,
                "stm_buffer": stm_count,    # v0.13
            }
            checks["router"] = router_stats  # v0.13

        except Exception as e:
            mem_issues.append(f"CogniMem 异常: {str(e)[:60]}")
            score -= 20

    if mem_issues:
        issues.append({"type": "memory", "detail": "; ".join(mem_issues), "severity": "medium"})

    # ── 4. API 健康（滑动窗口：最近 100 次调用） ──
    api_issues = []
    windowed_error_rate = _get_windowed_error_rate()
    total_calls = _HEALTH["api_calls"]
    total_errors = _HEALTH["api_errors"]
    if windowed_error_rate > 0.3:
        api_issues.append(f"API 错误率 {windowed_error_rate:.0%}（最近 100 次）")
        score -= 20
    elif windowed_error_rate > 0.1:
        api_issues.append(f"API 偶发错误 {windowed_error_rate:.0%}（最近 100 次）")
        score -= 5
    if _HEALTH["last_error_msg"]:
        age = time.time() - _HEALTH["last_error_time"]
        if age < 300:  # 5分钟内
            api_issues.append(f"最近错误: {_HEALTH['last_error_msg'][:50]}")
            score -= 10
    checks["api"] = {
        "total_calls": total_calls,
        "errors": total_errors,
        "windowed_error_rate": f"{windowed_error_rate:.1%}",
        "window_size": min(total_calls, _HEALTH["_window_max"]),
    }
    if api_issues:
        issues.append({"type": "api", "detail": "; ".join(api_issues), "severity": "high"})

    # ── 5. Agent 工具可用性 ──
    tool_issues = []
    if agent is None:
        tool_issues.append("Agent 引擎未初始化")
        score -= 20
    else:
        tool_count = len(tool_registry._tools) if tool_registry else 0
        if tool_count == 0:
            tool_issues.append("无可用工具")
            score -= 15
        checks["tools"] = tool_count
    if tool_issues:
        issues.append({"type": "agent", "detail": "; ".join(tool_issues), "severity": "high"})

    # ── 最终评分 ──
    score = max(0, min(100, score))
    if score >= 80:
        level = "healthy"
        label = "✅ 系统健康"
    elif score >= 50:
        level = "warning"
        label = f"⚠️ {score}"
    else:
        level = "critical"
        label = f"🔴 {score}"

    return {
        "score": score,
        "label": label,
        "level": level,
        "uptime_seconds": uptime,
        "checks": checks,
        "issues": issues,
    }


# ═══════════════════════════════════════════════════════════════
# 跨 Agent 记忆总线端点（受 Universal Agent OS 启发）
# ═══════════════════════════════════════════════════════════════

@app.post("/memory-bus")
async def cross_agent_recall(query: str = "", agent_ids: str = "",
                              top_k: int = 10):
    """跨 Agent 记忆总线：同时查询多个 Agent 的记忆"""
    if not query or not agent_ids:
        return {"status": "error", "message": "需要 query 和 agent_ids 参数"}
    ids = [a.strip() for a in agent_ids.split(",") if a.strip()]
    if not ids:
        return {"status": "error", "message": "至少指定一个 Agent ID"}
    try:
        result = cogni.recall_cross_agent(query=query, agent_ids=ids, top_k=top_k)
        return {
            "status": "ok",
            "count": result["count"],
            "facts": [
                {
                    "fact": f"{f.subject} {f.predicate} {f.object}",
                    "confidence": f.confidence,
                    "agent_id": f.agent_id,
                    "citation": f.citation,
                    "stale_warning": f.stale_warning,
                }
                for f in result["facts"]
            ],
            "sources": result["sources"],
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ═══════════════════════════════════════════════════════════════
# 审计日志端点（受 DREAM audit ledger 启发）
# ═══════════════════════════════════════════════════════════════

@app.get("/audit")
async def query_audit(agent_id: str = "",
                      operation: str = "",
                      limit: int = 50,
                      offset: int = 0,
                      since_hours: int = 0):
    """查询审计日志"""
    db = cogni.fact_network.db if cogni and cogni.fact_network else None
    if not db or not hasattr(db, 'query_audit'):
        return {"status": "error", "message": "审计日志不可用（无数据库）"}

    try:
        rows = db.query_audit(
            agent_id=agent_id,
            operation=operation,
            limit=limit,
            offset=offset,
            since_hours=since_hours,
        )
        return {"status": "ok", "count": len(rows), "entries": rows}
    except Exception as e:
        return {"status": "error", "message": str(e)}
