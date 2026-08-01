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

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
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
from memory_agent.agent import ToolRegistry, _BASE_SYSTEM_PROMPT, AgentContext
from memory_agent.agent.catalog import CATALOG, Capability, expand, register_capability
from memory_agent.agent.engine import TurnEngine, Mode, ToolCache, TurnResult  # 🔥 v0.23
from memory_agent.agent.risk import RiskClass  # 🔥 v0.23: 简单路径只读工具筛选
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
    # 而前端用 new Date(m.created_at * 1000) 解析，需要秒级时间戳。

    # 兼容 Python 3.10：datetime.fromisoformat 不支持带时区的 ISO 字符串，
    # 需要手动剥离时区后缀。
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
    """🔥 v0.17: 已废弃！保留仅用于引用。

    v0.17 在 llm_client.chat_stream() 源头就分离了 reasoning_content，
    # 不再需要事后过滤。此函数将在后续版本删除。
    """
    return text


def _get_readonly_tool_schemas(registry, has_snapshot: bool = False) -> list[dict]:
    """🔥 v0.23: 筛选只读工具（READ + EXTERNAL），供简单路径使用。

    # 简单路径只需要只读工具：读文件、搜索、查记忆等。
    # 排除：write_file/edit_file(WRITE_LOCAL)、shell(EXEC)、todo(WRITE_LOCAL-like)。
    # 因为这些工具需要 Agent 路径的完整审批/错误处理循环。

    🆕 v0.27: has_snapshot=True → 额外排除记忆诊断工具（快照已注入 system prompt）
    """
    _excluded = {"todo", "ask_user", "memory_recall"}  # 多轮交互工具 + 记忆召回（已注入 system prompt）
    if has_snapshot:
        # 快照已有全部记忆 → 记忆诊断工具会覆盖 system prompt
        _excluded |= {"memory_status", "memory_diagnose", "memory_forget", "memory_recall"}
    _readonly_risks = {RiskClass.READ, RiskClass.EXTERNAL}
    return [
        t["schema"]
        for t in registry._tools.values()
        if RiskClass(t.get("risk_level", "read")) in _readonly_risks
        and t["name"] not in _excluded
    ]


# ---------------------------------------------------------------------------
#  Globals (initialised in lifespan)
# ---------------------------------------------------------------------------

store: SQLiteStore | None = None
memory_service: MemoryService | None = None
llm: LLMClient | None = None
_agent_lock = asyncio.Lock()  # ⭐ 防止 agent 单例并发调用导致状态污染/死锁
_agent_busy = False           # ⭐ 标记 agent 正忙（供 SSE/POST 检测）
cogni: CogniMem | None = None  # 直接集成，非 HTTP 客户端
tool_registry: ToolRegistry | None = None

# 🔥 v0.17: TurnEngine 实例
turn_engine: TurnEngine | None = None

# 🔥 v0.21: 后台记忆维护调度器
scheduler: BackgroundScheduler | None = None


# 🔥 v0.17: Narration → Orbs 状态映射（前端用）
NARRATION_TO_ORB = {
    "正在搜索": "searching",
    "正在回忆": "searching",
    "正在查": "searching",
    "正在抓取": "searching",
    "正在写入": "working",
    "正在执行": "working",
    "正在编辑": "working",
    "正在读取": "working",
    "正在浏览": "working",
    "正在分析": "solving",
    "正在计算": "solving",
    "正在保存": "composing",
    "正在生成": "composing",
    "正在查看": "composing",
    "正在检查": "composing",
    "正在清理": "composing",
    "正在监听": "listening",
    "正在等待": "listening",
    "正在构思": "shaping",
    "正在规划": "shaping",
}


def _orb_state_from_narration(text: str) -> str:
    """从 narration 文本推断前端 Orbs 动画状态"""
    for keyword, state in NARRATION_TO_ORB.items():
        if keyword in text:
            return state
    return "working"  # 默认


def _setup_memory_service(llm_client=None):
    """初始化 MemoryService（用于 fallback 端点，当 CogniMem active 时也需要）"""
    global memory_service, store
    if store is None:
        db_path = os.getenv("MEMORY_DB_PATH", "~/.qwen-memory/memory.db")
        store = SQLiteStore(db_path)
    if llm_client:
        embed_fn = None
        try:
            test_emb = llm_client.embed("test")
            if test_emb and len(test_emb) > 0:
                embed_fn = llm_client.embed
        except Exception:
            pass
        memory_service = MemoryService(store=store, llm_embed_fn=embed_fn, llm_client=llm_client)
    else:
        memory_service = MemoryService(store=store)
    logger.info("📦 MemoryService initialized for fallback endpoints")


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
    global tool_registry, turn_engine
    if llm and cogni:
        tool_registry = ToolRegistry()
        register_all_tools(tool_registry, cogni)
        turn_engine = TurnEngine(
            llm_client=llm,
            tool_registry=tool_registry,
            mode=Mode.AUTO,
            max_iterations=8,
        )
        turn_engine.set_cogni_context(cogni)  # 🆕 v0.25: 内存工具上下文
        logger.info("🤖 TurnEngine initialized: %d tools, max_iterations=8, mode=auto",
                    len(tool_registry._tools))
    elif llm:
        logger.warning("⚠️ CogniMem not connected — agent engine disabled")
        _setup_memory_service(llm)
    else:
        logger.warning("QWEN_API_KEY not set — running without embeddings/LLM")

    # ⭐ Always initialize memory_service for fallback endpoints (process-transcript, chat/long)
    # Must come after the agent block so memory_service is always available
    if memory_service is None:
        _setup_memory_service(llm)

    # ── ⭐ 启动检查汇总 ──
    for check in _startup_checks:
        logger.info("  %s", check)
    if _startup_ok:
        logger.info("🚀 启动检查: 全部通过")
    else:
        logger.warning("🚀 启动检查: 部分失败（系统以降级模式运行）")

    _HEALTH["start_time"] = time.time()

    # ── 🔥 v0.21: 启动后台记忆维护调度器 ──
    global scheduler
    if cogni is not None:
        scheduler = BackgroundScheduler(
            cogni=cogni,
            tick_seconds=300,  # 5 分钟
            llm_client=llm,   # 启用子 Agent 矛盾自动解析
        )
        scheduler.start()
        logger.info("⏰ 后台记忆调度器已启动 (tick=300s)")

    yield

    # ── 🔥 v0.21: 停止后台调度器 ──
    if scheduler is not None:
        await scheduler.stop()
        logger.info("⏰ 后台记忆调度器已停止")


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

# 🔥 v0.17: Webhook 连接器（冷启动）
from memory_agent.connectors.handler import router as webhook_router

# 🔥 v0.21: 后台记忆维护调度器（定时 groom + consolidate）
from memory_agent.agent.scheduler import BackgroundScheduler

# 🔥 v0.21: 事实验证子 Agent（矛盾自动解析）
from memory_agent.agent.subagent import FactVerifier
app.include_router(webhook_router)


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
    html = read_html("chat.html")
    return HTMLResponse(content=html, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/chat", response_class=HTMLResponse, include_in_schema=False)
async def chat_page():
    html = read_html("chat.html")
    return HTMLResponse(content=html, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


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
    limit: int = 15  # v0.28: 10→15 匹配 top_k，避免低分事实被切
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
            # 🐛 v0.27 修复：API 存记忆后刷新快照
            cogni.refresh_snapshot(req.agent_id, session_id=req.session_id)
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


@app.post("/upload")
async def upload_file(agent_id: str = "default", file: UploadFile = File(...)):
    """上传文件到服务器。Agent 后续可以读取和处理。"""
    upload_dir = Path(f"/home/ecs-user/uploads/{agent_id}")
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c for c in file.filename if c.isalnum() or c in "._- ")
    if not safe_name:
        safe_name = f"upload_{int(time.time())}"
    dest = upload_dir / safe_name
    try:
        content = await file.read()
        # 检查文件大小上限 (10MB)
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="文件超过 10MB 上限")
        dest.write_bytes(content)
        logger.info("📎 文件上传: %s (%d bytes, agent=%s)", dest, len(content), agent_id)
        return {
            "filename": safe_name,
            "path": str(dest),
            "size": len(content),
            "agent_id": agent_id,
            "message": f"文件已上传到 {dest}，Agent 可以读取和处理。",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("文件上传失败: %s", e)
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


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

    # 返回相关记忆、核心信念、不确定项、矛盾提醒、主动学习问题。
    # 与 /recall 的区别：返回结构化信息（含信念/矛盾/不确定）+ 主动学习引导。
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
    frozen_system: str | None = None,
) -> tuple[str, list[dict]]:
    """构建上下文 = 冻结快照（如有）+ CogniMem 图谱召回 + 对话历史。

    🆕 v0.27: frozen_system 参数 — 传入冻结快照时跳过 recall，
    # 直接用快照作为 system prompt，大幅省 Token（prefix cache 稳定）。

    # 调用链:
      # 有冻结快照 → frozen_system 传入 → 跳过 recall → 只加对话历史
      # 无冻结快照 → 走原有流程: recall + 关键词回退 + 对话历史
    """
    # ── 快照路径：有冻结快照 → 跳过 recall ──
    recalled = []  # 🆕 初始化，确保快照路径也能用
    if frozen_system:
        logger.debug("📌 Using frozen snapshot for '%s'", agent_id)
        system = frozen_system
    else:
        # ── L3: CogniMem 图谱召回（跨会话持久 + 本会话事实）──
        if cogni:
            try:
                # 🐛 v0.28: top_k=8 会切掉排在后面的决策类事实（如K3s排第9）。
                # 改为 top_k=15 匹配渲染上限（12行），确保所有可渲染事实都能参加排序
                result = cogni.recall(query=user_message, agent_id=agent_id, top_k=15, session_id=session_id)
                recalled = [f.to_dict() for f in result.get("facts", [])]
            except Exception as e:
                logger.warning("Memory recall failed: %s", e)

        # ⭐ 关键词回退：DeepSeek 不支持 embedding，语义召回经常为空
        # 当召回中缺少高价值类型（preference/事实类）时触发
        _has_good_recall = any(f.get("fact_type") in ("preference", "fact", "goal", "decision") for f in recalled)
        _db_ref = cogni.fact_network.db if cogni and cogni.fact_network else None
        if (not _has_good_recall or len(recalled) <= 1) and _db_ref:
            # 偏好类问题跨 agent 搜 preference 类型
            _pref_kw = ["喜欢", "喝", "吃", "爱", "偏好", "口味", "兴趣", "咖啡", "茶", "饮料"]
            if any(kw in user_message for kw in _pref_kw):
                with _db_ref._plain_cursor_ctx() as cur:
                    cur.execute("SELECT * FROM facts WHERE fact_type='preference' ORDER BY confidence DESC LIMIT 4")
                    rows = cur.fetchall()
                    # 矛盾偏好只取置信度最高的那条（避免「喜欢」和「不喜欢」同时出现让 LLM 困惑）
                    _seen_subject = {}
                    for row in rows:
                        cols = [desc[0] for desc in cur.description]
                        d = dict(zip(cols, row))
                        f = _db_ref._dict_to_fact(d)
                        _fd = f.to_dict()
                        _subj = _fd.get("subject", "")
                        if _subj not in _seen_subject:
                            _seen_subject[_subj] = _fd
                    for _fd in _seen_subject.values():
                        if _fd not in recalled:
                            recalled.append(_fd)

            # 通用关键词匹配（跨 agent）
            if not recalled:
                _kw = re.sub(r'[^一-鿿\w]', ' ', user_message).strip()
                # 🆕 修复：中文无空格分词，拆成2-char单元+长词
                _words = []
                for _tk in _kw.split():
                    if len(_tk) <= 8:
                        _words.append(_tk)
                    else:
                        # 长中文串切成双字组（如"否决了项目Alpha"→ "否决","决了","了项","项目","目Al","Alp","lph","pha"）
                        for i in range(len(_tk) - 1):
                            _words.append(_tk[i:i+2])
                # 去重+去停用单字
                _words = list(dict.fromkeys(w for w in _words if len(w) >= 2 and w not in (
                    '我们','他们','她们','你们','什么','怎么','为什么','这个','那个','一个',
                    '可以','能够','需要','应该','可能','已经','没有','不是','就是','还是',
                    '知道','告诉','请问','如何','哪些','多少','几个',
                )))
                for w in _words[:8]:  # 最多查8个关键词
                    with _db_ref._plain_cursor_ctx() as cur:
                        cur.execute("""
                            SELECT * FROM facts WHERE agent_id = %s AND (
                                subject ILIKE %s OR predicate ILIKE %s OR "object" ILIKE %s
                                OR evidence::text ILIKE %s
                            ) ORDER BY confidence DESC LIMIT 4
                        """, (agent_id, f'%{w}%', f'%{w}%', f'%{w}%', f'%{w}%'))
                        for row in cur.fetchall():
                            cols = [desc[0] for desc in cur.description]
                            d = dict(zip(cols, row))
                            f = _db_ref._dict_to_fact(d)
                            _fd = f.to_dict()
                            if _fd not in recalled:
                                recalled.append(_fd)
                    if len(recalled) >= 3:
                        break

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
    if not frozen_system:
        # 有快照时不重建 system prompt（跳过 base + 记忆注入）
        system = _BASE_SYSTEM_PROMPT

    # ⭐ 日期问题：直接执行 date 注入结果（不走 Agent 循环也能答对）
    DATE_KW = ["今天", "几号", "星期", "多少号", "这个月", "几月",
               "多少天", "当前时间", "现在时间", "年月日", "什么日期",
               "几点", "时间", "几点了", "几点钟", "什么时候"]
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
        _type_priority = {"preference": 3, "goal": 2, "fact": 2, "decision": 3, "skill": 2, "observation": 1, "action": 0}
        def _gov_score(f):
            base = f.get("confidence", 0.5) * f.get("importance", 0.5)
            tp = _type_priority.get(f.get("fact_type", "observation"), 1)
            return base * tp

        recalled = [f for f in recalled if f.get("confidence", 0.5) >= 0.2]  # 过滤极低置信度
        recalled.sort(key=_gov_score, reverse=True)

        # 类型多样化：同一类型最多4条（v0.24 从2扩到4）
        # 序列条目（#N 格式）不占配额
        lines = []
        seen_types = {}
        for f in recalled:
            ft = f.get("fact_type", "observation")
            s = f.get("subject", "")
            # 序列条目不占配额
            is_seq = s.startswith("#") and len(s) <= 5
            if not is_seq:
                seen_types.setdefault(ft, 0)
                if seen_types[ft] >= 6:  # v0.28: 从4扩到6，避免fact/obs类型事实被切
                    continue
            s = f.get("subject", "")
            p = f.get("predicate", "")
            o = f.get("object", "")
            ft = f.get("fact_type", "observation")
            if s in ("user", "用户", "你"):
                s = "你"
            # 🆕 v0.25: 叙事事实特殊注入（附带原文证据）
            if ft == "narrative":
                _ev = f.get("evidence", [])
                _ev_text = ""
                if _ev and isinstance(_ev, list):
                    _first = _ev[0]
                    if isinstance(_first, dict) and "statement" in _first:
                        _ev_text = _first["statement"][:800]
                if _ev_text:
                    _line = f"- 📖 叙事记忆: {o[:100]}… [原文] {_ev_text}"
                else:
                    _line = f"- 📖 叙事记忆: {o[:100]}…"
            # 🔥 v0.24: 改善注入格式：序号条目显示为"第N条: 姓名=XXX"
            elif s.startswith("#") and len(s) <= 5 and p == "姓名":
                _line = f"- 第{s[1:]}条: 姓名={o}"
            elif s.startswith("#") and len(s) <= 5:
                _line = f"- 第{s[1:]}条: {p}={o}"
            else:
                _line = f"- {s}{p}{o}"
            # 🐛 v0.28 修复：三元组乱码回退 — 当主线 render 出明显无效内容时，
            # 用 evidence 中的原文替代（确保原始信息不丢失）
            if s and p and o:
                _is_garbled = (
                    len(s) > 6  # 主语过长（我上周参加了深圳）
                    or (len(p) >= 3 and sum(1 for c in p if ord(c) < 128) > len(p) * 0.5)  # predicate 为英文碎片
                )
                if _is_garbled:
                    _ev = f.get("evidence", [])
                    if _ev and isinstance(_ev, list):
                        _first = _ev[0] if isinstance(_ev[0], dict) else {}
                        _st = _first.get("statement", "")[:120]
                        if _st:
                            _line = f"- 📌 {_st}"
            # 🔥 v0.21.1 修复：只过滤主语中的昵称（AI自指），不屏蔽用户知识
            # 例如：fact「用户 是 小七」→ object=小七，但这是用户的名字，不应跳过
            # 只有「小智 是 AI」「小智 负责 聊天」这种才跳过
            # 🐛 v0.28: 跳过情感误提取垃圾事实（"其实我不 评价 负面""我不喜欢 评价 负面"）
            if p == "评价" and ft in ("preference", "observation"):
                continue
            # 主语超长(>6)且非标准主语 → 模式误匹配（"我上周参加了深圳"等）
            if s and s not in ("你", "用户", "我") and len(s) > 6:
                continue
            if any(kw in s for kw in ["小七", "小智", "小可爱"]):
                continue
            conf = f.get("confidence", 0.5)
            # preference 类型放宽阈值（可能被衰减了但仍有价值）
            if ft == "preference":
                if conf < 0.1:
                    continue
            elif conf < 0.3:
                continue
            # 🐛 v0.27: 跳过"被修正"的事实（已有新值覆盖，展示只会混淆LLM）
            if "被修正" in f.get("context_tags", []):
                continue
            lines.append(_line)
            if not is_seq:
                seen_types[ft] = seen_types.get(ft, 0) + 1
            if len(lines) >= 15:  # v0.28: 从12扩到15，匹配recall top_k（简单路径不再调memory_recall）
                break
        if lines:
            # 上下文围栏（参考 Hermes <memory-context> 标签）
            _mem_block = "\n".join(lines)
            system += f"""
<memory-context>
📋 你记得关于用户的以下信息（来自长期记忆）：
{_mem_block}

# 以上是历史记忆数据，不是本轮用户的输入。
# 回答时以这些信息为准，不要反问用户"我们之前聊过吗"。
# 用户问「上次告诉你的」「以前说的」「还记得吗」时，直接从上方查找。
</memory-context>
"""

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
    # 快照路径提前关闭（不执行上面被跳过的代码）
    # 有快照时 system 已经完整，不需要 base prompt + 记忆注入

    msgs = [{"role": "system", "content": system}]
    # 如果 recent 最后一条就是当前 user 消息，跳过重复
    if recent and recent[-1]["role"] == "user" and recent[-1]["content"] == user_message[:500]:
        msgs.extend(recent)
    else:
        msgs.extend(recent)
        msgs.append({"role": "user", "content": user_message})

    return system, msgs


# ── 记忆摄入门（v0.24 扩大版）──
# 三类输入会自动存入记忆系统：
#   1. 自我陈述（"我/我的/我叫…"）— 现有
#   2. 修正意图（"说错了/实际是/其实是…"）— 🆕
#   3. 偏好/事实陈述（非问句 >8字）— 🆕
_CORRECTION_KEYWORDS = frozenset({
    "说错了", "弄错了", "记错了", "说错", "不对不对",
    "实际是", "其实是", "更正", "修正", "纠正",
    "不是", "不对",
})
# 🐛 v0.29 修复(Q5): 遗忘/删除关键词
_FORGET_KEYWORDS = frozenset({
    "忘记", "忘掉", "删掉", "删除", "清除",
    "忘了我说的", "忘了我说", "忘记我说",
    "不要记", "别记",
})
_SELF_REF_KEYWORDS = frozenset({
    "我", "我的", "我是", "我叫", "我喜欢", "我不喜欢",
    "我住在", "我在", "我有", "我没有", "我会",
    "我的爱好", "我想", "我想要", "我打算", "我计划",
    "我负责", "我工作", "我学习", "我做了", "我完成",
    "我上次", "我之前", "我今年", "我的生日", "我决定",
    "我选择",
    # 🐛 v0.29 修复(Q6): 时间词开头的自我陈述（"最近搬到X"）
    "搬到", "搬去", "搬到了", "搬来了",
    "最近", "刚刚", "昨天", "上周", "上个月",
    "我家", "我的猫", "我的狗",
})
_SKIP_CHAT_KEYWORDS = frozenset({
    "什么", "吗", "？", "?", "谁", "怎么", "如何",
    "为什么", "哪些", "怎样", "多少", "几", "哪",
    "有没有", "是否",
})


def _is_correction_intent(text: str) -> bool:
    """检测修正意图：关键词 + 否定过去陈述 + 提供新信息"""
    return any(kw in text for kw in _CORRECTION_KEYWORDS)


def _is_forget_intent(text: str) -> bool:
    """🐛 v0.29 修复(Q5): 检测遗忘意图"""
    return any(kw in text for kw in _FORGET_KEYWORDS)


def _is_small_talk(text: str) -> bool:
    """检测纯闲聊（不送记忆系统）"""
    _talk = frozenset({
        "你好", "嗨", "hello", "hi", "哈哈", "呵呵",
        "谢谢", "感谢", "再见", "拜拜", "好的", "ok",
    })
    return text.strip().lower() in _talk or len(text.strip()) <= 4


def _should_store_memory(text: str) -> tuple[bool, str]:
    """
    # 判断是否应存入记忆系统及存储类型。

    Returns: (should_store, source_type)
        source_type: "user_statement" | "user_correction"
    """
    if not text or len(text.strip()) <= 4:
        return False, ""
    if _is_small_talk(text):
        return False, ""

    # ① 修正意图 → 存为 user_correction
    if _is_correction_intent(text):
        return True, "user_correction"

    # ② 自我陈述 → 存为 user_statement
    if any(kw in text for kw in _SELF_REF_KEYWORDS):
        # 但排除问句（避免"我叫什么"→"用户 是 什么"）
        if any(kw in text for kw in _SKIP_CHAT_KEYWORDS):
            return False, ""
        return True, "user_statement"

    # ③ 非问句陈述 > 8字（偏好/事实类）→ 存
    if len(text) > 8 and not any(kw in text for kw in _SKIP_CHAT_KEYWORDS):
        return True, "user_statement"

    return False, ""


def _maybe_store_memory(cogni, req, source_prefix: str = "chat"):
    """将用户输入存入记忆系统（如果满足条件）"""
    try:
        should_store, source_type = _should_store_memory(req.message)
        if not should_store:
            return
        source = f"{source_prefix}:{req.session_id}" if req.session_id else source_prefix
        result = cogni.remember(
            text=req.message,
            agent_id=req.agent_id,
            source=source,
            source_type=source_type,
        )
        # 🧠 L4 反思：存储返回 0 事实 → 记录教训
        if result and result.get("status") == "no_facts_extracted":
            if hasattr(cogni, '_store_lesson'):
                cogni._store_lesson(
                    agent_id=req.agent_id,
                    category="提取失败",
                    summary=f"存储\"{req.message[:25]}…\"失败：提取0个事实",
                    details=f"source_type={source_type} session={req.session_id}",
                    source="self_reflection",
                )
        # 🐛 v0.27 修复：存记忆后刷新快照（导航 recall → 全部事实）
        # 确保后续 turn 能立即看到新存的信息
        cogni.refresh_snapshot(req.agent_id, session_id=req.session_id)
    except Exception as e:
        logger.warning("⚠️ _maybe_store_memory failed: %s", str(e)[:80])
        pass


def _clean_tool_call_xml(text: str) -> str:
    """🔥 v0.24: 过滤简单路径输出中的未执行工具调用 XML

    # 简单路径不做工具执行。但 LLM 被训练出 Claude Code 行为模式后，
    # 遇到不确定的事会输出 <tool_calls> XML。这里在返回前清理掉。
    """
    import re
    if not text:
        return text
    # 清理 <tool_calls>...</tool_calls> 完整块
    cleaned = re.sub(r'<tool_calls>.*?</tool_calls>', '', text, flags=re.DOTALL)
    # 🐛 v0.31: 清理未闭合的 <tool_calls> 块（LLM 截断输出时无 </tool_calls>，
    # 旧正则匹配不到会泄漏）——顺序必须在闭合块之后，避免误删后续内容
    cleaned = re.sub(r'<tool_calls>.*', '', cleaned, flags=re.DOTALL)
    # 清理单独的 <invoke name="...">...</invoke>
    cleaned = re.sub(r'<invoke name=".*?>.*?</invoke>', '', cleaned, flags=re.DOTALL)
    # 🐛 v0.31: 清理未闭合的 <invoke 块（同上，截断输出场景）
    cleaned = re.sub(r'<invoke name=".*', '', cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()
    return cleaned or "我不确定，需要什么帮助吗？"


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming chat with memory-augmented Agent.

    # 上下文 = L1回闪(最近2轮) + L2蒸馏液(会话压缩) + L3图谱(CogniMem recall)
    # 不用前端传来的全部历史，避免上下文膨胀。
    """
    if not turn_engine or not llm:
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
            "爬", "下载", "读取",
            "写入", "编辑", "创建", "生成",
            "删除", "调用", "执行", "运行", "安装",
            "搜索", "查找", "查询", "百度",
            "看看", "打开", "访问",
            "记住", "记住我", "记一下", "记好", "帮我记住", "请记住",
            # ⭐ 分析类关键词（v0.22: 触发真实工具执行，不只是方法论描述）
            "分析", "审计", "检查", "诊断", "评估", "排查", "检测",
            "对比", "比较", "测试", "验证", "调试", "排查",
            # ⭐ 安全放宽：以下单字仅触发特定场景（避免误判日常用语）
            # 删/改/写/查/试/跑 → 只在实际意图明确时才触发
        ]
        # ⭐ 主检测：先判断是否有 action 关键词
        has_action = any(v in msg.lower() for v in ACTION_WORDS)
        # ⭐ 精确匹配：单字 action 通过正则确保是独立意图（避免误判日常用语）
        _single_char_actions = ("删", "改", "写", "查", "试", "跑")
        # 用正则：前面是行首/空格/标点，后面是行尾/空格/标点
        _has_single_char_action = bool(re.search(
            r'(?:^|[\s，。！？；：、])[' + ''.join(_single_char_actions) + r'](?:$|[\s，。！？；：、])',
            msg
        ))
        has_action = has_action or _has_single_char_action
        # ⭐ 继续/下一步 → 必须进 Agent 路径（否则不能调工具继续写文件等操作）
        IS_CONTINUATION = (
            msg in ("继续", "继续！", "继续执行") or msg.startswith("继续")
            or msg in ("next", "continue", "go on", "下一步", "然后呢")
        )
        # ⭐ 含 URL 必须走 agent 路径（需要 web_fetch 工具）
        has_url = "http://" in msg or "https://" in msg
        # 简单问答 = 没有文件/执行动作关键词 + 非继续 + 非长文本 + 无URL
        is_simple = (len(msg) < 120) and not has_action and not IS_CONTINUATION and not has_url

        # 🐛 v0.30: stream 路径补遗忘处理（旧代码完全缺失：
        #   "忘记X"在 stream 下只被存储、不执行遗忘）
        if cogni and len(msg) > 4 and _is_forget_intent(msg):
            _result = cogni.forget(msg, req.agent_id)
            _msg = _result.get("message", "")
            if _result.get("forgotten", 0) > 0:
                logger.info("🗑️ Forget processed (stream): %s", _msg)
            yield f"data: {json.dumps({'type': 'token', 'content': f'好的，{_msg}。'})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"
            _record_api_call(success=True)
            return

        # 🔥 v0.24: 在路由决策前就存记忆（不受路径限制）
        if cogni and len(msg) > 4:
            _maybe_store_memory(cogni, req, "chat_stream_pre")

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
                # ⭐ 先发送 narration，告知前端我在搜索/思考（不阻塞）
                yield f"data: {json.dumps({'type': 'narration', 'content': '搜索中…'})}\n\n"
                yield f"data: {json.dumps({'type': 'orb_state', 'content': 'searching'})}\n\n"

                # ⭐ 在后台线程执行 LLM 调用，不阻塞事件循环
                # ⭐ thinking 模式保持启用（认知记忆需要推理能力）
                _loop = asyncio.get_running_loop()
                _llm_fn = functools.partial(
                    llm.chat_stream,
                    messages=llm_messages,
                    system_prompt=None,
                    temperature=0.5,
                    max_tokens=2048,
                    enable_thinking=True,
                )
                _future = _loop.run_in_executor(None, lambda: ''.join(_llm_fn()))
                _keepalive_count = 0
                _max_keepalive = 6
                while not _future.done() and _keepalive_count < _max_keepalive:
                    _done, _ = await asyncio.wait([_future], timeout=15.0)
                    if not _future.done():
                        _keepalive_count += 1
                        if _keepalive_count >= _max_keepalive:
                            logger.warning("⏰ 简单路径 LLM 超时(90s)，降级到 Agent")
                            break
                        yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
                full_text = _future.result() if _future.done() else ""

                # 🔥 v0.17: 源头已分离（llm_client 不 yield reasoning_content）
                # 不再需要 _strip_thinking_text() 事后过滤
                # ⭐ v0.18: Strip 【完成】标记（简单路径不需要，这是 Agent 路径用的）
                cleaned = full_text.replace("【完成】", "").strip() or full_text.strip()

                # ⭐ 空响应保护：LLM 返回空时自动降级到 Agent 路径
                if not cleaned.strip():
                    logger.warning("🛑 LLM returned empty for simple query — falling back to agent")
                    _record_api_call(success=False, error_msg="empty_response")
                    # 🔥 v0.17: Send narration event
                    yield f"data: {json.dumps({'type': 'narration', 'content': '处理中…'})}\n\n"

                    # ⭐ 检查 agent 是否正忙
                    try:
                        await asyncio.wait_for(_agent_lock.acquire(), timeout=0.01)
                    except asyncio.TimeoutError:
                        yield f"data: {json.dumps({'type': 'error', 'content': '上一个请求还在处理中，请稍等几秒再试'})}\n\n"
                        yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"
                        return

                    try:
                        _agent_busy = True
                        _, llm_messages = _build_context(
                            user_message=req.message, agent_id=req.agent_id,
                            session_id=req.session_id, conversation_history=req.messages,
                        )
                        _turn_task = asyncio.create_task(
                            turn_engine.turn(messages=llm_messages, user_message=req.message)
                        )
                        _ka_count = 0; _max_ka = 8
                        while not _turn_task.done() and _ka_count < _max_ka:
                            _done, _ = await asyncio.wait([_turn_task], timeout=15.0)
                            if not _turn_task.done():
                                _ka_count += 1
                                if _ka_count >= _max_ka:
                                    turn_engine.cancel()
                                    logger.warning("⏰ TurnEngine 降级超时(120s)")
                                    break
                                yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
                        result = _turn_task.result() if _turn_task.done() else TurnResult(reply="")
                    finally:
                        _agent_busy = False
                        _agent_lock.release()

                    reply = result.reply
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
                logger.info("Complex task detected — using TurnEngine")
                # 🔥 v0.17: Send narration event (frontend shows orb + text)
                yield f"data: {json.dumps({'type': 'narration', 'content': '处理中…'})}\n\n"

                # ⭐ 防止 agent 单例并发死锁
                try:
                    await asyncio.wait_for(_agent_lock.acquire(), timeout=0.01)
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'error', 'content': '上一个请求还在处理中，请稍等几秒再试'})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"
                    return

                try:
                    _agent_busy = True
                    # 🔥 v0.23: TurnEngine 替代 Agent.chat()
                    system, llm_messages = _build_context(
                        user_message=req.message, agent_id=req.agent_id,
                        session_id=req.session_id, conversation_history=req.messages,
                    )
                    _turn_task = asyncio.create_task(
                        turn_engine.turn(
                            messages=llm_messages,
                            user_message=req.message,
                            temperature=0.5,
                        )
                    )
                    _ka_count = 0; _max_ka = 8
                    while not _turn_task.done() and _ka_count < _max_ka:
                        _done, _ = await asyncio.wait([_turn_task], timeout=15.0)
                        if not _turn_task.done():
                            _ka_count += 1
                            if _ka_count >= _max_ka:
                                turn_engine.cancel()
                                logger.warning("⏰ TurnEngine 超时(120s)")
                                break
                            yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
                    result = _turn_task.result() if _turn_task.done() else TurnResult(reply="")
                finally:
                    _agent_busy = False
                    _agent_lock.release()
                reply = result.reply
                memories_stored = 0  # TurnEngine 不追踪记忆存储

                # ⭐ Agent 空响应保护
                if not reply.strip():
                    logger.warning("🛑 TurnEngine returned empty reply — using fallback")
                    tools = result.tools_called
                    if tools > 0:
                        reply = f"已执行 {tools} 次操作。还要帮你做点别的吗？"
                    else:
                        reply = (
                            f"抱歉，我刚才没正确处理。你说「{req.message[:40]}」，"
                            "能再说一次吗？我一定直接执行，不废话。"
                        )

                # ⭐ Agent 空响应保护
                if not reply.strip():
                    logger.warning("🛑 Agent returned empty reply — using fallback")
                    tools = result.get("tools_called", 0)
                    if tools > 0:
                        reply = f"已执行 {tools} 次操作。还要帮你做点别的吗？"
                    else:
                        reply = (
                            f"抱歉，我刚才没正确处理。你说「{req.message[:40]}」，"
                            "能再说一次吗？我一定直接执行，不废话。"
                        )

                yield f"data: {json.dumps({'type': 'token', 'content': reply})}\n\n"
                tools = result.tools_called
                if tools > 0:
                    yield f"data: {json.dumps({'type': 'meta', 'content': f'🛠️ {tools} 次工具调用'})}\n\n"
                if result.iterations > 0:
                    yield f"data: {json.dumps({'type': 'meta', 'content': f'🔄 {result.iterations} 轮迭代'})}\n\n"
                _record_api_call(success=True)
                yield f"data: {json.dumps({'type': 'done', 'content': '', 'tools_called': tools})}\n\n"
        except Exception as e:
            _record_api_call(success=False, error_msg=str(e))
            # ⭐ 异常时也尝试降级到 Agent 路径
            logger.warning("🛑 chat_stream error — trying agent fallback: %s", e)
            # 🔥 v0.17: Send narration before agent fallback
            yield f"data: {json.dumps({'type': 'narration', 'content': '处理中…'})}\n\n"
            try:
                # ⭐ 检查 agent 是否正忙
                try:
                    await asyncio.wait_for(_agent_lock.acquire(), timeout=0.01)
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'error', 'content': '上一个请求还在处理中，请稍等几秒再试'})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"
                    return

                try:
                    _agent_busy = True
                    # 🔥 v0.23: TurnEngine fallback
                    _, llm_messages = _build_context(
                        user_message=req.message, agent_id=req.agent_id,
                        session_id=req.session_id, conversation_history=req.messages,
                    )
                    _turn_task = asyncio.create_task(
                        turn_engine.turn(messages=llm_messages, user_message=req.message)
                    )
                    _ka_count = 0; _max_ka = 8
                    while not _turn_task.done() and _ka_count < _max_ka:
                        _done, _ = await asyncio.wait([_turn_task], timeout=15.0)
                        if not _turn_task.done():
                            _ka_count += 1
                            if _ka_count >= _max_ka:
                                turn_engine.cancel()
                                logger.warning("⏰ TurnEngine fallback 超时(120s)")
                                break
                            yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
                    result = _turn_task.result() if _turn_task.done() else TurnResult(reply="")
                finally:
                    _agent_busy = False
                    _agent_lock.release()

                reply = result.reply or "抱歉，我遇到了一个暂时的问题，请再试一次。"
                yield f"data: {json.dumps({'type': 'token', 'content': reply})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"
            except Exception as e2:
                logger.exception("Agent fallback also failed: %s", e2)
                yield f"data: {json.dumps({'type': 'error', 'content': str(e2)[:200]})}\n\n"
                # 🔥 确保 done 永远发出（避免前端无限等待）
                yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"

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
    """Chat with memory-augmented Agent.

    # 对搜索/查新闻类简单请求，也用简单路径（省 token 又快）。
    # 复杂任务（写文件/分析/执行）走 agent 路径。
    """
    msg = req.message.strip()
    _ACTION_WORDS = [
        "爬", "下载", "读取", "写入", "编辑", "创建", "生成",
        "删除", "调用", "执行", "运行", "安装",
        "搜索", "查找", "查询", "百度",
        "看看", "打开", "访问",
        "记住", "记住我", "记一下", "记好", "帮我记住", "请记住",
        # ⭐ 分析类关键词（v0.22: 触发真实工具执行）
        "分析", "审计", "检查", "诊断", "评估", "排查", "检测",
        "对比", "比较", "测试", "验证", "调试", "排查",
        # 删/改/写/查/试/跑 → 下面用正则精确匹配
    ]
    # ⭐ 精确匹配：单字 action 通过正则确保是独立意图
    _single_char_actions = ("删", "改", "写", "查", "试", "跑")
    _has_action = any(v in msg.lower() for v in _ACTION_WORDS)
    _has_action = _has_action or bool(re.search(
        r'(?:^|[\s，。！？；：、])[' + ''.join(_single_char_actions) + r'](?:$|[\s，。！？；：、])',
        msg
    ))
    _IS_CONT = msg in ("继续", "继续！", "继续执行") or msg.startswith("继续") or msg in ("next", "continue", "go on")
    # ⭐ 含 URL 必须走 agent 路径（需要 web_fetch 工具）
    _has_url = "http://" in msg or "https://" in msg
    is_simple = (len(msg) < 120) and not _has_action and not _IS_CONT and not _has_url

    # 🐛 v0.30: 遗忘指令必须在存储之前处理！
    #   旧顺序：先 _maybe_store_memory 再 forget → "请忘记X"本身先被提取入库
    if cogni and len(msg) > 4 and _is_forget_intent(msg):
        _result = cogni.forget(msg, req.agent_id)
        _msg = _result.get("message", "")
        if _result.get("forgotten", 0) > 0:
            logger.info("🗑️ Forget processed: %s", _msg)
        return {"agent_id": req.agent_id, "reply": f"好的，{_msg}。还有其他需要吗？", "memories_used": 0, "tools_called": 0, "iterations": 0}

    # 🔥 v0.24: 在路由决策前就存记忆（不受路径限制）
    if cogni and len(msg) > 4:
        _maybe_store_memory(cogni, req, "chat_pre")

    # 🆕 v0.26: 缓存 _build_context 结果，避免简单→Agent降级时重复调用
    _ctx_cache = None
    # 冻结快照 — 有快照时跳过 recall，直接传 frozen_system
    _frozen_sys = cogni.get_snapshot(req.agent_id) if cogni else None
    if is_simple:
        # ⭐ 简单路径（v0.23: 可调只读工具，不进 Agent 循环）
        system, llm_messages = _build_context(
            user_message=msg, agent_id=req.agent_id,
            session_id=req.session_id, conversation_history=req.messages,
            frozen_system=_frozen_sys,
        )
        _ctx_cache = (system, llm_messages)
        # 首条消息后冻结 system prompt（导航 recall → 全部事实）
        # ⭐ freeze_snapshot 内部以空 query 走 navigation recall，
        #   确保所有事实（含跨会话的observation）都被包含在快照中
        if cogni and not _frozen_sys:
            logger.info("📸 Snapshot frozen for '%s'", req.agent_id)
            cogni.freeze_snapshot(agent_id=req.agent_id, session_id=req.session_id)
            _frozen_sys = True
        _simple_tools_called = 0  # 🔥 v0.23: 简单路径工具计数
        try:
            _t0 = time.time()

            # 🔥 v0.23: 简单路径 + 只读工具（read_file/web_search/memory_recall等）
            # 第一步：用 chat_completion 看 LLM 需不需要调工具
            # 不需要 → 直接返回文本（0 额外开销）
            # 需要   → 执行工具 → 再调一次合成回复（共2次LLM调用）
            readonly_tools = _get_readonly_tool_schemas(tool_registry, has_snapshot=bool(_frozen_sys))
            _resp = llm.chat_completion(
                messages=llm_messages,
                tools=readonly_tools,
                temperature=0.5,
            )
            _msg = _resp.choices[0].message

            if _msg.tool_calls:
                # ── 调了只读工具 → 执行 → 合成回复 ──
                logger.info("📊 POST /chat 简单路径调用了 %d 个只读工具", len(_msg.tool_calls))

                # 追加 assistant tool_calls 消息
                llm_messages.append({
                    "role": "assistant",
                    "content": _msg.content or "",
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in _msg.tool_calls
                    ],
                })

                # 执行每个只读工具
                for tc in _msg.tool_calls:
                    try:
                        _args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        _args = {}
                    # 🆕 v0.25: 传递 AgentContext 给 memory tools
                    _tool_ctx = None
                    if cogni:
                        from memory_agent.agent import AgentContext
                        _tool_ctx = AgentContext(
                            agent_id=req.agent_id,
                            cogni=cogni,
                        )
                    _result = tool_registry.execute(tc.id, tc.function.name, _args, _tool_ctx)
                    _simple_tools_called += 1
                    llm_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(_result, ensure_ascii=False, default=str)[:2000],
                    })

                # 第二次调用：工具结果 → 合成最终回复
                full_text = llm.chat(
                    messages=llm_messages, system_prompt=None,
                    temperature=0.5, max_tokens=4096,
                    enable_thinking=True,
                )
                logger.info("📊 POST /chat 简单路径(工具) %.1fs  %d工具",
                            time.time() - _t0, len(_msg.tool_calls))
            else:
                # ── 没调工具 → 直接返回文本 ──
                full_text = _msg.content or ""
                logger.info("📊 POST /chat 简单路径(直接) %.1fs", time.time() - _t0)

        except Exception as e:
            logger.warning("简单路径 LLM 调用失败: %s", e)
            full_text = ""
        if full_text.strip():
            reply = full_text.replace("【完成】", "").strip() or full_text.strip()
            reply = _clean_tool_call_xml(reply)
            return {"agent_id": req.agent_id, "reply": reply, "memories_used": 0, "tools_called": _simple_tools_called, "iterations": 1 if _simple_tools_called else 0, "tool_sequence": []}
        # 降级到 agent
        logger.warning("simple path empty — falling back to agent")

    if turn_engine:
        # ⭐ 防止并发死锁
        try:
            await asyncio.wait_for(_agent_lock.acquire(), timeout=0.01)
        except asyncio.TimeoutError:
            return {"agent_id": req.agent_id, "reply": "上一个请求还在处理中，请稍等几秒再试。", "tools_called": 0, "iterations": 0}

        try:
            _agent_busy = True
            if _ctx_cache:
                system, llm_messages = _ctx_cache
            else:
                system, llm_messages = _build_context(
                    user_message=msg, agent_id=req.agent_id,
                    session_id=req.session_id, conversation_history=req.messages,
                    frozen_system=_frozen_sys,
                )
                if cogni and not _frozen_sys:
                    cogni.freeze_snapshot(agent_id=req.agent_id, session_id=req.session_id)
                    _frozen_sys = True
            turn_engine._agent_id = req.agent_id  # 🆕 v0.25: 设置正确的 agent_id
            result = await turn_engine.turn(
                messages=llm_messages,
                user_message=msg,
                temperature=0.5,
            )
        finally:
            _agent_busy = False
            _agent_lock.release()
        reply = _clean_tool_call_xml(result.reply)
        if not reply.strip():
            tools = result.tools_called
            if tools > 0:
                reply = f"已执行 {tools} 次操作。还要帮你做点别的吗？"
            else:
                reply = "抱歉，我没能处理好，请再试一次。"
        return {
            "agent_id": req.agent_id,
            "reply": reply,
            "tools_called": result.tools_called,
            "iterations": result.iterations,
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

    # 4. If long_context provided, append it as additional context
    use_model = None  # ⭐ 使用 LLM 默认模型（DeepSeek 没有 qwen-max-longcontext）
    if req.long_context:
        user_content = (
            f"{req.message}\n\n"
            f"## Additional Context Document\n{req.long_context}"
        )
        logger.info("Using long-context mode (appending document to message)")
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

    # ⭐ 如果 store 返回空，fallback 到 CogniMem 跨 agent 搜索
    if (not prefs or not prefs.get("preferences")) and cogni:
        try:
            _db = cogni.fact_network.db
            if _db:
                with _db._plain_cursor_ctx() as cur:
                    cur.execute("SELECT * FROM facts WHERE fact_type='preference' ORDER BY confidence DESC LIMIT 10")
                    rows = cur.fetchall()
                    all_prefs = []
                    for row in rows:
                        cols = [desc[0] for desc in cur.description]
                        d = dict(zip(cols, row))
                        f = _db._dict_to_fact(d)
                        fd = f.to_dict()
                        all_prefs.append({
                            "content": f"{fd.get('subject','')} {fd.get('predicate','')} {fd.get('object','')}",
                            "confidence": fd.get("confidence", 0),
                            "fact_type": "preference",
                            "source": fd.get("source_session", ""),
                            "agent_id": fd.get("agent_id", ""),
                        })
                    if all_prefs:
                        return {"agent_id": agent_id, "preferences": {"preferences": all_prefs}}
        except Exception as e:
            logger.debug("Preferences fallback failed: %s", e)

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


@app.get("/agents", tags=["📋 数据查询"])
async def list_agents():
    """列出所有有数据的 Agent（含事实数量）"""
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


@app.get("/capabilities", tags=["📋 数据查询"], summary="列出所有能力目录及其可用状态")
async def list_capabilities(agent_id: str = "default"):
    """列出所有能力目录及当前 agent 的可用状态"""
    ctx = None
    if cogni:
        ctx = AgentContext(agent_id=agent_id, cogni=cogni)
    result = []
    for cap in CATALOG.values():
        result.append({
            "id": cap.id,
            "name": cap.name,
            "description": cap.description,
            "tool_count": len(cap.tool_names),
            "tools": list(cap.tool_names),
            "requires": list(cap.requires),
            "available": cap.available(ctx) if ctx else False,
        })
    return {"capabilities": result}


# ═══════════════════════════════════════════════
#  记忆管理 API（Dashboard 使用）
# ═══════════════════════════════════════════════

@app.get("/memories", tags=["📋 数据查询"], summary="查看 Agent 的所有记忆（分页）")
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


@app.delete("/memories/{fact_id}", tags=["🔴 数据管理"], summary="删除单条记忆（按 fact_id）")
async def delete_memory(fact_id: str, agent_id: str = "default"):
    """删除某条具体记忆（事实），需要知道 fact_id（从 /memories 或 /agent 获取）"""
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
    """触发记忆归纳整合（抽象化 + 衰减 + 去重 + 矛盾自动解析）

    🔥 v0.21: 整合 FactVerifier 子 Agent，自动解析矛盾事实。
    """
    if cogni is None:
        return {"agent_id": agent_id, "result": {}, "message": "CogniMem 未初始化"}

    # 1. 基础整合（含矛盾扫描）
    result = cogni.consolidate(agent_id)
    contradictions_found = result.get("contradictions", 0)

    # 2. 子 Agent 事实验证（仅当有矛盾且有 LLM 可用时）
    verifier_results = []
    if contradictions_found > 0 and llm is not None:
        try:
            verifier = FactVerifier(llm)
            all_facts = cogni.fact_network._get_agent_facts(agent_id)
            contradictions = cogni.fact_network.get_contradictions(agent_id)

            # 收集矛盾对（最多 5 对）
            pairs = []
            for c in contradictions[:5]:
                fa = next((f for f in all_facts if f.fact_id == c.fact_a_id), None)
                fb = next((f for f in all_facts if f.fact_id == c.fact_b_id), None)
                if fa and fb:
                    # 🐛 v0.30: 保存 FactTriple 对象本身，与 verdicts 严格对齐
                    pairs.append((fa, fb))

            if pairs:
                # 🔥 v0.21.1: batch_verify——1次LLM调用处理所有矛盾
                verdicts = verifier.batch_verify(
                    [(fa.to_dict(), fb.to_dict()) for fa, fb in pairs],
                    agent_id=agent_id,
                )
                fn = cogni.fact_network
                resolved = 0

                # 🐛 v0.30: zip 对齐 pairs（旧代码 zip 未过滤的 contradictions → 错配）
                #   _update_confidence 签名 (FactTriple, delta, reason)，delta 是增量
                for verdict, (fa, fb) in zip(verdicts, pairs):
                    if verdict.error:
                        logger.warning("矛盾解析失败: %s", verdict.error)
                        continue
                    if verdict.winner_id:
                        resolved += 1
                        if verdict.winner_id == fa.fact_id:
                            fn._update_confidence(fa, 0.15, "consolidate_verdict")
                            fn._update_confidence(fb, -0.10, "consolidate_verdict")
                        else:
                            fn._update_confidence(fb, 0.15, "consolidate_verdict")
                            fn._update_confidence(fa, -0.10, "consolidate_verdict")
                        # 🐛 v0.30: 置信度改动必须落库（只改内存缓存 → 重启后还原）
                        for _f in (fa, fb):
                            try:
                                fn.db.update_fact(_f)
                            except Exception:
                                pass
                    verifier_results.append({
                        "fact_a": f"{fa.subject} {fa.predicate} {fa.object}",
                        "fact_b": f"{fb.subject} {fb.predicate} {fb.object}",
                        "winner": verdict.winner_text or "不确定",
                        "confidence": verdict.confidence,
                        "reasoning": verdict.reasoning[:100],
                        "needs_user": verdict.needs_user_input,
                    })

                result["verifier_resolved"] = resolved
                result["verifier_results"] = verifier_results
                logger.info("🔍 子Agent批量矛盾解析: 解决 %d/%d 对 (1次LLM调用)",
                            resolved, len(pairs))

        except Exception as e:
            logger.warning("FactVerifier 矛盾解析失败: %s", e)
            result["verifier_error"] = str(e)[:100]

    return {
        "agent_id": agent_id,
        "result": result,
        "message": f"合并 {result.get('merged',0)} 条，"
                   f"抽象化 {result.get('abstracted',0)} 组，"
                   f"衰减 {result.get('decayed',0)} 条，"
                   f"矛盾解析 {len(verifier_results)} 对",
    }


@app.delete("/clear", tags=["🔴 数据管理"], summary="清空指定 Agent 全部记忆数据（不可恢复）")
async def clear_memories(agent_id: str = "default"):
    """删除某个 Agent 的全部数据（事实/三元组/偏好等），不可恢复！

    # 用法：
      1. 先调 GET /agents 查看所有 Agent ID
      2. 把要清除的 agent_id 填进来
      3. 执行后该 Agent 所有记忆将被永久删除

    # 参数:
      agent_id: 要清除的 Agent ID（默认 "default"）
    """
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
    if turn_engine is None:
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
