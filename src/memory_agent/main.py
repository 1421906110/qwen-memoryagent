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
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from memory_agent.models import MemoryRecord
from memory_agent.services.llm_client import LLMClient
from memory_agent.services.memory_service import MemoryService
from memory_agent.storage import SQLiteStore

# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("memory_agent")

# ---------------------------------------------------------------------------
#  Globals (initialised in lifespan)
# ---------------------------------------------------------------------------

store: SQLiteStore | None = None
memory_service: MemoryService | None = None
llm: LLMClient | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global store, memory_service, llm

    db_path = os.getenv("MEMORY_DB_PATH", "~/.qwen-memory/memory.db")
    store = SQLiteStore(db_path)

    # LLM — may be None if no API key configured yet
    api_key = os.getenv("QWEN_API_KEY", "")
    if api_key:
        llm = LLMClient(api_key=api_key)

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

# ---------------------------------------------------------------------------
#  Web UI — static HTML pages
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
TEMPLATES_DIR = HERE / "templates"
STATIC_DIR = HERE / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _read_html(name: str) -> str:
    path = TEMPLATES_DIR / name
    if not path.exists():
        raise HTTPException(404, f"Page {name} not found")
    return path.read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    return _read_html("chat.html")


@app.get("/chat", response_class=HTMLResponse, include_in_schema=False)
async def chat_page():
    return _read_html("chat.html")


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard_page():
    return _read_html("dashboard.html")


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
    """Retrieve relevant memories for an agent."""
    assert memory_service is not None

    # Generate query embedding if LLM available
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


@app.post("/chat")
async def chat(req: ChatRequest):
    """Chat with memory-augmented Qwen.

    Automatically:
      1. Recalls relevant memories
      2. Injects them as context
      3. Stores the exchange as new memories
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

    # 3. Generate answer with memory context
    memories_dict = [
        {
            "content": m.content,
            "memory_type": m.memory_type,
            "confidence": m.confidence,
            "created_at": m.created_at,
        }
        for m in recall_result.memories
    ]
    answer = llm.answer_with_memories(
        query=req.message,
        memories=memories_dict,
        preferences=prefs.get("preferences"),
    )

    # 4. Extract and store new memories from the exchange
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
    }


@app.get("/decay-trace/{memory_id}")
async def decay_trace(memory_id: str, days: int = 30, points: int = 50):
    """Visualize confidence decay curve for a specific memory."""
    assert memory_service is not None
    trace = memory_service.compute_decay_trace(memory_id, days=days, points=points)
    if "error" in trace:
        raise HTTPException(status_code=404, detail=trace["error"])
    return trace


@app.get("/decay-analysis")
async def decay_analysis(agent_id: str, min_confidence: float = 0.0):
    """Analyze decay state across all memories."""
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
    assert memory_service is not None
    stats = memory_service.merge_all(
        agent_id=agent_id,
        sim_threshold=sim_threshold,
        min_cluster_size=min_cluster_size,
        limit=limit,
    )
    return stats
