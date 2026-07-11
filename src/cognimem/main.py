"""
CogniMem API — FastAPI 服务层

运行模式:
    # 纯内存模式 (默认)
    cd ~/projects/qwen-memoryagent
    source .venv/bin/activate
    set -a; source .env; set +a
    uvicorn cognimem.main:app --reload --port 8001

    # 持久化模式
    COGNIMEM_DB=postgresql://localhost/cognimem uvicorn cognimem.main:app --reload --port 8001

    # LLM 提取 + 持久化（推荐）
    COGNIMEM_DB=postgresql://localhost/cognimem COGNIMEM_LLM=1 uvicorn cognimem.main:app --reload --port 8001

    # 或用 .env 文件自动加载
"""

import os
import logging
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel
from typing import Any

from cognimem.core.brain import CogniMem
from cognimem.core.db import DatabaseAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 加载 .env（优先项目根目录，兼容旧路径）──
env_paths = [
    Path(__file__).resolve().parents[2] / ".env",  # 项目根目录
    Path(__file__).parent / ".env",                 # 旧兼容
]
for env_path in env_paths:
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
        logger.info("📄 .env loaded: %s", env_path)
        break

# 自定义 Swagger UI 中文化
from fastapi.openapi.docs import get_swagger_ui_html

app = FastAPI(title="CogniMem 认知记忆系统", docs_url=None,
              description="人工智能体认知记忆引擎 — 三元组事实网络 + LLM 提取 + 置信度系统 + pgvector 向量搜索",
              version="0.2.0")

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    """CogniMem API 文档"""
    from fastapi.openapi.docs import get_swagger_ui_html
    from fastapi.responses import HTMLResponse
    html = get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="CogniMem API 文档",
    )
    js = "<script>new MutationObserver(function(){var v=document.querySelector('.info .version');if(v){v.remove();this.disconnect()}}).observe(document.body,{childList:true,subtree:true})</script>"
    body = html.body.decode() if isinstance(html.body, bytes) else str(html.body)
    body = body.replace("</body>", f"{js}</body>")
    return HTMLResponse(content=body)

# ── 数据库模式 ──
db_dsn = os.environ.get("COGNIMEM_DB", "")
if db_dsn:
    db = DatabaseAdapter(dsn=db_dsn)
    db.connect()
    logger.info("🚀 DB mode: connected to %s", db_dsn)
else:
    db = None
    logger.info("🧠 Memory-only mode: no database")

# ── LLM 模式 ──
use_llm = os.environ.get("COGNIMEM_LLM", "") in ("1", "true", "yes")
brain = CogniMem(db_adapter=db, use_llm=use_llm)


# ── Models ──

class RememberRequest(BaseModel):
    text: str
    source: str = ""
    agent_id: str = "default"


class RecallRequest(BaseModel):
    query: str
    agent_id: str = "default"
    top_k: int = 10


class ConfirmRequest(BaseModel):
    fact_id: str
    agent_id: str = "default"


# ── Endpoints ──

@app.get("/")
async def root():
    return {"service": "CogniMem", "version": "0.1.0", "status": "alive"}


@app.post("/remember")
async def api_remember(req: RememberRequest):
    """记住一条信息

将自然语言通过 LLM 提取为三元组，存入事实网络，自动检测矛盾"""
    result = brain.remember(req.text, req.source, req.agent_id)
    return result


@app.post("/recall")
async def api_recall(req: RecallRequest):
    """召回记忆

三级召回：L0 Cache → L1 精确匹配 → L2 语义扩展 → L3 向量搜索"""
    result = brain.recall(req.query, req.agent_id, req.top_k)
    # 序列化
    return {
        "query": req.query,
        "facts": [f.to_dict() for f in result["facts"]],
        "count": result["count"],
        "contradictions_warning": result.get("has_contradictions", False),
    }


@app.post("/ask")
async def api_ask(req: RecallRequest):
    """问答式召回（Agent 友好）

返回相关记忆、核心信念、不确定项、矛盾提醒"""
    return brain.ask(req.query, req.agent_id)


@app.post("/confirm")
async def api_confirm(req: ConfirmRequest):
    """确认事实 → 提升置信度（冷启动加速 ×2）"""
    return brain.confirm(req.fact_id, req.agent_id)


@app.post("/challenge")
async def api_challenge(req: ConfirmRequest):
    """质疑事实 → 降低置信度"""
    return brain.challenge(req.fact_id, req.agent_id)


@app.post("/consolidate")
async def api_consolidate(agent_id: str = "default"):
    """触发记忆整合 → 重复合并 + 模式提升 + 主动遗忘"""
    return brain.consolidate(agent_id)


@app.get("/stats")
async def api_stats(agent_id: str = "default"):
    """获取统计 → 总事实数 / 核心信念 / 按类型分布 / 路由命中率"""
    return brain.get_stats(agent_id)



@app.get("/health")
async def api_health():
    """引擎健康检测"""
    import time
    checks = {}
    issues = []
    score = 100

    # DB 状态
    if db:
        try:
            conn = db._get_conn()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            db._put_conn(conn)
            checks["db"] = "✅"
            # 表是否存在
            with db._plain_cursor_ctx() as cur:
                cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'facts')")
                checks["has_tables"] = "✅" if cur.fetchone()[0] else "⚠️"
        except Exception as e:
            checks["db"] = f"⚠️ {e}"
            issues.append({"type": "db", "detail": str(e)[:60], "severity": "high"})
            score -= 20
    else:
        checks["db"] = "🧠 内存模式（无数据库）"

    # Brain 状态
    if brain:
        try:
            stats = brain.get_stats("default")
            checks["brain"] = {
                "total_facts": stats.get("total_facts", 0),
                "core_beliefs": stats.get("core_beliefs", 0),
                "contradictions": stats.get("contradictions", 0),
            }
        except Exception as e:
            checks["brain"] = f"⚠️ {e}"
            issues.append({"type": "brain", "detail": str(e)[:60], "severity": "high"})
            score -= 20
    else:
        checks["brain"] = "❌ 未初始化"
        score -= 50

    score = max(0, min(100, score))
    level = "healthy" if score >= 80 else "warning" if score >= 50 else "critical"
    return {
        "service": "CogniMem",
        "version": "0.2.0",
        "score": score,
        "level": level,
        "checks": checks,
        "issues": issues,
        "timestamp": time.time(),
    }


@app.get("/versions/{fact_id}")
async def api_versions(fact_id: str):
    """获取事实的版本历史 → 追溯每次变更的原因和置信度变化"""
    # UUID 校验：非法格式返回 400
    try:
        uuid.UUID(fact_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail=f"非法 UUID 格式: {fact_id}")

    try:
        versions = brain.fact_network.get_versions(fact_id)
        return {"fact_id": fact_id, "versions": versions, "count": len(versions)}
    except Exception as e:
        logger.error("获取版本历史失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e)[:200])


class AnalyzeContradictionRequest(BaseModel):
    fact_id_a: str
    fact_id_b: str
    agent_id: str = "default"


@app.post("/resolve-contradiction")
async def api_resolve_contradiction(req: AnalyzeContradictionRequest):
    """用 LLM 分析两个事实之间的矛盾 → 返回 verdict / explanation / needs_confirmation"""
    result = brain.analyze_contradiction(
        req.fact_id_a, req.fact_id_b, req.agent_id
    )
    return result


@app.delete("/clear")
async def api_clear(agent_id: str = "default"):
    """清除某个 Agent 的所有记忆（含事实/矛盾/版本/缓存）"""
    if not db:
        brain.fact_network._lru_cache.clear()
        return {"agent_id": agent_id, "deleted": 0, "message": "内存模式，缓存已清"}

    facts = db.get_agent_facts(agent_id)
    count = len(facts)
    for f in facts:
        try:
            db.delete_fact(f.fact_id)
        except Exception as e:
            logger.warning("Failed to delete fact %s: %s", f.fact_id, e)
    brain.fact_network._lru_cache.clear()
    brain.fact_network._query_cache.clear()
    brain.fact_network._stats_cache.clear()
    return {"agent_id": agent_id, "deleted": count, "message": f"已清除 {count} 条记忆"}


# ── 启动验证 ──

@app.on_event("startup")
async def startup():
    """启动时运行验证（首次完整测试，后续快速启动跳过 LLM）"""
    logger.info("=" * 50)
    logger.info("CogniMem 启动验证")
    logger.info("=" * 50)

    startup_done_file = Path(__file__).parent / ".cognimem_startup_done"
    skip_test = os.environ.get("COGNIMEM_SKIP_STARTUP_TEST", "") in ("1", "true", "yes")

    if skip_test or startup_done_file.exists():
        # ⚡ 快速启动：只检查 DB 连通性，不调 LLM
        logger.info("⚡ 快速启动模式：跳过 LLM 验证测试")
        if db:
            try:
                conn = db._get_conn()
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                db._put_conn(conn)
                logger.info("✅ DB 连通性正常")
            except Exception as e:
                logger.warning(f"⚠️ DB 连接异常: {e}")
        logger.info("=" * 50)
        logger.info("CogniMem 启动完成（快速模式）")
        logger.info("=" * 50)
        return

    # 首次启动：完整验证（含 LLM 提取 + 矛盾检测 + 召回）
    logger.info("🔧 首次启动：运行完整验证...")

    r1 = brain.remember("我喜欢喝冰美式", "test_startup", "_startup")
    logger.info(f"✅ remember: {r1['status']} ({len(r1['facts'])} facts)")

    r2 = brain.remember("用户不喜欢喝冰美式", "test_startup", "_startup")
    logger.info(f"✅ contradiction check: {r2.get('contradictions_detected', 0)} detected")

    r3 = brain.recall("咖啡", "_startup")
    logger.info(f"✅ recall '咖啡': {r3['count']} facts found")

    stats = brain.get_stats("_startup")
    logger.info(f"✅ stats: {stats['total_facts']} facts, {stats['core_beliefs']} beliefs")

    # 清理测试数据
    if db:
        for f in db.get_agent_facts("_startup"):
            db.delete_fact(f.fact_id)
        brain.fact_network._lru_cache.clear()
        logger.info("✅ test data cleaned up")

    # 标记已完成首次验证，后续启动跳过 LLM
    startup_done_file.touch()
    logger.info("✅ 首次验证完成，已标记 .cognimem_startup_done")

    logger.info("=" * 50)
    logger.info("CogniMem 启动完成")
    logger.info("=" * 50)
