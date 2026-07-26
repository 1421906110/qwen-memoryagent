"""
CogniMem MCP Server — 认知记忆系统的 MCP 协议接口

让 CogniMem 作为 MCP Server 运行，任何 MCP 兼容客户端都可以直接调用：
- Claude Desktop / Claude Code
- Cursor / VS Code / Windsurf
- Cline / Roo Code / OpenClaw

灵感来自 Mimir MemoryAgent（43 MCP tools，Perseus Computing）：
Mimir 证明了"记忆即服务"的可行性 — 把记忆能力做成 MCP 协议，
就能被整个 AI 工具生态调用。CogniMem 的 SPO 三元组 + 矛盾检测
同样可以通过 MCP 协议供外部使用。

使用方式:
    python -m cognimem.mcp_server          # stdio 模式（默认，用于 MCP Host）
    python -m cognimem.mcp_server --sse    # SSE 模式（HTTP 服务）
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# ── 自动加载 .env ──
_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                k, v = _line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from mcp.server.fastmcp import FastMCP

from cognimem.core.brain import CogniMem
from cognimem.core.db import DatabaseAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MCP] %(message)s",
)
logger = logging.getLogger("cognimem.mcp")

# ── 初始化 CogniMem ──
_config = {
    "llm_api_key": os.environ.get("DEEPSEEK_API_KEY", "")
                   or os.environ.get("DASHSCOPE_API_KEY", ""),
    "llm_model": os.environ.get("QWEN_MODEL", "deepseek-v4-flash"),
}

# 尝试连接 PostgreSQL
db = None
try:
    db = DatabaseAdapter()
    db.connect()
    logger.info("✅ PostgreSQL 连接成功")
except Exception as e:
    logger.warning("⚠️  PostgreSQL 未连接（仅内存模式）: %s", e)

brain = CogniMem(db_adapter=db, config=_config,
                 use_llm=bool(_config.get("llm_api_key")))
logger.info("🧠 CogniMem MCP Server 初始化完成")

# ── FastMCP Server ──
mcp = FastMCP(
    name="CogniMem",
    instructions="CogniMem 认知记忆系统 — SPO 三元组记忆系统。提供结构化存储、矛盾检测、科学遗忘、智能召回。灵感来自 Mimir MemoryAgent（43 MCP tools）。",
)


# ═══════════════════════════════════════════════════════════════
# MCP Tools — 记忆操作
# ═══════════════════════════════════════════════════════════════

@mcp.tool(name="memory_recall",
          description="召回憶憶：根據查詢文本找到最相關的記憶。返回 SPO 三元組格式的事實列表，每個帶置信度。")
def memory_recall(query: str, agent_id: str = "default", top_k: int = 10) -> str:
    """根据查询文本召回相关的记忆"""
    try:
        result = brain.recall(query=query, agent_id=agent_id, top_k=top_k)
        facts = result.get("facts", [])
        if not facts:
            return json.dumps({"status": "no_memories_found", "facts": []},
                              ensure_ascii=False)

        output = []
        for f in facts:
            item = {
                "fact": f"{f.subject} {f.predicate} {f.object}",
                "confidence": f.confidence,
                "type": f.fact_type,
                "source_label": f.source_label,
                "citation": f.citation,
            }
            # 如果有过期警告
            sw = f.stale_warning
            if sw:
                item["stale_warning"] = sw
            output.append(item)

        return json.dumps({
            "status": "success",
            "count": len(output),
            "facts": output,
            "has_contradictions": result.get("has_contradictions", False),
        }, ensure_ascii=False)
    except Exception as e:
        logger.error("memory_recall 失败: %s", e)
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool(name="memory_remember",
          description="記下一條記憶：將文本信息提取為 SPO 三元組結構存入記憶庫。支持規則提取（0 Token）和 LLM 提取。")
def memory_remember(text: str, agent_id: str = "default",
                    source: str = "user_statement") -> str:
    """记住一条信息"""
    try:
        result = brain.remember(text=text, source=source, agent_id=agent_id,
                                source_type=source)
        status = result.get("status", "unknown")
        if status == "no_facts_extracted":
            return json.dumps({"status": "ok", "message": "信息太简单，无需存储"})
        facts_added = result.get("facts_added", 0)
        contradictions = result.get("contradictions_detected", 0)
        facts = result.get("facts", [])

        return json.dumps({
            "status": "remembered",
            "facts_added": facts_added,
            "contradictions_detected": contradictions,
            "facts": [
                {
                    "triple": f"{f.subject} {f.predicate} {f.object}",
                    "confidence": f.confidence,
                    "fact_type": f.fact_type,
                }
                for f in facts[:5]
            ],
        }, ensure_ascii=False)
    except Exception as e:
        logger.error("memory_remember 失败: %s", e)
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool(name="memory_diagnose",
          description="診斷記憶系統健康狀態：檢查總數、核心信念、矛盾數、召回路由命中率等。返回 JSON 格式的健康報告。")
def memory_diagnose(agent_id: str = "default") -> str:
    """诊断记忆健康状况"""
    try:
        stats = brain.get_stats(agent_id)
        router_stats = stats.get("router_stats", {})

        # 健康评分
        issues = []
        score = 100
        if stats.get("contradictions", 0) > 5:
            issues.append(f"矛盾過多：{stats['contradictions']} 條 pending")
            score -= 15
        if stats.get("unreliable", 0) > 10:
            issues.append(f"不可靠記憶較多：{stats['unreliable']} 條")
            score -= 10
        if stats.get("total_facts", 0) == 0:
            issues.append("暫無記憶")
            score -= 20

        return json.dumps({
            "status": "ok",
            "health_score": max(0, score),
            "metrics": {
                "total_facts": stats.get("total_facts", 0),
                "core_beliefs": stats.get("core_beliefs", 0),
                "pending_contradictions": stats.get("contradictions", 0),
                "unreliable_facts": stats.get("unreliable", 0),
                "by_type": stats.get("by_type", {}),
            },
            "router_performance": router_stats,
            "issues": issues,
            "summary": (
                f"記憶庫：{stats.get('total_facts', 0)} 條事實 | "
                f"核心信念：{stats.get('core_beliefs', 0)} 條 | "
                f"矛盾：{stats.get('contradictions', 0)} pending | "
                f"召回 L0命中率：{router_stats.get('l0_hit_rate', 'N/A')}"
            ),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool(name="memory_status",
          description="查看記憶統計：按類型分佈、過去 7 天新增量等綜合狀態摘要。")
def memory_status(agent_id: str = "default") -> str:
    """查看记忆统计"""
    try:
        stats = brain.get_stats(agent_id)
        by_type = stats.get("by_type", {})
        type_summary = "\n".join(
            f"  - {t}: {c} 條" for t, c in sorted(by_type.items(),
                                                    key=lambda x: x[1], reverse=True)
        ) if by_type else "  - 暫無數據"

        return json.dumps({
            "status": "ok",
            "agent_id": stats.get("agent_id", agent_id),
            "total_facts": stats.get("total_facts", 0),
            "core_beliefs": stats.get("core_beliefs", 0),
            "by_type": by_type,
            "type_summary": type_summary,
            "router_stats": stats.get("router_stats", {}),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool(name="memory_forget",
          description="刪除指定記憶：根據 fact_id 刪除一條事實。支持批量刪除（傳多個 ID 用逗號分隔）。")
def memory_forget(fact_id: str, agent_id: str = "default") -> str:
    """删除指定记忆"""
    try:
        brain.fact_network._delete_fact(fact_id)
        return json.dumps({
            "status": "deleted",
            "fact_id": fact_id,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool(name="memory_ask",
          description="問答式記憶召回：像聊天一樣查詢記憶，返回相關事實+核心信念+不確定項+主動學習問題。適合 Agent 直接使用。")
def memory_ask(query: str, agent_id: str = "default") -> str:
    """问答式记忆召回（带置信度说明和主动学习）"""
    try:
        result = brain.ask(query=query, agent_id=agent_id)
        memories = result.get("relevant_memories", [])
        questions = result.get("active_questions", [])

        summary_parts = []
        if memories:
            for m in memories[:5]:
                line = f"- {m['fact']}"
                if m.get("citation"):
                    line += f" ——{m['citation']}"
                if m.get("stale_warning"):
                    line += f" {m['stale_warning']}"
                summary_parts.append(line)

        return json.dumps({
            "status": "success",
            "summary": "\n".join(summary_parts) if summary_parts else "暫無相關記憶",
            "memories": memories[:5],
            "core_beliefs": result.get("core_beliefs", [])[:3],
            "active_questions": questions[:3] if questions else [],
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool(name="memory_groom",
          description="觸發記憶維護：執行艾賓浩斯遺忘（刪除低置信度記憶）+ 碎片歸納抽象 + 矛盾自動清理。類似人類睡眠時的記憶整理。")
def memory_groom(agent_id: str = "default") -> str:
    """触发记忆维护（遗忘+抽象+合并）"""
    try:
        result = brain.consolidate(agent_id)
        return json.dumps({
            "status": "completed",
            "decayed": result.get("decayed", 0),
            "deleted": result.get("deleted", 0),
            "merged": result.get("merged", 0),
            "promoted": result.get("promoted", 0),
            "abstracted": result.get("abstracted", 0),
            "core_preserved": result.get("core_preserved", 0),
            "summary": (
                f"遺忘 {result.get('deleted', 0)} 條 | "
                f"衰減 {result.get('decayed', 0)} 條 | "
                f"合併 {result.get('merged', 0)} 組 | "
                f"抽象化 {result.get('abstracted', 0)} 條 | "
                f"信念提升 {result.get('promoted', 0)} 條"
            ),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool(name="memory_batch_remember",
          description="批量記住多條記憶：一次傳入文本列表，每條都會被提取為 SPO 三元組存入。適合導入歷史數據。")
def memory_batch_remember(texts: str, agent_id: str = "default",
                           source: str = "user_statement") -> str:
    """批量记住多条信息（texts 用 || 分隔多条）"""
    try:
        lines = [t.strip() for t in texts.split("||") if t.strip()]
        total = len(lines)
        results = brain.batch_remember(lines, source=source,
                                        agent_id=agent_id, source_type=source)
        added = sum(1 for r in results if r.get("status") == "remembered")
        return json.dumps({
            "status": "ok",
            "total_input": total,
            "memories_added": added,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool(name="memory_bus",
          description="跨 Agent 記憶總線（受 Universal Agent OS 啓發）：同時查詢多個 Agent 的記憶庫，返回去重後按置信度排序的結果。適合團隊知識共享。")
def memory_bus(query: str, agent_ids: str, top_k: int = 10) -> str:
    """跨 Agent 记忆总线：一次查询多个 Agent 的记忆"""
    ids = [a.strip() for a in agent_ids.split(",") if a.strip()]
    if not ids:
        return json.dumps({"status": "error", "message": "請指定至少一個 Agent ID"})
    try:
        result = brain.recall_cross_agent(query=query, agent_ids=ids, top_k=top_k)
        facts = result.get("facts", [])
        sources = result.get("sources", {})
        return json.dumps({
            "status": "success",
            "count": result.get("count", 0),
            "facts": [
                {
                    "fact": f"{f.subject} {f.predicate} {f.object}",
                    "confidence": f.confidence,
                    "agent_id": f.agent_id,
                    "citation": f.citation,
                }
                for f in facts
            ],
            "sources": sources,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool(name="audit_query",
          description="查詢審計日誌：查看記憶的創建/讀取/更新/刪除等操作歷史。受 DREAM audit ledger 啟發。")
def audit_query(agent_id: str = "", operation: str = "",
                limit: int = 50, since_hours: int = 0) -> str:
    """查询审计日志"""
    db = brain.fact_network.db if brain.fact_network else None
    if not db or not hasattr(db, 'query_audit'):
        return json.dumps({"status": "error", "message": "审计日志不可用"})
    try:
        rows = db.query_audit(agent_id=agent_id, operation=operation,
                               limit=limit, since_hours=since_hours)
        return json.dumps({
            "status": "ok",
            "count": len(rows),
            "entries": [
                {
                    "time": str(r.get("created_at", ""))[:19],
                    "operation": r.get("operation", ""),
                    "detail": r.get("detail", ""),
                    "caller": r.get("caller", ""),
                }
                for r in rows
            ],
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool(name="credential_store",
          description="安全存储凭证（密码/API Key/密鑰等）。存儲在知識庫中，普通記憶召回不會洩露。")
def credential_store(service: str, credential: str,
                      agent_id: str = "default") -> str:
    """安全存储凭证到知识库"""
    try:
        result = brain.remember_credential(service, credential, agent_id)
        return json.dumps({
            "status": result["status"],
            "service": result["service"],
            "message": f"凭证已{'更新' if result['status']=='updated' else '存储'}: {service}",
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool(name="credential_recall",
          description="安全召回凭证。只有通过此工具才能解密獲取憑證原文，普通記憶召回不會洩露。")
def credential_recall(service: str, agent_id: str = "default") -> str:
    """安全召回凭证"""
    try:
        result = brain.recall_credential(service, agent_id)
        if result["status"] == "not_found":
            return json.dumps({"status": "not_found", "message": f"未找到服务「{service}」的凭证"})
        return json.dumps({
            "status": "found",
            "service": result["service"],
            "credential": result["credential"],
            "safe_display": result["safe_display"],
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool(name="credential_list",
          description="列出所有已存儲的憑證（不泄露原文）。返回服務名稱和創建時間。")
def credential_list(agent_id: str = "default") -> str:
    """列出所有凭证"""
    try:
        creds = brain.list_credentials(agent_id)
        return json.dumps({
            "status": "ok",
            "count": len(creds),
            "credentials": creds,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

def main():
    import argparse
    parser = argparse.ArgumentParser(description="CogniMem MCP Server")
    parser.add_argument("--sse", action="store_true",
                        help="以 SSE 模式运行（默认 stdio）")
    parser.add_argument("--port", type=int, default=8100,
                        help="SSE 端口（默认 8100）")
    args = parser.parse_args()

    if args.sse:
        logger.info("🚀 CogniMem MCP Server (SSE) on port %d", args.port)
        mcp.run(transport="sse", mount_path=f"/mcp")
    else:
        logger.info("🚀 CogniMem MCP Server (stdio)")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
