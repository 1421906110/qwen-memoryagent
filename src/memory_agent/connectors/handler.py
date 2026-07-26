"""
Webhook 连接器 — 冷启动消息处理器

🔥 相对优化（vs OpenWorker 25+常驻进程）：
  - 无常驻进程：外部服务通过 HTTP POST /webhook/{platform} 触发
  - 用完即止：Agent 处理完回复，没有常驻开销
  - 按需处理：消息入队列，Agent 异步消化

支持的 platform:
  - slack:  Slack Events API
  - github: GitHub Webhooks
  - telegram: Telegram Bot Webhook（需设置 webhook URL）
  - generic: 通用 JSON 格式

用法（外部服务 -> CogniMem）：
    POST /webhook/slack
    {"event": {"type": "message", "text": "你好", ...}}

    POST /webhook/github
    {"action": "opened", "issue": {"title": "...", ...}}
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger("agent.connectors")

router = APIRouter(prefix="/webhook")


# ── 消息队列（内存，Agent 异步消费） ──
_message_queue: list[dict] = []
_MAX_QUEUE = 100


# ── Slack 专用处理器 ──

@router.post("/slack")
async def slack_webhook(request: Request):
    """Slack Events API 回调

    处理: url_verification (challenge), event_callback (消息)
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Slack URL 验证
    if body.get("type") == "url_verification":
        challenge = body.get("challenge", "")
        logger.info("🔌 Slack URL verified")
        return {"challenge": challenge}

    # 事件回调
    event = body.get("event", {})
    if event.get("type") == "message" and event.get("text"):
        entry = {
            "id": f"slack_{event.get('ts', int(time.time()))}",
            "platform": "slack",
            "payload": {
                "channel": event.get("channel"),
                "user": event.get("user"),
                "text": event.get("text"),
                "thread_ts": event.get("thread_ts"),
            },
            "received_at": time.time(),
        }
        _message_queue.append(entry)

        logger.info("📬 Slack msg from %s: %.50s",
                    event.get("user", "?"), event.get("text", ""))

    return {"status": "ok"}


# ── 队列管理 ──

@router.get("/queue")
async def queue_status():
    """查看消息队列状态"""
    return {
        "size": len(_message_queue),
        "max": _MAX_QUEUE,
        "messages": [
            {
                "id": m["id"],
                "platform": m["platform"],
                "received_at": m["received_at"],
                "age_seconds": int(time.time() - m["received_at"]),
            }
            for m in _message_queue[-10:]
        ],
    }


@router.get("/queue/{msg_id}")
async def get_message(msg_id: str):
    """获取单条消息详情"""
    for m in _message_queue:
        if m["id"] == msg_id:
            return m
    raise HTTPException(status_code=404, detail="Message not found")


def pop_message() -> dict | None:
    """消费一条消息（Agent 轮询用）"""
    if _message_queue:
        return _message_queue.pop(0)
    return None


# ── 通用处理器（必须放在最后，避免抢特定路由） ──

@router.post("/{platform}")
async def webhook(platform: str, request: Request):
    """通用 Webhook 入口

    外部服务通过 HTTP POST 发送事件，
    消息入队列，Agent 按需异步处理。
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # 入队列
    entry = {
        "id": f"wh_{int(time.time() * 1000)}_{len(_message_queue)}",
        "platform": platform,
        "payload": body,
        "received_at": time.time(),
        "headers": dict(request.headers),
    }
    _message_queue.append(entry)

    # 裁剪队列
    while len(_message_queue) > _MAX_QUEUE:
        _message_queue.pop(0)

    logger.info("📬 Webhook %s: queued (queue=%d)", platform, len(_message_queue))

    return {
        "status": "queued",
        "id": entry["id"],
        "queue_size": len(_message_queue),
    }
