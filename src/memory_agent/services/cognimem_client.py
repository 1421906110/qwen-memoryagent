"""
CogniMem 客户端适配器 — 让 MemoryAgent 调用 CogniMem API
"""

import logging
import httpx
from typing import Any

logger = logging.getLogger(__name__)

API_BASE = "http://localhost:8001"


class CogniMemClient:
    """封装 CogniMem API 调用"""

    def __init__(self, base_url: str = API_BASE):
        self.base_url = base_url.rstrip("/")

    def remember(self, text: str, agent_id: str = "default",
                 source: str = "") -> dict:
        """记住信息 → CogniMem /remember"""
        try:
            r = httpx.post(f"{self.base_url}/remember", json={
                "text": text,
                "agent_id": agent_id,
                "source": source,
            }, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error("CogniMem remember failed: %s", e)
            return {"status": "error", "facts_added": 0}

    def recall(self, query: str, agent_id: str = "default",
               top_k: int = 10) -> dict:
        """召回 → CogniMem /recall"""
        try:
            r = httpx.post(f"{self.base_url}/recall", json={
                "query": query,
                "agent_id": agent_id,
                "top_k": top_k,
            }, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error("CogniMem recall failed: %s", e)
            return {"facts": [], "count": 0}

    def ask(self, query: str, agent_id: str = "default") -> dict:
        """问答式召回 → CogniMem /ask"""
        try:
            r = httpx.post(f"{self.base_url}/ask", json={
                "query": query,
                "agent_id": agent_id,
            }, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error("CogniMem ask failed: %s", e)
            return {"relevant_memories": []}

    def get_status(self, agent_id: str = "default") -> dict:
        """状态 → CogniMem /stats"""
        try:
            r = httpx.get(f"{self.base_url}/stats",
                          params={"agent_id": agent_id}, timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error("CogniMem stats failed: %s", e)
            return {"total_facts": 0, "core_beliefs": 0}

    def consolidate(self, agent_id: str = "default") -> dict:
        """触发记忆整合 → CogniMem /consolidate"""
        try:
            r = httpx.post(f"{self.base_url}/consolidate",
                          params={"agent_id": agent_id}, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error("CogniMem consolidate failed: %s", e)
            return {"status": "error"}

    @property
    def is_connected(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/", timeout=3)
            return r.status_code == 200
        except Exception:
            return False
