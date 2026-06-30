"""
MemoryAgent — Persistent AI memory layer for QwenCloud.

Core data models for memory entries, sessions, and agent state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryRecord:
    """A single memory entry with metadata."""

    id: str
    agent_id: str
    session_id: str
    content: str
    memory_type: str = "observation"  # observation | preference | fact | decision | goal
    confidence: float = 0.8
    tags: list[str] = field(default_factory=list)
    source: str = "conversation"
    created_at: float = 0.0  # unix ts
    updated_at: float = 0.0
    accessed_at: float = 0.0  # last recall time (for decay)
    access_count: int = 0
    superseded_by: str | None = None  # conflict resolution: id of newer memory
    embedding: list[float] | None = None

    def __post_init__(self):
        now = time.time()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now
        if not self.accessed_at:
            self.accessed_at = now


@dataclass
class SessionState:
    """Cross-session state for an agent."""

    agent_id: str
    session_id: str
    start_time: float
    end_time: float | None = None
    memory_count: int = 0
    active_preferences: dict[str, Any] = field(default_factory=dict)
    context_summary: str = ""


@dataclass
class RecallResult:
    memories: list[MemoryRecord]
    total_found: int
    query_time_ms: float
