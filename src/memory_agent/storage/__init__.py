"""
Storage abstraction for MemoryAgent.

Provides a pluggable storage backend interface and a built-in SQLite
implementation with FTS5 full-text search.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from memory_agent.models import MemoryRecord


def _compute_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    A = np.array(a, dtype=np.float32)
    B = np.array(b, dtype=np.float32)
    norm = np.linalg.norm(A) * np.linalg.norm(B)
    if norm == 0:
        return 0.0
    return float(np.dot(A, B) / norm)


class MemoryStore:
    """Pluggable memory store."""

    def store(self, memory: MemoryRecord) -> str:
        raise NotImplementedError

    def get(self, memory_id: str) -> MemoryRecord | None:
        raise NotImplementedError

    def search(
        self,
        agent_id: str,
        query_embedding: list[float] | None = None,
        memory_types: list[str] | None = None,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> list[MemoryRecord]:
        raise NotImplementedError

    def delete(self, memory_id: str) -> bool:
        raise NotImplementedError

    def count(self, agent_id: str) -> int:
        raise NotImplementedError

    def get_all_preferences(self, agent_id: str, limit: int = 50) -> list[MemoryRecord]:
        raise NotImplementedError

    def get_active_preferences(self, agent_id: str) -> dict[str, Any]:
        raise NotImplementedError


class SQLiteStore(MemoryStore):
    """SQLite-backed storage with JSON metadata and optional vector sim."""

    def __init__(self, db_path: str | Path = "~/.qwen-memory/memory.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL DEFAULT 'observation',
                confidence REAL NOT NULL DEFAULT 0.8,
                tags TEXT NOT NULL DEFAULT '[]',
                source TEXT NOT NULL DEFAULT 'conversation',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                accessed_at REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                superseded_by TEXT,
                embedding TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_id);
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);
            -- FTS5 for full-text search
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content, id UNINDEXED
            );
        """)
        self._conn.commit()

    def store(self, memory: MemoryRecord) -> str:
        if not memory.id:
            memory.id = f"mem_{uuid.uuid4().hex[:12]}"
        embedding_json = json.dumps(memory.embedding) if memory.embedding else None
        tags_json = json.dumps(memory.tags)
        self._conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, agent_id, session_id, content, memory_type, confidence,
                tags, source, created_at, updated_at, accessed_at,
                access_count, superseded_by, embedding)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                memory.id, memory.agent_id, memory.session_id,
                memory.content, memory.memory_type, memory.confidence,
                tags_json, memory.source, memory.created_at,
                memory.updated_at, memory.accessed_at,
                memory.access_count, memory.superseded_by, embedding_json,
            ),
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO memories_fts(id, content) VALUES (?, ?)",
            (memory.id, memory.content),
        )
        self._conn.commit()
        return memory.id

    def get(self, memory_id: str) -> MemoryRecord | None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return self._row_to_memory(row) if row else None

    def search(
        self,
        agent_id: str,
        query_embedding: list[float] | None = None,
        memory_types: list[str] | None = None,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> list[MemoryRecord]:
        sql = "SELECT * FROM memories WHERE agent_id = ? AND superseded_by IS NULL"
        params: list[Any] = [agent_id]

        if memory_types:
            placeholders = ",".join("?" * len(memory_types))
            sql += f" AND memory_type IN ({placeholders})"
            params.extend(memory_types)

        sql += " AND confidence >= ?"
        params.append(min_confidence)

        sql += " ORDER BY confidence DESC, created_at DESC LIMIT ?"
        params.append(limit * 3)  # fetch extra for re-ranking

        rows = self._conn.execute(sql, params).fetchall()
        memories = [self._row_to_memory(r) for r in rows if r]

        # Re-rank by cosine similarity if query embedding provided
        if query_embedding and memories:
            scored = []
            for m in memories:
                if m.embedding:
                    sim = _compute_similarity(query_embedding, m.embedding)
                else:
                    sim = 0.0
                # Blend similarity + confidence
                scored.append((sim * 0.7 + m.confidence * 0.3, m))
            scored.sort(key=lambda x: x[0], reverse=True)
            memories = [m for _, m in scored[:limit]]
        else:
            memories = memories[:limit]

        # Update accessed_at for recalled memories
        now = time.time()
        for m in memories:
            self._conn.execute(
                "UPDATE memories SET accessed_at = ?, access_count = access_count + 1 WHERE id = ?",
                (now, m.id),
            )
        self._conn.commit()

        return memories

    def search_fts(
        self, agent_id: str, query: str, limit: int = 10
    ) -> list[MemoryRecord]:
        """Full-text search via FTS5."""
        # Sanitize query: FTS5 doesn't like punctuation or operators
        sanitized = " ".join(
            w for w in query.split()
            if w.strip("?!,\"'.;:()[]{}")
        )
        if not sanitized.strip():
            return []
        try:
            sql = """
                SELECT m.* FROM memories m
                JOIN memories_fts fts ON m.id = fts.id
                WHERE m.agent_id = ? AND memories_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """
            rows = self._conn.execute(
                sql, (agent_id, sanitized, limit)
            ).fetchall()
            return [self._row_to_memory(r) for r in rows if r]
        except Exception:
            # Fallback: simple LIKE search
            like = f"%{query.lower()}%"
            sql = """
                SELECT * FROM memories
                WHERE agent_id = ? AND LOWER(content) LIKE ?
                LIMIT ?
            """
            rows = self._conn.execute(sql, (agent_id, like, limit)).fetchall()
            return [self._row_to_memory(r) for r in rows if r]

    def delete(self, memory_id: str) -> bool:
        c = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))
        self._conn.commit()
        return c.rowcount > 0

    def supersede(self, old_id: str, new_id: str):
        """Mark old memory as superseded by new memory (conflict resolution)."""
        self._conn.execute(
            "UPDATE memories SET superseded_by = ? WHERE id = ?",
            (new_id, old_id),
        )
        self._conn.commit()

    def count(self, agent_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE agent_id = ? AND superseded_by IS NULL",
            (agent_id,),
        ).fetchone()
        return row[0] if row else 0

    def get_all_preferences(self, agent_id: str, limit: int = 50) -> list[MemoryRecord]:
        """Get all preferences including superseded ones (for history tracing)."""
        rows = self._conn.execute(
            """SELECT * FROM memories
               WHERE agent_id = ? AND memory_type = 'preference'
               ORDER BY created_at DESC LIMIT ?""",
            (agent_id, limit),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows if r]

    def get_active_preferences(self, agent_id: str) -> dict[str, Any]:
        rows = self._conn.execute(
            """SELECT id, content, confidence, created_at FROM memories
               WHERE agent_id = ? AND memory_type = 'preference'
               AND superseded_by IS NULL
               ORDER BY confidence DESC, created_at DESC LIMIT 20""",
            (agent_id,),
        ).fetchall()
        return {
            "preferences": [
                {"id": r[0], "content": r[1], "confidence": r[2], "created_at": r[3]}
                for r in rows
            ]
        }

    def get_decaying_memories(self, agent_id: str, threshold_days: float = 7.0) -> list[MemoryRecord]:
        """Get memories that haven't been accessed recently (candidates for decay)."""
        cutoff = time.time() - (threshold_days * 86400)
        rows = self._conn.execute(
            """SELECT * FROM memories
               WHERE agent_id = ? AND accessed_at < ? AND superseded_by IS NULL
               ORDER BY accessed_at ASC LIMIT 50""",
            (agent_id, cutoff),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows if r]

    def _row_to_memory(self, row: sqlite3.Row | tuple) -> MemoryRecord | None:
        if not row:
            return None
        return MemoryRecord(
            id=row[0],
            agent_id=row[1],
            session_id=row[2],
            content=row[3],
            memory_type=row[4],
            confidence=row[5],
            tags=json.loads(row[6]) if isinstance(row[6], str) else (row[6] or []),
            source=row[7],
            created_at=row[8],
            updated_at=row[9],
            accessed_at=row[10],
            access_count=row[11],
            superseded_by=row[12],
            embedding=json.loads(row[13]) if isinstance(row[13], str) and row[13] else None,
        )
