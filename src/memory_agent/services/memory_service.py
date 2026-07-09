"""
MemoryAgent — Core memory service.

Implements the full memory lifecycle:
  Store → Recall → Groom (Decay + Conflict Resolution)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import Any

from memory_agent.models import MemoryRecord, RecallResult
from memory_agent.storage import MemoryStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Decay helpers  (Ebbinghaus-inspired)
# ---------------------------------------------------------------------------

def _decay_factor(age_hours: float, access_count: int) -> float:
    """Ebbinghaus-inspired confidence decay.

    New memories decay faster; frequently-accessed memories decay slower.
    Returns a multiplier in [0..1].
    """
    if access_count == 0:
        # never recalled — steep decay after ~6 hours
        return max(0.0, 1.0 - (age_hours / 24.0) ** 0.7)
    # each recall strengthens memory; decay flattens with access count
    half_life = 24.0 * (1.0 + max(0, access_count) ** 0.5)  # hours
    return max(0.0, 2.0 ** (-age_hours / half_life))


def _semantic_similarity(a: str, b: str) -> float:
    """Simple Jaccard + length heuristic for conflict detection.

    A proper implementation would use embeddings, but this is sufficient
    for detecting near-duplicate or directly contradictory statements.

    ⭐ 中文支持：对中文按字符切分（.split() 中文=单token，Jaccard 全0）
    """
    def _tokenize(s: str) -> set:
        """中英文混合分词：英文按空格，中文按单字"""
        import re
        tokens = set()
        for part in re.findall(r'[a-zA-Z]+|[0-9]+|[一-鿿]', s.lower()):
            tokens.add(part)
        return tokens

    set_a = _tokenize(a)
    set_b = _tokenize(b)
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
#  Memory Service
# ---------------------------------------------------------------------------

class MemoryService:
    """Core memory lifecycle manager."""

    def __init__(self, store: MemoryStore, llm_embed_fn=None, llm_client=None):
        self.store = store
        self._embed_fn = llm_embed_fn  # optional: callable(content) -> list[float]
        self._llm = llm_client  # optional: LLMClient for semantic comparison

    # ---- Write ----

    def remember(
        self,
        agent_id: str,
        session_id: str,
        content: str,
        memory_type: str = "observation",
        confidence: float = 0.8,
        tags: list[str] | None = None,
        source: str = "conversation",
    ) -> MemoryRecord:
        """Store a new memory."""
        now = time.time()

        memory = MemoryRecord(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            agent_id=agent_id,
            session_id=session_id,
            content=content,
            memory_type=memory_type,
            confidence=confidence,
            tags=tags or [],
            source=source,
            created_at=now,
            updated_at=now,
            accessed_at=now,
        )

        # Generate embedding if function available
        if self._embed_fn:
            try:
                memory.embedding = self._embed_fn(content)
            except Exception as e:
                logger.warning("Embedding failed for memory '%s': %s", str(content)[:50], e)

        # Conflict detection: check if this replaces a previous memory
        self._detect_and_resolve_conflicts(agent_id, memory)

        stored_id = self.store.store(memory)
        memory.id = stored_id
        return memory

    def forget(self, memory_id: str) -> bool:
        """Delete a memory by id."""
        return self.store.delete(memory_id)

    # ---- Read ----

    def recall(
        self,
        agent_id: str,
        query_text: str | None = None,
        query_embedding: list[float] | None = None,
        memory_types: list[str] | None = None,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> RecallResult:
        """Retrieve relevant memories."""
        t0 = time.time()

        # If we have an embed function and no embedding provided, generate one
        embedding = query_embedding
        if not embedding and query_text and self._embed_fn:
            try:
                embedding = self._embed_fn(query_text)
            except Exception as e:
                logger.warning("Embedding failed for search query '%s': %s", str(query_text)[:50], e)

        memories = self.store.search(
            agent_id=agent_id,
            query_embedding=embedding,
            memory_types=memory_types,
            limit=limit,
            min_confidence=min_confidence,
        )

        # Apply Ebbinghaus decay to confidence scores for display
        now = time.time()
        for m in memories:
            age_hours = (now - m.accessed_at) / 3600.0
            m.confidence *= _decay_factor(age_hours, m.access_count)

        # Re-sort by decayed confidence
        memories.sort(key=lambda m: m.confidence, reverse=True)

        elapsed = (time.time() - t0) * 1000
        return RecallResult(
            memories=memories[:limit],
            total_found=len(memories),
            query_time_ms=round(elapsed, 2),
        )

    # ---- Groom (maintenance) ----

    def groom(self, agent_id: str) -> dict[str, Any]:
        """Run maintenance: apply decay, prune forgotten memories.

        Returns grooming stats.
        """
        now = time.time()
        stats = {"decayed": 0, "pruned": 0, "conflicts_checked": 0}

        # Apply decay: find old low-access memories and lower confidence
        old_memories = self.store.get_decaying_memories(agent_id, threshold_days=2.0)
        for mem in old_memories:
            age_hours = (now - mem.accessed_at) / 3600.0
            factor = _decay_factor(age_hours, mem.access_count)
            new_conf = mem.confidence * factor
            if new_conf < 0.05:
                # Effectively forgotten — prune
                self.store.delete(mem.id)
                stats["pruned"] += 1
            elif new_conf < mem.confidence * 0.8:
                # Significant decay — record it
                stats["decayed"] += 1

        return stats

    # ---- Conflict Resolution ----

    def compute_decay_trace(
        self,
        memory_id: str,
        days: int = 30,
        points: int = 50,
    ) -> dict:
        """Compute confidence decay trace for a memory over time.

        Simulates the Ebbinghaus decay curve forward from creation,
        returning a time series for visualization.
        """
        mem = self.store.get(memory_id)
        if not mem:
            return {"error": "Memory not found"}

        now = time.time()
        created = mem.created_at
        timespan = days * 86400  # seconds
        traces = []

        for i in range(points):
            t = created + (timespan * i / points)
            age_hours = max(0, (t - mem.accessed_at) / 3600.0)
            eff_access = mem.access_count
            factor = _decay_factor(age_hours, eff_access)
            traces.append({
                "timestamp": t,
                "age_hours": round(age_hours, 1),
                "confidence": round(mem.confidence * factor, 4),
            })

        # Add key events: creation, last access
        key_events = [
            {"type": "created", "timestamp": mem.created_at, "confidence": mem.confidence},
        ]
        if mem.access_count > 0:
            key_events.append({
                "type": "last_accessed",
                "timestamp": mem.accessed_at,
                "confidence": round(mem.confidence * _decay_factor(
                    max(0, (mem.accessed_at - mem.accessed_at) / 3600.0),
                    mem.access_count,
                ), 4),
            })
        if mem.superseded_by:
            successor = self.store.get(mem.superseded_by)
            if successor:
                key_events.append({
                    "type": "superseded",
                    "timestamp": successor.created_at,
                    "confidence": 0.0,
                    "superseded_by": mem.superseded_by,
                })

        return {
            "memory_id": memory_id,
            "content": mem.content[:200],
            "memory_type": mem.memory_type,
            "initial_confidence": mem.confidence,
            "access_count": mem.access_count,
            "created_at": mem.created_at,
            "halflife_hours": round(24.0 * (1.0 + mem.access_count ** 0.5), 1),
            "trace": traces,
            "key_events": key_events,
        }

    def get_memory_decay_analysis(self, agent_id: str, min_confidence: float = 0.0) -> list[dict]:
        """Analyze decay state for all memories of an agent."""
        memories = self.store.search(
            agent_id=agent_id,
            limit=100,
            min_confidence=min_confidence,
        )
        now = time.time()
        analysis = []
        for m in memories:
            age_hours = (now - m.accessed_at) / 3600.0
            factor = _decay_factor(age_hours, m.access_count)
            effective_conf = m.confidence * factor
            analysis.append({
                "id": m.id,
                "content": m.content[:100],
                "memory_type": m.memory_type,
                "initial_confidence": m.confidence,
                "effective_confidence": round(effective_conf, 3),
                "age_hours": round(age_hours, 1),
                "access_count": m.access_count,
                "halflife_hours": round(24.0 * (1.0 + m.access_count ** 0.5), 1),
                "decay_factor": round(factor, 3),
                "needs_refresh": effective_conf < 0.3,
            })
        analysis.sort(key=lambda x: x["effective_confidence"])
        return analysis

    # ---- LLM-assisted semantic comparison ----

    def _llm_semantic_compare(self, text_a: str, text_b: str) -> dict:
        """Use LLM to compare two memories for semantic relationship.

        Returns: {relation, similarity_score, explanation}
          relation: near_duplicate | preference_override | contradiction | related | unrelated
        """
        if not self._llm:
            return {"relation": "unrelated", "similarity_score": 0.0, "explanation": "no LLM"}

        prompt = (
            "Compare these two memory entries. Determine their relationship:\n\n"
            f"Memory A: \"{text_a}\"\n"
            f"Memory B: \"{text_b}\"\n\n"
            "Respond with JSON: {\"relation\": \"near_duplicate|preference_override|contradiction|related|unrelated\", "
            "\"similarity_score\": 0.0-1.0, \"explanation\": \"short reason\"}\n"
            "near_duplicate: same info rephrased\n"
            "preference_override: B is a newer version of A's preference\n"
            "contradiction: B directly contradicts A\n"
            "related: topically related but not conflicting\n"
            "unrelated: no meaningful connection"
        )
        try:
            resp = self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=256,
            )
            result = json.loads(
                resp.strip().removeprefix("```json").removesuffix("```").strip()
            )
            return {
                "relation": result.get("relation", "unrelated"),
                "similarity_score": float(result.get("similarity_score", 0.0)),
                "explanation": result.get("explanation", ""),
            }
        except Exception as e:
            logger.warning("LLM semantic compare failed: %s", e)
            return {"relation": "unrelated", "similarity_score": 0.0, "explanation": str(e)}

    def _detect_and_resolve_conflicts(self, agent_id: str, new_mem: MemoryRecord):
        """Check if new memory contradicts/supersedes existing ones.

        Uses three methods in order:
          1. Embedding cosine similarity (if available)
          2. LLM semantic comparison (if LLM client available)
          3. Jaccard word overlap (fallback)

        For preferences: also builds an evolution chain.
        """
        existing = self.store.search(
            agent_id=agent_id,
            memory_types=[new_mem.memory_type],
            limit=20,
            min_confidence=0.3,
        )

        for old in existing:
            if old.id == new_mem.id or old.superseded_by:
                continue

            # Method 1: Embedding cosine similarity (if available)
            emb_sim = 0.0
            if (
                self._embed_fn
                and old.embedding
                and new_mem.embedding
            ):
                from memory_agent.storage import _compute_similarity
                emb_sim = _compute_similarity(old.embedding, new_mem.embedding)

            # Method 2: LLM semantic comparison (if available)
            relation = None
            if self._llm and emb_sim > 0.5:  # only uses LLM when embeddings suggest relation
                llm_result = self._llm_semantic_compare(new_mem.content, old.content)
                relation = llm_result.get("relation")

            # Method 3: Jaccard fallback
            jaccard = _semantic_similarity(new_mem.content, old.content)

            # Decision logic
            should_supersede = False
            reason = None

            if relation == "near_duplicate":
                should_supersede = True
                reason = "llm_near_duplicate"
            elif relation == "preference_override":
                should_supersede = True
                reason = "llm_preference_override"
            elif relation == "contradiction":
                # Contradiction: newer (higher confidence) wins
                if new_mem.confidence >= old.confidence:
                    should_supersede = True
                    reason = "llm_contradiction_newer_wins"
            elif jaccard > 0.85:
                should_supersede = True
                reason = "jaccard_near_duplicate"
            elif jaccard > 0.5 and new_mem.memory_type == "preference":
                should_supersede = True
                reason = "jaccard_preference_override"
            elif emb_sim > 0.85 and new_mem.memory_type == "preference":
                should_supersede = True
                reason = "embedding_preference_override"

            if should_supersede:
                self.store.supersede(old.id, new_mem.id)
                logger.info(
                    "Conflict: %s superseded %s (reason=%s, jaccard=%.2f, emb=%.2f)",
                    new_mem.id, old.id, reason, jaccard, emb_sim,
                )

    # ---- Preference evolution ----

    def get_preference_history(self, agent_id: str) -> list[dict]:
        """Get preference evolution history.

        Traces preference supersession chains to show how preferences
        changed over time.
        """
        all_prefs = self.store.get_all_preferences(agent_id, limit=50)

        active = self.store.get_active_preferences(agent_id)
        active_ids = {p.get("id") for p in active.get("preferences", [])}

        chains = []
        for mem in all_prefs:
            if not mem:
                continue
            chains.append({
                "id": mem.id,
                "content": mem.content,
                "confidence": mem.confidence,
                "memory_type": mem.memory_type,
                "created_at": mem.created_at,
                "superseded_by": mem.superseded_by,
                "is_active": mem.id in active_ids or mem.superseded_by is None,
            })
        return chains

    # ---- Memory Merge / Aggregation ----

    def _compute_pairwise_sim(self, a: str, b: str) -> float:
        """Compute pairwise similarity between two texts.
        Uses embedding cosine sim if available, falls back to Jaccard.
        """
        if self._embed_fn:
            try:
                emb_a = self._embed_fn(a)
                emb_b = self._embed_fn(b)
                from memory_agent.storage import _compute_similarity
                sim = _compute_similarity(emb_a, emb_b)
                if sim > 0.0:
                    return sim
            except Exception:
                pass
        return _semantic_similarity(a, b)

    def find_similar_clusters(
        self,
        agent_id: str,
        sim_threshold: float = 0.65,
        min_cluster_size: int = 2,
        limit: int = 100,
    ) -> list[list[MemoryRecord]]:
        """Find clusters of similar memories using pairwise similarity.

        Args:
            agent_id: Target agent.
            sim_threshold: Minimum pairwise similarity to be in same cluster.
            min_cluster_size: Minimum members for a cluster to be returned.
            limit: Max memories to scan.

        Returns:
            List of clusters, each cluster is a list of similar MemoryRecords.
        """
        # Fetch recent active memories
        memories = self.store.search(
            agent_id=agent_id,
            limit=limit,
            min_confidence=0.1,
        )

        if len(memories) < min_cluster_size:
            return []

        # Build similarity matrix (only upper triangle)
        n = len(memories)
        sim_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            sim_matrix[i][i] = 1.0
            for j in range(i + 1, n):
                sim = self._compute_pairwise_sim(
                    memories[i].content, memories[j].content
                )
                sim_matrix[i][j] = sim
                sim_matrix[j][i] = sim

        # Simple greedy clustering: start from highest-degree node each round
        assigned = set()
        clusters = []

        # Build degree: how many neighbors above threshold
        degrees = [
            sum(1 for j in range(n) if j != i and sim_matrix[i][j] >= sim_threshold)
            for i in range(n)
        ]

        while True:
            # Pick unassigned node with highest degree
            candidates = [
                i for i in range(n) if i not in assigned
            ]
            if not candidates:
                break
            seed = max(candidates, key=lambda i: degrees[i])

            # Build cluster around seed: include all unassigned nodes
            # that have sim >= threshold with the seed
            cluster = [memories[seed]]
            assigned.add(seed)

            added = True
            while added:
                added = False
                for j in range(n):
                    if j in assigned:
                        continue
                    # Check if j is similar to any existing cluster member
                    for cl in cluster:
                        idx = memories.index(cl)
                        if sim_matrix[idx][j] >= sim_threshold:
                            cluster.append(memories[j])
                            assigned.add(j)
                            added = True
                            break

            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)

        return clusters

    def merge_cluster(
        self,
        cluster: list[MemoryRecord],
    ) -> MemoryRecord | None:
        """Merge a cluster of similar memories into one aggregated memory.

        Uses LLM to generate a concise summary. Falls back to keeping the
        highest-confidence memory if LLM is unavailable.

        Args:
            cluster: A list of similar MemoryRecords to merge.

        Returns:
            The new merged MemoryRecord, or None if merge fails.
        """
        if len(cluster) < 2:
            return None

        # Sort by confidence desc, then created_at desc
        cluster.sort(key=lambda m: (-m.confidence, -m.created_at))

        contents = [m.content for m in cluster]
        top_mem = cluster[0]

        merged_text = None

        # Try LLM merge first
        if self._llm:
            prompt = (
                "You are a memory aggregation system. Merge the following "
                "similar memory entries into ONE concise summary. "
                "Capture ALL distinct information without losing details. "
                "Do NOT add new information.\n\n"
                "Memories:\n"
                + "\n".join(f"- {c}" for c in contents)
                + "\n\nReturn ONLY the merged summary text, no extra formatting."
            )
            try:
                resp = self._llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=512,
                )
                merged_text = resp.strip().strip('"\'')
                if len(merged_text) < 10:
                    merged_text = None
                logger.info(
                    "LLM merged %d memories: %s → %s",
                    len(cluster), contents[0][:50], merged_text[:80] if merged_text else "FAILED",
                )
            except Exception as e:
                logger.warning("LLM merge failed: %s — using fallback", e)

        # Fallback: keep highest-confidence memory
        if not merged_text:
            merged_text = top_mem.content
            logger.info("Fallback merge (no LLM): keeping top memory")

        # Ensure we have a session_id
        session_id = cluster[0].session_id

        # Collate tags
        all_tags = list(set(t for m in cluster for t in (m.tags or [])))

        # Compute merged confidence: average of top 3 or all if fewer
        confs = sorted([m.confidence for m in cluster], reverse=True)[:3]
        merged_conf = round(sum(confs) / len(confs), 3)

        # Determine merged memory type: majority vote
        type_votes = {}
        for m in cluster:
            t = m.memory_type or "observation"
            type_votes[t] = type_votes.get(t, 0) + 1
        merged_type = max(type_votes, key=type_votes.get)

        # Create merged memory
        now = time.time()
        merged = MemoryRecord(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            agent_id=cluster[0].agent_id,
            session_id=session_id,
            content=merged_text,
            memory_type=merged_type,
            confidence=merged_conf,
            tags=all_tags,
            source="merge",
            created_at=now,
            updated_at=now,
            accessed_at=now,
        )

        # Generate embedding if available
        if self._embed_fn:
            try:
                merged.embedding = self._embed_fn(merged_text)
            except Exception as e:
                logger.warning("Embedding failed during merge: %s", e)

        # Store merged memory
        stored_id = self.store.store(merged)
        merged.id = stored_id

        # Supersede all old memories in the cluster
        for old in cluster:
            if old.id != stored_id:
                self.store.supersede(old.id, stored_id)

        return merged

    def merge_all(
        self,
        agent_id: str,
        sim_threshold: float = 0.65,
        min_cluster_size: int = 2,
        limit: int = 100,
    ) -> dict:
        """Find and merge all clusters of similar memories for an agent.

        Returns merge statistics.
        """
        clusters = self.find_similar_clusters(
            agent_id=agent_id,
            sim_threshold=sim_threshold,
            min_cluster_size=min_cluster_size,
            limit=limit,
        )

        stats = {
            "agent_id": agent_id,
            "clusters_found": len(clusters),
            "memories_merged": 0,
            "memories_created": 0,
            "results": [],
        }

        for cluster in clusters:
            merged = self.merge_cluster(cluster)
            if merged:
                stats["memories_merged"] += len(cluster)
                stats["memories_created"] += 1
                stats["results"].append({
                    "merged_id": merged.id,
                    "merged_content": merged.content[:150],
                    "merged_confidence": merged.confidence,
                    "merged_type": merged.memory_type,
                    "source_count": len(cluster),
                    "source_ids": [m.id for m in cluster],
                })

        return stats
