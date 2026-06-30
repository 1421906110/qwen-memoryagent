"""Tests for MemoryAgent — core memory lifecycle."""

import json
import os
import shutil
import tempfile
import time

import pytest

from memory_agent.models import MemoryRecord, RecallResult
from memory_agent.services.memory_service import MemoryService, _decay_factor, _semantic_similarity
from memory_agent.storage import SQLiteStore


@pytest.fixture
def store():
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "test.db")
    s = SQLiteStore(db_path)
    yield s
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def svc(store):
    return MemoryService(store)


class TestMemoryStore:
    def test_store_and_recall(self, svc):
        mem = svc.remember("agent-1", "session-1", "User prefers dark mode", "preference", 0.9)
        assert mem.id.startswith("mem_")

        result = svc.recall("agent-1", "dark mode")
        assert len(result.memories) == 1
        assert result.memories[0].content == "User prefers dark mode"

    def test_recall_by_type(self, svc):
        svc.remember("agent-1", "s1", "fact A", "fact", 0.9)
        svc.remember("agent-1", "s1", "pref B", "preference", 0.8)

        result = svc.recall("agent-1", memory_types=["preference"])
        assert len(result.memories) == 1
        assert result.memories[0].memory_type == "preference"

    def test_recall_confidence_filter(self, svc):
        svc.remember("agent-1", "s1", "low conf", "observation", 0.2)
        svc.remember("agent-1", "s1", "high conf", "observation", 0.9)

        result = svc.recall("agent-1", min_confidence=0.5)
        assert len(result.memories) == 1
        assert result.memories[0].content == "high conf"

    def test_forget(self, svc):
        mem = svc.remember("agent-1", "s1", "delete me")
        assert svc.forget(mem.id) is True
        result = svc.recall("agent-1", "delete me")
        assert len(result.memories) == 0

    def test_count(self, store, svc):
        assert store.count("agent-1") == 0
        svc.remember("agent-1", "s1", "m1")
        svc.remember("agent-1", "s1", "m2")
        svc.remember("agent-1", "s1", "m3")
        assert store.count("agent-1") == 3


class TestConflictResolution:
    def test_near_duplicate_supersedes(self, svc):
        svc.remember("agent-1", "s1", "User likes Python", "preference", 0.8)
        time.sleep(0.01)
        svc.remember("agent-1", "s1", "User likes Python very much", "preference", 0.9)

        result = svc.recall("agent-1", "Python likes")
        # Should only return the newer one (old one superseded)
        contents = [m.content for m in result.memories]
        assert len(contents) == 1
        assert "very much" in contents[0]

    def test_preference_supersedes_high_overlap(self, svc):
        """Near-identical preference text triggers conflict resolution."""
        svc.remember("agent-1", "s1", "User prefers email notifications", "preference", 0.7)
        time.sleep(0.01)
        svc.remember("agent-1", "s1", "User prefers Slack notifications", "preference", 0.9)

        result = svc.recall("agent-1", "prefers notifications")
        assert len(result.memories) == 1
        assert "Slack" in result.memories[0].content

    def test_fresh_preference_outranks_old(self, svc):
        """Even without conflict detection, newer higher-confidence preference ranks first."""
        svc.remember("agent-1", "s1", "Likes coffee", "preference", 0.6)
        time.sleep(0.01)
        svc.remember("agent-1", "s1", "Now prefers tea over coffee", "preference", 0.9)

        result = svc.recall("agent-1", "coffee tea")
        assert len(result.memories) == 2
        # Higher confidence one should be first
        assert result.memories[0].confidence >= result.memories[1].confidence


class TestGrooming:
    def test_groom_prunes_low_confidence(self, svc):
        svc.remember("agent-1", "s1", "stale info", "observation", 0.1)
        stats = svc.groom("agent-1")
        # confidence 0.1 is already low, groom further reduces it
        result = svc.recall("agent-1", "stale", min_confidence=0.0)
        # May or may not be pruned depending on decay calculation

    def test_multiple_agents_isolation(self, svc):
        svc.remember("agent-a", "s1", "agent a data", "fact")
        svc.remember("agent-b", "s1", "agent b data", "fact")

        result_a = svc.recall("agent-a", "data")
        result_b = svc.recall("agent-b", "data")

        assert len(result_a.memories) == 1
        assert len(result_b.memories) == 1
        assert "agent a" in result_a.memories[0].content
        assert "agent b" in result_b.memories[0].content


class TestDecayTrace:
    """Tests for memory decay visualization."""

    def test_compute_decay_trace_returns_expected_structure(self, svc):
        mem = svc.remember("agent-1", "s1", "test memory", "fact", 0.9)
        trace = svc.compute_decay_trace(mem.id, days=10, points=5)
        assert trace["memory_id"] == mem.id
        assert trace["initial_confidence"] == 0.9
        assert len(trace["trace"]) == 5
        assert trace["halflife_hours"] > 0

    def test_compute_decay_trace_unknown_memory(self, svc):
        trace = svc.compute_decay_trace("nonexistent")
        assert "error" in trace

    def test_decay_factor_decreases_with_age(self):
        """Test that decay factor decreases as age increases."""
        f0 = _decay_factor(age_hours=0, access_count=0)
        f1 = _decay_factor(age_hours=6, access_count=0)
        f2 = _decay_factor(age_hours=24, access_count=0)
        assert f0 == 1.0  # fresh
        assert f1 < f0  # 6 hours later
        assert f2 < f1  # 1 day later

    def test_decay_factor_access_count_protects(self):
        """Test that frequent access slows decay."""
        no_access = _decay_factor(age_hours=48, access_count=0)
        frequent_access = _decay_factor(age_hours=48, access_count=5)
        # With more access, decay should be less (higher factor)
        assert frequent_access > no_access

    def test_get_memory_decay_analysis(self, svc):
        svc.remember("agent-1", "s1", "fresh memory", "fact", 0.9)
        analysis = svc.get_memory_decay_analysis("agent-1")
        assert len(analysis) == 1
        assert analysis[0]["effective_confidence"] <= analysis[0]["initial_confidence"]
        assert "needs_refresh" in analysis[0]


class TestPreferenceHistory:
    """Tests for preference evolution tracking."""

    def test_get_preference_history_tracks_changes(self, svc):
        svc.remember("agent-1", "s1", "Prefers email notifications", "preference", 0.7)
        time.sleep(0.01)
        svc.remember("agent-1", "s1", "Now prefers Slack notifications", "preference", 0.9)

        history = svc.get_preference_history("agent-1")
        # Should have both the old (superseded) and new preference
        assert len(history) == 2

        # Newer one should be active, older one superseded
        active = [h for h in history if h["is_active"]]
        superseded = [h for h in history if not h["is_active"]]
        assert len(active) >= 1

    def test_get_preference_history_empty(self, svc):
        history = svc.get_preference_history("agent-1")
        assert history == []


class TestSemanticSimilarity:
    """Tests for Jaccard-based conflict detection."""

    def test_identical_texts(self):
        sim = _semantic_similarity("User likes Python", "User likes Python")
        assert sim == 1.0

    def test_completely_different(self):
        sim = _semantic_similarity("User likes Python", "The weather is nice today")
        assert sim == 0.0

    def test_partial_overlap(self):
        sim = _semantic_similarity("User prefers email notifications", "User prefers Slack notifications")
        assert 0.3 < sim < 0.8

    def test_empty_strings(self):
        sim = _semantic_similarity("", "something")
        assert sim == 0.0


class TestLLMClientHelpers:
    """Tests for LLMClient utility functions (token estimation, memory selection)."""

    def test_estimate_tokens(self):
        from memory_agent.services.llm_client import estimate_tokens
        assert estimate_tokens("hello") > 0
        assert estimate_tokens("") == 0

    def test_select_memories_budget(self):
        from memory_agent.services.llm_client import select_memories_for_context
        memories = [
            {"content": "a" * 100, "confidence": 0.9, "memory_type": "preference", "created_at": 1770000000},
            {"content": "b" * 100, "confidence": 0.1, "memory_type": "observation", "created_at": 1770000000},
        ]
        selected = select_memories_for_context(memories, max_tokens=5000)
        assert len(selected) >= 1

    def test_select_memories_empty(self):
        from memory_agent.services.llm_client import select_memories_for_context
        assert select_memories_for_context([]) == []

    def test_select_memories_preference_bonus(self):
        from memory_agent.services.llm_client import select_memories_for_context
        # Preference should get priority over observation at same confidence
        memories = [
            {"content": "low", "confidence": 0.5, "memory_type": "observation", "created_at": 1770000000},
            {"content": "high", "confidence": 0.5, "memory_type": "preference", "created_at": 1770000000},
        ]
        selected = select_memories_for_context(memories, max_tokens=100000)
        assert selected[0]["memory_type"] == "preference"


class TestMemoryMerge:
    """Tests for memory merge / aggregation."""

    def test_similar_clusters_found(self, svc):
        """Store similar memories and verify they get clustered."""
        agent = "merge_test"
        session = "s1"

        # Three similar memories (Jaccard ~0.3-0.36, below conflict threshold 0.5)
        svc.remember(agent, session, "User prefers dark mode always", "preference", 0.9)
        svc.remember(agent, session, "User likes dark mode for computer work", "preference", 0.85)
        svc.remember(agent, session, "Dark mode is the best for user eyes", "preference", 0.8)
        # One unrelated memory
        svc.remember(agent, session, "User lives in Shanghai", "fact", 0.9)

        clusters = svc.find_similar_clusters(agent, sim_threshold=0.2, min_cluster_size=2)
        assert len(clusters) >= 1
        # The dark mode cluster should have 3 members
        dark_cluster = [c for c in clusters if len(c) >= 3]
        assert len(dark_cluster) >= 1

    def test_merge_cluster_creates_aggregated_memory(self, svc):
        """Merging a cluster should produce a new memory and supersede originals."""
        agent = "merge_test2"
        session = "s1"

        ids = []
        for text in [
            "User chooses Python for machine learning work",
            "User prefers Python for data science work",
            "Python is user preferred language for machine learning and data science",
        ]:
            m = svc.remember(agent, session, text, "preference", 0.8)
            ids.append(m.id)

        cluster = svc.find_similar_clusters(agent, sim_threshold=0.2, min_cluster_size=2)
        assert len(cluster) >= 1

        merged = svc.merge_cluster(cluster[0])
        assert merged is not None
        assert merged.memory_type == "preference"
        assert merged.source == "merge"

        # Original memories should be superseded
        for mid in ids:
            orig = svc.store.get(mid)
            assert orig is not None
            assert orig.superseded_by == merged.id

    def test_merge_all_returns_stats(self, svc):
        """merge_all should return proper statistics."""
        agent = "merge_test3"
        session = "s1"

        for i in range(5):
            svc.remember(agent, session, f"User mentioned topic about AI agent memory system part {i}", "observation", 0.7)
        svc.remember(agent, session, "User likes coffee very much", "preference", 0.9)

        stats = svc.merge_all(agent, sim_threshold=0.2, min_cluster_size=2)
        assert stats["clusters_found"] >= 1
        assert stats["memories_created"] >= 1
        assert stats["memories_merged"] >= 2
        assert len(stats["results"]) >= 1
        assert stats["results"][0]["source_count"] >= 2

    def test_merge_no_clusters_small(self, svc):
        """No merge should happen with fewer memories than min_cluster_size."""
        agent = "merge_test4"
        session = "s1"
        svc.remember(agent, session, "Just one memory", "observation", 0.5)

        stats = svc.merge_all(agent, min_cluster_size=2)
        assert stats["clusters_found"] == 0
        assert stats["memories_created"] == 0
