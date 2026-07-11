"""
智能召回路由 — CogniMem 三级路由

不是傻搜，而是按最短路径获取最相关的记忆。
"""

import logging
from typing import Any
from .models import FactTriple

logger = logging.getLogger(__name__)


class RecallRouter:
    """
    三级召回路由

    L0 — Cache Check (<1ms, 0 token 成本)
    L1 — 精确三元组匹配 (<3ms, 结构查询)
    L2 — 语义扩展 (tag/类型/关联)
    L3 — 向量搜索 (仅当以上全不满足时)
    """

    def __init__(self, fact_network):
        self.fn = fact_network
        self._stats = {"l0_hits": 0, "l1_hits": 0, "l2_hits": 0, "l3_hits": 0, "l3_fallsback": 0, "total": 0}

    def recall(self, query: str, agent_id: str = "default",
               context: dict | None = None, top_k: int = 10) -> list[FactTriple]:
        """
        三级路由召回。

        Args:
            query: 查询文本
            agent_id: Agent ID
            context: 上下文信息 (session_id, 话题标签等)
            top_k: 最大返回数

        Returns: 按置信度排序的事实列表
        """
        self._stats["total"] += 1
        q = query.lower().strip()
        ctx = context or {}
        session_id = ctx.get("session_id", "")
        seen: set[str] = set()
        results: list[FactTriple] = []

        # ── 空查询 → 返回所有事实（浏览/三元组列表模式）──
        if not q:
            for fact in self.fn._get_cached_facts(agent_id):
                if fact.fact_id not in seen:
                    results.append(fact)
                    seen.add(fact.fact_id)
            if self.fn.db:
                for f in self.fn.db.get_agent_facts(agent_id):
                    if f.fact_id not in seen:
                        results.append(f)
                        seen.add(f.fact_id)
                        self.fn._cache_put(f)
            if results:
                return self._rank_and_trim(results, top_k, query, session_id)
            return []

        # ── L0: Cache Check (0 token, <1ms) ──
        cache_hits = self._l0_cache_check(q, agent_id)
        for f in cache_hits:
            if f.fact_id not in seen:
                results.append(f)
                seen.add(f.fact_id)

        if results:
            self._stats["l0_hits"] += 1
            logger.debug(f"L0 cache hit: {len(results)} facts for '{query}'")
            return self._rank_and_trim(results, top_k, query, session_id)

        # ── L1: 精确三元组匹配 ──
        l1_hits = self._l1_exact_match(q, agent_id)
        for f in l1_hits:
            if f.fact_id not in seen:
                results.append(f)
                seen.add(f.fact_id)

        if results:
            self._stats["l1_hits"] += 1
            logger.debug(f"L1 exact match: {len(results)} facts for '{query}'")
            return self._rank_and_trim(results, top_k, query, session_id)

        # ── L1.5: BM25 关键词模糊匹配 ──
        bm25_hits = self._bm25_retrieve(q, agent_id, top_k)
        for f in bm25_hits:
            if f.fact_id not in seen:
                results.append(f)
                seen.add(f.fact_id)

        if results:
            logger.debug(f"L1.5 BM25: {len(results)} facts for '{query}'")
            return self._rank_and_trim(results, top_k, query, session_id)

        # ── L2: 语义扩展 ──
        l2_hits = self._l2_semantic_expand(q, agent_id, context)
        for f in l2_hits:
            if f.fact_id not in seen:
                results.append(f)
                seen.add(f.fact_id)

        # ── L3: 向量搜索 (pgvector) ──
        l2_only = bool(l2_hits)
        if len(results) < top_k:
            l3_hits = self.fn.recall_vector(query, agent_id, top_k)
            for f in l3_hits:
                if f.fact_id not in seen:
                    results.append(f)
                    seen.add(f.fact_id)

        if results:
            if not l2_only:
                # L2 未命中，L3 向量搜索兜底成功
                self._stats["l3_hits"] += 1
            else:
                self._stats["l2_hits"] += 1
        else:
            self._stats["l3_fallsback"] += 1
            logger.debug(f"All cache miss for '{query}', returning empty")

        return self._rank_and_trim(results, top_k, query, session_id)

    def _l0_cache_check(self, query: str, agent_id: str) -> list[FactTriple]:
        """L0: Cache 匹配 (subject/predicate/object + tags + evidence)"""
        matches = []
        for fact in self.fn._get_cached_facts(agent_id):
            if self.fn._match_query(fact, query):
                matches.append(fact)
                self.fn._record_access(fact)
        return matches

    def _l1_exact_match(self, query: str, agent_id: str) -> list[FactTriple]:
        """L1: 数据库精确匹配"""
        if not self.fn.db:
            return []
        return self.fn.db.search_facts(
            agent_id=agent_id,
            subject=query, object=query, predicate=query, tag=query,
            limit=20
        )

    # ═══════════════════════════════════════════════════
    # BM25 关键词检索（L1.5，精确匹配与语义扩展之间）
    # ═══════════════════════════════════════════════════

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """中文双字分词 + 英文单词切分

        中文用双字组（bigram）替代单字，大幅提升搜索精度。
        "项目5的预算" vs "项目4预算" — 单字重叠高，双字组能区分。
        """
        import re
        tokens = []
        # 英文：按空格
        words = re.findall(r'[a-zA-Z0-9_]+', text.lower())
        tokens.extend(words)

        # 中文：提取连续中文字符串，按双字组切分
        chinese_segments = re.findall(r'[一-鿿]+', text)
        for seg in chinese_segments:
            # 单字也保留（短查询时有用）
            for c in seg:
                tokens.append(c)
            # 双字组（核心改进）
            for i in range(len(seg) - 1):
                tokens.append(seg[i:i+2])

        return tokens

    def _bm25_score(self, query: str, doc_text: str,
                    avg_dl: float, N: int,
                    doc_freqs: dict[str, int],
                    k1: float = 1.5, b: float = 0.75) -> float:
        """计算单篇文档的 BM25 得分"""
        query_terms = self._tokenize(query)
        doc_terms = self._tokenize(doc_text)
        dl = len(doc_terms)
        if dl == 0:
            return 0.0

        score = 0.0
        for term in set(query_terms):
            if term not in doc_freqs:
                continue
            tf = doc_terms.count(term)
            idf = max(0.0, (N - doc_freqs[term] + 0.5) / (doc_freqs[term] + 0.5))
            score += (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl)) * idf
        return score

    def _bm25_retrieve(self, query: str, agent_id: str,
                       top_k: int = 10) -> list[FactTriple]:
        """BM25 关键词检索 — 在缓存的 fact 中做模糊匹配"""
        import math
        agent_facts = self.fn._get_cached_facts(agent_id)
        if not agent_facts:
            return []

        # 预处理：所有文档文本 + 统计
        docs = [
            f"{f.subject} {f.predicate} {f.object}"
            for f in agent_facts
        ]
        N = len(docs)
        avg_dl = sum(len(self._tokenize(d)) for d in docs) / max(N, 1)
        doc_freqs: dict[str, int] = {}
        for d in docs:
            for t in set(self._tokenize(d)):
                doc_freqs[t] = doc_freqs.get(t, 0) + 1

        # 打分
        scored = []
        for i, f in enumerate(agent_facts):
            s = self._bm25_score(query, docs[i], avg_dl, N, doc_freqs)
            if s > 0:
                scored.append((s, f))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:top_k]]

    def _l2_semantic_expand(self, query: str, agent_id: str,
                            context: dict | None = None) -> list[FactTriple]:
        """L2: 语义扩展 — tag/类型/关联事实"""
        results = []
        seen = set()

        # difflib 模糊匹配：对 cache 中的事实做相似度匹配
        q = query.lower().strip()
        import difflib
        for fact in self.fn._get_cached_facts(agent_id):
            if fact.fact_id in seen:
                continue
            text = f"{fact.subject} {fact.predicate} {fact.object}".lower()
            ratio = difflib.SequenceMatcher(None, q, text).ratio()
            # 短查询（<5字符）容易被 SequenceMatcher 误匹配，提高阈值
            min_ratio = 0.5 if len(q) < 5 else 0.35
            if ratio > min_ratio:
                results.append(fact)
                seen.add(fact.fact_id)

        # 从数据库按 tag 匹配
        if self.fn.db:
            tag_results = self.fn.db.search_facts(
                agent_id=agent_id, tag=query, limit=10
            )
            for f in tag_results:
                if f.fact_id not in seen:
                    results.append(f)
                    seen.add(f.fact_id)

        # 关联事实扩展 (通过 connected_facts)
        for f in list(results):
            for connected_id in f.connected_facts:
                if connected_id not in seen:
                    cf = self.fn._get_fact(connected_id)
                    if cf and cf.agent_id == agent_id:
                        results.append(cf)
                        seen.add(cf.fact_id)

        return results

    def _rank_and_trim(self, facts: list[FactTriple], top_k: int,
                        query: str = "", session_id: str = "") -> list[FactTriple]:
        """
        ⭐ 上下文感知排序。

        评分 = 置信度(45%) + 重要性(15%) + 新鲜度(20%) + 相关度(10%)
               + 证据权重(10%) + 同会话加分(额外)

        - 新鲜度: 越近访问的事实分越高（艾宾浩斯复习效应）
        - 相关度: 查询与事实的语义匹配度
        - 证据权重: 有原始来源的事实更可靠
        - 同会话加分: 本对话中产生的事实优先
        """
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        q = query.lower().strip() if query else ""

        def score(f: FactTriple) -> float:
            # 置信度 (45%)
            s = f.confidence * 0.45

            # 重要性 (15%)
            s += f.importance * 0.15

            # ⭐ 新鲜度 (20%): 越近访问越高
            try:
                accessed = datetime.fromisoformat(f.accessed_at)
                days = max(0, (now - accessed).days)
                recency = 1.0 / (1.0 + days * 0.5)
            except (ValueError, TypeError):
                recency = 0.3
            s += recency * 0.2

            # ⭐ 相关度 (10%)
            if q:
                import difflib
                fact_text = f"{f.subject} {f.predicate} {f.object}".lower()
                if q in fact_text:
                    relevance = 0.9
                else:
                    relevance = difflib.SequenceMatcher(None, q, fact_text).ratio()
                s += relevance * 0.1
            else:
                s += 0.05

            # ⭐ 证据权重 (10%): 有原始来源的事实更可靠
            has_evidence = bool(f.evidence) and any(
                hasattr(e, 'statement') and e.statement for e in f.evidence
            )
            if has_evidence:
                s += 0.1

            # ⭐ 同会话加分: 本对话产生的记忆优先（项目隔离 + 上下文连贯）
            if session_id and f.source_session == session_id:
                s += 0.15  # 同一会话的事实额外加权
            elif f.source_session:
                s += 0.05  # 至少有来源会话

            return s

        facts.sort(key=score, reverse=True)
        return facts[:top_k]

    def get_stats(self) -> dict:
        """获取路由命中统计"""
        s = self._stats
        total = s["total"] or 1
        return {
            "l0_hit_rate": f"{s['l0_hits']/total*100:.0f}%",
            "l1_hit_rate": f"{s['l1_hits']/total*100:.0f}%",
            "l2_hit_rate": f"{s['l2_hits']/total*100:.0f}%",
            "l3_hit_rate": f"{s['l3_hits']/total*100:.0f}%",
            "l3_fallback_rate": f"{s['l3_fallsback']/total*100:.0f}%",
            "total_queries": s["total"],
        }
