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
        ⭐ 六维上下文感知排序（6-Dimension Scoring）。

        受 ERINYS 6-signal + NaLog Blend Score 启发：
        评分 = 置信度(30%) + 重要性(10%) + 新鲜度(15%)
               + 相关度(15%) + 证据溯源(10%) + 过期度惩罚(10%)
               + 矛盾状态(5%) + 同会话加分(5%)

        ★ 六维 vs 旧版三维：
        - 新增「过期度惩罚」：半衰期越远，扣分越多 → 避免用过期记忆做决策
        - 新增「矛盾状态」：有 pending 矛盾的记忆降权 → 避免矛盾信息污染上下文
        - 新增「证据溯源」：用户陈述 > 工具结果 > AI 推断 → 不同来源不同权重
        - 改进「相关度」：从简单 difflib 改为 difflib + 关键词命中多级评分
        """
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        q = query.lower().strip() if query else ""

        # ── 1. 半衰期配置（与 fact_network._apply_decay 保持一致）──
        _HALF_LIFE_MAP = {
            "abstraction": 60.0,
            "core": 90.0,
        }

        def _calc_staleness(f: FactTriple, now_dt: datetime) -> float:
            """
            计算过期度 (0~1, 越高越过期)。

            半衰期 = f(编码层级, 置信度, 访问频率)
            天数 / 半衰期 → 指数映射为 0~1 的惩罚值。
            """
            try:
                accessed = datetime.fromisoformat(f.accessed_at)
                days = max(0, (now_dt - accessed).total_seconds() / 86400)
            except (ValueError, TypeError):
                return 0.0

            if days <= 0:
                return 0.0

            # 确定半衰期（天）
            hl = _HALF_LIFE_MAP.get(f.encoding_level, None)
            if hl is None:
                if f.confidence >= 0.6:
                    hl = 30.0
                elif f.confidence >= 0.3:
                    hl = 14.0
                else:
                    hl = 7.0

            # 访问频率修正
            if f.access_count > 10:
                hl *= 1.5
            elif f.access_count > 5:
                hl *= 1.2

            # stigma = 1 − 2^(−天数/半衰期)  → 半衰期时 stigma=0.5
            import math
            stigma = 1.0 - (0.5 ** (days / hl))
            return min(1.0, stigma)

        def score(f: FactTriple) -> float:
            # ★ ① 置信度 (30%) — 高置信优先
            s = f.confidence * 0.30

            # ★ ② 重要性 (10%) — 重要知识优先
            s += f.importance * 0.10

            # ★ ③ 新鲜度 (15%) — 越近访问越高
            try:
                accessed = datetime.fromisoformat(f.accessed_at)
                days = max(0, (now - accessed).total_seconds() / 86400)
                recency = 1.0 / (1.0 + days * 0.5)
            except (ValueError, TypeError):
                recency = 0.3
            s += recency * 0.15

            # ★ ④ 相关度 (15%) — 多级匹配
            if q:
                import difflib
                fact_text = f"{f.subject} {f.predicate} {f.object}".lower()
                if q in fact_text:
                    relevance = 0.9
                elif any(term in fact_text for term in q.split()):
                    relevance = 0.7
                elif q in f.fact_type.lower():
                    relevance = 0.5
                else:
                    relevance = difflib.SequenceMatcher(None, q, fact_text).ratio()
                s += relevance * 0.15
            else:
                s += 0.05

            # ★ ⑤ 证据溯源 (10%) — 来源可信度分层
            source_weights = {
                "user_statement": 1.0,      # 用户明确陈述
                "user_confirmation": 1.0,   # 用户确认
                "user_challenge": 0.8,      # 用户质疑（仍算直接）
                "agent_inference": 0.4,     # AI 推理（可信度较低）
                "tool_result": 0.7,         # 工具结果
                "system": 1.0,              # 系统操作
                "memory_abstraction": 0.6,  # 抽象归纳
            }
            source_type = f.evidence[0].source if f.evidence else ""
            sw = source_weights.get(source_type, 0.5)
            s += sw * 0.10

            # ★ ⑥ 过期度惩罚 (10%) — 超过半衰期越远扣分越多
            staleness = _calc_staleness(f, now)
            s += (1.0 - staleness) * 0.10  # 惩罚 = (1−staleness) 扣分

            # ★ ⑦ 矛盾状态 (5%) — 有 pending 矛盾的记忆降权
            if f.contradictions:
                s *= 0.95  # 有矛盾直接乘 0.95
                s -= min(len(f.contradictions) * 0.01, 0.05)  # 多个矛盾叠加扣

            # ★ ⑧ 同会话加分 (5%) — 当前会话产生的记忆优先
            if session_id and f.source_session == session_id:
                s += 0.05
            elif f.source_session:
                s += 0.02

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
