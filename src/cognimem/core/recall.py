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
    五级召回路由（v0.13 MIRIX 主动检索 + Weibull 衰减增强）

    P0 — Intent Classification (<0.01ms, 0 token)
    P1 — ★ Active Retrieval: 提取实体/主题/类型引导召回
    L0 — Cache Check (<1ms, 0 token 成本)
    L1 — 精确三元组匹配 (<3ms, 结构查询)
    L1.5 — BM25 关键词模糊匹配
    L2 — 语义扩展 (tag/类型/关联)
    L3 — 向量搜索 (仅当以上全不满足时)

    ★ v0.13 改进：
    - 主动检索：提取查询中的实体+预期事实类型，引导召回策略
    - Weibull 时间衰减：比指数衰减更科学，初始慢降+后期陡降
    """

    # ═══ 意图检测标记（零 LLM 开销）═══
    _EXPLORATORY_MARKERS = frozenset({
        '什么', '哪些', '如何', '怎么', '有没有',
        '是否', '为什么', '怎样', '多少', '几',
        '谁', '哪', '吗', '吧', '呢', '？', '?',
        '啥', '咋', '咋样', '为何', '何种', '哪个',
    })

    # ═══ 事实类型关键词映射（用于主动检索提取预期类型）═══
    _TYPE_KEYWORDS = {
        'preference': {'喜欢', '爱', '偏好', '口味', '兴趣', '爱好', '想要', '想喝', '想吃', '想玩'},
        'fact': {'是', '叫什么', '地址', '电话', '多少', '多大', '多远', '多久', '在哪里'},
        'goal': {'目标', '计划', '打算', '想实现', '要完成', '任务', '项目', '截止'},
        'decision': {'决定', '选择', '选了', '决定用', '决定买', '决定去'},
        'skill': {'会', '能', '懂', '会做', '会用', '熟练', '擅长'},
    }

    def __init__(self, fact_network):
        self.fn = fact_network
        self._stats = {"l0_hits": 0, "l1_hits": 0, "l2_hits": 0, "l3_hits": 0,
                       "l3_fallsback": 0, "total": 0,
                       "intent_factual": 0, "intent_exploratory": 0, "intent_navigation": 0,
                       "active_retrieval_hits": 0}
        self._last_intent = 'exploratory'

    # ═══════════════════════════════════════════════════════════════
    # ★ P0-1: PRISM 零 LLM 查询意图分类
    # ═══════════════════════════════════════════════════════════════

    @classmethod
    def _classify_query_intent(cls, query: str) -> str:
        """
        PRISM 启发式意图分类 — 零 LLM 开销。

        42.3% 查询可通过轻量路径处理：
        - 'navigation' → 浏览模式，全部返回
        - 'factual' → 事实性查询，精确/B25 即可
        - 'exploratory' → 探索性问题，走全管道

        PRISM 原文证实：意图路由不损失精度，且省 13× token。
        """
        if not query or not query.strip():
            return 'navigation'
        q = query.lower().strip()

        # 探索性问题 → 全管道
        for marker in cls._EXPLORATORY_MARKERS:
            if marker in q:
                return 'exploratory'

        # 短查询 + 具体名词 → 事实性查询
        if len(q) <= 15:
            return 'factual'

        # 默认走探索（全管道，不遗漏）
        return 'exploratory'

    # ═══════════════════════════════════════════════════════════════
    # ★ P1-1: MIRIX 主动检索 — 提取检索主题
    # ═══════════════════════════════════════════════════════════════

    @classmethod
    def _extract_retrieval_topic(cls, query: str) -> dict:
        """
        MIRIX 启发：从查询中提取检索主题/实体/预期事实类型。

        零 LLM 开销的启发式提取：
        - subject: 猜测查询的核心主体（第一个名词性短语）
        - expected_type: 猜测用户想要什么类型的事实
        - entities: 提取的关键实体列表

        Returns:
            {"subject": "", "expected_type": "", "entities": []}
        """
        if not query or not query.strip():
            return {"subject": "", "expected_type": "", "entities": []}

        q = query.strip()
        result: dict[str, Any] = {"subject": "", "expected_type": "", "entities": []}

        # 1. 提取预期事实类型
        import re
        for ftype, markers in cls._TYPE_KEYWORDS.items():
            for m in markers:
                if m in q:
                    result["expected_type"] = ftype
                    break
            if result["expected_type"]:
                break

        # 2. 提取实体：去停用词后提取双字词
        remove_words = {'什么', '哪些', '如何', '怎么', '有没有', '是否', '为什么',
                        '怎样', '谁', '哪', '吗', '吧', '呢', '的', '了', '是', '有',
                        '喜欢', '知道', '告诉', '请问', '这个', '那个', '一个',
                        '可以', '需要', '想要', '能够'}
        # 用双字组（bigram）提取实体候选
        import re
        chinese = re.findall(r'[一-鿿]+', q)
        entities_set = set()
        for seg in chinese:
            # 双字组
            for i in range(len(seg) - 1):
                bigram = seg[i:i+2]
                if bigram not in remove_words and len(bigram) >= 2:
                    entities_set.add(bigram)
            # 也保留完整词（2-4字）
            if 2 <= len(seg) <= 4 and seg not in remove_words:
                entities_set.add(seg)
        result["entities"] = list(entities_set)

        # 3. 猜测主体：entities 中最可能作为 subject 的那个
        if result["entities"]:
            # 通常第一个实体是主体
            result["subject"] = result["entities"][0]

        return result

    # ═══════════════════════════════════════════════════════════════
    # ★ P1-2: LiCoMemory Weibull 时间衰减
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _weibull_staleness(days: float, half_life_days: float,
                           shape_k: float = 1.5) -> float:
        """
        Weibull 分布计算过期度（替代指数衰减）。

        LiCoMemory 证明：Weibull 比指数更符合真实遗忘曲线。
        - k=1.5: 初始慢降（近期记忆保持好）+ 后期陡降（过期记忆快速衰减）
        - λ = half_life / (ln(2))^(1/k): 保证在 half_life 天时衰减到 50%

        Returns: 0~1 的过期度（0=最新，1=完全过期）
        """
        if days <= 0:
            return 0.0
        # 调整 λ 使得在 days=half_life_days 时 Weibull CDF = 0.5
        import math
        lam = half_life_days / (math.log(2) ** (1.0 / shape_k))
        # Weibull CDF: F(d) = 1 - exp(-(d/λ)^k)  → 0~1
        cdf = 1.0 - math.exp(-((days / lam) ** shape_k))
        return min(1.0, cdf)

    # ═══════════════════════════════════════════════════════════════
    # ★ P0-3: Mem0 语义缓存查询相似度
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _query_similarity(q1: str, q2: str) -> float:
        """字符级 n-gram (1-4) 重叠度，用于语义缓存匹配。"""
        if not q1 or not q2:
            return 0.0
        q1 = q1.lower().strip()
        q2 = q2.lower().strip()
        if q1 == q2:
            return 1.0
        grams1, grams2 = set(), set()
        for n in range(1, min(5, max(len(q1), len(q2)) + 1)):
            for i in range(len(q1) - n + 1):
                grams1.add(q1[i:i + n])
            for i in range(len(q2) - n + 1):
                grams2.add(q2[i:i + n])
        if not grams1 or not grams2:
            return 0.0
        intersection = grams1 & grams2
        union = grams1 | grams2
        return len(intersection) / max(len(union), 1)

    # ═══════════════════════════════════════════════════════════════
    # recall 主入口（v0.12 增强版）
    # ═══════════════════════════════════════════════════════════════

    def recall(self, query: str, agent_id: str = "default",
               context: dict | None = None, top_k: int = 10) -> list[FactTriple]:
        """
        四级路由召回（v0.12 意图感知）。

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

        # ★ P1-1: MIRIX 主动检索 — 提取检索主题
        retrieval_topic = self._extract_retrieval_topic(query)
        ctx["retrieval_topic"] = retrieval_topic
        if retrieval_topic["subject"]:
            logger.debug(f"Active retrieval: subject={retrieval_topic['subject']!r} "
                         f"type={retrieval_topic['expected_type']!r}")

        # ★ STM 事实 ID 集合（用于排序加分）
        stm_fact_ids: set[str] = set()
        if hasattr(self.fn, '_get_stm_facts'):
            for sf in self.fn._get_stm_facts(agent_id):
                stm_fact_ids.add(sf.fact_id)

        # ── P0: 查询意图分类（0 token）──
        intent = self._classify_query_intent(q)
        self._last_intent = intent
        self._stats[f"intent_{intent}"] = self._stats.get(f"intent_{intent}", 0) + 1

        # ── 空查询 → 返回所有事实（navigation 浏览模式）──
        if intent == 'navigation':
            # 包含 STM 中的事实
            stm_facts = self.fn._get_stm_facts(agent_id) if hasattr(self.fn, '_get_stm_facts') else []
            for fact in stm_facts:
                if fact.fact_id not in seen:
                    results.append(fact)
                    seen.add(fact.fact_id)
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
                return self._rank_and_trim(results, top_k, query, session_id, stm_ids=stm_fact_ids)
            return []

        # ── 语义缓存命中（P0-3: Mem0 语义缓存）──
        semantic_hit = None
        if hasattr(self.fn, '_semantic_cache_get'):
            semantic_hit = self.fn._semantic_cache_get(q)
        if semantic_hit is not None and intent == 'factual':
            self._stats["l0_hits"] += 1
            logger.debug(f"Semantic cache hit for '{query}' ({len(semantic_hit)} facts)")
            return semantic_hit

        # ── L0: Cache Check (0 token, <1ms) ──
        cache_hits = self._l0_cache_check(q, agent_id)
        for f in cache_hits:
            if f.fact_id not in seen:
                results.append(f)
                seen.add(f.fact_id)

        if results:
            self._stats["l0_hits"] += 1
            if hasattr(self.fn, '_semantic_cache_put') and intent == 'factual':
                self.fn._semantic_cache_put(q, results[:top_k])
            logger.debug(f"L0 cache hit: {len(results)} facts for '{query}'")
            return self._rank_and_trim(results, top_k, query, session_id, stm_ids=stm_fact_ids)

        # ── L1: 精确三元组匹配 ──
        l1_hits = self._l1_exact_match(q, agent_id)
        for f in l1_hits:
            if f.fact_id not in seen:
                results.append(f)
                seen.add(f.fact_id)

        if results:
            self._stats["l1_hits"] += 1
            if hasattr(self.fn, '_semantic_cache_put') and intent == 'factual':
                self.fn._semantic_cache_put(q, results[:top_k])
            logger.debug(f"L1 exact match: {len(results)} facts for '{query}'")
            return self._rank_and_trim(results, top_k, query, session_id, stm_ids=stm_fact_ids)

        # ── L1.5: BM25 关键词模糊匹配 ──
        bm25_hits = self._bm25_retrieve(q, agent_id, top_k)
        for f in bm25_hits:
            if f.fact_id not in seen:
                results.append(f)
                seen.add(f.fact_id)

        if results and intent == 'factual':
            # ★ factual 到此为止：不走 L2/L3（省向量搜索+语义计算）
            self._stats["l1_hits"] += 1
            if hasattr(self.fn, '_semantic_cache_put'):
                self.fn._semantic_cache_put(q, results[:top_k])
            return self._rank_and_trim(results, top_k, query, session_id, stm_ids=stm_fact_ids)

        if results:
            logger.debug(f"L1.5 BM25: {len(results)} facts for '{query}'")
            return self._rank_and_trim(results, top_k, query, session_id, stm_ids=stm_fact_ids)

        # ── L2: 语义扩展（仅 exploratory 触发）──
        l2_hits = self._l2_semantic_expand(q, agent_id, context)
        for f in l2_hits:
            if f.fact_id not in seen:
                results.append(f)
                seen.add(f.fact_id)

        # ── L3: 向量搜索 (pgvector，仅 exploratory) ──
        l2_only = bool(l2_hits)
        if len(results) < top_k:
            l3_hits = self.fn.recall_vector(query, agent_id, top_k)
            for f in l3_hits:
                if f.fact_id not in seen:
                    results.append(f)
                    seen.add(f.fact_id)

        if results:
            if not l2_only:
                self._stats["l3_hits"] += 1
            else:
                self._stats["l2_hits"] += 1
        else:
            self._stats["l3_fallsback"] += 1
            logger.debug(f"All miss for '{query}' (intent={intent})")

        return self._rank_and_trim(results, top_k, query, session_id, stm_ids=stm_fact_ids)

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
        """L2: 语义扩展 — tag/类型/关联事实（v0.13 主动检索增强）"""
        results = []
        seen: set[str] = set()
        ctx = context or {}
        topic = ctx.get("retrieval_topic", {})

        q = query.lower().strip()
        import difflib

        # ★ 主动检索：优先找匹配 subject + expected_type 的事实
        expected_type = topic.get("expected_type", "")
        topic_subject = topic.get("subject", "")
        topic_entities = topic.get("entities", [])

        for fact in self.fn._get_cached_facts(agent_id):
            if fact.fact_id in seen:
                continue
            text = f"{fact.subject} {fact.predicate} {fact.object}".lower()

            # 主动检索匹配：预期类型匹配优先
            type_match = expected_type and fact.fact_type == expected_type
            subject_match = topic_subject and topic_subject in fact.subject

            if type_match and subject_match:
                # 类型+主体都匹配 → 直接加入，不依赖相似度
                results.append(fact)
                seen.add(fact.fact_id)
                self._stats["active_retrieval_hits"] = \
                    self._stats.get("active_retrieval_hits", 0) + 1
                continue

            # 实体匹配
            entity_match = any(e in fact.subject or e in fact.object
                             for e in topic_entities) if topic_entities else False
            if entity_match:
                results.append(fact)
                seen.add(fact.fact_id)
                continue

            # 常规 difflib 模糊匹配
            ratio = difflib.SequenceMatcher(None, q, text).ratio()
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
                        query: str = "", session_id: str = "",
                        stm_ids: set | None = None) -> list[FactTriple]:
        """
        ⭐ 六维上下文感知排序（6-Dimension Scoring v0.12）。

        受 ERINYS 6-signal + NaLog Blend Score 启发：
        评分 = 置信度(30%) + 重要性(10%) + 新鲜度(15%)
               + 相关度(15%) + 证据溯源(10%) + 过期度惩罚(10%)
               + 矛盾状态(5%) + 同会话加分(5%)

        ★ v0.12 新增：
        - 🆕 STM 加分(5%) — 短期缓冲区的事实优先（AMP 启发）
        - 🆕 意图加权 — factual 查询加重精确匹配

        ★ 六维 vs 旧版三维：
        - 新增「过期度惩罚」：半衰期越远，扣分越多 → 避免用过期记忆做决策
        - 新增「矛盾状态」：有 pending 矛盾的记忆降权 → 避免矛盾信息污染上下文
        - 新增「证据溯源」：用户陈述 > 工具结果 > AI 推断 → 不同来源不同权重
        - 改进「相关度」：从简单 difflib 改为 difflib + 关键词命中多级评分
        """
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        q = query.lower().strip() if query else ""
        stm_ids = stm_ids or set()

        # ── 1. 半衰期配置（与 fact_network._apply_decay 保持一致）──
        _HALF_LIFE_MAP = {
            "abstraction": 60.0,
            "core": 90.0,
        }

        def _calc_staleness(f: FactTriple, now_dt: datetime) -> float:
            """
            ★ v0.13 Weibull 分布计算过期度 (0~1, 越高越过期)。

            LiCoMemory 验证：Weibull (k=1.5) 比指数衰减更符合真实遗忘曲线。
            - 近期（<半衰期/2）：衰减慢，信息保持好
            - 中期（≈半衰期）：加速衰减
            - 远期（>半衰期×2）：快速遗忘
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

            # Weibull CDF: F(d) = 1 - exp(-(d/λ)^k)
            # λ 调整使得在 d=half_life 时 F=0.5
            import math
            shape_k = 1.5
            lam = hl / (math.log(2) ** (1.0 / shape_k))
            stigma = 1.0 - math.exp(-((days / lam) ** shape_k))
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
            source_type = "unknown"
            if f.evidence and f.evidence[0].source:
                ev_source = f.evidence[0].source
                # 直接匹配已知来源类型
                if ev_source in source_weights:
                    source_type = ev_source
                # 否则从 evidence 内容推断（evidence 里的 statement 可能包含线索）
                else:
                    for known in source_weights:
                        if known in ev_source.lower():
                            source_type = known
                            break
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

            # ★ v0.12 ⑨ STM 缓冲区加分 (5%) — 刚记住的事实优先（AMP 启发）
            if f.fact_id in stm_ids:
                s += 0.05

            return s

        # ★ P1-3: 知识库过滤 — 普通召回不返回 credential 类型的事实
        facts = [f for f in facts if f.fact_type != "credential"]

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
            "intent_factual": s.get("intent_factual", 0),
            "intent_exploratory": s.get("intent_exploratory", 0),
            "intent_navigation": s.get("intent_navigation", 0),
            "intent_factual_pct": f"{s.get('intent_factual', 0)/total*100:.0f}%",
            "active_retrieval_hits": s.get("active_retrieval_hits", 0),
        }
