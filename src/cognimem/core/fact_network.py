"""
CogniMem 核心引擎 — 事实网络

核心能力：
1. 三元组存储与去重
2. 置信度贝叶斯更新
3. 矛盾检测与处理
4. 重要性评分与自适应 TTL
5. 主动遗忘与降级
"""

import logging
import json
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any
from collections import defaultdict, OrderedDict
from contextlib import contextmanager

from .models import FactTriple, Contradiction, Episode, EvidenceItem

logger = logging.getLogger(__name__)


# ── 默认配置 ──
DEFAULT_CONF_INITIAL = 0.6
DEFAULT_CONF_CONFIRM_BOOST = 0.1
DEFAULT_CONF_DENY_PENALTY = 0.2          # 直接否定（喜欢 vs 不喜欢）
DEFAULT_CONF_CONFLICT_PENALTY = 0.08     # 间接冲突（冰美式 vs 热美式）
DEFAULT_CONF_CORE_THRESHOLD = 0.9
DEFAULT_CONF_UNRELIABLE_THRESHOLD = 0.2
DEFAULT_CONF_DELETE_THRESHOLD = 0.1
DEFAULT_IMPORTANCE_BOOST_ON_ACCESS = 0.05
DEFAULT_IMPORTANCE_DECAY_PER_DAY = 0.02

# ── 中性谓词：不同对象不算矛盾 ⭐ 模块级常量，避免循环内重复创建 ──
_NEUTRAL_PREDICATES = {"请求", "请求了", "说", "说了", "问", "问了",
                      "提", "提了", "做", "做了", "用", "用了",
                      "跟", "跟了", "给", "给了",
                      # Agent Action Facts — 不同行为不是矛盾
                      "记住了", "创建了文件", "修改了文件",
                      "搜索了", "访问了网页", "执行了命令",
                      "读取了文件", "浏览了目录", "回想记忆"}

# ── 来源权重 ──
SOURCE_WEIGHTS = {
    "user_statement": 1.0,       # 用户明确陈述
    "user_confirmation": 1.0,    # 用户确认/纠正
    "user_challenge": 1.0,       # 用户质疑
    "agent_inference": 0.4,      # Agent 推理结论
    "tool_result": 0.7,          # 工具返回值
    "third_party": 0.2,          # 第三方引用
    "merged_evidence": 0.5,      # 重复陈述（合并）
    "consolidated": 0.3,         # 睡眠期自动提升
    "system": 1.0,               # 系统内部操作
}

# 冷启动: 前 N 次确认的权重倍数
COLD_START_CONFIRM_COUNT = 3
COLD_START_BOOST_MULTIPLIER = 2.0


class FactNetwork:
    """
    事实网络引擎 (CogniMem 核心)

    管理所有事实三元组，维护:
    - 事实图 (subject → predicate → object)
    - 置信度动态更新
    - 矛盾关系
    - 访问模式跟踪
    """

    def __init__(self, db_adapter=None, config: dict | None = None):
        self.db = db_adapter  # CockroachDB 适配器
        self._config = config or {}

        # 🔒 全局锁 — 所有缓存操作、consolidation 状态共享这把锁
        self._lock = threading.RLock()

        # 内存缓存 (工作记忆 L0) — 扩大 10 倍 + TTL 过期
        self._lru_cache: OrderedDict[str, FactTriple] = OrderedDict()
        self._cache_max = 1000
        self._cache_ttl = 300          # 5 分钟自动过期
        self._cache_timestamps: dict[str, float] = {}

        # 统计缓存 (30 秒 TTL，避免每次 stats 都遍历全库)
        self._stats_cache: dict[str, dict] = {}
        self._stats_cache_ttl = 30

        # 查询缓存 (60 秒 TTL，相同查询秒回)
        self._query_cache: dict[str, tuple[list, float]] = {}
        self._query_cache_ttl = 60

        # ⭐ 自动整合（Sleep-Cycle Consolidation）
        self._last_consolidation: dict[str, float] = {}  # agent_id → timestamp
        self._auto_consolidate_interval = 1800  # 30 分钟空闲后自动整合
        self._consolidating: set[str] = set()  # 正在整合中的 agent

        # 配置
        self.conf_initial = self._config.get("confidence_initial", DEFAULT_CONF_INITIAL)
        self.conf_confirm_boost = self._config.get("confidence_confirm_boost", DEFAULT_CONF_CONFIRM_BOOST)
        self.conf_deny_penalty = self._config.get("confidence_deny_penalty", DEFAULT_CONF_DENY_PENALTY)
        self.conf_conflict_penalty = self._config.get("confidence_conflict_penalty", DEFAULT_CONF_CONFLICT_PENALTY)
        self.conf_core_threshold = self._config.get("confidence_core_threshold", DEFAULT_CONF_CORE_THRESHOLD)
        self.conf_unreliable_threshold = self._config.get("confidence_unreliable_threshold", DEFAULT_CONF_UNRELIABLE_THRESHOLD)
        self.conf_delete_threshold = self._config.get("confidence_delete_threshold", DEFAULT_CONF_DELETE_THRESHOLD)

    # ═══════════════════════════════════════════
    # 写入
    # ═══════════════════════════════════════════

    def add_fact(self, fact: FactTriple, source_type: str = "user_statement") -> dict:
        """
        添加/更新事实。

        1. 去重: 相同三元组 (agent|subject|predicate|object) → 合并
        2. 矛盾检测: 检查与现有事实的冲突
        3. 存入缓存

        Args:
            fact: 事实三元组
            source_type: 来源类型 → 影响置信度权重
                         user_statement | agent_inference | tool_result | third_party

        Returns: {"status": "created"|"merged"|"contradiction_detected", "fact": fact}
        """
        with self._lock:
            existing = self._find_existing(fact.triple_key)

            if existing:
                # 已存在 → 合并 (增加证据, 提升置信度)
                return self._merge_facts(existing, fact, source_type)

            # 矛盾检测 (遍历已有事实)
            contradictions = self._detect_contradictions(fact)

            # 新事实 — 先写缓存，再写 DB（DB 失败则回滚缓存）
            self._cache_put(fact)
            db_ok = True
            if self.db:
                try:
                    self.db.save_fact(fact)
                except Exception as e:
                    # DB 失败 → 回滚缓存
                    self._lru_cache.pop(fact.fact_id, None)
                    self._cache_timestamps.pop(fact.fact_id, None)
                    logger.error("save_fact failed, cache reverted: %s", e)
                    raise

            if contradictions:
                # 矛盾等级: L1=deny(降双方) | L2=conflict(仅降新) | L3=context(仅标记)
                for c in contradictions:
                    fact.contradictions.append(c.fact_a_id)

                    if c.contradiction_type == "deny":
                        # L1: 直接否定 → 惩罚 0.2，降双方
                        existing_fact = self._get_fact(c.fact_a_id)
                        base_conf = existing_fact.confidence if existing_fact else self.conf_initial
                        fact.confidence = max(base_conf - self.conf_deny_penalty, 0.1)
                        if existing_fact:
                            from copy import deepcopy
                            old_ex = deepcopy(existing_fact)
                            self._update_confidence(
                                existing_fact, -self.conf_deny_penalty * 0.5,
                                f"contradicted_by_{fact.fact_id[:8]}",
                                source_type="system"
                            )
                            if self.db and old_ex.confidence != existing_fact.confidence:
                                self.db.save_version(
                                    existing_fact.fact_id, existing_fact.agent_id,
                                    old_ex, existing_fact,
                                    f"contradicted_by_{fact.fact_id[:8]}"
                                )
                    elif c.contradiction_type == "conflict":
                        # L2: 间接冲突 → 以已有事实为基准降 conf，不降对方
                        existing_fact = self._get_fact(c.fact_a_id)
                        base_conf = existing_fact.confidence if existing_fact else self.conf_initial
                        fact.confidence = max(base_conf - self.conf_conflict_penalty, 0.1)
                    # L3 context: 上下文变化 → 不降置信度，仅标记

                    if self.db:
                        self.db.save_contradiction(c)

                return {"status": "contradiction_detected", "fact": fact, "contradictions": contradictions}

            return {"status": "created", "fact": fact}

    def batch_add(self, facts: list[FactTriple], agent_id: str = "default",
                   source_type: str = "user_statement") -> list[dict]:
        """批量添加事实，返回每个结果"""
        return [self.add_fact(f, source_type) for f in facts]

    # ═══════════════════════════════════════════
    # 召回
    # ═══════════════════════════════════════════

    def recall(self, query: str, agent_id: str = "default", top_k: int = 10) -> list[FactTriple]:
        """
        三级路由召回（含查询缓存）：

        1. 查询缓存命中 → 0 DB 开销
        2. L0 Cache check — TTL 感知的内存缓存
        3. 精确匹配 — subject 或 object 匹配
        4. 语义扩展 — tag/类型匹配

        Returns: 按置信度排序的事实列表
        """
        with self._lock:
            cache_key = f"recall:{agent_id}:{query}:{top_k}"
            cached = self._get_query_cached(cache_key)
            if cached is not None:
                return cached

            results: list[FactTriple] = []
            seen: set[str] = set()
            q = query.lower().strip()

            # L1: Cache check (TTL 感知)
            self._evict_expired()
            for fact in self._lru_cache.values():
                if fact.agent_id != agent_id:
                    continue
                if self._match_query(fact, q):
                    if fact.fact_id not in seen:
                        results.append(fact)
                        seen.add(fact.fact_id)

            # L2: 精确匹配 (从数据库)
            if self.db and len(results) < top_k:
                db_facts = self.db.search_facts(
                    agent_id=agent_id,
                    subject=q, object=q, tag=q,
                    limit=top_k * 2
                )
                for f in db_facts:
                    if f.fact_id not in seen:
                        results.append(f)
                        seen.add(f.fact_id)
                        self._cache_put(f)  # 顺便预热缓存

            # 更新访问时间 + 重要性
            for f in results:
                self._record_access(f)

        # 按置信度排序
        results.sort(key=lambda f: f.confidence, reverse=True)
        results = results[:top_k]

        # ⭐ 抽象展开：若结果中有抽象事实，附带其子事实
        expanded_ids: set[str] = set(f.fact_id for f in results)
        for f in list(results):
            if f.encoding_level == "abstraction" and f.connected_facts:
                for child_id in f.connected_facts:
                    if child_id not in expanded_ids:
                        child = self._get_fact(child_id)
                        if child and child.encoding_level == "abstracted":
                            results.append(child)
                            expanded_ids.add(child_id)

        # 写入查询缓存
        self._set_query_cached(cache_key, results)
        return results

    def recall_vector(self, query: str, agent_id: str = "default",
                       top_k: int = 10) -> list[FactTriple]:
        """L3: 向量相似度搜索"""
        if not self.db:
            return []
        return self.db.search_facts_vector(agent_id, query, top_k)

    def recall_by_triple(self, subject: str, predicate: str | None = None,
                         agent_id: str = "default") -> list[FactTriple]:
        """精确三元组查询: 找出所有 (subject, predicate, ?) 的事实"""
        with self._lock:
            results = []
            for fact in self._get_agent_facts(agent_id):
                if fact.subject == subject:
                    if predicate is None or fact.predicate == predicate:
                        results.append(fact)
            results.sort(key=lambda f: f.confidence, reverse=True)
            return results

    def get_beliefs(self, agent_id: str = "default",
                    min_confidence: float = 0.9) -> list[FactTriple]:
        """获取核心信念 (置信度 > threshold)"""
        with self._lock:
            return [
                f for f in self._get_agent_facts(agent_id)
                if f.confidence >= min_confidence
            ]

    def get_contradictions(self, agent_id: str = "default",
                           resolution: str = "pending") -> list[Contradiction]:
        """获取待处理的矛盾（优先查 DB，否则走内存）"""
        if self.db:
            return self.db.get_contradictions(agent_id, resolution)
        return []

    def get_versions(self, fact_id: str) -> list[dict]:
        """获取事实的版本历史"""
        if self.db:
            return self.db.get_versions(fact_id)
        return []

    # ═══════════════════════════════════════════
    # 置信度管理
    # ═══════════════════════════════════════════

    def confirm_fact(self, fact_id: str, source: str = "user_confirmation") -> FactTriple | None:
        """确认事实 → 提升置信度（来源权重 = user_confirmation 1.0）"""
        with self._lock:
            fact = self._get_fact(fact_id)
            if not fact:
                return None
            from copy import deepcopy
            old = deepcopy(fact)
            self._update_confidence(fact, self.conf_confirm_boost, source,
                                    source_type="user_confirmation")
            fact.last_confirmed = datetime.now(timezone.utc).isoformat()
            if self.db and old.confidence != fact.confidence:
                self.db.save_version(fact.fact_id, fact.agent_id,
                                     old, fact, "confirmed")
            return fact

    def challenge_fact(self, fact_id: str, source: str = "user_challenge") -> FactTriple | None:
        """质疑事实 → 降低置信度（来源权重 = user_challenge 1.0）"""
        with self._lock:
            fact = self._get_fact(fact_id)
            if not fact:
                return None
            from copy import deepcopy
            old = deepcopy(fact)
            self._update_confidence(fact, -self.conf_deny_penalty, source,
                                    source_type="user_challenge")
            if self.db and old.confidence != fact.confidence:
                self.db.save_version(fact.fact_id, fact.agent_id,
                                     old, fact, "challenged")
            return fact

    # ═══════════════════════════════════════════
    # 矛盾检测 (核心创新)
    # ═══════════════════════════════════════════

    def _detect_contradictions(self, fact: FactTriple) -> list[Contradiction]:
        """
        检测新事实与已有事实的矛盾。

        矛盾等级:
          L1 deny    — 直接否定（喜欢 vs 不喜欢/不爱/讨厌）
          L2 conflict — 间接冲突（冰美式 vs 热美式）
          L3 context  — 上下文变化（截止日期从周五 → 周三）
        """
        context_predicates = {
            "是", "在", "去了", "住在", "位于", "来自",
            "开始", "结束", "截止", "推迟", "提前", "延长",
            "修改", "更新", "改成", "改为",
            "涨了", "跌了", "变成", "变为",
        }
        # 也匹配包含上下文关键词的复合谓词（如"截止日期是"→含"是"）
        context_keywords = {"是", "在", "截至", "截止", "位于", "开始", "结束"}
        is_context = (
            fact.predicate in context_predicates
            or any(kw in fact.predicate for kw in context_keywords)
        )

        # ⭐ 全面的否定谓词列表（来源: contradiction.py _NEGATION_MAP + 补充）
        neg_predicates = {
            "不喜欢", "不爱", "不想", "不要", "不是", "不会", "没有",
            "不需要", "不应该", "不愿意", "不能", "不可以",
            "拒绝", "讨厌", "反感", "厌恶",
            "未完成", "还没", "未曾",
        }

        # ⭐ Subject 归一化：user/用户/你 → 统一为"你"
        def _norm_subj(s: str) -> str:
            if s.lower() in ("user", "用户", "你"):
                return "你"
            return s
        fact_subj = _norm_subj(fact.subject)

        contradictions = []
        for existing in self._get_agent_facts(fact.agent_id):
            if existing.fact_id == fact.fact_id:
                continue
            existing_subj = _norm_subj(existing.subject)

            # 中性谓词：不同对象不算矛盾（"请求了 A" vs "请求了 B"完全正常）
            if existing.predicate in _NEUTRAL_PREDICATES or fact.predicate in _NEUTRAL_PREDICATES:
                continue

            # 规则1: 相同 subject+predicate, 不同 object
            if (existing_subj == fact_subj
                    and existing.predicate == fact.predicate
                    and existing.object != fact.object):
                # ⭐ 偏好类过滤：喜欢不同类别的东西（如咖啡 vs 火锅）不算矛盾
                is_preference = (
                    existing.fact_type == "preference"
                    or fact.fact_type == "preference"
                )
                if is_preference:
                    # ⭐ 偏好规则强化：喜欢不同东西不是矛盾，只有否定才矛盾
                    # 检查共享标签：同类别才标记为低级 conflict
                    existing_tags = set(existing.context_tags)
                    new_tags = set(fact.context_tags)
                    shared_tags = existing_tags & new_tags
                    if not shared_tags:
                        # 不同类别（咖啡 vs 美食）→ 跳过，不算矛盾
                        continue
                    # 同类别（冰美式 vs 热美式）→ 标低优先级 conflict
                    # 但如果是纯"喜欢"类偏好，不同物品不该算矛盾（人可以喜欢多样）
                    if "喜欢" in fact.predicate or "爱" in fact.predicate:
                        # 喜欢多个不同物品，完全正常 → 跳过
                        continue
                    ctype = "conflict"
                else:
                    ctype = "context" if is_context else "conflict"
                contradictions.append(Contradiction(
                    fact_a_id=existing.fact_id,
                    fact_b_id=fact.fact_id,
                    agent_id=fact.agent_id,
                    contradiction_type=ctype,
                    description=f"'{fact.subject}'的'{fact.predicate}': "
                                f"'{existing.object}' → '{fact.object}'"
                ))

            # 规则2: 否定关系检测 (双向) → L1 直接否定
            # 通用「不/没/未」前缀检测 + 已知否定谓词列表
            def _is_negated(pred: str) -> bool:
                if pred in neg_predicates:
                    return True
                # 通用不/没/未前缀
                if pred.startswith(("不", "没", "未")):
                    # 排除「不」开头的非否定词
                    exceptions = {"不错", "不少", "不一定", "不如", "不怎么", "不止"}
                    if pred not in exceptions:
                        return True
                return False

            def _is_positive(pred: str) -> bool:
                """判断是否肯定形式（与否定相对的正面谓词）"""
                # 已知肯定谓词
                positive_set = {
                    "喜欢", "爱", "想", "要", "是", "会", "有",
                    "需要", "应该", "愿意", "能", "可以",
                    "完成", "做好",
                }
                if pred in positive_set:
                    return True
                # 不在否定列表、不以不/没/未开头 → 视为肯定
                if not _is_negated(pred):
                    return True
                return False

            if (existing_subj == fact_subj
                    and existing.object == fact.object):
                # 方向A: 已存在是否定, 新事实是肯定 → 矛盾
                if _is_negated(existing.predicate) and _is_positive(fact.predicate):
                    contradictions.append(Contradiction(
                        fact_a_id=existing.fact_id,
                        fact_b_id=fact.fact_id,
                        agent_id=fact.agent_id,
                        contradiction_type="deny",
                        description=f"'{fact.subject}' '{fact.predicate}' '{fact.object}' "
                                    f"但之前记录 '{existing.predicate}'"
                    ))
                # 方向B: 新事实是否定, 已存在是肯定 → 矛盾
                elif _is_negated(fact.predicate) and _is_positive(existing.predicate):
                    contradictions.append(Contradiction(
                        fact_a_id=existing.fact_id,
                        fact_b_id=fact.fact_id,
                        agent_id=fact.agent_id,
                        contradiction_type="deny",
                        description=f"'{fact.subject}' '{fact.predicate}' '{fact.object}' "
                                    f"但之前记录 '{existing.predicate}'"
                    ))

        return contradictions

    # ═══════════════════════════════════════════
    # 遗忘与降级（⭐ 艾宾浩斯衰减曲线）
    # ═══════════════════════════════════════════

    def forget(self, agent_id: str = "default") -> dict:
        """
        艾宾浩斯主动遗忘引擎。

        不是粗暴删除，而是模拟人脑记忆衰减：
        - 每次访问 = 一次"复习" → 衰减重置
        - 核心信念半衰期 90 天，几乎不衰减
        - 低置信度事实半衰期 7 天，快速衰减
        - 置信度 < 0.1 → 删除

        Returns: {"decayed": N, "deleted": N, "core_preserved": N}
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            stats = {"decayed": 0, "deleted": 0, "core_preserved": 0}

            for fact in self._get_agent_facts(agent_id):
                old_conf = fact.confidence

                # 核心信念 → 跳过（几乎不衰减）
                if fact.is_core_belief or fact.encoding_level == "core":
                    stats["core_preserved"] += 1
                    continue

                # ⭐ 新事实免疫期：创建/访问 1 小时内不衰减，给新记忆巩固时间
                accessed = datetime.fromisoformat(fact.accessed_at)
                if (now - accessed).total_seconds() < 3600:
                    continue

                # 计算衰减
                new_conf = self._apply_decay(fact, now)

                if new_conf < self.conf_delete_threshold:
                    # 置信度过低 → 删除
                    self._delete_fact(fact.fact_id)
                    stats["deleted"] += 1
                    continue

                if new_conf < old_conf - 0.01:
                    # 有实质衰减 → 更新
                    fact.confidence = new_conf
                    fact.encoding_level = "compressed" if new_conf < 0.3 else fact.encoding_level
                    self._cache_put(fact)
                    if self.db:
                        self.db.update_fact(fact)
                        self.db.log_confidence_change(
                            fact.fact_id, fact.agent_id,
                            old_conf, new_conf, "ebbinghaus_decay"
                        )
                    stats["decayed"] += 1

                # 低重要性 + 长期未访问 → 压缩
                accessed = datetime.fromisoformat(fact.accessed_at)
                days_since_access = (now - accessed).days
                if (fact.importance < 0.3
                        and days_since_access > 14
                        and fact.encoding_level == "raw"):
                    fact.encoding_level = "compressed"
                    fact.evidence = fact.evidence[:1] if fact.evidence else []
                    if self.db:
                        self.db.update_fact(fact)
                    stats.setdefault("compressed", 0)
                    stats["compressed"] += 1

        return stats

    def _apply_decay(self, fact: FactTriple, now) -> float:
        """
        艾宾浩斯衰减公式。

        半衰期因记忆强度而异：
        - 抽象概念: 60 天
        - 高置信 (>0.6): 30 天
        - 普通: 14 天
        - 低置信 (<0.3): 7 天

        公式: new_conf = conf × 0.5^(days/halflife)
        """
        accessed = datetime.fromisoformat(fact.accessed_at)
        days = (now - accessed).days
        if days <= 0:
            return fact.confidence

        # 半衰期（天）
        if fact.encoding_level == "abstraction":
            half_life = 60.0
        elif fact.confidence >= 0.6:
            half_life = 30.0
        elif fact.confidence >= 0.3:
            half_life = 14.0
        else:
            half_life = 7.0

        # 访问频率修正：访问越多，衰减越慢
        if fact.access_count > 10:
            half_life *= 1.5
        elif fact.access_count > 5:
            half_life *= 1.2

        decay = 0.5 ** (days / half_life)
        return max(0.0, fact.confidence * decay)

    # ═══════════════════════════════════════════
    # 睡眠期整合
    # ═══════════════════════════════════════════

    def consolidate(self, agent_id: str = "default",
                    llm_extractor=None) -> dict:
        """
        记忆整合 (模拟睡眠期整理)

        执行顺序（重要！）：
        1. ⭐ 记忆抽象化 — 先找模式，从碎片归纳高层知识
        2. 重复合并 — 去重（LLM 泛化产生的相同三元组）
        3. 可信模式提升 — 高频访问 → 信念
        4. 遗忘 — 清理低置信度事实
        """
        # ⭐ 整个 consolidate 在一把锁下执行，避免并发读写不一致
        with self._lock:
            results = {"merged": 0, "promoted": 0, "abstracted": 0,
                       "contradictions_resolved": 0}

            # 1. ⭐ 记忆抽象化（先执行，在合并前找到模式）
            try:
                abstract_result = self._abstract_memories(agent_id, llm_extractor)
                results["abstracted"] = abstract_result.get("abstracted", 0)
            except Exception as e:
                logger.error(f"❌ Abstraction step failed: {e}", exc_info=True)
                results["abstracted"] = 0

            # 2. 重复合并：完全相同三元组 → 去重
            try:
                groups = defaultdict(list)
                for f in self._get_agent_facts(agent_id):
                    if f.encoding_level in ("core", "abstraction"):
                        continue
                    groups[(f.subject, f.predicate, f.object)].append(f)

                for key, group in groups.items():
                    if len(group) > 1:
                        best = max(group, key=lambda f: f.confidence)
                        for f in group:
                            if f.fact_id != best.fact_id:
                                best.evidence.extend(f.evidence)
                                best.access_count += f.access_count
                                best.importance = max(best.importance, f.importance)
                                self._delete_fact(f.fact_id)
                                results["merged"] += 1
            except Exception as e:
                logger.error(f"❌ Merge step failed: {e}", exc_info=True)

            # 3. 高频访问 → 提升为信念
            try:
                for f in self._get_agent_facts(agent_id):
                    if f.access_count >= 10 and f.confidence >= 0.7:
                        old_conf = f.confidence
                        boost = min(0.05 * (f.access_count // 5), 0.15)
                        self._update_confidence(f, boost, "consolidated_pattern",
                                                source_type="consolidated")
                        if self.db and old_conf != f.confidence:
                            from copy import deepcopy
                            old = deepcopy(f)
                            self.db.save_version(f.fact_id, f.agent_id,
                                                 old, f, "consolidated")
                        results["promoted"] += 1
            except Exception as e:
                logger.error(f"❌ Promotion step failed: {e}", exc_info=True)

            # 4. ⭐ 矛盾解决：自动清理可解决的矛盾
            try:
                pending = self.get_contradictions(agent_id)
                resolved = 0
                for c in pending:
                    fa = self._get_fact(c.fact_a_id)
                    fb = self._get_fact(c.fact_b_id)
                    if not fa or not fb:
                        continue
                    # 偏好类矛盾：喜欢不同东西不算矛盾 → 直接删除矛盾记录
                    if "喜欢" in fa.predicate or "喜欢" in fb.predicate:
                        if self.db:
                            with self.db._plain_cursor_ctx() as cur:
                                cur.execute("DELETE FROM contradictions WHERE id = %s", (c.id,))
                        resolved += 1
                        continue
                    # deny 类：低置信度那个删掉
                    if c.contradiction_type == "deny":
                        to_remove = fa if fa.confidence < fb.confidence else fb
                        self._delete_fact(to_remove.fact_id)
                        resolved += 1
                        continue
                    # conflict 类：伪矛盾，只删除矛盾记录，两个事实都保留
                    if c.contradiction_type == "conflict" and self.db:
                        with self.db._plain_cursor_ctx() as cur:
                            cur.execute("DELETE FROM contradictions WHERE id = %s", (c.id,))
                        resolved += 1
                        continue
                if resolved:
                    logger.info("✅ Resolved %d contradictions for '%s'", resolved, agent_id)
                results["contradictions_resolved"] = resolved
            except Exception as e:
                logger.error(f"❌ Contradiction resolution failed: {e}", exc_info=True)

            # 5. 遗忘
            try:
                forget_result = self.forget(agent_id)
                results.update(forget_result)
            except Exception as e:
                logger.error(f"❌ Forget step failed: {e}", exc_info=True)

        return results

    def _maybe_auto_consolidate(self, agent_id: str,
                                 llm_extractor=None) -> dict | None:
        """
        ⭐ 空闲自动整合（Sleep-Cycle Consolidation），异步。

        系统空闲超过 5 分钟 → 后台线程触发整合，不阻塞请求。
        同一 agent 不会同时跑两个整合。

        Returns: 立即返回 None（触发时开后台线程）
        """
        with self._lock:
            now = time.time()
            last = self._last_consolidation.get(agent_id, 0)

            if now - last < self._auto_consolidate_interval:
                return None  # 还没到时间

            if agent_id in self._consolidating:
                logger.debug(f"Consolidation already in progress for '{agent_id}' — skipping")
                return None

            logger.info(
                f"🌙 Auto-consolidation triggered for '{agent_id}' "
                f"(idle: {now - last:.0f}s, async)"
            )
            self._consolidating.add(agent_id)
            self._last_consolidation[agent_id] = now

        # 后台线程异步执行（在锁外启动，避免死锁）
        def _run():
            try:
                result = self.consolidate(agent_id, llm_extractor)
                logger.info(f"✅ Consolidation done for '{agent_id}': {result}")
            except Exception as e:
                logger.error(f"❌ Consolidation failed for '{agent_id}': {e}", exc_info=True)
            finally:
                with self._lock:
                    self._consolidating.discard(agent_id)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return {"status": "async_started"}

    # ═══════════════════════════════════════════
    # ⭐ 记忆抽象化（核心创新）
    # ═══════════════════════════════════════════

    def _abstract_memories(self, agent_id: str,
                           llm_extractor=None) -> dict:
        """
        记忆抽象化：从碎片事实中归纳高层知识。

        算法：
        1. 按 (subject, predicate) 分组
        2. 组内按共享标签聚类 → 每个聚类生成一个抽象
        3. 同组 >= 2 个事实 → 尝试找到共同上层类别
        4. 创建抽象事实（置信度 = avg(子事实) × 1.1）
        5. 子事实标记为 "abstracted"，链接到抽象事实
        """
        groups = defaultdict(list)
        for f in self._get_agent_facts(agent_id):
            if f.encoding_level in ("core", "abstraction", "abstracted"):
                continue
            if f.fact_type == "observation":
                continue
            groups[(f.subject, f.predicate)].append(f)

        results = {"abstracted": 0, "groups_scanned": len(groups)}

        for (subject, predicate), group in groups.items():
            if len(group) < 2:
                continue

            # ⭐ 组内聚类：按共享标签拆分成子组
            clusters = self._cluster_by_tags(group)

            for cluster in clusters:
                if len(cluster) < 2:
                    continue

                category = self._find_category(cluster, llm_extractor)
                if not category:
                    continue

                existing_abstract = self._find_existing(
                    f"{agent_id}|{subject}|{predicate}|{category}"
                )
                if existing_abstract and existing_abstract.encoding_level == "abstraction":
                    continue

                avg_conf = sum(f.confidence for f in cluster) / len(cluster)
                child_ids = [f.fact_id for f in cluster]
                child_objects = [f.object for f in cluster[:5]]

                abstract = FactTriple(
                    subject=subject,
                    predicate=predicate,
                    object=category,
                    agent_id=agent_id,
                    fact_type="abstraction",
                    confidence=min(1.0, avg_conf * 1.1),
                    importance=max(f.importance for f in cluster),
                    encoding_level="abstraction",
                    context_tags=list(set(
                        t for f in cluster for t in f.context_tags
                    ))[:5],
                    connected_facts=child_ids,
                    evidence=[EvidenceItem(
                        source="memory_abstraction",
                        statement=f"归纳自 {len(cluster)} 条相关记忆: "
                                  f"{', '.join(child_objects)}"
                    )],
                )

                self._cache_put(abstract)
                if self.db:
                    self.db.save_fact(abstract)

                for child in cluster:
                    child.encoding_level = "abstracted"
                    if abstract.fact_id not in child.connected_facts:
                        child.connected_facts.append(abstract.fact_id)
                    self._cache_put(child)
                    if self.db:
                        self.db.update_fact(child)

                results["abstracted"] += 1
                logger.info(
                    f"🧠 Abstracted: '{subject} {predicate} {category}' "
                    f"from {len(cluster)} facts (conf={abstract.confidence:.2f})"
                )

        return results

    def _cluster_by_tags(self, group: list[FactTriple]) -> list[list[FactTriple]]:
        """按共享标签将事实聚类（连通分量算法）"""
        if len(group) <= 1:
            return [group]

        # 构建标签→事实映射
        tag_to_facts: dict[str, set[int]] = defaultdict(set)
        for i, f in enumerate(group):
            for tag in f.context_tags:
                tag_to_facts[tag].add(i)

        # 并查集合并共享标签的事实
        parent = list(range(len(group)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pa] = pb

        for indices in tag_to_facts.values():
            if len(indices) >= 2:
                it = iter(indices)
                first = next(it)
                for idx in it:
                    union(first, idx)

        # 收集聚类
        cluster_map: dict[int, list[FactTriple]] = defaultdict(list)
        for i, f in enumerate(group):
            cluster_map[find(i)].append(f)

        return list(cluster_map.values())

    def _find_category(self, group: list[FactTriple],
                       llm_extractor=None) -> str | None:
        """为一组事实找到共同上层类别（tags→LLM→词重叠）"""
        tag_counts: dict[str, int] = {}
        for f in group:
            for tag in f.context_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        shared = [(t, c) for t, c in tag_counts.items() if c >= 2]
        if shared:
            shared.sort(key=lambda x: x[1], reverse=True)
            return shared[0][0]

        if llm_extractor:
            try:
                return self._llm_categorize(group, llm_extractor)
            except Exception as e:
                logger.debug(f"LLM categorize failed: {e}")

        objects = [f.object for f in group]
        common_words = set(objects[0])
        for o in objects[1:]:
            common_words &= set(o)
        if common_words:
            return "".join(sorted(common_words)[:3])

        return None

    def _llm_categorize(self, group: list[FactTriple],
                        llm_extractor) -> str | None:
        """用 LLM 归纳一组事实的共同类别（约 50 tok）"""
        import json as _json
        objects = [f.object for f in group]
        subject = group[0].subject
        predicate = group[0].predicate

        prompt = (
            f"以下是关于「{subject}」的「{predicate}」相关条目：\n"
            + "\n".join(f"- {o}" for o in objects)
            + "\n\n这些条目属于什么共同类别？用一个简洁的中文词组回答（2-6个字）。"
            + "\n只返回 JSON: {\"category\": \"类别名\"}"
        )

        try:
            import openai
            client = openai.OpenAI(
                api_key=llm_extractor.api_key,
                base_url=llm_extractor.base_url,
            )
            r = client.chat.completions.create(
                model=llm_extractor.model,
                messages=[
                    {"role": "system", "content": "你是分类专家。只返回 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1, max_tokens=100,
                response_format={"type": "json_object"},
            )
            data = _json.loads(r.choices[0].message.content or "{}")
            cat = data.get("category", "").strip()
            return cat if cat else None
        except Exception:
            return None

    # ═══════════════════════════════════════════
    # 内部方法
    # ═══════════════════════════════════════════

    def _update_confidence(self, fact: FactTriple, delta: float, reason: str,
                           source_type: str = "user_statement"):
        """
        更新置信度（含来源权重 + 冷启动加速）。

        Args:
            fact: 目标事实
            delta: 基础变化量（正=提升，负=降低）
            reason: 变化原因
            source_type: 来源类型 → 映射 SOURCE_WEIGHTS
        """
        # 1. 来源权重
        source_weight = SOURCE_WEIGHTS.get(source_type, 0.5)
        adjusted_delta = delta * source_weight

        # 2. 冷启动加速: 前 COLD_START_CONFIRM_COUNT 次正面更新 ×2
        if adjusted_delta > 0 and fact.access_count <= COLD_START_CONFIRM_COUNT:
            adjusted_delta *= COLD_START_BOOST_MULTIPLIER

        # 3. 应用
        old = fact.confidence
        fact.confidence = max(0.0, min(1.0, fact.confidence + adjusted_delta))
        logger.debug(
            f"Confidence: {fact.triple_key} {old:.2f}→{fact.confidence:.2f} "
            f"(base={delta:+.3f}, weight={source_weight}, cold={adjusted_delta/delta if delta else 1:.1f}x, {reason})"
        )

        # 4. DB 日志（忽略 FK 错误，避免缓存态事实未入库导致报错）
        if self.db and abs(adjusted_delta) > 0.001:
            try:
                self.db.log_confidence_change(
                    fact.fact_id, fact.agent_id, old, fact.confidence, reason
                )
            except Exception:
                logger.exception("Failed to log confidence change for fact %s", fact.fact_id)

    def _record_access(self, fact: FactTriple):
        """记录访问"""
        fact.access_count += 1
        fact.accessed_at = datetime.now(timezone.utc).isoformat()
        fact.importance = min(1.0, fact.importance + DEFAULT_IMPORTANCE_BOOST_ON_ACCESS)

    def _match_query(self, fact: FactTriple, query: str) -> bool:
        """判断事实是否匹配查询 (检查 subject/predicate/object + tags + evidence)"""
        if not query:
            return False
        q = query.lower()

        # 直接匹配三元组
        if (q in fact.subject.lower()
                or q in fact.object.lower()
                or q in fact.predicate.lower()):
            return True

        # 匹配标签
        for tag in fact.context_tags:
            if q in tag.lower() or tag.lower() in q:
                return True

        # 匹配证据原文
        for ev in fact.evidence:
            if q in ev.statement.lower():
                return True

        return False

    def _find_existing(self, triple_key: str) -> FactTriple | None:
        """按三元组 key 查找已有事实（TTL 感知缓存优先）"""
        with self._lock:
            self._evict_expired()
            for f in self._lru_cache.values():
                if f.triple_key == triple_key:
                    return f
            if self.db:
                fact = self.db.find_by_triple_key(triple_key)
                if fact:
                    self._cache_put(fact)
                return fact
            return None

    def _get_fact(self, fact_id: str) -> FactTriple | None:
        """按 fact_id 获取事实（优先 TTL 缓存）"""
        cached = self._cache_get(fact_id)
        if cached:
            return cached
        if self.db:
            fact = self.db.get_fact(fact_id)
            if fact:
                self._cache_put(fact)
            return fact
        return None

    def _get_agent_facts(self, agent_id: str) -> list[FactTriple]:
        """获取某 agent 的所有事实（TTL 缓存优先，DB 补充，去重）"""
        with self._lock:
            self._evict_expired()
            seen: set[str] = set()
            facts: list[FactTriple] = []

            for f in self._lru_cache.values():
                if f.agent_id == agent_id:
                    facts.append(f)
                    seen.add(f.fact_id)

            if self.db:
                for f in self.db.get_agent_facts(agent_id):
                    if f.fact_id not in seen:
                        facts.append(f)
                        seen.add(f.fact_id)
                        self._cache_put(f)  # 预热缓存

            return facts

    def _get_cached_facts(self, agent_id: str) -> list[FactTriple]:
        """线程安全地获取缓存的 facts（供 RecallRouter 使用）。
        只返回缓存中的 facts，不查 DB。"""
        with self._lock:
            self._evict_expired()
            return [f for f in self._lru_cache.values() if f.agent_id == agent_id]

    def _delete_fact(self, fact_id: str):
        """删除事实"""
        with self._lock:
            old_fact = self._lru_cache.get(fact_id)
            self._lru_cache.pop(fact_id, None)
            self._cache_timestamps.pop(fact_id, None)
            if self.db:
                try:
                    self.db.delete_fact(fact_id)
                except Exception as e:
                    # DB 删除失败 → 恢复缓存
                    if old_fact:
                        self._cache_put(old_fact)
                    logger.error("delete_fact DB failed, cache restored: %s", e)
                    return
            logger.info(f"Fact deleted: {fact_id}")

    def _merge_facts(self, existing: FactTriple, new_fact: FactTriple,
                     source_type: str = "user_statement") -> dict:
        """合并两个相同三元组的事实"""
        with self._lock:
            from copy import deepcopy
            old = deepcopy(existing)  # 快照，用于版本链

            # 合并证据
            existing.evidence.extend(new_fact.evidence)
            # 提升置信度（来源权重 = merged_evidence 0.5）
            if new_fact.confidence > existing.confidence:
                self._update_confidence(existing,
                                        self.conf_confirm_boost * 0.5,
                                        "merged_evidence",
                                        source_type="merged_evidence")
            existing.access_count += 1
            existing.accessed_at = datetime.now(timezone.utc).isoformat()
            existing.importance = min(1.0, existing.importance + 0.05)

            self._cache_put(existing)
            if self.db:
                self.db.update_fact(existing)
                self.db.save_version(existing.fact_id, existing.agent_id,
                                     old, existing, "merged")

            return {"status": "merged", "fact": existing}

    def _cache_put(self, fact: FactTriple):
        """更新 LRU 缓存（TTL 感知 + 容量扩展至 1000）"""
        with self._lock:
            now = time.time()
            self._lru_cache[fact.fact_id] = fact
            self._cache_timestamps[fact.fact_id] = now
            # 移到末尾（最近使用）
            self._lru_cache.move_to_end(fact.fact_id, last=True)

            # 先清理过期条目
            self._evict_expired(now)

            # 超容量 → 淘汰最旧的（OrderedDict 第一个）
            while len(self._lru_cache) > self._cache_max:
                oldest_id, _ = next(iter(self._lru_cache.items()))
                self._lru_cache.pop(oldest_id, None)
                self._cache_timestamps.pop(oldest_id, None)

    def _evict_expired(self, now: float | None = None):
        """清理所有过期缓存条目"""
        with self._lock:
            if now is None:
                now = time.time()
            expired = [
                fid for fid, ts in self._cache_timestamps.items()
                if now - ts > self._cache_ttl
            ]
            for fid in expired:
                self._lru_cache.pop(fid, None)
                self._cache_timestamps.pop(fid, None)
            if expired:
                logger.debug(f"Cache TTL eviction: {len(expired)} entries")

    def _cache_get(self, fact_id: str) -> FactTriple | None:
        """从缓存获取（自动检查 TTL + 更新访问时间）"""
        with self._lock:
            fact = self._lru_cache.get(fact_id)
            if fact is None:
                return None
            ts = self._cache_timestamps.get(fact_id, 0)
            if time.time() - ts > self._cache_ttl:
                self._lru_cache.pop(fact_id, None)
                self._cache_timestamps.pop(fact_id, None)
                return None
            # 移到末尾（LRU）
            self._lru_cache.move_to_end(fact_id, last=True)
            return fact

    def _get_stats_cached(self, agent_id: str) -> dict | None:
        """获取缓存的统计（30 秒 TTL）"""
        with self._lock:
            entry = self._stats_cache.get(agent_id)
            if entry:
                ts = entry.get("_ts", 0)
                if time.time() - ts < self._stats_cache_ttl:
                    return {k: v for k, v in entry.items() if k != "_ts"}
            return None

    def _set_stats_cached(self, agent_id: str, stats: dict):
        """写入统计缓存"""
        with self._lock:
            stats["_ts"] = time.time()
            self._stats_cache[agent_id] = stats

    def _get_query_cached(self, cache_key: str) -> list | None:
        """获取缓存的查询结果（60 秒 TTL）"""
        with self._lock:
            entry = self._query_cache.get(cache_key)
            if entry:
                results, ts = entry
                if time.time() - ts < self._query_cache_ttl:
                    return results
                else:
                    del self._query_cache[cache_key]
            return None

    def _set_query_cached(self, cache_key: str, results: list):
        """写入查询缓存"""
        with self._lock:
            self._query_cache[cache_key] = (results, time.time())
            # 查询缓存也限制大小
            if len(self._query_cache) > 200:
                oldest_key = min(self._query_cache, key=lambda k: self._query_cache[k][1])
                del self._query_cache[oldest_key]
