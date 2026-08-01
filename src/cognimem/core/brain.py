"""
CogniMem 大脑 — 核心编排器

整合:
- 提取器 (NLP → 三元组)
- 事实网络 (核心引擎)
- 矛盾检测 (创新)
- 递归路由
- 睡眠整合
"""

import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any
from .models import FactTriple, EvidenceItem, Contradiction
from .extractor import TripleExtractor
from .llm_extractor import LLMTripleExtractor
from .fact_network import FactNetwork, _NEUTRAL_PREDICATES
from .recall import RecallRouter

logger = logging.getLogger(__name__)


class CogniMem:
    """
    CogniMem 认知记忆系统主入口

    使用方式:
        brain = CogniMem()
        brain.remember("我喜欢喝冰美式")
        result = brain.recall("用户想喝什么")

    启用 LLM 提取:
        brain = CogniMem(use_llm=True)
        # 或设置环境变量 DASHSCOPE_API_KEY
    """

    def __init__(self, db_adapter=None, config: dict | None = None,
                 use_llm: bool = False):
        self.config = config or {}
        self.extractor = TripleExtractor()

        # LLM 提取器（有条件才启用）
        self.llm_extractor = None
        api_key = (self.config.get("llm_api_key", "")
                   or os.environ.get("DEEPSEEK_API_KEY", "")
                   or os.environ.get("DASHSCOPE_API_KEY", "")
                   or os.environ.get("QWEN_API_KEY", ""))
        if use_llm and api_key:
            model = (self.config.get("llm_model", "")
                     or os.environ.get("QWEN_MODEL", "deepseek-v4-flash"))
            self.llm_extractor = LLMTripleExtractor(
                api_key=api_key,
                model=model,
            )
            logger.info("🤖 LLM extractor enabled: %s", self.llm_extractor.model)

        self.fact_network = FactNetwork(db_adapter, config)
        self.recall_router = RecallRouter(self.fact_network)

        # 🆕 v0.27: 冻结快照 — 会话启动时冻结，整轮不动（Hermes借鉴）
        self._snapshot = {}  # {agent_id: {"system": str, "agent_id": str, "created_at": float}}

    def freeze_snapshot(self, agent_id: str = "default",
                        user_message: str = "",
                        session_id: str = "") -> str | None:
        """生成并冻结当前记忆快照。返回冻结的 system prompt 文本。

        Hermes 思路：会话启动时一次性加载记忆 → prefix cache 稳定。
        SPO 管简单事实 + 快照管整体上下文。

        🐛 v0.27 修复：空 query 触发导航 recall（返回全部事实），
        而非仅当前消息相关的片段。避免快照重建后丢失其他记忆。
        """
        from memory_agent.main import _build_context
        try:
            system, llm_messages = _build_context(
                user_message="",  # 空 query → 导航 recall → 返回全部事实
                agent_id=agent_id,
                session_id=session_id,
                conversation_history=[],
            )
            self._snapshot[agent_id] = {
                "system": system,
                "agent_id": agent_id,
                "created_at": time.time(),
            }
            return system
        except Exception as e:
            logger.warning("freeze_snapshot failed: %s", e)
            return None

    def get_snapshot(self, agent_id: str = "default") -> str | None:
        """获取当前 agent 的冻结快照。不存在或过期返回 None。"""
        snap = self._snapshot.get(agent_id)
        if snap is None:
            return None
        return snap.get("system")

    def refresh_snapshot(self, agent_id: str = "default",
                         user_message: str = "",
                         session_id: str = "") -> str | None:
        """主动刷新快照（LLM memory_remember 后调用）。"""
        self._snapshot.pop(agent_id, None)
        return self.freeze_snapshot(agent_id, user_message, session_id)

    def has_snapshot(self, agent_id: str = "default") -> bool:
        """检查是否有有效快照。"""
        return agent_id in self._snapshot

    # ── 写入 ──

    # ═══════════════════════════════════════════════════════════════
    # ★ L4 反思框架 — 从错误中自动学习，逐步逼近完美
    # ═══════════════════════════════════════════════════════════════
    # 借鉴 self-improving-agent 的反思模式，但直接存储到 SPO 图谱。
    # 每次发现错误/异常/空结果时，记录一条 lesson 事实。
    # 后续 recall 时自动加载相关 lesson，避免重复犯错。

    _LESSON_CATEGORIES = {
        "提取失败": "LLM/规则提取返回空或无意义事实",
        "用户修正": "用户纠正了系统的输出或记忆",
        "召回为空": "用户提问但recall返回0条",
        "工具错误": "工具执行失败",
        "快照过时": "冻结快照与当前记忆状态不一致",
    }

    def _store_lesson(self, agent_id: str, category: str,
                       summary: str, details: str,
                       source: str = "self_reflection") -> dict:
        """存储一条经验教训到 SPO 图谱。

        L4 反思框架核心：把错误/失败/异常转化为可检索的 lesson 事实。
        同类 lesson 重复出现时自动提权。

        Args:
            agent_id: 归属 agent
            category: 教训类别（见 _LESSON_CATEGORIES）
            summary: 一句话总结（作为 predicate/object）
            details: 详细描述（作为 evidence）
            source: 来源（"self_reflection"/"user_correction"/"tool_error"）
        """
        if category not in self._LESSON_CATEGORIES:
            category = "提取失败"

        # 先查是否已有同类 lesson（同 summary 视为重复）
        existing = self.fact_network.recall_by_triple("系统", "学到了", agent_id)
        for f in existing:
            if f.object == summary[:100]:
                # 重复 lesson → 提权 + 更新证据
                f.confidence = min(f.confidence + 0.1, 0.8)
                f.importance = min(f.importance + 0.05, 0.8)
                f.access_count += 1
                f.evidence.append(EvidenceItem(
                    source=source,
                    statement=details[:500],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))
                self.fact_network._cache_put(f)
                if self.fact_network.db:
                    try:
                        self.fact_network.db.update_fact(f)
                    except Exception:
                        pass
                logger.info("📘 Lesson reinforced: %s (conf=%.2f)", summary[:40], f.confidence)
                return {"status": "reinforced", "fact": f}

        # 新 lesson
        fact = FactTriple(
            subject="系统",
            predicate="学到了",
            object=summary[:100],
            agent_id=agent_id,
            fact_type="lesson",
            confidence=0.5,
            importance=0.3,
            encoding_level="raw",
            context_tags=["反思", category],
            source_session=f"lesson:{source}",
            evidence=[EvidenceItem(
                source=source,
                statement=details[:500],
                timestamp=datetime.now(timezone.utc).isoformat(),
            )],
        )
        result = self.fact_network.add_fact(fact)
        if self.fact_network.db:
            try:
                # 写入后使快照失效，下次自动重建
                self._snapshot.pop(agent_id, None)
            except Exception:
                pass
        logger.info("📘 Lesson stored: %s (cat=%s)", summary[:40], category)
        return result

    # ⭐ 不需要 LLM 提取的文本模式（省 Token 省时间）
    _SKIP_LLM_PATTERNS = [
        r"^完成了一个任务",
        r"^任务步骤结果",
        r"^用户信息:",
        r"^用户提问:",
        r"^用戶請求了",        # agent 自我反思
        r"^用户請求了",        # agent 自我反思（简体）
        r"^ping$",
        r"^hi$",
        r"^你好",
        r"^\d+\+?\d*=?\??$",  # 数学表达式
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-",  # UUID
    ]

    @classmethod
    def _should_skip_llm(cls, text: str) -> bool:
        """判断文本是否太简单/系统化，不需要 LLM 提取"""
        if not text or len(text.strip()) < 8:
            return True  # 短文本无需 LLM
        for pat in cls._SKIP_LLM_PATTERNS:
            if re.search(pat, text):
                return True
        return False

    def remember(self, text: str, source: str = "",
                 agent_id: str = "default",
                 source_type: str = "user_statement") -> dict:
        """记住一条信息。流程: 提取三元组 → 检测矛盾 → 存入事实网络

        ⭐ 规则优先：先试规则提取（0 Token），够用就不调 LLM。
        """
        # ⭐ 空闲自动整合
        self.fact_network._maybe_auto_consolidate(
            agent_id, self.llm_extractor
        )
        # 1. 规则提取（0 Token，<1ms）— 所有情况都先试
        facts = self.extractor.extract(text, source, agent_id)
        rules_found_good = facts and any(f.confidence >= 0.6 for f in facts)

        # 2. LLM 精提 — 仅在以下情况才调：
        #    a) 规则没提取到好结果（无事实或置信度 < 0.6）
        #    b) 文本不是系统日志/简单短句（_should_skip_llm）
        need_llm = False
        if self.llm_extractor and not rules_found_good:
            if not self._should_skip_llm(text):
                need_llm = True
            else:
                logger.info(
                    "⚡ Skipped LLM extract (trivial text): %s",
                    text[:40],
                )

        if need_llm:
            try:
                llm_facts = self.llm_extractor.extract(text, source, agent_id)
                if llm_facts:
                    facts = llm_facts
                    source_type = "agent_inference"
            except Exception as e:
                logger.warning("⚠️ LLM extractor failed, using rule-only: %s", str(e)[:80])

        if not facts:
            # 🧠 L4 反思：提取完全失败 → 记录教训
            _skip_reason = "too_short" if len(text.strip()) < 8 else "no_pattern_match"
            if _skip_reason == "no_pattern_match":
                try:
                    self._store_lesson(
                        agent_id=agent_id,
                        category="提取失败",
                        summary=f"文本\"{text[:30]}…\"无规则匹配且LLM提取返回空",
                        details=f"source_type={source_type} text_len={len(text)} text={text[:200]}",
                        source="self_reflection",
                    )
                except Exception:
                    pass
            return {"status": "no_facts_extracted", "facts": []}

        # ⭐ 增强证据链：每条事实都带原始来源文本
        # 必须在原文保底检查之前运行，否则空 evidence 的 fact 会导致保底误判追加重复
        for f in facts:
            has_source = False
            for ev in f.evidence:
                if isinstance(ev, EvidenceItem) and ev.statement:
                    has_source = True
                    break
            if not has_source:
                f.evidence.append(EvidenceItem(
                    source=source or source_type,
                    statement=text[:300],  # 保留原文前300字
                ))
            if source and source.startswith("session:"):
                f.source_session = source

        # ⭐ 原文保底：证据链已补齐，检查原文是否可搜索
        # 防止规则匹配只截取部分内容导致关键字丢失（如"负责消息队列中间件"丢了"消息队列"）
        # ⚠️ 用正常化空格后再比，避免 evidence 保存时多空格导致匹配失败 → 追加重复
        _norm = lambda s: ' '.join(s.split())
        has_original = any(
            isinstance(ev, EvidenceItem) and _norm(text[:100]) in _norm(ev.statement)
            for f in facts for ev in f.evidence
        )
        if not has_original:
            # 简单抽取标签（不依赖 extractor 实例）
            _keywords = [w for w in text.split() if len(w) >= 2][:5]
            facts.append(FactTriple(
                subject="用户",
                predicate="说了",
                object=text[:500],
                agent_id=agent_id,
                fact_type="observation",
                confidence=0.5,
                source_session=source,
                context_tags=_keywords if _keywords else None,
                evidence=[EvidenceItem(source=source or "unknown", statement=text[:300])],
            ))

        # 🆕 v0.25: 紧急指令/核心规则 高优先级标记
        # 防止被大量无关对话冲刷出STM
        _priority_keywords = {"紧急指令", "核心", "必须记住", "第一优先级", "最重要",
                             "优先提醒", "不能忘", "一定记住"}
        if any(kw in text for kw in _priority_keywords):
            for f in facts:
                f.importance = max(f.importance, 0.95)
                f.confidence = max(f.confidence, 0.85)
                if "优先" not in f.context_tags:
                    f.context_tags = list(f.context_tags) + ["优先"]
            logger.info("🔴 紧急指令标记: importance=0.95")

        # 🔥 v0.24: 修正处理 — source_type="user_correction" 时自动覆盖旧事实
        if source_type == "user_correction":
            for f in facts:
                if "修正" not in f.context_tags:
                    f.context_tags = list(f.context_tags) + ["修正"]
                f.confidence = max(f.confidence, 0.8)  # 修正事实高置信度

            # 🔍 查找可覆盖的旧事实：先精确再模糊
            for f in facts:
                if not f.subject or not f.predicate:
                    continue

                # ① 精确匹配：同subject + 同predicate
                candidates = self.fact_network.recall_by_triple(
                    f.subject, f.predicate, agent_id
                )
                # ② 模糊搜索：遍历agent所有事实，找主题匹配的
                # 修正场景下，新值（"6月10日"）不可能出现在旧事实里
                # 所以改成按主题词搜索：从predicate/evidence提主题词
                _topic_words = set()
                # 从新事实predicate提： "日期是" → ["日期"]
                for w in ["生日", "日期", "名字", "年龄", "电话", "工作", "公司"]:
                    if w in f.predicate or w in f.object:
                        _topic_words.add(w)
                # 从修正文本提
                for ev in f.evidence:
                    if isinstance(ev, EvidenceItem):
                        _txt = ev.statement or ""
                        for w in ["生日", "日期", "名字", "年龄", "电话", "工作", "公司"]:
                            if w in _txt:
                                _topic_words.add(w)

                all_facts = self.fact_network._get_agent_facts(agent_id)
                for old in all_facts:
                    if old.fact_id == f.fact_id:
                        continue
                    # subject 归一化后比较
                    _old_subj_norm = "用户" if old.subject in ("我", "你", "您", "user") else old.subject
                    _f_subj_norm = "用户" if f.subject in ("我", "你", "您", "user") else f.subject
                    if _old_subj_norm != _f_subj_norm:
                        continue
                    # 匹配主题词
                    _matched = False
                    for tw in _topic_words:
                        if tw in old.predicate or tw in old.object:
                            _matched = True
                            break
                        for ev in old.evidence:
                            if isinstance(ev, EvidenceItem) and tw in (ev.statement or ""):
                                _matched = True
                                break
                        if _matched:
                            break
                    if _matched:
                        candidates.append(old)
                        continue
                    # 日期模式匹配：仅当新事实和旧事实都是日期类谓词时才纳入
                    # 🐛 v0.30: 旧代码只看 old.predicate 含"月""日" →
                    #   修正"生日是6月10日"会误伤"纪念日是5月20日"等其他日期事实
                    if any(dw in f.predicate for dw in ("生日", "日期")) and \
                       any(dw in old.predicate for dw in ("生日", "日期")):
                        candidates.append(old)
                        continue
                    # 被修正标记的事实也加入
                    if "被修正" in old.context_tags:
                        candidates.append(old)

                for old in candidates:
                    if old.fact_id == f.fact_id:
                        continue
                    if old.object == f.object:
                        continue  # 相同值不是修正
                    # 旧事实降置信度（被修正了）
                    old.confidence = min(old.confidence, 0.3)
                    old.importance = max(old.importance * 0.5, 0.1)
                    if "被修正" not in old.context_tags:
                        old.context_tags = list(old.context_tags) + ["被修正"]
                    if f.fact_id not in old.connected_facts:
                        old.connected_facts.append(f.fact_id)

                    # 🔥 继承旧事实的关键词到新事实（保证召回能找到）
                    for tag in old.context_tags:
                        if tag not in f.context_tags and tag not in ("被修正", "修正"):
                            f.context_tags = list(f.context_tags) + [tag]
                    # 也把旧事实的 predicate 关键词加进去
                    _old_pred_words = old.predicate.replace("生日", "").replace("日期", "")
                    if _old_pred_words and _old_pred_words not in f.context_tags:
                        f.context_tags = list(f.context_tags) + [_old_pred_words]
                    # 从旧事实evidence中提取主题词（"我的生日是5月10日"→"生日"）
                    for ev in old.evidence:
                        if isinstance(ev, EvidenceItem) and ev.statement:
                            _topic_words = [w for w in ["生日","名字","年龄","电话","地址","工作","公司","学校"] if w in ev.statement]
                            for tw in _topic_words:
                                if tw not in f.context_tags:
                                    f.context_tags = list(f.context_tags) + [tw]

                    # 🔥 继承旧事实的精确谓词（"日期是"→"生日是"）
                    # 旧事实的predicate更具体时使用
                    _old_pred_clean = old.predicate.replace("生日", "").replace("日期", "").strip()
                    _new_pred_clean = f.predicate.replace("生日", "").replace("日期", "").strip()
                    if len(_old_pred_clean) < len(_new_pred_clean) or ("生日" in old.predicate and "生日" not in f.predicate):
                        f.predicate = old.predicate
                        logger.info("🔄 谓词继承: %s → %s", _new_pred_clean, f.predicate)
                    self.fact_network._cache_put(old)
                    if self.fact_network.db:
                        try:
                            self.fact_network.db.update_fact(old)
                        except Exception:
                            pass
                    logger.info(
                        "🔄 修正覆盖: (%s, %s, %s) 降权旧事实 (%s, %s, %s)",
                        f.subject, f.predicate, f.object,
                        old.subject, old.predicate, old.object,
                    )

        # 🐛 v0.28 修复: 隐式修正检测 — 只对fact类型生效
        # 用户说"我叫测试用户"应覆盖"我叫张三"（唯一属性修正）
        # ⚠️ preference/goal 是累加型（喜欢吃日式料理 + 吃苦瓜 是两件事不互斥）
        #    不能因为"喜欢吃苦瓜"就把"喜欢吃日式料理"打为"被修正"
        # 🐛 v0.29 修复: 超级通用谓语"是"不触发隐式修正（Q1 希腊脚误杀）
        #   "用户 是 希腊脚"和"用户 是 事件视界"是完全不同的信息，不是相互修正
        _BROAD_PREDICATES = frozenset({"是"})
        if source_type not in ("user_correction",):
            for f in facts:
                if f.fact_type != "fact":
                    continue
                if not f.subject or not f.predicate:
                    continue
                if f.predicate in _BROAD_PREDICATES:
                    continue  # "是"太通用，不触发隐式修正
                old_facts = self.fact_network.recall_by_triple(f.subject, f.predicate, agent_id)
                for old in old_facts:
                    if old.fact_id == f.fact_id or old.fact_id in [x.fact_id for x in facts]:
                        continue
                    if old.object == f.object:
                        continue
                    if old.confidence < 0.3 or f.confidence < 0.5:
                        continue
                    if "被修正" not in old.context_tags:
                        old.context_tags = list(old.context_tags) + ["被修正"]
                    old.confidence = min(old.confidence, 0.3)
                    if "修正" not in f.context_tags:
                        f.context_tags = list(f.context_tags) + ["修正"]
                    f.confidence = max(f.confidence, 0.75)
                    self.fact_network._cache_put(old)
                    if self.fact_network.db:
                        try:
                            self.fact_network.db.update_fact(old)
                        except Exception:
                            pass
                    logger.info(
                        "🔄 隐式修正: (%s, %s, %s) 覆盖 (%s, %s, %s)",
                        f.subject, f.predicate, f.object,
                        old.subject, old.predicate, old.object,
                    )

        # 🐛 v0.28: 反义谓词检测 — "不喜欢X了" → 降低"喜欢X"旧事实置信度
        # preference 类更新（"不喜欢美式了→现在喜欢拿铁"）需要降权旧"喜欢美式"
        for f in facts:
            if f.fact_type != "preference" or not f.subject or not f.predicate:
                continue
            if f.predicate == "不喜欢" and f.object:
                old_likes = self.fact_network.recall_by_triple(f.subject, "喜欢", agent_id)
                for old in old_likes:
                    if old.fact_id == f.fact_id:
                        continue
                    # 检查是否同话题（对象有重叠词）
                    _new_obj = f.object.replace("了", "").strip()
                    _overlap = any(w in old.object and len(w) >= 2 for w in [_new_obj])
                    if not _overlap and _new_obj not in old.object and old.object not in _new_obj:
                        continue
                    if old.confidence >= 0.5:
                        logger.info("🔄 反义降权: (%s, %s, %s) 旧(%s, %s, %s) conf=%.2f→%.2f",
                                    f.subject, f.predicate, f.object,
                                    old.subject, old.predicate, old.object,
                                    old.confidence, old.confidence * 0.5)
                    old.confidence *= 0.5
                    if "被修正" not in old.context_tags:
                        old.context_tags = list(old.context_tags) + ["被修正"]
                    self.fact_network._cache_put(old)
                    if self.fact_network.db:
                        try:
                            self.fact_network.db.update_fact(old)
                        except Exception:
                            pass

        # 🐛 v0.29 修复(Q6): 属性自动更新 — "搬到X" → 更新"住在X"
        # 当用户说"搬到通州了"，旧"住在朝阳区"应被自动降权
        _RELOCATION_VERBS = frozenset({"搬到", "搬去", "搬到了", "搬去了", "搬来", "搬来了", "搬到"})
        for f in facts:
            if f.fact_type != "fact" or not f.subject or not f.object:
                continue
            if f.predicate not in _RELOCATION_VERBS:
                continue
            # 找到同 subject 的"住在"事实
            old_residence = self.fact_network.recall_by_triple(f.subject, "住在", agent_id)
            for old in old_residence:
                if old.fact_id == f.fact_id:
                    continue
                if old.object == f.object:
                    continue  # 搬到同一个地方，不处理
                if old.confidence >= 0.4:
                    logger.info("🔄 搬迁移权: (%s, %s, %s) 旧(%s, %s, %s) conf=%.2f→%.2f",
                                f.subject, f.predicate, f.object,
                                old.subject, old.predicate, old.object,
                                old.confidence, old.confidence * 0.3)
                old.confidence *= 0.3
                if "被修正" not in old.context_tags:
                    old.context_tags = list(old.context_tags) + ["被修正"]
                # 在新事实上加"修正"标签，提高其权重
                if "修正" not in f.context_tags:
                    f.context_tags = list(f.context_tags) + ["修正"]
                f.confidence = max(f.confidence, 0.75)
                self.fact_network._cache_put(old)
                if self.fact_network.db:
                    try:
                        self.fact_network.db.update_fact(old)
                    except Exception:
                        pass

        # 3. 批量添加 (含矛盾检测 + 来源权重)
        results = self.fact_network.batch_add(facts, agent_id, source_type)

        # 🆕 v0.25: 叙事跨会话链接
        # 如果新事实中包含叙事类型，与已有的叙事事实建立 connected_facts
        try:
            self._link_narratives(agent_id, facts)
        except Exception as e:
            logger.warning("叙事链接失败: %s", e)

        # 3. 检查是否有矛盾
        contradictions = [
            r for r in results if r.get("status") == "contradiction_detected"
        ]

        # ★ 审计日志（受 DREAM audit ledger 启发）
        db = getattr(self.fact_network, 'db', None)
        if db and hasattr(db, 'log_audit'):
            for r in results:
                f = r.get("fact")
                if f:
                    op = "create" if r.get("status") == "created" else \
                         "update" if r.get("status") == "merged" else "contradiction"
                    db.log_audit(
                        agent_id=agent_id,
                        fact_id=f.fact_id,
                        operation=op,
                        detail=f"存储事实: {f.subject} {f.predicate} {f.object}",
                        metadata={"source_type": source_type, "status": r.get("status")},
                        caller="brain.remember",
                    )

        response = {
            "status": "remembered",
            "facts_added": len(facts),
            "facts": [r["fact"] for r in results],
        }

        if contradictions:
            response["contradictions_detected"] = len(contradictions)
            response["contradiction_details"] = [
                c.get("contradictions", [])
                for c in contradictions
            ]

        return response

    def batch_remember(self, texts: list[str], source: str = "",
                       agent_id: str = "default",
                       source_type: str = "user_statement") -> list[dict]:
        """批量记住多条信息"""
        return [self.remember(t, source, agent_id, source_type) for t in texts]

    # ── 召回 ──

    def recall(self, query: str, agent_id: str = "default",
               top_k: int = 10, context: dict | None = None,
               session_id: str = "") -> dict:
        """
        召回记忆。

        三级路由自动选择最优路径。

        Args:
            query: 查询
            agent_id: Agent ID
            top_k: 最大返回数
            context: 上下文 (话题标签等)
            session_id: 当前会话 ID（用于同会话事实加权）
        """
        # ⭐ 空闲自动整合
        self.fact_network._maybe_auto_consolidate(
            agent_id, self.llm_extractor
        )
        ctx = dict(context or {})
        if session_id:
            ctx["session_id"] = session_id
        facts = self.recall_router.recall(query, agent_id, ctx, top_k)
        pending_contradictions = self.fact_network.get_contradictions(agent_id)

        # 🧠 L4 反思：用户提问但 recall 为空 → 记录教训（过滤问候/短句）
        _is_greeting = query.strip() in ("你好", "hi", "hello", "hey", "在吗", "")
        if not facts and len(query) > 6 and not _is_greeting and agent_id:
            _lesson_facts = [f for f in facts if f.fact_type == "lesson"]
            if not _lesson_facts:
                try:
                    self._store_lesson(
                        agent_id=agent_id,
                        category="召回为空",
                        summary=f"查询\"{query[:25]}…\"召回0条相关记忆",
                        details=f"query_len={len(query)} top_k={top_k}",
                        source="self_reflection",
                    )
                except Exception:
                    pass

        return {
            "facts": facts,
            "count": len(facts),
            "has_contradictions": len(pending_contradictions) > 0,
            "contradictions": pending_contradictions if pending_contradictions else None,
        }

    def recall_cross_agent(self, query: str, agent_ids: list[str],
                           top_k: int = 10) -> dict:
        """
        ★ 跨 Agent 记忆总线（受 Universal Agent OS 多Agent记忆总线启发）。

        从多个 Agent 的记忆库中同时召回，去重后按置信度排序返回。
        适合：团队知识共享/多角色记忆池。

        Args:
            query: 查询文本
            agent_ids: 要查询的 Agent ID 列表
            top_k: 最大返回数

        Returns:
            {"facts": [...], "count": N, "sources": {agent_id: count}}
        """
        from collections import Counter
        all_facts = []
        seen_ids = set()
        source_counts = Counter()

        for aid in agent_ids[:10]:  # 最多查 10 个 agent
            try:
                result = self.recall(query, aid, top_k=top_k // len(agent_ids[:10]))
                for f in result.get("facts", []):
                    if f.fact_id not in seen_ids:
                        seen_ids.add(f.fact_id)
                        all_facts.append(f)
                        source_counts[aid] += 1
            except Exception as e:
                logger.warning("Cross-agent recall failed for '%s': %s", aid, e)
                continue

        # 跨 Agent 排序（置信度降序）
        all_facts.sort(key=lambda f: f.confidence, reverse=True)
        return {
            "facts": all_facts[:top_k],
            "count": min(len(all_facts), top_k),
            "sources": dict(source_counts.most_common()),
        }

    def ask(self, query: str, agent_id: str = "default") -> dict:
        """
        问答式召回 — 适合 Agent 直接使用。

        - 自动召回相关事实
        - 附带置信度说明
        - 提醒矛盾信息
        - ⭐ 主动学习：检测到矛盾时生成引导性问题

        Returns: Agent 可以直接使用的结构
        """
        result = self.recall(query, agent_id, top_k=5)

        beliefs = self.fact_network.get_beliefs(agent_id, min_confidence=0.6)
        uncertainties = [
            f for f in self.fact_network._get_agent_facts(agent_id)
            if 0.2 <= f.confidence < 0.6
        ]

        # ═══ 主动学习：检测矛盾，生成引导性问题 ═══
        active_questions = []
        pending_contradictions = result.get("contradictions") or []
        if pending_contradictions:
            for c in pending_contradictions[:3]:  # 最多 3 个
                fa = self.fact_network._get_fact(c.fact_a_id)
                fb = self.fact_network._get_fact(c.fact_b_id)
                if fa and fb:
                    if c.contradiction_type == "deny":
                        # L1 直接否定 → 引导用户澄清哪个是对的
                        active_questions.append(
                            f"我注意到关于「{fa.subject}」，"
                            f"之前记录是「{fa.predicate}{fa.object}」，"
                            f"但后来又说「{fb.predicate}{fb.object}」。"
                            f"哪个是准确的？"
                        )
                    elif c.contradiction_type == "conflict":
                        # L2 间接冲突 → 提醒注意
                        active_questions.append(
                            f"有个小矛盾：你提到过「{fa.subject}{fa.predicate}{fa.object}」，"
                            f"但同时也说过「{fb.subject}{fb.predicate}{fb.object}」。"
                            f"这两者好像不太一致，能帮我澄清一下吗？"
                        )
                    elif c.contradiction_type == "context":
                        # L3 上下文变化 → 确认是否更新了
                        active_questions.append(
                            f"关于「{fa.subject}」的信息有变化："
                            f"从「{fa.object}」变成了「{fb.object}」。"
                            f"是更新了吗？"
                        )

        # 不确定项也生成主动询问
        if uncertainties and not active_questions:
            low_conf_facts = uncertainties[:2]
            for f in low_conf_facts:
                active_questions.append(
                    f"我不太确定「{f.subject} {f.predicate} {f.object}」是否准确"
                    f"（可信度 {f.confidence:.0%}），你能确认一下吗？"
                )

        return {
            "query": query,
            "relevant_memories": [
                {
                    "fact": f"{f.subject} {f.predicate} {f.object}",
                    "confidence": f.confidence,
                    "type": f.fact_type,
                    # ★ 来源引用（受 RuleMemory provenance 启发）
                    "citation": f.citation,
                    "source_label": f.source_label,
                    # ★ 过期警告（受 RuleMemory stale-assumption 检测启发）
                    "stale_warning": f.stale_warning,
                }
                for f in result["facts"]
            ],
            "core_beliefs": [
                {
                    "belief": f"{f.subject} {f.predicate} {f.object}",
                    "confidence": f.confidence,
                    "citation": f.citation,
                }
                for f in beliefs[:3]
            ],
            "uncertainties": [
                {
                    "fact": f"{f.subject} {f.predicate} {f.object}",
                    "confidence": f.confidence,
                    "note": "这条我不太确定，需要你确认",
                    "citation": f.citation,
                }
                for f in uncertainties[:3]
            ],
            "contradictions_warning": result["contradictions"] is not None,
            "active_questions": active_questions,  # ⭐ 主动学习问题
        }

    # ── 确认与质疑 ──

    def confirm(self, fact_id: str, agent_id: str = "default") -> dict:
        """确认一个事实 → 置信度提升"""
        fact = self.fact_network.confirm_fact(fact_id, "user_confirmation")
        # ★ 审计日志
        if fact:
            db = getattr(self.fact_network, 'db', None)
            if db and hasattr(db, 'log_audit'):
                db.log_audit(agent_id, "confirm", f"确认事实: {fact.subject} {fact.predicate} {fact.object}",
                             fact_id=fact_id, caller="brain.confirm")
        return {"status": "confirmed" if fact else "not_found"}

    def challenge(self, fact_id: str, agent_id: str = "default") -> dict:
        """质疑一个事实 → 置信度降低"""
        fact = self.fact_network.challenge_fact(fact_id, "user_challenge")
        # ★ 审计日志
        if fact:
            db = getattr(self.fact_network, 'db', None)
            if db and hasattr(db, 'log_audit'):
                db.log_audit(agent_id, "challenge", f"质疑事实: {fact.subject} {fact.predicate} {fact.object}",
                             fact_id=fact_id, caller="brain.challenge")
        return {"status": "challenged" if fact else "not_found"}

    def resolve_contradiction(self, contradiction_id: str,
                              resolution: str) -> dict:
        """解决矛盾"""
        return {"status": "resolved", "resolution": resolution}

    def analyze_contradiction(self, fact_id_a: str, fact_id_b: str,
                               agent_id: str = "default") -> dict:
        """用 LLM 分析两个事实之间的矛盾"""
        fa = self.fact_network._get_fact(fact_id_a)
        fb = self.fact_network._get_fact(fact_id_b)
        if not fa or not fb:
            return {"error": "fact not found"}

        # 确定矛盾类型
        contradictions = self.fact_network.get_contradictions(agent_id)
        ctype = "deny"
        for c in contradictions:
            if c.fact_a_id == fact_id_a and c.fact_b_id == fact_id_b:
                ctype = c.contradiction_type
                break

        if not self.llm_extractor:
            return {"verdict": ctype, "needs_confirmation": ctype == "deny",
                    "explanation": "LLM 未启用，使用规则判断"}

        prompt = (
            f"分析事实矛盾：\n\n"
            f"A: ({fa.subject}, {fa.predicate}, {fa.object}) conf={fa.confidence:.2f}\n"
            f"B: ({fb.subject}, {fb.predicate}, {fb.object}) conf={fb.confidence:.2f}\n"
            f"类型: {ctype}\n\n"
            f"JSON 输出：\n"
            f'{{"verdict":"contradiction|context|misunderstanding",'
            f'"explanation":"中文分析","needs_confirmation":true/false}}'
        )

        try:
            import json, openai
            client = openai.OpenAI(
                api_key=self.llm_extractor.api_key,
                base_url=self.llm_extractor.base_url,
            )
            r = client.chat.completions.create(
                model=self.llm_extractor.model,
                messages=[
                    {"role": "system", "content": "你是一个矛盾分析专家。只返回 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1, max_tokens=300,
                response_format={"type": "json_object"},
            )
            return json.loads(r.choices[0].message.content or "{}")
        except Exception as e:
            return {"error": str(e), "verdict": ctype,
                    "needs_confirmation": ctype == "deny"}

    # ── 维护 ──

    def forget(self, query: str, agent_id: str = "default") -> dict:
        """🐛 v0.29 修复(Q5): 根据关键词遗忘特定记忆"""
        db = getattr(self.fact_network, 'db', None)
        if not db:
            return {"forgotten": 0, "message": "无数据库连接"}
        try:
            # 从 query 提取关键词（去掉"忘记/删掉"等指令词）
            # 🐛 v0.30: "我的"必须整体匹配（旧正则先匹配"我"→ 剩"的银行卡信息"，
            #   开头的"的"导致后续匹配全部失败）
            _clean = re.sub(r'(?:请|帮我)?(?:忘记|忘掉|删掉|删除|清除|不要记|别记)(?:我刚才说的|我的|我|的|这个|那个|这条)?', '', query).strip()
            if not _clean or len(_clean) < 2:
                return {"forgotten": 0, "message": "请指定要遗忘什么信息", "hint": '例如"忘记我说的银行卡信息"'}

            # 提取关键词（去掉"是敏感数据"等后缀）
            _clean = re.sub(r'[，,。.！!？?](?:那|这|它).*$', '', _clean)
            # 去末尾标点
            _clean = _clean.strip('，。！？,.!?、；;')
            # 🐛 v0.30: 剥离宽泛后缀（"银行卡信息"→"银行卡"）
            #   用户说"银行卡信息"，存的事实是"银行卡后四位" — 必须用核心词匹配
            _core = re.sub(r'(?:信息|号码|密码|账号|账户|资料|情况|内容|数据|细节)$', '', _clean)
            if len(_core) >= 2:
                _clean = _core
            _words = [w for w in re.split(r'[的，,\s]', _clean) if len(w) >= 2]

            all_facts = self.fact_network._get_agent_facts(agent_id)
            ids_to_forget = set()
            for f in all_facts:
                _text = f"{f.subject} {f.predicate} {f.object} {f.evidence[0].statement if f.evidence else ''}"
                # ① 精确匹配完整关键词
                if _clean in _text:
                    ids_to_forget.add(f.fact_id)
                # ② 分词单词匹配（银行卡/信用卡/密码等）
                if any(w in _text for w in _words):
                    ids_to_forget.add(f.fact_id)

            if not ids_to_forget:
                return {"forgotten": 0, "message": f"未找到与「{_clean}」相关的记忆"}

            # 🐛 v0.29: UPDATE 和 DELETE 必须分开事务。
            # DELETE 语句失败会导致整个事务回滚，之前的 UPDATE 全部丢失！
            with db._plain_cursor_ctx() as cur:
                for fid in ids_to_forget:
                    cur.execute(
                        "UPDATE facts SET confidence = 0, importance = 0, "
                        "context_tags = array_append(context_tags, '已遗忘') "
                        "WHERE fact_id = %s AND agent_id = %s",
                        (fid, agent_id))
            # 单独事务删矛盾记录（表不存在也要保证 UPDATE 已提交）
            # 🐛 v0.30: 列名错误！contradictions 表只有 fact_a_id/fact_b_id，
            #   旧代码写 related_fact_id → 每次报"column does not exist"被吞掉
            #   → 遗忘后矛盾记录永久残留，ask() 持续追问旧矛盾
            try:
                with db._plain_cursor_ctx() as cur2:
                    id_list = list(ids_to_forget)
                    # 🐛 v0.30: uuid 列必须 ::text 再比较（ANY(ARRAY['uuid']) 会报
                    #   "operator does not exist: uuid = text"）
                    cur2.execute(
                        "DELETE FROM contradictions WHERE fact_a_id::text = ANY(%s) OR fact_b_id::text = ANY(%s)",
                        (id_list, id_list))
            except Exception as _e2:
                logger.warning("⚠️ 矛盾记录清理失败: %s", _e2)

            # 清除缓存 + 快照
            self.fact_network._clear_agent_cache(agent_id)
            self._snapshot.pop(agent_id, None)

            logger.info("🗑️ Forget agent '%s': %d facts for '%s'", agent_id, len(ids_to_forget), _clean)
            return {"forgotten": len(ids_to_forget), "message": f"已遗忘 {len(ids_to_forget)} 条相关记忆"}
        except Exception as e:
            logger.error("Forget failed: %s", e)
            return {"forgotten": 0, "message": str(e)}

    def reset_agent(self, agent_id: str = "default") -> dict:
        """清除指定 Agent 的所有记忆（含 FK 关联表）"""
        db = getattr(self.fact_network, 'db', None)
        if not db:
            return {"deleted": 0, "message": "无数据库连接"}
        try:
            with db._plain_cursor_ctx() as cur:
                tables = ["confidence_log", "fact_versions", "contradictions",
                          "facts", "episodes", "working_memory_snapshots"]
                total = 0
                for table in tables:
                    cur.execute(f"DELETE FROM {table} WHERE agent_id = %s", (agent_id,))
                    total += cur.rowcount
            # ★ 审计日志
            if hasattr(db, 'log_audit'):
                db.log_audit(agent_id, "delete", f"清除 Agent 所有记忆（{total} 行）",
                             caller="brain.reset_agent")
            # 🐛 v0.28: 清内存缓存（否则矛盾检测引用已删 fact_id 导致外键冲突）
            self.fact_network._clear_agent_cache(agent_id)
            # 🐛 v0.29 修复: /clear 后必须清除内存快照，否则旧数据污染新会话（Q10/Q17）
            self._snapshot.pop(agent_id, None)
            logger.info("🗑️ Reset agent '%s': %d rows deleted, snapshot cleared", agent_id, total)
            return {"deleted": total, "message": "记忆已清除"}
        except Exception as e:
            logger.error("Reset agent failed: %s", e)
            return {"deleted": 0, "message": str(e)}

    def consolidate(self, agent_id: str = "default") -> dict:
        """触发睡眠期记忆整合（含抽象化 + 矛盾定期扫描）"""
        result = self.fact_network.consolidate(agent_id,
                                               llm_extractor=self.llm_extractor)

        # ★ 矛盾定期扫描：检查非动作、非中性事实间的矛盾
        try:
            contradictions_found = self._scan_contradictions(agent_id)
            if contradictions_found:
                result["contradictions"] = contradictions_found
        except Exception as e:
            logger.warning("矛盾扫描失败: %s", e)

        # ★ 审计日志
        db = getattr(self.fact_network, 'db', None)
        if db and hasattr(db, 'log_audit'):
            detail = (f"维护完成: 遗忘{result.get('deleted',0)}条 "
                     f"衰减{result.get('decayed',0)}条 "
                     f"抽象{result.get('abstracted',0)}条"
                     f"矛盾{result.get('contradictions',0)}条")
            db.log_audit(agent_id, "consolidation", detail, caller="brain.consolidate")
        return result

    def _scan_contradictions(self, agent_id: str) -> int:
        """主动扫描所有非动作事实间的矛盾"""
        import uuid
        facts = self.fact_network.get_all_facts(agent_id)
        candidates = [f for f in facts
                      if f.fact_type not in ("action", "credential")
                      and f.predicate not in _NEUTRAL_PREDICATES]
        found = 0
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a, b = candidates[i], candidates[j]
                if a.subject != b.subject or a.object != b.object:
                    continue

                def _is_neg(p):
                    return p.startswith(("不", "没", "未")) or p in {"讨厌", "反感", "拒绝"}
                def _is_pos(p):
                    return p in {"喜欢", "爱", "要", "想", "是", "有", "需要", "能", "可以"}

                if _is_neg(a.predicate) and _is_pos(b.predicate):
                    c = Contradiction(fact_a_id=a.fact_id, fact_b_id=b.fact_id,
                                      agent_id=agent_id, contradiction_type="deny",
                                      description=f"'{a.subject}' '{a.predicate}' '{a.object}' "
                                                  f"矛盾于 '{b.predicate}'")
                    self.fact_network.db.save_contradiction(c)
                    found += 1
                elif _is_pos(a.predicate) and _is_neg(b.predicate):
                    c = Contradiction(fact_a_id=b.fact_id, fact_b_id=a.fact_id,
                                      agent_id=agent_id, contradiction_type="deny",
                                      description=f"'{b.subject}' '{b.predicate}' '{b.object}' "
                                                  f"矛盾于 '{a.predicate}'")
                    self.fact_network.db.save_contradiction(c)
                    found += 1
        if found:
            logger.info("🔍 矛盾扫描: 发现 %d 条新矛盾", found)
        return found

    # ═══════════════════════════════════════════════════════════════
    # ★ v0.25: 叙事跨会话链接
    # ═══════════════════════════════════════════════════════════════

    def _link_narratives(self, agent_id: str, new_facts: list):
        """
        新存储的叙事事实与已有叙事事实建立 connected_facts。

        跨会话叙事链接：第1章"王磊" → 第3章"同样有王磊" → 建链
        这样 recall 时通过 connected_facts 能找回所有关联章节。
        """
        new_narratives = [f for f in new_facts if f.fact_type == "narrative"]
        if not new_narratives:
            return

        all_facts = self.fact_network._get_agent_facts(agent_id)
        existing_narratives = [f for f in all_facts
                               if f.fact_type == "narrative"
                               and f.fact_id not in {nf.fact_id for nf in new_narratives}]
        if not existing_narratives:
            return

        linked = 0
        _SKIP_TAGS = frozenset({"长文本", "长文本摘要", "叙事", "情感", "positive", "negative"})

        for new_n in new_narratives:
            new_tags = set(t for t in new_n.context_tags if t not in _SKIP_TAGS)
            # 从 evidence 中提取实体词用于匹配
            _new_ev_text = ""
            for ev in new_n.evidence:
                if isinstance(ev, EvidenceItem):
                    _new_ev_text += (ev.statement or "") + " "

            for old in existing_narratives:
                old_tags = set(t for t in old.context_tags if t not in _SKIP_TAGS)
                shared = new_tags & old_tags

                # 证据文本中的共享实体匹配
                if not shared:
                    _old_ev_text = ""
                    for ev in old.evidence:
                        if isinstance(ev, EvidenceItem):
                            _old_ev_text += (ev.statement or "") + " "
                    # 提取两个叙事中都出现的中文双字组实体（如"王磊"）
                    import re as _re
                    _words_new = set(_re.findall(r'[一-鿿]{2}', _new_ev_text))
                    _words_old = set(_re.findall(r'[一-鿿]{2}', _old_ev_text))
                    _common = _words_new & _words_old
                    # 过滤常用虚词
                    _skip_words = frozenset({
                        "一个", "没有", "我们", "他们", "自己", "这个", "那个", "什么",
                        "怎么", "可以", "知道", "就是", "不是", "但是", "因为", "所以",
                        "时候", "突然", "然后", "起来", "出现", "发现", "看见", "听到",
                        "觉得", "感觉", "开口", "询问", "回答", "离开", "走进", "走出",
                        "先生", "小姐", "女士", "对方", "眼前", "时代", "这封", "一天",
                        "再次", "继续", "终于", "开始", "一直", "一起", "下面", "上面",
                    })
                    _common = {w for w in _common if w not in _skip_words}
                    # 优先选 2 字的人名风格实体
                    _candidates = sorted(_common, key=lambda w: len(w))
                    if _candidates:
                        shared = {_candidates[0]}

                if shared:
                    tag = next(iter(shared))
                    if new_n.fact_id not in old.connected_facts:
                        old.connected_facts = list(old.connected_facts) + [new_n.fact_id]
                        self.fact_network._cache_put(old)
                    if old.fact_id not in new_n.connected_facts:
                        new_n.connected_facts = list(new_n.connected_facts) + [old.fact_id]
                        self.fact_network._cache_put(new_n)
                    linked += 1
                    logger.info("🔗 叙事链接: %s ↔ %s 共享=%s",
                                new_n.fact_id[:8], old.fact_id[:8], tag)

        if linked:
            # 同步到 DB
            db = getattr(self.fact_network, 'db', None)
            if db:
                for f in new_narratives + existing_narratives:
                    try:
                        db.update_fact(f)
                    except Exception:
                        pass
            logger.info("🔗 叙事跨会话链接完成: %d 条链接", linked)

    # ═══════════════════════════════════════════════════════════════
    # ★ P1-3: 知识库模块（MIRIX Knowledge Vault 启发）
    # ═══════════════════════════════════════════════════════════════

    def remember_credential(self, service: str, credential: str,
                            agent_id: str = "default") -> dict:
        """安全存储凭证（密码/API Key/密钥等）。

        与普通 remember 的区别：
        - 存储为 fact_type='credential'
        - 对象值做简单 XOR 混淆（非真正加密，仅供基础保护）
        - 普通 recall 不会返回 credential 数据
        - 必须通过 recall_credential() 明确调用

        Args:
            service: 服务名称（如 "微信", "GitHub", "openai_api_key"）
            credential: 凭证原文
            agent_id: Agent ID
        """
        # 简单混淆（XOR + base64，防明文存储）
        import base64
        key = 0x5A
        masked_bytes = bytes(c ^ key for c in credential.encode('utf-8'))
        obfuscated = base64.b64encode(masked_bytes).decode('ascii')

        fact = FactTriple(
            subject=service,
            predicate="凭证",
            object=obfuscated,
            agent_id=agent_id,
            fact_type="credential",
            confidence=1.0,
            importance=0.9,
            encoding_level="credential",
            evidence=[EvidenceItem(
                source="credential_store",
                statement=f"知识库: {service} 的凭证已安全存储",
            )],
        )

        # 跳过矛盾检测和进化（凭证不产生矛盾，不参与链接）
        fn = self.fact_network
        # 按 subject 查找已有凭证（而非 triple_key，因 object 是混淆值每次不同）
        existing = None
        for f in fn._get_agent_facts(agent_id):
            if f.subject == service and f.fact_type == "credential":
                existing = f
                break
        if existing:
            # 已存在 → 更新
            existing.object = obfuscated
            existing.confidence = 1.0
            existing.accessed_at = datetime.now(timezone.utc).isoformat()
            fn._cache_put(existing)
            if fn.db:
                fn.db.update_fact(existing)
            logger.info(f"🔐 Credential updated: {service}")
            return {"status": "updated", "service": service}

        # 新凭证
        fn._cache_put(fact)
        fn._add_to_stm(fact)
        if fn.db:
            try:
                fn.db.save_fact(fact)
            except Exception as e:
                logger.error("Credential DB save failed: %s", e)
        logger.info(f"🔐 Credential stored: {service}")
        return {"status": "stored", "service": service}

    def recall_credential(self, service: str,
                          agent_id: str = "default") -> dict:
        """安全召回凭证。

        只有通过此方法才能获取解密后的凭证原文。
        普通 recall 不会泄露 credential 数据。

        Args:
            service: 服务名称
            agent_id: Agent ID

        Returns:
            {"service": "", "credential": "", "status": "found"|"not_found"}
        """
        import base64
        fn = self.fact_network
        # 精确查找 credential 事实
        for f in fn._get_agent_facts(agent_id):
            if f.subject == service and f.fact_type == "credential":
                # 解码
                try:
                    masked = base64.b64decode(f.object.encode('ascii'))
                    key = 0x5A
                    decoded = bytes(b ^ key for b in masked).decode('utf-8')
                except Exception:
                    decoded = f.object  # 解码失败返回原始混淆值
                return {
                    "status": "found",
                    "service": service,
                    "credential": decoded,
                    "safe_display": f.safe_display,
                }
        return {"status": "not_found", "service": service}

    def list_credentials(self, agent_id: str = "default") -> list[dict]:
        """列出所有已存储的凭证（不泄露原文）。"""
        services = []
        for f in self.fact_network._get_agent_facts(agent_id):
            if f.fact_type == "credential":
                services.append({
                    "service": f.subject,
                    "safe_display": f.safe_display,
                    "created_at": f.created_at[:10],
                })
        return services

    def get_stats(self, agent_id: str = "default") -> dict:
        """获取统计信息（v0.12 含 STM 计数）"""
        facts = self.fact_network._get_agent_facts(agent_id)
        contradictions = self.fact_network.get_contradictions(agent_id)
        return {
            "agent_id": agent_id,
            "total_facts": len(facts),
            "core_beliefs": len([f for f in facts if f.is_core_belief]),
            "unreliable": len([f for f in facts if f.is_unreliable]),
            "contradictions": len(contradictions),
            "by_type": self._count_by_type(facts),
            "router_stats": self.recall_router.get_stats(),
            # ★ v0.12 新增
            "stm_buffer": self.fact_network._stm_count(agent_id),
        }

    def _count_by_type(self, facts: list) -> dict:
        counts = {}
        for f in facts:
            counts[f.fact_type] = counts.get(f.fact_type, 0) + 1
        return counts
