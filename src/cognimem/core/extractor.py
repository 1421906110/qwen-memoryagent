"""
事实提取器 — 从自然语言提取三元组（v0.20 全面加强版）

改进：
1. 谓词从 39 → 68 个
2. 所有匹配模式都收集，不只取第一个
3. 同类型谓词合并（"爱吃"="喜欢"）
4. 仅当无任何模式匹配时才 fallback 到 observation
"""

import re
import logging
from datetime import datetime, timezone
from typing import Any
from .models import FactTriple, EvidenceItem

logger = logging.getLogger(__name__)


class TripleExtractor:
    """
    三元组提取器 v0.20

    策略:
    1. 规则匹配: 收集所有命中的模式
    2. 无模式命中 → observation fallback
    """

    # ═══ 谓词表（68 个，按语义分组）═══
    # 值 = (predicate, fact_type)
    _PREDICATES = {
        # ── 喜好类（preference）──
        "喜欢": ("喜欢", "preference"),
        "不喜欢": ("不喜欢", "preference"),
        "爱吃": ("喜欢", "preference"),
        "爱喝": ("喜欢", "preference"),
        "爱玩": ("喜欢", "preference"),
        "爱去": ("喜欢", "preference"),
        "爱": ("喜欢", "preference"),
        "最爱": ("喜欢", "preference"),
        "超爱": ("喜欢", "preference"),
        "讨厌": ("讨厌", "preference"),
        "反感": ("讨厌", "preference"),
        "受不了": ("受不了", "preference"),
        "戒了": ("不吃了", "preference"),

        # ── 事实类（fact）──
        "是": ("是", "fact"),
        "叫": ("叫", "fact"),
        "叫做": ("叫", "fact"),
        "有": ("有", "fact"),
        "没有": ("没有", "fact"),
        "在": ("在", "fact"),
        "住在": ("住在", "fact"),
        "位于": ("在", "fact"),
        "来自": ("来自", "fact"),
        "做": ("做", "fact"),
        "从事": ("从事", "fact"),
        "工作": ("工作", "fact"),
        "学习": ("学习", "fact"),
        "去了": ("去过", "fact"),
        "去过": ("去过", "fact"),
        "到过": ("去过", "fact"),
        "参加": ("参加过", "fact"),
        "参加了": ("参加过", "fact"),
        "参与": ("参与过", "fact"),
        "参与了": ("参与过", "fact"),
        "组织": ("组织过", "fact"),
        "组织了": ("组织过", "fact"),
        "访问": ("访问过", "fact"),
        "访问了": ("访问过", "fact"),
        "参观": ("参观过", "fact"),
        "参观了": ("参观过", "fact"),
        "出席": ("出席过", "fact"),
        "出席了": ("出席过", "fact"),
        "搬到": ("搬到", "fact"),        # 🐛 v0.29: 搬家检测
        "搬去": ("搬到", "fact"),        # 🐛 v0.29: 搬家检测
        "搬到了": ("搬到", "fact"),      # 🐛 v0.29: 搬家检测
        "搬来了": ("搬到", "fact"),      # 🐛 v0.29: 搬家检测
        "调高": ("调高", "fact"),        # 🐛 v0.29: 数值状态
        "调高到": ("调高到", "fact"),    # 🐛 v0.29: 数值状态
        "调低": ("调低", "fact"),        # 🐛 v0.29: 数值状态
        "调低到": ("调低到", "fact"),    # 🐛 v0.29: 数值状态
        "调到": ("调到", "fact"),        # 🐛 v0.30: 数值状态初始值（状态机 base）
        "调成": ("调到", "fact"),        # 🐛 v0.30: 同上
        "设为": ("设置为", "fact"),      # 🐛 v0.30: 初始值设置（状态机 base）
        "设置为": ("设置为", "fact"),    # 🐛 v0.30: 同上
        "设定": ("设置为", "fact"),      # 🐛 v0.30: 同上
        "设定为": ("设置为", "fact"),    # 🐛 v0.30: 同上
        "买了": ("买了", "fact"),
        "换了": ("换了", "fact"),        # 🐛 v0.30: Q13 "后来换了MacBook"
        "换成": ("换了", "fact"),        # 🐛 v0.30: 同上
        "用过": ("用过", "fact"),
        "吃过": ("吃过", "fact"),
        "喝过": ("喝过", "fact"),
        "看过": ("看过", "fact"),
        "读过": ("读过", "fact"),
        "学过": ("学过", "fact"),
        "我负责": ("负责", "fact"),
        "负责": ("负责", "fact"),           # 逗号延续：负责用户增长
        "担任": ("担任", "fact"),           # 担任前端架构师
        "目标是": ("目标是", "goal"),         # 目标是高级工程师
        "理想是": ("理想是", "goal"),         # 理想是成为CTO
        "打算": ("打算", "goal"),
        "打算去": ("打算去", "goal"),

        # ── 能力类（skill）──
        "会": ("会", "skill"),
        "不会": ("不会", "skill"),
        "能": ("能", "skill"),
        "不能": ("不能", "skill"),
        "可以": ("可以", "skill"),
        "擅长": ("擅长", "skill"),
        "精通": ("擅长", "skill"),
        "会做": ("会做", "skill"),
        "会用": ("会用", "skill"),

        # ── 目标/意愿类（goal）──
        "想": ("想", "goal"),
        "想要": ("想要", "goal"),
        "想去": ("想去", "goal"),
        "想吃": ("想吃", "goal"),
        "想喝": ("想喝", "goal"),
        "想玩": ("想玩", "goal"),
        "想看": ("想看", "goal"),
        "想学": ("想学", "goal"),
        "要": ("要", "goal"),
        "不要": ("不要", "goal"),
        "需要": ("需要", "goal"),
        "不需要": ("不需要", "goal"),
        "打算": ("打算", "goal"),
        "计划": ("计划", "goal"),
        "准备": ("准备", "goal"),

        # ── 感受类（observation）──
        "觉得": ("觉得", "observation"),
        "认为": ("认为", "observation"),
        "感觉": ("感觉", "observation"),
        "挺好": ("挺好", "preference"),          # 🐛 v0.29: 情感补全
        "挺香": ("挺香", "preference"),          # 🐛 v0.29: "还挺香的"
        "好喝": ("好喝", "preference"),          # 🐛 v0.29: 真好喝
        "好吃": ("好吃", "preference"),          # 🐛 v0.29: 很好吃

        # ── 决策类（decision）──
        "决定": ("决定", "decision"),
        "选择": ("选择", "decision"),
        "选了": ("选了", "decision"),
        "决定用": ("决定用", "decision"),
        "决定买": ("决定买", "decision"),
        "决定去": ("决定去", "decision"),
    }

    # ═══ 同义词映射（多个词映射到同一个 predicate）═══
    _SYNONYM_MAP = {
        "爱喝": "喜欢", "爱吃": "喜欢", "爱玩": "喜欢", "爱去": "喜欢",
        "最爱": "喜欢", "超爱": "喜欢",
        "想喝": "想", "想吃": "想", "想看": "想", "想玩": "想", "想学": "想",
        "没见过": "没去过",
        "叫": "是", "叫做": "是",
        "位于": "在",
        "不会做": "不会", "不会用": "不会",
    }

    # ═══ 疑问词（v0.30）═══
    # 宾语含疑问词 → 是提问不是陈述，不存储
    _QUESTION_WORDS = ("多少", "什么", "哪个", "哪些", "哪里", "哪儿",
                       "怎么", "为什么", "是否", "是不是", "吗", "呢",
                       "几", "谁", "何时")

    # ═══ 修正检测模式（v0.24 新增）═══
    _CORRECTION_PATTERNS = [
        # "实际是X" / "其实是X"
        (r"(?:实际|其实|应该)(?:上|来说)?是(.+?)(?:[，。！？,!?]|$)", True),
        # "不是X，是Y" / "不是X，Y"
        (r"不(?:对|是)(.+?)(?:[，。])?(?:，)?(?:实际|其实|应该|而)?是(.+?)(?:[，。！？,!?]|$)", False),
        # "说错了，是X" / "弄错了，是X"
        (r"(?:说错|弄错|记错|搞错)[了了]?[，,]\s*(?:实际|其实|应该)?(?:是)?(.+?)(?:[，。！？,!?]|$)", True),
        # "更正一下，X" / "修正一下，X"
        (r"(?:更正|修正|纠正)(?:一下|一下下)?[，,]\s*(.+?)(?:[，。！？,!?]|$)", True),
    ]

    def extract(self, text: str, source: str = "",
                agent_id: str = "default") -> list[FactTriple]:
        """
        从文本中提取所有可能的三元组。

        ⭐ 收集所有模式匹配结果，不取最先匹配的那个。
        ⭐ v0.24: 新增修正意图识别。
        """
        facts = []
        if text is None:
            return facts
        text = text.strip()
        if not text:
            return facts

        # 🐛 v0.28: 时间词开头的陈述句自动补"我"（"上周参加了…"→"我上周参加了…"）
        # 使得模式1的时间词间隔模式能匹配到谓词
        _t_start = r"^(?:上[周月天次]|这[周月天]|下[周月天]|本[周月天]|昨[天夜]|今[天夜]|前[天]|后[天]|最近|刚刚|以前|曾经|已经)"
        if re.match(_t_start, text) and not re.match(r'^(?:我|你|用户|我们|您)', text):
            text = "我" + text

        # 🆕 策略0: 修正检测（v0.24）— 先于其他规则，优先级最高
        correction_facts = self._extract_correction(text, source, agent_id)
        if correction_facts:
            facts.extend(correction_facts)
            return facts  # 修正模式匹配后直接返回，不走其他提取

        # 策略1: 规则匹配 — 收集所有命中
        pattern_facts, raw_obs_facts = self._extract_all(text, source, agent_id)
        # 🐛 v0.30: 统一过滤疑问句垃圾事实
        #   "我的音量实际是多少？" → 旧代码提取 (用户, 的智能音箱音量实际, 多少)
        #   提问不是陈述，不该存储。覆盖所有模式，而不是逐个模式打补丁。
        pattern_facts = self._filter_question_facts(pattern_facts)
        facts.extend(pattern_facts)

        # 🆕 v0.25 — 策略1.5: 情感极性检测
        # 检测用户对实体的情感倾向（无需"我"触发）
        # "苹果生态系统真的强" → (苹果, 评价, 正面)
        # "小米广告太多了" → (小米, 评价, 负面)
        sentiment_facts = self._extract_sentiment(text, source, agent_id)
        facts.extend(sentiment_facts)

        # 🆕 v0.25 — 策略1.8: 长文本叙事检测（不用 LLM 也能存叙事事实）
        # 当规则提取只产出了低质量结果（片段垃圾匹配）时
        # 或者文本有叙事标记（第X章）时，用长文本提取代替
        _has_chapter_marker = bool(re.search(r'第[一二三四五六七八九十\d]+[章节回部]', text))
        _is_longish = len(text) > 80 or (_has_chapter_marker and len(text) > 20)
        if _is_longish and not sentiment_facts:
            # 检测结果质量：subject含标点/超长/片段
            _bad_subject = lambda s: (
                any(c in s for c in "。，？！；：、""''")
                or len(s) > 5
                or (len(s) >= 4 and s.startswith(("城", "信", "家", "房", "门", "客", "书", "桌")))
            )
            _has_bad_data = any(_bad_subject(f.subject) for f in facts) if facts else True
            if _has_bad_data and len(facts) <= 3:
                facts = []
                long_facts = self._extract_long_text_fallback(text, source, agent_id)
                facts.extend(long_facts)

        # 策略2: 原文保底 — 当 pattern_facts 质量不足时，保留原始文本作为观察
        # 质量判断：subject 超长(>6) 或 predicate 不含中文（英文碎片）→ 低质量
        if raw_obs_facts:
            _has_quality_pattern = any(
                len(f.subject) <= 6
                and any('一' <= c <= '鿿' for c in f.predicate)
                for f in pattern_facts
            ) if pattern_facts else False
            if not _has_quality_pattern:
                facts.append(raw_obs_facts[0])

        # 策略3: 真·什么都没有 → observation fallback
        if not facts:
            facts.append(FactTriple(
                subject="用户",
                predicate="说了",
                object=text[:200],
                agent_id=agent_id,
                fact_type="observation",
                confidence=0.5,
                source_session=source,
                evidence=[EvidenceItem(
                    source=source or "unknown",
                    statement=text[:500],
                )],
            ))

        return facts

    @staticmethod
    def _filter_question_facts(facts: list[FactTriple]) -> list[FactTriple]:
        """丢弃宾语是疑问词/疑问句的事实（提问不是陈述，不该存储）"""
        out = []
        for f in facts:
            if f.object.endswith(("?", "？")) or any(
                    w in f.object for w in TripleExtractor._QUESTION_WORDS):
                continue
            out.append(f)
        return out

    # ═══ v0.24: 修正意图提取 ═══
    def _extract_correction(self, text: str, source: str,
                            agent_id: str) -> list[FactTriple]:
        """
        提取修正意图中的事实。
        "抱歉我上次说错了实际是6月10日"
        → (用户, 生日是, 6月10日)  [从原文推断predicate]
        → context_tags=["修正"]
        """
        facts = []

        # ① 模式匹配
        best_match = None  # (matched_text, new_value, is_single)
        for pat, is_single in self._CORRECTION_PATTERNS:
            m = re.search(pat, text)
            if m:
                if is_single:
                    new_val = m.group(1).strip()
                    best_match = (m.group(0), new_val, True)
                else:
                    # "不是X，是Y" 模式 → 旧值X, 新值Y
                    old_val, new_val = m.group(1).strip(), m.group(2).strip()
                    best_match = (m.group(0), new_val, False)
                break

        if not best_match:
            return []

        matched_text, new_value, is_single = best_match
        if not new_value or len(new_value) > 40:
            return []
        # 🐛 v0.30: 疑问句不是修正！"实际是多少？"是提问，不是更正
        #   旧代码提取出 (用户, 是, 多少) 垃圾事实
        if new_value.endswith(("?", "？")) or any(
                w in new_value for w in self._QUESTION_WORDS):
            logger.debug("⏭️ 修正模式命中疑问句，跳过: %r", new_value)
            return []

        # ② 从原文推断 predicate
        # "生日是5月10日" → 有"生日"，用"生日是"
        # "实际是6月10日" → 原文有"生日/名字/年龄"等关键词
        inferred_predicate = self._infer_correction_predicate(text)

        fact = FactTriple(
            subject="用户",
            predicate=inferred_predicate or "是",
            object=new_value,
            agent_id=agent_id,
            fact_type="fact",
            confidence=0.8,  # 修正的事实用更高置信度
            importance=0.7,
            source_session=source,
            context_tags=["修正"],
            evidence=[EvidenceItem(
                source=source or "correction",
                statement=text[:500],
            )],
        )
        facts.append(fact)

        # ③ "不是X，是Y" 模式：额外存一条旧值否定
        if not is_single:
            # 找到 matched_text 中第一个匹配的旧值
            old_match = re.search(r"不(?:对|是)(.+?)(?:[，。])", matched_text)
            if old_match:
                old_val = old_match.group(1).strip()
                if old_val:
                    old_neg = FactTriple(
                        subject="用户",
                        predicate=inferred_predicate or "不是",
                        object=old_val,
                        agent_id=agent_id,
                        fact_type="fact",
                        confidence=0.5,
                        importance=0.3,
                        source_session=source,
                        context_tags=["修正", "被否定"],
                        evidence=[EvidenceItem(
                            source=source or "correction",
                            statement=text[:500],
                        )],
                    )
                    facts.append(old_neg)

        return facts

    @staticmethod
    def _infer_correction_predicate(text: str) -> str | None:
        """
        从上下文推断修正内容的谓词。

        "实际是6月10日" + 原文"我的生日是5月10日"
        → 原文有"生日" → 用"生日是"

        当前实现：从修正文本本身提取线索词。
        """
        # 线索词 → 谓词映射
        _CLUES = {
            "生日": "生日是",
            "名字": "名字是",
            "电话": "电话是",
            "地址": "地址是",
            "邮箱": "邮箱是",
            "邮件": "邮箱是",
            "公司": "公司是",
            "工作": "工作是",
            "学校": "学校是",
            "年龄": "年龄是",
            "年份": "年份是",
        }
        for clue, predicate in _CLUES.items():
            if clue in text:
                return predicate

        # "今年X岁" → 年龄是
        if re.search(r"[0-9]+岁", text):
            return "年龄是"

        # 日期的语义提示
        if re.search(r"[0-9]+月[0-9]+日", text) or re.search(r"[0-9]+-[0-9]+", text):
            # 如果原文有"生日"就返回"生日是"，否则返回"日期是"
            if "生日" in text:
                return "生日是"
            return "日期是"

        return None

    # ═══ v0.25: 情感极性提取 ═══
    def _extract_sentiment(self, text: str, source: str,
                           agent_id: str) -> list[FactTriple]:
        """
        从文本中提取实体级别的情感极性事实。

        "苹果生态系统真的强" → (苹果, 评价, 正面, confidence=0.48)
        "小米广告太多了" → (小米, 评价, 负面, confidence=0.50)

        作为规则补充：即使 pattern 匹配到了其他事实，情感也单独提取。
        """
        from .sentiment import SentimentEngine
        result = SentimentEngine.analyze(text)
        if not result:
            return []

        sentiment_val = "正面" if result["sentiment"] == "positive" else "负面"
        fact = FactTriple(
            subject=result["entity"],
            predicate="评价",
            object=sentiment_val,
            agent_id=agent_id,
            fact_type="preference",
            confidence=result["score"] * 0.6,  # 推理所得，降低初始置信度
            importance=0.4,
            context_tags=["情感", result["sentiment"]],
            source_session=source,
            evidence=[EvidenceItem(
                source=source or "sentiment_analysis",
                statement=text[:300],
            )],
        )
        logger.debug("🧠 情感提取: (%s, %s, %s) score=%.2f",
                     result["entity"], "评价", sentiment_val, result["score"])
        return [fact]

    # ═══ v0.25: 长文本叙事保底（无 LLM 时使用）═══
    def _extract_long_text_fallback(self, text: str, source: str,
                                     agent_id: str) -> list[FactTriple]:
        """
        当无 LLM 可用时，为长文本生成一个合理的摘要事实。
        用简单的规则检测是否为叙事，并尝试提取人物名作为标签。
        """
        from .llm_extractor import LLMTripleExtractor
        is_narrative = LLMTripleExtractor._detect_narrative(text)
        characters = LLMTripleExtractor._extract_characters(text) if is_narrative else []

        _tags = ["长文本"]
        _ftype = "observation"
        if is_narrative:
            _tags += ["叙事"] + characters
            _ftype = "narrative"

        facts = [FactTriple(
            subject="用户",
            predicate="提供了",
            object=text[:500],
            agent_id=agent_id,
            fact_type=_ftype,
            confidence=0.4,
            importance=0.3,
            source_session=source,
            context_tags=_tags,
            evidence=[EvidenceItem(
                source=source or "long_text",
                statement=text[:1000],
            )],
        )]
        if is_narrative:
            logger.info("📖 叙事检测(无LLM): %d字 角色=%s", len(text), characters)
        return facts

    def _extract_all(self, text: str, source: str,
                     agent_id: str) -> tuple[list[FactTriple], list[FactTriple]]:
        """
        收集所有模式匹配的结果。
        Returns: (pattern_facts, raw_observations)
        """
        pattern_facts = []
        raw_observations = []
        seen_triples = set()  # 去重

        # ── 模式1: "我/用户/你 + [程度副词] + 谓词 + 宾语" ──
        # 🆕 v0.26: 支持"我最喜欢Python""我特别讨厌加班"等程度副词结构
        # 🐛 v0.30: 长谓词优先（"调高到"先于"调高"），已消费的文本区间
        #   不允许短谓词再次命中 → "我把音量调高到了60%" 只产
        #   (调高到, 音量60%)，不再多产 (调高, 音量到了60%) 冗余事实
        _consumed_spans: list[tuple[int, int]] = []
        for word, (pred, ftype) in sorted(self._PREDICATES.items(),
                                          key=lambda kv: -len(kv[0])):
            _adv = r"(?:最(?!近)|很|非常|特别|有点|比较|真|超|更|太|极[其|为]|又|也|还|顺手|终于|就|才)?"
            # 🐛 v0.28: 时间词隔开也匹配（"我上周参加了"→"参加"在"上周"后面）
            #   ⚠️ 必须用精确的时间词列表，不能用 [^，。] 通配（会误配"我想在"）
            _t = r"(?:上[周月天次]|这[周月天]|下[周月天]|本[周月天]|去[年月]|今[年月天]|明[年天]|昨[天夜]|前[天]|后[天]|最近|刚刚|以前|曾经|已经)?"
            patterns = [
                rf"(?:我|我们|用户|[他她]){_adv}\s*{re.escape(word)}(?:了)?\s*(.+?)(?:[，。！？,!?]|$)",
                rf"(?:你|您){_adv}\s*{re.escape(word)}(?:了)?\s*(.+?)(?:[，。！？,!?]|$)",
                rf"(?:我|我们|用户|[他她]){_adv}{_t}{re.escape(word)}(?:了)?\s*(.+?)(?:[，。！？,!?]|$)",
                rf"(?:你|您){_adv}{_t}{re.escape(word)}(?:了)?\s*(.+?)(?:[，。！？,!?]|$)",
                # 🐛 v0.29: "我把X调高了" — 宾语前置结构（我把音量调低到30%）
                # ⚠️ f-string 里 {1,12} 必须写成 {{1,12}}，否则会被当作格式占位符！
                # 🐛 v0.30: 主语与"把"之间允许修饰词（"我又把音量调低了15个百分点"）
                rf"(?:我|我们|用户|[他她])(?:又|也|还|顺手|终于|就|才|{_t})?把(.{{1,12}}){re.escape(word)}(?:了)?\s*(.+?)(?:[，。！？,!?]|$)",
            ]
            _spans_this_word: list[tuple[int, int]] = []
            for _pi, pat in enumerate(patterns):
                for m in re.finditer(pat, text):
                    # 🐛 v0.30: 命中区间已被更长谓词消费 → 跳过
                    if any(m.start() >= s and m.end() <= e
                           for s, e in _consumed_spans):
                        continue
                    # 🐛 v0.29: "我把X调低"模式有2个捕获组：
                    # group(1)=X(把的宾语)，group(2)=数值 → object = X+数值
                    if _pi >= 4:
                        _g1 = m.group(1).strip() if m.group(1) else ""
                        _g2 = m.group(2).strip() if m.group(2) else ""
                        obj = (_g1 + _g2).strip() if _g1 and _g2 else (_g1 or _g2)
                    else:
                        obj = m.group(1).strip()
                    if not obj or len(obj) > 60:
                        continue
                    # 规范化谓词（同义词映射）
                    final_pred = self._SYNONYM_MAP.get(word, pred)
                    key = (agent_id, "用户", final_pred, obj)
                    if key in seen_triples:
                        continue
                    seen_triples.add(key)
                    pattern_facts.append(FactTriple(
                        subject="用户",
                        predicate=final_pred,
                        object=obj,
                        agent_id=agent_id,
                        fact_type=ftype,
                        source_session=source,
                        evidence=[EvidenceItem(
                            source=source or "unknown",
                            statement=text[:300],
                        )],
                        context_tags=self._extract_tags(text),
                    ))
                    _spans_this_word.append((m.start(), m.end()))
            # 本词条所有命中区间标记为已消费（拦截更短的后续谓词）
            _consumed_spans.extend(_spans_this_word)

        # ── 模式2: 实体关系（X的Y很Z）──
        # "今天天气很好" → (今天, 天气, 很好)
        # ⚠️ 先检查"我的本职是X/我的专业是X"等定义式结构，避免被通用模式切碎
        for m in re.finditer(r"我的(?:本职工作|专业|职业|工作|本职)(?:是|在)(.+?)(?:[，。！？,!?]|$)", text):
            job = m.group(1).strip()
            if job and len(job) <= 40:
                key = (agent_id, "用户", "工作是", job)
                if key not in seen_triples:
                    seen_triples.add(key)
                    pattern_facts.append(FactTriple(
                        subject="用户", predicate="工作是", object=job,
                        agent_id=agent_id, fact_type="fact",
                        source_session=source,
                        evidence=[EvidenceItem(source=source or "unknown", statement=text[:300])],
                    ))
        # ── 模式3: "我叫X" / "名字是X" ──
        for m in re.finditer(r"(?:我|用户)(?:的)?(?:名字|昵称|姓名|全名)(?:是|叫)?(.+?)(?:[，。！？,!?]|$)", text):
            name = m.group(1).strip()
            if name and len(name) <= 20:
                key = (agent_id, "用户", "名字", name)
                if key not in seen_triples:
                    seen_triples.add(key)
                    pattern_facts.append(FactTriple(
                        subject="用户",
                        predicate="名字",
                        object=name,
                        agent_id=agent_id,
                        fact_type="fact",
                        source_session=source,
                        evidence=[EvidenceItem(
                            source=source or "unknown",
                            statement=text[:300],
                        )],
                    ))

        # ── 模式4: "我在X工作" / "我住在X" ──
        for m in re.finditer(r"我在(.+?)(?:工作|上班|学习|读书)(?:[，。！？,!?]|$)", text):
            place = m.group(1).strip()
            if place:
                key = (agent_id, "用户", "工作在", place)
                if key not in seen_triples:
                    seen_triples.add(key)
                    pattern_facts.append(FactTriple(
                        subject="用户", predicate="工作在", object=place,
                        agent_id=agent_id, fact_type="fact",
                        source_session=source,
                        evidence=[EvidenceItem(source=source or "unknown", statement=text[:300])],
                    ))
        # 目前在X就职 / 目前在X任职
        for m in re.finditer(r"(?:目前|现在)(?:在)(.+?)(?:就职|任职)(?:[，。！？,!?]|$)", text):
            place = m.group(1).strip()
            if place:
                key = (agent_id, "用户", "工作在", place)
                if key not in seen_triples:
                    seen_triples.add(key)
                    pattern_facts.append(FactTriple(
                        subject="用户", predicate="工作在", object=place,
                        agent_id=agent_id, fact_type="fact",
                        source_session=source,
                        evidence=[EvidenceItem(source=source or "unknown", statement=text[:300])],
                    ))

        # ── 模式5: 数量/度量（"我今年X岁" / "今年X岁" / "我的Y是Z个"）──
        for m in re.finditer(r"(?:我)?今年(.+?)(?:岁|岁了)(?:[，。！？,!?]|$)", text):
            age = m.group(1).strip()
            if age:
                key = (agent_id, "用户", "年龄", age)
                if key not in seen_triples:
                    seen_triples.add(key)
                    pattern_facts.append(FactTriple(
                        subject="用户", predicate="年龄", object=age,
                        agent_id=agent_id, fact_type="fact",
                        source_session=source,
                        evidence=[EvidenceItem(source=source or "unknown", statement=text[:300])],
                    ))

        # 🆕 v0.25: 模式5.5: 我X是Y（X为属性/度量/状态）
        # 🐛 v0.27: 跳过"帮我记住："等命令前缀导致的错误匹配（"帮我"→prop=记住：我的数据库）
        for m in re.finditer(r"(?<!帮)我([^，。！？]{2,10})是(.+?)(?:[，。！？,!?]|$)", text):
            prop = m.group(1).strip()
            val = m.group(2).strip()
            # 🐛 v0.30: 剥离"的"前缀（"我的银行卡后四位是2210" → "银行卡后四位"）
            #   旧代码 prop="的银行卡后四位" 带垃圾前缀
            if prop.startswith("的"):
                prop = prop[1:].strip()
            # 验证 prop 不含标点（防止"记住：我的数据库"等错误提取）
            _has_punct = any(c in prop for c in "：:；;，。！？、【】《》\"'")
            if _has_punct:
                continue
            if prop and val and len(val) <= 40 and prop not in ("", "就", "才", "正", "还"):
                key = (agent_id, "用户", prop, val)
                if key not in seen_triples:
                    seen_triples.add(key)
                    pattern_facts.append(FactTriple(
                        subject="用户", predicate=prop, object=val,
                        agent_id=agent_id, fact_type="fact",
                        source_session=source,
                        evidence=[EvidenceItem(source=source or "unknown", statement=text[:300])],
                    ))

        # 🐛 v0.30: 模式5.6: "X设为Y" / "X设置为Y" — 数值状态初始值
        #   "我的智能音箱音量设为30%" → (用户, 设置为, 智能音箱音量30%)
        #   状态机的 _base_val 依赖此谓词，缺了初始值会被当操作增量算错
        for m in re.finditer(r"([^，。！？]{1,12}?)(?:设为|设置为|设定为|调整为)(.+?)(?:[，。！？,!?]|$)", text):
            _sprop = m.group(1).strip()
            _sval = m.group(2).strip()
            if _sprop and _sval and len(_sval) <= 30:
                # 去"我的/我们的"前缀（"我的智能音箱音量"→"智能音箱音量"）
                _sprop = re.sub(r'^(?:我的|我们的|我)', '', _sprop)
                if _sprop:
                    _snum = re.findall(r'(\d+(?:\.\d+)?)', _sval)
                    if _snum:  # 只存带数字的设置（"设为30%"），避免"设为静音"误提取
                        key = (agent_id, "用户", "设置为", _sprop + _sval)
                        if key not in seen_triples:
                            seen_triples.add(key)
                            pattern_facts.append(FactTriple(
                                subject="用户", predicate="设置为", object=_sprop + _sval,
                                agent_id=agent_id, fact_type="fact",
                                source_session=source,
                                evidence=[EvidenceItem(source=source or "unknown", statement=text[:300])],
                            ))

        # 🐛 v0.29 修复(Q17): 联系信息提取 — "地址/电话/门禁"逗号分列
        # "我的地址是科苑路15号3楼，联系电话138****2210，门禁按502。"
        # → (用户, 地址, 科苑路15号3楼) (用户, 联系电话, 138****2210) (用户, 门禁, 502)
        _CONTACT_PATS = [
            (r'(?:地址|住址)(?:是|为|在)?\s*([^，,。]{2,40})', "地址"),
            (r'(?:联系电话|电话|手机|联系方式)(?:是|为|：|:)?\s*([0-9*#+()\- ]{5,25})', "联系电话"),
            (r'(?:门禁|门禁号|门禁密码|门牌)(?:是|为|按|：|:)?\s*([0-9*#]{1,10})', "门禁"),
        ]
        for _cpat, _cpred in _CONTACT_PATS:
            for _cm in re.finditer(_cpat, text):
                _cval = _cm.group(1).strip()
                if _cval and len(_cval) <= 40:
                    key = (agent_id, "用户", _cpred, _cval)
                    if key not in seen_triples:
                        seen_triples.add(key)
                        pattern_facts.append(FactTriple(
                            subject="用户", predicate=_cpred, object=_cval,
                            agent_id=agent_id, fact_type="fact",
                            source_session=source,
                            evidence=[EvidenceItem(source=source or "unknown", statement=text[:300])],
                            context_tags=[_cpred, "联系信息"],
                        ))

        # 🐛 v0.29 修复(Q2): 序列提取 — "先X，再Y，最后是Z"
        # "我家三只猫的绝育顺序是：先小黑，再橘座，最后是警长。"
        # → (#1, 绝育顺序, 小黑) (#2, 绝育顺序, 橘座) (#3, 绝育顺序, 警长)
        _seq_m = re.search(r'([^，。！？]{2,8})顺序(?:是|为)[：:]\s*(.+)', text)
        if _seq_m:
            _seq_name_raw = _seq_m.group(1).strip()
            # 提取核心名词（去掉"的""我的""我们"等前缀）
            _seq_name = re.sub(r'^(?:我|我的|我们|你们的|你|你的)?(?:的)?', '', _seq_name_raw)
            if not _seq_name:
                _seq_name = _seq_name_raw
            _seq_body = _seq_m.group(2).strip()
            _seq_items = re.split(r'[，,]\s*', _seq_body)
            _seq_order = 1
            for _item in _seq_items:
                _item_clean = re.sub(r'^(?:先|然后|再|最后|接着|随后|接下来|第一步|第二步|第三步|第四步|最后一步)(?:是|再|来)?', '', _item).strip()
                _item_clean = _item_clean.rstrip('。.!！，,')
                if _item_clean and len(_item_clean) <= 20:
                    _seq_key = f"#{_seq_order}"
                    key = (agent_id, _seq_key, _seq_name, _item_clean)
                    if key not in seen_triples:
                        seen_triples.add(key)
                        pattern_facts.append(FactTriple(
                            subject=_seq_key, predicate=_seq_name, object=_item_clean,
                            agent_id=agent_id, fact_type="fact",
                            source_session=source,
                            evidence=[EvidenceItem(source=source or "unknown", statement=text[:300])],
                            context_tags=[_seq_name, "顺序", "序列"],
                        ))
                    _seq_order += 1

        # 🐛 v0.29 修复(Q12): 条件规则提取 — "但如果X，可以Y"/"要是X就Y"
        # "但如果诺兰导演的，可以破例。" → (用户, 条件规则, 诺兰导演的 可以破例)
        _cond_pats = [
            r'(?:但)?(?:如果|要是)(.+?)[，,]\s*(?:就|则|可以|才能|才能)?(.+?)(?:[，。！？]|$)',
        ]
        for _cpat in _cond_pats:
            _cm = re.search(_cpat, text)
            if _cm:
                _cond = _cm.group(1).strip()
                _action = _cm.group(2).strip()
                if _cond and _action and len(_cond) <= 30 and len(_action) <= 30:
                    _cond_text = f"如果{_cond}，则{_action}"
                    key = (agent_id, "用户", "条件规则", _cond_text)
                    if key not in seen_triples:
                        seen_triples.add(key)
                        pattern_facts.append(FactTriple(
                            subject="用户", predicate="条件规则", object=_cond_text,
                            agent_id=agent_id, fact_type="decision",
                            source_session=source,
                            evidence=[EvidenceItem(source=source or "unknown", statement=text[:300])],
                            context_tags=["规则", "条件"],
                        ))

        # ── 🆕 模式5.7: 决策/事件提取（第三人称事实）──
        # "CTO陈刚否决了项目Alpha的微服务方案" → (CTO陈刚, 否决, 项目Alpha微服务方案)
        # "王强提议项目Alpha采用微服务架构"    → (王强, 提议, 项目Alpha采用微服务架构)
        _DECISION_VERBS = [
            ("否决(?:了)?", "否决"),
            ("批准(?:了)?", "批准"),
            ("提议", "提议"),
            ("提出", "提出"),
            ("分享", "分享"),
            ("暂停(?:了)?", "暂停"),
            ("冻结(?:了)?", "冻结"),
            ("取消(?:了)?", "取消"),
            ("决定", "决定"),
            ("宣布(?:了)?", "宣布"),
            ("通过(?:了)?", "通过"),
            ("拒绝(?:了)?", "拒绝"),
            ("反对", "反对"),
            ("支持", "支持"),
            ("建议", "建议"),
            ("叫停(?:了)?", "叫停"),
            ("启动(?:了)?", "启动"),
            ("推迟(?:了)?", "推迟"),
            ("解冻(?:了)?", "解冻"),
            ("扩招(?:到|了)?", "扩招"),
            ("裁员(?:了)?", "裁员"),
            ("迁(?:移到|到|至)", "迁移"),
            ("切换(?:到|至|为)", "切换"),
        ]
        for verb_pat, verb_pred in _DECISION_VERBS:
            for m in re.finditer(
                r"([一-鿿\w]{2,12})" + verb_pat + r"(.+?)(?:[，。！？；]|$)",
                text
            ):
                subj = m.group(1).strip()
                obj = m.group(2).strip()
                if not subj or not obj:
                    continue
                # 跳过第一人称：模式1已处理
                if subj in ("我", "你", "用户", "我们", "你们", "您"):
                    continue
                # 去掉宾语末尾标点
                obj = re.sub(r'[，。！？；：、]+$', '', obj)
                if len(obj) > 120 or len(obj) < 2:
                    continue
                # 主语不能以连词/标点开头
                if subj[0] in ("的", "了", "过", "在", "是", "有", "，", "。", "和", "与"):
                    continue
                # 主语非纯ASCII（必须是中文名或中英混合）
                _zh_chars = sum(1 for c in subj if '一' <= c <= '鿿')
                if _zh_chars == 0 and len(subj) > 4:
                    continue
                key = (agent_id, subj, verb_pred, obj)
                if key in seen_triples:
                    continue
                seen_triples.add(key)
                pattern_facts.append(FactTriple(
                    subject=subj,
                    predicate=verb_pred,
                    object=obj,
                    agent_id=agent_id,
                    fact_type="decision",
                    confidence=0.7,
                    source_session=source,
                    evidence=[EvidenceItem(
                        source=source or "unknown",
                        statement=text[:300],
                    )],
                    context_tags=self._extract_tags(text),
                ))

        # ── 🆕 模式5.8: X的Y提取改善（处理中文+ASCII混排主体）──
        # 原始模式(.{1,6})的(.{2,8})对"项目Alpha的微服务方案"切错
        # 改为非贪婪 + 宽主体 + 谓词不含标点
        for m in re.finditer(
            r"([一-鿿\w]{1,10})的([一-鿿\w]{2,8})(?:很|非常|特别|有点|比较|真)?(.+?)(?:[，。！？,!?]|$)",
            text
        ):
            subj_s, pred_s, obj_s = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            if subj_s and pred_s and obj_s and 2 <= len(obj_s) <= 30:
                # 跳过模式5.7已覆盖的决策类（避免重复）
                if any(vp[1] == pred_s for vp in _DECISION_VERBS):
                    continue
                # 🐛 v0.28: 全中文属性大于4字不是有效中文属性（"产品经理大"→跳过）
                _pred_all_cn = all('一' <= c <= '鿿' for c in pred_s)
                if _pred_all_cn and len(pred_s) > 4:
                    continue
                # 🐛 v0.30: 属性含"是/了" → "X的Y是Z"结构被贪婪吞掉（"银行卡后四位是2"）
                #   "我的银行卡后四位是2210" 被切出垃圾 (我, 银行卡后四位是2, 210)
                #   X是Y 结构由模式5.5/修正模式负责，这里跳过
                if "是" in pred_s or "了" in pred_s:
                    continue
                # 🐛 修复 v0.28: X的Y 过度匹配 — predicate 如果以ASCII为主（英文字母>50%），
                # 说明是英文单词被截断（如Kubernet），不是有效的中文属性
                _pred_ascii = sum(1 for c in pred_s if ord(c) < 128)
                if _pred_ascii > len(pred_s) * 0.5 and len(pred_s) >= 3:
                    continue
                key = (agent_id, subj_s, pred_s, obj_s)
                if key not in seen_triples:
                    seen_triples.add(key)
                    pattern_facts.append(FactTriple(
                        subject=subj_s,
                        predicate=pred_s,
                        object=obj_s,
                        agent_id=agent_id,
                        fact_type="observation",
                        source_session=source,
                        evidence=[EvidenceItem(
                            source=source or "unknown",
                            statement=text[:300],
                        )],
                    ))

        # ── 模式6: 逗号延续（"我喜欢喝冰美式，住在北京朝阳区"）──
        # 在模式1匹配完后，对未覆盖的中文逗号后片段做二次匹配
        # 把文本按逗号拆分成短句，逐句用所有谓词再匹配
        # ⭐ 只在文本含中文逗号且 pattern_facts 少于预期时才触发
        comma_count = text.count("，") + text.count(",")
        if comma_count >= 1 and len(pattern_facts) <= comma_count + 1:
            segments = re.split(r'[，,]\s*', text)
            _unmatched_segments = []  # 🐛 v0.28: 记录无模式匹配的片段，转raw observation保底
            for seg in segments:
                seg = seg.strip()
                if not seg or len(seg) < 2:
                    continue
                # 跳过已有「你/您」开头的
                if re.match(r'^(?:你|您)\s*', seg):
                    continue
                # 对无主语的片段尝试每个谓词（🆕 v0.26: 支持"最喜欢""很讨厌"等程度副词）
                _adv6 = r"(?:最(?!近)|很|非常|特别|有点|比较|真|超|更|太|极[其|为])?"
                # 🐛 v0.30: _t6 补"周末/周X/后来"（Q7: "周末在家喜欢听爵士"）
                _t6 = r"(?:上[周月天次]|这[周月天]|下[周月天]|本[周月天]|周[末六日一二三四五]|昨[天夜]|今[天夜]|前[天]|后[天]|现在|最近|刚刚|以前|曾经|已经|后来|之后|平时)?"
                # 🐛 v0.30: 谓词前允许地点词（"周末在家喜欢听爵士"→"在家"+喜欢）
                _loc6 = r"(?:在家|在公司|在单位|在学校|在办公室|在)?"
                _matched = False
                for word, (pred, ftype) in self._PREDICATES.items():
                    m = re.match(rf'{_t6}{_adv6}{_loc6}{re.escape(word)}(?:了)?\s*(.+?)(?:[，。！？,!?]|$)', seg)
                    if m:
                        _matched = True
                        obj = m.group(1).strip()
                        if not obj or len(obj) > 60:
                            continue
                        final_pred = self._SYNONYM_MAP.get(word, pred)
                        key = (agent_id, "用户", final_pred, obj)
                        if key not in seen_triples:
                            seen_triples.add(key)
                            pattern_facts.append(FactTriple(
                                subject="用户",
                                predicate=final_pred,
                                object=obj,
                                agent_id=agent_id,
                                fact_type=ftype,
                                source_session=source,
                                evidence=[EvidenceItem(
                                    source=source or "unknown",
                                    statement=text[:300],
                                )],
                                context_tags=self._extract_tags(text),
                            ))
                        break  # 每个片段只匹配第一个谓词
                if not _matched:
                    _unmatched_segments.append(seg)

            # 🐛 v0.28: 无匹配的逗号片段 → 转 raw observation 保底
            # 确保即使谓词表缺词，信息也不会丢（BM25 仍可通过原文检索）
            for _seg in _unmatched_segments:
                if len(_seg) >= 4:
                    raw_observations.append(FactTriple(
                        subject="用户",
                        predicate="说了",
                        object=_seg[:200],
                        agent_id=agent_id,
                        fact_type="observation",
                        confidence=0.45,
                        source_session=source,
                        evidence=[EvidenceItem(
                            source=source or "unknown",
                            statement=_seg[:300],
                        )],
                    ))

        # ── 模式7: 原始文本也暂存（如果 pattern_facts 为空时用）──
        if not pattern_facts and len(text) > 4:
            # 用普通谓词做保底标签
            _ptype = "observation"
            if any(kw in text for kw in ["喜欢", "爱", "讨厌", "想", "要"]):
                _ptype = "preference"
            raw_observations.append(FactTriple(
                subject="用户",
                predicate="说了",
                object=text[:200],
                agent_id=agent_id,
                fact_type=_ptype,
                confidence=0.5,
                source_session=source,
                evidence=[EvidenceItem(source=source or "unknown", statement=text[:300])],
            ))

        return pattern_facts, raw_observations

    def extract_tags(self, text: str) -> list[str]:
        """公开接口：外部调用关键词提取"""
        return self._extract_tags(text)

    def _extract_tags(self, text: str) -> list[str]:
        """提取关键词标签"""
        common_tags = {
            "工作", "学习", "生活", "运动", "美食", "旅行",
            "咖啡", "茶", "编程", "设计", "音乐", "电影",
            "天气", "交通", "购物", "健康", "理财", "宠物",
            "游戏", "读书", "跑步", "健身", "游泳", "摄影",
            "投资", "创业", "股票", "基金",
        }
        tag_aliases = {
            "冰美式": "咖啡", "拿铁": "咖啡", "美式": "咖啡",
            "Python": "编程", "Java": "编程", "JS": "编程",
            "爬山": "运动", "瑜伽": "运动",
        }
        found = []
        for tag in common_tags:
            if tag in text:
                found.append(tag)
        for keyword, tag in tag_aliases.items():
            if keyword in text and tag not in found:
                found.append(tag)
        return found[:5]
