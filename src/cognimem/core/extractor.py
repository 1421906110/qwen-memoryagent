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
        "买了": ("买了", "fact"),
        "用过": ("用过", "fact"),
        "吃过": ("吃过", "fact"),
        "喝过": ("喝过", "fact"),
        "看过": ("看过", "fact"),
        "读过": ("读过", "fact"),
        "学过": ("学过", "fact"),
        "我负责": ("负责", "fact"),

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

    def extract(self, text: str, source: str = "",
                agent_id: str = "default") -> list[FactTriple]:
        """
        从文本中提取所有可能的三元组。

        ⭐ 收集所有模式匹配结果，不取最先匹配的那个。
        ⭐ 仅当真实无任何匹配时，才 fallback 到 observation。
        """
        facts = []
        if text is None:
            return facts
        text = text.strip()
        if not text:
            return facts

        # 策略1: 规则匹配 — 收集所有命中
        pattern_facts, raw_obs_facts = self._extract_all(text, source, agent_id)
        facts.extend(pattern_facts)

        # 策略2: 如果规则提取出任何有意义的匹配（非 observation），直接用
        # 如果有 raw_obs 且无 pattern 匹配，选 raw_obs 中最优的
        if not pattern_facts and raw_obs_facts:
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

    def _extract_all(self, text: str, source: str,
                     agent_id: str) -> tuple[list[FactTriple], list[FactTriple]]:
        """
        收集所有模式匹配的结果。
        Returns: (pattern_facts, raw_observations)
        """
        pattern_facts = []
        raw_observations = []
        seen_triples = set()  # 去重

        # ── 模式1: "我/用户/你 + 谓词 + 宾语" ──
        for word, (pred, ftype) in self._PREDICATES.items():
            patterns = [
                rf"(?:我|我们|用户|[他她])\s*{re.escape(word)}\s*(.+?)(?:[，。！？,.!?]|$)",
                rf"(?:你|您)\s*{re.escape(word)}\s*(.+?)(?:[，。！？,.!?]|$)",
            ]
            for pat in patterns:
                for m in re.finditer(pat, text):
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

        # ── 模式2: 实体关系（X的Y很Z）──
        # "今天天气很好" → (今天, 天气, 很好)
        for m in re.finditer(r"(.{1,6})的(.{2,8})(?:很|非常|特别|有点|比较|真)?(.+?)(?:[，。！？,.!?]|$)", text):
            subj, pred, obj = m.groups()
            subj_s, pred_s, obj_s = subj.strip(), pred.strip(), obj.strip()
            if subj_s and pred_s and obj_s and 1 <= len(obj_s) <= 30:
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

        # ── 模式3: "我叫X" / "名字是X" ──
        for m in re.finditer(r"(?:我|用户)(?:的)?(?:名字|昵称|姓名|全名)(?:是|叫)?(.+?)(?:[，。！？,.!?]|$)", text):
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
        for m in re.finditer(r"我在(.+?)(?:工作|上班|学习|读书)(?:[，。！？,.!?]|$)", text):
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

        # ── 模式5: 数量/度量（"我今年X岁" / "我的Y是Z个"）──
        for m in re.finditer(r"我今年(.+?)(?:岁|岁了)(?:[，。！？,.!?]|$)", text):
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

        # ── 模式6: 逗号延续（"我喜欢喝冰美式，住在北京朝阳区"）──
        # 在模式1匹配完后，对未覆盖的中文逗号后片段做二次匹配
        # 把文本按逗号拆分成短句，逐句用所有谓词再匹配
        # ⭐ 只在文本含中文逗号且 pattern_facts 少于预期时才触发
        comma_count = text.count("，") + text.count(",")
        if comma_count >= 1 and len(pattern_facts) <= comma_count + 1:
            segments = re.split(r'[，,]\s*', text)
            for seg in segments:
                seg = seg.strip()
                if not seg or len(seg) < 2:
                    continue
                # 跳过已有「我」开头的（模式1已处理）
                if re.match(r'^(?:我|我们|用户|你|您|[他她])\s*', seg):
                    continue
                # 对无主语的片段尝试每个谓词
                for word, (pred, ftype) in self._PREDICATES.items():
                    m = re.match(rf'{re.escape(word)}\s*(.+?)(?:[，。！？,.!?]|$)', seg)
                    if m:
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
