"""
事实提取器 — 从自然语言提取三元组

将用户输入/对话内容 转换为 FactTriple。
"""

import re
import logging
from datetime import datetime, timezone
from typing import Any
from .models import FactTriple, EvidenceItem

logger = logging.getLogger(__name__)


class TripleExtractor:
    """
    三元组提取器

    从自然语言中提取 (subject, predicate, object) 三元组。

    策略:
    1. 规则匹配: 预定义句法模式
    2. LLM 辅助: 复杂句子用 LLM 拆解 (TODO)
    3. 默认 fallback: 整句作为 observation
    """

    # 常见谓词
    _PREDICATES = {
        "喜欢", "不喜欢", "爱吃", "爱喝", "爱玩", "爱去",
        "是", "不是", "叫", "叫做", "有", "没有",
        "要", "不要", "想去", "想去", "想要", "想",
        "在", "住在", "位于", "来自",
        "做", "从事", "工作", "学习",
        "会", "不会", "能", "不能", "可以",
        "觉得", "认为", "感觉",
        "买了", "用过", "去过", "吃过", "喝过",
        "需要", "不需要", "讨厌", "反感",
    }

    def extract(self, text: str, source: str = "",
                agent_id: str = "default") -> list[FactTriple]:
        """
        从文本中提取所有可能的三元组。

        Args:
            text: 用户输入原文
            source: 来源标识 (session/conversation ID)
            agent_id: Agent ID
        """
        facts = []
        if text is None:
            return facts
        text = text.strip()
        if not text:
            return facts

        # 策略1: 规则匹配
        facts.extend(self._extract_by_patterns(text, source, agent_id))

        # 策略2: 如果没提取到, 作为 observation 存
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

    def _extract_by_patterns(self, text: str, source: str,
                             agent_id: str) -> list[FactTriple]:
        """规则匹配提取"""
        facts = []

        # 模式1: "我[喜欢]X" / "我[不爱吃]Y"
        # 匹配: 我/用户/我们 + 谓词 + 宾语
        for pred in self._PREDICATES:
            patterns = [
                rf"(?:我|我们|用户)\s*{pred}\s*(.+?)(?:[，。！？,.!?]|$)",
                rf"你\s*{pred}\s*(.+?)(?:[，。！？,.!?]|$)",
                rf"(?:他|她)\s*{pred}\s*(.+?)(?:[，。！？,.!?]|$)",
            ]
            for pat in patterns:
                matches = re.findall(pat, text)
                for obj in matches:
                    obj = obj.strip()
                    if obj and len(obj) < 100:
                        facts.append(FactTriple(
                            subject="user",
                            predicate=pred,
                            object=obj,
                            agent_id=agent_id,
                            fact_type=self._classify_predicate(pred),
                            source_session=source,
                            evidence=[EvidenceItem(
                                source=source or "unknown",
                                statement=text[:300],
                            )],
                            context_tags=self._extract_tags(text),
                        ))

        # 模式2: 实体关系 (e.g. "今天深圳的天气很好")
        # "X的Y很Z" → (X, Y, Z)
        match = re.search(r"(.+)的(.{2,}?)(.+?)(?:[，。！？,.!?]|$)", text)
        if match:
            subj, pred, obj = match.groups()
            subj_s, pred_s, obj_s = subj.strip(), pred.strip(), obj.strip()
            if subj_s and pred_s and obj_s and len(obj_s) < 50:
                facts.append(FactTriple(
                    subject=subj_s,
                    predicate=pred_s,
                    object=obj_s,
                    agent_id=agent_id,
                    fact_type="observation",
                    source_session=source,
                ))

        return facts

    def _classify_predicate(self, pred: str) -> str:
        """根据谓词推断事实类型"""
        pref_like = {"喜欢", "不喜欢", "爱吃", "爱喝", "爱玩", "讨厌", "反感"}
        goal_words = {"想", "想去", "想要", "要", "需要", "不想"}
        fact_words = {"是", "叫", "有", "在", "住在", "来自", "做"}

        if pred in pref_like:
            return "preference"
        elif pred in goal_words:
            return "goal"
        elif pred in fact_words:
            return "fact"
        return "observation"

    def _extract_tags(self, text: str) -> list[str]:
        """提取关键词标签"""
        # 简单提取: 名词性关键词
        common_tags = {
            "工作", "学习", "生活", "运动", "美食", "旅行",
            "咖啡", "茶", "编程", "设计", "音乐", "电影",
            "天气", "交通", "购物", "健康", "理财",
        }
        # 别名映射: "冰美式" → "咖啡"
        tag_aliases = {
            "冰美式": "咖啡", "拿铁": "咖啡", "美式": "咖啡",
            "Python": "编程", "Java": "编程", "JS": "编程",
        }
        found = []
        for tag in common_tags:
            if tag in text:
                found.append(tag)
        for keyword, tag in tag_aliases.items():
            if keyword in text and tag not in found:
                found.append(tag)
        return found
