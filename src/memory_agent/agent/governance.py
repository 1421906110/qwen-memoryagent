"""
Memory Governance — 生成前记忆治理过滤器

在召回之后、注入 prompt 之前运行。
不是 LLM 判断，是确定性规则，保证可审计。

4 态策略（参考 ERINYS Care Memory）:
  ✅ SELECTED    → 安全且相关，注入 prompt
  ⚠️ CONFLICTED  → 有矛盾/不一致，注入 + 标注矛盾
  ⛔ DEMOTED     → 低置信度/过时，不注入但可查
  🚫 BLOCKED     → PII/敏感信息，彻底拦截
"""

import logging
import re
from enum import Enum

logger = logging.getLogger("agent.governance")


class MemoryState(Enum):
    SELECTED = "selected"
    CONFLICTED = "conflicted"
    DEMOTED = "demoted"
    BLOCKED = "blocked"


# PII / 敏感关键词（中文 + 英文）
SENSITIVE_PATTERNS = [
    r"密码|口令|passwd|password",
    r"身份证|id.?card|ssn",
    r"银行卡|credit.?card|借记卡",
    r"手机号|phone|mobile|电话号码",
    r"地址|住址|address|家庭住址",
    r"信用卡|cvv|cvc|银行卡号",
    r"验证码|otp|2fa|mfa",
]


class MemoryGovernor:
    """
    记忆治理器 — 确定性规则过滤，不依赖 LLM。

    Usage:
        governor = MemoryGovernor()
        results = governor.filter(recalled_facts)
        for r in results:
            if r.state == MemoryState.SELECTED:
                prompt_memories.append(r.fact)
    """

    def filter(self, facts: list[dict],
               min_confidence: float = 0.3) -> list[dict]:
        """
        对召回的事实列表执行治理过滤。

        每个事实被标注 state + reason，供下游使用。

        Args:
            facts: 召回的事实列表（每项含 subject/predicate/object/confidence/…）
            min_confidence: 最低置信度阈值

        Returns:
            增强后的事实列表，每项增加 _governance 字段:
            {"state": MemoryState, "reason": str, "fact": 原数据}
        """
        results = []
        for f in facts:
            result = self._evaluate(f, min_confidence)
            results.append(result)
        return results

    def filter_selected(self, facts: list[dict]) -> list[dict]:
        """只返回 SELECTED + CONFLICTED 的事实（可直接注入 prompt）"""
        return [
            r["fact"] for r in self.filter(facts)
            if r["state"] in (MemoryState.SELECTED, MemoryState.CONFLICTED)
        ]

    def get_conflicts(self, facts: list[dict]) -> list[dict]:
        """返回被标记为 CONFLICTED 的事实"""
        return [
            r["fact"] for r in self.filter(facts)
            if r["state"] == MemoryState.CONFLICTED
        ]

    def get_states(self, facts: list[dict]) -> dict:
        """返回治理统计"""
        r = self.filter(facts)
        return {
            "total": len(r),
            "selected": sum(1 for x in r if x["state"] == MemoryState.SELECTED),
            "conflicted": sum(1 for x in r if x["state"] == MemoryState.CONFLICTED),
            "demoted": sum(1 for x in r if x["state"] == MemoryState.DEMOTED),
            "blocked": sum(1 for x in r if x["state"] == MemoryState.BLOCKED),
        }

    def _evaluate(self, fact: dict,
                  min_confidence: float) -> dict:
        """评估一条事实的治理状态"""
        confidence = fact.get("confidence", 0.5)
        content = self._fact_text(fact)
        tags = fact.get("context_tags", []) or []
        contradictions = fact.get("contradictions", []) or []

        # 🚫 BLOCKED: PII/敏感信息
        for pat in SENSITIVE_PATTERNS:
            if re.search(pat, content, re.IGNORECASE):
                return {
                    "state": MemoryState.BLOCKED,
                    "reason": f"含敏感信息：{pat[:20]}",
                    "fact": fact,
                    "confidence": confidence,
                }

        # ⛔ DEMOTED: 低置信度
        if confidence < min_confidence:
            return {
                "state": MemoryState.DEMOTED,
                "reason": f"置信度 {confidence:.2f} 低于阈值 {min_confidence}",
                "fact": fact,
                "confidence": confidence,
            }

        # ⚠️ CONFLICTED: 有矛盾标记
        if contradictions:
            return {
                "state": MemoryState.CONFLICTED,
                "reason": f"存在 {len(contradictions)} 条矛盾记录，已标注",
                "fact": fact,
                "confidence": confidence,
            }

        # ✅ SELECTED: 通过
        return {
            "state": MemoryState.SELECTED,
            "reason": "安全且相关，注入 prompt",
            "fact": fact,
            "confidence": confidence,
        }

    @staticmethod
    def _fact_text(fact: dict) -> str:
        """拼接事实文本用于关键词匹配"""
        parts = [
            str(fact.get("subject", "")),
            str(fact.get("predicate", "")),
            str(fact.get("object", "")),
        ]
        text = " ".join(parts)
        # 也匹配 evidence 原文
        for ev in fact.get("evidence", []):
            if isinstance(ev, dict):
                text += " " + ev.get("statement", "")
        return text
