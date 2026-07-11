"""
记忆治理器 — MemoryGovernor v2.0

在召回之后、注入 prompt 之前运行。
确定性规则（0 Token），不依赖 LLM，保证可审计。

受 ERINYS CareDog 4-state 治理 + 6-signal scoring 启发：

## 4 态策略
  ✅ SELECTED   → 安全且相关，注入 prompt
  ⚠️ CONFLICTED  → 有矛盾/不一致，注入 + 标注矛盾
  ⛔ DEMOTED     → 低置信度/过期/不可靠来源，不注入但可查
  🚫 BLOCKED     → PII/敏感信息，彻底拦截

## 六维信号评估（v2.0 新增）
  - Sensitivity    (敏感度): 含 PII/敏感词 → BLOCK
  - Staleness      (过期度): 超过半衰期 → DEMOTE
  - Conflict       (矛盾状态): 有矛盾/不一致 → CONFLICTED
  - Importance     (重要性): 语义重要性 → SELECTED/DEMOTED
  - Recency        (新鲜度): 最近访问时间 → 影响 DEMOTE 幅度
  - SourceTrust    (来源可信度): 用户陈述 > 工具结果 > AI 推断

## Cached Facts 感知（v2.0 新增）
  直接接受 FactTriple 对象，保留结构化信息。
  兼容旧版 dict 格式。
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any

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
    r"token[=:]|api.?key[=:]|secret[=:]",
]

# 来源可信度权重（用于 source_provenance 评估）
SOURCE_TRUST_WEIGHTS = {
    "user_statement": 1.0,       # 用户明确陈述 → 最可信
    "user_confirmation": 1.0,    # 用户确认
    "user_challenge": 0.8,       # 用户质疑
    "merges_evidence": 0.7,      # 多重证据
    "tool_result": 0.7,          # 工具返回结果
    "memory_abstraction": 0.6,   # 自动归纳
    "system": 0.5,               # 系统内部
    "agent_inference": 0.4,      # AI 推理 → 不可靠
    "consolidated": 0.4,         # 自动提升
    "third_party": 0.2,          # 第三方 → 低可信
}

# 不安全的来源（来源类型本身就应该被 demote）
UNTRUSTED_SOURCES = {"agent_inference", "consolidated", "third_party"}


class MemoryGovernor:
    """
    记忆治理器 v2.0 — 确定性规则 + 六维信号评估。

    Usage:
        governor = MemoryGovernor()
        results = governor.filter(recalled_facts)  # 支持 FactTriple 或 dict
        safe = governor.filter_selected(recalled_facts)
    """

    def __init__(self):
        self._stats: dict[str, int] = {"selected": 0, "conflicted": 0,
                                        "demoted": 0, "blocked": 0, "total": 0}

    @property
    def stats(self) -> dict:
        """获取治理统计快照"""
        s = dict(self._stats)
        return s

    def reset_stats(self):
        self._stats = {"selected": 0, "conflicted": 0,
                       "demoted": 0, "blocked": 0, "total": 0}

    def filter(self, facts: list[Any],
               min_confidence: float = 0.3) -> list[dict]:
        """
        对召回的事实列表执行治理过滤。

        支持 FactTriple 对象和 dict。

        Args:
            facts: 召回的事实列表（FactTriple 或 dict）
            min_confidence: 最低置信度阈值（低于此 → DEMOTED）

        Returns:
            增强后的事实列表，每项增加 _governance 字段:
            {"state": MemoryState, "reason": str, "signals": dict, "fact": dict}
        """
        results = []
        for f in facts:
            fact_dict = self._to_dict(f)
            state, reason, signals = self._evaluate(fact_dict, min_confidence)
            results.append({
                "state": state,
                "reason": reason,
                "signals": signals,
                "fact": fact_dict,
            })
            self._stats[state.value] = self._stats.get(state.value, 0) + 1
            self._stats["total"] = self._stats.get("total", 0) + 1
        return results

    def filter_selected(self, facts: list[Any],
                        min_confidence: float = 0.3) -> list[dict]:
        """只返回 SELECTED + CONFLICTED 的事实（可直接注入 prompt）"""
        return [
            r["fact"] for r in self.filter(facts, min_confidence)
            if r["state"] in (MemoryState.SELECTED, MemoryState.CONFLICTED)
        ]

    def get_conflicts(self, facts: list[Any],
                      min_confidence: float = 0.3) -> list[dict]:
        """返回被标记为 CONFLICTED 的事实"""
        return [
            r["fact"] for r in self.filter(facts, min_confidence)
            if r["state"] == MemoryState.CONFLICTED
        ]

    def get_states(self, facts: list[Any],
                   min_confidence: float = 0.3) -> dict:
        """返回治理统计"""
        r = self.filter(facts, min_confidence)
        return {
            "total": len(r),
            "selected": sum(1 for x in r if x["state"] == MemoryState.SELECTED),
            "conflicted": sum(1 for x in r if x["state"] == MemoryState.CONFLICTED),
            "demoted": sum(1 for x in r if x["state"] == MemoryState.DEMOTED),
            "blocked": sum(1 for x in r if x["state"] == MemoryState.BLOCKED),
        }

    def governance_summary(self, facts: list[Any],
                           min_confidence: float = 0.3) -> str:
        """
        生成治理摘要文本（可注入 prompt，让 LLM 了解治理情况）。

        ★ v2.0: 在上下文中注入治理信息，增加透明度。
        格式:
        ─── 记忆治理报告 ───
        ✅ 6 条通过 | ⚠️ 2 条有矛盾 | ⛔ 1 条降级 | 🚫 0 条拦截
        ⚠️ 矛盾提醒：用户之前说「不喜欢喝咖啡」但最近说「喜欢喝冰美式」
        ⛔ 降级原因：1 条记忆来自 AI推断（置信度低）
        """
        r = self.filter(facts, min_confidence)
        selected = [x for x in r if x["state"] == MemoryState.SELECTED]
        conflicted = [x for x in r if x["state"] == MemoryState.CONFLICTED]
        demoted = [x for x in r if x["state"] == MemoryState.DEMOTED]
        blocked = [x for x in r if x["state"] == MemoryState.BLOCKED]

        lines = ["─── 记忆治理报告 ───"]
        lines.append(
            f"✅ {len(selected)} 条通过 | "
            f"⚠️ {len(conflicted)} 条有矛盾 | "
            f"⛔ {len(demoted)} 条降级 | "
            f"🚫 {len(blocked)} 条拦截"
        )

        if conflicted:
            for c in conflicted[:3]:
                fact = c["fact"]
                lines.append(
                    f"⚠️ 矛盾提醒：记忆「{fact.get('subject','?')} "
                    f"{fact.get('predicate','?')} {fact.get('object','?')}」"
                    f" — {c['reason']}"
                )

        if demoted:
            demote_reasons = {}
            for d in demoted:
                reason = d["reason"]
                demote_reasons[reason] = demote_reasons.get(reason, 0) + 1
            for reason, count in demote_reasons.items():
                lines.append(f"⛔ 降级{count}条：{reason}")

        if blocked:
            for b in blocked:
                lines.append(f"🚫 拦截{1}条：{b['reason']}")

        lines.append("───")
        return "\n".join(lines)

    def _evaluate(self, fact: dict,
                  min_confidence: float) -> tuple[MemoryState, str, dict]:
        """
        六维信号评估一条事实。
        评估顺序: block → demote → conflict → select
        """
        signals = self._assess_signals(fact)

        # 🚫 BLOCKED: 敏感信息（优先级最高）
        if signals["sensitivity"] >= 1.0:
            return (MemoryState.BLOCKED,
                    f"含敏感信息", signals)

        # 🚫 BLOCKED: 不安全来源（拦截 AI 推理/第三方 的事实）
        if signals["source_trust"] <= 0.3 and signals["confidence"] < 0.5:
            return (MemoryState.BLOCKED,
                    f"来源不可靠（可信度 {signals['source_trust']:.1f}）", signals)

        # ⛔ DEMOTED: 过期度过高
        if signals["staleness"] > 0.8:
            return (MemoryState.DEMOTED,
                    f"记忆已严重过期（{signals.get('staleness_desc','')}）", signals)

        # ⛔ DEMOTED: 低置信度
        if fact.get("confidence", 0.5) < min_confidence:
            return (MemoryState.DEMOTED,
                    f"置信度 {fact['confidence']:.2f} 低于阈值 {min_confidence}",
                    signals)

        # ⛔ DEMOTED: 不可信来源 + 低重要性
        if signals["source_trust"] <= 0.4 and signals.get("importance", 0.5) < 0.3:
            return (MemoryState.DEMOTED,
                    f"来源可信度低（{signals.get('source_label','?')}）且重要性低",
                    signals)

        # ⛔ DEMOTED: 过期且重要性低
        if signals["staleness"] > 0.6 and signals.get("importance", 0.5) < 0.3:
            return (MemoryState.DEMOTED,
                    f"记忆过期且不重要", signals)

        # ⚠️ CONFLICTED: 有矛盾标记
        if signals.get("has_contradictions", False):
            cont_count = signals.get("contradiction_count", 0)
            return (MemoryState.CONFLICTED,
                    f"存在 {cont_count} 条矛盾记录", signals)

        # ✅ SELECTED: 通过所有检查
        return (MemoryState.SELECTED,
                "安全且相关", signals)

    def _assess_signals(self, fact: dict) -> dict:
        """
        六维信号评估。

        Returns:
            sensitivity: 0~1, 含敏感信息则=1
            staleness: 0~1, 超过半衰期越远越高
            source_trust: 0~1, 来源可信度
            has_contradictions: bool
            contradiction_count: int
            importance: 0~1
            recency: 0~1
            confidence: 0~1
        """
        now = datetime.now(timezone.utc)
        signals = {}

        # ★ Dimension 1: Sensitivity（敏感度）
        content = self._fact_text(fact)
        sensitivity = 0.0
        for pat in SENSITIVE_PATTERNS:
            if re.search(pat, content, re.IGNORECASE):
                sensitivity = 1.0
                break
        signals["sensitivity"] = sensitivity

        # ★ Dimension 2: Staleness（过期度）
        staleness, staleness_desc = self._calc_staleness(fact, now)
        signals["staleness"] = staleness
        signals["staleness_desc"] = staleness_desc

        # ★ Dimension 3: Conflict（矛盾）
        contradictions = fact.get("contradictions", []) or []
        # FactTriple 对象可能将 contradictions 存为 ID 列表
        if isinstance(contradictions, list):
            signals["has_contradictions"] = len(contradictions) > 0
            signals["contradiction_count"] = len(contradictions)
        else:
            signals["has_contradictions"] = False
            signals["contradiction_count"] = 0

        # ★ Dimension 4: Importance（重要性）
        signals["importance"] = fact.get("importance", 0.5)

        # ★ Dimension 5: Source Trust（来源可信度）
        source_type = ""
        evidence = fact.get("evidence", [])
        if evidence and isinstance(evidence, list):
            first = evidence[0]
            if isinstance(first, dict):
                source_type = first.get("source", "")
            elif hasattr(first, "source"):
                source_type = first.source
        signals["source_label"] = source_type
        signals["source_trust"] = SOURCE_TRUST_WEIGHTS.get(source_type, 0.5)

        # ★ Dimension 6: Recency（新鲜度）
        accessed_str = fact.get("accessed_at", "")
        if isinstance(accessed_str, str) and accessed_str:
            try:
                accessed = datetime.fromisoformat(accessed_str)
                days = max(0, (now - accessed).total_seconds() / 86400)
                signals["recency"] = 1.0 / (1.0 + days * 0.5)
                signals["days_since_access"] = round(days, 1)
            except (ValueError, TypeError):
                signals["recency"] = 0.5
        else:
            signals["recency"] = 0.5

        # Confidence
        signals["confidence"] = fact.get("confidence", 0.5)

        return signals

    @staticmethod
    def _calc_staleness(fact: dict, now: datetime) -> tuple[float, str]:
        """
        计算过期度 (0~1) + 描述文本。

        使用与 fact_network.py 相同的半衰期参数。
        """
        accessed_str = fact.get("accessed_at", "")
        if isinstance(accessed_str, str) and accessed_str:
            try:
                accessed = datetime.fromisoformat(accessed_str)
                days = max(0, (now - accessed).total_seconds() / 86400)
            except (ValueError, TypeError):
                return 0.0, ""
        else:
            return 0.0, ""

        if days <= 0:
            return 0.0, ""

        # 确定半衰期（天）
        encoding = fact.get("encoding_level", "raw")
        conf = fact.get("confidence", 0.5)
        access_count = fact.get("access_count", 1)

        hl_map = {"abstraction": 60.0, "core": 90.0}
        hl = hl_map.get(encoding, 14.0 if conf < 0.3 else 30.0 if conf >= 0.6 else 14.0)

        if access_count > 10:
            hl *= 1.5
        elif access_count > 5:
            hl *= 1.2

        # stigma = 1 - 0.5^(days/hl)
        import math
        stigma = 1.0 - (0.5 ** (days / hl))
        if stigma > 0.8:
            desc = f"{int(days)}天未访问，远超半衰期{int(hl)}天"
        elif stigma > 0.6:
            desc = f"{int(days)}天未访问，超过半衰期{int(hl)}天"
        elif stigma > 0.4:
            desc = f"接近半衰期（{int(days)}天/{int(hl)}天）"
        else:
            desc = ""
        return min(1.0, stigma), desc

    @staticmethod
    def _to_dict(fact) -> dict:
        """将 FactTriple 或 dict 统一转为 dict"""
        if isinstance(fact, dict):
            return fact
        # FactTriple 对象 → dict
        if hasattr(fact, "to_dict"):
            return fact.to_dict()
        # 尝试 dataclass asdict
        try:
            from dataclasses import asdict
            return asdict(fact)
        except (TypeError, ImportError):
            pass
        # 暴力转换
        result = {}
        for attr in ["subject", "predicate", "object", "fact_id", "confidence",
                      "importance", "encoding_level", "evidence", "contradictions",
                      "connected_facts", "context_tags", "source_session",
                      "created_at", "accessed_at", "access_count", "fact_type"]:
            if hasattr(fact, attr):
                result[attr] = getattr(fact, attr)
        return result

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
            elif hasattr(ev, "statement"):
                text += " " + ev.statement
        return text
