"""
事实验证子 Agent — 对标 OpenWorker `coworker/tools/subagent.py` 的 explore 模式

## 🔥 相对 OpenWorker 的 Token 优化

OpenWorker explore 用 TurnEngine + 工具，每次 spawn 都有工具 schema 开销。
CogniMem 的矛盾验证不需要工具——LLM 自己就能推理矛盾双方的可信度。
去掉工具层，省下 ~200tok/次 的工具定义 + 工具调用循环。

## v0.21.1 优化（2026-07-29）

| 版本 | 方式 | 工具 | 每对 Token | N对 Token |
|------|------|------|-----------|----------|
| v0.21 | 每对单独调引擎 | web_search + think | ~500 | ~1500 |
| v0.21.1 | 批量一次调LLM | 无 | ~75 | ~75(分摊) |

## 用法

在 `cogni.consolidate()` 检测到矛盾后 → `FactVerifier.batch_verify()` →
同时分析所有矛盾对 → 返回裁决列表 → 更新置信度
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, fields
from typing import Any, Optional

logger = logging.getLogger("agent.verifier")


# ── 裁决结构 ──

@dataclass
class Verdict:
    """事实验证子 Agent 的裁决结果"""
    fact_a_id: str
    fact_b_id: str
    winner_id: str                    # 更可信的事实 ID（空 = 都不确定）
    winner_text: str                  # 胜出事实的文本
    confidence: float                 # 裁决置信度 (0~1)
    reasoning: str                    # 推理过程摘要
    needs_user_input: bool = False    # 是否最终需要用户确认
    error: Optional[str] = None


# ── FactVerifier ──

class FactVerifier:
    """只读事实验证子 Agent — 🔥 零工具，纯推理

    对标 OpenWorker `coworker/tools/subagent.py` 的 explore 模式。
    但 CogniMem 的矛盾验证不需要工具——LLM 自己就能推理。

    Usage:
        verifier = FactVerifier(llm_client)
        verdicts = verifier.batch_verify([
            (fact_a_dict, fact_b_dict),
            (fact_c_dict, fact_d_dict),
        ])
    """

    def __init__(self, llm_client):
        """
        Args:
            llm_client: LLMClient 实例（需要 chat() 方法）
        """
        self.llm = llm_client

    # ── 公开方法 ──

    def batch_verify(self, pairs: list[tuple[dict, dict]],
                     agent_id: str = "default") -> list[Verdict]:
        """🔥 批量验证多对矛盾——一次 LLM 调用搞定所有（省 Token）。

        对比 v0.21 逐对调用 TurnEngine，v0.21.1 改为：
        - 一次 LLM chat 调用处理最多 5 对矛盾
        - 无工具、无 TurnEngine 开销
        - 零工具 schema 传输

        Args:
            pairs: [(fact_a, fact_b), ...] 最多 5 对
            agent_id: Agent ID（仅日志用）

        Returns:
            list[Verdict]
        """
        if not pairs:
            return []

        pairs = pairs[:5]  # 最多处理 5 对
        logger.info("🔍 子Agent批量验证 %d 对矛盾", len(pairs))

        # 构建一次性提示词
        task = self._build_batch_task(pairs)

        try:
            # 🔥 直接 LLM chat，不走 TurnEngine，省工具 schema 开销
            reply = self.llm.chat(
                messages=[{"role": "user", "content": task}],
                temperature=0.3,
                max_tokens=1024,
            )
        except Exception as e:
            logger.exception("FactVerifier 批量验证失败")
            return [
                Verdict(
                    fact_a_id=p[0].get("fact_id", ""),
                    fact_b_id=p[1].get("fact_id", ""),
                    winner_id="", winner_text="",
                    confidence=0.0, reasoning=f"验证异常: {e}", error=str(e),
                )
                for p in pairs
            ]

        # 解析结果
        return self._parse_batch(reply, pairs)

    def verify_contradiction(self, fact_a: dict, fact_b: dict,
                             agent_id: str = "default") -> Verdict:
        """验证一对矛盾事实（单对接口，内部调 batch_verify）

        Args:
            fact_a: 第一条事实 dict
            fact_b: 第二条事实 dict
            agent_id: Agent ID

        Returns:
            Verdict 裁决结果
        """
        return self.batch_verify([(fact_a, fact_b)])[0]

    def research_fact(self, fact: dict, question: str = "") -> Verdict:
        """验证单条事实的真实性（可选联网查证）

        Args:
            fact: 事实 dict
            question: 额外验证问题

        Returns:
            Verdict（winner_id 用 fact_id 替代，confidence 表示可信度）
        """
        text = f"{fact.get('subject','')} {fact.get('predicate','')} {fact.get('object','')}"
        task = (
            f"验证以下说法的可靠性:\n"
            f"「{text}」\n"
            f"置信度(0~1):\n"
            f"参考: {question}\n" if question else
            f"「{text}」\n"
            f"置信度(0~1):\n"
        )

        try:
            reply = self.llm.chat(
                messages=[{"role": "user", "content": task}],
                temperature=0.3,
                max_tokens=512,
            )
            conf = self._extract_confidence(reply) or 0.5
            return Verdict(
                fact_a_id=fact.get("fact_id", ""), fact_b_id="",
                winner_id=fact.get("fact_id", ""), winner_text=text,
                confidence=conf, reasoning=(reply or "")[:500],
            )
        except Exception as e:
            return Verdict(
                fact_a_id=fact.get("fact_id", ""), fact_b_id="",
                winner_id="", winner_text="",
                confidence=0.0, reasoning=f"验证异常: {e}", error=str(e),
            )

    # ── 内部方法 ──

    @staticmethod
    def _build_batch_task(pairs: list[tuple[dict, dict]]) -> str:
        """🔥 构建批量矛盾验证提示词——短、无工具指令

        每条矛盾用编号标识，LLM 按编号输出裁决。
        """
        items = []
        for i, (a, b) in enumerate(pairs, 1):
            ta = f"{a.get('subject','')} {a.get('predicate','')} {a.get('object','')}"
            tb = f"{b.get('subject','')} {b.get('predicate','')} {b.get('object','')}"
            ca = a.get("confidence", 0.5)
            cb = b.get("confidence", 0.5)
            items.append(
                f"#{i} A:「{ta}」(置信{ca:.2f})  vs  B:「{tb}」(置信{cb:.2f})"
            )

        return (
            "分析以下矛盾事实，判断每条中 A 更可靠还是 B 更可靠：\n\n"
            + "\n".join(items)
            + "\n\n按输出格式（每行一条）:\n"
            + "\n".join(f"#{i}: A / B / UNSURE" for i in range(1, len(pairs) + 1))
        )

    @staticmethod
    def _parse_batch(reply: str, pairs: list[tuple[dict, dict]]) -> list[Verdict]:
        """🔥 批量解析 LLM 回复，提取每条矛盾的裁决"""
        verdicts = []
        lines = reply.strip().split("\n")

        for i, (fact_a, fact_b) in enumerate(pairs, 1):
            winner_id = ""
            winner_text = ""
            confidence = 0.5
            reasoning = ""

            # 找 #i: 行
            for line in lines:
                line = line.strip()
                if not line.startswith(f"#{i}") and not line.startswith(f"#{i}:"):
                    continue

                # #i: A 或 #i: B 或 #i: UNSURE
                content = line.split(":", 1)[1].strip() if ":" in line else line
                content = content.replace(f"#{i}", "").strip()

                if content.upper().startswith("A"):
                    winner_id = fact_a.get("fact_id", "")
                    winner_text = FactVerifier._fmt(fact_a)
                elif content.upper().startswith("B"):
                    winner_id = fact_b.get("fact_id", "")
                    winner_text = FactVerifier._fmt(fact_b)

            # 从全文提取置信度和理由
            for line in lines:
                ls = line.strip()
                if f"#{i}" in ls or (i < len(pairs) and f"#{i+1}" in ls):
                    continue
                m = re.search(r"(?:置信度|CONFIDENCE)[：:\s]*([0-9.]+)", ls, re.IGNORECASE)
                if m and confidence == 0.5:
                    try:
                        confidence = max(0.0, min(1.0, float(m.group(1))))
                    except ValueError:
                        pass

            # 兜底：从全文中的"#i"段提取
            all_lines = reply.split("\n")
            in_section = False
            section_text = ""
            for line in all_lines:
                if f"#{i}" in line:
                    in_section = True
                elif in_section:
                    if f"#{i+1}" in line or (in_section and not line.strip()):
                        break
                    section_text += line + "\n"
            reasoning = section_text.strip() or reply[:200]

            if not winner_id:
                needs_user = True
            else:
                needs_user = confidence < 0.3

            verdicts.append(Verdict(
                fact_a_id=fact_a.get("fact_id", ""),
                fact_b_id=fact_b.get("fact_id", ""),
                winner_id=winner_id, winner_text=winner_text,
                confidence=confidence, reasoning=reasoning[:200],
                needs_user_input=needs_user,
            ))

        return verdicts

    @staticmethod
    def _extract_confidence(text: str) -> Optional[float]:
        """从文本中提取置信度"""
        if not text:
            return None
        m = re.search(r"置信度[：:\s]*([0-9.]+)", text)
        if m:
            try:
                return max(0.0, min(1.0, float(m.group(1))))
            except ValueError:
                pass
        m = re.search(r"CONFIDENCE[：:\s]*([0-9.]+)", text, re.IGNORECASE)
        if m:
            try:
                return max(0.0, min(1.0, float(m.group(1))))
            except ValueError:
                pass
        return None

    @staticmethod
    def _fmt(fact: dict) -> str:
        return f"{fact.get('subject','')} {fact.get('predicate','')} {fact.get('object','')}"
