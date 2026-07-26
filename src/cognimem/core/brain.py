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

    # ── 写入 ──

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
            llm_facts = self.llm_extractor.extract(text, source, agent_id)
            if llm_facts:
                facts = llm_facts
                source_type = "agent_inference"

        if not facts:
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

        # 3. 批量添加 (含矛盾检测 + 来源权重)
        results = self.fact_network.batch_add(facts, agent_id, source_type)

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
            logger.info("🗑️ Reset agent '%s': %d rows deleted", agent_id, total)
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
