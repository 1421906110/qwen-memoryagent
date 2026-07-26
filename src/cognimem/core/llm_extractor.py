"""
LLM 三元组提取器 — 用千问替代规则匹配

为什么不用纯规则？
  规则匹配准确率 ~60%，复杂句子基本废。
  LLM 提取准确率 ~90%+，还能处理隐含关系。

为什么不用纯 LLM？
  每次提取 50-150 tok，1000 次 ≈ 5-15 万 tok。
  所以策略是: 规则兜底 + LLM 精提。

使用方式：
  export DASHSCOPE_API_KEY=sk-xxx
  # 或写入 .env 文件
"""

import json
import os
import logging
import re
import difflib
from collections import OrderedDict
from typing import Any

from .models import FactTriple, EvidenceItem

logger = logging.getLogger(__name__)


# ── 默认配置 ──
DEFAULT_MODEL = "deepseek-v4-flash"          # DeepSeek V4
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.1

# Qwen / DashScope 兼容配置
QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
QWEN_DEFAULT_MODEL = "qwen-plus"


class LLMTripleExtractor:
    """
    LLM 三元组提取器

    用千问（DashScope）从自然语言中提取三元组。
    也支持 OpenAI 兼容接口（换 base_url 即可）。

    策略：
    1. 优先 LLM 提取（结构化 JSON 输出）
    2. 如果 LLM 调用失败 → fallback 到规则提取
    """

    def __init__(self, api_key: str = "", model: str = DEFAULT_MODEL,
                 base_url: str = "", config: dict | None = None):
        self.api_key = (
            api_key
            or os.environ.get("DEEPSEEK_API_KEY", "")
            or os.environ.get("DASHSCOPE_API_KEY", "")
            or os.environ.get("QWEN_API_KEY", "")
        )
        # 自动检测 backend：如果设了 QWEN_API_KEY 但没设 base_url，用 DashScope
        has_qwen_key = bool(os.environ.get("QWEN_API_KEY"))
        self.model = model
        self.base_url = base_url or os.environ.get(
            "QWEN_BASE_URL", os.environ.get("LLM_BASE_URL", "")
        )
        if not self.base_url:
            if has_qwen_key or self.api_key == os.environ.get("QWEN_API_KEY"):
                self.base_url = QWEN_BASE_URL
                if model == DEFAULT_MODEL:
                    self.model = os.environ.get("QWEN_MODEL", QWEN_DEFAULT_MODEL)
            else:
                self.base_url = DEFAULT_BASE_URL
        self.config = config or {}
        self.max_tokens = self.config.get("max_tokens", DEFAULT_MAX_TOKENS)
        self.temperature = self.config.get("temperature", DEFAULT_TEMPERATURE)

        # ⭐ 推理链缓存：相似句子复用提取结果
        # key=原文, value=(时间戳, [triple_meta, ...])
        self._extraction_cache: OrderedDict[str, tuple[float, list[dict]]] = OrderedDict()
        self._cache_max = 500
        self._cache_hits = 0
        self._cache_misses = 0

    # ── 公共接口 ──

    def extract(self, text: str, source: str = "",
                agent_id: str = "default") -> list[FactTriple]:
        """
        提取三元组。混合策略（省 Token）：

        1. ⭐ 简单句 → 规则提取（0 Token）
        2. 推理链缓存命中 → 直接复用（0 Token）
        3. LLM 提取（仅复杂/模糊句子）
        4. 规则 fallback
        """
        if not text or not text.strip():
            return []

        # ⭐ 混合模式：简单句直接用规则提取（省 20-30% Token）
        if self._is_simple_sentence(text):
            facts = self._simple_rule_extract(text, source, agent_id)
            if facts:
                logger.debug(f"⚡ Simple rule: {len(facts)} facts from '{text[:40]}'")
                return facts

        # ⭐ 推理链缓存
        if self.api_key:
            cached = self._find_cached_extraction(text, source, agent_id)
            if cached is not None:
                self._cache_hits += 1
                logger.debug(
                    f"⚡ Cache hit ({self._cache_hits}/{self._cache_hits + self._cache_misses}): "
                    f"'{text[:40]}'"
                )
                return cached

        # LLM 提取（超长文本直接走规则 fallback，避免浪费 token + 提取失效）
        if self.api_key and len(text) <= 400:
            try:
                facts = self._llm_extract(text, source, agent_id)
                if facts:
                    self._cache_misses += 1
                    self._cache_extraction(text, facts)
                    logger.debug(f"LLM extracted {len(facts)} facts from '{text[:50]}'")
                    return facts
            except Exception as e:
                logger.warning(f"LLM extract failed, fallback to rules: {e}")

        return self._rule_fallback(text, source, agent_id)

    # ── LLM 提取 ──

    def _llm_extract(self, text: str, source: str,
                     agent_id: str) -> list[FactTriple]:
        """调用千问提取三元组"""
        import openai

        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        prompt = self._build_prompt(text)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if not content:
            return []

        return self._parse_response(content, text, source, agent_id)

    def _system_prompt(self) -> str:
        return """你是一个事实提取器。从用户输入中提取 (主体, 谓词, 客体) 三元组。

规则：
1. 每个三元组表示一个独立的事实关系
2. 主体通常是"用户"（或对话中的具体实体）
3. 谓词用简洁的中文动词（喜欢/是/在/有/想/需要/用/做/去等）
4. 客体是谓词作用的对象
5. 给每个事实打标签：preference(偏好) | fact(事实) | goal(目标) | observation(观察)
6. 提取关键词标签（3个以内）
7. 注意检测隐含关系（"戒咖啡" → 不喜欢咖啡）

输出格式（JSON）：
{
  "triples": [
    {
      "subject": "用户",
      "predicate": "喜欢",
      "object": "喝冰美式",
      "fact_type": "preference",
      "tags": ["咖啡", "饮品"],
      "importance": 0.7
    }
  ]
}

只返回 JSON，不要加解释。如果不确定就提取为 observation 类型。
"""

    def _build_prompt(self, text: str) -> str:
        return f"从以下文本中提取事实三元组：\n\n{text}"

    def _parse_response(self, content: str, original: str,
                        source: str, agent_id: str) -> list[FactTriple]:
        """解析 LLM 返回的 JSON"""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # 尝试提取 JSON 部分
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return []
            else:
                return []

        facts = []
        triples = data.get("triples", []) if isinstance(data, dict) else data if isinstance(data, list) else []

        if isinstance(triples, list):
            for item in triples:
                if not isinstance(item, dict):
                    continue
                subject = str(item.get("subject", "")).strip()
                predicate = str(item.get("predicate", "")).strip()
                obj = str(item.get("object", "")).strip()
                if not subject or not predicate or not obj:
                    continue

                fact_type = str(item.get("fact_type", "observation"))
                importance = float(item.get("importance", 0.5))
                tags = item.get("tags", [])
                if isinstance(tags, str):
                    tags = [tags]

                facts.append(FactTriple(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    agent_id=agent_id,
                    fact_type=fact_type if fact_type in {
                        "preference", "fact", "goal", "decision",
                        "observation", "skill"
                    } else "observation",
                    importance=min(1.0, max(0.1, importance)),
                    context_tags=tags[:5],
                    source_session=source,
                    evidence=[EvidenceItem(
                        source=source or "llm_extraction",
                        statement=original[:500],
                    )],
                ))

        # 一条都没提取到 → 存为 observation
        if not facts:
            facts.append(FactTriple(
                subject="用户",
                predicate="说了",
                object=original[:200],
                agent_id=agent_id,
                fact_type="observation",
                confidence=0.5,
                source_session=source,
                evidence=[EvidenceItem(
                    source=source or "llm_extraction",
                    statement=original[:500],
                )],
            ))

        return facts

    # ── ⭐ 混合提取：简单句规则（0 Token）──

    # 匹配顺序重要：越具体的模式越靠前！
    SIMPLE_PATTERNS = [
        # (正则, subject, predicate, object_group, fact_type)
        # ── 工作/项目类（优先级最高：避免被通用模式误匹配）──
        (r"我(?:负责|在[做干搞弄])(.+?)(?:[，。！？,.!?]|$)", "用户", "负责", 1, "fact"),
        (r"我(?:做[完了好过]|完成[了]?|搞[定完][了]?)(.+?)(?:[，。！？,.!?]|$)", "用户", "完成了", 1, "fact"),
        (r"我(?:没|没有|还没)(?:做|完成|开始|去)(.+?)(?:[，。！？,.!?]|$)", "用户", "未完成", 1, "fact"),
        (r"我(?:擅长|精通|专长是?)(.+?)(?:[，。！？,.!?]|$)", "用户", "擅长", 1, "skill"),
        # ── 决策/计划类（在项目进度前：避免"我打算X上线"误匹配进度模式）──
        (r"我(?:决定|打算|计划|准备)(?:要|去)?(.+?)(?:[，。！？,.!?]|$)", "用户", "决定", 1, "decision"),
        (r"我(?:选择|选[了]?|定[了]?)(.+?)(?:[，。！？,.!?]|$)", "用户", "选择", 1, "decision"),
        # ── 项目进度/状态变化（用捕获组动态提取主语）──
        (r"(.+?)的?(?:截止日期|deadline|ddl)(?:是|改成?|改为?|推迟到?|延期到?)?(.+?)(?:[，。！？,.!?]|$)", r"\1", "截止日期", 2, "fact"),
        (r"(.+?)进度(?:是|为|到|达到)?(.+?)(?:[，。！？,.!?]|$)", r"\1", "进度", 2, "fact"),
        (r"(.+?)(?:已经|已)?(完成|做完|搞定|上线|发布|提交|交付)[了]?", r"\1", "状态", 2, "fact"),
        (r"(.+?)(?:延期|推迟|delay|推后)[了]?", r"\1", "状态", "已延期", "fact"),
        # ── 时间/日期类 ──
        (r"今天(?:是|礼拜|星期)?(.+?)(?:[，。！？,.!?]|$)", "今天", "是", 1, "fact"),
        (r"现在(?:是)?(.+?)(?:[，。！？,.!?]|$)", "现在", "时间", 1, "fact"),
        (r"我今年(.+?)(?:[，。！？,.!?]|$)", "用户", "年龄", 1, "fact"),
        # ── 日常状态类 ──
        (r"我(?:觉得|感觉|认为)(.+?)(?:[，。！？,.!?]|$)", "用户", "感觉", 1, "observation"),
        (r"我(?:最近|这?几天|这段?时间|最近在)(.+?)(?:[，。！？,.!?]|$)", "用户", "最近", 1, "observation"),
        (r"我(?:已经|早就)?(?:吃完?|喝过?|看过?|读过?|写完?)(.+?)(?:[，。！？,.!?]|$)", "用户", "做过", 1, "fact"),
        (r"我(?:想|想去?)(?:去|学|看|吃|喝|买|玩|做)?(.+?)(?:[，。！？,.!?]|$)", "用户", "想去", 1, "goal"),
        # ── 属性类（我是X / 我在X / 我有X） ──
        (r"我(?:是|系)(?:一[个名位])?(.+?)(?:[，。！？,.!?]|$)", "用户", "是", 1, "fact"),
        (r"我(?:有)(?:一[个些])?(.+?)(?:[，。！？,.!?]|$)", "用户", "有", 1, "fact"),
        (r"我在(.+?)(?:工作|上班|学习|读书)(?:[，。！？,.!?]|$)", "用户", "工作在", 1, "fact"),
        # ── 偏好类 ──
        (r"我(?:很|非常|特别|超|最)?喜欢(.+?)(?:[，。！？,.!?]|$)", "用户", "喜欢", 1, "preference"),
        (r"我(?:很|非常|特别|超|最)?[爱热]爱(.+?)(?:[，。！？,.!?]|$)", "用户", "喜欢", 1, "preference"),
        (r"我(?:不|没)(?:喜欢|爱|想|要)(.+?)(?:[，。！？,.!?]|$)", "用户", "不喜欢", 1, "preference"),
        (r"用户(?:不|没)(?:喜欢|爱|想|要)(.+?)(?:[，。！？,.!?]|$)", "用户", "不喜欢", 1, "preference"),
        (r"我(?:讨厌|厌恶|烦)(.+?)(?:[，。！？,.!?]|$)", "用户", "讨厌", 1, "preference"),
        (r"我想?(?:要|想|需要|要)(.+?)(?:[，。！？,.!?]|$)", "用户", "想要", 1, "goal"),
        # ── 基础事实类 ──
        (r"我(?:的)?(?:名字)?[是叫]+(.+?)(?:[，。！？,.!?]|$)", "用户", "名字", 1, "fact"),
        (r"我[是](?:一[个名位])?(.+?)(?:[，。！？,.!?]|$)", "用户", "是", 1, "fact"),
        (r"我去过(.+?)(?:[，。！？,.!?]|$)", "用户", "去过", 1, "fact"),
        (r"我(?:会|能|可以)(.+?)(?:[，。！？,.!?]|$)", "用户", "会", 1, "skill"),
        (r"我[在住](?:在)?(.+?)(?:[，。！？,.!?]|$)", "用户", "在", 1, "fact"),
        (r"我[有](?:一个?)?(.+?)(?:[，。！？,.!?]|$)", "用户", "有", 1, "fact"),
    ]

    def _is_simple_sentence(self, text: str) -> bool:
        """判断是否简单句（可用规则精确提取，无需 LLM）"""
        text = text.strip()
        # 太长的句子大概率不是简单句
        if len(text) > 40:
            return False
        # 包含逗号、但是、然而等复杂结构的跳过
        complex_markers = ["但是", "然而", "因为", "所以", "虽然", "而且", "不过", "于是"]
        if any(m in text for m in complex_markers):
            return False
        # 匹配已知模式
        for pattern, *_ in self.SIMPLE_PATTERNS:
            if re.search(pattern, text):
                return True
        return False

    def _simple_rule_extract(self, text: str, source: str,
                              agent_id: str) -> list[FactTriple]:
        """用正则模式提取简单句（0 Token，<1ms）

        ⭐ v0.20: 收集所有匹配，不只取第一个。
         "我喜欢喝冰美式，住在北京朝阳区" → 2 条事实。
        """
        text = text.strip()
        facts = []
        seen_keys = set()

        for pattern, subject, predicate, obj_group, fact_type in self.SIMPLE_PATTERNS:
            for m in re.finditer(pattern, text):
                # 支持动态捕获组引用（如 r"\1" 表示第一个捕获组）
                def _resolve(val, default=""):
                    if isinstance(val, str) and val.startswith("\\"):
                        try:
                            idx = int(val[1:])
                            return (m.group(idx) or default).strip()
                        except (ValueError, IndexError):
                            return default
                    return val

                subj = _resolve(subject, "用户")
                pred = _resolve(predicate, "")
                obj = m.group(obj_group).strip() if isinstance(obj_group, int) else _resolve(obj_group, "")
                if not obj or len(obj) > 40:
                    continue

                # 去重：同一文本同一谓词同一宾语不重复存
                key = (subj, pred, obj)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                facts.append(FactTriple(
                    subject=subj,
                    predicate=pred,
                    object=obj,
                    agent_id=agent_id,
                    fact_type=fact_type if fact_type in {
                        "preference", "fact", "goal", "decision",
                        "observation", "skill"
                    } else "observation",
                    confidence=0.65,
                    importance=0.5,
                    source_session=source,
                    evidence=[EvidenceItem(
                        source=source or "simple_rule",
                        statement=text,
                    )],
                ))
        return facts

    # ── 规则 Fallback ──

    def _rule_fallback(self, text: str, source: str,
                       agent_id: str) -> list[FactTriple]:
        """简单的规则 fallback（比什么都不做强）"""
        # 提取 "X 的 Y" 模式
        facts = []
        match = re.search(r"(.+?)的(.+?)(?:很|非常|特别|有点|比较)?(.+?)(?:[，。！？,.!?]|$)", text)
        if match:
            subj, pred, obj = match.groups()
            if subj and pred and obj and len(obj) < 50:
                facts.append(FactTriple(
                    subject=subj.strip(),
                    predicate=pred.strip(),
                    object=obj.strip(),
                    agent_id=agent_id,
                    fact_type="observation",
                    source_session=source,
                ))

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
                    source=source or "fallback",
                    statement=text[:500],
                )],
            ))

        return facts

    # ═══════════════════════════════════════════
    # ⭐ 推理链缓存引擎
    # ═══════════════════════════════════════════

    def _cache_extraction(self, text: str, facts: list[FactTriple]):
        """缓存 LLM 提取结果（存元数据，不存完整对象避免 UUID 冲突）"""
        import time as _time
        triples_meta = []
        for f in facts:
            triples_meta.append({
                "subject": f.subject,
                "predicate": f.predicate,
                "object": f.object,
                "fact_type": f.fact_type,
                "tags": list(f.context_tags),
                "importance": f.importance,
            })
        self._extraction_cache[text] = (_time.time(), triples_meta)
        # LRU 淘汰
        while len(self._extraction_cache) > self._cache_max:
            self._extraction_cache.popitem(last=False)

    def _find_cached_extraction(self, text: str, source: str,
                                 agent_id: str) -> list[FactTriple] | None:
        """
        查找缓存：用 difflib 比较新文本与缓存文本的相似度。

        - > 0.8: 直接复用（0 Token）
        - 0.5-0.8: 复用但降置信度（低风险场景）
        - < 0.5: 不走缓存
        """
        if not self._extraction_cache:
            return None

        import time as _time
        now = _time.time()
        best_ratio = 0.0
        best_meta = None
        best_text = ""

        # 找最相似的缓存条目（过期 1 小时以上的跳过）
        for cached_text, (ts, meta) in self._extraction_cache.items():
            if now - ts > 3600:  # 1 小时过期
                continue
            ratio = difflib.SequenceMatcher(None, text, cached_text).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_meta = meta
                best_text = cached_text

        if best_meta is None:
            return None

        # 相似度判断
        if best_ratio >= 0.8:
            # 高相似 → 直接复用，尝试适配 object
            return self._adapt_cached_triples(
                best_meta, best_text, text, source, agent_id,
                confidence_multiplier=0.95
            )
        elif best_ratio >= 0.5:
            # 中相似 → 复用但降置信度
            return self._adapt_cached_triples(
                best_meta, best_text, text, source, agent_id,
                confidence_multiplier=0.85
            )

        return None

    def _adapt_cached_triples(self, cached_meta: list[dict],
                               old_text: str, new_text: str,
                               source: str, agent_id: str,
                               confidence_multiplier: float = 1.0
                               ) -> list[FactTriple]:
        """从缓存元数据重建 FactTriple，尝试适配文本差异"""
        import difflib as _difflib

        # 简单的词级差异检测：找出新旧文本中不同的部分
        sm = _difflib.SequenceMatcher(None, old_text, new_text)
        replacements: dict[str, str] = {}
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'replace':
                old_part = old_text[i1:i2]
                new_part = new_text[j1:j2]
                replacements[old_part] = new_part

        facts = []
        for meta in cached_meta:
            obj = meta["object"]
            # 尝试用新文本中的词替换 object
            for old_part, new_part in replacements.items():
                if old_part in obj:
                    obj = obj.replace(old_part, new_part)

            facts.append(FactTriple(
                subject=meta["subject"],
                predicate=meta["predicate"],
                object=obj,
                agent_id=agent_id,
                fact_type=meta.get("fact_type", "observation"),
                confidence=0.6 * confidence_multiplier,
                importance=float(meta.get("importance", 0.5)),
                context_tags=list(meta.get("tags", [])),
                source_session=source,
                evidence=[EvidenceItem(
                    source=source or "cached_extraction",
                    statement=new_text[:500],
                )],
            ))

        return facts

    def get_cache_stats(self) -> dict:
        """获取缓存命中统计"""
        total = self._cache_hits + self._cache_misses
        return {
            "cache_size": len(self._extraction_cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": f"{self._cache_hits / total * 100:.1f}%" if total > 0 else "0%",
            "tokens_saved_estimate": self._cache_hits * 80,  # 平均每次省 ~80 tok
        }

    # ── 健康检查 ──

    def check_connection(self) -> dict:
        """测试 LLM API 连通性"""
        if not self.api_key:
            return {"status": "no_api_key", "message": "未设置 API Key（支持 QWEN_API_KEY / DASHSCOPE_API_KEY / DEEPSEEK_API_KEY）"}

        try:
            import openai
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "返回 JSON: {\"ok\": true}"}],
                temperature=0.1,
                max_tokens=50,
                response_format={"type": "json_object"},
            )
            return {
                "status": "ok",
                "model": self.model,
                "response": resp.choices[0].message.content,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
