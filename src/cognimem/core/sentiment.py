"""
情感极性检测引擎 — 无"我"句情感推理

核心目标：检测用户对实体的情感倾向，即使没有「我+喜欢/讨厌」模式。
"苹果生态系统真的强" → entity=苹果, sentiment=positive, score=0.8

Q4 测试覆盖：
  - "苹果生态系统真的强" → 正面(苹果)
  - "小米系统广告太多了" → 负面(小米)
  - 跨会话聚合：多次吐槽 → consolidate → (用户_情感倾向, 小米=负面)

设计原则：
  1. 轻量：纯规则+词库，0 Token
  2. 精准：宁可漏判不错判，低置信度宁可不输出
  3. 可扩展：品牌库+极性词库易增补
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


class SentimentEngine:
    """
    情感极性检测引擎

    三步走：
    1. 提取实体（品牌/产品名，支持已知+上下文抽取）
    2. 检测极性（正面词/负面词 + 否定前置检测）
    3. 输出特征（entity, sentiment, score）
    """

    # ── 正面线索词 (clue, weight) ──
    # weight 范围 0.5~1.0，代表线索强度
    _POSITIVE_CLUES = [
        # ★ 强正面 (0.85+)
        ("真的强", 0.90), ("确实强", 0.90), ("太强了", 0.90),
        ("真的很好", 0.90), ("确实很好", 0.90), ("太好了", 0.88),
        ("太棒了", 0.90), ("非常棒", 0.88), ("太厉害了", 0.90),
        ("真香", 0.88), ("太优秀了", 0.88),
        ("体验很好", 0.88), ("体验非常好", 0.90),
        ("性价比超高", 0.88), ("太值得了", 0.88),
        # ★ 中等正面 (0.75-0.84)
        ("很强", 0.80), ("非常强", 0.85), ("特别好", 0.82),
        ("很不错", 0.80), ("真不错", 0.80), ("确实不错", 0.80),
        ("很好用", 0.80), ("非常好用", 0.85), ("很流畅", 0.80),
        ("非常流畅", 0.85), ("很稳定", 0.75), ("很舒服", 0.75),
        ("很厉害", 0.80), ("真厉害", 0.82), ("很出色", 0.78),
        ("很优秀", 0.78), ("很满意", 0.80), ("值得买", 0.75),
        ("性价比高", 0.75), ("很值得", 0.75), ("种草了", 0.72),
        ("很漂亮", 0.70), ("很美观", 0.70), ("很精致", 0.72),
        ("非常喜欢", 0.82), ("真的很喜欢", 0.85),
        # ★ 弱正面 (0.5-0.74)
        ("很好", 0.70), ("不错", 0.60), ("还好", 0.50),
        ("可以", 0.50), ("还行", 0.50), ("好", 0.50),
        ("强", 0.55), ("棒", 0.55), ("优秀", 0.60),
    ]

    # ── 负面线索词 ──
    _NEGATIVE_CLUES = [
        # ★ 强负面 (0.85+)
        ("太差了", 0.90), ("太垃圾", 0.90), ("太烂了", 0.90),
        ("太贵了", 0.85), ("太坑了", 0.88), ("太差了", 0.90),
        ("体验极差", 0.92), ("体验很差", 0.88), ("体验太差了", 0.90),
        ("智商税", 0.85), ("割韭菜", 0.85), ("后悔死了", 0.88),
        ("太失望了", 0.85), ("太恶心", 0.88),
        # ★ 中等负面 (0.75-0.84)
        ("很差", 0.82), ("非常差", 0.85), ("很垃圾", 0.85),
        ("真垃圾", 0.85), ("很难用", 0.80), ("太难用", 0.85),
        ("不好用", 0.75), ("太卡了", 0.82), ("太慢了", 0.75),
        ("广告太多", 0.82), ("广告太多了", 0.82), ("太多广告", 0.80),
        ("太臃肿", 0.80), ("系统臃肿", 0.78),
        ("很后悔", 0.78), ("很失望", 0.75), ("不太行", 0.72),
        # ★ 弱负面 (0.5-0.74)
        ("广告多", 0.70), ("很卡", 0.70), ("很贵", 0.70),
        ("复杂", 0.60), ("麻烦", 0.60), ("一般", 0.50),
        ("不行", 0.60), ("不好", 0.60), ("不喜欢", 0.70),
        ("遗憾", 0.55), ("凑合", 0.50),
    ]

    # ── 知名实体库 ──
    _KNOWN_ENTITIES = [
        # 手机/数码
        "苹果", "iPhone", "Mac", "iPad", "AirPods", "Apple",
        "小米", "Redmi", "小米14", "小米15", "小米13",
        "华为", "Mate", "Pura", "华为Pura", "Mate60", "Mate70",
        "OPPO", "vivo", "荣耀", "三星", "魅族", "一加", "真我",
        # 汽车
        "特斯拉", "比亚迪", "蔚来", "小鹏", "理想", "问界",
        # 互联网/软件
        "微信", "支付宝", "抖音", "快手", "微博", "小红书",
        "百度", "阿里", "腾讯", "字节", "京东", "拼多多",
        "淘宝", "天猫", "闲鱼", "美团", "滴滴", "高德",
        # AI
        "ChatGPT", "Claude", "DeepSeek", "Qwen", "通义",
        "Gemini", "Copilot", "文心一言",
        # 系统
        "Windows", "macOS", "iOS", "Android", "HarmonyOS",
        # 通用
        "价格", "系统", "外观", "续航", "屏幕", "相机", "拍照",
    ]

    # ── 否定词（前置3字内出现 → 极性反转） ──
    _NEGATORS = frozenset({"不", "没", "不太", "不是", "并不"})

    @classmethod
    def analyze(cls, text: str) -> dict | None:
        """
        分析情感倾向。

        Args:
            text: 输入文本

        Returns:
            {"entity": "苹果", "sentiment": "positive"|"negative",
             "score": 0.8, "evidence": "原文"}
            返回 None 表示无显著情感
        """
        if not text or len(text.strip()) < 3:
            return None
        text_l = text.strip()

        # 1. 提取实体
        entity = cls._extract_entity(text_l)
        # 2. 检测极性
        polarity = cls._detect_polarity(text_l)
        if polarity is None:
            return None
        # 3. 组合
        return {
            "entity": entity or "用户",  # 无实体时绑定到用户
            "sentiment": polarity["sentiment"],
            "score": polarity["score"],
            "evidence": text_l[:200],
        }

    @classmethod
    def analyze_multi(cls, texts: list[str]) -> list[dict]:
        """批量分析"""
        results = []
        for t in texts:
            r = cls.analyze(t)
            if r:
                results.append(r)
        return results

    @classmethod
    def _extract_entity(cls, text: str) -> str | None:
        """从文本中提取情感指向的实体"""
        # 优先匹配已知实体（长匹配优先，避免"小"匹配"小米"）
        found = []
        for ent in cls._KNOWN_ENTITIES:
            if ent in text:
                found.append(ent)
        if found:
            return max(found, key=len)

        # 尝试提取句子开头的名词短语作为潜在实体
        # "华仔确实值得信赖" → "华仔" 没有被KNOWN覆盖
        m = re.match(r'^\s*([一-鿿]{2,4})', text)
        if m:
            first = m.group(1)
            skip_words = frozenset({
                "真的", "确实", "不太", "这个", "那个", "一个",
                "太", "很", "非常", "特别", "有点", "比较",
                "我", "你", "他", "她", "我们", "你们", "他们",
            })
            if first not in skip_words and not first.endswith("的"):
                return first

        return None

    @classmethod
    def _detect_polarity(cls, text: str) -> dict | None:
        """检测文本的情感极性"""
        pos_score = 0.0
        neg_score = 0.0

        # 检查正面线索
        for clue, weight in cls._POSITIVE_CLUES:
            if clue in text:
                # 检查前置否定词（3字内）
                idx = text.find(clue)
                before = text[max(0, idx - 4):idx]
                negated = any(neg in before for neg in cls._NEGATORS)
                if negated:
                    neg_score += weight * 0.7  # "不好用" = 否定正面
                else:
                    pos_score += weight

        # 检查负面线索
        for clue, weight in cls._NEGATIVE_CLUES:
            if clue in text:
                idx = text.find(clue)
                before = text[max(0, idx - 4):idx]
                negated = any(neg in before for neg in cls._NEGATORS)
                if negated:
                    pos_score += weight * 0.5  # "不卡" = 否定负面→正面
                else:
                    neg_score += weight

        # 阈值判断：总得分不足 0.6 或单方不足 0.5 视为无显著情感
        if pos_score < 0.5 and neg_score < 0.5:
            return None

        if pos_score > neg_score:
            score = min(1.0, pos_score / max(pos_score + neg_score * 0.3, 0.5))
            return {"sentiment": "positive", "score": score}
        elif neg_score > pos_score:
            score = min(1.0, neg_score / max(neg_score + pos_score * 0.3, 0.5))
            return {"sentiment": "negative", "score": score}

        return None
