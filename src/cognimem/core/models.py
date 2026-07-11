"""
CogniMem 数据模型

核心设计：以"事实三元组"为最小存储单位。
不存文本，存 (subject, predicate, object) + 置信度 + 证据链。
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any
import json
import uuid


@dataclass
class EvidenceItem:
    """单条证据"""
    source: str          # 来源: 会话ID/文件/用户输入
    statement: str       # 原文
    timestamp: str = ""  # ISO 时间 (自动填充)


@dataclass
class FactTriple:
    """
    事实三元组 — CogniMem 最小存储单位

    不是存"用户喜欢喝冰美式"这条文本，
    而是存 (用户, 喜欢, 冰美式) 这个关系 + 元数据。
    """
    subject: str               # 主体
    predicate: str             # 谓词
    object: str                # 客体
    agent_id: str = "default"
    fact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    fact_type: str = "general"  # preference|fact|goal|decision|observation|skill
    confidence: float = 0.6     # 置信度 0~1
    importance: float = 0.5     # 重要性 0~1
    encoding_level: str = "raw" # raw|compressed|core

    # 元数据
    evidence: list = field(default_factory=list)       # [EvidenceItem, ...]
    contradictions: list = field(default_factory=list)  # [fact_id, ...]
    connected_facts: list = field(default_factory=list) # [fact_id, ...]
    context_tags: list = field(default_factory=list)    # ["tag1", "tag2"]
    source_session: str = ""

    # 时序
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    accessed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_confirmed: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    access_count: int = 1
    expires_at: str = ""

    def __post_init__(self):
        """初始化后校验：confidence/importance 必须为 0~1 有效浮点数"""
        import math
        if not isinstance(self.confidence, (int, float)) or math.isnan(self.confidence):
            self.confidence = 0.6
        self.confidence = max(0.0, min(1.0, self.confidence))
        if not isinstance(self.importance, (int, float)) or math.isnan(self.importance):
            self.importance = 0.5
        self.importance = max(0.0, min(1.0, self.importance))
        # encoding_level 校验：仅允许已知值
        valid_levels = {"raw", "compressed", "core", "abstraction", "abstracted"}
        if self.encoding_level not in valid_levels:
            self.encoding_level = "raw"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @property
    def triple_key(self) -> str:
        """去重 key: JSON 数组，避免字段内 | 引发碰撞"""
        return json.dumps([self.agent_id, self.subject, self.predicate, self.object],
                          ensure_ascii=False)

    @property
    def is_reliable(self) -> bool:
        return self.confidence >= 0.6

    @property
    def is_core_belief(self) -> bool:
        return self.confidence >= 0.9

    @property
    def is_unreliable(self) -> bool:
        return self.confidence < 0.2

    @property
    def confidence_label(self) -> str:
        """可读的置信度等级标签"""
        if self.confidence >= 0.9:
            return "确信"
        elif self.confidence >= 0.7:
            return "可靠"
        elif self.confidence >= 0.5:
            return "可能"
        elif self.confidence >= 0.3:
            return "存疑"
        else:
            return "不可靠"

    @property
    def source_label(self) -> str:
        """可读的来源标签"""
        if self.evidence:
            src = self.evidence[0].source if hasattr(self.evidence[0], 'source') else ''
            if src:
                return {"user_statement": "用户陈述",
                        "user_confirmation": "用户确认",
                        "agent_inference": "AI推断",
                        "tool_result": "工具结果",
                        "system": "系统"}.get(src, src)
        return "未知"


@dataclass
class Contradiction:
    """矛盾记录"""
    fact_a_id: str
    fact_b_id: str
    agent_id: str = "default"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    contradiction_type: str = "deny"  # deny(直接否定) | conflict(间接冲突) | context(上下文变化)
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolution: str = "pending"  # pending|resolved_a|resolved_b|both_false


@dataclass
class Episode:
    """时序事件 (Episodic Memory)"""
    agent_id: str
    summary: str
    session_id: str = ""
    fact_refs: list = field(default_factory=list)
    importance: float = 0.5
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    occurred_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
