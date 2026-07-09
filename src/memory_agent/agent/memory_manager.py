"""
Memory Manager — intelligent memory filtering for the Agent.

The agent should NOT store everything it sees or hears.
It should only store what matters:

  ✅ User preferences ("我喜欢美式咖啡")
  ✅ Important facts ("公司地址是北京朝阳区")
  ✅ Task outcomes ("已爬取 example.com 的数据")
  ✅ Knowledge learned ("Python 的 requests 库可以发 HTTP 请求")

  ❌ Greetings ("你好", "早上好")
  ❌ Acknowledgements ("好的", "明白了", "知道了")
  ❌ Small talk ("今天天气不错")
  ❌ Trivial tool output ("返回200 OK")

This class is the gatekeeper between the agent and CogniMem.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("agent.memory")


# ── Trivial patterns to IGNORE ──

_TRIVIAL_PATTERNS = [
    # Greetings
    r"^(你好|您好|嗨|hi|hello|hey|早上好|下午好|晚上好|晚安)[!！。.]*$",
    r"^(大家好|你们好)$",
    # Acknowledgements
    r"^(好的|好[的嘛]|可以|没问题|明白[了]?|知道[了]?|收到|了解|OK|ok|好的吧|行[吧]?|嗯[嗯]?)$",
    r"^(谢谢|多谢|感谢|thank|thanks|thx)[!！。.]*$",
    r"^(不用谢|不客气|you are welcome|welcome)$",
    r"^(没事|没关系|it's ok|no problem)$",
    # Very short
    r"^[!！?？。.]{1,3}$",
    r"^[0-9]{1,2}$",
    r"^.{0,2}$",  # 1-2 chars
    # Fillers
    r"^(继续|继续吧|请继续|接着说|然后呢|还有吗|还有呢)$",
    r"^(是的|对的|没错|对啊|就是|嗯嗯)$",
    # Trivial single words
    r"^(测试|test|debug|demo|试试)$",
]

# ── Important signals (if any of these appear, the info is likely important) ──

_IMPORTANT_TRIGGERS = [
    "喜欢", "不喜欢", "爱好", "习惯", "偏好", "口味",
    "住在", "工作在", "我在", "我是", "我叫",
    "记住", "记一下", "别忘了", "重要",
    "密码", "账号", "邮箱", "电话", "地址",
    "公司", "职位", "部门", "项目",
    "学会了", "发现", "搞定了", "完成了",
    "配置", "设置", "设定",
    "parent", "config", "setting", "password",
    "prefer", "like", "love", "hate",
]


class MemoryManager:
    """
    Decides what's worth remembering and what's not.

    Usage:
        mgr = MemoryManager()
        if mgr.should_store("我喜欢美式咖啡"):
            cogni.remember(...)

    Also extracts "important findings" from tool execution results.
    """

    @staticmethod
    def should_store(text: str, source: str = "user") -> bool:
        """Check if a piece of text is worth storing in long-term memory.

        Args:
            text: The content to evaluate.
            source: "user" | "agent" | "tool_result"

        Returns:
            True if this should be stored as a memory.
        """
        if not text or not text.strip():
            return False

        text = text.strip()

        # ── Trivial content (never store) ──
        for pattern in _TRIVIAL_PATTERNS:
            if re.match(pattern, text):
                logger.debug("Ignored trivial: %s", text[:40])
                return False

        # ── Too short (nothing meaningful) ──
        # 注意：中文一个词就2-3字，"我住在北京"=5字是有意义的
        if len(text) < 4:
            return False

        # ── Tool results: only store if they contain data ──
        if source == "tool_result":
            # Raw JSON-like results are rarely worth storing
            if text.startswith("{") or text.startswith("["):
                return False
            # Status messages
            if any(kw in text.lower() for kw in
                   ["success", "ok", "done", "200", "fetched"]):
                return False

        # ── Agent self-talk: rarely worth storing ──
        if source == "agent":
            if any(kw in text for kw in
                   ["让我", "我先", "下一步", "继续", "正在"]):
                return False

        # ── Important signals ──
        for trigger in _IMPORTANT_TRIGGERS:
            if trigger in text:
                logger.debug("Important trigger '%s' in: %s", trigger, text[:50])
                return True

        # ── User-provided information: only store if important triggers match ──
        # (caller _store_important_memories now uses _IMPORTANT_TRIGGERS directly)
        if source == "user":
            return True  # kept for backward compat, caller already filters

        return False

    @staticmethod
    def extract_important(findings: list[str]) -> list[str]:
        """Filter a list of potential memories, keeping only important ones."""
        return [f for f in findings if MemoryManager.should_store(f, source="tool_result")]

    @staticmethod
    def should_forget(memory: dict) -> bool:
        """Check if an EXISTING memory should be forgotten (low confidence + trivial).

        This is for the consolidation/grooming process.
        """
        confidence = memory.get("confidence", 0.5)
        content = memory.get("content", "") or memory.get("fact", "")

        # Very low confidence on already-trivial content → forget
        if confidence < 0.2 and len(content) < 30:
            return True

        # Content that was important once but is now outdated
        # (confidence decay handles this primarily, this is a secondary check)
        return False

    @staticmethod
    def categorize(text: str) -> str:
        """Suggest a memory type (fact, preference, goal, etc.) based on content."""
        if any(kw in text for kw in ["喜欢", "不喜欢", "prefer", "like", "爱好", "口味"]):
            return "preference"
        if any(kw in text for kw in ["目标", "goal", "计划", "打算", "想", "希望", "want"]):
            return "goal"
        if any(kw in text for kw in ["密码", "password", "账号", "account", "email", "邮箱"]):
            return "credential"
        if any(kw in text for kw in ["完成", "done", "搞定了", "finished", "complet"]):
            return "achievement"
        if any(kw in text for kw in ["配置文件", "config", "setting", "设置"]):
            return "config"
        return "fact"
