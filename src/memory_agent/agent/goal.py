"""
Goal Context — track what the agent is trying to accomplish.

The core insight: a real agent doesn't just respond to each user message.
It keeps the END GOAL in mind and keeps working until done.

Architecture:
  GoalContext stores the user's original request, the agent's interpretation,
  a queue of sub-goals, and completion criteria. Every iteration of the
  agent loop checks "are we done yet?" instead of "did the LLM stop calling tools?"

This is the single most important change to turn CogniMem into a real agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("agent.goal")


class GoalStatus(Enum):
    PENDING = "pending"        # Created but not started
    PLANNING = "planning"      # Agent is figuring out steps
    RUNNING = "running"        # Actively being worked on
    COMPLETED = "completed"    # All sub-goals done
    FAILED = "failed"          # Cannot be achieved
    PARTIAL = "partial"        # Some done, some failed


@dataclass
class SubGoal:
    """One step in the agent's task plan."""
    description: str            # What this step achieves
    status: GoalStatus = GoalStatus.PENDING
    tool_used: str | None = None
    result: dict | None = None
    retries: int = 0
    max_retries: int = 3
    error: str | None = None


@dataclass
class GoalContext:
    """
    Tracks the agent's current task end-to-end.

    Usage:
        goal = GoalContext(user_request="帮我爬 example.com 的内容")
        goal.plan = [
            SubGoal("获取页面 HTML"),
            SubGoal("解析标题和正文"),
            SubGoal("保存为文件"),
        ]
        # Agent executes each sub-goal, checking goal.is_complete() after each
    """

    # ── Origin ──
    original_request: str                # User's raw message
    description: str = ""                # Agent's interpreted goal

    # ── Planning ──
    plan: list[SubGoal] = field(default_factory=list)
    current_step: int = 0
    auto_plan: bool = True              # Whether agent should plan before acting

    # ── Execution state ──
    status: GoalStatus = GoalStatus.PENDING
    max_retries_per_step: int = 3
    iterations_used: int = 0
    tools_called: int = 0
    completed_sub_goals: list[SubGoal] = field(default_factory=list)
    failed_sub_goals: list[SubGoal] = field(default_factory=list)

    # ── Memories to store at the end ──
    important_facts: list[str] = field(default_factory=list)

    # ── Results ──
    final_result: str = ""

    def set_description(self, desc: str) -> None:
        """Set the interpreted goal description from the LLM."""
        self.description = desc
        logger.info("🎯 Goal: %s", desc[:100])

    def add_step(self, description: str) -> SubGoal:
        """Add a sub-goal step."""
        sg = SubGoal(description=description)
        self.plan.append(sg)
        return sg

    def set_plan(self, steps: list[str]) -> None:
        """Set full plan from a list of step descriptions."""
        self.plan = [SubGoal(description=s) for s in steps]
        logger.info(
            "📋 Plan: %d steps — %s",
            len(steps), "; ".join(s[:50] for s in steps),
        )

    def current(self) -> SubGoal | None:
        """Get the current sub-goal (or None if done)."""
        if self.current_step < len(self.plan):
            return self.plan[self.current_step]
        return None

    def advance(self) -> SubGoal | None:
        """Mark current step as completed and move to next."""
        current = self.current()
        if current:
            current.status = GoalStatus.COMPLETED
            self.completed_sub_goals.append(current)
            logger.info("✅ Step %d/%d done: %s",
                        self.current_step + 1, len(self.plan), current.description[:60])
        self.current_step += 1
        return self.current()

    def mark_failed(self, error: str) -> None:
        """Mark current step as failed."""
        current = self.current()
        if current:
            current.status = GoalStatus.FAILED
            current.error = error
            self.failed_sub_goals.append(current)
            logger.warning("❌ Step %d/%d failed: %s — %s",
                          self.current_step + 1, len(self.plan),
                          current.description[:60], error[:100])
        self.current_step += 1

    def is_complete(self) -> bool:
        """Check if ALL steps are done (completed or failed)."""
        if not self.plan:
            return False
        completed = len(self.completed_sub_goals) + len(self.failed_sub_goals)
        return completed >= len(self.plan)

    def should_retry(self, error: str) -> bool:
        """Decide if current step should be retried based on error type."""
        current = self.current()
        if not current:
            return False
        if current.retries >= current.max_retries:
            return False
        # Don't retry certain errors
        if not error:
            return True  # 无错误信息 → 默认重试
        no_retry_keywords = ["permission denied", "invalid syntax",
                             "does not exist", "no such file",
                             # ⭐ 中文 locale 错误提示
                             "权限不足", "权限拒绝", "权限被拒绝",
                             "文件不存在", "找不到文件",
                             "语法错误", "语法不正确",
                             "不存在", "无法访问", "被拒绝"]
        if any(kw in error.lower() for kw in no_retry_keywords):
            return False
        current.retries += 1
        return True

    def get_summary(self) -> dict:
        """Get a summary of goal progress for the LLM prompt."""
        done = len(self.completed_sub_goals)
        failed = len(self.failed_sub_goals)
        total = len(self.plan)
        return {
            "original_request": self.original_request,
            "description": self.description,
            "progress": f"{done + failed}/{total}",
            "completed": done,
            "failed": failed,
            "total": total,
            "status": self.status.value,
        }

    def to_dict(self) -> dict:
        """Serialise for API response."""
        return {
            "original_request": self.original_request,
            "description": self.description,
            "plan": [s.description for s in self.plan],
            "completed": len(self.completed_sub_goals),
            "failed": len(self.failed_sub_goals),
            "total": len(self.plan),
            "status": self.status.value,
            "tools_called": self.tools_called,
            "iterations_used": self.iterations_used,
            "important_facts_added": len(self.important_facts),
            "final_result": self.final_result[:500] if self.final_result else "",
        }
