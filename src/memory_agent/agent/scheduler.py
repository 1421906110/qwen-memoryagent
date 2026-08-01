"""
后台记忆维护调度器 — 对标 OpenWorker `coworker/automation/scheduler.py`

## 设计

```text
Scheduler (async tick loop)
  ├─ run-once-catch-up (startup: 检查上次运行时间，补做错过的整理)
  ├─ tick every T seconds:
  │   ├─ groom_runner:  衰减/清理
  │   └─ consolidate_runner: 抽象化/去重/合并（每 N 次 tick 跑一次）
  └─ skip-on-overlap (同一任务不堆叠)
```

## 对比 OpenWorker

| 特性 | OpenWorker Scheduler | CogniMem Scheduler |
|------|---------------------|-------------------|
| 用途 | 执行定时自动化任务 | 定时记忆维护 |
| tick | 30s | 300s (可配置) |
| catch-up | run-once-catch-up | ✅ 同 |
| overlap guard | skip-on-overlap | ✅ 同 |
| spawn | spawned Task | await 执行（不阻塞主循环） |
| runner | 外部注入 | 直接调 cogni.consolidate |
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("agent.scheduler")

# 默认配置（秒）
_DEFAULT_TICK = 300       # 5 分钟
_DEFAULT_GROOM_INTERVAL = 1    # 每次 tick 都跑 groom（轻量）
_DEFAULT_CONSOLIDATE_EVERY = 6  # 每 6 次 tick 跑一次 consolidate（30分钟）


class BackgroundScheduler:
    """后台记忆维护调度器

    对标 OpenWorker `coworker/automation/scheduler.Scheduler`。

    Usage:
        scheduler = BackgroundScheduler(cogni=cogni_instance)
        scheduler.start()   # 启动后台循环
        ...
        await scheduler.stop()  # 应用关闭时停止
    """

    def __init__(
        self,
        cogni,
        *,
        tick_seconds: float = _DEFAULT_TICK,
        groom_runner: Optional[Callable[[str], Any]] = None,
        consolidate_runner: Optional[Callable[[str], Any]] = None,
        extra_tick: Optional[Callable[[], Awaitable[None]]] = None,
        llm_client=None,  # 🔥 v0.21: 子 Agent 事实验证需要
        # 如果 cogni 是 None 时也不报错（降级模式）
        allow_degraded: bool = True,
    ):
        """
        Args:
            cogni: CogniMem 实例（或其 compatible 对象，有 consolidate() 方法）
            tick_seconds: 后台循环间隔（秒）
            groom_runner: 自定义 groom 函数（默认调 cogni.consolidate()）
            consolidate_runner: 自定义 consolidate 函数
            extra_tick: 每次 tick 额外执行的回调
            llm_client: LLMClient 实例（传此参数启用 FactVerifier 矛盾自动解析）
            allow_degraded: cogni=None 时不崩溃
        """
        self.cogni = cogni
        self.tick_seconds = tick_seconds
        self.extra_tick = extra_tick
        self.allow_degraded = allow_degraded
        self._llm_client = llm_client

        # 运行状态
        self._task: Optional[asyncio.Task] = None
        self._running_groom: set[str] = set()        # overlap guard
        self._running_consolidate: set[str] = set()   # overlap guard
        self._tick_count: int = 0
        self._consolidate_every = _DEFAULT_CONSOLIDATE_EVERY

        # 统计
        self._stats = {
            "ticks": 0,
            "grooms": 0,
            "consolidates": 0,
            "errors": 0,
            "last_groom_time": 0,
            "last_consolidate_time": 0,
            "skipped_overlap": 0,
        }

        # 自定义 runner（默认用 cogni.consolidate 同时处理 groom 和 consolidate）
        self._groom_runner = groom_runner
        self._consolidate_runner = consolidate_runner

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    # ── 生命周期 ──

    def start(self) -> None:
        """启动后台循环（非阻塞）"""
        if self._task is not None:
            logger.info("Scheduler already running")
            return
        self._task = asyncio.create_task(self._loop())
        logger.info("⏰ 调度器已启动 (tick=%ds, consolidate_every=%dticks)",
                     self.tick_seconds, self._consolidate_every)

    async def stop(self) -> None:
        """停止后台循环"""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("⏰ 调度器已停止 (统计: %s)", self._stats)

    # ── 主循环 ──

    async def _loop(self) -> None:
        """主循环

        对标 OpenWorker `Scheduler._loop()`：
        - 首次 tick = catch-up（运行当前需要维护的）
        - 后续按 tick_seconds 间隔
        """
        # First pass = catch-up
        try:
            await self._tick(is_catchup=True)
            self._tick_count += 1
        except Exception as e:
            logger.exception("调度器 catch-up 失败: %s", e)

        while True:
            await asyncio.sleep(self.tick_seconds)
            self._tick_count += 1
            try:
                await self._tick(is_catchup=False)
            except Exception as e:
                logger.exception("调度器 tick 失败: %s", e)
                self._stats["errors"] += 1

    async def _tick(self, *, is_catchup: bool = False) -> None:
        """一次 tick：跑 groom + （按需）consolidate

        Args:
            is_catchup: 是否是 catch-up 首次运行
        """
        self._stats["ticks"] += 1
        tag = "catchup" if is_catchup else "tick"
        logger.debug("⏰ 调度器 %s #%d", tag, self._tick_count)

        if not self.cogni:
            if self.allow_degraded:
                return
            raise RuntimeError("CogniMem not available")

        # ── 1. Groom（每次 tick 都跑，轻量操作）──
        #    对所有活跃 agent 执行衰减清理
        try:
            await self._run_groom()
        except Exception as e:
            logger.exception("groom 失败: %s", e)
            self._stats["errors"] += 1

        # ── 2. Consolidate（每 N 次 tick 跑一次，较重）──
        if self._tick_count % self._consolidate_every == 0 or is_catchup:
            try:
                await self._run_consolidate()
            except Exception as e:
                logger.exception("consolidate 失败: %s", e)
                self._stats["errors"] += 1

        # ── 3. Extra tick ──
        if self.extra_tick is not None:
            try:
                await self.extra_tick()
            except Exception as e:
                logger.exception("调度器 extra_tick 失败: %s", e)

        # 统计（不是 catch-up 时才记录 tick）
        if not is_catchup:
            self._stats["last_groom_time"] = time.time()

    # ── 具体任务 ──

    async def _get_active_agents(self) -> list[str]:
        """获取所有有数据的 agent_id

        返回 ["default"] 兜底，不抛异常。
        """
        try:
            if hasattr(self.cogni, 'fact_network') and self.cogni.fact_network:
                db = self.cogni.fact_network.db
                if db:
                    from contextlib import contextmanager
                    try:
                        with db._cursor_ctx() as cur:
                            cur.execute("""
                                SELECT DISTINCT agent_id FROM facts
                            """)
                            rows = cur.fetchall()
                            if rows:
                                return [r[0] for r in rows]
                    except Exception:
                        pass
        except Exception:
            pass
        return ["default"]

    async def _run_groom(self) -> None:
        """对所有活跃 agent 执行 groom（衰减清理）

        skip-on-overlap：同一 agent 的 groom 不堆叠。
        对标 OpenWorker `Scheduler.run_task()` 的 overlap guard。
        """
        agents = await self._get_active_agents()

        for agent_id in agents:
            if agent_id in self._running_groom:
                self._stats["skipped_overlap"] += 1
                logger.debug("⏭️ 跳过 groom(%s) — 上一次还在运行", agent_id)
                continue

            self._running_groom.add(agent_id)
            try:
                if self._groom_runner:
                    result = self._groom_runner(agent_id)
                else:
                    # 默认：调 cogni.consolidate（自带衰减+去重）
                    result = await asyncio.to_thread(
                        self.cogni.consolidate, agent_id
                    )
                decayed = result.get("decayed", 0) if isinstance(result, dict) else 0
                if decayed > 0:
                    logger.info("🗑️ 调度器 groom(%s): 衰减 %d 条", agent_id, decayed)
                self._stats["grooms"] += 1
            except Exception as e:
                logger.warning("groom(%s) 失败: %s", agent_id, e)
                self._stats["errors"] += 1
            finally:
                self._running_groom.discard(agent_id)

    async def _run_consolidate(self) -> None:
        """对所有活跃 agent 执行 consolidate（抽象化+去重）

        skip-on-overlap 同 groom。
        只对数据量 > 10 条的 agent 执行（过少没必要）。
        """
        agents = await self._get_active_agents()

        for agent_id in agents:
            if agent_id in self._running_consolidate:
                self._stats["skipped_overlap"] += 1
                logger.debug("⏭️ 跳过 consolidate(%s) — 上一次还在运行", agent_id)
                continue

            # 跳过数据量过少的 agent
            try:
                if hasattr(self.cogni, 'get_stats'):
                    stats = await asyncio.to_thread(self.cogni.get_stats, agent_id)
                    total = stats.get("total_facts", 0) if isinstance(stats, dict) else 0
                    if total < 10:
                        continue
            except Exception:
                pass

            self._running_consolidate.add(agent_id)
            try:
                if self._consolidate_runner:
                    result = self._consolidate_runner(agent_id)
                else:
                    result = await asyncio.to_thread(
                        self.cogni.consolidate, agent_id
                    )
                merged = result.get("merged", 0) if isinstance(result, dict) else 0
                abstracted = result.get("abstracted", 0) if isinstance(result, dict) else 0
                contradictions = result.get("contradictions", 0) if isinstance(result, dict) else 0
                if merged > 0 or abstracted > 0:
                    logger.info("🧹 调度器 consolidate(%s): 合并%d 抽象化%d 矛盾%d",
                                agent_id, merged, abstracted, contradictions)

                # 🔥 v0.21.1: 子 Agent 批量解析矛盾（1次LLM调用替代N次）
                if contradictions > 0 and self._llm_client is not None:
                    try:
                        from memory_agent.agent.subagent import FactVerifier
                        verifier = FactVerifier(self._llm_client)
                        all_facts = self.cogni.fact_network._get_agent_facts(agent_id)
                        contradictions_list = self.cogni.fact_network.get_contradictions(agent_id)

                        pairs = []
                        for c in contradictions_list[:5]:
                            fa = next((f for f in all_facts if f.fact_id == c.fact_a_id), None)
                            fb = next((f for f in all_facts if f.fact_id == c.fact_b_id), None)
                            if fa and fb:
                                pairs.append((fa, fb))  # 🐛 保存对象本身，与 verdicts 严格对齐

                        if pairs:
                            verdicts = verifier.batch_verify(
                                [(fa.to_dict(), fb.to_dict()) for fa, fb in pairs],
                                agent_id=agent_id,
                            )
                            fn = self.cogni.fact_network
                            resolved = 0
                            # 🐛 修复: zip 必须对齐 pairs（pairs 已过滤缺失，长度可能 < contradictions_list[:5]，
                            #    旧代码 zip 错位会把 verdict 应用到错误的 contradiction 上）
                            for verdict, (fa, fb) in zip(verdicts, pairs):
                                if verdict.error or not verdict.winner_id:
                                    continue
                                resolved += 1
                                # 🐛 修复: _update_confidence 签名是 (FactTriple, delta, reason)，
                                #    delta 是增量不是绝对值；旧代码传 fact_id+绝对值 → AttributeError/缺参
                                if verdict.winner_id == fa.fact_id:
                                    fn._update_confidence(fa, 0.15, "scheduler_contradiction_verdict")
                                    fn._update_confidence(fb, -0.10, "scheduler_contradiction_verdict")
                                else:
                                    fn._update_confidence(fb, 0.15, "scheduler_contradiction_verdict")
                                    fn._update_confidence(fa, -0.10, "scheduler_contradiction_verdict")
                                # 🐛 v0.30: 裁决置信度必须落库（只改内存缓存 → 重启还原）
                                for _f in (fa, fb):
                                    try:
                                        fn.db.update_fact(_f)
                                    except Exception:
                                        pass
                            if resolved:
                                logger.info("🔍 调度器批量矛盾解析(%s): 解决 %d/%d 对 (1次LLM)", agent_id, resolved, len(pairs))
                    except Exception as e:
                        logger.warning("调度器矛盾解析失败: %s", e)

                self._stats["consolidates"] += 1
                self._stats["last_consolidate_time"] = time.time()
            except Exception as e:
                logger.warning("consolidate(%s) 失败: %s", agent_id, e)
                self._stats["errors"] += 1
            finally:
                self._running_consolidate.discard(agent_id)
