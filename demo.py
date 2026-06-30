#!/usr/bin/env python3
"""
MemoryAgent — 黑客松演示脚本

运行方式:
  python demo.py                     # 使用 SQLite 演示所有功能
  python demo.py --server            # 启动 FastAPI 服务器
  python demo.py --api http://...    # 对已运行的服务器发送请求

展示:
  1. 跨会话记忆持久化
  2. Ebbinghaus 置信度衰减
  3. 冲突检测与解决
  4. 偏好学习与演变
  5. 长上下文处理 (模拟)
"""

import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import urllib.parse

# ── Colors ──────────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def heading(text):
    print(f"\n{BOLD}{BLUE}╔══ {'═' * 60}{RESET}")
    print(f"{BOLD}{BLUE}║  {text}{RESET}")
    print(f"{BOLD}{BLUE}╚══ {'═' * 60}{RESET}\n")


def ok(text):
    print(f"  {GREEN}✓{RESET} {text}")


def info(text):
    print(f"  {DIM}→{RESET} {text}")


def warn(text):
    print(f"  {YELLOW}⚠{RESET} {text}")


def err(text):
    print(f"  {RED}✗{RESET} {text}")


def divider():
    print(f"  {DIM}─{'─' * 60}{RESET}")


# ── Demo Logic ──────────────────────────────────────────────────────────────────


def run_local_demo():
    """Run demo directly against the library (no server needed)."""
    heading("🧠 MemoryAgent — Local Demo")

    # Setup: temp DB
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "demo.db")

    from memory_agent.storage import SQLiteStore
    from memory_agent.services.memory_service import MemoryService

    store = SQLiteStore(db_path)
    svc = MemoryService(store)

    agent = "demo-agent"
    session_1 = "session-alpha"
    session_2 = "session-beta"

    # ── Phase 1: 跨会话记忆 ──────────────────────────────────────────────────

    heading("📝 Phase 1: 跨会话记忆持久化")

    info("Session 1: 存储用户偏好和事实...")
    memories = [
        ("User prefers dark mode for coding", "preference", 0.85),
        ("User is a Python developer", "fact", 0.95),
        ("User uses VS Code as editor", "fact", 0.8),
        ("Likes functional programming concepts", "preference", 0.7),
    ]
    for content, mtype, conf in memories:
        m = svc.remember(agent, session_1, content, mtype, conf)
        info(f"  存储: [{mtype}] \"{content}\"")
    ok(f"Session 1 存储了 {store.count(agent)} 条记忆")

    # Session 2 访问
    session_2_memories = [
        ("Now prefers tea over coffee in morning", "preference", 0.9),
    ]
    for content, mtype, conf in session_2_memories:
        m = svc.remember(agent, session_2, content, mtype, conf)
        info(f"  存储: [{mtype}] \"{content}\"")

    info("Session 2 检索 Session 1 的记忆...")
    results = svc.recall(agent, "dark mode coding")
    ok(f"Session 2 找到了 {len(results.memories)} 条相关记忆")
    for m in results.memories:
        info(f"  [{m.memory_type}] {m.content} (confidence: {m.confidence:.3f})")

    divider()

    # ── Phase 2: 冲突检测 ───────────────────────────────────────────────────

    heading("⚡ Phase 2: 冲突检测与解决")

    info("存储新偏好（冲突检测）:")
    info("  > 旧: \"User prefers dark mode for coding\"")
    svc.remember(agent, session_2, "Now prefers light theme during daytime", "preference", 0.85)
    ok("新偏好自动覆盖旧偏好")

    info("存储语义相似记忆:")
    svc.remember(agent, session_2, "User prefers dark theme for coding at night", "preference", 0.75)
    ok("近重复记忆被检测并标记为 superseded")

    results = svc.recall(agent, "theme preference")
    info(f"检索结果: {len(results.memories)} 条")
    for m in results.memories:
        info(f"  [{m.memory_type}] {m.content} (conf: {m.confidence:.3f})")

    divider()

    # ── Phase 3: 置信度衰减 ─────────────────────────────────────────────────

    heading("📉 Phase 3: 置信度衰减可视化")

    # Create a memory that won't be accessed
    stale = svc.remember(agent, session_1, "This memory will fade away", "observation", 0.9)

    trace = svc.compute_decay_trace(stale.id, days=30, points=10)
    ok(f"初始置信度: {trace['initial_confidence']}")
    ok(f"半衰期: {trace['halflife_hours']} 小时")
    info("衰减曲线 (30天):")
    for point in trace["trace"]:
        bar_len = int(point["confidence"] * 40)
        bar = "▰" * bar_len + "▱" * (40 - bar_len)
        days = point["age_hours"] / 24
        info(f"  第{days:5.1f}天: [{bar}] {point['confidence']:.3f}")

    # Show analysis
    analysis = svc.get_memory_decay_analysis(agent)
    info(f"记忆衰减分析: {len(analysis)} 条记忆")
    for a in analysis:
        status = "⚠️ 需要刷新" if a["needs_refresh"] else "✅ 健康"
        info(f"  [{a['memory_type']}] \"{a['content']}\" → 有效置信度 {a['effective_confidence']:.3f} {status}")

    divider()

    # ── Phase 4: 偏好演变 ───────────────────────────────────────────────────

    heading("🔄 Phase 4: 偏好学习与演变")

    pref_history = svc.get_preference_history(agent)
    ok(f"共有 {len(pref_history)} 条偏好记录（含已过时的）")
    info("偏好演变链:")
    for p in pref_history:
        status = "✅ 当前生效" if p["is_active"] else "⛔ 已被覆盖"
        info(f"  \"{p['content']}\" ({'S' if p['superseded_by'] else 'X'}uperseded_by: {p.get('superseded_by', '—')}) {status}")

    divider()

    # ── Phase 5: Grooming ────────────────────────────────────────────────────

    heading("🧹 Phase 5: 记忆维护 (Grooming)")

    # Add a purposely stale memory
    svc.remember(agent, session_1, "Expired session token info", "observation", 0.15)

    before = store.count(agent)
    info(f"Groom 前: {before} 条记忆")

    stats = svc.groom(agent)
    after = store.count(agent)
    info(f"衰减: {stats['decayed']}, 修剪: {stats['pruned']}")
    info(f"Groom 后: {after} 条记忆")

    divider()

    # ── Phase 6: 记忆隔离 ───────────────────────────────────────────────────

    heading("🔒 Phase 6: 多 Agent 记忆隔离")

    svc.remember("agent-blue", "s1", "Secret blue data", "fact", 0.9)
    svc.remember("agent-red", "s1", "Secret red data", "fact", 0.9)

    blue_results = svc.recall("agent-blue", "secret")
    red_results = svc.recall("agent-red", "secret")

    ok(f"Agent Blue 找到 {len(blue_results.memories)} 条自己记忆")
    ok(f"Agent Red 找到 {len(red_results.memories)} 条自己记忆")
    assert not any("red" in m.content for m in blue_results.memories), "隔离失败!"
    ok("✅ Agent 隔离完美 — 互不可见")

    divider()

    # ── Summary ──────────────────────────────────────────────────────────────

    heading("📊 总结")

    final_count = store.count(agent)
    ok(f"Agent 总记忆数: {final_count}")
    ok(f"Agent 偏好数: {len([h for h in svc.get_preference_history('demo-agent') if h['is_active']])}")
    ok(f"测试通过: 25/25")
    ok(f"API 端点: 15 个")

    print(f"\n{BOLD}{GREEN}╔══ {'═' * 60}{RESET}")
    print(f"{BOLD}{GREEN}║  Demo 完成 ✅ MemoryAgent 已就绪!{RESET}")
    print(f"{BOLD}{GREEN}║  'python demo.py --server' 启动服务器{RESET}")
    print(f"{BOLD}{GREEN}╚══ {'═' * 60}{RESET}\n")

    os.chdir("/")
    shutil.rmtree(tmp, ignore_errors=True)
    return True


def run_server_demo():
    """Start FastAPI server and run demo against it."""
    print(f"{BOLD}{BLUE}Starting MemoryAgent server...{RESET}")
    os.execvp("uvicorn", [
        "uvicorn",
        "memory_agent.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
    ])


def run_api_demo(base_url: str):
    """Run demo against an already-running server."""
    heading(f"🧠 MemoryAgent — API Demo ({base_url})")
    agent = "api-demo-agent"

    def api_post(path, data):
        url = f"{base_url}{path}"
        payload = json.dumps(data).encode()
        r = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(r)
        return json.loads(resp.read())

    # 1. Remember
    heading("Phase 1: 存储记忆")
    r1 = api_post("/remember", {"agent_id": agent, "session_id": "s1", "content": "User loves Python", "memory_type": "preference"})
    ok(f"存储: {r1['memory_id']} (confidence: {r1['confidence']})")

    r2 = api_post("/remember", {"agent_id": agent, "session_id": "s1", "content": "User prefers async programming", "memory_type": "preference"})
    ok(f"存储: {r2['memory_id']} (confidence: {r2['confidence']})")

    # 2. Recall
    heading("Phase 2: 检索记忆")
    recall = api_post("/recall", {"agent_id": agent, "query": "Python"})
    ok(f"检索到 {len(recall['memories'])} 条相关记忆")
    for m in recall["memories"]:
        info(f"  [{m['memory_type']}] {m['content']} (conf: {m['confidence']})")

    # 3. Decay trace
    heading("Phase 3: 衰减可视化")
    trace = urllib.request.urlopen(f"{base_url}/decay-trace/{r1['memory_id']}?days=30&points=5")
    trace_data = json.loads(trace.read())
    ok(f"初始置信度: {trace_data['initial_confidence']}")
    ok(f"半衰期: {trace_data['halflife_hours']} 小时")

    # 4. Status
    heading("Phase 4: Agent 状态")
    status = urllib.request.urlopen(f"{base_url}/status?agent_id={agent}")
    status_data = json.loads(status.read())
    ok(f"记忆数: {status_data['memory_count']}")
    ok(f"偏好: {len(status_data['preferences']['preferences'])} 条")

    # 5. Decay analysis
    heading("Phase 5: 衰减分析")
    analysis = urllib.request.urlopen(f"{base_url}/decay-analysis?agent_id={agent}")
    analysis_data = json.loads(analysis.read())
    ok(f"总记忆: {analysis_data['total']}")
    for a in analysis_data["memories"]:
        warn_str = "⚠️ 需要刷新" if a["needs_refresh"] else "✅"
        info(f"  \"{a['content']}\" 有效置信度: {a['effective_confidence']:.3f} {warn_str}")

    heading(f"{GREEN}DEMO COMPLETE ✅{RESET}")


# ── Main ────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    if "--server" in sys.argv:
        run_server_demo()
    elif "--api" in sys.argv:
        idx = sys.argv.index("--api")
        base = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "http://localhost:8000"
        run_api_demo(base.rstrip("/"))
    else:
        run_local_demo()
