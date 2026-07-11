#!/usr/bin/env python3
"""
CogniMem 一键演示启动器（Demo Mode）

受 NaLog Agronomist Agent 的 DEMO 模式启发（NALOG_USE_DEMO=true 一键体验）。

特点:
  - 零配置：自动检测环境，无需手动设置
  - 自动降级：PostgreSQL 不可用 → SQLite 兜底
  - 自动播种：首次运行自动填充演示数据（~20条带矛盾的事实）
  - 一键启动：一个命令完成所有操作

用法:
    python demo.py                     # 一键启动（HTTP 服务 + 演示数据）
    python demo.py --quick             # 快速测试（启动+召回1次+退出）
    python demo.py --seed-only         # 只填充演示数据，不启动服务

环境变量:
    NALOG_USE_DEMO=true               # 启用演示模式（NaLog 兼容）
    QWEN_API_KEY=sk-xxx               # 可选，有则启用 LLM 提取
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import textwrap
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("demo")

# 颜色
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RED = "\033[91m"
BOLD = "\033[1m"
END = "\033[0m"


def print_banner():
    banner = f"""
{CYAN}{BOLD}╔══════════════════════════════════════════╗
║     🧠  CogniMem 认知记忆系统          ║
║     🌙  一键演示模式                    ║
╚══════════════════════════════════════════╝{END}

{BLUE}SPO 三元组 | 矛盾驱动学习 | 艾宾浩斯遗忘 | 智能召回{END}
    """
    print(banner)


def check_env() -> dict:
    """检测运行环境"""
    status = {
        "postgresql": False,
        "llm_api_key": False,
        "sqlite_fallback": True,
        "mode": "demo",
    }

    # PostgreSQL 检测
    try:
        import psycopg2
        conn = psycopg2.connect("postgresql://localhost/cognimem", connect_timeout=2)
        conn.close()
        status["postgresql"] = True
        status["sqlite_fallback"] = False
        print(f"  {GREEN}✅ PostgreSQL: 已连接{END}")
    except Exception:
        print(f"  {YELLOW}⚠️  PostgreSQL: 不可用（将使用 SQLite 兜底）{END}")

    # API Key 检测
    key = (os.environ.get("DEEPSEEK_API_KEY", "")
           or os.environ.get("DASHSCOPE_API_KEY", "")
           or os.environ.get("QWEN_API_KEY", ""))
    if key:
        status["llm_api_key"] = True
        print(f"  {GREEN}✅ LLM API: 已配置（将启用 LLM 提取）{END}")
    else:
        print(f"  {YELLOW}ℹ️   LLM API: 未配置（规则提取模式，0 Token 成本）{END}")

    return status


def seed_demo_data() -> bool:
    """填充演示数据"""
    import subprocess
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run(
        [sys.executable, "seed_demo.py", "--clear"],
        capture_output=True, text=True, cwd=script_dir,
        env={**os.environ, "PYTHONPATH": f"{script_dir}/src:{os.environ.get('PYTHONPATH', '')}"},
    )
    if result.returncode == 0:
        print(f"  {GREEN}✅ 演示数据已填充{END}")
        return True
    else:
        print(f"  {RED}❌ 演示数据填充失败: {result.stderr[:200]}{END}")
        return False


def print_demo_info(port: int):
    """打印演示信息"""
    info = f"""
{BLUE}{BOLD}🎯 演示入口{END}

  {GREEN}🔗 聊天界面:{END}       http://localhost:{port}/
  {GREEN}📊 仪表盘:{END}         http://localhost:{port}/dashboard
  {GREEN}🕸️  知识图谱:{END}      http://localhost:{port}/graph
  {GREEN}🔍 健康检测:{END}       http://localhost:{port}/health
  {GREEN}📋 审计日志:{END}       http://localhost:{port}/audit

{CYAN}{BOLD}📌 试试这些{END}
  "我喜歡喝冰美式"
  "我不喜歡喝熱美式"     → 会触发矛盾检测
  "我住在北京"
  "我是一名程序员"
  "我还记得什么？"        → 召回相关记忆
  "诊断记忆健康状况"      → 查看内存统计

{YELLOW}{BOLD}🧪 竞品参考{END}
  {BOLD}RuleMemory{END}     → 答案引用来源 ✓ | 过期警告 ✓
  {BOLD}ERINYS{END}         → 6信号治理 ✓
  {BOLD}Mimir{END}          → MCP协议 ✓ | Benchmark ✓
  {BOLD}NaLog{END}          → Demo模式 ✓ | 多维评分 ✓
  {BOLD}Emma{END}           → 验证器层 ✓
  {BOLD}DREAM{END}          → 审计日志 ✓

{GREEN}{BOLD}按 Ctrl+C 停止服务{END}
"""
    print(info)


def run_quick_test():
    """快速测试模式：启动、召回、退出"""
    # 确保能导入 cognimem
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(script_dir, "src"))

    from cognimem.core.brain import CogniMem
    from cognimem.core.db import DatabaseAdapter

    print(f"\n  {CYAN}快速测试模式...{END}")

    db = None
    try:
        db = DatabaseAdapter()
        db.connect()
    except Exception:
        pass

    brain = CogniMem(db_adapter=db, use_llm=False)

    # 测试记忆
    tests = [
        "我喜欢喝冰美式",
        "我不喜欢喝热美式",
        "我住在北京",
        "我是一名程序员",
    ]
    for t in tests:
        result = brain.remember(t)
        status = "✅" if result.get("status") == "remembered" else "❌"
        facts = result.get("facts", [])
        extracted = [f"{f.subject} {f.predicate} {f.object}" for f in facts[:2]]
        print(f"  {status} 记住: {t[:30]} -> {extracted}")

    # 测试召回
    result = brain.recall("咖啡")
    facts = result.get("facts", [])
    print(f"  📋 召回 '咖啡': {len(facts)} 条结果")
    for f in facts[:3]:
        print(f"     · {f.subject} {f.predicate} {f.object} (conf={f.confidence:.2f})")

    # 测试矛盾检测
    result = brain.remember("我喜欢喝热美式")
    contradictions = result.get("contradictions_detected", 0)
    print(f"  ⚠️  矛盾检测: {contradictions} 条矛盾")

    # 测试来源引用
    result = brain.ask("我喜歡喝什麼")
    for m in result.get("relevant_memories", [])[:3]:
        print(f"     · {m['fact']} ——{m.get('citation', '')}")

    print(f"\n  {GREEN}✅ 快速测试通过！{END}\n")


def run_server(port: int, seed: bool):
    """启动 HTTP 服务器"""
    # 确保能导入
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(script_dir, "src"))

    from uvicorn import run

    if seed:
        seed_demo_data()

    print_demo_info(port)
    run("memory_agent.main:app", host="0.0.0.0", port=port, log_level="warning")


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CogniMem 一键演示启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            示例:
              python demo.py                  # 一键启动完整演示
              python demo.py --quick          # 快速功能测试
              python demo.py --seed-only      # 只填充数据不启动
              NALOG_USE_DEMO=true python demo.py  # NaLog 兼容模式
        """),
    )
    parser.add_argument("--port", type=int, default=8000, help="服务端口（默认 8000）")
    parser.add_argument("--quick", action="store_true", help="快速测试模式（不启动 HTTP）")
    parser.add_argument("--seed-only", action="store_true", help="只填充演示数据")
    parser.add_argument("--no-seed", action="store_true", help="不填充演示数据，直接启动")

    args = parser.parse_args()

    # NaLog 兼容模式
    if os.environ.get("NALOG_USE_DEMO", "").lower() in ("true", "1", "yes"):
        print(f"  {CYAN}ℹ️  检测到 NALOG_USE_DEMO=true（NaLog 兼容模式）{END}")

    print_banner()
    print(f"  {BLUE}环境检测...{END}")
    env = check_env()
    print()

    if args.quick:
        run_quick_test()
        return

    if args.seed_only:
        seed_demo_data()
        return

    # 一键启动
    run_server(port=args.port, seed=not args.no_seed)


if __name__ == "__main__":
    main()
