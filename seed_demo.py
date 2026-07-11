#!/usr/bin/env python3
"""
seed_demo.py — CogniMem 演示数据填充脚本

为评委准备一套高质量演示数据，展示系统的核心能力：
  - 多种事实类型：preference / fact / goal / decision / observation / action
  - 矛盾检测：直接否定 "喜欢" vs "不喜欢"（同品类）
  - 关系图谱：多个三元组形成知识网络
  - 置信度差异：从 0.3（不确定）到 0.95（确信）
  - 抽象化素材：同一 subject+predicate 多 object 且同标签

用法:
    cd ~/projects/qwen-memoryagent
    source .venv/bin/activate
    python seed_demo.py                    # 填充 default agent
    python seed_demo.py --agent demo_user  # 填充指定 agent
    python seed_demo.py --clear            # 清空再填充

数据内容（default agent，共 ~20 条）：
  - 8 条偏好（含 2 条矛盾用于展示矛盾检测）
  - 4 条事实信息
  - 2 个目标
  - 2 条决策历史
  - 2 条观察记录
  - 2 条行为记录（action）
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("seed_demo")

# ── 确保能找到 cognimem 模块 ──
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ── .env 加载 ──
_env_path = os.path.join(_THIS_DIR, ".env")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def parse_args():
    parser = argparse.ArgumentParser(description="CogniMem 演示数据填充")
    parser.add_argument("--agent", default="default", help="目标 agent ID（默认: default）")
    parser.add_argument("--clear", action="store_true", help="填充前先清空该 agent 数据")
    parser.add_argument("--db", default="", help="PostgreSQL DSN（默认从 .env 读 COGNIMEM_DB）")
    return parser.parse_args()


def get_db(dsn: str):
    """连接 PostgreSQL 并返回 DatabaseAdapter"""
    from cognimem.core.db import DatabaseAdapter
    if not dsn:
        dsn = os.environ.get("COGNIMEM_DB", "postgresql://localhost/cognimem")
    db = DatabaseAdapter(dsn=dsn)
    db.connect()
    # 确保表存在（首次运行自动建表，已存在则跳过）
    try:
        db.create_tables()
    except Exception:
        pass  # 表已存在，正常
    return db


def clear_agent(db, agent_id: str):
    """清空指定 agent 的所有数据（FK 顺序）"""
    with db._plain_cursor_ctx() as cur:
        cur.execute("DELETE FROM contradictions WHERE agent_id = %s", (agent_id,))
        cur.execute("DELETE FROM confidence_log WHERE agent_id = %s", (agent_id,))
        cur.execute("DELETE FROM fact_versions WHERE agent_id = %s", (agent_id,))
        cur.execute("DELETE FROM facts WHERE agent_id = %s", (agent_id,))
    logger.info("🗑️ 已清空 agent '%s' 的所有数据", agent_id)


def seed_agent(db, agent_id: str):
    """填充演示数据"""
    from cognimem.core.models import FactTriple, EvidenceItem

    now = datetime.now(timezone.utc)
    facts = []

    def add(subj, pred, obj, ftype="general", conf=0.6, imp=0.5,
            level="raw", tags=None, session=""):
        f = FactTriple(
            subject=subj, predicate=pred, object=obj,
            agent_id=agent_id, fact_type=ftype, confidence=conf,
            importance=imp, encoding_level=level,
            context_tags=tags or [],
            source_session=session or f"demo:{ftype}",
            evidence=[EvidenceItem(source="demo_seed", statement=f"{subj}{pred}{obj}")],
        )
        facts.append(f)

    # ═══════════════════════════════════════════
    #  偏好（preference）— 包含矛盾用于展示
    # ═══════════════════════════════════════════

    add("user", "喜欢", "喝冰美式咖啡", "preference", 0.85, 0.6,
        tags=["咖啡", "饮品"], session="demo:pref:coffee")
    add("user", "不喜欢", "喝热美式", "preference", 0.80, 0.5,
        tags=["咖啡", "饮品"], session="demo:pref:coffee")
    add("user", "喜欢", "吃火锅", "preference", 0.75, 0.5,
        tags=["美食", "川菜"], session="demo:pref:food")
    add("user", "喜欢", "吃日料", "preference", 0.70, 0.4,
        tags=["美食", "日料"], session="demo:pref:food")
    add("user", "喜欢", "蓝色", "preference", 0.90, 0.3,
        tags=["颜色", "设计"], session="demo:pref:design")
    add("user", "喜欢", "极简风格", "preference", 0.80, 0.4,
        tags=["设计", "风格"], session="demo:pref:design")

    # ⭐ 矛盾对：同品类 "喜欢" vs "不喜欢" → 触发矛盾检测
    # 注意：这两条通过 brain.remember() 走完整管道（含矛盾检测），不直接写库
    # 先保存，后面再专门处理
    add("user", "喜欢", "喝冰美式", "preference", 0.70, 0.6,
        tags=["咖啡", "饮品"], session="demo:contradiction")
    add("user", "不喜欢", "喝冰美式", "preference", 0.65, 0.6,
        tags=["咖啡", "饮品"], session="demo:contradiction")

    # ═══════════════════════════════════════════
    #  事实（fact）— 结构化信息
    # ═══════════════════════════════════════════

    add("user", "住在", "深圳南山", "fact", 0.95, 0.7,
        tags=["位置", "深圳"], session="demo:fact:location")
    add("user", "职业", "程序员", "fact", 0.90, 0.6,
        tags=["工作", "编程"], session="demo:fact:work")
    add("user", "会", "Python", "fact", 0.85, 0.5,
        tags=["编程", "技能"], session="demo:fact:skill")
    add("user", "会用", "React", "fact", 0.70, 0.4,
        tags=["编程", "前端"], session="demo:fact:skill")

    # ═══════════════════════════════════════════
    #  目标（goal）— 未来计划
    # ═══════════════════════════════════════════

    add("user", "目标", "三个月学会 DevOps", "goal", 0.80, 0.8,
        tags=["学习", "职业发展"], session="demo:goal")
    add("user", "目标", "年底前瘦 10 斤", "goal", 0.60, 0.7,
        tags=["健康", "运动"], session="demo:goal")

    # ═══════════════════════════════════════════
    #  决策（decision）— 历史选择
    # ═══════════════════════════════════════════

    add("user", "选择了", "MacBook Pro 作为主力开发机", "decision", 0.90, 0.6,
        tags=["工具", "硬件"], session="demo:decision")
    add("user", "选择了", "VS Code 作为编辑器", "decision", 0.85, 0.5,
        tags=["工具", "软件"], session="demo:decision")

    # ═══════════════════════════════════════════
    #  观察（observation）— 系统自动提取
    # ═══════════════════════════════════════════

    add("用户", "说了", "今天深圳好热，想喝冰的", "observation", 0.50, 0.3,
        tags=["天气", "深圳"], session="demo:obs")
    add("用户", "说了", "项目 Q3 要上线，时间很赶", "observation", 0.50, 0.4,
        tags=["工作", "项目"], session="demo:obs")

    # ═══════════════════════════════════════════
    #  行为记录（action）— Agent 执行记录
    # ═══════════════════════════════════════════

    add("小明", "创建了文件", "/home/user/project/README.md", "action", 0.95, 0.4,
        tags=["文件", "文档"], session="demo:action")
    add("小明", "搜索了", "如何优化 PostgreSQL 查询性能", "action", 0.90, 0.3,
        tags=["搜索", "数据库"], session="demo:action")

    # ═══════════════════════════════════════════
    #  批量写入 DB
    # ═══════════════════════════════════════════

    saved = 0
    for f in facts:
        try:
            db.save_fact(f)
            saved += 1
        except Exception as e:
            logger.warning("跳过重复事实: %s %s %s (%s)", f.subject, f.predicate, f.object, e)

    logger.info("✅ Agent '%s': 写入 %d/%d 条事实", agent_id, saved, len(facts))

    # 打印摘要
    by_type = {}
    for f in facts:
        by_type.setdefault(f.fact_type, 0)
        by_type[f.fact_type] += 1
    logger.info("📊 类型分布: %s", dict(by_type))

    # ═══════════════════════════════════════════
    #  ⭐ 矛盾检测演示：删除两条矛盾事实，用 cogni.remember() 重走管道
    #  ═══════════════════════════════════════════
    #  db.save_fact 直接写库跳过矛盾检测，
    #  这里先删掉矛盾对，再用 CogniMem 引擎过一遍完整管道。
    from cognimem.core.brain import CogniMem
    brain = CogniMem(db_adapter=db, use_llm=False)

    # 找到并删除已保存的矛盾对
    existing = db.get_agent_facts(agent_id)
    for f in existing:
        if f.object == "喝冰美式" and f.predicate in ("喜欢", "不喜欢"):
            db.delete_fact(f.fact_id)
            logger.info("🗑️ 删除矛盾事实: %s %s %s", f.subject, f.predicate, f.object)

    # 通过 cogni 管道重新添加 → 触发矛盾检测
    contradictions_expected = 0
    for conflict_text in [
        "我喜欢喝冰美式",
        "我不喜欢喝冰美式",
    ]:
        try:
            result = brain.remember(conflict_text, source="demo:contradiction", agent_id=agent_id)
            if result.get("contradictions_detected"):
                contradictions_expected += result["contradictions_detected"]
                logger.info("⚡ 矛盾检测触发: %s → %d 对", conflict_text, result["contradictions_detected"])
        except Exception as e:
            logger.warning("矛盾检测跳过: %s", e)

    # 召回验证
    result = brain.recall("冰美式", agent_id, top_k=5)
    logger.info("🔍 召回验证「冰美式」: 找到 %d 条事实", result["count"])
    for f in result["facts"]:
        logger.info("   → %s %s %s (conf=%.2f, type=%s)",
                     f.subject, f.predicate, f.object, f.confidence, f.fact_type)

    contradictions = brain.fact_network.get_contradictions(agent_id)
    if contradictions:
        logger.info("⚡ 矛盾检测: %d 对矛盾已标记", len(contradictions))
        for c in contradictions:
            logger.info("   → %s  vs  %s  [%s]", c.fact_a_id[:8], c.fact_b_id[:8], c.contradiction_type)
    else:
        logger.info("ℹ️ 未检测到矛盾（需要 consolidate 后处理）")

    # 打印 dashboard 数据验证
    stats = brain.get_stats(agent_id)
    logger.info("📈 统计: %d 条事实, %d 个核心信念, %d 条矛盾",
                 stats["total_facts"], stats["core_beliefs"], stats["contradictions"])


def main():
    args = parse_args()

    db = get_db(args.db)

    if args.clear:
        clear_agent(db, args.agent)

    seed_agent(db, args.agent)

    logger.info("=" * 50)
    logger.info("✅ seed_demo 完成！")
    logger.info("   启动服务后访问 http://localhost:9999/dashboard 查看数据")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
