# CogniMem 改进报告：对照阿里官方需求的差距分析与改进方案

> **日期：** 2026-07-03 | **截止：** 2026-07-10 05:00 GMT+8
> **赛道：** MemoryAgent（持久化认知存储）
> **官方要求：** Build an Agent with persistent memory that autonomously accumulates experience, remembers user preferences, and makes increasingly accurate decisions across multi-turn, cross-session interactions.

---

## 目录

1. [官方需求逐项对照](#1-官方需求逐项对照)
2. [竞品分析：赢家做了什么](#2-竞品分析赢家做了什么)
3. [差距总览](#3-差距总览)
4. [改进方案详述](#4-改进方案详述)
5. [改进工作量估算](#5-改进工作量估算)
6. [推荐执行路线](#6-推荐执行路线)

---

## 1. 官方需求逐项对照

### 1.1 赛道描述要求

官方原文：
> *Build an Agent with persistent memory that autonomously accumulates experience, remembers user preferences, and makes increasingly accurate decisions across multi-turn, cross-session interactions.*

| 关键词 | 官方要求 | 我们现状 | 差距 |
|--------|---------|---------|------|
| **Agent** | 是一个 Agent（智能体），不是 API/存储系统 | ❌ 只是一个存储后端 + 聊天界面 | 🔴 致命 |
| **persistent memory** | 持久化记忆 | ⚠️ 内存模式，重启丢数据 | 🟡 中 |
| **autonomously accumulates experience** | 自动积累经验 | ✅ 混合提取 + 主动学习 | ✅ |
| **remembers user preferences** | 记住用户偏好 | ✅ 偏好类三元组 | ✅ |
| **cross-session** | 跨会话 | ✅ agent_id 隔离 | ✅ |
| **increasingly accurate decisions** | 决策越来越准 | ⚠️ 置信度更新有，但 Agent 没接 | 🟡 中 |

### 1.2 官方强调的 focus 方向

官方原文：
> *Participants should focus on: efficient memory storage and retrieval, timely forgetting of outdated information, and recalling critical memories within limited context windows.*

| 方向 | 我们现状 | 评分 |
|------|---------|------|
| **efficient memory storage and retrieval**（高效存储检索） | 三级路由 + 多级缓存 + 规则 0 tok 提取 | ✅ 强项 |
| **timely forgetting**（及时遗忘） | 艾宾浩斯衰减 + 5min 自动整合 | ✅ 强项 |
| **recalling critical memories within limited context windows**（有限上下文内召回关键记忆） | 上下文感知排序 + token 预算选择 | ✅ 有 |
| **Agent 自主决策** | ❌ 没有 | 🔴 缺失 |

### 1.3 提交要求

| # | 要求 | 现状 |
|---|------|------|
| 1 | 公开 GitHub 仓库 + 开源协议 | ⚠️ 仓库有但代码不是最新，缺 LICENSE |
| 2 | 阿里云部署证明 | ✅ 已部署到 ECS |
| 3 | 架构图 | ✅ 已有 |
| 4 | 演示视频 | ❌ 未录 |
| 5 | 项目描述文本 | ✅ 已有 |
| 6 | 使用 Qwen 模型 | ✅ 已用 Qwen 3.7+ |

---

## 2. 竞品分析：赢家做了什么

### 2.1 ERINYS Care Memory（记忆治理方向赢家）

**核心创新：** Deterministic Memory Governance（确定性记忆治理）

```
召回的所有记忆 → 4 态策略过滤 → 只有通过/冲突的进入 prompt
                  ↑
           不是 LLM 判断，是硬规则
```

**4 态策略：**
| 状态 | 含义 | 处理 |
|------|------|------|
| ✅ SELECTED | 安全、相关 | 注入 prompt |
| ⚠️ CONFLICTED | 有矛盾的记忆 | 注入 + 标注矛盾 |
| ⛔ DEMOTED | 低置信度/过时 | 不注入，但可查 |
| 🚫 BLOCKED | PII/敏感信息 | 彻底拦截 |

**我们可以学的：** 
- CogniMem 已经有矛盾检测，但在存储时做，不是生成前
- 加一道生成前记忆治理规则，把"召回→全塞"改成"召回→过滤→注入"

### 2.2 Universal Agent OS（全功能 Agent 方向）

**核心创新：** 4 大记忆支柱 + 混合检索 + 116 个测试

| 支柱 | 说明 |
|------|------|
| 🧠 State Memory | 当前任务状态 |
| 👤 Persona Memory | 用户画像 |
| 💣 Minefield Memory | 踩坑记录（之前犯过的错）|
| 🔮 Code Soul Memory | 代码风格/偏好 |

**混合检索：**
```
BM25 关键词匹配 → 得分 × 0.4
语义向量匹配   → 得分 × 0.3
新鲜度        → 得分 × 0.2
重要性        → 得分 × 0.1
───────────────
加权总分排序 → 取 top_k
```

**我们可以学的：**
- CogniMem 的 `recall.py` 已经有类似的评分公式（置信度×50%+重要性×20%+新鲜度×20%+相关度×10%），但少了 BM25/keyword 那一路
- 多 Agent 交互 demo

### 2.3 RuleMemory（冲突解决方向）

**核心创新：** 冲突 supersession（新事实覆盖旧事实时，旧事实标记为 superseded 而非删除）

**我们可以学的：**
- CogniMem 的 `fact_network.py` 已经有 `_merge_facts()` 和版本链，但 supersession 的 UI 展示不够清晰

### 2.4 Mimir MemoryAgent（全栈工程方向）

**核心创新：** Rust 后端 + 27 个 MCP 工具 + AES-256 加密

**我们可以学的：**
- 工程完整度——Docker Compose、加密、完善的 README
- 但我们不需要 Rust，Python 够用

---

## 3. 差距总览

### 按严重度排列

| # | 差距 | 严重度 | 工作量 | 说明 | 状态 |
|---|------|--------|--------|------|------|
| 1 | **不是 Agent** — 只会一问一答 | 🔴 P0 | 中 | 目标驱动循环 + 10工具 | ✅ **v0.6** |
| 2 | **无自主规划** — 不会拆任务 | 🔴 P0 | 小 | 预规划 LLM 生成步骤 | ✅ **v0.6** |
| 3 | **无自动纠错** — 报错只会返回 | 🔴 P0 | 小 | FixExecutor 4种修复 | ✅ **v0.6** |
| 4 | **记忆无过滤** — 全存全不存 | 🟡 P1 | 小 | MemoryManager 过滤 | ✅ **v0.6** |
| 5 | **搜索慢** — DuckDuckGo 被墙 | 🟡 P1 | 小 | HTTP代理 + 缓存 | ✅ **v0.6** |
| 6 | **生成前记忆治理** | 🔴 P0 | 中 | MemoryGovernor 4态 | ✅ **v0.5** |
| 7 | **BM25 关键词检索** | 🟡 P1 | 小 | recall.py 增加 | ✅ **v0.5** |
| 8 | **Docker Compose** | 🟡 P1 | 小 | docker-compose.yml | ✅ **v0.5** |
| 9 | **内存模式重启丢数据** | 🟢 P2 | 极小 | PostgreSQL 持久化 | ✅ **v0.5** |
| 10 | **GitHub 仓库同步** | 🔴 P0 | 中 | 暂未同步 | 🔲 待做 |
| 11 | **LICENSE 文件** | 🟢 P2 | 极小 | 未加 | 🔲 待做 |
| 12 | **演示视频** | 🔴 P0 | ~2h | 未录 | 🔲 待做 |
| 13 | **Devpost 提交** | 🔴 P0 | ~1h | 未填 | 🔲 待做 |

---

## 4. 改进方案详述

### 🔴 P0-1：把 Agent 引擎接上 `/chat` ✅已修

**之前：** `/chat` endpoint 调 `llm.answer_with_memories()`，直接调 LLM 回答，不涉及工具调用。

**现在：** `/chat` endpoint 调 `agent.chat()`，走 Think→Act→Observe 循环。Agent 可以自主决定调工具（搜索/文件/Shell/记忆操作）。

**效果验证：**
- `你好` → 回复正常，0 工具调用 ✅
- `帮我搜一下今天的天气` → Agent 调用了 web_search 工具，2 轮思考后追问城市 ✅

**改动文件：**
| 文件 | 改动 |
|------|------|
| `llm_client.py` | 新增 `chat_completion()` 方法（支持 tools 参数）|
| `main.py` | 启动时创建 Agent 实例，`/chat` 改用 agent.chat() |
| `chat.html` | 显示工具调用序列和轮次信息 |

```
agent/__init__.py  ← Agent 类 + Agent.chat() 方法
agent/tools.py     ← 10 个工具 + ToolRegistry
agent/modules.py   ← 模块系统
```

**改动量：**

| 文件 | 改动 | 行数 |
|------|------|------|
| `main.py` | `/chat` endpoint：创建 Agent 实例替代直接 LLM 调用 | ~30 行 |
| `main.py` | 启动时创建 ToolRegistry + 注册工具 | ~10 行 |
| `chat.html` | 显示 tool_calls + tool_sequence | ~50 行 |
| `llm_client.py` | 加 `chat_completion()` 方法（支持 tools 参数） | ~20 行 |
| **总计** | | **~110 行** |

**Agent 对话 UI 改动（chat.html）：**

```
现在：                         改后：
用户 → LLM → 回复              用户 → Agent → 思考 → 调工具 → 观察结果 → 回复
                                     ↓
                                 显示思考过程
                                 🧠 正在分析...
                                 🛠️ 调用了 web_search("...")
                                 👀 找到了3条结果
                                 💬 最终回复
```

**Agent 记忆治理流程（内置在 Agent.chat() 中）：**

```
Agent.chat() 内部:
  1. 从 CogniMem 召回记忆
  2. 🔥 新增：记忆治理过滤（筛选安全/相关的）
  3. 注入筛选后的记忆到 system prompt
  4. 调 LLM（可能触发工具调用）
  5. 执行工具 → 观察结果
  6. 循环直到 LLM 输出最终文本
  7. 存储会话摘要到 CogniMem
```

---

### 🔴 P0-2：生成前记忆治理

**现状：** 召回的所有记忆全部注入 prompt，没有筛选。

**改后：** 在召回之后、注入 prompt 之前，加一道治理过滤。

**设计：**
```python
class MemoryGovernor:
    """
    记忆治理过滤器 — 在召回之后、注入之前运行。

    4 态策略（参考 ERINYS）:
      SELECTED  → 安全且相关，注入 prompt
      CONFLICTED → 有矛盾，注入 + 标注矛盾
      DEMOTED  → 低置信度/过时，不注入但可查
      BLOCKED  → PII/敏感信息，彻底拦截

    规则都是确定性的（不是 LLM 判断），保证可审计。
    """
```

**过滤规则：**
| 条件 | 判定 | 处理 |
|------|------|------|
| 置信度 ≥ 0.6 | ✅ SELECTED | 注入 prompt |
| 置信度 0.3-0.6 | ⚠️ DEMOTED | 不注入（但保留在库里）|
| 置信度 < 0.3 | 🚫 BLOCKED | 不注入，建议清理 |
| 有矛盾标记 | ⚠️ CONFLICTED | 注入 + 标注矛盾 |
| 含"密码/PII/身份证"等关键词 | 🚫 BLOCKED | 不注入 |

**改动量：** 新建 `cognimem/core/governance.py`，约 80 行。

---

### 🟡 P1-1：BM25 关键词检索

**现状：** `recall.py` 只有 L0 缓存/L1 精确/L2 语义/L3 向量，缺少 BM25 关键词模糊匹配。

**改后：** L2 语义扩展中增加 BM25 评分分支。

```python
# 在 _l2_semantic_expand() 中增加：
# 1. 对 query 分词
# 2. 对每个缓存事实计算 BM25 得分
# 3. 得分 > 阈值 → 加入结果集
```

**改动量：** `recall.py` 加一个 `_bm25_score()` 方法，约 30 行（Python 有 `rank_bm25` 库，或自己实现简单 BM25）。

---

### 🟡 P1-2：Docker Compose

**现状：** 部署靠手动 scp + 手动启动。

**改后：** `docker-compose.yml` 一键启动：

```yaml
services:
  cognimem:
    build: .
    ports: ["8001:8001"]
    environment:
      - COGNIMEM_DB=postgresql://postgres:password@db/cognimem
    depends_on: [db]

  ui:
    build: .
    ports: ["8000:8000"]
    environment:
      - QWEN_API_KEY=${QWEN_API_KEY}

  db:
    image: postgres:16
    environment:
      - POSTGRES_DB=cognimem
```

**改动量：** 新建 `docker-compose.yml` + `Dockerfile`，约 40 行。

---

### 🟡 P1-3：评测基准

**现状：** 没有量化指标证明"我的系统多好"。

**改后：** 写评测脚本，输出关键指标：

| 指标 | 说明 |
|------|------|
| 召回准确率 | 100 条测试查询中正确命中的比例 |
| 召回延迟 (P50/P95) | 毫秒 |
| 矛盾检测 F1 | 精准率 + 召回率 |
| Token 节省率 | 规则提取 vs 全 LLM 提取的 token 对比 |
| 衰减正确性 | 不同访问模式下的置信度变化是否符合艾宾浩斯曲线 |

**改动量：** 新建 `benchmark/` 目录，约 100 行。

---

## 5. 改进工作量估算

| 优先级 | 改进项 | 工作量 | 文件数 | 对得分的提升 | 状态 |
|--------|--------|--------|--------|------------|------|
| 🔴 P0-1 | Agent 引擎接线+目标驱动 | **~200 行** | 4 文件 | **极高** | ✅ **v0.6** |
| 🔴 P0-2 | 自主预规划 | **~60 行** | 1 文件 | **高** | ✅ **v0.6** |
| 🔴 P0-3 | 自动纠错修复 | **~100 行** | 2 文件 | **高** | ✅ **v0.6** |
| 🟡 P1-1 | 记忆自管理 | **~80 行** | 1 文件 | 中 | ✅ **v0.6** |
| 🟡 P1-2 | 搜索性能优化 | **~50 行** | 2 文件 | 中 | ✅ **v0.6** |
| 🟡 P1-3 | 模型切换（DeepSeek） | **~10 行** | 1 文件 | 中 | ✅ **v0.6** |
| 🟢 P2 | LICENSE + .gitignore | 2 行 | 2 文件 | 低 | 🔲 待做 |

**总计：~500 行代码，90% 已完成**

---

## 6. 推荐执行路线

### 实际执行（✅ 已完成 7/4）

```
7/4:
  ✅ Phase 1: 目标驱动执行循环     ← Agent 能连续调工具直到完成
  ✅ Phase 2: 自主预规划           ← Agent 先拆步骤再执行
  ✅ Phase 3: 自动纠错重试         ← pip install / mkdir / 备份
  ✅ Phase 4: 记忆自管理           ← 废话过滤 + 重要自动存
  ✅ 搜索优化                     ← HTTP代理 + 缓存 + 超时优化
  ✅ 模型切换                     ← Qwen → DeepSeek
```

### 剩余待做

```
7/5:
  🔳 GitHub 同步 + LICENSE       (~30分钟)
  
7/6:
  🎥 录演示视频                  (~2小时)
  📝 Devpost 提交                (~1小时)

7/7~7/8:
  🎯 可选优化 + 最终检查
  ✍️ Blog 文章（额外奖金）
```

---

## 结论

**CogniMem 的核心引擎很强——存储、提取、矛盾、遗忘、抽象化都在前列。**

但**缺了最关键的一层：Agent**。赛道叫 MemoryAgent，评委想要的是"一个有记忆的智能体"，不是一个"带 UI 的数据库"。

好消息是 Agent 引擎代码已经写好，只是没接线。P0-1 是改动最小、收益最大的——~110 行代码就能把后端变成 Agent，把差距补上。
