# 🧠 CogniMem 认知记忆系统 — 完整技术文档

> **版本:** v0.13 | **更新:** 2026-07-12
> **赛道:** Qwen Cloud MemoryAgent | **许可证:** MIT
> **仓库:** [github.com/1421906110/qwen-memoryagent](https://github.com/1421906110/qwen-memoryagent)
> **演示:** [http://42.121.253.80:8000](http://42.121.253.80:8000)

---

## 📑 目录

- [一、项目概览](#一项目概览)
- [二、系统架构](#二系统架构)
- [三、快速开始](#三快速开始)
- [四、记忆系统](#四记忆系统)
- [五、v0.13 核心特性详解](#五v013-核心特性详解)
- [六、Agent 引擎](#六agent-引擎)
- [七、MCP Server](#七mcp-server)
- [八、仪表盘与知识图谱](#八仪表盘与知识图谱)
- [九、API 端点完整清单](#九api-端点完整清单)
- [十、数据库结构](#十数据库结构)
- [十一、配置说明](#十一配置说明)
- [十二、部署说明](#十二部署说明)
- [十三、技术栈](#十三技术栈)
- [十四、竞品对比优势](#十四竞品对比优势)
- [十五、测试与质量](#十五测试与质量)
- [十六、常见问题与排错](#十六常见问题与排错)

---

## 一、项目概览

### 一句话介绍

让 AI 拥有长期记忆——不是死记硬背对话原文，而是像人一样：**理解要点、归纳规律、发现矛盾、自然遗忘**。

### 四大设计原则

| # | 原则 | 核心体现 |
|:-:|:----|:---------|
| 1 | 🧠 **更智能** | 完整 Agent 引擎（12 工具 + 目标驱动 + 自主规划 + 自反思） |
| 2 | 💰 **更省 Token** | 规则提取覆盖 70%（0 token）, L0 缓存命中直接返回 |
| 3 | ⚡ **更省资源** | PostgreSQL + pgvector 一体, 无需外部缓存服务, 单机可跑 |
| 4 | 🚀 **算法更创新** | SPO 三元组结构推理、矛盾驱动学习、Weibull 科学遗忘 |

### 核心能力矩阵

| 能力 | 级别 | 说明 |
|:----|:----:|:-----|
| SPO 结构化记忆 | ✅ **独家** | (主体-谓词-客体) 三元组, 可推理知识网络 |
| 零成本提取 | ✅ | 规则 70% + 缓存 20%, LLM 调用 < 10% |
| 三层矛盾检测 | ✅ | 否定词(0tok) → 语义向量(0tok) → LLM 裁决 |
| 碎片自动归纳 | ✅ | 同类事实自动抽象为核心信念 |
| 科学遗忘 | ✅ | Ebbinghaus + Weibull 双曲线 |
| 四层召回流水线 | ✅ | L0 缓存 → L1 拒绝 → L2 向量 → L3 图谱漫游 |
| 零LLM路由 | ✅ | 无需 LLM 精确分类查询意图 |
| 语义缓存 | ✅ | 相似查询命中缓存, 毫秒级返回 |
| 记忆进化 | ✅ | 相关事实自动链接 |
| STM 缓冲区 | ✅ | FIFO 淘汰 + 自动 Consolidate |
| 主动检索 | ✅ | 按实体类型精确召回 |
| Agent 引擎 | ✅ | 目标驱动 + 预规划 + 自纠错 + 经验学习 |
| 跨 Agent 总线 | ✅ | 一次查询多个 Agent 记忆 |
| 知识图谱 | ✅ | 力导向图, 8 种语义类型 |
| MCP 协议 | ✅ | 13 个 MCP 工具 |
| 仪表盘 | ✅ | 健康检测 / 指标卡 / 趋势图 |
| 凭证安全存储 | ✅ | 自动掩码, 普通召回排除 |
| 审计日志 | ✅ | 全操作可追溯 |
| 移动端适配 | ✅ | 响应式 CSS + 侧栏切换 |

---

## 二、系统架构

### 架构总览

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         用户浏览器 / MCP 客户端                             │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
                      ┌──────▼──────┐
                      │  FastAPI    │  HTTP / SSE / MCP
                      │  (uvicorn)  │
                      └──────┬──────┘
                             │
            ┌────────────────┼─────────────────┐
            │                │                  │
      ┌─────▼─────┐   ┌─────▼──────┐   ┌──────▼──────┐
      │ 聊天界面    │   │ 仪表盘      │   │ 知识图谱     │
      │ (chat.html)│   │(dashboard) │   │ (graph)     │
      └─────┬─────┘   └─────┬──────┘   └──────┬──────┘
            │                │                  │
            └────────────────┼──────────────────┘
                             │
                      ┌──────▼──────────────────────┐
                      │   Agent 引擎                  │
                      │   ┌────────────────────────┐ │
                      │   │ 目标驱动循环 + 预规划    │ │
                      │   │ 12 内置工具             │ │
                      │   │ 自反思 + 经验学习       │ │
                      │   │ Governance 4 态过滤     │ │
                      │   │ 自动纠错 + 降级         │ │
                      │   └────────────────────────┘ │
                      └──────┬───────────────────────┘
                             │
                      ┌──────▼──────────────────────┐
                      │   CogniMem 记忆引擎            │
                      │   ┌────────────────────────┐ │
                      │   │ 三层提取 (规则/缓存/LLM) │ │
                      │   │ 三层矛盾检测             │ │
                      │   │ 碎片归纳抽象             │ │
                      │   │ Weibull 科学遗忘         │ │
                      │   │ 四层召回流水线           │ │
                      │   │ 零LLM路由 / STM缓冲区    │ │
                      │   │ 语义缓存 / 记忆进化      │ │
                      │   │ 主动检索                 │ │
                      │   └────────────────────────┘ │
                      └──────┬───────────────────────┘
                             │
                      ┌──────▼──────────────────────┐
                      │   PostgreSQL + pgvector        │
                      │   ┌────────────────────────┐ │
                      │   │ facts (事实网络)         │ │
                      │   │ versions (版本链)       │ │
                      │   │ contradictions (矛盾)   │ │
                      │   │ agents (智能体)         │ │
                      │   │ embedding (向量索引)    │ │
                      │   │ lessons (经验教训)      │ │
                      │   └────────────────────────┘ │
                      └──────┬───────────────────────┘
                             │
                      ┌──────▼──────┐
                      │  Qwen LLM   │  DashScope API
                      │ (3.7-Plus)  │
                      └─────────────┘
```

### 模块分层

| 层 | 模块 | 路径 | 行数 | 职责 |
|:--|:----|:----|:---:|:-----|
| **Web** | 主入口 | `memory_agent/main.py` | 1743 | HTTP 端点 + 页面路由 + 健康检测 + 仪表盘 API |
| | 引擎入口 | `cognimem/main.py` | 336 | 引擎独立 HTTP 端点 |
| | MCP Server | `cognimem/mcp_server.py` | 432 | MCP 协议接口 (13 工具) |
| **Agent** | Agent 核心 | `memory_agent/agent/__init__.py` | ~1000 | 目标驱动循环 + 预规划 + 自反思 + 纠错 |
| | 工具集 | `memory_agent/agent/tools.py` | ~500 | 12 内置工具实现 |
| | Governance | `memory_agent/agent/governance.py` | — | 4 态记忆治理过滤 |
| | 反射器 | `memory_agent/agent/reflector.py` | — | 任务后自我反思, 存教训 |
| | 验证器 | `memory_agent/agent/validator.py` | — | 工具调用结果验证 |
| | 记忆管理器 | `memory_agent/agent/memory_manager.py` | — | 重要记忆智能存储 |
| | 模块系统 | `memory_agent/agent/modules.py` | — | 高频任务 → 模块沉淀 |
| | 目标系统 | `memory_agent/agent/goal.py` | — | GoalContext 规划与追踪 |
| **核心引擎** | Brain | `cognimem/core/brain.py` | 612 | 记忆引擎总入口 (remember/recall/ask/consolidate) |
| | Fact Network | `cognimem/core/fact_network.py` | 1470 | 事实网络 + 矛盾检测 + 衰减 + 版本链 |
| | Recall | `cognimem/core/recall.py` | 674 | 4 层召回流水线 + 路由统计 |
| | DB | `cognimem/core/db.py` | 725 | PostgreSQL + pgvector 适配器, 连接池管理 |
| | 规则提取器 | `cognimem/core/extractor.py` | 163 | 正则模板提取三元组 (0 token) |
| | LLM 提取器 | `cognimem/core/llm_extractor.py` | — | Qwen LLM 精提取, JSON 结构化输出 |
| | 模型 | `cognimem/core/models.py` | — | FactTriple / Contradiction / EvidenceItem |
| **服务** | LLM Client | `memory_agent/services/llm_client.py` | — | Qwen/DeepSeek 兼容客户端, 流式/非流式 |
| | Memory Service | `memory_agent/services/memory_service.py` | — | 记忆服务封装 |
| | CogniMem Client | `memory_agent/services/cognimem_client.py` | — | Agent ↔ CogniMem 桥接 |

### 数据流

**记忆存入流程：**
```
用户输入 → 规则提取(0tok, <1ms) → 置信度 ≥ 0.6? → 是 → 矛盾检测 → 存入事实网络
                                      ↓ 否
                                文本太简单? ──是──→ 跳过 LLM(节省 token)
                                      ↓ 否
                                LLM 精提取(50-150 tok) → 矛盾检测 → 存入事实网络
                                                              ↓
                                                        有矛盾? → 记录矛盾对 → 待用户确认
```

**记忆召回流程：**
```
用户提问 → L0 语义缓存命中? → 直接返回缓存 (毫秒级)
            ↓ 否
        零LLM路由 → 分类查询意图 (factual/exploratory/navigation)
            ↓
         L1 Deny 过滤 (排除矛盾/敏感)
            ↓
         L2 Embedding 向量召回 (pgvector, <25ms)
            ↓
         L3 图谱漫游 (关联事实扩展)
            ↓
         综合评分 (置信度 × 相关性 × 新鲜度) → Top-K 注入上下文
```

---

## 三、快速开始

### 3.1 环境要求

```bash
# Python 3.10+
# PostgreSQL 14+
# 安装 pgvector
```

### 3.2 配置

```bash
# .env 文件
QWEN_API_KEY=sk-xxx                          # Qwen DashScope
DEEPSEEK_API_KEY=sk-xxx                      # 或 DeepSeek
QWEN_MODEL=qwen-plus                         # 模型名
COGNIMEM_DB=postgresql://user:pass@localhost/cognimem
COGNIMEM_LLM=true                            # 启用 LLM 提取
MCP_PORT=8100                                # MCP SSE 端口
```

### 3.3 启动

```bash
# 安装依赖
pip install -r requirements.txt

# 启动主服务
cd src
python -m uvicorn memory_agent.main:app --host 0.0.0.0 --port 8000

# 或启动 MCP Server（标准 I/O 模式）
python -m cognimem.mcp_server

# 或 MCP SSE 模式
python -m cognimem.mcp_server --sse --port 8100
```

### 3.4 验证

```bash
curl http://localhost:8000/health
# → {"score": 100, "level": "healthy"}
```

---

## 四、记忆系统

### 4.1 SPO 三元组结构

区别于纯文本存储, CogniMem 将信息拆解为 (主体, 谓词, 客体) 三元组：

| 用户说 | 提取结果 | 类型 |
|:-------|:---------|:----:|
| "我喜欢喝冰美式" | `(用户, 喜欢, 喝冰美式)` | preference |
| "我不喜欢喝热美式" | `(用户, 不喜欢, 喝热美式)` | preference |
| "我住在北京" | `(用户, 住在, 北京)` | fact |
| "我是一名程序员" | `(用户, 职业, 程序员)` | fact |
| "我打算三个月学会 React" | `(用户, 目标, 三个月学会 React)` | goal |
| "我决定用 FastAPI 做后端" | `(用户, 决策, 用 FastAPI 做后端)` | decision |

三元组结构支持精确推理：
> 搜"住在北京的程序员" → `(?, 住在, 北京) ∩ (?, 职业, 程序员)` → 精确命中

### 4.2 三层提取策略

| 层 | 方法 | Token | 覆盖 | 延迟 |
|:--|:----|:----:|:----:|:---:|
| **L1** | 正则规则模板 (60+ 句式) | **0** | ~70% | < 1ms |
| **L2** | 缓存复用 (同句式/同用户命中) | **0** | ~20% | < 1ms |
| **L3** | Qwen LLM 精提取 (JSON 结构化) | 50-150 tok | < 10% | ~500ms |

**模板示例：**
```
"我喜欢 X" → (用户, 喜欢, X)
"我不喜欢 X" → (用户, 不喜欢, X)
"我叫 X" → (用户, 名字, X)
"我想去 X" → (用户, 想去, X)
```

### 4.3 三层矛盾检测

| 层 | 方法 | Token | 原理 |
|:--|:----|:----:|:-----|
| **L1** | 否定词 Jaccard 匹配 | **0** | "喜欢" vs "不喜欢" → 否定前缀对比 → 矛盾 |
| **L2** | Embedding 余弦相似度 | **0** | "咖啡" vs "火锅" → 向量距离远 → 不同偏好 |
| **L3** | Qwen LLM 语义裁决 | 少量 | 前两层无法确定时做最终判断 |

**智能容错：**
- 中性谓词自动跳过：`(用户, 请求, 读文件)` vs `(用户, 请求, 搜网络)` → 不误报
- 跨类别偏好不冲突："喜欢咖啡" vs "喜欢火锅" → 独立维度
- 矛盾驱动学习：检测到矛盾后主动提问"你之前说喜欢冰美式，现在说不喜欢，是变了吗？"

### 4.4 碎片自动归纳

零散同类信息自动汇总为高层概念：

> `(我, 喜欢, 冰美式)` + `(我, 喜欢, 热美式)` + `(我, 喜欢, 冷萃)`
> → 归纳为 **"用户对咖啡有偏好"**（core_belief, type=abstraction）

归纳自动触发条件：系统空闲 5 秒后自动 consolidation，无需人工干预。

### 4.5 科学遗忘

**Ebbinghaus 曲线（基础版）：**
```
half_life = 24h × (1 + √access_count)
置信度 = 初始值 × 2^(-经过时间 / half_life)
```

**Weibull 衰减（v0.13 增强版，k=1.5）：**

| 天数 | 衰减量 | 说明 |
|:---:|:-----:|:-----|
| 0 | 0.0000 | 记忆完好 |
| 7 | 0.0752 | 缓慢开始 |
| 30 | 0.5000 | 半衰点 |
| 90 | 0.9727 | 几乎遗忘 |
| 180 | ≈1.0000 | 完全遗忘 |

**Ebbinghaus vs Weibull：**
- Ebbinghaus：均匀衰减, 前期忘太快, 后期忘太慢
- Weibull：**前期慢（刚记住牢固）+ 后期快（过时快速淘汰）**, 更接近人类遗忘

### 4.6 四层召回流水线

```
用户提问
  ↓
L0 语义缓存：相似查询命中缓存? → 直接返回 (0 token, <1ms)
  ↓ (未命中)
L1 Deny 过滤：排除与问题矛盾/敏感的信息
  ↓ (通过)
L2 Embedding 召回：pgvector 余弦相似度, Top-N
  ↓ (召回结果)
L3 图谱漫步：从召回结果沿 SPO 关系扩展关联事实
  ↓
综合评分：置信度 × 相关性 × 新鲜度 → Top-K 注入上下文
```

### 4.7 噪音自动过滤

| 场景 | 处理 |
|:----|:-----|
| `curl / wget / ping / traceroute` 命令 | 自动跳过, 不收入记忆库 |
| `web_fetch` 结果 | 中间数据, 无记忆价值 → 跳过 |
| 同一对话多次 `web_search` | 只存第一次结果 |
| 凭证类型事实 | `recall()` 中自动排除, 不泄露 |

### 4.8 版本链

每条记忆的完整生命周期可追溯：

```
created(0.6) → confirmed(0.7) → challenged(0.5) → confirmed(0.8) → decayed(0.3) → pruned
```

每次变化记录：时间、原因、新旧置信度。通过 `/versions/{id}` 端点可查。

---

## 五、v0.13 核心特性详解

### 5.1 P0-1 零LLM路由

不需要 LLM 调用就能精确分类查询意图：

| 输入 | 分类 | 说明 |
|:----|:----|:-----|
| "冰美式" | `factual` | 事实查询 |
| "为什么..." | `exploratory` | 探索性 |
| "你好" | `navigation` | 导航/问候 |
| "如何使用" | `exploratory` | 操作类 |
| "" (空) | `navigation` | 默认 |

**意义：** LLM 零开销，毫秒级响应，能扛高并发。只有需要深度语义时才调 LLM。

### 5.2 P0-2 记忆进化（相关事实自动链接）

存入"小七是老大"和"小七是项目经理"后，系统自动识别共享同一主体(`小七`)，建立 `connected_facts` 链接。

- 自动关联，不影响矛盾检测
- 链接用于推理和上下文构建（L3 图谱漫游时持续扩大相关记忆）
- 召回时携带关联事实，提供更丰富上下文

### 5.3 P0-3 语义缓存

| 缓存已有 | 新查询 | 相似度 | 命中? |
|:--------|:------|:-----:|:----:|
| "冰美式" | "冰美式咖啡" | > 0.5 | ✅ |
| — | 空字符串 | = 0.0 | ✅ |
| 完全相同 | 完全相同 | = 1.0 | ✅ |

缓存命中直接返回结果，跳过 LLM 调用，响应时间从秒级降至毫秒级。

### 5.4 P0-4 STM 缓冲区（短期记忆）

```
存入: 5 条 → STM count = 5
FIFO: 35 条 → 自动淘汰至 ≤ 30
Flush: 手动 → 归零
Consolidate: 自动合并 STM 到长期记忆
```

- 隔离高频短期操作，减少碎片化存储
- consolidate 时批量处理到长期记忆

### 5.5 P1-1 主动检索（按实体类型精确召回）

| 输入 | 提取类型 |
|:----|:--------:|
| "小七喜欢喝什么咖啡" | `preference` |
| "项目截止日期是什么时候" | `fact` |
| "打算去日本旅游" | `goal` |
| "会做数据分析" | `skill` |

### 5.6 P1-2 Weibull 衰减

见 §4.5 科学遗忘章节。

### 5.7 P1-3 知识库（凭证安全存储）

```python
# 存储（自动掩码）
remember_credential('GitHub', 'token_abc')
# → 存储: 'token_abc'
# → 展示: 'tok***abc'

# 召回
recall_credential('GitHub')
# → credential: 'token_abc'
# → safe_display: 'tok***abc'

# 列表（全部掩码，不泄露原文）
list_credentials()
# → [{'service': 'GitHub', 'safe_display': 'tok***abc'}]
```

**安全设计：** 凭证类型事实在 `recall()` 中自动排除，普通对话不会泄露凭证。

### 5.8 嵌入算法（纯 Python 哈希 v2.0）

无需外部模型，不依赖 Embedding API：

| 特性 | 说明 |
|:----|:-----|
| 维度 | 384 维 |
| n-gram | 1~5 字符多粒度 |
| 位置加权 | 靠前 n-gram 权重更高 |
| 多哈希分布 | 3 个种子减少碰撞 |
| 中文感知 | 识别常见中文双字词 |
| IDF 平滑 | 高频 n-gram 自动降权 |
| L2 归一化 | 向量归一化到单位长度 |

**对比 Qwen text-embedding-v3：** 准确度略低 (相关对 0.24+ vs 0.35+)，但零成本、零延迟、可离线。

---

## 六、Agent 引擎

### 6.1 目标驱动循环

```
用户消息
  ↓
Step 1: 召回记忆 + Governance 过滤
  ↓
Step 2: 创建 GoalContext, 拆解任务
  ↓
Step 3: 预规划 → 生成步骤列表
  ↓
Step 4: LOOP (最多 30 次迭代)
  ├─ LLM 思考 → 调用工具 → 验证结果
  ├─ LLM 思考 → 直接回复文本
  └─ ← 检查目标完成? → 未完成继续
  ↓ 完成
Step 5: 回复用户
  ↓
Step 6: 自反思 → 记录经验教训
  ↓
Step 7: 智能存储重要记忆
```

### 6.2 12 个内置工具

| 工具 | 用途 | 类别 |
|:----|:-----|:----:|
| `read_file` | 读取本地文件 | 📁 文件 |
| `write_file` | 写入文件 | 📁 文件 |
| `edit_file` | 编辑文件（指定行替换）| 📁 文件 |
| `list_dir` | 列出目录内容 | 📁 文件 |
| `shell` | 执行 Shell 命令 | ⚙️ 命令 |
| `web_fetch` | 抓取网页内容 | 🌐 网络 |
| `web_search` | 联网搜索 | 🌐 网络 |
| `memory_recall` | 查询记忆 | 🧠 记忆 |
| `memory_remember` | 存入记忆 | 🧠 记忆 |
| `memory_status` | 查看记忆统计 | 🧠 记忆 |
| `memory_diagnose` | 诊断记忆健康 | 🧠 记忆 |
| `memory_forget` | 删除指定记忆 | 🧠 记忆 |

### 6.3 自主规划 + 预规划

Agent 收到请求后自动拆解步骤：

```
用户: "帮我查一下 Python 异步编程的资料，整理成笔记保存到桌面"

助手自动规划:
  1. web_search("Python 异步编程教程 2026")
  2. web_fetch(最有价值的链接)
  3. 整理关键信息为笔记
  4. write_file → 保存到桌面
```

### 6.4 自反思 + 经验学习

每次任务完成后自动反思：
- 工具调用哪些成功/失败？
- 失败原因是什么？如何改进？
- 记录经验到 `lessons` 表
- 下次同类任务自动召回以往教训，避免重蹈覆辙

### 6.5 自动纠错

| 场景 | 纠错行为 |
|:----|:---------|
| 搜索无结果 | 自动换关键词重试 |
| 工具返回错误 | 分析错误 → 调整参数重试 |
| LLM 返回空 | 降级到 Agent 路径 |
| Agent 也返回空 | 带上下文的友好 fallback 回复 |
| 文件写入失败 | 检查路径权限 → 尝试备选位置 |

### 6.6 Governance 治理层

**4 态记忆治理（受 ERINYS 启发）：**

| 状态 | 含义 | 处理 |
|:----|:-----|:-----|
| ✅ SELECTED | 安全、相关 | 直接注入 prompt |
| ⚠️ CONFLICTED | 有矛盾 | 注入 + 标注矛盾提醒 LLM |
| ⛔ DEMOTED | 低置信度/过时 | 不注入, 但可通过查询获取 |
| 🚫 BLOCKED | PII / 敏感信息 | 彻底拦截, 永不注入 |

### 6.7 上下文窗口管理

Agent 做多步任务时会积累大量中间消息。CogniMem 实时监控上下文大小：
- 超过 **24K token** 时自动裁剪最早的无用轮次
- 保留最近的系统消息、工具结果和用户对话
- 确保 Agent 不因上下文超限而崩溃

### 6.8 稳定性与容错

| 机制 | 说明 |
|:----|:-----|
| DB 连接池健康检测 | 每次请求前 `SELECT 1` 探活，坏连接自动替换 |
| LLM 3 次重试 | 指数退避 (1s/3s/9s) |
| 全局异常捕获 | 所有端点统一返回结构化 JSON，不裸抛 500 |
| systemd 开机自启 | 重启后服务自动恢复 |
| HTML 模板缓存键 | `mtime` 缓存, 修改文件自动刷新, 无需重启 |

---

## 七、MCP Server

### 7.1 13 个 MCP 工具

| 工具 | 用途 | 返回 |
|:----|:-----|:-----|
| `memory_recall` | 根据查询文本召回 SPO 三元组记忆 | 事实列表 + 置信度 + 来源 |
| `memory_remember` | 提取文本为三元组存入 | 提取数量 + 矛盾数 |
| `memory_diagnose` | 诊断系统健康状态 | 健康分 + 各维度指标 |
| `memory_status` | 统计概览 | 按类型分布 + 路由统计 |
| `memory_forget` | 删除指定事实 | 删除确认 |
| `memory_ask` | 问答式召回 | 记忆 + 核心信念 + 主动问题 |
| `memory_groom` | 触发记忆维护 | 遗忘/衰减/合并/抽象数量 |
| `memory_batch_remember` | 批量存入 (`||` 分隔) | 成功数 |
| `memory_bus` | 跨 Agent 总线查询 | 多 Agent 去重结果 |
| `audit_query` | 查询审计日志 | 操作历史列表 |
| `credential_store` | 存储凭证（自动掩码）| 服务 + 状态 |
| `credential_recall` | 召回凭证原文 | 原文 + 掩码显示 |
| `credential_list` | 列出已存凭证 | 服务列表（不泄露原文）|

### 7.2 使用方式

```bash
# stdio 模式（默认，用于 MCP Host）
python -m cognimem.mcp_server

# SSE 模式（HTTP 服务）
python -m cognimem.mcp_server --sse --port 8100
```

兼容所有 MCP 客户端：Claude Desktop、Claude Code、Cursor、VS Code、Windsurf、Cline、Roo Code、OpenClaw。

---

## 八、仪表盘与知识图谱

### 8.1 三页面统一入口

| 页面 | 路由 | 功能 |
|:----|:----|:-----|
| 💬 聊天 | `/chat` | 对话 + 流式输出 + 记忆召回 |
| 📊 仪表盘 | `/dashboard` | 统计 + 管理 + 健康检测 |
| 🔗 知识图谱 | `/graph` | SPO 关系可视化 |

三页面共享 `localStorage` 同步 agent 选择，切换无缝。

### 8.2 仪表盘模块

**指标卡（4 个）：**
| 卡片 | 数据来源 |
|:----|:---------|
| 🧠 总记忆数 | facts 表 COUNT |
| 🔮 抽象概念 | fact_type = abstraction |
| ❤️ 活跃偏好 | fact_type = preference, confidence > 0.6 |
| ⚠️ 需关注项 | 矛盾数 + 低置信度 + API 错误 |

**系统健康卡（5 维度）：**
| 维度 | 检测方式 |
|:----|:---------|
| DB 连接 | `SELECT 1` 探活 |
| LLM 状态 | 快速预热 ping |
| 配置完整性 | API Key / 模型名检查 |
| 工具可用性 | 12 工具注册状态 |
| API 错误率 | 滑动窗口（最近 100 次）|

**v0.13 新增卡片：**
- 📦 知识库状态 — 凭证存储/召回统计
- ⚡ STM 缓冲区 — FIFO 淘汰监控
- 🧭 智能路由 — L0-L3 命中率 + 零LLM路由分类
- 🏥 系统健康 — 实时评分 + 问题列表

**图表区：**
- 记忆分布图：doughnut (preference/fact/action/goal/decision/observation/abstraction)
- 增长趋势图：累计记忆量时间曲线
- 记忆分类柱状图：各类别百分比

**活动日志：** 滚动显示最近记忆事件（创建/确认/挑战/衰减/删除）。

**关键洞察：** 自动生成记忆分析摘要（最多类型、最新记忆、低置信度告警等）。

### 8.3 知识图谱

| 特性 | 说明 |
|:----|:------|
| 可视化引擎 | Canvas 力导向图, 支持拖拽/缩放 |
| 节点 = 实体 | SPO 三元组的 subject 和 object |
| 边 = 关系 | SPO 三元组的 predicate |
| 8 种语义类型 | 不同颜色标识 |
| 抽象节点 | 虚线边框, 视觉区分原始事实 vs 高层归纳 |
| 类型筛选 | Checkbox 多选过滤 |
| 节点详情 | 右侧面板显示字段、关系列表、关联事实 |

**8 种语义类型颜色映射：**
| 类型 | 颜色 | 说明 |
|:----|:----:|:-----|
| preference | 🟢 绿 | 偏好 |
| fact | 🔵 蓝 | 事实 |
| action | 🟠 橙 | 行为 |
| goal | 🟣 紫 | 目标 |
| decision | 🔴 红 | 决策 |
| observation | 🟡 黄 | 观察 |
| entity | ⚪ 灰 | 实体 |
| abstraction | ⬜ 虚框 | 抽象归纳 |

---

## 九、API 端点完整清单

### 9.1 主服务（memory_agent, :8000）

| 方法 | 路由 | 说明 | v0.13 |
|:----|:-----|:-----|:-----:|
| GET | `/` | 服务存活检测 | ✅ |
| GET | `/chat` | 聊天界面（Web） | ✅ |
| GET | `/dashboard` | 仪表盘（Web） | ✅ |
| GET | `/graph` | 知识图谱（Web） | ✅ |
| GET | `/agents` | Agent 列表 | ✅ |
| GET | `/health` | 健康检测 + 综合评分 | ✅ |
| GET | `/stats` | 统计数据 | ✅ |
| GET | `/status` | Agent 状态 | ✅ |
| GET | `/memories` | 记忆列表（分页）| ✅ |
| GET | `/memories/search` | 记忆搜索 | ✅ |
| GET | `/preferences` | 偏好/事实列表 | ✅ |
| GET | `/preferences/history` | 偏好演变历史 | ✅ 修 bug |
| GET | `/decay-trace/{id}` | 衰减曲线可视化 | ✅ 修 bug |
| GET | `/decay-analysis` | 衰减分析 | ✅ |
| GET | `/versions/{fact_id}` | 事实版本历史 | ✅ |
| GET | `/memory-graph` | 图谱 JSON 数据 | ✅ |
| GET | `/audit` | 审计日志查询 | ✅ |
| POST | `/chat` | 聊天（非流式）| ✅ |
| POST | `/chat/stream` | 聊天（SSE 流式）| ✅ |
| POST | `/chat/long` | 超长上下文 (1M token) | ✅ |
| POST | `/remember` | 存入记忆 | ✅ |
| POST | `/recall` | 召回记忆 | ✅ |
| POST | `/confirm` | 确认事实 ↑ 置信度 | ✅ |
| POST | `/challenge` | 质疑事实 ↓ 置信度 | ✅ |
| POST | `/consolidate` | 记忆整合 | ✅ |
| POST | `/groom` | 记忆维护 | ✅ 修 bug |
| POST | `/merge` | 记忆合并 | ✅ 修 bug |
| POST | `/process-transcript` | 文档批量提取 | ✅ |
| POST | `/memory-bus` | 跨 Agent 总线查询 | ✅ |
| DELETE | `/memories/{fact_id}` | 删除事实 | ✅ |
| DELETE | `/clear` | 清空所有记忆 | ✅ |

### 9.2 引擎独立服务（cognimem, :8001）

| 方法 | 路由 | 说明 |
|:----|:-----|:-----|
| GET | `/` | 服务存活 |
| GET | `/stats` | 统计数据 |
| GET | `/health` | 健康检查 |
| GET | `/versions/{fact_id}` | 版本历史 |
| POST | `/remember` | 存入记忆 |
| POST | `/recall` | 召回记忆 |
| POST | `/ask` | 问答式召回 |
| POST | `/confirm` | 确认事实 |
| POST | `/challenge` | 质疑事实 |
| POST | `/consolidate` | 记忆整合 |
| POST | `/resolve-contradiction` | 解决矛盾对 |
| DELETE | `/clear` | 清空所有 |

---

## 十、数据库结构

### 10.1 核心表

**facts — 事实网络主表**
```sql
fact_id         UUID PRIMARY KEY
agent_id        VARCHAR(64) NOT NULL
triple_key      VARCHAR(512) UNIQUE    -- subject|predicate|object
subject         TEXT NOT NULL
predicate       TEXT NOT NULL
object          TEXT NOT NULL
fact_type       VARCHAR(32) DEFAULT 'general'
confidence      FLOAT DEFAULT 0.6
importance      FLOAT DEFAULT 0.5
encoding_level  VARCHAR(16) DEFAULT 'raw'
evidence        JSONB DEFAULT '[]'     -- [{source, statement, timestamp}]
contradictions  TEXT[] DEFAULT '{}'    -- 关联的矛盾 fact_id
connected_facts TEXT[] DEFAULT '{}'    -- 关联事实
context_tags    TEXT[] DEFAULT '{}'
source_label    VARCHAR(128)
citation        TEXT
stale_warning   VARCHAR(256)
ttl_seconds     INTEGER
expires_at      TIMESTAMPTZ
embedding       vector(384)            -- pgvector 向量索引
created_at      TIMESTAMPTZ DEFAULT NOW()
updated_at      TIMESTAMPTZ DEFAULT NOW()
```

**versions — 版本链**
```sql
version_id  UUID PRIMARY KEY
fact_id     UUID NOT NULL REFERENCES facts(fact_id)
agent_id    VARCHAR(64) NOT NULL
subject     TEXT
predicate   TEXT
object      TEXT
confidence  FLOAT
timestamp   TIMESTAMPTZ DEFAULT NOW()
reason      VARCHAR(256)    -- 'created' / 'confirmed' / 'challenged' / 'decayed' / 'abstracted'
```

**contradictions — 矛盾记录**
```sql
id            UUID PRIMARY KEY
agent_id      VARCHAR(64) NOT NULL
fact_id_a     UUID NOT NULL
fact_id_b     UUID NOT NULL
conflict_type VARCHAR(32)    -- 'negation' / 'semantic' / 'llm_judged'
status        VARCHAR(16) DEFAULT 'pending'  -- 'pending' / 'resolved'
resolution    TEXT           -- 矛盾解决说明
created_at    TIMESTAMPTZ DEFAULT NOW()
resolved_at   TIMESTAMPTZ
```

**agents — 智能体**
```sql
agent_id    VARCHAR(64) PRIMARY KEY
name        VARCHAR(128)
created_at  TIMESTAMPTZ DEFAULT NOW()
last_active TIMESTAMPTZ DEFAULT NOW()
```

### 10.2 关键索引

- `facts.agent_id` → BTREE 索引
- `facts.triple_key` → 唯一索引
- `facts.embedding` → HNSW 向量索引 (pgvector)
- `versions(fact_id, timestamp)` → 复合索引
- `contradictions.agent_id` → BTREE 索引

---

## 十一、配置说明

### 11.1 环境变量

| 变量 | 默认值 | 必须 | 说明 |
|:----|:------|:----:|:-----|
| `QWEN_API_KEY` | — | ⚠️ | Qwen DashScope API Key |
| `DEEPSEEK_API_KEY` | — | ⚠️ | DeepSeek Key（二选一）|
| `QWEN_MODEL` | `deepseek-chat` | 可选 | 模型名 |
| `COGNIMEM_DB` | — | 可选 | PostgreSQL DSN，缺省 = 内存模式 |
| `COGNIMEM_LLM` | `false` | 可选 | `true` 启用 LLM 提取 |
| `MCP_PORT` | `8100` | 可选 | MCP SSE 端口 |

⚠️：`QWEN_API_KEY` 或 `DEEPSEEK_API_KEY` 至少配一个。

### 11.2 硬件要求

| 项 | 最低 | 推荐 |
|:--|:----|:----:|
| CPU | 1 核 | 2 核 |
| RAM | 512 MB | 2 GB |
| 存储 | 5 GB | 20 GB (含 PostgreSQL) |
| 数据库 | PostgreSQL 14+ | PostgreSQL 16 + pgvector |

### 11.3 依赖清单

```
fastapi>=0.138       Web 框架
uvicorn[standard]     ASGI 服务器
pydantic>=2          数据校验
sqlalchemy>=2        ORM (可选)
psycopg2-binary      PostgreSQL 驱动
httpx>=0.28          HTTP 客户端
jinja2>=3            模板引擎
python-multipart     表单解析
aiofiles             异步文件 I/O
python-dotenv        .env 加载
numpy>=2.5           数值计算 (embedding)
mcp>=1.0             MCP 协议
```

---

## 十二、部署说明

### 12.1 生产服务器信息

| 项 | 值 |
|:--|:---|
| 地址 | `42.121.253.80:8000` |
| 系统 | Ubuntu 22.04 LTS |
| 用户 | `ecs-user` |
| 项目路径 | `/home/ecs-user/qwen-memoryagent` |
| 服务管理 | `systemctl [start\|stop\|status\|restart] cognimem.service` |
| 日志 | `/tmp/qwen-agent3.log` |
| 数据库 | PostgreSQL 16（本地 :5432, systemd 管理）|

### 12.2 systemd 服务

```ini
[Unit]
Description=CogniMem 认知记忆系统
After=network.target postgresql.service

[Service]
Type=simple
User=ecs-user
WorkingDirectory=/home/ecs-user/qwen-memoryagent
Environment=PYTHONPATH=/home/ecs-user/qwen-memoryagent/src
ExecStart=/usr/bin/python3 -m uvicorn memory_agent.main:app --host 0.0.0.0 --port 8000 --app-dir /home/ecs-user/qwen-memoryagent/src
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 12.3 部署步骤

```bash
# 1. 同步代码
cd /Users/baikai/projects/qwen-memoryagent
rsync -avz --delete \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude '.pytest_cache' --exclude '.env' --exclude '*.log' \
  ./ ecs-user@42.121.253.80:/home/ecs-user/qwen-memoryagent/

# 2. 重启服务
ssh ecs-user@42.121.253.80 "sudo systemctl restart cognimem.service"

# 3. 验证
curl http://42.121.253.80:8000/health
```

### 12.4 Docker 部署

```bash
# 构建
docker build -t cognimem .

# 运行（需要先启动 PostgreSQL）
docker run -d --name cognimem \
  -p 8000:8000 \
  -e QWEN_API_KEY=sk-xxx \
  -e COGNIMEM_DB=postgresql://... \
  cognimem
```

### 12.5 踩坑记录

| 问题 | 解决 |
|:----|:-----|
| `.venv/bin/python3` 是 symlink | 用 `/usr/bin/python3`，包装在 `~/.local/lib/python3.10/` |
| `pkill -f uvicorn` 误杀 SSH | 用 `systemctl restart` 代替 |
| 首次启动报 `facts 表不存在` | 自动建表，等 3 秒再查 |
| pgvector 未安装 | 连接串改用 `PGOPTIONS='-c search_path=public'` 降级纯文本 |

---

## 十三、技术栈

| 层 | 选型 | 版本 | 理由 |
|:--|:----|:----:|:-----|
| **LLM** | Qwen3.7-Plus + Qwen3.6-Flash | DashScope | 推理/提取 + 快速响应; 1M token |
| **后端语言** | Python | 3.10+ | 生态成熟, AI 原生 |
| **Web 框架** | FastAPI | 0.138+ | 性能接近 Go |
| **服务器** | Uvicorn | — | ASGI 高性能 |
| **数据库** | PostgreSQL | 16 | 关系 + 向量一体化 |
| **向量搜索** | pgvector | — | HNSW 索引, < 25ms |
| **Embedding** | 纯 Python 哈希 v2.0 | — | 零依赖, 零成本 |
| **前端** | Vanilla JS + Chart.js + Canvas | — | 零框架依赖 |
| **MCP** | FastMCP | 1.0+ | MCP 协议 |
| **部署** | 阿里云 ECS + systemd | Ubuntu 22.04 | 开机自启 |
| **容器** | Docker (可选) | — | 迁移友好 |

---

## 十四、竞品对比优势

| 维度 | **CogniMem** | Universal Agent OS | Mimir | Engram |
|:----|:-----------:|:-----------------:|:-----:|:------:|
| SPO 三元组结构 | ✅ **独家** | ❌ 自由文本 | ❌ 结构化实体 | ❌ 类型化记忆 |
| 三层压缩 | ✅ **独家** | ❌ | ❌ | ❌ |
| 健康分系统 | ✅ **独家** | ❌ | ❌ | ❌ |
| Agent 自主规划 | ✅ | ❌ | ❌ | ❌ |
| 知识图谱可视化 | ✅ **独家** | ❌ | ❌ | ❌ |
| 三层矛盾检测 | ✅ **独家** | ❌ | ❌ | ✅ LLM 单层 |
| 零LLM路由 | ✅ **独家** | ❌ | ❌ | ❌ |
| 记忆进化（事实链接）| ✅ | ❌ | ❌ | ❌ |
| 语义缓存 | ✅ | ❌ | ❌ | ❌ |
| 科学遗忘 | ✅ Ebbinghaus+Weibull | ❌ | ✅ Ebbinghaus | ✅ Ebbinghaus |
| 混合搜索 | ✅ 4 层流水线 | ✅ BM25+语义 | ✅ FTS5+向量 | ✅ 向量+重要性 |
| MCP 工具数量 | ✅ 13 个 | ✅ 6 个 | ✅ 27-43 个 | ✅ 3 个 |
| 跨 Agent 总线 | ✅ | ✅ | ❌ | ❌ |
| 凭证安全存储 | ✅ | ❌ | ✅ AES 加密 | ❌ |
| 审计日志 | ✅ | ❌ | ❌ | ❌ |
| 仪表盘 | ✅ 完整 Web UI | ❌ | ❌ TUI | ✅ 记忆板 |
| 噪音自动过滤 | ✅ | ❌ | ❌ | ❌ |
| 中文深度优化 | ✅ | ❌ | ❌ | ❌ |
| Python 纯原生 | ✅ | ✅ | ❌ Rust | ❌ TS |
| 部署 | ✅ 阿里云 ECS | ✅ 阿里云 ECS | ❌ 本地优先 | ✅ 阿里云 FC |

---

## 十五、测试与质量

### 15.1 v0.13 测试覆盖

| 测试维度 | 数量 | 通过率 |
|:--------|:----:|:------:|
| 自动化脚本 (test_v013.sh) | 18 项 | ✅ 100% |
| 深度验证 (comprehensive) | 102 项 | ✅ 100% |
| pytest | 26 项 | ✅ 100% |
| 主服务 API 端点 | 31 项 | ✅ 100% |
| 引擎 API 端点 | 12 项 | ✅ 91.7% |
| 边界与错误处理 | 8 项 | ✅ 100% |
| **稳定性测试综合** | **28 项** | ✅ **100%** |

### 15.2 稳定性场景

| 场景 | 操作量 | 结果 |
|:----|:-----:|:----:|
| 单线程循环压力 | 1000 次 @ 18op/s | ✅ 无错误 |
| 50 线程并发 | 2500 次混合操作 | ✅ 0 死锁 |
| 全 API 混合负载 | 500 次 12 种 API 交替 | ✅ 无错误 |
| 超长文本 | 10000 字 | ✅ 正常处理 |
| 特殊字符 | 12 种 (含 emoji/控制符) | ✅ 全部通过 |
| 重复内容去重 | 100 条完全相同 | ✅ 正常 |
| 空值边界 | 空文本/空查询/非法 UUID | ✅ 优雅处理 |
| LRU 缓存监控 | 内存稳定性 | ✅ 线性增长, 无泄漏 |
| HTTP 稳定性 | 200 次跨 15 端点 | ✅ 0 异常 |
| 凭证循环 | 300 次 | ✅ 无丢失/无泄漏 |
| 长时间运行 | 1000 次持续操作, 55.3s | ✅ 稳定 |

### 15.3 Bug 修复（v0.13）

| ID | 严重度 | 问题 | 修复 |
|:--|:------|:-----|:-----|
| BUG-1 | 🔴 严重 | `/preferences/history` 500 | 添加 cogni 分支 |
| BUG-2 | 🔴 中 | `/decay-trace/{id}` 500 + agent_id 硬编码 | 修复分支 + 参数化 |
| BUG-3 | 🟡 中 | `/groom` 缺少 cogni 分支 | 添加 → consolidate |
| BUG-4 | 🟡 中 | `/merge` 缺少 cogni 分支 | 添加 → consolidate |
| BUG-6 | 🟢 低中 | 非法 UUID 导致 DB 崩溃 | UUID 校验 + 400 返回 |
| BUG-8 | 🟢 低 | 缺少 `/health` 端点 | 新增完整健康检查 |

---

## 十六、常见问题与排错

### 16.1 服务无法启动

| 现象 | 原因 | 解决 |
|:----|:-----|:-----|
| `ModuleNotFoundError` | 依赖未装全 | `pip install -r requirements.txt` |
| `Connection refused` 数据库 | PostgreSQL 未启动 | `sudo systemctl start postgresql` |
| LLM 返回空 | API Key 未配或失效 | 检查环境变量 `QWEN_API_KEY` |
| 健康分 < 80 | API 错误率过高 | 检查日志 `/tmp/qwen-agent3.log` |

### 16.2 记忆系统异常

| 现象 | 原因 | 解决 |
|:----|:-----|:-----|
| 召回为空 | 向量搜索降级纯文本 | `CREATE EXTENSION vector;` |
| 矛盾误报 | 跨类别偏好冲突 | 检查矛盾记录, 手动解决 |
| 提取不到三元组 | 句式不在规则中 + LLM 不可用 | 检查 API Key, 或补充规则 |
| Agent 返回空回复 | LLM 调用异常 | 检查网络/API 配额 |

### 16.3 性能参考

| 操作 | 延迟 | 说明 |
|:----|:---:|:-----|
| 规则提取 | < 1ms | 0 token |
| L0 缓存命中 | < 1ms | 0 token |
| L2 向量召回 | 10-25ms | pgvector HNSW |
| LLM 提取 | 200-800ms | 取决于网络 |
| 矛盾检测 L1+L2 | < 5ms | 0 token |
| Consolidate (100 条) | 1-3s | 后台自动触发 |

---

*文档生成于 2026-07-12 | CogniMem v0.13 | 592 行 → 941 行*
