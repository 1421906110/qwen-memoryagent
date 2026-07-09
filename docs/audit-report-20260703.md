# CogniMem 认知记忆系统 — 改进审计报告

> **审计日期：** 2026-07-03
> **AI 行业背景：** Claude Opus 4.8 / Sonnet 5, DeepSeek Thinking, Qwen 3.7+, 上下文窗口普遍 100k-1M tokens
> **审计范围：** CogniMem 引擎 (`~/Desktop/cognimem/`) + MemoryAgent (`~/projects/qwen-memoryagent/`)

---

## 目录

1. [四个「更」评分总览](#1-四个更评分总览)
2. [🔴 关键问题详细分析](#2-关键问题详细分析)
3. [行业对标分析](#3-行业对标分析)
4. [改进优先级路线图](#4-改进优先级路线图)
5. [各组件改进清单](#5-各组件改进清单)

---

## 1. 四个「更」评分总览

```
🧠 更智能    ████████░░  4/5  ← 核心架构好，实现有坑
💰 更省Token ██████░░░░  3/5  ← 规则+缓存策略对，但落地不彻底
⚡ 更省资源  ██████░░░░  3/5  ← O(n) 扫库是硬伤
🚀 更创新    ████████░░  4/5  ← 矛盾驱动+抽象化是真创新
```

**综合：3.5/5 — 方向全对，实现粗糙**

---

## 2. 关键问题详细分析

### 🔴 P0 — 矛盾检测代码双份

| 文件 | 行数 | 功能 | 状态 |
|------|------|------|------|
| `core/fact_network.py` | 317-433 (117行) | 内置 `_detect_contradictions()` | ✅ 实际在用，覆盖全面 |
| `core/contradiction.py` | 15-103 (89行) | `ContradictionDetector.detect()` | ❌ **从未被调用** |

**对比差异：**

| 能力 | `fact_network.py` | `contradiction.py` |
|------|------------------|-------------------|
| L1 直接矛盾（同 subject+predicate，不同 object） | ✅ | ✅ |
| L2 否定矛盾（喜欢 vs 不喜欢） | ✅ 含通用「不/没/未」前缀检测 | ✅ 仅 `_NEGATION_MAP` 精确匹配 |
| L3 上下文变化（截止周五→周三） | ✅ 含 context_predicates 白名单 | ❌ |
| 偏好类去噪（喜欢咖啡 vs 喜欢火锅不算矛盾） | ✅ 共享标签才标记 conflict | ❌ |
| 肯定形式判断（"会"是"不会"的正面） | ✅ `_is_positive()` | ❌ |
| 否定例外排除（"不错""不少"不算否定） | ✅ exceptions 集合 | ❌ |

**影响：** 如果未来要升级矛盾检测，改 `contradiction.py` 是完全白费功夫。两套代码维护成本 double。

**修复方案：** 
1. 删除 `contradiction.py` 的 `ContradictionDetector` 类
2. 将 `ContradictionResolver` 重构到 `fact_network.py` 或独立的 `resolver.py`
3. `fact_network._detect_contradictions()` → 提取为独立模块方法，方便单测

---

### 🔴 P1 — LLM 提取不区分难易，每条都调

当前流程（`brain.remember()`）：

```
用户输入 → 有 LLM? → 调 LLM 提取（≈500-1000 tok）→ 规则提取 fallback
                   ↘ 无 LLM? → 规则提取（0 tok）
```

**问题：** 即使规则能提取的三元组（如"我喜欢吃火锅"→ `用户|喜欢|吃火锅`），也会先调 LLM。

**应然流程：**

```
用户输入 → 规则提取 → 提取到且置信度≥0.6? → 直接入库（0 tok）
                   ↘ 没提到或置信度<0.6? → LLM 提取（≈500 tok）
                                          → LLM 提取结果和规则提取结果合并去重
```

**收益估算：**

| 场景 | 当前 tok/条 | 改进后 tok/条 | 节省 |
|------|-----------|-------------|------|
| 简单句（80% 场景） | 500-1000 | 0 | 100% |
| 复杂句（20% 场景） | 500-1000 | 500-1000 | 0% |
| 加权平均 | ~600 | ~120 | **~80% 节省** |

**影响：** 这是四个「更」中最关键的改进。直接切中"更省 Token"的目标。

---

### 🔴 P2 — 全量 O(n) 扫库

以下方法都遍历全部事实：

| 方法 | 位置 | 复杂度 | 影响 |
|------|------|--------|------|
| `_get_agent_facts()` | `fact_network.py:927` | O(n) | stats、矛盾检测、整合全依赖它 |
| `recall()` 中的 L0 缓存匹配 | `fact_network.py:196` | O(cache) | 遍历全部缓存条目 |
| `_detect_contradictions()` | `fact_network.py:348` | O(n) | 新事实和每个已有事实比对 |
| `forget()` 衰减循环 | `fact_network.py:454` | O(n) | 遍历每个事实算衰减 |
| `_abstract_memories()` | `fact_network.py:656` | O(n) | 遍历所有非抽象事实 |

**当前规模下（~10条/agent）毫无问题，但设计上限制了扩展性。** 目标应该是能支撑单 agent 10 万+ 条事实时仍保持响应。

**修复方案（分阶段）：**
1. 短期：`_get_agent_facts()` 加 LIMIT + 分页参数
2. 中期：矛盾检测改用 DB 端查询（`WHERE subject = ? AND (predicate IN negated_set OR ...)`），不用全量拉到内存
3. 长期：pgvector 索引 + 专门的事实图索引

---

### 🟡 P3 — 同步整合阻塞请求

`consolidate()` 和 `_maybe_auto_consolidate()` 都在请求线程里跑：

```
POST /consolidate → brain.consolidate() → 抽象化(调LLM) → 去重 → 提升 → 遗忘(全量遍历)
```

如果抽象化调了 LLM（`_llm_categorize`），阻塞时间可能 >3s。

**修复方案：**
- 用 `asyncio.to_thread()` 或 `threading.Thread` 把 consolidate 丢后台
- 加一个 `/_consolidate-status` 端点查看整合进度
- 自动整合触发时返回 202 Accepted，不等到整合完成才响应

---

### 🟡 P4 — LLM 提取器模型配置碎片化

| 位置 | 默认模型 | 配置方式 |
|------|---------|---------|
| `cognimem/core/brain.py:52` | `deepseek-chat` | `DEEPSEEK_API_KEY` 或 `DASHSCOPE_API_KEY` |
| `cognimem/core/llm_extractor.py` | `deepseek-chat` | 同上 |
| MemoryAgent `llm_client.py` | `qwen-plus` | `QWEN_API_KEY` + `QWEN_BASE_URL` |
| MemoryAgent `.env` | `qwen3.7-plus` | 环境变量 |

**问题：** CogniMem 引擎有自己的 LLM 配置路径，MemoryAgent 有另一套。如果 CogniMem 引擎启动时没设 `DASHSCOPE_API_KEY`，LLM 提取器就不工作。但实际上 MemoryAgent 已经配好了 Qwen key，只是没传给 CogniMem。

**修复方案：**
- CogniMem 引擎的 `llm_extractor.py` 增加 `QWEN_API_KEY` + `QWEN_BASE_URL` 支持
- MemoryAgent 启动时将 Qwen 配置透传给 CogniMem 引擎
- 统一模型配置到一个共享模块

---

### 🟡 P5 — 对话历史只在浏览器 localStorage

上次已修了一部分（前端传 `messages`、后端注入），但对比"更省资源"目标，对话历史应该：
- 不需要每次都传全部历史（token 浪费）
- 应该自动从存储系统召回相关的历史消息

---

## 3. 行业对标分析

### vs Mem0 (2025-2026)

| 维度 | Mem0 | CogniMem |
|------|------|----------|
| 存储结构 | 向量 + 元数据 | 三元组 + 图 + 向量 |
| 矛盾检测 | ❌ 无 | ✅ L1/L2/L3 + 主动追问 |
| 抽象化 | ❌ 无 | ✅ 碎片→高层归纳 |
| 遗忘曲线 | ❌ 固定 TTL | ✅ 艾宾浩斯自适应 |
| 提取成本 | 每次调 LLM | ✅ 规则 0 tok + LLM 兜底（当前未完美实现）|
| 置信度 | ❌ 无 | ✅ 贝叶斯 + 来源权重 |

### vs 普通 RAG (2024-2026)

| 维度 | 普通 RAG | CogniMem |
|------|---------|----------|
| 存储 | 向量（块级） | 三元组（事实级） |
| 理解深度 | 语义相似度 | 结构匹配 + 矛盾检测 |
| 知识演进 | 无 | 置信度变化 + 抽象化 |
| Token 效率 | 低（每次检索送 chunk） | 高（三级缓存 + 规则优先） |

**优势定位：** CogniMem 在"结构化认知"方向上是领先的。矛盾检测 + 抽象化 + 主动学习是竞品没有的组合。

**劣势：** 工程实现粗糙（O(n)扫库、同步阻塞、规则提取覆盖面窄）。

---

## 4. 改进优先级路线图

```
紧急 ┼──────────────────────────────────────>
    │                                          
    │  P0 矛盾检测合并     P3 异步consolidate   
    │  P1 规则优先提取                         
    │  P5 Qwen配置统一                         
    │                                          
    ├──────────────────────────────────────>
       Qwen 黑客松截止(7/9)           长远

短平快（立刻能改）：P0 → P5
黑客松后：          P1 → P2 → P3 → P4
```

### 执行状态

| 优先级 | 任务 | 状态 | 工量 | 收益 |
|--------|------|------|------|------|
| P0 | 矛盾检测去重 + 清理死代码 | ✅ **已修** | 15min | 删除 ~100 行无用代码 |
| P4→P5 | Qwen 配置统一（API Key + Base URL + Model） | ✅ **已修** | 15min | LLM 提取走 Qwen |
| P1.1 | SIMPLE_PATTERNS 扩充 10+ 模式 | ✅ **已修** | 10min | 更多场景 0 tok 提取 |
| P1.2 | 规则优先 LLM 兜底（brain 层调优） | 🔲 待修 | 1h | Token 节省 80% |
| P3 | asyncio 后台 consolidate | 🔲 待修 | 30min | 不阻塞请求 |
| P2 | 数据库端矛盾检测 + 分页 | 🔲 待修 | 2h | 可扩展至 10 万条 |

---

## 5. 各组件改进清单

### `core/fact_network.py` (核心引擎)

```
[P0] 将 _detect_contradictions() 提取为独立函数，删除 contradiction.py 的重复代码
[P2] _get_agent_facts() 加分页参数
[P2] 矛盾检测改用 DB 端过滤
[P3] _maybe_auto_consolidate() 改异步触发
[P1] recall() 加入语义缓存（"咖啡"≈"喝咖啡"）
```

### `core/contradiction.py`

```
[已修] 删除 ContradictionDetector 类（从未调用过）
[保留] ContradictionResolver（供 future 使用）
[修改] 文件只保留 ContradictionResolver，移除全部 ~100 行死代码
```

### `core/brain.py` (编排器)

```
[已修] LLM 提取器增加 QWEN_API_KEY / QWEN_BASE_URL / QWEN_MODEL 支持
[已修] 删除废弃的 self.contradiction_detector 实例化
[待修] remember() 改为：规则提取→够用→不入库→才调 LLM（llm_extractor 已实现，brain 层需调优）
```

### `core/llm_extractor.py` (LLM 提取)

```
[已修] 增加 Qwen 3.7+ 作为支持的模型后端
[已修] 自动检测 backend：有 QWEN_API_KEY 且未设 base_url 时走 DashScope
[已修] 扩展 SIMPLE_PATTERNS：新增时间/日期、日常状态、属性类等 10+ 模式
[已修] check_connection() 提示包含 QWEN_API_KEY
[待修] 更智能的简单句判定（当前 ≤40 字符 + 无复杂标记，可放宽）
```

### `core/extractor.py` (规则提取)

```
[已修] llm_extractor.py 的 SIMPLE_PATTERNS 已扩（extractor.py 作为纯规则 fallback 保留）
[待修] 增加置信度评分：模式匹配精确度影响初始置信度
[待修] 增加同义词/近义词映射（"爱喝"→"喜欢"）
```

### `core/recall.py` (召回路由)

```
[P2] L0 缓存改为语义缓存（n-gram 模糊匹配）
[P2] 数据库端搜索加 LIMIT，避免全量扫描
```

---

## 总结

**CogniMem 方向正确，但工程实现有 gap。**

最令人兴奋的部分（矛盾驱动学习、抽象化、艾宾浩斯遗忘）已经跑通，但：

- 代码有重复（矛盾检测两套）
- 优化不彻底（规则优先没落地 → Token 浪费）
- 扩展性堪忧（全量扫库）
- 配置碎片化（Qwen vs DeepSeek 两套）

趁 Qwen 黑客松还没截止，优先搞 **P0（合并矛盾代码）** 和 **P5（Qwen 配置统一）**，这两项改动最小、让引擎立刻能用。

---

*报告生成：2026-07-03*
*AI 行业背景：Claude Opus 4.8 时代 — 记忆系统从"有没有"进入"好不好/省不省"阶段*
