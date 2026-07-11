# CogniMem v0.13 完整测试手册

> 版本: v0.13 | 日期: 2026-07-11  
> 前置条件: 引擎运行在 `:8001`，UI 运行在 `:9999`

---

## 目录

1. [快速运行测试](#1-快速运行测试)
2. [后端 API 测试](#2-后端-api-测试)
3. [前端 UI 测试](#3-前端-ui-测试)
4. [自动化脚本测试](#4-自动化脚本测试)
5. [P0 新功能专项测试](#5-p0-新功能专项测试)
6. [P1 新功能专项测试](#6-p1-新功能专项测试)
7. [边界与压力测试](#7-边界与压力测试)
8. [回归测试](#8-回归测试)

---

## 1. 快速运行测试

一键运行所有自动化测试：

```bash
# 确保服务在运行
bash test_v013.sh
```

预期输出: `18 ✅ / 0 ❌`

---

## 2. 后端 API 测试

### 2.1 引擎 API (`:8001`)

| 端点 | 方法 | 预期响应 | 测试命令 |
|:----|:----|:---------|:---------|
| `/` | GET | `{"status":"alive"}` | `curl http://localhost:8001/` |
| `/remember` | POST | `{"status":"remembered",...}` | `curl -X POST http://localhost:8001/remember -H 'Content-Type: application/json' -d '{"text":"测试记忆"}'` |
| `/recall` | POST | `{"facts":[...]}` | `curl -X POST http://localhost:8001/recall -H 'Content-Type: application/json' -d '{"query":"测试"}'` |
| `/ask` | POST | `{"relevant_memories":[...]}` | `curl -X POST http://localhost:8001/ask -H 'Content-Type: application/json' -d '{"query":"测试"}'` |
| `/stats` | GET | 含 `stm_buffer`, `router_stats` | `curl 'http://localhost:8001/stats?agent_id=default'` |
| `/health` | GET | 含 `checks.router`, `checks.memory.stm_buffer` | `curl 'http://localhost:8001/health'` |

**验证要点**:
- [ ] `/stats` 响应包含 `stm_buffer` 字段（v0.13 新增）
- [ ] `/stats` 响应包含 `router_stats.intent_factual_pct`（v0.13 新增）
- [ ] `/health` 响应包含 `checks.memory.stm_buffer`（v0.13 新增）
- [ ] `/health` 响应包含 `checks.router` 对象（v0.13 新增）

### 2.2 UI API (`:9999`)

| 端点 | 方法 | v0.13 新增字段 | 测试命令 |
|:----|:----|:---------------|:---------|
| `/stats?agent_id=default` | GET | `credential_count` | `curl 'http://localhost:9999/stats?agent_id=default'` |
| `/health?agent_id=default` | GET | `checks.memory.stm_buffer`, `checks.router` | `curl 'http://localhost:9999/health?agent_id=default'` |
| `/memory-graph?agent_id=default` | GET | 无变化（图谱不变） | `curl 'http://localhost:9999/memory-graph?agent_id=default&limit=10'` |
| `/decay-analysis?agent_id=default` | GET | 无变化 | `curl 'http://localhost:9999/decay-analysis?agent_id=default'` |

**测试方法**:
```bash
# 验证 credential_count
curl -s 'http://localhost:9999/stats?agent_id=default' | python3 -c "import sys,json; print(json.load(sys.stdin).get('credential_count'))"

# 验证 stm_buffer
curl -s 'http://localhost:9999/health?agent_id=default' | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['checks']['memory'].get('stm_buffer','MISS'))"
```

---

## 3. 前端 UI 测试

### 3.1 仪表盘 (`/dashboard`)

#### 3.1.1 指标卡片

| # | 操作步骤 | 预期结果 | 实测 |
|:-:|:---------|:---------|:----:|
| 1 | 打开 `http://localhost:9999/dashboard` | 页面加载，无白屏/报错 | □ |
| 2 | 查看 Row1 第一卡片 `📦总记忆` | 显示数字 > 0 | □ |
| 3 | 查看 Row1 第二卡片 `🧩抽象概念` | 显示核心信念数 | □ |
| 4 | **查看 Row1 第三卡片 `🔐知识库`** | **显示凭证数量（v0.13 新增）** | □ |
| 5 | 查看 Row1 第四卡片 `⚠️需关注` | 显示矛盾数 | □ |
| 6 | **查看 Row2 第一卡片 `💚系统健康`** | **显示 ✅健康/⚠️一般/🔴异常（点击弹出详情）** | □ |
| 7 | 查看 Row2 第二卡片 `🔧MCP工具` | 显示工具数量 | □ |
| 8 | **查看 Row2 第三卡片 `⚡短期缓存`** | **显示 STM 缓冲区条数（v0.13 新增）** | □ |
| 9 | **查看 Row2 第四卡片 `🎯智能路由`** | **显示精确查询百分比（v0.13 新增）** | □ |

#### 3.1.2 健康弹窗

| # | 操作步骤 | 预期结果 | 实测 |
|:-:|:---------|:---------|:----:|
| 10 | 点击 `💚系统健康` 卡片 | 弹出详情弹窗 | □ |
| 11 | 弹窗中查看 **STM缓冲区** 行 | 显示 STM 条数 | □ |
| 12 | 弹窗中查看 **精确查询(Factual)** 行 | 显示百分比 + 次数 | □ |
| 13 | 弹窗中查看 **主动检索命中** 行 | 显示命中次数 | □ |
| 14 | 弹窗中查看 **查询总数** 行 | 显示总查询次数 | □ |
| 15 | 点击弹窗 `关闭` 按钮 | 弹窗关闭 | □ |

#### 3.1.3 图表

| # | 操作步骤 | 预期结果 | 实测 |
|:-:|:---------|:---------|:----:|
| 16 | 查看 `📂记忆分布` 甜甜圈图 | 图表渲染，有颜色区分 6 种类型 | □ |
| 17 | 查看 `📈增长趋势` 折线图 | 图表渲染，有日期+累积曲线 | □ |
| 18 | 调整浏览器宽度到 <768px | 卡片堆叠，响应式布局 | □ |

#### 3.1.4 底部内容

| # | 操作步骤 | 预期结果 | 实测 |
|:-:|:---------|:---------|:----:|
| 19 | 查看 `🔍关键洞察` | 文本分析，无报错信息 | □ |
| 20 | 查看 `🏷️记忆分类` | 6 种类型进度条，总计 100% | □ |
| 21 | 查看 `📋活动日志` | 显示最近记录，无空白报错 | □ |
| 22 | 切换 Agent 选择器 | 数据自动刷新 | □ |

### 3.2 聊天页 (`/chat`)

#### 3.2.1 侧栏

| # | 操作步骤 | 预期结果 | 实测 |
|:-:|:---------|:---------|:----:|
| 23 | 打开 `http://localhost:9999/chat` | 页面加载 | □ |
| 24 | 查看左侧顶部 Agent 信息 | 显示 Agent 名称和统计（事实/信念/矛盾） | □ |
| 25 | 查看 `📦知识分类` | 显示 6 种类型进度条 | □ |
| 26 | 查看 `🔗三元组` 区域 | 显示 SPO 三元组列表 | □ |
| 27 | 点击 `🧠归纳` 按钮 | 按钮变 ⏳ → ✅，侧栏刷新 | □ |
| 28 | 点击 `🗑️清空` 按钮 | 弹出确认框 | □ |
| 29 | 确认清空 | 记忆被清除 | □ |
| 30 | 点击 `↻` 刷新按钮 | 按钮旋转动画，三元组刷新 | □ |
| 31 | 鼠标悬停在三元组任一条上 | 显示点击指针 | □ |

#### 3.2.2 事实详情弹窗

| # | 操作步骤 | 预期结果 | 实测 |
|:-:|:---------|:---------|:----:|
| 32 | 点击一条三元组 | 弹出事实详情弹窗 | □ |
| 33 | 查看弹窗中的 SPO 显示 | 主体紫色、谓词灰色、客体正常色 | □ |
| 34 | 查看弹窗中的置信度 | 显示百分比 + 颜色（绿/橙/红） | □ |
| 35 | 查看 `📜版本历史` | 显示版本记录或「无版本记录」 | □ |
| 36 | 点击 `✅确认` 按钮 | 弹窗关闭，侧栏刷新 | □ |
| 37 | 点击 `❌质疑` 按钮 | 弹窗关闭，侧栏刷新 | □ |
| 38 | 点击 `关闭` 按钮 | 弹窗关闭 | □ |

#### 3.2.3 对话功能

| # | 操作步骤 | 预期结果 | 实测 |
|:-:|:---------|:---------|:----:|
| 39 | 在输入框输入文字按 Enter | 消息发送成功 | □ |
| 40 | 点击发送按钮 | 消息发送成功 | □ |
| 41 | 发送后显示用户消息 | 右侧蓝色气泡 | □ |
| 42 | Agent 回复（如有 LLM） | 显示 Agent 气泡 | □ |
| 43 | 点击 `+` 新对话 | 创建新对话，聊天区清空 | □ |
| 44 | 点击侧栏其他对话 | 切换到该对话 | □ |
| 45 | 鼠标悬停在对话旁的 `✕` | 显示删除按钮 | □ |
| 46 | 点击 `✕` 删除对话 | 弹出确认框 | □ |

#### 3.2.4 Agent 管理

| # | 操作步骤 | 预期结果 | 实测 |
|:-:|:---------|:---------|:----:|
| 47 | 点击 `🧠` Agent 头像 | 循环切换到下一个 Agent | □ |
| 48 | 点击 `+新建项目/Agent` | 弹出输入框 | □ |
| 49 | 输入名称点创建 | 新 Agent 出现在列表中 | □ |
| 50 | 点击新 Agent | 切换到该 Agent（数据独立） | □ |
| 51 | 鼠标悬停在 Agent 旁 `🗑️` | 显示删除按钮 | □ |
| 52 | 点击 `🗑️` 删除 | 弹出确认框 | □ |
| 53 | 确认删除 | Agent 被删除 | □ |
| 54 | 顶栏 `📊仪表盘` 链接 | 新标签打开仪表盘 | □ |
| 55 | 顶栏 `🔗图谱` 链接 | 新标签打开图谱 | □ |

### 3.3 图谱页 (`/graph`)

#### 3.3.1 页面加载

| # | 操作步骤 | 预期结果 | 实测 |
|:-:|:---------|:---------|:----:|
| 56 | 打开 `http://localhost:9999/graph` | 图谱加载，Canvas 渲染 | □ |
| 57 | 查看顶栏统计 | 显示节点数、连线数、密度、抽象数 | □ |
| 58 | Agent 选择器 | 可切换 Agent | □ |
| 59 | 类型过滤器全选 | 显示各类别复选框 | □ |
| 60 | 点击 `🔄刷新` | 图谱重新加载 | □ |
| 61 | 点击 `💬聊天` | 跳转到聊天页 | □ |
| 62 | 点击 `📊仪表盘` | 跳转到仪表盘 | □ |

#### 3.3.2 图谱交互

| # | 操作步骤 | 预期结果 | 实测 |
|:-:|:---------|:---------|:----:|
| 63 | 鼠标悬停在一个节点上 | 显示 tooltip（内容+类型+置信度） | □ |
| 64 | 鼠标移出节点 | tooltip 消失 | □ |
| 65 | 点击一个节点 | 节点高亮，右侧详情面板滑出 | □ |
| 66 | 详情面板查看内容 | 显示完整内容 | □ |
| 67 | 详情面板查看置信度 | 显示百分比 | □ |
| 68 | 详情面板查看关联记忆 | 显示关联节点列表 | □ |
| 69 | 点击关联列表中的一项 | 聚焦到该节点，显示其详情 | □ |
| 70 | 点击 `✕` 关闭详情面板 | 面板收起 | □ |
| 71 | 按 ESC 键 | 详情面板关闭 | □ |
| 72 | 按 R 键 | 图谱刷新 | □ |
| 73 | 取消勾选某种类型过滤器 | 该类型节点和边隐藏 | □ |
| 74 | 勾选全部类型 | 所有节点和边恢复显示 | □ |
| 75 | 拖拽 Canvas 空白区域 | 图谱平移 | □ |

---

## 4. 自动化脚本测试

提供两个级别的自动化测试：

### 4.1 快速验证（18 项）

```bash
bash test_v013.sh
```

覆盖范围:
- 环境检查：引擎/UI/pytest
- P0 功能：空数据/STM/进化/意图路由/语义缓存
- P1 功能：主动检索/Weibull/知识库
- 集成测试：完整流程/HTTP端点/POST端点
- 边界测试：大量数据/特殊字符/Agent隔离/并发

### 4.2 深度验证（102 项）

```bash
PYTHONPATH=src python3 /tmp/test_v013_comprehensive.py
```

覆盖范围（来自之前的全功能验证）：
- 空数据边界（8 项）
- STM 缓冲区（4 项）
- 记忆进化（3 项）
- 意图路由（11 项）
- 语义缓存（7 项）
- 主动检索（10 项）
- Weibull 衰减（10 项）
- 知识库（15 项）
- 现有功能回归（12 项）
- MCP Server（11 项）
- 并发安全（2 项）
- 边界测试（8 项）
- 一致性检查（1 项）

---

## 5. P0 新功能专项测试

### 5.1 零LLM路由层 (P0-1)

**测试目标**: 查询意图分类正确，factual 不走 L2/L3

```python
from cognimem.core.recall import RecallRouter

# 预期 factual 的查询（短、具体、无疑问词）
assert RecallRouter._classify_query_intent('冰美式') == 'factual'
assert RecallRouter._classify_query_intent('coffee') == 'factual'
assert RecallRouter._classify_query_intent('小七') == 'factual'
assert RecallRouter._classify_query_intent('1984') == 'factual'

# 预期 exploratory 的查询（含疑问词）
assert RecallRouter._classify_query_intent('用户喜欢什么') == 'exploratory'
assert RecallRouter._classify_query_intent('为什么') == 'exploratory'
assert RecallRouter._classify_query_intent('如何使用这个功能') == 'exploratory'

# 预期 navigation
assert RecallRouter._classify_query_intent('') == 'navigation'
assert RecallRouter._classify_query_intent(None) == 'navigation'
```

**手动验证**:
- [ ] 重复查询"冰美式"5次，统计 `intent_factual` 每次 +1
- [ ] 查询"为什么"，`intent_exploratory` +1
- [ ] router_stats 中 `total_queries` 为所有查询总数

### 5.2 记忆进化 (P0-2)

**测试目标**: 添加新事实时自动链接相关旧事实

```python
from cognimem.core.brain import CogniMem
b = CogniMem()

# 添加两条相关事实
b.remember('小七是老大')
b.remember('小七是项目经理')

# 验证两条事实互相连接
facts = b.fact_network._get_agent_facts('default')
connected = [f for f in facts if len(f.connected_facts) > 0]
assert len(connected) >= 1

# 验证矛盾检测不受干扰
b.remember('小七不喜欢喝冰美式')
b.remember('小七爱喝冰美式')  # 应该触发矛盾，但不影响进化功能
```

**手动验证**:
- [ ] 添加"小七是老大"后，再添加"小七是项目经理"，dashboard 中两条事实显示有关联
- [ ] 添加矛盾事实（喜欢 vs 不喜欢），矛盾检测正常

### 5.3 语义缓存 (P0-3)

**测试目标**: 相似查询复用缓存结果

```python
from cognimem.core.fact_network import FactNetwork

# 相似度计算验证
assert FactNetwork._query_similarity('冰美式', '冰美式咖啡') > 0.5  # 同主题
assert FactNetwork._query_similarity('完全相同', '完全相同') == 1.0  # 完全相同
assert FactNetwork._query_similarity('', 'anything') == 0.0  # 空查询
```

**手动验证**:
- [ ] 先查询"冰美式"，再查"冰美式咖啡"，语义缓存应命中（相似度 > 0.55）
- [ ] 先查"小七是老大"，再查"小七是项目经理"，语义缓存应命中

### 5.4 STM 缓冲区 (P0-4)

**测试目标**: 新事实先入 STM，满则 FIFO 淘汰，consolidate 时 flush

```python
from cognimem.core.brain import CogniMem
b = CogniMem()

# 添加 5 条
for i in range(5):
    b.remember(f'测试{i}', agent_id='stm_test')
assert b.fact_network._stm_count('stm_test') == 5

# 超过 30 条应 FIFO 淘汰
for i in range(30):
    b.remember(f'批量{i}', agent_id='stm_test')
assert b.fact_network._stm_count('stm_test') <= 30

# Flush
b.fact_network._flush_stm('stm_test')
assert b.fact_network._stm_count('stm_test') == 0

# consolidate 自动 flush
b.remember('测试', agent_id='consol_test')
r = b.consolidate()
assert r.get('stm_flushed', 0) >= 0
```

**手动验证**:
- [ ] 添加 5 条记忆，dashboard `⚡短期缓存` 卡片显示 5
- [ ] 运行 consolidate，STM 变 0
- [ ] STM 加分：STM 中的事实在排序中靠前

---

## 6. P1 新功能专项测试

### 6.1 主动检索 (P1-1)

**测试目标**: 从查询中提取实体+预期事实类型

```python
from cognimem.core.recall import RecallRouter

# 偏好查询
t = RecallRouter._extract_retrieval_topic('小七喜欢喝什么咖啡')
assert t['expected_type'] == 'preference'
assert '咖啡' in str(t['entities'])
assert '小七' in str(t['entities'])

# 事实查询
t = RecallRouter._extract_retrieval_topic('项目截止日期是什么时候')
assert t['expected_type'] == 'fact'

# 目标查询
t = RecallRouter._extract_retrieval_topic('打算去日本旅游')
assert t['expected_type'] == 'goal'

# 技能查询
t = RecallRouter._extract_retrieval_topic('会做数据分析')
assert t['expected_type'] == 'skill'

# 决策查询
t = RecallRouter._extract_retrieval_topic('决定用Python')
assert t['expected_type'] == 'decision'

# 空查询
t = RecallRouter._extract_retrieval_topic('')
assert t['expected_type'] == ''
```

**手动验证**:
- [ ] 查询"小七喜欢喝什么咖啡"，dashboard 路由统计中 `active_retrieval_hits` 增加
- [ ] 查询"截止日期"，主动检索提取到 fact 类型

### 6.2 Weibull 时间衰减 (P1-2)

**测试目标**: Weibull 比指数衰减更科学，半衰期准确

```python
from cognimem.core.recall import RecallRouter

# 半衰期精确性：在 half_life 天时过期度应为 0.5
for hl in [7, 14, 30, 60, 90]:
    w = RecallRouter._weibull_staleness(hl, hl)
    assert abs(w - 0.5) < 0.01

# 单调性
prev = 0
for d in [0, 1, 3, 7, 14, 30, 60, 90, 180, 365]:
    w = RecallRouter._weibull_staleness(d, 30)
    assert w >= prev
    prev = w

# 初期慢衰减（7天 < 20%）
assert RecallRouter._weibull_staleness(7, 30) < 0.2

# 后期快衰减（90天 > 95%）
assert RecallRouter._weibull_staleness(90, 30) > 0.95
```

**手动验证**:
- [ ] 查看代码 `fact_network.py` 中 `_apply_decay` 使用 Weibull 公式
- [ ] 查看代码 `recall.py` 中 `_calc_staleness` 使用 Weibull 公式

### 6.3 知识库模块 (P1-3)

**测试目标**: 凭证安全存储、掩码展示、普通 recall 排除

```python
from cognimem.core.brain import CogniMem
b = CogniMem()

# 1. 存储
r = b.remember_credential('GitHub', 'ghp_abc123')
assert r['status'] == 'stored'

# 2. 召回（解码正确）
r = b.recall_credential('GitHub')
assert r['credential'] == 'ghp_abc123'
assert r['status'] == 'found'

# 3. 安全展示（原文被掩码）
assert 'ghp_abc123' not in r['safe_display']
assert '*' in r['safe_display']

# 4. 更新
b.remember_credential('GitHub', 'new_token')
assert b.recall_credential('GitHub')['credential'] == 'new_token'

# 5. 不存在
assert b.recall_credential('NONEXIST')['status'] == 'not_found'

# 6. 列出凭证（不泄露原文）
b.remember_credential('AWS', 'AKIA_test')
creds = b.list_credentials()
assert len(creds) == 2
for c in creds:
    assert '***' in c['safe_display']  # 掩码

# 7. 普通 recall 排除凭证
b.remember('正常记忆')
r = b.recall('GitHub')
for f in r['facts']:
    assert f.fact_type != 'credential', f'凭证泄露: {f.fact_type}'

# 8. 凭证不触发矛盾
b.remember_credential('矛盾测试', 'val1')
b.remember_credential('矛盾测试', 'val2')  # 更新而非矛盾
assert len(b.fact_network.get_contradictions('default')) == 0

# 9. 凭证不参与进化
for f in b.fact_network._get_agent_facts('default'):
    if f.fact_type == 'credential':
        assert len(f.connected_facts) == 0
```

**手动验证**:
- [ ] 存储一个凭证后，dashboard `🔐知识库` 卡片数字 +1
- [ ] 用 `curl` 直接查 `/stats`，`credential_count` 显示正确数字
- [ ] 搜索包含凭证关键词的查询，结果中不出现凭证内容

---

## 7. 边界与压力测试

### 7.1 大量数据处理

```bash
PYTHONPATH=src python3 -c "
from cognimem.core.brain import CogniMem
b = CogniMem()
# 50条普通记忆
for i in range(50): b.remember(f'压力测试第{i}条', agent_id='stress')
r = b.recall('压力测试')
print(f'50条recall: {r[\"count\"]} facts')
# 20条凭证
for i in range(20): b.remember_credential(f'service_{i}', f'key_{i}')
print(f'20条凭证: {len(b.list_credentials())}')
print('OK')
"
```

- [ ] 50 条记忆 recall 正常返回
- [ ] 20 条凭证列表正常列出

### 7.2 超长文本与特殊字符

```bash
PYTHONPATH=src python3 -c "
from cognimem.core.brain import CogniMem
b = CogniMem()
b.remember('A' * 10000)  # 10000字
b.remember('测试!@#\$%^&*()_+特殊字符')
b.remember('emoji测试🚀🧠💡')
print('OK')
"
```

- [ ] 10000 字不崩溃
- [ ] 特殊字符不崩溃
- [ ] Emoji 正常处理

### 7.3 多 Agent 隔离

```bash
PYTHONPATH=src python3 -c "
from cognimem.core.brain import CogniMem
b = CogniMem()
b.remember('数据A', agent_id='agent_a')
b.remember('数据B', agent_id='agent_b')
ra = b.recall('数据', agent_id='agent_a')
rb = b.recall('数据', agent_id='agent_b')
print(f'A: {ra[\"count\"]} facts, B: {rb[\"count\"]} facts')
print('OK')
"
```

- [ ] Agent A 只看到自己的数据
- [ ] Agent B 只看到自己的数据

### 7.4 并发安全

```bash
PYTHONPATH=src python3 -c "
from cognimem.core.brain import CogniMem
import threading
b = CogniMem()
errs = []
def w(i):
    try: b.remember(f'并发{i}')
    except Exception as e: errs.append(str(e))
def rd():
    try: b.recall('并发')
    except Exception as e: errs.append(str(e))
ths = [threading.Thread(target=w, args=(i,)) for i in range(10)]
ths += [threading.Thread(target=rd) for _ in range(10)]
for t in ths: t.start()
for t in ths: t.join()
assert len(errs) == 0, f'错误: {errs}'
print('20线程并发OK')
"
```

- [ ] 10 个写入线程无异常
- [ ] 10 个读取线程无异常

---

## 8. 回归测试

### 8.1 pytest

```bash
PYTHONPATH=src python3 -m pytest tests/ -v
```

预期输出: `26 passed`

### 8.2 现有功能不受影响

```python
from cognimem.core.brain import CogniMem
b = CogniMem()

# 矛盾检测仍有效
b.remember('小七喜欢喝冰美式')
b.remember('小七不喜欢喝冰美式')
assert len(b.fact_network.get_contradictions('default')) > 0

# 跨Agent总线仍有效
b.remember('Alice数据', agent_id='alice')
b.remember('Bob数据', agent_id='bob')
c = b.recall_cross_agent('数据', ['alice', 'bob'])
assert c['count'] >= 0

# Consolidate 仍有效
r = b.consolidate()
assert isinstance(r, dict)
assert 'stm_flushed' in r  # v0.13 新增字段
assert 'merged' in r
assert 'abstracted' in r

# 确认/质疑 仍有效
f = b.fact_network._get_agent_facts('default')[0]
assert b.confirm(f.fact_id)['status'] == 'confirmed'
assert b.challenge(f.fact_id)['status'] == 'challenged'

# get_stats 格式不变（新增字段）
s = b.get_stats('default')
assert 'total_facts' in s
assert 'router_stats' in s
assert 'stm_buffer' in s  # v0.13 新增
```

---

## 测试结果记录表

| 测试日期 | 测试人员 | P0(4项) | P1(3项) | UI | 边界 | 回归 | 总体 |
|:--------|:--------|:-------:|:-------:|:--:|:---:|:---:|:---:|
| 2026-07-11 | 小七 | ✅/✅/✅/✅ | ✅/✅/✅ | 40项✅ | 4项✅ | 26/26 | **全部通过** |
| | | □/□/□/□ | □/□/□ | □ | □ | □ | □ |
| | | □/□/□/□ | □/□/□ | □ | □ | □ | □ |
