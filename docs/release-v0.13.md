# 🧠 CogniMem v0.13 版本发布报告

> **发布日期**: 2026-07-12  
> **标签**: `v0.13`  
> **部署地址**: [http://42.121.253.80:8000](http://42.121.253.80:8000)  
> **仓库**: [1421906110/qwen-memoryagent](https://github.com/1421906110/qwen-memoryagent)

---

## 一、版本概览

v0.13 是 CogniMem 的**稳定性里程碑版本**，在保持核心架构不变的前提下，完成了：

1. **4 项 P0 级核心功能**（零LLM路由/记忆进化/语义缓存/STM缓冲区）
2. **3 项 P1 级高级功能**（主动检索/Weibull衰减/知识库）
3. **6 个 Bug 修复**（含 4 个 cogni 分支缺失 + UUID校验 + /health端点）
4. **完整部署流水线**（阿里云 ECS + systemd 开机自启）
5. **28 项稳定性测试全部通过**

### 质量指标

| 维度 | 结果 |
|:----|:----:|
| 自动化测试 (18项) | ✅ 100% |
| 深度验证 (102项) | ✅ 100% |
| pytest (26项) | ✅ 100% |
| HTTP端点 (13项) | ✅ 100% |
| 稳定性测试 (28项) | ✅ 100% |
| 架构健康度 | ⭐⭐⭐⭐⭐ (96.5分) |

---

## 二、P0 核心功能

### P0-1 零LLM路由 — 不需要LLM也能精确分类查询意图

```python
classify_query_intent('冰美式')       → 'factual'      # 事实查询
classify_query_intent('为什么')         → 'exploratory'  # 探索性查询
classify_query_intent('')              → 'navigation'   # 导航/空查询
classify_query_intent('coffee')        → 'factual'      # 英文也支持
classify_query_intent('如何使用')       → 'exploratory'  # 操作类
```

**意义**: 零LLM开销，毫秒级响应，能扛高并发。只有需要深度语义理解时才调用LLM。

### P0-2 记忆进化 — 相关事实自动链接

存入"小七是老大"和"小七是项目经理"后，系统自动识别两个事实共享同一主体(`小七`)，建立 `connected_facts` 链接。

**特性**:
- 自动关联，不影响矛盾检测
- 链接用于后续推理和上下文构建
- 召回时携带关联事实，提供更丰富上下文

### P0-3 语义缓存 — 相似查询命中缓存

| 缓存查询 | 新查询 | 相似度 |
|:---------|:------|:------:|
| 冰美式 | 冰美式咖啡 | > 0.5 ✅ |
| — | 空字符串 | = 0.0 ✅ |
| 完全相同 | 完全相同 | = 1.0 ✅ |

缓存命中直接返回结果，跳过LLM调用，**响应时间从秒级降至毫秒级**。

### P0-4 STM 缓冲区 — 短期记忆管理

```
存入: 5条 → STM count = 5
FIFO: 35条 → 自动淘汰至 ≤30
Flush: 手动 → 归零
Consolidate: 自动合并STM到长期记忆
```

STM 缓冲区隔离高频短期操作，consolidate 时才批量处理，减少碎片化存储。

---

## 三、P1 高级功能

### P1-1 主动检索 — 按实体类型精确召回

| 输入 | 提取类型 |
|:----|:--------:|
| "小七喜欢喝什么咖啡" | `preference` |
| "项目截止日期是什么时候" | `fact` |
| "打算去日本旅游" | `goal` |
| "会做数据分析" | `skill` |
| (空) | `` (空类型) |

### P1-2 Weibull 衰减 — 更真实的遗忘曲线

采用 Weibull 分布 (`k=1.5`) 模拟人类遗忘：

```
30天半衰期:
  d=0   衰减=0.0000  → 记忆完好
  d=7   衰减=0.0752  → 缓慢开始遗忘  
  d=30  衰减=0.5000  → 恰好一半
  d=90  衰减=0.9727  → 几乎遗忘
  d=180 衰减≈1.0000  → 完全遗忘
```

相比 Ebbinghaus 曲线，Weibull 的优势：
- **初期更慢**（刚记住时不容易忘）
- **后期更快**（过了半衰期后快速衰减）
- **参数灵活**（调整 k 值可适配不同记忆类型）

### P1-3 知识库 — 安全凭证管理

```python
# 存储（自动掩码）
remember_credential('GitHub', 'token_abc')
# → 存储: 'token_abc'
# → 展示: 'tok***abc'  (自动掩码)

# 召回
recall_credential('GitHub')
# → credential: 'token_abc'
# → safe_display: 'tok***abc'

# 更新已有凭证
remember_credential('GitHub', 'new_token')
# → 自动覆盖旧值

# 列表（全部掩码）
list_credentials()
# → [{'service': 'GitHub', 'safe_display': 'tok***abc'}, ...]

# 安全：普通recall排除凭证
recall('GitHub')
# → 结果中不包含 credential 类型的事实
```

**安全设计**: 凭证类型事实在 `recall()` 中自动排除，不会因对话上下文泄露。

---

## 四、Bug 修复详表

| ID | 严重度 | 文件 | 问题 | 修复 |
|:--|:------|:----|:----|:-----|
| BUG-1 | 🔴 严重 | `memory_agent/main.py:982` | `/preferences/history` 在 cogni 模式下 500 | 添加 cogni 分支，从 `fact_network._get_agent_facts()` 按 `fact_type=="preference"` 过滤 |
| BUG-2 | 🔴 中 | `memory_agent/main.py:706` | `/decay-trace/{id}` 500 + 硬编码 `agent_id="*"`永不命中 | 添加 cogni 分支 + 改用 `agent_id` 参数(默认default) |
| BUG-3 | 🟡 中 | `memory_agent/main.py:960` | `/groom` 缺少 cogni 分支 | cogni 分支 → `cogni.consolidate()` |
| BUG-4 | 🟡 中 | `memory_agent/main.py:1012` | `/merge` 缺少 cogni 分支 | cogni 分支 → `cogni.consolidate()` |
| BUG-6 | 🟢 低中 | `cognimem/main.py:227` | 引擎 `/versions/{id}` 非法UUID导致DB崩溃 | 添加 `uuid.UUID()` 格式校验 + try/except，非法返回 400 |
| BUG-8 | 🟢 低 | `cognimem/main.py:170` | 引擎端口 `:8001` 缺少 `/health` | 新增完整健康检查端点(DB/Brain/评分) |

### 修复前后对比

```
修复前: 4个端点 → 500 Internal Server Error
修复后: 全部端点 → 200 OK ✅
```

---

## 五、稳定性测试结果

| 测试场景 | 操作量 | 结果 |
|:--------|:-----:|:----:|
| 单线程循环压力 | 1000次操作 @ 18op/s | ✅ 无错误 |
| 50线程并发 | 2500次混合操作 | ✅ 0死锁/0竞争 |
| 全API混合负载 | 500次12种API交替 | ✅ 无错误 |
| 超长文本 | 10000字A | ✅ 正常处理 |
| 特殊字符 | 12种(含emoji/控制符) | ✅ 全部通过 |
| 重复内容 | 100条完全相同 | ✅ 正常去重 |
| 空值边界 | 空文本/空查询/非法UUID | ✅ 优雅处理 |
| 内存稳定性 | LRU缓存监控 | ✅ 线性增长，无泄漏 |
| HTTP稳定性 | 200次跨15端点 | ✅ 0异常 |
| 知识库稳定性 | 300次凭证循环 | ✅ 无丢失/无泄漏 |
| 长时间运行 | 1000次持续操作 | ✅ 55.3s持续稳定 |

---

## 六、部署架构

```
┌──────────────┐     ┌──────────────────────────────┐
│   用户浏览器   │────▶│  阿里云ECS 42.121.253.80:8000  │
└──────────────┘     │                              │
                     │  ┌──────────────────────────┐│
                     │  │  uvicorn (systemd 管理)   ││
                     │  │  memory_agent.main:app    ││
                     │  │  Python 3.10 + FastAPI    ││
                     │  └──────────┬───────────────┘│
                     │             │                  │
                     │  ┌──────────▼───────────────┐│
                     │  │  CogniMem 引擎 (进程内)    ││
                     │  │  - 事实网络               ││
                     │  │  - Weibull 衰减           ││
                     │  │  - 语义缓存               ││
                     │  │  - 知识库                 ││
                     │  └──────────┬───────────────┘│
                     │             │                  │
                     │  ┌──────────▼───────────────┐│
                     │  │  PostgreSQL 16            ││
                     │  │  cognimem 数据库          ││
                     │  │  7张表: facts/versions/   ││
                     │  │  contradictions/agents... ││
                     │  └──────────────────────────┘│
                     └──────────────────────────────┘
```

### 服务管理

```bash
# 启动/停止/重启
sudo systemctl start|stop|restart cognimem.service

# 查看状态
sudo systemctl status cognimem.service

# 查看日志
tail -f /tmp/qwen-agent3.log

# 开机自启（已配置）
sudo systemctl enable cognimem.service
```

---

## 七、API 端点清单

| 方法 | 路径 | 说明 | 状态 |
|:----|:----|:----|:----:|
| GET | `/` | 服务存活检测 | ✅ |
| GET | `/dashboard` | Web仪表盘 | ✅ |
| GET | `/chat` | 聊天界面 | ✅ |
| GET | `/graph` | 知识图谱 | ✅ |
| GET | `/agents` | Agent管理 | ✅ |
| GET | `/health` | 健康检查+评分 | ✅ |
| GET | `/stats` | 统计数据 | ✅ |
| GET | `/memories` | 记忆列表 | ✅ |
| GET | `/preferences/history` | 偏好演变历史 | ✅ 修复 |
| GET | `/decay-trace/{id}` | 衰减曲线可视化 | ✅ 修复 |
| POST | `/remember` | 存入记忆 | ✅ |
| POST | `/recall` | 召回记忆 | ✅ |
| POST | `/confirm` | 确认事实 ↑置信度 | ✅ |
| POST | `/challenge` | 质疑事实 ↓置信度 | ✅ |
| POST | `/consolidate` | 记忆整合 | ✅ |
| POST | `/groom` | 记忆整理 | ✅ 修复 |
| POST | `/merge` | 记忆合并 | ✅ 修复 |

---

## 八、版本路线图

```
v0.1  ─── 原型/MVP
  │
v0.2  ─── API层/FastAPI端点
  │
v0.3  ─── 记忆引擎核心(三元组/衰减/多级召回)
  │
v0.4  ─── Web UI(聊天/仪表盘/图谱)
  │
v0.5  ─── 冲突检测/置信度/版本追踪
  │
v0.6  ─── Agent四大能力/DeepSeek切换
  │
v0.7  ─── 健康检测/三层压缩/Evidence链
  │
v0.8  ─── 12工具/线程锁/滑动窗口/Dashboard
  │
v0.9  ─── 仪表盘最终定稿
  │
v0.10 ─── PostgreSQL持久化
  │
v0.11 ─── Agent升级版
  │
v0.12 ─── 认知增强版
  │
v0.13 ─── ⭐ 稳定增强版(当前)
          6 Bug修复 + ECS部署 + 28项稳定性测试
```

---

## 九、待改进

| 项目 | 优先级 | 说明 |
|:----|:------|:-----|
| pgvector 安装 | P2 | 向量搜索当前降级为纯文本，安装后恢复语义搜索 |
| Nginx 反向代理 | P2 | 当前直接暴露 :8000，加 nginx 可做 SSL/限流 |
| 超大文本限制 | P2 | 50000字记得超时(>10s)，建议限制 ≤20000字符 |
| 参数命名规范 | P3 | BUG-5/7/9 等 API 风格对齐 |
| Docker 部署 | P3 | 当前用 systemd 裸跑，Docker 化后更易迁移 |

---

## 十、致谢

- **测试报告**: 全面覆盖 197 项测试，定位 6 个真实 Bug
- **稳定性测试**: 28 项场景覆盖边界/并发/长时/HTTP 全链路
- **阿里云 ECS**: 杭州节点，PostgreSQL 16 + Python 3.10

---

*报告生成于 2026-07-12 | 版本 v0.13*
