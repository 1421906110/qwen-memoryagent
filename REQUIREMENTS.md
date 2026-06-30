# 🧠 MemoryAgent — Hackathon 提交需求清单

> **比赛：** Global AI Hackathon Series with QwenCloud
> **赛道：** Track 1 — MemoryAgent（持久化认知存储）
> **奖金：** $7,000 现金 + $3,000 云代金券
> **截止：** 2026-07-09
> **提交平台：** Devpost

---

## 1. 📋 硬性提交要求

| # | 要求 | 说明 | 当前状态 | 负责人 |
|---|------|------|---------|--------|
| 1 | **公开代码仓库** | GitHub 公开仓库，包含全部源码 | ✅ 已有 | 小七 |
| 2 | **开源协议文件** | 仓库根目录必须包含 LICENSE（如 MIT, Apache 2.0） | ❌ 缺失 | 小七 |
| 3 | **阿里云部署证明** | 项目必须部署在阿里云上，提交时提供可访问 URL | ❌ 未部署 | 老大（注册） |
| 4 | **架构图** | 系统设计可视化图（嵌入 README 或单独提交） | ❌ 未画 | 小七 |
| 5 | **演示视频** | 3-5 分钟视频，展示项目功能 | ❌ 未录 | 老大 + 小七 |
| 6 | **项目描述文本** | Devpost 上填写项目说明 | ❌ 未写 | 小七 |
| 7 | **必须使用 Qwen 模型** | 调用 QwenCloud（DashScope）的 qwen-plus/qwen-max 等模型 | ❌ 目前用 DeepSeek | 老大（注册后） |
| 8 | **加入 QwenCloud Discord** | 参赛者需加入官方 Discord | ❌ 未加入 | 老大 |

---

## 2. 🎯 赛道核心要求（MemoryAgent）

### 必须实现的功能

> 官方描述：*Build an autonomous agent with persistent memory capable of accumulating experience, tracking user preferences, and making accurate decisions across multi-turn, cross-session interactions.*

| # | 要求 | 说明 | 当前状态 |
|---|------|------|---------|
| 1 | **跨会话记忆持久化** | Session A 记住的，Session B 能查到 | ✅ 已实现 |
| 2 | **用户偏好追踪** | 从对话中提取并持久化用户偏好 | ✅ 已实现（偏好演变链） |
| 3 | **高效向量存储/检索** | embedding + 相似度搜索 | ⚠️ FTS5 已可用，向量需 Qwen embedding |
| 4 | **智能遗忘机制** | 旧记忆自动衰减/清理 | ✅ Ebbinghaus 衰减 + Groom |
| 5 | **上下文窗口优化** | 在 token 预算内选最相关的记忆 | ✅ 已实现 |

### 加分项

| # | 要求 | 说明 | 当前状态 |
|---|------|------|---------|
| 1 | 记忆冲突解决 | 新事实覆盖旧事实 | ✅ 3 层检测 |
| 2 | 1M Token 长上下文 | qwen-max-longcontext | ✅ 代码已写，需 Qwen Key 验证 |
| 3 | 可视化界面 | 非强制但推荐 | ✅ Web UI 已上线 |
| 4 | 记忆关系图谱 | 展示记忆之间的关联 | ❌ 未实现 |
| 5 | 评测数据 | 召回率/冲突检测准确率 | ❌ 未做 |

---

## 3. 📅 冲刺排期（修正版）

| 天数 | 日期 | 任务 | 说明 | 依赖 |
|------|------|------|------|------|
| Day 1 | 7/1 ✅ | **Web UI** | 聊天界面 + 仪表盘 | 已完成 |
| **Day 2** | **7/2** | **🔴 阿里云注册 + Qwen Key** | 注册阿里云 → DashScope 开 API → 拿免费代金券 | 老大操作 |
| **Day 3** | **7/3** | **🔴 换 Qwen 模型 + 全链路联调** | 把 DeepSeek 换成 qwen-plus，测试 embedding + chat + 长上下文 | Day 2 |
| **Day 4** | **7/4** | **🔴 部署到阿里云 ECS** | 弹性云服务器部署，配公网 IP | Day 3 |
| Day 5 | 7/5 | **LICENSE + 架构图** | MIT 协议 + Draw.io 架构图 | 无依赖 |
| Day 6 | 7/6 | **Demo 视频录制** | 3-5 分钟录屏 + 剪辑 | Day 4（部署好了再录） |
| Day 7 | 7/7 | **Devpost 提交** | 填写所有信息 + 上传视频 + 提交链接 | Day 5-6 |
| Day 8 | 7/8 | **缓冲日** | 修补 + 最终检查 | — |

---

## 4. 🔴 阻塞项（必须老大操作）

### 4.1 阿里云注册
- 打开 [aliyun.com](https://www.aliyun.com) 注册账号
- 可以用支付宝/淘宝/钉钉/GitHub（你已经绑了 GitHub）
- **免费试用：** 新用户 3 个月免费 ECS + 3 个月免费 API 额度

### 4.2 QwenCloud（DashScope）API Key
- 进入 [DashScope 控制台](https://dashscope.console.aliyun.com/)
- 左侧 **API-KEY 管理** → **创建 API Key**
- 填写免费 hackathon 算力申请（官方说参赛者可申请免费 credits）

### 4.3 加入 QwenCloud Discord
- Devpost 上有 Discord 链接
- 用于接收比赛通知 + 技术支持

### 4.4 部署到阿里云 ECS
- 创建 ECS 实例（免费试用期内免费）
- SSH 登录 → 装 Python → 跑起服务
- 开 8000 端口安全组 → 公网可访问

---

## 5. 🛠 技术栈确认

| 组件 | 开发期 | 提交期 |
|------|--------|--------|
| LLM | DeepSeek-v4-flash（测试用） | **qwen-plus**（替换后） |
| Embedding | FTS5 降级 | **text-embedding-v3/v4**（替换后） |
| 存储 | SQLite | SQLite |
| 框架 | FastAPI + Uvicorn | FastAPI + Uvicorn |
| Web UI | Chart.js + 原生 JS | Chart.js + 原生 JS |
| 部署 | localhost:8000 | **阿里云 ECS 公网 IP** |

---

## 6. ✅ 已完成清单（截至 7/1）

### 核心引擎
- [x] 记忆存储/检索/删除（CRUD）
- [x] FTS5 全文搜索
- [x] Ebbinghaus 置信度衰减
- [x] 冲突检测（Jaccard + Embedding + LLM）
- [x] 偏好学习与演变追踪
- [x] 跨会话持久化
- [x] 多 Agent 隔离
- [x] 智能上下文窗口优化
- [x] 1M Token 长上下文代码（需 Qwen Key 验证）

### API（15 个端点）
- [x] `/remember` — 存储记忆
- [x] `/recall` — 检索记忆
- [x] `/chat` — 记忆增强对话
- [x] `/chat/long` — 长上下文对话
- [x] `/process-transcript` — 批量提取
- [x] `/decay-trace/{id}` — 衰减曲线
- [x] `/decay-analysis` — 衰减分析
- [x] `/preferences` — 偏好列表
- [x] `/preferences/history` — 偏好演变链
- [x] `/groom` — 记忆维护
- [x] `/status` — 状态总览

### Web UI
- [x] 聊天界面（记忆侧边栏实时更新）
- [x] 仪表盘（置信度柱状图 + 衰减曲线 + 记忆明细 + 偏好演变）

### 测试
- [x] 25 个测试全部通过
- [x] Demo 脚本（6 阶段演示）

---

## 7. ⏳ 待办清单

### P0 — 截止前必须完成
- [ ] **阿里云注册 + Qwen API Key**
- [ ] **替换 DeepSeek → Qwen 模型**
- [ ] **部署到阿里云 ECS**
- [ ] **录制 Demo 视频**

### P1 — 重要但不阻塞
- [ ] 添加 LICENSE 文件（MIT）
- [ ] 画架构图（Draw.io）
- [ ] 完善 README
- [ ] 加入 QwenCloud Discord

### P2 — 有更好
- [ ] 记忆关系图谱可视化
- [ ] Benchmark 评测数据
- [ ] 记忆合并/聚合
