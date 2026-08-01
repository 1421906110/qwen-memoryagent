# CogniMem v0.27 架构总图

> 生成时间：2026-07-30 | 版本：v0.27 | 模型：DeepSeek v4-flash | 健康分：95

---

## 一、整体分层架构

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                         🧑 用户 / API 客户端                         │
 │                     HTTP / SSE / WebSocket                         │
 └───────────────────────────┬─────────────────────────────────────────┘
                             │
 ┌───────────────────────────▼─────────────────────────────────────────┐
 │                       🌐 Web 层 (main.py)                          │
 │                                                                     │
 │  FastAPI (uvicorn :8000)  +  Dashboard (HTML)                      │
 │                                                                     │
 │  ├─ GET  /health          ─ 健康检测 (95分)                         │
 │  ├─ POST /chat            ─ 聊天主入口 ★                           │
 │  ├─ POST /chat/stream     ─ 流式聊天                               │
 │  ├─ POST /remember        ─ 存记忆                                 │
 │  ├─ POST /recall          ─ 召记忆                                 │
 │  ├─ GET  /stats           ─ 统计                                   │
 │  ├─ GET  /memories        ─ 记忆列表                               │
 │  ├─ GET  /preferences     ─ 偏好查询                               │
 │  ├─ GET  /agents          ─ Agent 列表                             │
 │  ├─ POST /groom           ─ 记忆维护                               │
 │  ├─ POST /consolidate     ─ 整合记忆                               │
 │  ├─ POST /confirm         ─ 确认事实                               │
 │  ├─ POST /challenge       ─ 质疑事实                               │
 │  └─ GET  /capabilities🆕 ─ 能力目录 (v0.27)                        │
 │                                                                     │
 │  ┌────────────────────────────────────────────────────────────┐     │
 │  │  🧊 冻结快照 (v0.27 Phase 0)                               │     │
 │  │                                                             │     │
 │  │  cogni._snapshot[agent_id] = {                              │     │
 │  │    'system': system,     # 首条构建后冻结                    │     │
 │  │    'agent_id': agent_id,                                    │     │
 │  │    'created_at': timestamp                                  │     │
 │  │  }                                                          │     │
 │  │                                                             │     │
 │  │  后续请求 → frozen_system → 跳过 recall → prefix cache 命中 │     │
 │  └────────────────────────────────────────────────────────────┘     │
 └───────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │                ⚙️ Agent 引擎层 (memory_agent/agent/)                │
 │                                                                     │
 │  ┌────────────────────┐    ┌──────────────────────────────────┐     │
 │  │  🟢 简单路径        │    │  🔵 复杂路径 (TurnEngine)        │     │
 │  │  (main.py 内联)    │    │  (engine.py)                     │     │
 │  │                     │    │                                  │     │
 │  │  msg < 120 字符     │    │  有工具调用/URL/ACTION_WORDS     │     │
 │  │  无 ACTION_WORDS    │    │  → 进入 agent 循环               │     │
 │  │  无 URL             │    │                                  │     │
 │  │                     │    │  ├─ max_iterations=8             │     │
 │  │  1次 LLM 调用       │    │  ├─ 权限检查 (RiskClass)         │     │
 │  │  (或1次工具+1次合成) │    │  ├─ 工具缓存 LRU                │     │
 │  │                     │    │  ├─ 并发只读工具执行              │     │
 │  │  3-6s               │    │  └─ 10-30s                      │     │
 │  └────────┬───────────┘    └──────────┬───────────────────────┘     │
 │           │                           │                              │
 │           └───────────┬───────────────┘                              │
 │                       │                                               │
 │           ┌───────────▼───────────┐                                   │
 │           │  🛡️ XML 清洗双重保护    │                                   │
 │           │                     │                                   │
 │           │  engine._clean_xml  │  ← v0.27: 统一函数                │
 │           │  main._clean_tool_  │  ← v0.24: 双重保护                │
 │           │     call_xml        │                                   │
 │           │                     │                                   │
 │           │  正则清洗:           │                                   │
 │           │  <tool_calls>...</>  │                                   │
 │           │  <invoke name="..."> │                                   │
 │           └─────────────────────┘                                   │
 │                                                                     │
 │  ┌──────────────────────────────────────────────────────────────┐   │
 │  │  📦 工具系统                                               │   │
 │  │                                                             │   │
 │  │  registry.py  ← 注册 19+ 工具 (read/write/shell/search)     │   │
 │  │  risk.py      ← RiskClass (READ/WRITE_LOCAL/EXEC/EXTERNAL)  │   │
 │  │  tools.py     ← 工具实现                                    │   │
 │  │  permissions.py ← 三级模式 DISCUSS/INTERACTIVE/AUTO         │   │
 │  │  validator.py ← 安全检查                                    │   │
 │  └─────────────────────────────────────────────────────────────┘   │
 │                                                                     │
 │  ┌──────────────────────────────────────────────────────────────┐   │
 │  │  📋 Capability 能力目录🆕 (catalog.py, v0.27 Phase 3)        │   │
 │  │                                                             │   │
 │  │  6 种能力，按 requires 筛选：                                │   │
 │  │  ┌──────────┬──────────────┬──────────┬──────────┐          │   │
 │  │  │ capability  │ tools  │ requires   │ status   │          │   │
 │  │  ├──────────┼──────────────┼──────────┼──────────┤          │   │
 │  │  │ memory   │ recall/reme │ cogni     │ ✅ ECS   │          │   │
 │  │  │          │ mber/status │           │          │          │   │
 │  │  │ web      │ search      │ (none)    │ ✅ ECS   │          │   │
 │  │  │          │ /fetch      │           │          │          │   │
 │  │  │ filesys  │ read/write/ │ workspace │ ❌ no ctx│          │   │
 │  │  │          │ edit/grep   │           │          │          │   │
 │  │  │ shell    │ shell       │ executor  │ ❌ no ctx│          │   │
 │  │  │ todo     │ todo        │ workspace │ ❌ no ctx│          │   │
 │  │  │ code     │ search/glob │ workspace │ ❌ no ctx│          │   │
 │  │  └──────────┴──────────────┴──────────┴──────────┘          │   │
 │  │                                                             │   │
 │  │  expand(ids, ctx) → 自动跳过不满足 requires 的能力           │   │
 │  │  不破坏现有 registry，只引用工具名                           │   │
 │  └──────────────────────────────────────────────────────────────┘   │
 │                                                                     │
 │  ┌──────────────────────────────────────────────────────────────┐   │
 │  │  📝 上下文压缩🆕 (v0.27 Phase 2)                             │   │
 │  │                                                             │   │
 │  │  _prune_messages (engine.py + __init__.py)                  │   │
 │  │                                                             │   │
 │  │  触发: token_count > 24000                                  │   │
 │  │  策略: 保留 system + 首条 user + 最近 8 轮                   │   │
 │  │        补全 tool_calls 配对                                  │   │
 │  │        被裁剪消息 → LLM 1次摘要 → 插入 [对话摘要]            │   │
 │  └──────────────────────────────────────────────────────────────┘   │
 │                                                                     │
 │  ┌──────────────────────────────────────────────────────────────┐   │
 │  │  ⏰ 后台调度 (scheduler.py)                                  │   │
 │  │  每 5min → groom / 每 30min → consolidate+矛盾解析           │   │
 │  └──────────────────────────────────────────────────────────────┘   │
 └───────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │                   🧠 核心记忆引擎 (cognimem/core/)                   │
 │                                                                     │
 │  ┌──────────────────────────────────────────────────────────────┐   │
 │  │  brain.py — CogniMem 主类                                    │   │
 │  │                                                             │   │
 │  │  ├─ remember()       → 存事实 (去重+矛盾检测)                │   │
 │  │  ├─ recall()         → BM25 + 关键词回搜 evidence            │   │
 │  │  ├─ _snapshot[ ]🆕   → 快照字典 (v0.27 Phase 0)             │   │
 │  │  ├─ get_snapshot()   → 读快照                                │   │
 │  │  └─ has_snapshot()   → 快照存在判断                          │   │
 │  └─────────────────────────────────────────────────────────────┘   │
 │                                                                     │
 │  ┌────────────────────┐   ┌────────────────────┐                   │
 │  │  extractor.py      │   │  fact_network.py   │                   │
 │  │  ─────────────────  │   │  ─────────────────  │                   │
 │  │  规则提取 (0 Token) │   │  事实存储/去重      │                   │
 │  │  68+ 谓词          │   │  矛盾检测           │                   │
 │  │  20+ 决策动词      │   │  置信度管理         │                   │
 │  │  程度副词支持      │   │  consolidate 整合   │                   │
 │  │  垃圾三元组过滤    │   │  证据链管理         │                   │
 │  └────────────────────┘   └────────────────────┘                   │
 │                                                                     │
 │  ┌────────────────────┐   ┌────────────────────┐                   │
 │  │  recall.py         │   │  models.py         │                   │
 │  │  ─────────────────  │   │  ─────────────────  │                   │
 │  │  BM25 召回         │   │  FactTriple 数据类  │                   │
 │  │  中文双字拆分      │   │  SPO 结构定义       │                   │
 │  │  agent_id 过滤     │   │  序列化             │                   │
 │  │  置信度排序        │   │                     │                   │
 │  └────────────────────┘   └────────────────────┘                   │
 │                                                                     │
 │  ┌────────────────────┐   ┌────────────────────┐                   │
 │  │  llm_extractor.py  │   │  sentiment.py      │                   │
 │  │  LLM 提取 (兜底)   │   │  情感分析           │                   │
 │  └────────────────────┘   └────────────────────┘                   │
 └───────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │                    💾 存储层 (memory_agent/storage/)                │
 │                                                                     │
 │  ┌────────────────────┐   ┌────────────────────┐                   │
 │  │  SQLiteStore       │   │  PostgreSQL         │                   │
 │  │  (本地开发)        │   │  (ECS 生产)         │                   │
 │  │                    │   │                     │                   │
 │  │  facts 表 (SPO)    │   │  facts 表 (SPO)     │                   │
 │  │  FTS 全文索引      │   │  FTS 全文索引       │                   │
 │  │  evidence 列搜索   │   │  evidence 列搜索    │                   │
 │  └────────────────────┘   └────────────────────┘                   │
 └─────────────────────────────────────────────────────────────────────┘
```

---

## 二、请求处理流程（POST /chat）

```
用户输入
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. 路由决策                                                    │
│                                                                 │
│  msg < 120 字符 && 无 ACTION_WORDS && 无 URL && 非"继续"       │
│       │                      │                                  │
│       ✅ 是                  ❌ 否                               │
│       ▼                      ▼                                  │
│   🟢 简单路径             🔵 Agent 路径 (TurnEngine)            │
└─────────────────────────────────────────────────────────────────┘
         │                          │
         ▼                          ▼
┌──────────────────┐    ┌──────────────────────────────────────┐
│ 2. _build_context │    │ 2. _build_context                   │
│                   │    │                                      │
│  frozen_system?   │    │  frozen_system?                     │
│  ├─ ✅ → system = │    │  ├─ ✅ → system = frozen_system     │
│  │   frozen_system│    │  │   (跳过 recall)                  │
│  │   (跳过 recall)│    │  └─ ❌ → cogni.recall()             │
│  └─ ❌ → cogni.   │    │          + 关键词回退                │
│       recall()    │    │          + <memory-context> 围栏     │
│       + 关键词回退 │    │                                      │
│       + <memory-  │    │  首条 → snapshot 冻结                │
│       context>    │    │                                      │
│       围栏        │    │                                      │
│                   │    │                                      │
│  首条 → snapshot  │    │                                      │
│  冻结             │    │                                      │
└──────────────────┘    └──────────┬───────────────────────────┘
         │                         │
         ▼                         ▼
┌──────────────────┐    ┌──────────────────────────────────────┐
│ 3. LLM 调用      │    │ 3. TurnEngine.turn()                 │
│                   │    │    (最多 8 轮)                       │
│  chat_completion │    │                                      │
│  + 只读工具      │    │  循环:                              │
│                   │    │    ├─ LLM.complete()                 │
│  有 tool_calls?   │    │    ├─ 有 tool_calls?                 │
│  ├─ ✅ 执行工具   │    │    │  ├─ 权限检查 (RiskClass)       │
│  │   → 合成回复   │    │    │  ├─ 缓存命中? → 0 Token        │
│  └─ ❌ 直接返回   │    │    │  ├─ 并发/顺序执行               │
│                   │    │    │  └─ 追加结果 → 回 LLM           │
│  3-6s             │    │    └─ 无 → 提前退出                  │
│                   │    │                                      │
│                   │    │  _prune_messages 🆕                  │
│                   │    │  (token > 24000 → 摘要裁剪)         │
│                   │    │                                      │
│                   │    │  10-30s                              │
└────────┬─────────┘    └──────────┬───────────────────────────┘
         │                         │
         └──────────┬──────────────┘
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. XML 清洗 双重保护 🆕                                        │
│                                                                 │
│  简单路径: main._clean_tool_call_xml()                         │
│  Agent路径: engine._clean_xml() + main._clean_tool_call_xml()   │
│                                                                 │
│  正则:                                                          │
│  r'<tool_calls>.*?</tool_calls>'  (re.DOTALL)                  │
│  r'<invoke name=".*?>.*?</invoke>' (re.DOTALL)                  │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. 返回响应                                                    │
│                                                                 │
│  {                                                              │
│    "reply": "清除了的文本",                                     │
│    "tools_called": N,                                           │
│    "iterations": N,                                             │
│    "memories_used": N                                           │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、冻结快照生命周期（v0.27 Phase 0 核心创新）

```
首次请求 ───────────────────────────────────────────────────────────────
                                                                        
用户: "你好"                                                             
    │                                                                    
    ▼                                                                    
_build_context(user_message="你好", frozen_system=None)                  
    │                                                                    
    ├─ cogni.recall("你好")     ← 走 recall（BM25 + 关键词回退）       
    ├─ 构建 system = _BASE_SYSTEM_PROMPT + <memory-context> + 对话历史    
    ├─ LLM 回复 "你好！我是小明... "                                      
    │                                                                    
    ▼                                                                    
cogni._snapshot["default"] = {                                          
    'system': system,        ← 冻结首次构建的完整 system prompt          
    'agent_id': 'default',                                              
    'created_at': 1732812345.0                                           
}                                                                        
    │                                                                    
    ▼                                                                    
响应 → 用户                                                              
                                                                        
─────────────────────────────────────────────────────────────────────────

后续请求 ───────────────────────────────────────────────────────────────

用户: "你还记得上次我说了什么吗？"                                        
    │                                                                    
    ▼                                                                    
_build_context(user_message="...", frozen_system="<冻结的system>")       
    │                                                                    
    ├─ 直接 system = frozen_system  ← skip recall！ 省 200-500ms         
    ├─ 不加新 <memory-context>     ← 快照已包含全部记忆                  
    ├─ 只加对话历史                                                      
    ├─ 屏蔽记忆相关工具 (读/写/诊断/遗忘)                                
    │                                                                    
    ▼                                                                    
LLM 收到 ↙                                                                
<system>                                                                  
你是小明，带长期记忆的 AI 助手。                                          
...                                                                       
<memory-context>         ← 快照自带的记忆                                 
📋 你记得关于用户的以下信息：                                              
- (小明, 是, AI助手)                                                      
- (用户, 是, 老大)                                                        
...                                                                       
</memory-context>                                                         
</system>                                                                 
    │                                                                    
    ▼                                                                    
LLM 回复（基于快照信息，自然衔接）                                        
prefix cache 命中 → 更快                                                 
                                                                        
─────────────────────────────────────────────────────────────────────────

效果:
  - 每轮省 recall BM25 + 关键词回退 + SQL 查询 ≈ 2-5ms
  - prefix cache 从不命中 → 首轮后100%命中
  - 无重复工具调用覆盖记忆系统
  - 围栏自然区别"历史记忆" vs "本轮新输入"
```

---

## 四、v0.27 5大 Phase 文件改动

| Phase | 改了什么 | 文件 | 设计原则 |
|-------|---------|------|---------|
| **Phase 0** | `frozen_system` 参数 + `_snapshot` 字典 | `main.py:707-722` | 增量参数，不破现有逻辑 |
| 冻结快照 | 快照存储（main.py层） | `main.py:1397-1404` | 避免brain→main循环import |
| | `get_snapshot/has_snapshot` 方法 | `brain.py:65-111` | 纯读方法，无副作用 |
| **Phase 1A** | `<memory-context>` 围栏标签 | `main.py:892-903` | 注入时包标签，不改格式 |
| 上下文围栏 | 工具屏蔽 (快照时排除诊断工具) | `main.py:125-137` | 条件参数，不破无快照场景 |
| **Phase 1B** | System prompt "🧠 记忆" 指导语 | `__init__.py:50-53` | 纯文本提示，不改逻辑 |
| LLM自主写入 | 引导 LLM 调 `memory_remember` | | 对应 Hermes LLM驱动写入 |
| **Phase 2** | `_prune_messages` LLM 摘要 | `engine.py:532-598` | 保留最近8轮+摘要，不丢核心 |
| 上下文压缩 | 超过24000 token触发 | | LLM调用仅1次，非逐轮 |
| **Phase 3** | Capability 数据类 + CATALOG | `catalog.py`(新文件) | 增量层，不改变registry |
| 能力目录 | `/capabilities` API端点 | `main.py:2130-2147` | 只读端点 |
| | AgentContext workspace/executor | `__init__.py:204-205` | 可选字段 |
| **XML清洗** | `_clean_xml()` 统一函数 | `engine.py:49-55` | 4条返回路径全保护 |
| 全局修复 | main.py双重保护保留 | `main.py:1035-1049` | 第二道防线 |

---

## 五、v0.27 数据流（谁调谁）

```
main.py (FastAPI)
    │
    ├─ _build_context()                    ← 构建 system + 消息
    │    ├─ cogni.recall()                 ← 图谱召回 (无快照时)
    │    │    ├─ fact_network.query()       ← BM25 + SQL
    │    │    └─ recall.combine()          ← 融合排序
    │    └─ <memory-context> 包裹          ← 围栏注入
    │
    ├─ llm.chat_completion() / chat()      ← 简单路径
    │    └─ tool_registry.execute()        ← 只读工具执行
    │
    ├─ TurnEngine.turn()                  ← Agent 路径
    │    ├─ _prune_messages()              ← 上下文压缩 🆕
    │    ├─ self.llm.complete()            ← LLM 调用 (循环)
    │    ├─ _check_permission()            ← 权限检查
    │    ├─ tool_registry.execute()        ← 工具执行
    │    │    └─ read/write/shell/search   ← 19+工具
    │    └─ _clean_xml()                   ← XML清洗 🆕
    │
    ├─ _clean_tool_call_xml()              ← 第二道清洗
    │
    ├─ cogni._snapshot[agent_id] = {...}   ← 快照冻结
    │
    └─ _maybe_store_memory()               ← 对话后存记忆
         └─ cogni.remember()
              ├─ extractor.extract()       ← 规则提取 (0 Token)
              ├─ llm_extractor.extract()   ← LLM兜底提取
              └─ fact_network.store()      ← 去重+矛盾+存储
```

---

## 六、参考来源标注

```
 Hermes ─────────────────── OpenWorker ──────────────── Claude ──
                           │
    ├─ 冻结快照 🆕         ├─ Capability 目录 🆕      ├─ 对话即记忆
    │  (首条构建后跳过     │  (catalog.py：能力       │  (不自动提取
    │  后续 recall)        │   声明+requires筛选)    │   额外记忆)
    │                      │                          │
    ├─ <memory-context> 🆕 ├─ _prune_messages 🆕      │
    │  (围栏标签区分记忆   │  (上下文压缩:摘要+       │
    │    vs 新输入)        │   保留最近N轮)            │
    │                      │                          │
    ├─ LLM驱动写入 🆕     ├─ RiskClass (align 参考)   │
    │  (system prompt 引导 │  已在 v0.22 实现)         │
    │  调 memory_remember) │                          │
    │                      └─ Standing Rules 参考      │
    └─ 记忆上下文翻转       (system prompt 铁律)       │
       (Hermes 的                                                       
        memory-context
        fencing pattern)
```

---

> **五更符合度：**
> - 🏗️ **地基更牢**：Capability增量设计不破坏registry；快照不改变recall逻辑；XML清洗统一4路径
> - 💰 **更省Token**：冻结快照让后续请求跳过 recall；prefix cache 从不命中→100%命中
> - ⚡ **更省资源**：简单路径3-6s；TurnEngine 10-30s（原Agent.chat 40-50s）
> - 🧠 **更智能**：LLM自主写入+上下文摘要；Capability按上下文智能过滤
> - 🚀 **算法创新**：Hermes+OpenWorker+Claude三家合理综合，不做照搬
