# CogniMem 终极审计 — 第五轮

**方法**: 绕过测试手册，纯代码级边界测试 + 符号执行推演  
**范围**: GoalContext / MemoryManager / SelfReflector / MemoryGovernor / FrequencyTracker / CogniMemClient / DB层 / triple_key  
**之前未覆盖**: cognimem_client、governance、goal、reflector、modules、llm_extractor

---

## 🔴 P0 — 运行时数据损坏

### Bug X1: triple_key 分隔符冲突 → 事实静默丢失

- **位置**: `src/cognimem/core/models.py:65` — `triple_key` 属性
- **根因**: key 格式为 `agent_id|subject|predicate|object`，用 `|` 连接，但当字段内出现 `|` 时，`split("|")` 无法正确还原

**完整攻击链**:
```
事实A: (张三, 住在, 北京|上海)     → triple_key = "default|张三|住在|北京|上海"
事实B: (张三, 住在|北京, 上海)     → triple_key = "default|张三|住在|北京|上海"  ❌ 碰撞!

store(A) → _find_existing(key) → cache miss → 正常存入
store(B) → _find_existing(key) → cache hit  → 误判为A的重复 → merge到A上 → B丢失
```

- **影响**:
  - `_find_existing()` 返回错误 fact → merge 到不对的 fact
  - `find_by_triple_key()` 用 `split("|", 3)` 还原 → 拿错 subject/predicate/object → DB 查错行
  - DB 中 UNIQUE 约束在 `(agent_id,subject,predicate,object)` 列，但 triple_key 匹配给了错误的事实，DB 约束无法保护

**触发概率**: 低但确实会发生——任何用户输入含 `|`（如 markdown 表格、管道命令、双城地址）

---

## 🟡 P1 — 功能性缺陷

### Bug X2: `should_retry` 仅处理英文错误

- **位置**: `src/memory_agent/agent/goal.py:149`
- **代码**: `no_retry_keywords = ["permission denied", "invalid syntax", "does not exist", "no such file"]`
- **问题**: 中文 locale 系统中系统错误可能是中文，如："权限被拒绝"、"文件不存在"
- **验证**: `should_retry("权限不足")` → True（本应 False）→ 无限重试
- **影响**: 中文环境下，权限错误/文件不存在错误会无限重试

### Bug X3: `should_store` 方法对 user 源始终返回 True

- **位置**: `src/memory_agent/agent/memory_manager.py:131`
- **代码**: `if source == "user": return True  # kept for backward compat`
- **问题**: 方法签名暗示过滤逻辑，但 user 消息全放行
- **影响**: 如果未来有人直接调用此方法（不通过调用方预过滤），所有用户消息都会被存
- **注**: 当前调用方 `_store_important_memories` 已预过滤，所以实际影响小

---

## 🟠 P2 — 数据质量

### Bug X4: ILIKE 通配符未转义

- **位置**: `src/cognimem/core/db.py:386,389,392`
- **代码**: `params["subject"] = f"%{subject}%"`  → SQL: `ILIKE '%with_underscore%'`
- **问题**: 用户搜索 `a_b` 时，`_` 是 SQL 单字符通配符 → 匹配 `aXb`、`aYb` 等
- **同样**: 搜索 `100%` 时 `%` 也是通配符
- **影响**: 搜索结果多出无关匹配，降低精确召回

### Bug X5: `encoding_level` 无校验

- **位置**: `src/cognimem/core/models.py`
- **问题**: 可存入任意值如 `"corrupted"`、`""`
- **影响**: consolidation 基于 `encoding_level` 做决策（如 core 级别跳过 decay），非法值可能导致错误跳过

---

## 🔵 P3 — 设计问题

### Bug X6: `_FAILED_DOMAINS` 模块级无锁
- 已在前轮报告，未修复

---

## ✅ 第五轮新发现确认正常

| 检查 | 结果 |
|------|------|
| GoalContext 状态机 | ✅ 全部正确（含边界：空plan/None current/complete） |
| SelfReflector | ✅ 含空输入 |
| FrequencyTracker | ✅ |
| MemoryGovernor | ✅ |
| CogniMemClient 无参构造 | ✅ |
| DB search_facts 参数化 | ✅ |
| 连接池归还 | ✅ |
| consolidate RLock 安全 | ✅ |
| EvidenceItem 空值 | ✅ |
| confidence_label 全路径 | ✅ |
| __post_init__ 各种边界 | ✅ |
| 空plan advance | ✅ |

---

## 全量汇总（五轮累计）

| 级别 | Bug # | 说明 | 状态 |
|------|-------|------|------|
| P0 | X1 | triple_key `\|` 分隔符冲突 | 🆕 新发现 |
| P1 | X2 | should_retry 仅英文 | 🆕 新发现 |
| P1 | X3 | should_store user=全放行 | 🆕 新发现 |
| P2 | X4 | ILIKE `_`/`%` 未转义 | 🆕 新发现 |
| P2 | X5 | encoding_level 无校验 | 🆕 新发现 |
| P3 | X6 | _FAILED_DOMAINS 无锁 | 已知未修 |

**前四轮 16 个 Bug 全修 ✅**

---

## 修复优先级

```
P0 立即 — Bug X1 triple_key 碰撞（数据静默丢失）
  修复: 改用不可见字符分隔，如 \x00 或采用 hash(tuple) 方案

P1 重要 — Bug X2 should_retry 中文错误处理

P2 改善 — X4 ILIKE转义, X5 encoding_level校验
```

