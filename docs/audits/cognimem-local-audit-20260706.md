# CogniMem 本地深度审计报告

**方法**: 直接 import 全模块，不依赖运行中服务器  
**范围**: 模型层 → 提取器 → 工具函数 → 存储层 → Agent 工具 → 测试套件  
**时间**: 2026-07-06

---

## 总览

| 层 | 文件 | 状态 |
|----|------|------|
| 模型 | `models.py` (cognimem + memory_agent) | ⚠️ 4个校验缺失 |
| 提取器 | `extractor.py` | ✅ 无崩溃 |
| 工具函数 | `llm_client.py`, `memory_service.py` | ❌ 3个运行时Bug |
| 存储层 | `SQLiteStore` | ✅ 正常 |
| Agent工具 | `tools.py` | ⚠️ 1个并发风险 |
| 测试套件 | `test_memory.py` | ✅ 26/26 全通过 |

---

## 🔴 严重 — 运行时崩溃

### Bug L1: `_decay_factor` 负 access_count 崩溃

- **位置**: `src/memory_agent/services/memory_service.py:37`
- **代码**: `half_life = 24.0 * (1.0 + access_count ** 0.5)`
- **触发**: `access_count < 0` → `(-n) ** 0.5` = complex → `'>' not supported between instances of 'complex' and 'float'`
- **影响**: 任何 access_count 传入负数直接炸

### Bug L2: `_semantic_similarity` 对中文完全失效

- **位置**: `src/memory_agent/services/memory_service.py:41-49`
- **代码**: `set_a = set(a.lower().split())`
- **根因**: `.split()` 按空格分词，中文没有空格 → 整句被当成一个 token → Jaccard 始终为 0 或 1
- **验证**:
  ```
  "我喜欢咖啡" vs "我讨厌咖啡" → sim=0.0000 ❌ 应为 >0
  "深圳天气" vs "深圳天气很好"  → sim=0.0000 ❌ 应为 >0
  ```
- **影响**: 中文冲突检测、重复合并、相似聚类全部失败

### Bug L3: `_is_retryable` 漏掉 TimeoutError

- **位置**: `src/memory_agent/services/llm_client.py:75-86`
- **根因**: `str(TimeoutError())` = `""`（空字符串），"timeout" 不在空串中
- **验证**:
  ```
  TimeoutError()             → is_retryable=False ❌ 应为 True
  TimeoutError("timed out")  → is_retryable=False ❌ "timed out" ≠ "timeout"
  ```
- **影响**: OpenAI SDK 超时不重试，直接抛给用户

---

## 🟡 中 — 数据模型校验缺失

### Bug L4: FactTriple 允许非法 confidence 值

- **位置**: `src/cognimem/core/models.py`
- **问题**: `confidence` 字段可接受 -0.1、1.5、inf、NaN
- **影响**: 脏数据入库，影响 decay 计算、路由排序、core_belief 判断

### Bug L5: FactTriple 允许非法 importance 值

- 同样可接受 -0.5、2.0、NaN
- 影响 decay 压缩逻辑

### Bug L6: EvidenceItem 允许空 statement

- 无 content 的证据记录无意义但可存入

### Bug L7: 超长字段无截断

- subject/object 可存入 10000 字符，无限制
- 影响 DB 存储和向量计算

---

## 🔵 低 — 并发/工程问题

### Bug L8: `_FAILED_DOMAINS` 无锁

- **位置**: `src/memory_agent/agent/tools.py`
- `_FAILED_DOMAINS` 是模块级 dict，多协程并发读写
- 影响：域名屏蔽统计可能不准确

### Bug L9: 工具缺少统一参数校验

- 12 个工具中，`read_file`/`write_file`/`shell` 等工具在缺少必填参数时抛 `KeyError`
- agent 传参时要靠 LLM 自行构造参数，无框架级校验
- 实际不严重因为 Agent loop 中 LLM 会补参数

---

## ✅ 已确认正常

| 检查项 | 结果 |
|--------|------|
| `extract(None)` 崩溃 | ✅ 已修复 |
| 提取器全边界输入 | ✅ None/空/emoji/超长/控制字符/日文/XSS 全不崩溃 |
| 测试套件 26/26 | ✅ |
| SQLiteStore CRUD | ✅ |
| SQLiteStore SQL 注入 | ✅ 参数化查询 |
| SQL 注入 (psycopg2) | ✅ 参数化 |
| 循环引入 | ✅ |
| 代码语法 | ✅ |
| `estimate_tokens` | ✅ |
| `_api_call_with_retry` 退避 | ✅ |
| `_is_retryable` 429/503 | ✅ |
| OpenAI timeout=30 | ✅ |
| `_HEALTH` 锁 | ✅ 已加 threading.Lock |
| 12 工具已注册 | ✅ |

---

## 修复优先级

```
P0 紧急 ─────────────────────────────────────
  L1  decay_factor 负access_count崩溃
  L2  semantic_similarity 中文失效
  L3  is_retryable 漏TimeoutError

P1 重要 ─────────────────────────────────────
  L4  FactTriple confidence 无校验
  L5  FactTriple importance 无校验

P2 改进 ─────────────────────────────────────
  L6  EvidenceItem 空statement
  L7  超长字段无截断
  L8  _FAILED_DOMAINS 无锁
```

---

*审计方式: 直接 import + 边界测试 + 源码审查。3 个 P0 运行时 Bug 之前所有手动测试和服务器测试都无法发现。*
