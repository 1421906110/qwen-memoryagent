# CogniMem 全盘排查手册

快速定位系统中存在的一切不合理与 Bug。

---

## 1. 启动验证（3 秒）

```bash
# 服务是否活着
curl -s http://localhost:9999/health | python3 -m json.tool

# 日志有无异常
grep -c "ERROR" /tmp/qwen-agent.log
tail -20 /tmp/qwen-agent.log | grep -i "error\|traceback\|exception"

# 工具数量
curl -s http://localhost:9999/health | python3 -c "import sys,json; print(json.load(sys.stdin)['checks']['tools'])"
# 期望: 12
```

---

## 2. 内存系统排查

### 2.1 数据完整性

```bash
# 事实总数 + 矛盾率
curl -s 'http://localhost:9999/stats?agent_id=default'

# 浏览全部记忆（检查乱入/重复/异常）
curl -s 'http://localhost:9999/memories?agent_id=default&limit=100' \
  | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f'Total: {d[\"total\"]}')
for m in d['memories']:
    text = f\"{m.get('subject','')} {m.get('predicate','')} {m.get('object','')}\"
    # 检查空字段
    if not m['subject'] or not m['predicate'] or not m['object']:
        print(f'  ⚠️ EMPTY FIELD: {m[\"fact_id\"]}')
    # 检查超长文本
    if len(text) > 200:
        print(f'  ⚠️ OVERLONG: {m[\"fact_id\"]} ({len(text)} chars)')
    # 检查置信度异常
    if m['confidence'] <= 0 or m['confidence'] > 1:
        print(f'  ⚠️ BAD CONFIDENCE: {m[\"fact_id\"]} = {m[\"confidence\"]}')
print('Done')
"
```

### 2.2 矛盾检测

```bash
# 查看未解决的矛盾
psql cognimem -c "SELECT id, fact_a_id, fact_b_id, contradiction_type, description FROM contradictions WHERE resolution='pending' LIMIT 20;"

# 矛盾率过高说明提取器太敏感或用户频繁改口
# 正常值: < 20%
```

### 2.3 提取器测试

```bash
# 测试规则提取（直接调 brain.remember 看 extract 结果）
# 关键检查: 中英文混合、特殊字符、空输入
python3 -c "
from cognimem.core.extractor import TripleExtractor
e = TripleExtractor()

# 边界情况
tests = [
    '',                    # 空字符串
    'hi',                  # 英文问候
    '我',                  # 单字
    'a' * 1000,            # 超长文本
    '你好！！！？？？',     # 特殊字符
    '深圳的天气很好',       # re.search vs re.match（之前修过的bug）
    '我喜欢喝冰美式',       # 标准偏好
    '我不喜欢喝咖啡',       # 否定
    None,                  # None 输入
]
for t in tests:
    try:
        r = e.extract(t)
        print(f'[{repr(t)[:30]}] → {len(r)} facts')
    except Exception as ex:
        print(f'[{repr(t)[:30]}] ❌ {ex}')
"
```

---

## 3. Agent 排查

### 3.1 工具调用跟踪

日志搜索以下关键词判断 Agent 行为是否正常：

```bash
# 工具调用频率
grep -c "Tool call" /tmp/qwen-agent.log

# 工具失败
grep "Tool.*failed" /tmp/qwen-agent.log

# 自我修复触发
grep -c "Auto-fix\|Self-Reflection\|Auto-fix worked" /tmp/qwen-agent.log

# 规划行为
grep "规划\|Plan:" /tmp/qwen-agent.log

# 记忆存储
grep -c "Stored.*memories" /tmp/qwen-agent.log

# 无限循环检测（迭代次数过多）
grep "max iterations\|Hit max" /tmp/qwen-agent.log
```

### 3.2 Agent 是否卡死

```bash
# 检查 agent 回话模式：如果 is_simple 判断错误，
# 简单问题走了复杂 agent 路径，浪费工具调用
grep "Complex task\|Simple Q&A" /tmp/qwen-agent.log | tail -20
# 期望：大部分简单问答走 Simple Q&A
```

### 3.3 内存膨胀检查

```bash
# 系统提示词是否重复追加（之前修过 goal progress 标题不匹配的bug）
grep -c "📋 進度" /tmp/qwen-agent.log

# 检查上下文 token 估算（正常应该在 500-2000 之间）
# 如果超过 5000 说明有累积
```

---

## 4. 数据库排查

### 4.1 连接池

```bash
# 检查是否有连接泄露：连接数持续增长
psql cognimem -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';"

# 检查僵尸连接
psql cognimem -c "SELECT pid, state, query_start, query FROM pg_stat_activity WHERE state = 'idle in transaction' OR state = 'idle';"
```

### 4.2 数据一致性

```bash
# 缓存 vs DB 数据量差异（缓存 5min TTL，理论值接近）
# 缓存中的事实数
curl -s 'http://localhost:9999/stats?agent_id=default' | python3 -c "import sys,json; print(json.load(sys.stdin)['total_facts'])"

# DB 中的事实数
psql cognimem -c "SELECT count(*) FROM facts WHERE agent_id='default';"

# 两个数字应一致。如果差异大，说明缓存-DB 一致性有问题
```

### 4.3 孤立数据检查

```bash
# contradictions 引用了已删除的 fact
psql cognimem -c "
SELECT c.id, c.fact_a_id, c.fact_b_id
FROM contradictions c
LEFT JOIN facts fa ON fa.fact_id = c.fact_a_id
LEFT JOIN facts fb ON fb.fact_id = c.fact_b_id
WHERE fa.fact_id IS NULL OR fb.fact_id IS NULL;
"

# fact_versions 引用了已删除的 fact
psql cognimem -c "
SELECT v.id, v.fact_id
FROM fact_versions v
LEFT JOIN facts f ON f.fact_id = v.fact_id
WHERE f.fact_id IS NULL
LIMIT 10;
"
```

---

## 5. 并发排查

### 5.1 锁覆盖检查

每个修改共享状态的操作都应被 `threading.RLock` 保护。
检查以下模式是否遗漏：

```bash
# 查找所有直接访问 _lru_cache 的地方（应该在锁内）
grep -rn "_lru_cache" src/cognimem/core/ \
  | grep -v "test\|__pycache__\|\.pyc" \
  | grep -v "with self._lock\|_get_cached_facts"
# 期望：没有输出（所有访问都在锁内）

# 查找所有 _get_conn 调用（应该在 _conn_ctx/_cursor_ctx 内）
grep -rn "\._get_conn()" src/ \
  | grep -v "test\|__pycache__\|\.pyc" \
  | grep -v "_conn_ctx\|_get_conn(self)"
# 期望：只看到 _conn_ctx 和 _get_conn 定义本身
```

### 5.2 后台线程安全

```bash
# consolidation 后台线程是否正确释放锁
grep "_consolidating" src/cognimem/core/fact_network.py
# 期望：add/remove 都在 with self._lock 内
```

---

## 6. 前端排查

### 6.1 网络请求

```bash
# 打开浏览器 Developer Tools → Network 标签
# 检查:
# 1. 所有请求的 agent_id 参数是否一致（别硬编码 default）
# 2. 是否有 4xx/5xx 错误
# 3. 请求耗时是否合理（>10s 说明有问题）
```

### 6.2 Common JS 错误

在浏览器控制台执行：

```javascript
// 检查 _streaming 锁是否正确释放
// 在 sendMsg 结束后，_streaming 应为 false
console.log('_streaming =', typeof _streaming !== 'undefined' ? _streaming : 'undefined');

// 检查 esc 函数是否存在
console.log('esc =', typeof esc === 'function' ? 'OK' : 'MISSING');

// 检查 agent selector 是否工作
const sel = document.getElementById('agent-select');
console.log('Agent selector:', sel ? sel.value : 'NOT FOUND');
```

### 6.3 XSS 检查

```bash
# 检查 innerHTML 拼接是否转义（重点检查 content/text 变量）
grep -rn "innerHTML.*+.*content\|innerHTML.*+.*text" src/memory_agent/templates/
# 如果输出包含 ${content} 或 ${text} 且没有调用 esc()，就是 XSS 风险
```

---

## 7. 依赖排查

```bash
# 检查所有 import 是否有效
python3 -c "
import ast, os
errors = []
for root, dirs, files in os.walk('src'):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            try:
                with open(path) as fh:
                    ast.parse(fh.read())
            except SyntaxError as e:
                errors.append(f'{path}:{e.lineno} {e.msg}')
if errors:
    for e in errors:
        print(f'❌ {e}')
else:
    print('✅ All files parse OK')
"

# 检查未使用的 import
# 手动搜索以下常见的死 import 模式:
# "from .llm_client import" 在 llm_client.py 内部
```

---

## 8. 健康分诊断

健康分异常时逐项排查：

```bash
# 配置问题
curl -s http://localhost:9999/health | python3 -c "
import sys,json
h = json.load(sys.stdin)
print(f'Score: {h[\"score\"]}')
print(f'Config: {h[\"checks\"].get(\"config\")}')
print(f'Tools: {h[\"checks\"].get(\"tools\")}')
print(f'Memory: {h[\"checks\"].get(\"memory\", {})}')
print(f'API: {h[\"checks\"].get(\"api\", {})}')
print(f'Issues: {len(h.get(\"issues\", []))}')
for i in h.get('issues', []):
    print(f'  [{i[\"severity\"]}] {i[\"detail\"]}')
"
```

低分常见原因：
- 60-80: 矛盾率偏高 → 运行 `/consolidate`
- 40-60: API 错误率高 → 检查 API Key / 网络
- < 40: CogniMem 未初始化 / 严重配置缺失

---

## 9. 回归测试命令

每次修改后执行：

```bash
# 1. 语法
python3 -c "
import ast, os
for root, dirs, files in os.walk('src'):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            with open(path) as fh: ast.parse(fh.read(), filename=path)
print('✅ All syntax OK')
"

# 2. 健康
curl -s http://localhost:9999/health | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['score']>=50, f'Health too low: {d[\"score\"]}'; print(f'Health: {d[\"score\"]}')"

# 3. 记忆
curl -s 'http://localhost:9999/memories?agent_id=default&limit=1' > /dev/null \
  && echo '✅ Memories API OK'

# 4. 聊天
curl -s http://localhost:9999/chat -X POST -H "Content-Type: application/json" \
  -d '{"agent_id":"default","session_id":"test","message":"hi"}' > /dev/null \
  && echo '✅ Chat API OK'

# 5. 日志
test $(grep -c ERROR /tmp/qwen-agent.log) -eq 0 && echo '✅ 0 errors' || echo '⚠️ Has errors'
```
