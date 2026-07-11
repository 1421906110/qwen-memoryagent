# 🧠 CogniMem v0.9 核心功能测试手册

> **版本**: v0.9  
> **测试日期**: 2026-07-06  
> **测试范围**: Agent 记忆核心功能（Action Facts + 自验证 + 三级召回 + 工具链）  
> **前置条件**: 服务运行 `http://localhost:9999`，PostgreSQL cognimem 库可用

---

## 目录

1. [测试环境与工具](#1-测试环境与工具)
2. [基础记忆功能](#2-基础记忆功能)
3. [Action Facts — Agent 行为记忆](#3-action-facts--agent-行为记忆)
4. [三级召回测试](#4-三级召回测试)
5. [自验证系统测试](#5-自验证系统测试)
6. [Agent 工具链测试](#6-agent-工具链测试)
7. [跨话题记忆测试](#7-跨话题记忆测试)
8. [矛盾检测与学习](#8-矛盾检测与学习)
9. [多 Agent 隔离测试](#9-多-agent-隔离测试)
10. [边界与压力测试](#10-边界与压力测试)
11. [回归测试](#11-回归测试)

---

## 1. 测试环境与工具

```bash
# 服务健康检查
curl -s http://localhost:9999/health | python3 -m json.tool
# 期望: score >= 90, level: "healthy", 0 ERROR

# 查询数据库
psql -d cognimem -c "SELECT count(*) FROM facts;"

# 查看日志
tail -f /tmp/cognimem-server.log

# 清理测试数据（慎用！）
# psql -d cognimem -c "DELETE FROM facts WHERE agent_id='test-xxx';"
```

---

## 2. 基础记忆功能

### 2.1 直接记忆存储

```bash
curl -s -X POST http://localhost:9999/memory \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "test-basic",
    "text": "用户喜欢喝冰美式咖啡",
    "memory_type": "preference"
  }' | python3 -m json.tool
```

**验证点：**
- 返回 `status: "ok"`
- 数据库 `facts` 表有该 agent 的记录
- 提取的三元组包含 `(用户, 喜欢, 冰美式咖啡)` 或类似

### 2.2 直接召回

```bash
curl -s -X POST http://localhost:9999/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "用户喜欢什么咖啡", "agent_id": "test-basic"}' | python3 -m json.tool
```

**验证点：**
- `count` > 0
- 返回的记忆内容与存储的相符
- 置信度 > 0.5

### 2.3 空查询 → 返回全部

```bash
curl -s -X POST http://localhost:9999/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "", "agent_id": "test-basic"}' | python3 -c "
import sys,json
data = json.load(sys.stdin)
print(f'共 {data[\"count\"]} 条记忆')
for m in data['memories'][:5]:
    print(f'  {m[\"content\"]} (conf={m[\"confidence\"]})')
"
```

**期望：** 返回该 agent 所有事实

### 2.4 清空 Agent 记忆

```bash
curl -s -X POST "http://localhost:9999/groom?agent_id=test-basic"
```

**验证点：** `deleted` > 0

---

## 3. Action Facts — Agent 行为记忆

> ⭐ **v0.9 核心新功能**：Agent 的工具调用结果存为结构化 FactTriple

### 3.1 写文件 → 自动存行为记忆

```bash
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "帮我在桌面写一个 hello.py，内容是 print(\"Hello World\")",
    "agent_id": "test-action",
    "session_id": "test-action-1"
  }' | python3 -c "
import sys,json
d = json.load(sys.stdin)
print(f'回复: {d.get(\"reply\",\"\")[:100]}')
print(f'工具: {d.get(\"tools_called\",0)}')
"
```

**验证点（必测 ⭐）：**
```bash
# 1. 检查数据库是否有 action 类型的事实
psql -d cognimem -c "
SELECT subject, predicate, object, fact_type, confidence 
FROM facts 
WHERE agent_id='test-action' AND fact_type='action';
"

# 期望: 看到 (test-action, 创建了文件, /Users/baikai/Desktop/hello.py) @0.9
#       而不是 (用户, 说了, 完成了一个任务) @0.5
```

### 3.2 搜刮做过的行为

```bash
curl -s -X POST http://localhost:9999/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "test-action 之前做了什么", "agent_id": "test-action"}' | python3 -c "
import sys,json
data = json.load(sys.stdin)
print(f'搜到 {data[\"count\"]} 条:')
for m in data['memories']:
    print(f'  {m[\"content\"]} (conf={m[\"confidence\"]}, type={m.get(\"fact_type\",\"?\")})')
"
```

**期望：** 返回 `(test-action, 创建了文件, hello.py)` @0.9 ✅

### 3.3 搜具体内容

```bash
for q in "hello.py" "创建了文件" "print" "桌面"; do
  echo "=== 搜: $q ==="
  curl -s -X POST http://localhost:9999/recall \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"$q\", \"agent_id\": \"test-action\"}" | python3 -c "
import sys,json
data = json.load(sys.stdin)
for m in data['memories']: print(f'  {m[\"content\"]}')
print(f'  共 {len(data[\"memories\"])} 条')
"
done
```

**期望：**
- "hello.py" → 命中 action fact ✅
- "创建了文件" → 命中 action fact ✅
- "print" → 可能命中（取决于 ILIKE）
- "桌面" → 可能命中（object 路径含 Desktop）

### 3.4 搜索工具调用

```bash
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "搜索一下最近的AI新闻",
    "agent_id": "test-action",
    "session_id": "test-action-2"
  }' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'工具: {d.get(\"tools_called\",0)}')"
```

**验证点：**
```bash
# 搜创建和搜索两种行为
curl -s -X POST http://localhost:9999/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "test-action 做了什么", "agent_id": "test-action"}' | python3 -c "
import sys,json
data = json.load(sys.stdin)
for m in data['memories']:
    if m.get('fact_type') == 'action':
        print(f'  ✅ {m[\"content\"]}')
print(f'  共 {len(data[\"memories\"])} 条')
"
```

**期望：** 同时看到创建文件和搜索两种 action fact ✅

---

## 4. 三级召回测试

### 4.1 L1 精确匹配

```bash
# 先存一条事实
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "记住：用户叫张三，在北京工作",
    "agent_id": "test-recall",
    "session_id": "test-recall-1"
  }' 2>&1 | grep -c "工具" 

# 精确查询
curl -s -X POST http://localhost:9999/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "张三", "agent_id": "test-recall"}' | python3 -c "
import sys,json
print(f'L1命中: {json.load(sys.stdin)[\"count\"]} 条')
"
```

### 4.2 L1.5 BM25 模糊匹配

```bash
# 查询词与存储事实部分重叠
curl -s -X POST http://localhost:9999/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "张先生在北京", "agent_id": "test-recall"}' | python3 -c "
import sys,json
data = json.load(sys.stdin)
print(f'BM25命中: {data[\"count\"]} 条')
for m in data['memories'][:3]:
    print(f'  {m[\"content\"]} (conf={m[\"confidence\"]})')
"
```

**期望：** "北京"重叠 → 应找到事实

### 4.3 搜索无相关记忆

```bash
curl -s -X POST http://localhost:9999/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "今天天气怎么样", "agent_id": "test-recall"}' | python3 -c "
import sys,json
data = json.load(sys.stdin)
assert data['count'] == 0, '不应该搜到天气相关'
print('✅ 无关查询不返回结果')
"
```

---

## 5. 自验证系统测试

> ⭐ **v0.9 核心新功能**：工具调用后自动验证结果

### 5.1 正常写文件 → 验证通过（无声无息）

```bash
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "在桌面创建一个test_verify.txt，内容是 verified",
    "agent_id": "test-verify",
    "session_id": "test-verify-1"
  }' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'工具: {d.get(\"tools_called\",0)}')"

# 验证文件真实存在
ls -la ~/Desktop/test_verify.txt && echo "✅ 文件存在" || echo "❌ 文件不存在"
```

### 5.2 验证器日志检查

```bash
grep -i "验证\|_verified\|_issues\|自动重试\|verify" /tmp/cognimem-server.log | grep -v consolidation | tail -10
```

**期望：** 没有写文件的验证失败日志（成功时验证器静默通过）

### 5.3 shell 失败 → 验证器捕获

```bash
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "运行一个不存在的命令 xyz_invalid_cmd_123",
    "agent_id": "test-verify",
    "session_id": "test-verify-2"
  }' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'回复: {d.get(\"reply\",\"\")[:120]}')"
```

**验证点：**
```bash
grep "自验证失败" /tmp/cognimem-server.log
# 期望: ⚠️ 自验证失败: shell — 命令退出码 127: ...
```

### 5.4 空文件 → 验证器触发重试

**手动测试：** 先创建一个空文件，然后让 agent 覆盖写内容

```bash
# 准备空文件
echo -n "" > ~/Desktop/empty_test.txt

# 让 agent 写文件
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "把桌面的empty_test.txt改成内容 \"filled by agent\"",
    "agent_id": "test-verify",
    "session_id": "test-verify-3"
  }' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'回复: {d.get(\"reply\",\"\")[:120]}')"

# 验证文件有内容
cat ~/Desktop/empty_test.txt
```

---

## 6. Agent 工具链测试

### 6.1 多步骤复杂任务

```bash
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "搜索今天的AI新闻，把结果保存到桌面 ai_news_today.txt",
    "agent_id": "test-tools",
    "session_id": "test-tools-1"
  }' | python3 -c "
import sys,json
d = json.load(sys.stdin)
print(f'回复: {d.get(\"reply\",\"\")[:150]}')
print(f'工具调用: {d.get(\"tools_called\",0)}')
print(f'执行次数: {d.get(\"iterations\",0)}')
"
```

**验证点：**
- 文件已创建：`ls -la ~/Desktop/ai_news_today.txt`
- Action fact 存了两条：搜索 + 创建文件
- 跨话题能搜到这两条

### 6.2 工具调用稳定性（连续5次）

```bash
for i in 1 2 3 4 5; do
  echo "=== 第 $i 次 ==="
  curl -s -X POST http://localhost:9999/chat \
    -H "Content-Type: application/json" \
    -d "{
      \"message\": \"在桌面创建 test_run_$i.txt\",
      \"agent_id\": \"test-tools\",
      \"session_id\": \"test-tools-$i\"
    }" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'工具: {d.get(\"tools_called\",0)}')"
done
```

**期望：** 5 次全部成功，`tools_called` >= 1 每次

### 6.3 Agent 自主规划任务

```bash
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "对比一下 Python 和 JavaScript，写到桌面 lang_compare.txt",
    "agent_id": "test-tools",
    "session_id": "test-tools-plan"
  }' | python3 -c "
import sys,json
d = json.load(sys.stdin)
print(f'回复: {d.get(\"reply\",\"\")[:150]}')
print(f'工具: {d.get(\"tools_called\",0)}, 迭代: {d.get(\"iterations\",0)}')
"
```

**期望：** Agent 自动规划步骤（搜索→整理→写文件），不等待用户中间确认

---

## 7. 跨话题记忆测试

> ⭐ **核心场景**：模拟"小明做完贪吃蛇就忘"问题

### 7.1 先做事，换话题

```bash
# 步骤1：让 agent 做事
echo "=== 步骤1: 做事 ==="
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "在桌面写一个 notes.md，内容是今天的测试笔记",
    "agent_id": "test-cross",
    "session_id": "test-cross-1"
  }' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'工具: {d.get(\"tools_called\",0)}')"

echo ""
echo "=== 步骤2: 换完全不相关的话题 ==="
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "今天天气怎么样？",
    "agent_id": "test-cross",
    "session_id": "test-cross-2"
  }' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'回复: {d.get(\"reply\",\"\")[:80]}')"

echo ""
echo "=== 步骤3: 回来问做了什么 ==="
curl -s -X POST http://localhost:9999/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "之前做了什么", "agent_id": "test-cross"}' | python3 -c "
import sys,json
data = json.load(sys.stdin)
for m in data['memories']:
    if m.get('fact_type') == 'action':
        print(f'  ✅ {m[\"content\"]} (conf={m[\"confidence\"]})')
    else:
        print(f'  {m[\"content\"]} (conf={m[\"confidence\"]})')
print(f'共 {len(data[\"memories\"])} 条')
"
```

**期望：** 搜"之前做了什么" → 看到 `(test-cross, 创建了文件, notes.md)` @0.9 ✅
**关键：** 不再说"我好像还没做"

### 7.2 跨话题 + agent 聊天调用

```bash
# 模拟完整对话：先做事 → 聊别的 → 回来问
SESSION="cross-chat-$(date +%s)"

echo "=== 消息1: 让 agent 做事 ==="
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"message\": \"帮我写一个python脚本统计桌面文件数量\",
    \"agent_id\": \"test-cross-chat\",
    \"session_id\": \"$SESSION\"
  }" > /dev/null

echo "=== 消息2: 换话题聊几句 ==="
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"message\": \"推荐几本编程书\",
    \"agent_id\": \"test-cross-chat\",
    \"session_id\": \"$SESSION\"
  }" > /dev/null

echo "=== 消息3: 回来问 ==="
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"message\": \"我刚才让你做了什么？\",
    \"agent_id\": \"test-cross-chat\",
    \"session_id\": \"$SESSION\"
  }" | python3 -c "
import sys,json
d = json.load(sys.stdin)
reply = d.get('reply','')
print(f'回复: {reply[:200]}')
if 'python' in reply.lower() or '脚本' in reply or '统计' in reply or '文件' in reply:
    print('✅ Agent 记得之前做了什么！')
else:
    print('⚠️ Agent 可能不记得，检查 recall 结果')
"
```

**期望：** Agent 回复提及之前的 python 脚本任务

---

### 7.5 复杂任务场景

> 真实世界复杂任务：多步骤、排错、多源调研、从经验学习

#### 7.5.1 排错任务：代码有问题，让 agent 诊断修复

**场景描述：** 先准备一个有 bug 的 Python 脚本，让 agent 读代码 → 发现问题 → 修复 → 验证

```bash
# 先准备一个有 bug 的文件
cat > ~/Desktop/buggy.py << 'EOF'
def calc_average(nums)
    total = sum(nums)
    return total / len(nums)

result = calc_average([10, 20, 30, 40, 50])
print("平均数是: " + result)
EOF

echo "=== 让 agent 修复 buggy.py ==="
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "桌面上有个 buggy.py 有 bug，帮我读代码找出问题并修复",
    "agent_id": "test-complex",
    "session_id": "test-complex-debug"
  }' | python3 -c "
import sys,json
d = json.load(sys.stdin)
print(f'工具: {d.get(\"tools_called\",0)}')
print(f'迭代: {d.get(\"iterations\",0)}')
print(f'回复: {d.get(\"reply\",\"\")[:200]}')
"

echo ""
echo "=== 验证修复后的文件 ==="
python3 ~/Desktop/buggy.py 2>&1 && echo "✅ 脚本运行成功" || echo "❌ 脚本仍有问题"

# 清理
rm -f ~/Desktop/buggy.py
```

**期望：**
- Agent 读取了文件（read_file）
- 识别到两个 bug：`def calc_average(nums):` 缺冒号、`print("平均数是: " + result)` 不能拼接 int
- 修复后文件能正常运行 ✅

---

#### 7.5.2 多源调研任务：搜索多个来源，综合输出

**场景描述：** 让 agent 搜索某个主题，从多个来源获取信息，综合成一份结构化报告

```bash
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "帮我调研一下 2026 年最值得学习的编程语言 top 3，每个语言查一下特点、应用场景和薪资范围，汇总写到桌面 programming_languages_2026.md",
    "agent_id": "test-complex",
    "session_id": "test-complex-research"
  }' | python3 -c "
import sys,json
d = json.load(sys.stdin)
print(f'工具: {d.get(\"tools_called\",0)}')
print(f'迭代: {d.get(\"iterations\",0)}')
print(f'回复: {d.get(\"reply\",\"\")[:200]}')
"

echo ""
echo "=== 验证输出文件 ==="
wc -l ~/Desktop/programming_languages_2026.md 2>/dev/null
echo "--- 文件内容预览 ---"
head -20 ~/Desktop/programming_languages_2026.md 2>/dev/null
```

**预期行为：**
- Agent 自动规划调研步骤（拆成 3 次搜索，每个语言搜一次）
- 或单次搜索覆盖多个语言，然后拆解结果
- 最终输出文件包含 3 个语言的完整信息
- 过程中不中途问用户"接下来做什么"

---

#### 7.5.3 工具失败恢复：搜索失败，agent 换方案

**场景描述：** 故意让一个搜索失败，看 agent 能否自动换方案

```bash
echo "=== 让 agent 搜一个肯定会失败的内容 ==="
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "搜一下 asdlfkjasldfkjalsdkfjalskdfj 这个公司怎么样",
    "agent_id": "test-complex",
    "session_id": "test-complex-retry"
  }' | python3 -c "
import sys,json
d = json.load(sys.stdin)
print(f'工具: {d.get(\"tools_called\",0)}')
print(f'回复: {d.get(\"reply\",\"\")[:200]}')
# 应该看到 agent 尝试搜索，没结果，然后诚实说没找到
reply = d.get('reply','')
if '没找到' in reply or '没有' in reply or '找不到' in reply:
    print('✅ Agent 诚实地报告没找到')
else:
    print('⚠️ Agent 可能编造了答案，检查回复')
"
```

**期望：**
- Agent 尝试搜索
- 搜索结果为空
- 验证器可能标记 `_verified: false`
- Agent 不编造答案，直接说"没找到"或"搜不到信息"
- **不应** 编造公司信息

---

#### 7.5.4 记忆复用任务：先学经验，再应用到新任务

**场景描述：** 先让 agent 做一个搜索+保存的任务，存为经验。再做一个类似的搜索+保存任务，看它是否能复用上一次的经验（不自嗨、不多搜）

```bash
echo "=== 任务1: 搜索+保存（建立经验）==="
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "查一下 Rust 语言的特点和优势，写到桌面 rust_info.md",
    "agent_id": "test-complex",
    "session_id": "test-complex-learn"
  }' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'任务1: {d.get(\"tools_called\",0)} 次工具')"

echo ""
echo "=== 任务2: 类似任务（看能不能借鉴经验）==="
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "查一下 Go 语言的特点和优势，写到桌面 go_info.md",
    "agent_id": "test-complex",
    "session_id": "test-complex-learn-2"
  }' | python3 -c "
import sys,json
d = json.load(sys.stdin)
tc = d.get('tools_called', 0)
print(f'任务2: {tc} 次工具')
if tc <= 4:
    print('✅ Agent 可能借鉴了经验，工具调用合理')
else:
    print('⚠️ Agent 可能过度搜索，检查是否自嗨')
print(f'回复: {d.get(\"reply\",\"\")[:100]}')
"

echo ""
echo "=== 验证: 回到之前说 Rust==="
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "我之前让你查过什么编程语言？",
    "agent_id": "test-complex",
    "session_id": "test-complex-learn-3"
  }' | python3 -c "
import sys,json
d = json.load(sys.stdin)
reply = d.get('reply','')
if 'Rust' in reply or 'rust' in reply.lower():
    print('✅ Agent 记得 Rust 任务')
else:
    print(f'⚠️ 可能不记得了: {reply[:80]}')
if 'Go' in reply or 'go' in reply.lower():
    print('✅ Agent 也记得 Go 任务')
"

# 清理
rm -f ~/Desktop/rust_info.md ~/Desktop/go_info.md
```

**预期行为：**
- 任务2 比任务1 用的工具数更少（说明经验起了作用，不自嗨）
- agent 记得之前查过 Rust 和 Go（Action Facts 召回）

---

#### 7.5.5 长对话记忆：10轮以上保持连贯

**场景描述：** 模拟较长的多轮对话，验证 agent 在长上下文中记忆不丢失

```bash
SESSION="test-complex-long-$$"

echo "=== 轮次1-4: 随机聊天（刷上下文）==="
for msg in "你好" "推荐一部电影" "为什么天是蓝色的" "帮我写一个斐波那契函数"; do
  curl -s -X POST http://localhost:9999/chat \
    -H "Content-Type: application/json" \
    -d "{\"message\": \"$msg\", \"agent_id\": \"test-complex\", \"session_id\": \"$SESSION\"}" > /dev/null
done

echo "=== 轮次5: 让 agent 记住一件事 ==="
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"记住：我养了一只叫旺财的柯基犬\", \"agent_id\": \"test-complex\", \"session_id\": \"$SESSION\"}" > /dev/null

echo "=== 轮次6-10: 继续刷话题 ==="
for msg in "你会做什么" "现在几点" "讲个笑话" "地球有多重" "什么是区块链"; do
  curl -s -X POST http://localhost:9999/chat \
    -H "Content-Type: application/json" \
    -d "{\"message\": \"$msg\", \"agent_id\": \"test-complex\", \"session_id\": \"$SESSION\"}" > /dev/null
done

echo "=== 轮次11: 回来问还记得旺财吗 ==="
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"我养的狗叫什么名字？\", \"agent_id\": \"test-complex\", \"session_id\": \"$SESSION\"}" | python3 -c "
import sys,json
d = json.load(sys.stdin)
reply = d.get('reply','')
if '旺财' in reply or '柯基' in reply:
    print('✅ 10轮后还记得！🐕')
else:
    print(f'⚠️ 可能不记得了: {reply[:100]}')
"
```

**期望：** 10轮对话后，agent 仍能通过 CogniMem 召回"养了叫旺财的柯基犬"这一事实

---

## 8. 矛盾检测与学习

### 8.1 信息更新 → 旧信息被挑战

```bash
# 先存一条
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "记住：用户工作在阿里巴巴",
    "agent_id": "test-conflict",
    "session_id": "test-conflict-1"
  }' > /dev/null

# 更新信息
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "不对，我工作在腾讯，不是阿里巴巴",
    "agent_id": "test-conflict",
    "session_id": "test-conflict-2"
  }' > /dev/null

# 检查矛盾
curl -s "http://localhost:9999/stats?agent_id=test-conflict" | python3 -c "
import sys,json
d = json.load(sys.stdin)
print(f'矛盾数: {d.get(\"contradictions\",\"?\")}')
print(f'事实总数: {d.get(\"total_facts\",\"?\")}')
"
```

**验证点：** 应该有矛盾记录

### 8.2 主动学习问题

```bash
curl -s -X POST http://localhost:9999/remember \
  -H "Content-Type: application/json" \
  -d '{"query": "用户的工作", "agent_id": "test-conflict"}' | python3 -m json.tool
```

**期望：** 返回矛盾相关的问题

---

## 9. 多 Agent 隔离测试

### 9.1 不同 Agent 记忆隔离

```bash
# Agent A 存信息
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "记住：用户是老大",
    "agent_id": "boss-agent",
    "session_id": "boss-1"
  }' > /dev/null

# Agent B 搜不到 A 的信息
curl -s -X POST http://localhost:9999/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "用户是", "agent_id": "worker-agent"}' | python3 -c "
import sys,json
data = json.load(sys.stdin)
count = data['count']
print(f'Worker搜boss的记忆: {count} 条')
assert count == 0, '❌ Agent 隔离失败！Worker 搜到了 Boss 的记忆'
if count == 0: print('✅ Agent 隔离正常')
"
```

### 9.2 Agent 数量压力

```bash
# 创建 10 个 agent 各做一件事
for i in $(seq 1 10); do
  curl -s -X POST http://localhost:9999/chat \
    -H "Content-Type: application/json" \
    -d "{
      \"message\": \"记住 agent_$i 的编号是 $i\",
      \"agent_id\": \"agent_$i\",
      \"session_id\": \"stress-$i\"
    }" > /dev/null
done

# 验证每个 agent 的记忆独立
for i in $(seq 1 3); do
  curl -s -X POST http://localhost:9999/recall \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"编号\", \"agent_id\": \"agent_$i\"}" | python3 -c "
import sys,json
data = json.load(sys.stdin)
print(f'agent_$i: {data[\"count\"]} 条记忆')
" 
done
```

**期望：** 每个 agent 只看到自己的记忆

---

## 10. 边界与压力测试

### 10.1 空消息

```bash
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "",
    "agent_id": "test-edge",
    "session_id": "test-edge-1"
  }' | python3 -c "import sys,json; print(json.load(sys.stdin).get('detail','??')[:100])"
```

**期望：** 返回 422 验证错误

### 10.2 超长消息

```bash
LONG_MSG=$(python3 -c "print('A' * 50000)")
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"message\": \"$LONG_MSG\",
    \"agent_id\": \"test-edge\",
    \"session_id\": \"test-edge-2\"
  }" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'回复长度: {len(d.get(\"reply\",\"\"))}')"
```

**期望：** 不崩溃，有合理回复

### 10.3 不存在的 Agent

```bash
curl -s -X POST http://localhost:9999/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "测试", "agent_id": "不存在的_agent_xyz"}' | python3 -c "
import sys,json
data = json.load(sys.stdin)
print(f'不存在的agent: {data[\"count\"]} 条')
assert data['count'] == 0, '不存在的agent应该返回0条'
"
```

### 10.4 并发请求

```bash
# 同时发 5 个请求（模拟多用户）
for i in 1 2 3 4 5; do
  curl -s -X POST http://localhost:9999/chat \
    -H "Content-Type: application/json" \
    -d "{
      \"message\": \"写一个 concurrency_test_$i.txt\",
      \"agent_id\": \"test-con\",
      \"session_id\": \"test-con-$i\"
    }" > /dev/null &
done
wait
echo "5个并发请求完成"
ls -la ~/Desktop/concurrency_test_*.txt 2>/dev/null | wc -l | xargs echo "创建的文件数:"
```

---

## 11. 回归测试

### 11.1 服务健康

```bash
echo "=== 健康检查 ==="
curl -s http://localhost:9999/health | python3 -c "
import sys,json
d = json.load(sys.stdin)
print(f'分: {d[\"score\"]}')
print(f'级: {d[\"level\"]}')
print(f'行: {d[\"uptime_seconds\"]:.0f}s')
assert d['score'] >= 80, f'健康分异常: {d[\"score\"]}'
assert d['level'] == 'healthy', f'健康等级异常: {d[\"level\"]}'
print('✅ 健康检查通过')
"
```

### 11.2 API 端点完整性

```bash
echo "=== API 端点测试 ==="
for endpoint in "/health" "/stats?agent_id=default" "/status"; do
  status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:9999$endpoint)
  echo "  GET $endpoint → $status"
  assert $status == 200
done

for endpoint in "/chat" "/recall" "/groom?agent_id=default" "/memory" "/remember"; do
  status=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:9999$endpoint)
  echo "  POST $endpoint → $status"
done
```

### 11.3 测试后清理

```bash
# 清理本次测试创建的文件
rm -f ~/Desktop/test_*.txt
rm -f ~/Desktop/hello.py
rm -f ~/Desktop/ai_news_today.txt
rm -f ~/Desktop/lang_compare.txt
rm -f ~/Desktop/notes.md
rm -f ~/Desktop/concurrency_test_*.txt
rm -f ~/Desktop/snake_game.html
rm -f ~/Desktop/hello_world.py
rm -f ~/Desktop/empty_test.txt

echo "✅ 测试文件已清理"

# 清理测试 agent 的数据库记录
# psql -d cognimem -c "DELETE FROM facts WHERE agent_id LIKE 'test-%';"
# echo "✅ 测试数据已清理"
```

---

## 测试报告模板

测试完成后，把结果填到下面的模板：

```markdown
# CogniMem v0.9 测试报告

测试日期: 2026-07-06
测试人: 
服务版本: v0.9
健康分: 

## 测试结果汇总

| 模块 | 用例 | 结果 |
|------|------|:----:|
| 基础记忆 | 存/取/空查询 | ⬜ |
| Action Facts | 行为存/搜/跨话题 | ⬜ |
| 三级召回 | L1/L1.5/L3 | ⬜ |
| 自验证 | 写/shell/重试 | ⬜ |
| 工具链 | 多步/稳定/规划 | ⬜ |
| 跨话题 | 做事→换→回问 | ⬜ |
| 矛盾检测 | 更新/挑战 | ⬜ |
| 多Agent | 隔离/数量 | ⬜ |
| 边界 | 空/超长/并发 | ⬜ |

## 发现的问题

| 问题 | 严重度 | 状态 |
|------|:------:|:----:|
| ... | P0/P1/P2 | 待修/已修/跳过 |

## 结论

[通过/有条件通过/不通过]
```

---

> **文档版本**: v0.9  
> **最后更新**: 2026-07-06  
> **对应项目**: qwen-memoryagent (CogniMem)
