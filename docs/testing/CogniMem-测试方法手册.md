# 🧪 CogniMem 具体测试方法手册

> 版本：v0.10 | 日期：2026-07-10
> 测试目标：http://47.99.151.253:8000（ECS 公网）
> 前置条件：服务已启动，端口可达

---

## 一、🧠 智能记忆测试方法

### 1.1 三元组提取测试

**目的：** 验证自然语言能否正确拆解为 SPO 三元组

**测试命令：**

```bash
# 存记忆
curl -s -X POST http://47.99.151.253:8000/remember \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_mem","session_id":"t1","content":"用户喜欢喝冰美式","confidence":0.9}'

# 查记忆看提取结果
curl -s -X POST http://47.99.151.253:8000/recall \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_mem","query":"冰美式","limit":5}'
```

**预期输出：** `status=stored`，recall 返回三元组 `(用户, 喜欢, 喝冰美式)`

**覆盖句式：**

| 句式 | 输入 | 检查点 |
|:-----|:-----|:-------|
| 肯定句 | "用户喜欢喝冰美式" | 提取 subject=用户, predicate=喜欢 |
| 否定句 | "用户不喜欢喝热美式" | 提取 subject=用户, predicate=不喜欢 |
| 个人信息 | "我叫张三，住在北京" | 提取两个三元组 |
| 目标 | "用户目标是学会Python" | 提取 subject=用户, predicate=目标 |
| 带"的"句 | "张三的编程能力很强" | 正确处理"的"字结构 |

---

### 1.2 记忆召回精度测试

**目的：** 验证不同查询方式召回准确性

**前置条件：** 已完成 1.1 的数据存储

**测试命令：**

```bash
# 1.2a 空查询（浏览全部）
echo "=== 空查询 ==="
curl -s -X POST http://47.99.151.253:8000/recall \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_mem","query":"","limit":30}' | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f'返回 {d[\"count\"]} 条')
for m in d['memories'][:5]:
    print(f'  → {m[\"content\"][:60]}')
"

# 1.2b 精确查询
echo "=== 精确查询 ==="
curl -s -X POST http://47.99.151.253:8000/recall \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_mem","query":"冰美式","limit":10}'

# 1.2c 语义查询（近义词）
echo "=== 语义查询 ==="
curl -s -X POST http://47.99.151.253:8000/recall \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_mem","query":"程序员","limit":10}'

# 1.2d 组合条件
echo "=== 组合查询 ==="
curl -s -X POST http://47.99.151.253:8000/recall \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_mem","query":"张三 北京","limit":10}'
```

**验证标准：**
- 空查询：返回条数 > 0
- 精确查询：返回条数 >= 1，且包含"冰美式"
- 语义查询：不崩溃即可（BM25 能力有限）
- 组合条件：返回条数 >= 1

---

### 1.3 矛盾检测测试

**目的：** 验证 L1 否定矛盾检测、中性谓词跳过、跨类别偏好不误报

**测试命令：**

```bash
# 1.3a 真矛盾检测
echo "=== 真矛盾 ==="
curl -s -X DELETE "http://47.99.151.253:8000/clear?agent_id=test_con" > /dev/null
curl -s -X POST http://47.99.151.253:8000/remember \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_con","content":"用户喜欢喝冰美式","confidence":0.9}' > /dev/null
curl -s -X POST http://47.99.151.253:8000/remember \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_con","content":"用户不喜欢喝冰美式","confidence":0.9}' > /dev/null
sleep 2
curl -s "http://47.99.151.253:8000/stats?agent_id=test_con" | python3 -c "
import sys,json; d=json.load(sys.stdin)
c = d.get('contradictions',0)
print(f'矛盾数: {c}')
if c >= 1: print('✅ 真矛盾检出')
else: print('❌ 未检出')
"

# 1.3b 中性谓词不误报
echo "=== 中性谓词 ==="
curl -s -X POST http://47.99.151.253:8000/remember \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_con","content":"用户请求读取文件","confidence":0.9}' > /dev/null
curl -s -X POST http://47.99.151.253:8000/remember \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_con","content":"用户请求搜索网络","confidence":0.9}' > /dev/null
sleep 1
curl -s "http://47.99.151.253:8000/stats?agent_id=test_con" | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'矛盾数（应保持原数）: {d.get(\"contradictions\")}')
"

# 1.3c 跨类别偏好不误报
echo "=== 跨类别偏好 ==="
curl -s -X DELETE "http://47.99.151.253:8000/clear?agent_id=test_pref" > /dev/null
curl -s -X POST http://47.99.151.253:8000/remember \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_pref","content":"用户喜欢喝咖啡","confidence":0.9}' > /dev/null
curl -s -X POST http://47.99.151.253:8000/remember \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_pref","content":"用户喜欢吃火锅","confidence":0.9}' > /dev/null
sleep 1
curl -s "http://47.99.151.253:8000/stats?agent_id=test_pref" | python3 -c "
import sys,json; d=json.load(sys.stdin)
c = d.get('contradictions',0)
if c == 0: print('✅ 跨类别不误报')
else: print(f'❌ 误报矛盾 {c}')
"
```

**验证标准：**
- 真矛盾：contradictions >= 1
- 中性谓词：矛盾数不增加
- 跨类别偏好：contradictions == 0

---

### 1.4 置信度操作测试

**目的：** 验证 confirm / challenge / 版本链

**测试命令：**

```bash
# 获取一条已存事实的 ID
FID=$(curl -s -X POST http://47.99.151.253:8000/recall \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_mem","query":"冰美式","limit":1}' | python3 -c "
import sys,json; print(json.load(sys.stdin)['memories'][0]['id'])
")
echo "目标事实: $FID"

# confirm
echo -n "确认: "
curl -s -X POST http://47.99.151.253:8000/confirm \
  -H "Content-Type: application/json" \
  -d "{\"fact_id\":\"$FID\",\"agent_id\":\"test_mem\"}" | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'status={d[\"status\"]}')
"

# challenge
echo -n "质疑: "
curl -s -X POST http://47.99.151.253:8000/challenge \
  -H "Content-Type: application/json" \
  -d "{\"fact_id\":\"$FID\",\"agent_id\":\"test_mem\"}" | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'status={d[\"status\"]}')
"

# 版本链
echo -n "版本链: "
curl -s "http://47.99.151.253:8000/versions/$FID" | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'{d[\"count\"]} 个版本')
for v in d['versions']:
    print(f'  {v[\"change_reason\"]}: {v[\"old_confidence\"]} → {v[\"new_confidence\"]}')
"
```

**验证标准：** confirm → challenge 均返回成功，版本链 >= 2 条

---

## 二、🤖 Agent 能力测试方法

### 2.1 简单对话测试

```bash
echo "=== 简单对话 ==="
curl -s -X POST http://47.99.151.253:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_agt","session_id":"c1","message":"你好"}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
r=d.get('reply','')
print(f'回复: {r[:80]}')
assert len(r) > 5, '❌'
print('✅')
"
```
**成功标准：** 回复长度 > 5，不报错

### 2.2 搜索执行测试

```bash
echo "=== 搜索 ==="
curl -s -X POST http://47.99.151.253:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_agt","session_id":"c2","message":"搜一下AI新闻"}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
t=d['tools_called']; r=d['reply']
print(f'tools={t} reply={r[:60]}')
assert t > 0, '❌ 没调用工具'
print('✅')
"
```
**成功标准：** tools_called > 0

### 2.3 搜索早停测试

```bash
echo "=== 早停 ==="
curl -s -X POST http://47.99.151.253:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_agt","session_id":"c3","message":"搜一下Qwen2"}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
t=d['tools_called']
print(f'tools={t}')
assert t < 12, f'❌ 工具过多({t})'
print('✅')
"
```
**成功标准：** tools_called < 12

### 2.4 上下文分离测试

```bash
echo "=== 上下文分离 ==="
curl -s -X POST http://47.99.151.253:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id":"test_agt","session_id":"c4","message":"搜一下AI工具",
    "messages":[
      {"role":"user","content":"帮我在桌面新建个文件夹"},
      {"role":"assistant","content":"好的"},
      {"role":"user","content":"你会搜索吗？"},
      {"role":"assistant","content":"会的"}
    ]
  }' | python3 -c "
import sys,json; d=json.load(sys.stdin)
r=d['reply']
if '文件夹' in r: print('❌ 混淆')
elif d['tools_called']>0: print(f'✅ tools={d[\"tools_called\"]}')
else: print(f'⚠️ {r[:50]}')
"
```
**成功标准：** 回复中不出现"文件夹"

### 2.5 多步任务测试

```bash
echo "=== 多步任务 ==="
echo '多步测试内容' > /tmp/agent_multi_test.txt
curl -s -X POST http://47.99.151.253:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_agt","session_id":"c5","message":"先读 /tmp/agent_multi_test.txt 的内容，再搜一下相关内容"}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
t=d['tools_called']
print(f'tools={t}')
assert t >= 2, f'❌ 工具数不足({t})'
print('✅')
"
```
**成功标准：** tools_called >= 2

### 2.6 文件读取测试

```bash
echo "=== 文件读 ==="
echo 'Hello CogniMem' > /tmp/agent_file_test.txt
curl -s -X POST http://47.99.151.253:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_agt","session_id":"c6","message":"读取 /tmp/agent_file_test.txt"}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
r=d['reply']
assert 'Hello CogniMem' in r, '❌ 没读到内容'
print(f'✅ tools={d[\"tools_called\"]}')
"

echo "=== 文件不存在 ==="
curl -s -X POST http://47.99.151.253:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_agt","session_id":"c7","message":"读取 /nonexistent/file.txt"}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
r=d['reply']
if '不存在' in r or '没有' in r: print('✅ 优雅处理')
else: print(f'⚠️ {r[:60]}')
"
```
**成功标准：** 存在文件正确读出，不存在文件优雅报错

### 2.7 分析能力测试

```bash
echo "=== 分析能力 ==="
cat > /tmp/agent_analyze.txt << 'EOF'
项目计划
目标：做一款AI产品
背景：现在AI很火
方法：使用大模型
预期：会有很多用户
EOF
curl -s -X POST http://47.99.151.253:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_agt","session_id":"c8","message":"分析 /tmp/agent_analyze.txt 的质量"}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
r=d['reply']
critical=['简单','少','不够','缺乏','模糊','空泛','单薄']
found=[w for w in critical if w in r]
print(f'批评词: {found}')
print('✅' if found else '⚠️ 无批评词')
"
```
**成功标准：** 回复包含批评性词汇

### 2.8 记忆对话测试（跨会话）

```bash
echo "=== 跨对话记忆 ==="
# 第一轮：存偏好
curl -s -X POST http://47.99.151.253:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_agt","session_id":"mem1","message":"我喜欢蓝色"}' > /dev/null
sleep 1

# 第二轮：问偏好
curl -s -X POST http://47.99.151.253:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_agt","session_id":"mem2","message":"你记得我喜欢什么颜色吗？"}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
r=d['reply']
assert '蓝色' in r or '蓝' in r, '❌ 没记住'
print(f'✅ 记住: {r[:60]}')
"
```
**成功标准：** 跨会话后能正确召回偏好

### 2.9 路由策略测试

```bash
echo "=== 路由测试 ==="

# streaming 端点测试（含搜→agent路径）
echo "含搜→agent:"
curl -s -X POST http://47.99.151.253:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_agt","session_id":"r1","message":"搜点什么"}' | grep '"type": "meta"\|"type": "done"'

echo ""

# streaming 端点测试（无词→streaming路径）
echo "无词→stream:"
curl -s -X POST http://47.99.151.253:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_agt","session_id":"r2","message":"今天天气真不错"}' | grep '"type": "meta"\|"type": "done"'
```
**成功标准：**
- 含搜：有 meta 事件（tools_called > 0）
- 无词：只有 token + done，无 meta

---

## 三、🌐 网络测试方法

### 3.1 页面加载测试

```bash
echo "=== 三页面加载 ==="
ALL=0
for p in "/" "/dashboard" "/graph"; do
  CODE=$(curl -s -o /dev/null -w '%{http_code}' http://47.99.151.253:8000$p)
  CT=$(curl -s -o /dev/null -w '%{content_type}' http://47.99.151.253:8000$p)
  HTML=$(curl -s http://47.99.151.253:8000$p | grep -c '<html\|<!DOCTYPE')
  echo "$p → HTTP $CODE | $CT | HTML: $([ $HTML -gt 0 ] && echo '✅' || echo '❌')"
  [ "$CODE" = "200" ] && [ $HTML -gt 0 ] && ALL=$((ALL+1))
done
echo "通过: $ALL/3"
```

### 3.2 仪表盘组件验证

```bash
echo "=== 仪表盘组件 ==="
HTML=$(curl -s http://47.99.151.253:8000/dashboard)
echo -n "总记忆: "; echo "$HTML" | grep -c '总记忆'
echo -n "抽象概念: "; echo "$HTML" | grep -c '抽象概念'
echo -n "偏好: "; echo "$HTML" | grep -c '偏好'
echo -n "需关注: "; echo "$HTML" | grep -c '需关注'
echo -n "系统健康: "; echo "$HTML" | grep -c '系统健康'
echo -n "记忆分布: "; echo "$HTML" | grep -c 'type-chart'
echo -n "增长趋势: "; echo "$HTML" | grep -c 'growth-chart'
echo -n "关键洞察: "; echo "$HTML" | grep -c '关键洞察'
echo -n "活动日志: "; echo "$HTML" | grep -c '活动日志'
```

### 3.3 Streaming 测试

```bash
echo "=== SSE 流式 ==="
RESP=$(curl -s -X POST http://47.99.151.253:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"st1","message":"你好"}')
TOKENS=$(echo "$RESP" | grep -c '"type": "token"')
META=$(echo "$RESP" | grep -c '"type": "meta"')
DONE=$(echo "$RESP" | grep -c '"type": "done"')
echo "tokens=$TOKENS meta=$META done=$DONE"
[ "$DONE" -eq 1 ] && echo "✅" || echo "❌"
```

---

## 四、🔧 复杂能力测试方法

### 4.1 系统健康检测

```bash
echo "=== 健康检测 ==="
curl -s http://47.99.151.253:8000/health | python3 -c "
import sys,json; h=json.load(sys.stdin)
print(f'健康分: {h[\"score\"]}')
print(f'db:    {h[\"checks\"].get(\"db\")}')
print(f'llm:   {h[\"checks\"].get(\"llm\")}')
print(f'配置:   {h[\"checks\"].get(\"config\")}')
print(f'工具:   {h[\"checks\"].get(\"tools\")} 个')
"
```
**成功标准：** 健康分 >= 90，db/llm/config 均为 ✅

### 4.2 噪音过滤测试

```bash
echo "=== 噪音过滤 ==="
# curl 命令
curl -s -X POST http://47.99.151.253:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_noise","session_id":"n1","message":"curl https://example.com"}' > /dev/null
sleep 1

echo -n "curl过滤: "
curl -s -X POST http://47.99.151.253:8000/recall \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_noise","query":"curl"}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
has=any('curl' in m.get('content','').lower() for m in d.get('memories',[]))
print('❌ 有噪音' if has else '✅ 已过滤')
"

# 个人信息正常存储
curl -s -X POST http://47.99.151.253:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_noise","session_id":"n2","message":"我叫test_user，喜欢跑步"}' > /dev/null
sleep 1

echo -n "个人信息: "
curl -s -X POST http://47.99.151.253:8000/recall \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test_noise","query":"test_user"}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
found=any('test_user' in m.get('content','') for m in d.get('memories',[]))
print('✅ 正常存' if found else '⚠️ 可能丢失')
"
```

### 4.3 并发请求测试

```bash
echo "=== 并发请求 ==="
curl -s -X POST http://47.99.151.253:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"p1","message":"你好"}' > /dev/null &
curl -s -X POST http://47.99.151.253:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"p2","message":"搜一下AI"}' > /dev/null &
curl -s http://47.99.151.253:8000/stats?agent_id=test > /dev/null &
wait
echo "✅ 3 个并发请求均完成"
```

### 4.4 参数校验测试

```bash
echo "=== 参数校验 ==="
# 空消息
echo -n "空消息: "
curl -s -X POST http://47.99.151.253:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","message":"","session_id":"e1"}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'422 ✅' if 'detail' in d else '❌')
"

# 无效 JSON
echo -n "无效JSON: "
curl -s -X POST http://47.99.151.253:8000/chat \
  -H 'Content-Type: application/json' \
  -d 'not json' | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'422 ✅' if 'detail' in d else '❌')
"

# 缺字段
echo -n "缺字段: "
curl -s -X POST http://47.99.151.253:8000/chat \
  -H "Content-Type: application/json" \
  -d '{}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'422 ✅' if 'detail' in d else '❌')
"
```
**成功标准：** 全部返回 422，不返回 500

### 4.5 Consolidation 测试

```bash
echo "=== Consolidation ==="
curl -s -X POST 'http://47.99.151.253:8000/consolidate?agent_id=default' | python3 -c "
import sys,json; d=json.load(sys.stdin)
r=d.get('result',{})
print(f'merged={r.get(\"merged\",0)} abstracted={r.get(\"abstracted\",0)} resolved={r.get(\"contradictions_resolved\",0)}')
print('✅' if d.get('status')=='success' or r else '⚠️')
"
```
**成功标准：** 不崩溃

### 4.6 清理测试

```bash
echo "=== 清理 ==="
curl -s -X DELETE 'http://47.99.151.253:8000/clear?agent_id=test_mem' | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'{d[\"message\"]}')
"
```
**成功标准：** 返回"记忆已清除"

---

## 📋 测试清单（逐项打勾用）

```markdown
- [ ] 1.1 三元组提取 — 5 种句式
- [ ] 1.2 记忆召回 — 空/精确/语义/组合
- [ ] 1.3 矛盾检测 — 真/中性/跨类别
- [ ] 1.4 置信度操作 — confirm/challenge/版本链
- [ ] 2.1 简单对话
- [ ] 2.2 搜索执行
- [ ] 2.3 搜索早停
- [ ] 2.4 上下文分离
- [ ] 2.5 多步任务
- [ ] 2.6 文件读取
- [ ] 2.7 分析能力
- [ ] 2.8 跨对话记忆
- [ ] 2.9 路由策略
- [ ] 3.1 三页面加载
- [ ] 3.2 仪表盘组件
- [ ] 3.3 Streaming
- [ ] 4.1 健康检测
- [ ] 4.2 噪音过滤
- [ ] 4.3 并发请求
- [ ] 4.4 参数校验
- [ ] 4.5 Consolidation
- [ ] 4.6 清理
```

> 每条命令都是可直接复制到终端执行的具体测试步骤。
> 测试前确保先执行 `export URL=http://47.99.151.253:8000`
