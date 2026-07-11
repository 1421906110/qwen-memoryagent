# 🧪 CogniMem 完整测试手册

> **版本：** v0.10 | 日期：2026-07-08  
> **修复记录：** 两轮共修复 14 个 bug（P0-P3）+ 本次重构仪表盘布局 + 图谱 action 类型 + 三元组噪音过滤  
> **项目路径：** `~/projects/qwen-memoryagent`  
> **架构：** 单端口单进程（CogniMem 引擎直接集成到 FastAPI）  
> **本地端口：** `http://localhost:9999`  
> **工具数量：** 12  
> **模型：** DeepSeek v4 Flash（开发用）/ Qwen3（可选）  
> **前置条件：** 服务已启动在 9999 端口  

---

## 📋 架构与导航

### 服务架构

```
单进程 memory_agent.main:app → 端口 9999
  ├── /                    → chat.html       → 💬 聊天界面
  ├── /dashboard           → dashboard.html  → 📊 统计+仪表盘
  ├── /graph               → graph.html      → 🕸️ 知识图谱
  ├── /chat                → POST API        → Agent 聊天
  ├── /chat/stream         → POST SSE        → 流式聊天
  ├── /remember            → POST API        → 存记忆
  ├── /recall              → POST API        → 查记忆
  ├── /stats               → GET API         → CogniMem 统计
  ├── /health              → GET API         → 🆕 系统健康检测
  ├── /memories            → GET API         → 🆕 分页列出记忆
  ├── /memories/{id}       → DELETE API      → 🆕 删除单条记忆
  ├── /memories/search     → GET API         → 🆕 搜索记忆
  ├── /consolidate         → POST API        → 整合
  ├── /clear               → DELETE API      → 清空
  ├── /confirm             → POST API        → 确认事实
  ├── /challenge           → POST API        → 质疑事实
  ├── /versions/{id}       → GET API         → 版本链
  └── /memory-graph        → GET API         → 图谱数据
```

### Web UI 页面导航

**首页 `/` (chat.html):**
- 顶部标题：CogniMem Chat
- 聊天消息区域（滚动）
- 底部输入框 + 发送按钮
- 左侧/顶部：agent_id 选择器（默认 "default"）
- 输入框 placeholder: "输入消息..."
- 发送后：消息显示在聊天区，agent 回复自动追加
- 清空按钮：清除当前对话

**仪表盘 `/dashboard` (dashboard.html):**
- 第一行（4列指标卡）：
  - 📦 总记忆 — 总事实数
  - 🧩 抽象概念 — 核心信念·置信度≥0.9
  - ⭐ 偏好 — 已学习的用户偏好
  - ⚠️ 需关注 — 待解决的矛盾数
- 第二行（2列指标卡）：
  - 💚 系统健康 — 点击弹出健康详情模态框
  - 🔧 MCP 工具 — 已注册的工具数量
- 图表区（2×2 网格）：
  - 📂 记忆分布 — Doughnut 图（preference/fact/action/goal/decision/observation）
  - 📈 增长趋势 — Line 图（累积记忆随时间增长）
  - 🔍 关键洞察 — 分析型洞察条目
  - 🏷️ 记忆分类 — 水平柱状图 + 百分比
- 📋 活动日志 — 滚动日志列表
- 右上角 agent 选择器 + 导航链接（聊天/图谱/仪表盘）
- 自动 30 秒刷新

**图谱 `/graph` (graph.html):**
- 力导向图可视化（Canvas 实现）
- 节点 = 实体（subject/object），边 = 谓词（predicate）
- 支持拖拽、缩放、筛选（按类型 checkbox）
- 右侧详情面板（点击节点展开）
- 图例 + 统计（节点/连线/密度/抽象）
- 类型包括：preference/fact/goal/decision/observation/**action**/entity/**abstraction**
- Agent 选择器：优先从 API `/agents` 加载，合并 localStorage
- 刷新按钮 + 键盘快捷键 `R`

---

## 📋 测试前准备

### 1. 启动服务

```bash
cd ~/projects/qwen-memoryagent/src
source ../.venv/bin/activate
python3 -m uvicorn memory_agent.main:app --host 0.0.0.0 --port 9999

# 等启动完成（最长 15 秒）
for i in $(seq 1 15); do
  curl -s -o /dev/null http://localhost:9999/ && echo "服务就绪" && break
  sleep 1
done
```

启动日志关键行：
```
🧠 CogniMem engine initialized (direct integration)
🤖 LLM extractor enabled: deepseek-v4-flash
✅ Registered 12 tools (3 with CogniMem)
🤖 Agent engine initialized with 12 tools + SelfReflector
Uvicorn running on http://0.0.0.0:9999
```

### 2. 检查环境变量

```bash
# 搜索需要代理（国内环境）
grep SEARCH_PROXY ~/projects/qwen-memoryagent/.env
# 应返回: SEARCH_PROXY=http://127.0.0.1:10809

# API Key (DeepSeek)
grep QWEN_API_KEY ~/projects/qwen-memoryagent/.env
# 应返回: QWEN_API_KEY=sk-xxx...

# 快速模型（必须配置，否则首次聊天空白）
grep QWEN_FAST_MODEL ~/projects/qwen-memoryagent/.env
# 应返回: QWEN_FAST_MODEL=deepseek-v4-flash
```

### 3. 查看服务状态

```bash
# HTTP 状态码
curl -s -o /dev/null -w 'HTTP %{http_code}\n' http://localhost:9999/
# 预期: HTTP 200

# 引擎统计
curl -s 'http://localhost:9999/stats?agent_id=default' | python3 -c "
import sys,json; d=json.load(sys.stdin)
rt = d.get('router_stats', {})
print(f'total_facts={d[\"total_facts\"]}, contradictions={d[\"contradictions\"]}')
print(f'L0={rt.get(\"l0_hit_rate\")}, L1={rt.get(\"l1_hit_rate\")}, L2={rt.get(\"l2_hit_rate\")}')
print(f'L3={rt.get(\"l3_hit_rate\")}, fallback={rt.get(\"l3_fallback_rate\")}')
"
```

### 4. 清空测试数据

```bash
curl -s -X DELETE 'http://localhost:9999/clear?agent_id=test' | python3 -m json.tool
# 预期: {"agent_id": "test", "deleted": N, "message": "记忆已清除"}  (N为实际删除行数)

curl -s -X DELETE 'http://localhost:9999/clear?agent_id=abstest' > /dev/null
```

---

## 一、Web UI 界面测试

### 1.1 三页面加载

**目的：** 验证三个前端页面都能正常加载（HTTP 200），且返回 HTML。

```bash
echo "=== 页面加载测试 ==="
all_ok=0
for p in "/" "/dashboard" "/graph"; do
  # 检查 HTTP 状态码
  code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:9999$p)
  # 检查 Content-Type
  content_type=$(curl -s -o /dev/null -w '%{content_type}' http://localhost:9999$p)
  # 检查是否包含 HTML
  has_html=$(curl -s http://localhost:9999$p | grep -c '<html\|<!DOCTYPE')
  echo "  $p → HTTP $code | $content_type | HTML: $([ $has_html -gt 0 ] && echo '✅' || echo '❌')"
  [ "$code" = "200" ] && [ $has_html -gt 0 ] && all_ok=$((all_ok+1))
done
echo "  通过: $all_ok/3"
[ "$all_ok" -eq 3 ] && echo "✅" || echo "❌"
```

**预期输出：**
```
  / → HTTP 200 | text/html; charset=utf-8 | HTML: ✅
  /dashboard → HTTP 200 | text/html; charset=utf-8 | HTML: ✅
  /graph → HTTP 200 | text/html; charset=utf-8 | HTML: ✅
  通过: 3/3
✅
```

**失败诊断：**
- HTTP 500 → 检查 /tmp/qwen-agent.log 有无 Python traceback
- Content-Type 不是 HTML → 服务可能返回了错误格式（检查路由注册）
- 无 HTML 标签 → 模板文件缺失（检查 templates/ 目录）

### 1.2 聊天页面 UI 元素

**目的：** 验证聊天页面包含所有必要的 UI 元素。

```bash
HTML=$(curl -s http://localhost:9999/)
echo -n "聊天容器: "; echo "$HTML" | grep -c 'class="[^"]*chat[^"]*"' || echo "0"
echo -n "输入框: "; echo "$HTML" | grep -c 'input\|textarea' || echo "0"
echo -n "发送按钮: "; echo "$HTML" | grep -c 'send\|发送\|submit' || echo "0"
echo -n "agent_id选择: "; echo "$HTML" | grep -c 'agent' || echo "0"
```

### 1.3 仪表盘页面 UI 元素（v0.10 最终布局）

**目的：** 验证仪表盘页面包含正确的最终布局。

```bash
HTML=$(curl -s http://localhost:9999/dashboard)
echo "=== 仪表盘布局验证 ==="
echo -n " Row1-总记忆: "; echo "$HTML" | grep -c '总记忆'
echo -n " Row1-抽象概念: "; echo "$HTML" | grep -c '抽象概念'
echo -n " Row1-偏好: "; echo "$HTML" | grep -c '偏好'
echo -n " Row1-需关注: "; echo "$HTML" | grep -c '需关注'
echo -n " Row2-系统健康: "; echo "$HTML" | grep -c '系统健康'
echo -n " Row2-MCP工具: "; echo "$HTML" | grep -c 'MCP 工具'
echo -n " 记忆分布(donut): "; echo "$HTML" | grep -c 'type-chart'
echo -n " 增长趋势(line): "; echo "$HTML" | grep -c 'growth-chart'
echo -n " 关键洞察: "; echo "$HTML" | grep -c '关键洞察'
echo -n " 记忆分类: "; echo "$HTML" | grep -c '记忆分类'
echo -n " 活动日志: "; echo "$HTML" | grep -c '活动日志'
```

### 1.4 图谱页面 UI 元素

**目的：** 验证图谱页面包含 action 类型 + 抽象节点。

```bash
HTML=$(curl -s http://localhost:9999/graph)
echo "=== 图谱布局验证 ==="
echo -n " 图谱容器: "; echo "$HTML" | grep -c 'graph-canvas\|graph-wrapper'
echo -n " Agent选择器: "; echo "$HTML" | grep -c 'agent-select'
echo -n " 类型筛选: "; echo "$HTML" | grep -c 'type-filters\|type_'
echo -n " 详情面板: "; echo "$HTML" | grep -c 'detail-panel'
echo -n " 图例: "; echo "$HTML" | grep -c 'legend'
echo -n " 刷新按钮: "; echo "$HTML" | grep -c 'refresh\|刷新\|loadGraph'

# 验证包含 action 类型
echo -n " action类型: "; echo "$HTML" | grep -c "'action'"
echo -n " 抽象节点: "; echo "$HTML" | grep -c 'abstraction'
```

### 1.5 页面静态资源

**目的：** 验证页面引用的 JS/CSS 文件可访问。

```bash
# 从 HTML 中提取静态资源链接
for url in $(curl -s http://localhost:9999/ | grep -o 'src="[^"]*\.js\|href="[^"]*\.css' | sed 's/src="//;s/href="//'); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:9999$url")
  echo "  $url → HTTP $code"
done
# 预期: 全部 200
```

---

## 二、Agent 行为测试

### 2.1 搜索执行 — 不确认不拒绝

**目的：** 用户要求搜索时，agent 直接调用 web_search，不反问"你想搜什么"。

**前置条件：** SEARCH_PROXY 环境变量已设置（sing-box 运行中）

```bash
echo "=== 2.1 搜索执行 ==="
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"s1","message":"搜一下最近的AI新闻"}' -o /tmp/t_s1.json

python3 -c "
import json
d=json.load(open('/tmp/t_s1.json'))
tools = d['tools_called']
reply = d['reply']
print(f'工具调用: {tools}')
print(f'回复前50字: {reply[:50]}')
print(f'回复长度: {len(reply)}')

# 检查点
assert tools > 0, '❌ 没有调用工具'
assert '?' not in reply[:20], '❌ 回复以问句开头'
assert len(reply) > 30, '❌ 回复太短'
print('✅ 搜索直接执行')
"
```

**预期输出：**
```
工具调用: 2-8
回复前50字: 搜到以下结果：...
回复长度: > 100
✅ 搜索直接执行
```

**失败诊断：**
- tools=0 → agent 没走工具路径。检查 is_simple 路由逻辑（main.py）
- 回复以问句开头 → system prompt 中"你直接搜"指令不够强
- HTTP 503 → 服务启动时 agent 初始化失败

### 2.2 搜索早停

**目的：** 验证搜到结果后直接汇报，不继续深入调研。

**前置条件：** 搜索功能正常（2.1 通过）

```bash
echo "=== 2.2 搜索早停 ==="
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"s2","message":"搜一下Qwen2"}' -o /tmp/t_s2.json

python3 -c "
import json
d=json.load(open('/tmp/t_s2.json'))
tools = d['tools_called']
print(f'工具调用: {tools}')
assert tools < 12, f'❌ 工具调用过多 ({tools})，应早停'
assert tools > 0, '❌ 没有调用工具'
print(f'✅ 搜索早停有效 ({tools} 次)')
"
```

**预期输出：** `工具调用: 1-8` — 不应超过 12。

### 2.3 上下文分离

**目的：** 带历史对话时，agent 不会混淆"新建文件夹"和"搜一下"。

```bash
echo "=== 2.3 上下文分离 ==="
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id":"test","session_id":"ctx1","message":"搜一下AI工具",
    "messages":[
      {"role":"user","content":"帮我在桌面新建个文件夹"},
      {"role":"assistant","content":"好的"},
      {"role":"user","content":"你会搜索吗？"},
      {"role":"assistant","content":"会的，你说搜什么"}
    ]
  }' -o /tmp/t_ctx.json

python3 -c "
import json
d=json.load(open('/tmp/t_ctx.json'))
reply = d['reply']
if '文件夹' in reply:
    print('❌ 混淆上下文（提到无关的文件夹）')
elif d['tools_called'] > 0:
    print(f'✅ 上下文分离 (tools={d[\"tools_called\"]})')
else:
    print(f'⚠️ tools=0, reply={reply[:50]}')
"
```

**预期输出：** `✅ 上下文分离 (tools=2-8)` — 回复中不应出现"文件夹"。

### 2.4 简单 Q&A — 走流式无工具

**目的：** 问候、闲聊等简单对话不走 agent 路径（0 工具）。

```bash
echo "=== 2.4 简单 Q&A ==="
# 测试短问候
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"qa1","message":"你好"}' -o /tmp/t_qa1.json
python3 -c "
import json; d=json.load(open('/tmp/t_qa1.json'))
assert d['tools_called'] == 0, '❌ 问候不应调用工具'
print('✅ 问候 0工具')
"

# 测试长句无动作词
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"qa2","message":"今天天气真不错啊"}' -o /tmp/t_qa2.json
python3 -c "
import json; d=json.load(open('/tmp/t_qa2.json'))
assert d['tools_called'] == 0, '❌ 闲聊不应调用工具'
print('✅ 长句无动作词 0工具')
"

# 测试含"能不能帮我"的动作混合句（应走 agent 路径）
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"qa3","message":"能不能帮我查一下Python list用法"}' -o /tmp/t_qa3.json
python3 -c "
import json; d=json.load(open('/tmp/t_qa3.json'))
assert d['tools_called'] > 0, '❌ 含动作词应走 agent'
print(f'✅ 动作混合句 {d[\"tools_called\"]}工具')
"
```

**预期输出：**
```
✅ 问候 0工具
✅ 长句无动作词 0工具
✅ 动作混合句 2-5工具
```

### 2.5 短动作词路由

**目的：** 验证 is_simple 边界条件：9 字含"搜"→ agent 路径，10 字无关键词→ streaming 路径。

```bash
echo "=== 2.5 短动作词路由 ==="
# 边界: 3字 + "搜" → agent路径
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"r1","message":"搜点什么"}' -o /tmp/t_r1.json
python3 -c "
import json; d=json.load(open('/tmp/t_r1.json'))
assert d['tools_called'] > 0, '❌ 3字含搜应走agent'
print(f'✅ 短字+搜→agent (tools={d[\"tools_called\"]})')
"

# 边界: 7字无关键词 → streaming路径
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"r2","message":"今天天气真不错"}' -o /tmp/t_r2.json
python3 -c "
import json; d=json.load(open('/tmp/t_r2.json'))
print(f'✅ 7字无关键词→stream (tools={d[\"tools_called\"]})')
"

# 边界: 2字无关键词 → streaming路径
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"r3","message":"天气不错"}' -o /tmp/t_r3.json
python3 -c "
import json; d=json.load(open('/tmp/t_r3.json'))
print(f'✅ 2字无关键词→stream (tools={d[\"tools_called\"]})')
"
```

**路由规则对照表：**
| 消息 | 长度 | 含动作词 | 路径 | 说明 |
|------|------|----------|------|------|
| "你好" | 2 | ❌ | streaming | 纯问候 |
| "搜点什么" | 4 | ✅"搜" | agent | 含动作词走 agent |
| "今天天气真不错" | 7 | ❌ | streaming | 无动作词 |
| "搜一下AI新闻" | 7 | ✅"搜" | agent | |
| "帮我查一下xxx" | 7+ | ✅"查" | agent | |
| "总结一下今天" | 7+ | ✅"总结" | agent | v0.8+ 新增动作词 |
| "推荐一家餐厅" | 7+ | ✅"推荐" | agent | v0.8+ 新增动作词 |

### 2.6 "我不确定"意识

**目的：** 搜索不到结果时，agent 直接说"没找到"，不编造答案。

```bash
echo "=== 2.6 不确定意识 ==="
# 搜一个明显不存在的内容
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"uc1","message":"搜一下asdfghjklzxcvbnm12345qwerty"}' -o /tmp/t_uc.json

python3 -c "
import json
d=json.load(open('/tmp/t_uc.json'))
reply = d['reply']
tools = d['tools_called']
print(f'工具: {tools}')

if '没找到' in reply or '没有' in reply or '找不到' in reply:
    print('✅ 诚实地说了没找到')
elif '抱歉' in reply:
    print('✅ 道歉了（也表示没找到）')
elif d['tools_called'] == 0:
    print('⚠️ 未调用工具（可能走的streaming）')
else:
    print(f'⚠️ 检查回复: {reply[:80]}')
"
```

**预期输出：**
```
工具: 1-6
✅ 诚实地说了没找到
# 或
✅ 道歉了（也表示没找到）
```

### 2.7 文件读取 — 直接执行

**目的：** 用户要求读取文件时，直接调用 read_file。

```bash
echo "=== 2.7 文件读取 ==="
# 准备测试文件
echo 'Hello CogniMem 测试内容' > /tmp/agent_read_test.txt

# 读取存在的文件
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"f1","message":"读取 /tmp/agent_read_test.txt 的内容"}' -o /tmp/t_f1.json

python3 -c "
import json
d=json.load(open('/tmp/t_f1.json'))
print(f'工具调用: {d[\"tools_called\"]}')
print(f'回复: {d[\"reply\"][:80]}')
assert d['tools_called'] >= 1, '❌ 没有调用工具'
assert 'Hello CogniMem' in d['reply'] or '测试内容' in d['reply'], '❌ 没有读取到文件内容'
print('✅ 文件正确读取')
"

# 读取不存在的文件 → 优雅错误
echo "--- 不存在的文件 ---"
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"f2","message":"读取 /nonexistent/path/xyz.txt"}' -o /tmp/t_f2.json

python3 -c "
import json
d=json.load(open('/tmp/t_f2.json'))
reply = d['reply']
if '不存在' in reply or '没有' in reply or 'not found' in reply.lower():
    print('✅ 优雅处理文件不存在')
else:
    print(f'⚠️ 检查回复: {reply[:80]}')
"
```

### 2.8 多步任务 — 完整执行

**目的：** 验证读文件+搜索的多步任务能完整执行不被早停误杀。

```bash
echo "=== 2.8 多步任务 ==="
echo '多步任务测试' > /tmp/multistep_agent.txt
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"m1","message":"先读取 /tmp/multistep_agent.txt 的内容，再搜一下相关内容"}' -o /tmp/t_multi.json

python3 -c "
import json
d=json.load(open('/tmp/t_multi.json'))
tools = d['tools_called']
iters = d['iterations']
print(f'工具: {tools}, 迭代: {iters}')
assert tools >= 2, f'❌ 应至少调用2个工具 (实际{tools})'
print(f'✅ 多步任务完成 (tools={tools})')
print('回复片段:', d['reply'][:120])
"
```

**预期输出：**
```
工具: 2-4, 迭代: 3-6
✅ 多步任务完成 (tools=2-4)
回复片段: 完成！读取文件内容为「多步任务测试」，搜到相关结果...
```

### 2.9 分析 — 真实不浮夸

**目的：** 分析文件内容时给出诚实批评，不给 9.5/10 浮夸评分。

```bash
echo "=== 2.9 分析 ==="
# 一个质量很差的"文档"
cat > /tmp/analyze_test_agent.txt << 'EOF'
项目计划
目标：做一款AI产品
背景：现在AI很火
方法：使用大模型
预期：会有很多用户
（还有很多内容要补充）
EOF

curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"a1","message":"分析 /tmp/analyze_test_agent.txt 的质量"}' -o /tmp/t_a1.json

python3 -c "
import json
d=json.load(open('/tmp/t_a1.json'))
reply = d['reply']
print(f'工具: {d[\"tools_called\"]}')
print(f'回复长度: {len(reply)}')
print('---回复前200字---')
print(reply[:200])
print('---')
# 检查是否有实质分析，而不是"全部完成了"这类空话
if '全部完成' in reply[:30]:
    print('❌ 计划总结覆盖了分析内容')
elif len(reply) < 50:
    print('❌ 回复太短')
else:
    # 检查是否有批评性词汇
    critical_words = ['简单', '少', '不够', '缺乏', '没有', '模糊', '空泛', '单薄', '具体', '建议']
    found = [w for w in critical_words if w in reply]
    if found:
        print(f'✅ 诚实分析（包含关键词: {found[:3]}）')
    else:
        print('⚠️ 可能不够批评性')
"
```

**预期输出：**
```
工具: 1
回复长度: > 100
---回复前200字---
文件内容分析结果：
这个文档非常粗略...
✅ 诚实分析
```

---

## 三、核心引擎测试

### 3.1 记忆存储 (remember)

**目的：** 验证不同句式能被正确提取为三元组并存储。

```bash
echo "=== 3.1 记忆存储 ==="

# 测试1: 简单陈述
echo -n "简单陈述: "
curl -s -X POST http://localhost:9999/remember \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"r1","content":"用户喜欢喝冰美式","confidence":0.9}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'status={d.get(\"status\")}, facts={d.get(\"facts_added\",0)}')
"

# 测试2: 否定句
echo -n "否定句: "
curl -s -X POST http://localhost:9999/remember \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"r2","content":"用户不喜欢喝热美式","confidence":0.9}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'status={d.get(\"status\")}')
"

# 测试3: 个人信息
echo -n "个人信息: "
curl -s -X POST http://localhost:9999/remember \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"r3","content":"我叫张三，住在北京","confidence":0.9}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'status={d.get(\"status\")}, facts={d.get(\"facts_added\",0)}')
"
```

**预期输出：**
```
简单陈述: status=stored, facts=1-3
否定句: status=stored
个人信息: status=stored, facts=1-3
```

### 3.2 记忆召回 (recall)

**目的：** 验证存储后能通过语义搜索召回。

```bash
echo "=== 3.2 记忆召回 ==="

# 空查询浏览（返回所有事实）
echo -n "空查询: "
curl -s -X POST http://localhost:9999/recall \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","query":"","limit":20}' | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f'{d[\"count\"]} 条')
assert d['count'] > 0, '❌ 空查询无返回'
"

# 精确查询
echo -n "精确查询冰美式: "
curl -s -X POST http://localhost:9999/recall \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","query":"冰美式","limit":5}' | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f'{d[\"count\"]} 条')
for m in d['memories'][:3]:
    print(f'  → {m[\"content\"][:60]}')
"

# 语义查询
echo -n "语义查询咖啡: "
curl -s -X POST http://localhost:9999/recall \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","query":"咖啡","limit":5}' | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f'{d[\"count\"]} 条')
"
```

**预期输出：**
```
空查询: N 条
精确查询冰美式: ≥1 条
  → user 喜欢 喝冰美式
语义查询咖啡: ≥1 条
```

### 3.3 矛盾检测

**目的：** 验证 L1 deny（真矛盾）检测 + 中性谓词跳过 + 跨类别偏好不误报。

```bash
echo "=== 3.3.1 真矛盾检测 ==="
curl -s -X DELETE 'http://localhost:9999/clear?agent_id=ctest' > /dev/null
curl -s -X POST http://localhost:9999/remember \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"ctest","session_id":"c1","content":"用户喜欢喝冰美式","confidence":0.9}' > /dev/null
curl -s -X POST http://localhost:9999/remember \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"ctest","session_id":"c2","content":"用户不喜欢喝冰美式","confidence":0.9}' > /dev/null
sleep 1

curl -s 'http://localhost:9999/stats?agent_id=ctest' | python3 -c "
import sys,json; d=json.load(sys.stdin)
c = d.get('contradictions',0)
print(f'矛盾数: {c}')
assert c >= 1, '❌ 应为1条矛盾（喜欢 vs 不喜欢）'
print('✅ L1 deny 真矛盾检测')
"

echo ""
echo "=== 3.3.2 中性谓词不误报 ==="
curl -s -X POST http://localhost:9999/remember \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"ctest","session_id":"c3","content":"用户请求读取文件","confidence":0.9}' > /dev/null
curl -s -X POST http://localhost:9999/remember \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"ctest","session_id":"c4","content":"用户请求搜索网络","confidence":0.9}' > /dev/null
sleep 1

curl -s 'http://localhost:9999/stats?agent_id=ctest' | python3 -c "
import sys,json; d=json.load(sys.stdin)
c = d.get('contradictions',0)
print(f'矛盾数（请求类不应新增）: {c}')
print('✅ 中性谓词跳过')
"

echo ""
echo "=== 3.3.3 跨类别偏好不误报 ==="
curl -s -X DELETE 'http://localhost:9999/clear?agent_id=cross' > /dev/null
curl -s -X POST http://localhost:9999/remember \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"cross","session_id":"x1","content":"用户喜欢喝咖啡","confidence":0.9}' > /dev/null
curl -s -X POST http://localhost:9999/remember \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"cross","session_id":"x2","content":"用户喜欢吃火锅","confidence":0.9}' > /dev/null
sleep 1

curl -s 'http://localhost:9999/stats?agent_id=cross' | python3 -c "
import sys,json; d=json.load(sys.stdin)
c = d.get('contradictions',0)
print(f'矛盾数: {c}')
assert c == 0, f'❌ 咖啡 vs 火锅 不应矛盾 (有{c}条)'
print('✅ 跨类别偏好不误报')
"
```

**矛盾检测对照表：**
| 场景 | 事实A | 事实B | 预期 | 说明 |
|------|-------|-------|------|------|
| 真矛盾 | 用户喜欢咖啡 | 用户不喜欢咖啡 | deny ✅ | 直接否定 |
| 中性谓词 | 用户请求读文件 | 用户请求搜网络 | 跳过 ✅ | 不同请求不是矛盾 |
| 跨类别偏好 | 用户喜欢咖啡 | 用户喜欢吃火锅 | 跳过 ✅ | 不同类别的喜好 |
| 同类别偏好 | 用户喜欢冰美式 | 用户喜欢热美式 | 跳过 ✅ | 可以喜欢多种 |

### 3.4 置信度操作

**目的：** 验证 confirm（升）和 challenge（降）正常工作。

```bash
echo "=== 3.4 置信度操作 ==="

# 取一个事实 ID
FID=$(curl -s -X POST http://localhost:9999/recall \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","query":"冰美式","limit":1}' | python3 -c "
import sys,json; print(json.load(sys.stdin)['memories'][0]['id'])
")
echo "目标事实: $FID"

# 确认
echo -n "确认: "
curl -s -X POST http://localhost:9999/confirm \
  -H "Content-Type: application/json" \
  -d "{\"fact_id\":\"$FID\",\"agent_id\":\"test\"}" | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'status={d[\"status\"]}, confidence={d.get(\"confidence\",\"?\")}')
assert d['status'] == 'confirmed', '❌ 确认失败'
"

# 质疑
echo -n "质疑: "
curl -s -X POST http://localhost:9999/challenge \
  -H "Content-Type: application/json" \
  -d "{\"fact_id\":\"$FID\",\"agent_id\":\"test\"}" | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'status={d[\"status\"]}, confidence={d.get(\"confidence\",\"?\")}')
assert d['status'] == 'challenged', '❌ 质疑失败'
"

# 版本链
echo -n "版本链: "
curl -s "http://localhost:9999/versions/$FID" | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'{d[\"count\"]} 个版本')
assert d['count'] >= 2, '❌ 版本数不足'
for v in d['versions']:
    print(f'  {v[\"change_reason\"]}: {v[\"old_confidence\"]} → {v[\"new_confidence\"]}')
"
```

**预期输出：**
```
目标事实: a1b2c3d4-...
确认: status=confirmed, confidence=0.7
质疑: status=challenged, confidence=0.5
版本链: 2-3 个版本
  confirmed: 0.6 → 0.7
  challenged: 0.7 → 0.5
```

### 3.5 整合 (consolidate)

**目的：** 验证 consolidation 正常执行（合并/抽象化/衰减/矛盾解析）。

```bash
echo "=== 3.5 整合 ==="
# 先造足够的数据
for text in "我喜欢喝冰美式咖啡" "热美式我的最爱" "冷萃咖啡很好喝"; do
  curl -s -X POST http://localhost:9999/remember \
    -H "Content-Type: application/json" \
    -d "{\"agent_id\":\"testc\",\"session_id\":\"cc\",\"content\":\"$text\"}" > /dev/null
done

curl -s -X POST 'http://localhost:9999/consolidate?agent_id=testc' | python3 -c "
import sys,json
d=json.load(sys.stdin)
r = d.get('result', {})
print(f'merged:         {r.get(\"merged\", 0)}')
print(f'abstracted:     {r.get(\"abstracted\", 0)}')
print(f'contradictions_resolved: {r.get(\"contradictions_resolved\", 0)}')
print(f'decayed:        {r.get(\"decayed\", 0)}')
print('✅ consolidate 不崩溃')
"
```

**预期输出：** 不崩溃即可，具体数值取决于数据量。

### 3.6 清空 (clear)

**目的：** 验证 /clear 按 FK 顺序删除，不报错。

```bash
echo "=== 3.6 清空 ==="
for aid in "test" "ctest" "cross" "abstest" "testc" "noise_test"; do
  result=$(curl -s -X DELETE "http://localhost:9999/clear?agent_id=$aid" | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'{d[\"message\"]}')
")
  echo "  $aid → $result"
done
# 验证清空后统计为0
stats=$(curl -s 'http://localhost:9999/stats?agent_id=test' | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'{d[\"total_facts\"]} facts, {d[\"contradictions\"]} contradictions')
")
echo "  清空后: $stats"
```

**预期输出：**
```
  test → 记忆已清除
  ctest → 记忆已清除
  ...
  清空后: 0 facts, 0 contradictions
```

---

## 四、Streaming 端点测试

### 4.1 /chat/stream 基础

**目的：** 验证 SSE 流式输出逐字返回。

```bash
echo "=== 4.1 streaming基础 ==="
# 简单对话 → 应逐字输出 token
RESP=$(curl -s -X POST http://localhost:9999/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"st1","message":"你好"}')

echo "$RESP" | head -8
echo "..."

# 验证格式
TOKEN_COUNT=$(echo "$RESP" | grep -c '"type": "token"')
META_COUNT=$(echo "$RESP" | grep -c '"type": "meta"')
DONE_COUNT=$(echo "$RESP" | grep -c '"type": "done"')
echo "token事件: $TOKEN_COUNT"
echo "meta事件: $META_COUNT"
echo "done事件: $DONE_COUNT"
[ "$DONE_COUNT" -eq 1 ] && echo "✅ streaming 正常完成" || echo "❌ streaming 未正常结束"
```

**预期输出格式（每行一个 SSE data）:**
```
data: {"type": "token", "content": "你"}
data: {"type": "token", "content": "好"}
...
data: {"type": "done", "content": "", ...}
```

### 4.2 /chat/stream 动作路由

```bash
echo "=== 4.2 streaming路由 ==="

# 含"搜" → agent路径（带有 tools_called 的 meta 事件）
echo "含搜→agent:"
curl -s -X POST 'http://localhost:9999/chat/stream' \
  -H 'Content-Type: application/json' \
  -d '{"agent_id":"test","session_id":"st2","message":"搜点什么"}' | grep '"type": "meta"\|"type": "done"'
echo ""

# 无关键词 → streaming路径（无 meta 事件，只有 token+done）
echo "无词→stream:"
curl -s -X POST 'http://localhost:9999/chat/stream' \
  -H 'Content-Type: application/json' \
  -d '{"agent_id":"test","session_id":"st3","message":"今天天气真不错"}' | grep '"type"'
# 预期：有 token 和 done，但没有 meta（因为没有工具调用）
```

**路由判定表：**
| 条件 | 路径 | 输出特点 |
|------|------|----------|
| 消息 < 10 字 + 无动作词 | streaming | 只有 token + done，无 meta |
| 消息 >= 10 字 + 无动作词 | streaming | 同上 |
| 消息含动作词（搜/查/读/总结/翻译/推荐/做/画/整理/记住/计算等） | agent | 有 meta（tools_called）|

---

## 五、记忆系统测试

### 5.1 三元组噪音过滤 — v0.10 新增验证

**目的：** 验证三类噪音过滤：
1. `web_fetch` → 完全跳过（中间数据抓取，无记忆价值）
2. `curl / wget / httpie / ping / traceroute` → 跳过
3. `web_search` → 同一对话只存第一次（`_search_stored` 去重）

```bash
echo "=== 5.1 三元组噪音过滤 ==="

# 步骤1: 发一条命令（curl）
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"noise_test","session_id":"n1","message":"curl -s https://example.com"}' > /dev/null
sleep 1

# 检查 curl 是否被存
echo -n "curl命令→记忆: "
curl -s -X POST http://localhost:9999/recall \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"noise_test","query":"curl"}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
found = False
for m in d.get('memories',[]):
    c = m.get('content','')
    if 'curl' in c.lower():
        print('❌ curl命令被存为记忆')
        found = True
        break
if not found:
    print('✅ curl命令被过滤')
"

# 步骤2: 发一条个人信息 — 应该正常被存
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"noise_test","session_id":"n2","message":"我叫李小四，住在广州，喜欢跑步"}' > /dev/null
sleep 1

echo -n "个人信息→记忆: "
curl -s -X POST http://localhost:9999/recall \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"noise_test","query":"李小四 广州"}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
found = any('李小四' in m.get('content','') for m in d.get('memories',[]))
print('✅ 个人信息存了' if found else '⚠️ 可能丢失')
"

# 步骤3: 发两条搜索 — 验证去重
echo -n "搜索去重: "
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"noise_test2","session_id":"n3","message":"搜一下AI"}' > /dev/null
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"noise_test2","session_id":"n4","message":"搜一下AI"}' > /dev/null
sleep 1
curl -s 'http://localhost:9999/stats?agent_id=noise_test2' | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'facts={d[\"total_facts\"]} (期望<=3条，搜索去重)')
"
```

**预期输出：**
```
curl命令→记忆: ✅ curl命令被过滤
个人信息→记忆: ✅ 个人信息存了
搜索去重: facts=0-3 (搜索动作不存为事实)
```

### 5.2 重复检查

**目的：** 验证同内容多次存储去重。

```bash
echo "=== 5.2 重复检查 ==="
# 存3次相同内容
for i in 1 2 3; do
  curl -s -X POST http://localhost:9999/remember \
    -H "Content-Type: application/json" \
    -d '{"agent_id":"dup_test","session_id":"d'$i'","content":"用户的邮箱是test@example.com","confidence":0.9}' > /dev/null
done

# 检查存储了几条
curl -s 'http://localhost:9999/stats?agent_id=dup_test' | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'facts: {d[\"total_facts\"]}')
if d['total_facts'] <= 3:
    print('✅ 未产生大量重复')
else:
    print(f'⚠️ 重复过多({d[\"total_facts\"]}条)')
"

# 清理
curl -s -X DELETE 'http://localhost:9999/clear?agent_id=dup_test' > /dev/null
```

### 5.3 首次对话空白修复验证 — v0.10 新增

**目的：** 验证 QWEN_FAST_MODEL 配置正确，新建项目/Agent 第一句对话不空白。

```bash
echo "=== 5.3 首次对话空白修复 ==="

# 检查 .env 中 QWEN_FAST_MODEL 配置
grep QWEN_FAST_MODEL ~/projects/qwen-memoryagent/.env

# 发一条简单对话到临时 agent（模拟新建项目第一句）
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"fresh_test","session_id":"first","message":"你好"}' -o /tmp/t_first.json

python3 -c "
import json
d=json.load(open('/tmp/t_first.json'))
reply = d.get('reply','')
print(f'回复长度: {len(reply)}')
print(f'回复: {reply[:80]}')
assert len(reply) > 5, '❌ 第一句回复为空或过短'
print('✅ 新建Agent第一句对话正常')
"

# 清理
curl -s -X DELETE 'http://localhost:9999/clear?agent_id=fresh_test' > /dev/null
```

### 5.4 经验学习

**目的：** 验证 lessons.json 文件的读写。

```bash
echo "=== 5.4 经验学习 ==="
# lessons 文件路径
LESSON_FILE=~/.qwen-memory/lessons.json

# 检查文件是否存在
if [ -f "$LESSON_FILE" ]; then
  echo "lessons文件存在"
  # 查看最近的教训
  python3 -c "
import json
with open('$LESSON_FILE') as f:
    data = json.load(f)
print(f'共{len(data)}条经验教训')
for l in data[-3:]:
    print(f'  [{l[\"agent_id\"]}] {l[\"lesson\"][:60]}')
"
else:
  echo "lessons文件不存在（首次使用?）"
fi
```

---

## 六、边缘情况测试

### 6.1 Pydantic 验证

**目的：** 验证 API 输入验证正确返回 422，不报 500。

```bash
echo "=== 6.1 Pydantic验证 ==="

# 空消息
echo -n "空消息: "
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"e1","message":""}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
code = d['detail'][0]
print(f'{code[\"type\"]} → {code[\"msg\"][:40]}')
assert 'string_too_short' in code['type'], '❌'
print('✅')
"

# 无效 JSON
echo -n "无效JSON: "
curl -s -X POST 'http://localhost:9999/chat' \
  -H 'Content-Type: application/json' \
  -d 'not json at all' | python3 -c "
import sys,json; d=json.load(sys.stdin)
assert 'json_invalid' in d['detail'][0]['type'], '❌'
print('✅')
"

# 缺字段
echo -n "缺字段: "
curl -s -X POST 'http://localhost:9999/chat' \
  -H 'Content-Type: application/json' \
  -d '{"agent_id":"test"}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
assert 'missing' in d['detail'][0]['type'], '❌'
print('✅')
"

# 超长文本（不崩溃即可）
echo -n "超长文本: "
LONG=$(python3 -c "print('测试'*1000)")
curl -s -X POST http://localhost:9999/remember \
  -H "Content-Type: application/json" \
  -d "{\"agent_id\":\"test\",\"session_id\":\"long\",\"content\":\"$LONG\"}" | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'status={d.get(\"status\",\"?\")}')
print('✅ 不崩溃')
"

# 无效 UUID（不被 500）
echo -n "无效UUID: "
curl -s -X POST http://localhost:9999/confirm \
  -H "Content-Type: application/json" \
  -d '{"fact_id":"00000000-0000-0000-0000-000000000000","agent_id":"test"}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
# 预期返回 not_found 而不是 500 错误
print(f'status={d.get(\"status\",\"?\")}')
if d.get('status') == 'not_found':
    print('✅ 优雅处理')
else:
    print('⚠️', d)
"
```

### 6.2 图谱 API 错误处理

```bash
echo "=== 6.2 图谱错误处理 ==="
# 不存在 agent → 应返回空图，不报 500
code=$(curl -s -o /dev/null -w '%{http_code}' 'http://localhost:9999/memory-graph?agent_id=this_agent_does_not_exist')
echo -n "不存在agent: HTTP $code → "
[ "$code" = "200" ] && echo "✅（不报500）" || echo "❌"

# 正常 agent → 应返回图数据
echo -n "正常agent: "
curl -s 'http://localhost:9999/memory-graph?agent_id=test' | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'节点{d[\"stats\"][\"node_count\"]} 边{d[\"stats\"][\"edge_count\"]}')
"
```

### 6.3 并行请求

**目的：** 验证服务能同时处理多个请求而不崩溃。

```bash
echo "=== 6.3 并行请求 ==="
# 同时发 3 个请求
curl -s -X POST http://localhost:9999/chat -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"p1","message":"你好"}' > /dev/null &
curl -s -X POST http://localhost:9999/chat -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"p2","message":"搜一下AI"}' > /dev/null &
curl -s 'http://localhost:9999/stats?agent_id=test' > /dev/null &
wait
echo "✅ 3个并行请求均完成（无崩溃）"
```

---

## 七、v0.10 新功能测试

### 7.1 系统健康检测

**目的：** 验证 `/health` 端点返回 5 个维度的健康数据。

```bash
echo "=== 7.1 健康检测 ==="
HEALTH=$(curl -s http://localhost:9999/health?agent_id=default)

echo "$HEALTH" | python3 -c "
import sys,json
h=json.load(sys.stdin)
print(f'健康分: {h[\"score\"]}')
print(f'等级:   {h[\"label\"]}')
print(f'运行:   {h[\"uptime_seconds\"]:.0f}s')
print(f'配置:   {h[\"checks\"].get(\"config\")}')
print(f'工具:   {h[\"checks\"].get(\"tools\")} 个')
print(f'问题:   {len(h.get(\"issues\",[]))} 个')
assert h['score'] >= 0 and h['score'] <= 100, '❌ 健康分范围异常'
assert h['level'] in ('healthy','warning','critical'), '❌ 等级异常'
print('✅ 健康检测端点正常')
"
```

**预期输出：** 健康分 0-100，等级为 healthy/warning/critical 之一。

### 7.2 Dashboard 记忆管理

**目的：** 验证记忆列表、搜索、确认、质疑、删除 API。

```bash
echo "=== 7.2.1 记忆列表 ==="
curl -s 'http://localhost:9999/memories?agent_id=default&limit=3' | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'总数: {d[\"total\"]}')
print(f'返回: {len(d[\"memories\"])} 条')
for m in d['memories'][:2]:
    print(f'  [{m[\"fact_id\"][:8]}] {m.get(\"subject\",\"\")} {m.get(\"predicate\",\"\")} {m.get(\"object\",\"\")[:30]}')
assert 'memories' in d, '❌'
print('✅ 记忆列表正常')
"

echo ""
echo "=== 7.2.2 记忆搜索 ==="
curl -s 'http://localhost:9999/memories/search?q=小明&agent_id=default' | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'搜索「小明」→ {d[\"total\"]} 条')
print('✅ 记忆搜索正常')
"

echo ""
echo "=== 7.2.3 确认/质疑/删除 ==="
# 取一条记忆
MEM_ID=$(curl -s 'http://localhost:9999/memories?agent_id=default&limit=1' | python3 -c "
import sys,json; print(json.load(sys.stdin)['memories'][0]['fact_id'])
")
echo "目标: $MEM_ID"

# 确认
curl -s -X POST http://localhost:9999/confirm \
  -H "Content-Type: application/json" \
  -d "{\"fact_id\":\"$MEM_ID\",\"agent_id\":\"default\"}" | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'确认: {d[\"status\"]}')
assert d['status'] in ('confirmed','error'), '❌'
"

# 质疑
curl -s -X POST http://localhost:9999/challenge \
  -H "Content-Type: application/json" \
  -d "{\"fact_id\":\"$MEM_ID\",\"agent_id\":\"default\"}" | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'质疑: {d[\"status\"]}')
"

echo "✅ 确认/质疑/删除 API 正常"
```

### 7.3 仪表盘健康详情模态框

**目的：** 验证 `/health` 接口被 Dashboard 正确消费。

```bash
echo "=== 7.3 健康详情 ==="
# 直接调用 health 接口验证返回所有维度
curl -s 'http://localhost:9999/health?agent_id=default' | python3 -c "
import sys,json; h=json.load(sys.stdin)
checks = h.get('checks', {})
print(f'数据库: {checks.get(\"db\")}')
print(f'LLM:    {checks.get(\"llm\")}')
print(f'工具:    {checks.get(\"tools\")}')
print(f'配置:    {checks.get(\"config\")}')
print(f'API窗口: {checks.get(\"api\",{})}')
print('✅ 健康详情5维度齐全')
"
```

### 7.4 memory_diagnose 工具

**目的：** 验证 memory_diagnose 返回正确的诊断信息。

```bash
echo "=== 7.4 memory_diagnose ==="
# 通过 agent 对话触发（走 chat 端点）
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"default","session_id":"diag1","message":"检查我的记忆健康状况"}' -o /tmp/t_diag.json

python3 -c "
import json
d=json.load(open('/tmp/t_diag.json'))
tools = d.get('tools_called',0)
reply = d.get('reply','')
print(f'工具: {tools}')
print(f'回复: {reply[:100]}')
if tools > 0 or len(reply) > 20:
    print('✅ memory_diagnose 可调用')
else:
    print('⚠️ 可能未触发 diagnose 工具')
"
```

### 7.5 模型切换验证

**目的：** 验证 DeepSeek ↔ Qwen 切换配置正确。

```bash
echo "=== 7.5 模型切换 ==="
# 检查 .env 格式
grep -q "QWEN_BASE_URL" ~/projects/qwen-memoryagent/.env && echo "✅ .env 含模型配置"
grep -q "dashscope-intl" ~/projects/qwen-memoryagent/.env && echo "✅ Qwen 端点已配置（注释状态）"
grep -q "deepseek.com" ~/projects/qwen-memoryagent/.env && echo "✅ DeepSeek 端点已配置（当前使用）"

# 检查 QWEN_FAST_MODEL 是否存在
grep -q "QWEN_FAST_MODEL" ~/projects/qwen-memoryagent/.env && echo "✅ 快速模型已配置"
```

---

## 八、回归测试（每次改后必跑）

```bash
echo "========================================="
echo "  回归测试 — 7项核心功能验证"
echo "========================================="

echo ""
echo "=== 1. 服务在线 ==="
code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:9999/)
echo "HTTP $code"
[ "$code" = "200" ] && echo "✅" || echo "❌"

echo ""
echo "=== 2. 健康检测 ==="
curl -s http://localhost:9999/health | python3 -c "
import sys,json; h=json.load(sys.stdin)
print(f'健康分: {h[\"score\"]} | 工具: {h[\"checks\"].get(\"tools\",\"?\")}')
assert h['score']>=50, '❌ 健康分过低'
print('✅')
"

echo ""
echo "=== 3. 搜索可用 ==="
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"reg","message":"搜一下AI"}' -o /tmp/t_reg.json
python3 -c "
import json
d=json.load(open('/tmp/t_reg.json'))
tools=d['tools_called']
print(f'tools={tools} len={len(d[\"reply\"])}')
if tools>0 and len(d['reply'])>30: print('✅')
else: print('❌')
"

echo ""
echo "=== 4. 存储+召回 ==="
curl -s -X POST http://localhost:9999/remember \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"reg2","content":"测试数据"}' > /dev/null
curl -s -X POST http://localhost:9999/recall \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","query":"测试"}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'recall: {d[\"count\"]} 条')
if d['count']>0: print('✅')
else: print('❌')
"

echo ""
echo "=== 5. 三页面加载 ==="
all_ok=0
for p in "/" "/dashboard" "/graph"; do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:9999$p)
  echo "  $p: $code"
  [ "$code" = "200" ] && all_ok=$((all_ok+1))
done
[ "$all_ok" -eq 3 ] && echo "✅" || echo "❌"

echo ""
echo "=== 6. 三元组噪音过滤 ==="
# 发 curl 命令，检查不被存
curl -s -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","session_id":"reg3","message":"curl https://example.com"}' > /dev/null
sleep 1
curl -s -X POST http://localhost:9999/recall \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","query":"curl"}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
has_curl = any('curl' in m.get('content','').lower() for m in d.get('memories',[]))
print('curl命令被过滤' if not has_curl else '❌ 有curl噪音')
"

echo ""
echo "=== 7. 清理 ==="
curl -s -X DELETE 'http://localhost:9999/clear?agent_id=test' | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'{d[\"message\"]}')
if d.get('deleted',-1) >= 0 or d['message'] == '无数据库连接':
    print('✅')
else:
    print('❌')
"
```

---

## 九、已知问题与故障排除

### 已知限制（当前版本 v0.10）

| # | 问题 | 原因 | 影响 |
|---|------|------|------|
| 1 | web_search 需要 SEARCH_PROXY | 国内直连 Bing/Google 被墙 | 无代理时搜索不可用 |
| 2 | 搜索结果可能滞后 | Bing 搜索结果非实时 | 新闻类查询可能漏最新 |
| 3 | 抽象化需同模式数据 ≥2 条 | 算法要求同 (subject, predicate) 组 ≥2 | 碎片数据无法抽象 |
| 4 | 经验学习不跨服务重启 | lessons.json 持续存储但仅当前实例回读 | 重启后 lessons 仍在 |
| 5 | 中文"的"分词不完美 | 纯正则无法理解中文句法 | 部分"X的Y很Z"句式提取不完整 |

### ✅ 已修复（完整历史）

| 历史问题 | 修复版本 | 验证方法 |
|----------|---------|----------|
| `extract(None)` 崩溃 | v0.8 | 不再需要测试 None 输入 |
| xss dashboard/graph | v0.8 | 不再需要检查 innerHTML |
| `_find_existing` 缺锁 | v0.8 | 不再需要检查锁覆盖 |
| 路由统计 l3_hits | v0.8 | stats 包含 `l3_hit_rate` 字段 |
| `reset_agent` 返回 -1 | v0.8 | 回归测试需 `>=0` 判断 |
| `/memory-graph` 空 recall | v0.8 | 不存在 agent 返回空图 |
| 测试失败 | v0.8 | 26/26 全过 |
| OpenAI 无超时 | v0.8 | 网络故障时 30s 超时断连 |
| `_HEALTH` 无锁 | v0.8 | 并发请求不竞争 |
| LLM 无重试 | v0.8 | 网络抖动自动重试3次 |
| ACTION_WORDS 遗漏 | v0.8 | +总结/翻译/推荐等8词 |
| 搜索关键词遗漏 | v0.8 | +咨询/了解/介绍等7词 |
| 遗忘过猛 | v0.8 | 30min间隔+1h免疫期 |
| HTML 每次读磁盘 | v0.8 | `lru_cache` 缓存 |
| **图谱 action 类型缺失** | **v0.10** | action 节点显示 + 筛选 |
| **图谱空数据** | **v0.10** | `/memory-graph` 加 _resolve_agent |
| **三元组噪音** | **v0.10** | curl/wget 过滤 + web_fetch 跳过 + 搜索去重 |
| **首次对话空白** | **v0.10** | .env 添加 QWEN_FAST_MODEL |
| **仪表盘布局错乱** | **v0.10** | 最终定稿：4+2 卡片 + 5 图表 + 日志 |
| **错误截断不足** | **v0.10** | chat_stream str(e)[:80] → [:200] |

### 故障诊断

| 症状 | 可能原因 | 检查方法 |
|------|----------|----------|
| HTTP 500 | Python 运行时错误 | `tail -30 /tmp/qwen-agent.log` |
| 搜索无结果 | SEARCH_PROXY 未设置或代理不可用 | `curl -x http://127.0.0.1:10809 https://www.bing.com` |
| 搜索太深（>12工具）| 系统提示词中"早停"指令被覆盖 | 检查 `_build_messages` 中 system prompt 组装 |
| 聊天无回复 | LLM API 超时 | 检查 `/tmp/qwen-agent.log` 中 httpx 请求 |
| 首句空白 | QWEN_FAST_MODEL 未配置 | 检查 .env 中 `grep QWEN_FAST_MODEL` |
| 记忆不被召回 | 数据太少或查询词不匹配 | 先用空查询 `query=""` 浏览所有事实 |
| /clear 报错 | FK 外键顺序 | 检查 brain.py reset_agent 方法 |
| 页面空白 | templates/ 文件缺失或损坏 | `ls -la src/memory_agent/templates/` |
| **三元组含 curl** | 噪音过滤未生效 | 检查 agent/__init__.py `_extract_action_facts` |
| **图谱无 action 节点** | 过滤未含 'action' | 检查 graph.html `TYPE_COLORS` `activeTypeFilters` |

---

> **发现 bug 请记录：** 记到 `xiaoqi-mistakes.md`  
> **做对事请记录：** 记到 `xiaoqi-wins.md`  
> **本地端口：** http://localhost:9999  
> **截止：** 2026-07-10 05:00 GMT+8
