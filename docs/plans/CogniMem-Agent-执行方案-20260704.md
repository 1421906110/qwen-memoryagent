# CogniMem → 真·AI Agent 执行方案

> **时间：** 2026-07-04 | **截止：** 2026-07-10
> **目标：** 让 CogniMem 拥有和小七一样的干活能力——自主规划、自我反思、目标驱动、记忆自管理

---

## 一、现状诊断：Agent 为什么不是 Agent

### 当前执行循环

```
用户: "帮我爬一下这个页面"
  → Agent 调一次 web_fetch → 返回结果
  → 结束（等用户说下一步）
```

**问题：循环是用户驱动的，不是目标驱动的。** Agent 不会：
1. 自己决定"下一步做什么"
2. 发现错误后自己重试或换方法
3. 记住最终目标（调 10 次工具直到完成）
4. 完成目标后自己存记忆

### 技术根因

| 根因 | 代码位置 | 说明 |
|------|---------|------|
| 一轮只执行一次工具调用 | `agent/__init__.py` Agent.chat() | 虽然有多轮循环，但每轮只等 LLM 决定下一步，没有"完成判断" |
| 没有目标跟踪 | `agent/__init__.py` 缺 GoalContext | Agent 不知道"最终目标是什么"，做完一步就汇报 |
| 没有错误恢复 | `agent/__init__.py` try/except | 工具失败只是返回 error 字符串，不自动重试或换策略 |
| 没有主动存记忆 | `agent/__init__.py` Step 5 被删除 | 之前删了自动存（怕污染三元组），但应该改为**智能判断后存** |

---

## 二、执行架构：5 层能力

```
┌─────────────────────────────────────────────────────────────┐
│                     🎯 Goal Planner                          │
│              任务拆解 → 子目标 → 排序执行                       │
├─────────────────────────────────────────────────────────────┤
│                     🔄 Task Executor                          │
│              执行 → 检查结果 → 决定继续/重试/换方法             │
├─────────────────────────────────────────────────────────────┤
│                     ⚡ Tool Runner                            │
│              调工具 → 解析结果 → 返回结构化数据                  │
├─────────────────────────────────────────────────────────────┤
│                     🧠 Self-Reflector                         │
│              错误分析 → 策略调整 → 下次避免同类错误               │
├─────────────────────────────────────────────────────────────┤
│                     💾 Memory Manager                         │
│              判断"这个重要吗" → 决定存/忘 → 管理置信度            │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、Phase 1：目标驱动执行循环 🔴 P0（今天）

### 改造 Agent.chat()

**当前：**
```
1. 召回记忆 → 治理过滤 → 注入 prompt
2. LLM 思考 → 调工具 → 返回结果 → 结束
```

**改后：**
```
1. 用户说"帮我爬这个页面"
2. Agent 创建 GoalContext:
   - 目标: "获取页面内容并结构化保存"
   - 子目标: []
   - 完成条件: "success = 内容已获取"
3. 循环直到完成或 max_iterations:
   a. LLM 思考下一步
   b. 调工具
   c. 检查结果:
      - 失败 → 自动重试（最多 3 次）
      - 部分成功 → 继续下一步
      - 完成 → 跳出循环
4. 判断是否需要存记忆:
   - "这个信息对以后有用吗?"
   - 有用 → memory_remember
5. 回复用户 + 总结做了啥
```

### 新增文件

#### `agent/goal.py` — Goal Context 跟踪

```python
@dataclass
class GoalContext:
    """跟踪 Agent 当前任务的最终目标"""
    original_request: str           # 用户原始请求
    description: str               # Agent 理解的最终目标
    sub_goals: list[SubGoal]       # 子目标队列
    completed_sub_goals: list      # 已完成
    max_retries: int = 3           # 每个工具的失败重试次数
    completion_criteria: str = ""  # 完成条件的自然语言描述
    status: str = "pending"        # pending | running | done | failed
    
    def is_complete(self, tool_results: list[dict]) -> bool:
        """判断目标是否完成"""
        # 核心逻辑：检查子目标是否全部完成
        # 或 LLM 判断"目标是否已经达成"
```

#### `agent/reflector.py` — 自我反思

```python
class SelfReflector:
    """错误后自动分析原因并调整策略"""
    
    def analyze_failure(self, tool_name: str, args: dict, error: str) -> dict:
        """分析失败原因 → 返回建议"""
        # 模式匹配常见错误
        # "Connection refused" → "服务没启动，试试启动它"
        # "timeout" → "加长超时时间或换方法"
        # "not found" → "路径不对，先 list_dir 找到正确路径"
    
    def suggest_alternative(self, tool_name: str, failure_reason: str) -> str | None:
        """建议替代方案"""
        # web_fetch 失败 → "试试 curl 代替"
        # shell 命令失败 → "试试 pip install 缺少的包"
```

### 改造 `agent/__init__.py` — Agent.chat()

**主要改动：**

```python
def chat(self, message, agent_id, ...) -> dict:
    # 0. 创建 GoalContext
    goal = self._parse_goal(message)
    
    # 1. 召回 + 治理
    memories = self._recall_and_filter(message)
    
    # 2. 目标驱动循环
    for iteration in range(max_iterations):
        # 2a. 检查是否应该结束
        if goal.is_complete(tool_results_this_round):
            break
            
        # 2b. LLM 思考下一步
        response = self.llm.chat_completion(
            messages=openai_messages,
            tools=tool_defs,
        )
        
        # 2c. 执行工具
        for tc in response.choices[0].message.tool_calls:
            result = self.tools.execute(tc.id, tc.function.name, args, ctx)
            
            # 2d. 失败 → 自动重试 / 换方法
            if "error" in result:
                # 先尝试重试
                for retry in range(goal.max_retries):
                    result = self.tools.execute(tc.id, tc.function.name, args, ctx)
                    if "error" not in result:
                        break
                # 重试都失败 → 反思 + 换方法
                if "error" in result:
                    suggestion = self.reflector.analyze_failure(
                        tc.function.name, args, result["error"]
                    )
                    # 把反思结果喂给 LLM，让它决定下一步
                    ...
    
    # 3. 智能存记忆
    important_memories = self._extract_important_facts(
        message, goal, tool_results
    )
    if important_memories:
        self._store_memories(important_memories)
    
    # 4. 返回
    return {"reply": final_reply, "tools_called": ..., "goal_achieved": ...}
```

### 改动的文件

| 文件 | 改动 | 预估行数 |
|------|------|---------|
| `agent/__init__.py` | 重写 Agent.chat() → 目标驱动循环 | ~80 行 |
| `agent/goal.py` | **新建** GoalContext 类 | ~60 行 |
| `agent/reflector.py` | **新建** SelfReflector 类 | ~50 行 |
| `agent/tools.py` | 无改动（工具定义不变） | 0 行 |

---

## 四、Phase 2：自主任务规划 🔴 P0（明天）

### 需要的能力

```
用户: "帮我写一个爬虫爬这个页面"

Phase 1（当前）:
  Agent: 调 web_fetch → 返回 HTML → "好了"
  ❌ 用户没看到结构化数据

Phase 2（目标）:
  Agent 自己拆:
    Step 1: web_fetch(url)           → 获取页面
    Step 2: 分析 HTML → 发现需要解析  → 用 shell 调 python
    Step 3: 写一个解析脚本            → 提取标题/内容
    Step 4: 测试解析结果              → 验证成功
    Step 5: 存记忆"爬虫写好了"         → 下次可以直接复用
    Step 6: 回复用户                  → "搞定了，数据在 xxx"
  ✅ 用户看到完整结果
```

### 实现方案

#### 方案 A：LLM 预规划（推荐）

在 Agent 响应前，先让 LLM 生成一个`任务计划`：

```
用户: "帮我爬这个页面 www.example.com"

Agent 内部:
  1. 调 LLM（不调工具）→ 让我先规划:
     分析: 这是一个新闻页面，需要:
     - 步骤1: web_fetch 获取页面
     - 步骤2: 用 Python 解析 HTML
     - 步骤3: 提取新闻标题/日期/内容
     - 步骤4: 保存为 Markdown 文件
     - 步骤5: 用 memory_remember 存为记忆
     
  2. 执行计划（自动推进每一步）
     
  3. 汇报完成
```

**关键：先想再做，不是边想边做。**

### 改动的文件

| 文件 | 改动 | 预估行数 |
|------|------|---------|
| `agent/__init__.py` | chat() 开始时加预规划步骤 | ~40 行 |
| `agent/goal.py` | 扩展 GoalContext 支持预规划步骤 | ~30 行 |

---

## 五、Phase 3：自我反思 + 纠错循环 🔴 P0（明天）

### 当前行为

| 场景 | 当前 | 期望 |
|------|------|------|
| web_fetch 超时 | 返回 "error: timeout" | 自动重试 + 换 user-agent |
| shell 命令 not found | 返回 "error: not found" | `pip install` → 重试 |
| 写文件目录不存在 | 返回 "error: no such dir" | 创建目录 → 重试 |
| LLM 返回格式错误 | crash | 截断/重解析 → 继续 |

### 实现方案

```python
class SelfReflector:
    """错误分析 + 自动修复"""
    
    # 错误模式 → 修复策略
    FIX_PATTERNS = [
        (r"connect(ion)? refused", "service_down", 
         "服务没启动，尝试 systemctl start xxx"),
        (r"timeout", "network_slow",
         "网络太慢，加长超时或换 curl -sSL"),
        (r"not found|No such file", "path_missing",
         "路径不存在，先创建父目录"),
        (r"command not found", "missing_dependency",
         "缺少依赖，pip install / apt install"),
        (r"Permission denied", "no_permission",
         "权限不够，尝试 chmod 或 sudo"),
    ]
    
    def analyze(self, tool_name: str, error: str) -> dict:
        """返回: {pattern_matched, fix_action, fix_prompt}"""
    
    def generate_fix_prompt(self, tool_name: str, error: str, 
                             original_args: dict) -> str:
        """生成 LLM 能理解的修复提示"""
        # "web_fetch 超时了，建议加长 timeout 或用 curl 代替"
```

### 改动的文件

| 文件 | 改动 | 预估行数 |
|------|------|---------|
| `agent/reflector.py` | 扩展 SelfReflector 的修复逻辑 | ~60 行 |
| `agent/__init__.py` | chat() 工具调用失败时接入 reflector | ~20 行 |

---

## 六、Phase 4：记忆自管理 🟡 P1（7/6）

### 需要的能力

| 场景 | 当前 | 期望 |
|------|------|------|
| 用户说"我喜欢喝咖啡" | 存成记忆 ✅ | 已支持 |
| 用户说"今天天气不错" | 也存成记忆 ❌ | 不存（废话）|
| Agent 爬完一个网站 | 不存 ❌ | 存"爬取记录" |
| Agent 学会了新技能 | 不存 ❌ | 存"这个任务下次怎么做" |

### 实现方案

```python
class MemoryFilter:
    """智能判断该记什么、不该记什么"""
    
    def should_remember(self, text: str, source: str) -> bool:
        """返回 True/False"""
        
        # 1. 废话规则
        IGNORE_PATTERNS = [
            "天气不错", "早上好", "晚安", "你好",
            "明白了", "知道了", "好的",
        ]
        
        # 2. 重要信号
        IMPORTANT_SIGNALS = [
            "喜欢", "不喜欢", "偏好", "习惯",
            "记住", "不要", "总是", "从来不",
            "我是", "我叫", "我住在",
        ]
        
        # 3. 信息量判断（太短的内容不记）
        if len(text) < 10:
            return False
        
        # 4. Agent 自产内容（工具执行结果）
        if source == "agent_tool_result":
            # 只有"成功/失败/学到了新东西"才记
            ...
```

### 改动的文件

| 文件 | 改动 | 预估行数 |
|------|------|---------|
| `agent/memory_manager.py` | **新建** MemoryFilter 类 | ~80 行 |
| `agent/__init__.py` | chat() 存记忆前调用 MemoryFilter | ~10 行 |

---

## 七、Phase 5：提交材料 🟡 P1（7/7~7/8）

| # | 事项 | 优先级 | 估算 |
|---|------|--------|------|
| 1 | GitHub 仓库同步（ECS → GitHub） | 🔴 P0 | ~30 分钟 |
| 2 | 添加 LICENSE（MIT） | 🟢 P2 | ~5 分钟 |
| 3 | 更新 README.md | 🟡 P1 | ~1 小时 |
| 4 | 录演示视频（展示 Agent 自主执行） | 🔴 P0 | ~2 小时 |
| 5 | 截图 ECS 部署证明 | 🟡 P1 | ~15 分钟 |
| 6 | 更新 Devpost 提交 | 🔴 P0 | ~1 小时 |
| 7 | 写 Blog 文章（可选，额外奖金） | 🟢 P2 | ~2 小时 |

---

## 八、完整执行路线

### 今天（7/4）—— Phase 1：目标驱动执行

```
✅ 备份完成
⬜ 1a. 创建 agent/goal.py (60行)
⬜ 1b. 创建 agent/reflector.py (50行)  
⬜ 1c. 改造 agent/__init__.py → 目标驱动循环 (80行)
⬜ 1d. 验证：Agent 能自动完成"帮我爬xxx → 存记忆"全程
```

### 明天（7/5）—— Phase 2+3：自主规划 + 自我反思

```
⬜ 2a. 预规划步骤（LLM 先拆任务再执行）
⬜ 2b. 扩展 SelfReflector 修复逻辑
⬜ 2c. 验证：Agent 爬虫失败后自动重试/换方法
```

### 7/6 —— Phase 4：记忆自管理

```
⬜ 3a. 创建 agent/memory_manager.py
⬜ 3b. 验证：废话不存，重要自动存
⬜ 3c. 更新 README + 架构图
```

### 7/7~7/8 —— 提交冲刺

```
⬜ 4a. GitHub 同步 + LICENSE
⬜ 4b. 录演示视频（重点展示 Agent 自主能力）
⬜ 4c. Devpost 提交
⬜ 4d. 可选：写 Blog
```

---

## 九、关键验证场景

### 场景 1：网站爬取（自主执行 + 目标驱动）

```
用户: "帮我爬 https://example.com 的内容"

期望行为:
  1. Agent 分析 → 规划步骤
  2. web_fetch → 获取 HTML
  3. 发现 raw HTML 不好读 → 用 Python 解析
  4. 提取标题/正文 → 保存为文件
  5. 存记忆"已爬取 example.com"
  6. 回复"搞定了，数据在 xxx"
  
❌ 当前: 只做 web_fetch → 给用户看 HTML → 结束
✅ 目标: 全程自动完成
```

### 场景 2：错误恢复

```
用户: "运行这个 Python 脚本"

期望行为:
  1. 尝试 python xxx.py
  2. 报错 "Module not found: requests"
  3. Agent 自动: pip install requests → 重试
  4. 又报错 → 分析错误 → 修复 → 重试
  5. 最终成功 → 回复结果

❌ 当前: 报错就返回"error"
✅ 目标: 自动修复直到成功或确认无法修复
```

### 场景 3：跨会话记忆

```
Session 1: 
  用户: "我住在北京"
  Agent: "好的，已记住你住北京"

Session 2:
  用户: "帮我查一下明天的天气"
  Agent: 自动查 "北京 天气"
  → 回复 "北京明天晴天 25°C"

❌ 当前: 可能查到上海的天气（没主动用记忆）
✅ 目标: 自动用已记住的信息提升回答质量
```

---

## 十、架构变更总览

### 新增文件

| 文件 | 用途 | 预估行数 |
|------|------|---------|
| `agent/goal.py` | Goal Context 跟踪 + 完成判断 | ~60 行 |
| `agent/reflector.py` | 自我反思 + 自动修复策略 | ~110 行 |
| `agent/memory_manager.py` | 记忆自管理（该记/不该记判断） | ~80 行 |

### 修改文件

| 文件 | 改动 | 预估行数 |
|------|------|---------|
| `agent/__init__.py` | chat() 重构 → 目标驱动 + 预规划 + 反思接入 | ~100 行 |
| `main.py` | 如果 Agent 初始化需要新参数 | ~10 行 |

### 总数

```
新增: 3 文件, ~250 行
修改: 2 文件, ~110 行
总代码: ~360 行
```

---

## 十一、风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| Qwen 模型 function calling 不稳定 | 中 | 工具调用失败 | 加重试 + fallback 到 chat |
| 阿里云 ECS 网络受限 | 低 | web tools 用不了 | 本地验证，ECS 只做展示 |
| 10K token 限制不够用 | 中 | 长对话失败 | 只注入关键记忆 + 截断历史 |
| 用户需求变化 | 低 | 方向偏离 | 每完成一个 Phase 就验证 |

---

## 结论

**现在的 CogniMem 是一个优秀的记忆存储系统，但不是 Agent。**

要把变成真正的 AI Agent，核心不是加功能，而是**改造执行循环**——从"用户说了才动"变成"自己规划→执行→反思→记忆→完成"。

今天是 7/4，截止 7/10，我们有 6 天。前 3 天集中做 Phase 1~3（执行能力），后 3 天做提交材料。**第一天就要跑通"Agent 自主完成一个复杂任务"的完整流程。**
