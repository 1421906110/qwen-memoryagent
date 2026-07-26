"""
ToolRegistry — 自动 JSON Schema 生成 + @tool 装饰器

🔥 相对优化（vs OpenWorker）：
  - 零外部依赖：用 inspect.signature 替代 aisuite
  - 运行时开销 0：schema 在 import 时生成一次后缓存
  - 装饰器模式：加工具只需 3 行代码

用法：
    @tool
    def web_search(query: str, max_results: int = 5):
        \"\"\"搜索网络信息

        :param query: 搜索关键词
        :param max_results: 最大结果数
        \"\"\"
        ...
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from typing import Any, Callable, Optional

logger = logging.getLogger("agent.registry")


# ===================================================================
#  Auto Schema — 从函数签名 + docstring 自动生成 JSON Schema
#  🔥 零外部依赖，import 时生成一次
# ===================================================================

TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


def _schema_from_func(func: Callable) -> dict:
    """从函数签名 + docstring 自动生成 OpenAI 格式 JSON Schema

    🔥 零外部依赖（OpenWorker 用 aisuite），纯标准库
    🔥 运行时成本：0 —— import 时生成一次后缓存

    Args:
        func: 要生成 schema 的函数

    Returns:
        OpenAI function calling 格式的 schema dict
    """
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        # 类型映射
        param_type = TYPE_MAP.get(param.annotation, "string")

        # 从 docstring 提取参数描述
        # 支持 :param name: desc 和 name: desc 两种格式
        desc = ""
        for pattern in [
            rf":param\s+{re.escape(name)}:\s*(.+?)(?:\n|$)",
            rf"{re.escape(name)}:\s*(.+?)(?:\n|$)",
        ]:
            m = re.search(pattern, doc)
            if m:
                desc = m.group(1).strip()
                break

        properties[name] = {"type": param_type, "description": desc or f"Parameter {name}"}

        if param.default is inspect.Parameter.empty:
            required.append(name)

    # 函数描述 = docstring 第一行
    first_line = doc.split("\n")[0] if doc else ""

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": first_line,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


# ===================================================================
#  ToolRegistry
# ===================================================================

class ToolRegistry:
    """工具注册中心

    支持两种注册方式：
    1. @tool 装饰器
    2. 手动 register()

    OpenWorker 用 aisuite Tools() 做 schema 转换，
    CogniMem 用 inspect.signature（零依赖）。
    """

    def __init__(self):
        self._tools: dict[str, dict] = {}
        self._categories: dict[str, list[str]] = {}

    def register(self, *args, **kwargs):
        """注册工具

        支持两种调用方式：

        装饰器模式:
            @registry.register
            def my_tool(...): ...

        手动模式（两种风格均可）:
            # 🔥 新风格（推荐）
            registry.register(name="x", description="...",
                              parameters={...}, executor=func)

            # 🔥 旧风格（兼容，v0.17 legacy）
            registry.register("name", "description", parameters, executor,
                              tool_type="builtin", category="general")

        Args:
            *args: 旧风格位置参数 (name, description, parameters, executor)
            **kwargs: 新风格或装饰器模式
                func: 装饰器模式传入的函数
                name: 手动模式时工具名
                description: 手动模式时工具描述
                parameters: 手动模式时参数 schema
                executor: 手动模式时执行函数
                tool_type: 工具类型（builtin/module）
                category: 工具分类

        Returns:
            装饰器模式返回原函数，手动模式返回 None
        """
        # ── 检测调用风格 ──
        # 旧风格：register("name", "desc", params, executor, ...)
        if len(args) >= 2:
            name, description = args[0], args[1]
            parameters = args[2] if len(args) > 2 else kwargs.get("parameters")
            executor = args[3] if len(args) > 3 else kwargs.get("executor")
            tool_type = kwargs.get("tool_type", "builtin")
            category = kwargs.get("category", "general")

            schema = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description or "",
                    "parameters": parameters or {"type": "object", "properties": {}},
                },
            }
            self._tools[name] = {
                "name": name,
                "schema": schema,
                "func": executor,
                "type": tool_type,
                "category": category,
            }
            self._categories.setdefault(category, []).append(name)
            logger.debug("🔧 Registered tool (legacy style): %s", name)
            return None

        # ── 新风格：kwargs only ──
        func = kwargs.pop("func", None)
        name = kwargs.pop("name", None)
        description = kwargs.pop("description", None)
        parameters = kwargs.pop("parameters", None)
        executor = kwargs.pop("executor", None)
        tool_type = kwargs.pop("tool_type", "builtin")
        category = kwargs.pop("category", "general")

        # 装饰器模式：第一个参数是函数
        if len(args) == 1 and callable(args[0]):
            func = args[0]
        elif func is not None and callable(func):
            pass
        else:
            func = None

        if func is not None:
            # 装饰器模式：自动生成 schema
            name = func.__name__
            schema = _schema_from_func(func)
            self._tools[name] = {
                "name": name,
                "schema": schema,
                "func": func,
                "type": tool_type,
                "category": category,
            }
            self._categories.setdefault(category, []).append(name)
            logger.debug("🔧 Registered tool (auto-schema): %s", name)
            return func

        # 手动模式
        if not name or not executor:
            raise ValueError("手动注册需要 name 和 executor")

        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description or "",
                "parameters": parameters or {"type": "object", "properties": {}},
            },
        }
        self._tools[name] = {
            "name": name,
            "schema": schema,
            "func": executor,
            "type": tool_type,
            "category": category,
        }
        self._categories.setdefault(category, []).append(name)
        logger.debug("🔧 Registered tool (manual): %s", name)
        return None

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Optional[dict]:
        return self._tools.get(name)

    def schemas(self) -> list[dict]:
        """返回所有工具的 OpenAI function calling schema 列表"""
        return [t["schema"] for t in self._tools.values()]

    def list_tools(self) -> list[dict]:
        """兼容旧接口：返回完整 tool definitions"""
        return self.schemas()

    def list_for_ui(self) -> list[dict]:
        return [
            {"name": t["name"], "description": t["schema"]["function"]["description"],
             "type": t["type"], "category": t["category"]}
            for t in self._tools.values()
        ]

    def list_by_category(self) -> dict[str, list[dict]]:
        result = {}
        for t in self._tools.values():
            cat = t["category"]
            result.setdefault(cat, []).append({
                "name": t["name"],
                "description": t["schema"]["function"]["description"],
                "type": t["type"],
            })
        return result

    def execute(self, tool_call_id: str, name: str, args: dict,
                ctx=None) -> dict:
        """执行工具

        Args:
            tool_call_id: LLM 分配的 tool_call_id
            name: 工具名
            args: 参数字典
            ctx: AgentContext（可选）

        Returns:
            工具执行结果 dict
        """
        tool = self._tools.get(name)
        if not tool:
            return {"error": f"未知工具: {name}"}

        logger.info("🧰 Tool call: %s(%s)", name, json.dumps(args)[:200])

        try:
            result = tool["func"](tool_call_id, args, ctx)

            # 更新 ctx（如果有）
            if ctx is not None:
                ctx.tools_called += 1
                if tool["type"] == "module":
                    ctx.modules_used += 1

            return result
        except Exception as e:
            logger.exception("Tool %s failed", name)
            return {"error": str(e)}


# ===================================================================
#  @tool 装饰器（快捷引用）
# ===================================================================

_registry = ToolRegistry()


def tool(func=None, *, registry: ToolRegistry = None):
    """@tool 装饰器

    用法：
        @tool
        def my_tool(param1: str, param2: int = 0):
            \"\"\"工具描述

            :param param1: 参数1说明
            :param param2: 参数2说明
            \"\"\"
            ...

    如果不传 registry，使用全局默认 _registry。
    """
    r = registry or _registry
    if func is not None:
        return r.register(func)
    # 带参数的用法：@tool(registry=xxx)
    return lambda f: r.register(f)


def get_registry() -> ToolRegistry:
    """获取全局默认 ToolRegistry"""
    return _registry
