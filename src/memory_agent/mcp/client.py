"""
CogniMem MCP Client — 按需连接 + 空闲自动关闭

🔥 相对优化（vs OpenWorker 常驻进程）：
  - 按需连接：用户要求时才启动 MCP 进程（省资源）
  - 5min 空闲自动关闭：长期不用自动释放（省资源）
  - 不预加载工具列表：用的时候再查（省Token）

用法：
    mcp = MCPManager()
    await mcp.connect("playwright", "npx", ["@playwright/mcp"])
    tools = await mcp.list_tools("playwright")
    result = await mcp.call_tool("playwright", "click", {"selector": "#btn"})
    # 5 分钟不用自动关闭
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

logger = logging.getLogger("agent.mcp")


class _Connection:
    """单个 MCP Server 连接"""

    def __init__(self, session, tools: list[dict]):
        self.session = session
        self.tools = tools
        self.last_used = time.time()
        self.created = time.time()

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_used

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created


class MCPManager:
    """MCP 客户端管理器

    🔥 按需连接：不用的时候没有常驻进程
    🔥 5min 空闲自动关闭：长期不用自动释放
    """

    def __init__(self, idle_timeout: int = 300):
        """
        Args:
            idle_timeout: 空闲超时秒数（默认 300s = 5min）
        """
        self._conns: dict[str, _Connection] = {}
        self._idle_timeout = idle_timeout
        self._cleanup_task: Optional[asyncio.Task] = None

    async def connect(self, name: str, command: str,
                      args: list[str] | None = None) -> list[dict]:
        """连接一个 MCP Server（按需）

        Args:
            name: 唯一标识
            command: 可执行文件路径
            args: 命令行参数

        Returns:
            可用工具列表
        """
        # 已有连接 → 更新 last_used 并返回
        existing = self._conns.get(name)
        if existing is not None:
            existing.last_used = time.time()
            return existing.tools

        # 🔥 按需启动 MCP 进程
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=command,
                args=args or [],
            )

            read, write = await stdio_client(params).__aenter__()
            session = await ClientSession(read, write).__aenter__()
            await session.initialize()

            # 获取工具列表
            result = await session.list_tools()
            tools = [
                {"name": t.name, "description": t.description,
                 "inputSchema": t.inputSchema}
                for t in result.tools
            ]

            conn = _Connection(session, tools)
            self._conns[name] = conn
            logger.info("🔌 MCP connected: %s (%d tools)", name, len(tools))

            # 启动自动清理（如果未启动）
            self._ensure_cleanup()

            return tools
        except Exception as e:
            logger.error("MCP connect failed: %s: %s", name, e)
            raise

    async def list_tools(self, name: str) -> list[dict]:
        """获取某个 MCP Server 的工具列表"""
        conn = self._conns.get(name)
        if conn is None:
            raise RuntimeError(f"MCP server not connected: {name}")
        conn.last_used = time.time()
        return conn.tools

    async def call_tool(self, name: str, tool: str,
                        arguments: dict | None = None) -> Any:
        """调用某个 Server 的工具"""
        conn = self._conns.get(name)
        if conn is None:
            raise RuntimeError(f"MCP server not connected: {name}")
        conn.last_used = time.time()

        result = await conn.session.call_tool(tool, arguments or {})
        return result.content if hasattr(result, 'content') else result

    def get_all_tools(self) -> dict[str, list[dict]]:
        """获取所有已连接 Server 的聚合工具列表"""
        result = {}
        for name, conn in self._conns.items():
            result[name] = conn.tools
        return result

    async def disconnect(self, name: str):
        """主动断开某个 MCP Server"""
        conn = self._conns.pop(name, None)
        if conn:
            try:
                await conn.session.__aexit__(None, None, None)
                logger.info("🔌 MCP disconnected: %s", name)
            except Exception as e:
                logger.warning("MCP disconnect error: %s: %s", name, e)

    async def disconnect_all(self):
        """断开所有连接"""
        for name in list(self._conns.keys()):
            await self.disconnect(name)

    def _ensure_cleanup(self):
        """启动空闲自动清理任务"""
        if self._cleanup_task is None or self._cleanup_task.done():
            loop = asyncio.get_running_loop()
            self._cleanup_task = loop.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        """🔥 周期性检查空闲连接并关闭"""
        while True:
            await asyncio.sleep(60)  # 每分钟检查一次
            for name, conn in list(self._conns.items()):
                if conn.idle_seconds > self._idle_timeout:
                    logger.info("🔌 Closing idle MCP: %s (idle=%ds)",
                                name, conn.idle_seconds)
                    await self.disconnect(name)

    @property
    def stats(self) -> dict:
        """连接统计"""
        return {
            "connected": len(self._conns),
            "connections": {
                name: {
                    "tools": len(conn.tools),
                    "idle_seconds": conn.idle_seconds,
                    "age_seconds": conn.age_seconds,
                }
                for name, conn in self._conns.items()
            },
        }
