"""DATORYX MCP Client Manager.

Completes the bidirectional MCP story: datoryx_mcp/mcp_server.py exposes
DATORYX agents *to* other MCP clients; this module lets DATORYX agents
themselves act *as* an MCP client, calling out to external MCP servers
(filesystem, GitHub, browser automation, security scanners, etc).

Configuration lives in `mcp_servers.json` (see mcp_servers.example.json)
in the same shape as Claude Desktop's config, so an existing config can be
copied in directly:

    {
      "mcpServers": {
        "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]},
        "github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"], "env": {"GITHUB_TOKEN": "..."}}
      }
    }

Usage from an agent:

    from core.plugin_manager.mcp_client import get_client_manager
    result = await get_client_manager().call_tool("filesystem", "read_file", {"path": "/tmp/x.txt"})

Every call degrades gracefully (returns an error dict, never raises) if a
server isn't configured, isn't running, or the call fails - consistent
with the rest of the codebase's "degrade don't crash" approach.
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional

_CONFIG_PATH = Path(__file__).parent.parent.parent / "mcp_servers.json"


class MCPClientManager:
    """Owns one persistent MCP client session per configured external
    server, opened lazily on first use and reused across calls.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self._config_path = config_path or _CONFIG_PATH
        self._sessions: Dict[str, Any] = {}
        self._exit_stack = AsyncExitStack()
        self._lock = asyncio.Lock()

    def _load_config(self) -> Dict[str, Any]:
        if not self._config_path.exists():
            return {}
        try:
            return json.loads(self._config_path.read_text()).get("mcpServers", {})
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[mcp_client] failed to read {self._config_path}: {exc}")
            return {}

    def list_configured_servers(self) -> List[str]:
        return list(self._load_config().keys())

    async def _get_session(self, server_name: str):
        if server_name in self._sessions:
            return self._sessions[server_name]

        async with self._lock:
            if server_name in self._sessions:  # re-check after acquiring lock
                return self._sessions[server_name]

            servers = self._load_config()
            if server_name not in servers:
                raise KeyError(
                    f"No MCP server named '{server_name}' in {self._config_path}. "
                    f"Configured: {list(servers.keys())}"
                )

            spec = servers[server_name]
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            env = {**os.environ, **spec.get("env", {})}
            params = StdioServerParameters(
                command=spec["command"], args=spec.get("args", []), env=env,
            )
            read, write = await self._exit_stack.enter_async_context(stdio_client(params))
            session = await self._exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._sessions[server_name] = session
            return session

    async def list_tools(self, server_name: str) -> List[Dict[str, Any]]:
        try:
            session = await self._get_session(server_name)
            result = await session.list_tools()
            return [{"name": t.name, "description": t.description} for t in result.tools]
        except Exception as exc:
            return [{"error": str(exc)}]

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        try:
            session = await self._get_session(server_name)
            result = await session.call_tool(tool_name, arguments)
            content = [
                {"type": block.type, "text": getattr(block, "text", None)}
                for block in result.content
            ]
            return {"success": not result.isError, "content": content}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def close(self):
        await self._exit_stack.aclose()
        self._sessions.clear()


_manager: Optional[MCPClientManager] = None


def get_client_manager() -> MCPClientManager:
    global _manager
    if _manager is None:
        _manager = MCPClientManager()
    return _manager
