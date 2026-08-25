"""DATORYX Operator - MCP Server.

This is the piece that makes the whole agent fleet MCP-native: it walks the
AgentRegistry once at startup and auto-generates one MCP tool per
(agent, capability) pair -- currently 20 agents x their registered
capabilities. Any MCP client (Claude Desktop, Claude Code, Cursor, a
custom Operator UI) can `mcp connect` to this process over stdio and get
every DATORYX agent as a first-class tool, with real LLM-backed execution
behind it (not a stub).

Run:
    python -m datoryx_mcp.mcp_server        (from backend/, stdio transport)

Add to an MCP client config, e.g. Claude Desktop's claude_desktop_config.json:
    {
      "mcpServers": {
        "datoryx": {
          "command": "python",
          "args": ["-m", "datoryx_mcp.mcp_server"],
          "cwd": "/path/to/datoryx-operator/backend"
        }
      }
    }
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

# Make `backend/` importable as the project root regardless of cwd.
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from mcp.server import MCPServer  # noqa: E402  (real mcp SDK, not this package)
from mcp.server.stdio import stdio_server  # noqa: E402

from agents.registry import AgentRegistry  # noqa: E402

_TYPE_MAP = {
    "str": "string", "string": "string",
    "int": "integer", "integer": "integer",
    "float": "number", "number": "number",
    "bool": "boolean", "boolean": "boolean",
    "list": "array", "dict": "object",
}


def _json_schema_for(params: Dict[str, Any]) -> Dict[str, Any]:
    """Turn an AgentCapability.parameters dict (name -> loose type string)
    into a JSON Schema `properties` block MCP clients can render as a form.
    """
    props = {}
    for pname, ptype in (params or {}).items():
        json_type = _TYPE_MAP.get(str(ptype).lower(), "string")
        props[pname] = {"type": json_type}
    return {
        "type": "object",
        "properties": props,
        "required": list(props.keys()),
    }


def build_server() -> MCPServer:
    registry = AgentRegistry()
    server = MCPServer(
        name="datoryx-operator",
        title="DATORYX Operator",
        description=(
            "Multi-agent AI operator: 20 specialized, LLM-backed agents "
            "(coding, research, data science, security, devops, etc.) "
            "exposed as MCP tools, backed by real multi-provider LLM "
            "routing with local/offline fallback."
        ),
        version="1.0.0",
    )

    tool_count = 0
    for agent_name, agent, capability in registry.capability_index():
        tool_id = f"{agent_name}.{capability.name}"
        schema = _json_schema_for(capability.parameters)

        def _make_handler(agent=agent, capability=capability):
            def _handler(**kwargs) -> Dict[str, Any]:
                result = agent.execute({
                    "tool": capability.name,
                    "params": kwargs,
                })
                return {
                    "success": result.success,
                    "data": result.data,
                    "message": result.message,
                    "agent": agent.name,
                }
            return _handler

        handler = _make_handler()
        handler.__name__ = tool_id.replace(".", "_")
        handler.__doc__ = f"[{agent.name}] {capability.description}"

        server.tool(
            name=tool_id,
            description=f"[{agent.name}] {capability.description}",
        )(handler)
        tool_count += 1

    # One catch-all tool per agent too, for free-form queries that don't map
    # cleanly onto a single structured capability (mirrors BaseAgent's own
    # default tool, so this is still real LLM execution, not a stub).
    for agent_name, agent in registry.all().items():
        def _make_default_handler(agent=agent):
            def _handler(query: str, context: str = "") -> Dict[str, Any]:
                result = agent.execute({
                    "tool": "default",
                    "params": {"query": query, "context": {"note": context} if context else {}},
                })
                return {"success": result.success, "data": result.data, "message": result.message}
            return _handler

        server.tool(
            name=f"{agent_name}.ask",
            description=f"Ask {agent.name} a free-form question in its area of expertise: {agent.description}",
        )(_make_default_handler())
        tool_count += 1

    print(f"[datoryx-mcp] registered {tool_count} tools across {len(registry.names())} agents", file=sys.stderr)
    return server


async def _run():
    server = build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main():
    asyncio.run(_run())


if __name__ == "__main__":
    main()
