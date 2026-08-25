"""DATORYX Base Agent - Foundation for all specialized agents."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timezone
import json
import re
import uuid


@dataclass
class AgentCapability:
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_tools: List[str] = field(default_factory=list)


@dataclass
class AgentResult:
    success: bool
    data: Any = None
    message: str = ""
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Base class for all DATORYX agents."""

    def __init__(self, name: str, description: str = "", llm: Any = None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.capabilities: List[AgentCapability] = []
        self.tools: Dict[str, Callable] = {}
        self.memory: Dict[str, Any] = {}
        self.status: str = "idle"
        self.performance_log: List[Dict[str, Any]] = []

        # Every agent is LLM-backed. If no LLMManager instance was injected,
        # create one - it auto-registers whichever provider API keys are
        # present in the environment and degrades to a (clearly labeled)
        # simulated response if none are configured, so agents never crash
        # for lack of credentials.
        if llm is None:
            from models.llm.manager import LLMManager
            llm = LLMManager()
        self.llm = llm

        self._register_capabilities()
        self._register_tools()
        # Every agent gets a real 'default' tool - a catch-all that answers
        # a free-form natural-language query using the LLM, framed with this
        # agent's expertise. This exists so callers that don't know (or
        # don't need to know) a specific structured tool name - like
        # Datoryx.process_query() routing a raw user query to whichever
        # agents seem relevant - always hit a real, working tool instead of
        # a hardcoded "default" that no agent subclass happens to define.
        # setdefault() so a subclass can still override "default" explicitly
        # if it wants bespoke catch-all behavior.
        self.tools.setdefault("default", self._default_tool)

    # ------------------------------------------------------------------
    # LLM helpers shared by every agent's tool implementations
    # ------------------------------------------------------------------

    def _llm_text(self, system_prompt: str, user_prompt: str,
                   temperature: float = 0.4, max_tokens: int = 1200) -> str:
        """Run a single-turn completion and return the raw text."""
        from shared.llm_helpers import llm_text
        return llm_text(self.llm, system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens)

    def _llm_json(self, system_prompt: str, user_prompt: str,
                   temperature: float = 0.3, max_tokens: int = 1500) -> Any:
        """Run a completion whose system prompt instructs the model to return
        JSON only, then parse it. Tolerates markdown code fences and stray
        prose around the JSON payload; falls back to a dict wrapping the raw
        text if parsing still fails, so callers always get *something* usable
        instead of a crash.
        """
        from shared.llm_helpers import llm_json
        return llm_json(self.llm, system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens)

    def _call_external_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call out to an external MCP server (filesystem, GitHub, browser,
        security scanners, etc.) configured in mcp_servers.json. Any agent
        can reach beyond its 20-agent-fleet built-ins this way - this is
        the client half of DATORYX's bidirectional MCP support (the server
        half is datoryx_mcp/mcp_server.py, which exposes agents outward).

        Synchronous wrapper around the async MCPClientManager so agent tool
        implementations (which are sync) can call it directly. Degrades to
        an error dict rather than raising if the server isn't configured or
        the call fails, consistent with this codebase's philosophy.
        """
        import asyncio
        from core.plugin_manager.mcp_client import get_client_manager

        async def _run():
            return await get_client_manager().call_tool(server_name, tool_name, arguments)

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                # Already inside an event loop (e.g. called from async API code) -
                # run in a fresh loop on a worker thread instead of nesting.
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, _run()).result()
            return asyncio.run(_run())
        except Exception as exc:
            return {"success": False, "error": str(exc)}


    @abstractmethod
    def _register_capabilities(self):
        """Register agent capabilities. Override in subclass."""
        pass

    @abstractmethod
    def _register_tools(self):
        """Register agent tools. Override in subclass."""
        pass

    def execute(self, task: Dict[str, Any]) -> AgentResult:
        """Execute a task using the appropriate tool."""
        import time
        start = time.time()
        self.status = "working"

        try:
            tool_name = task.get("tool", "default")
            params = task.get("params", {})

            if tool_name in self.tools:
                result = self.tools[tool_name](**params)
                execution_time = time.time() - start

                agent_result = AgentResult(
                    success=True,
                    data=result,
                    message=f"{self.name} completed task successfully",
                    execution_time=execution_time,
                    metadata={"agent": self.name, "tool": tool_name}
                )
            else:
                agent_result = AgentResult(
                    success=False,
                    message=f"Tool '{tool_name}' not found in {self.name}",
                    execution_time=time.time() - start
                )
        except Exception as e:
            agent_result = AgentResult(
                success=False,
                message=str(e),
                execution_time=time.time() - start
            )

        self.status = "idle"
        self.performance_log.append({
            "task": task,
            "result": agent_result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        return agent_result

    def get_capabilities(self) -> List[AgentCapability]:
        """Get all registered capabilities."""
        return self.capabilities

    def _default_tool(self, query: str = "", context: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Catch-all tool: answer a free-form query through this agent's
        lens using the LLM. Real (LLM-backed), not a stub - just generic
        rather than one of the agent's specific structured tools.

        Any extra kwargs a caller passes (beyond query/context) are folded
        into the prompt as additional context rather than raising a
        TypeError, since this is meant to be tolerant of whatever shape a
        generic caller sends.
        """
        tool_names = list(self.tools.keys())
        system_prompt = (
            f"You are {self.name}, an AI agent. {self.description}\n"
            f"Your available specialized capabilities are: {', '.join(tool_names) or 'general reasoning'}.\n"
            "Respond directly and practically to the user's request from your area of expertise. "
            "If the request would be better served by one of your specific capabilities, mention which "
            "one and what it would need."
        )
        extra_context = {**(context or {}), **kwargs}
        user_prompt = query if not extra_context else f"{query}\n\nAdditional context: {extra_context}"

        answer = self._llm_text(system_prompt, user_prompt)
        return {"answer": answer, "agent": self.name, "handled_by": "default"}

    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "capabilities": len(self.capabilities),
            "tools": list(self.tools.keys()),
            "tasks_completed": len(self.performance_log)
        }

    def learn(self, key: str, value: Any):
        """Store learned information."""
        self.memory[key] = value

    def recall(self, key: str) -> Optional[Any]:
        """Recall learned information."""
        return self.memory.get(key)
