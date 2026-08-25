"""DATORYX Agent Registry.

Discovers every specialized agent shipped under backend/agents/<name>/agent.py
and exposes them as a single flat map: agent_key -> BaseAgent instance.

This is the piece that makes the whole system MCP-able: instead of hand
writing an MCP tool per agent per capability, the MCP server (see
backend/mcp/server.py) walks this registry once and auto-generates one MCP
tool per (agent, capability) pair. Add a new agent under agents/<name>/agent.py
that follows the existing BaseAgent contract and it shows up as MCP tools
with zero extra wiring.
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Dict, Type

from agents.base import BaseAgent

_AGENTS_DIR = Path(__file__).parent


def discover_agent_classes() -> Dict[str, Type[BaseAgent]]:
    """Scan agents/<name>/agent.py modules and return {name: AgentClass}.

    A subpackage is included if it defines exactly one BaseAgent subclass in
    its agent.py. Packages without an agent.py (e.g. this file's own
    __pycache__) are silently skipped.
    """
    found: Dict[str, Type[BaseAgent]] = {}
    for entry in sorted(_AGENTS_DIR.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        agent_file = entry / "agent.py"
        if not agent_file.exists():
            continue
        module_name = f"agents.{entry.name}.agent"
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - defensive, keeps one bad agent from killing the rest
            print(f"[registry] skipping {module_name}: {exc}")
            continue
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseAgent)
                and attr is not BaseAgent
                and attr.__module__ == module_name
            ):
                found[entry.name] = attr
                break
    return found


class AgentRegistry:
    """Lazily instantiates and caches every discovered agent, sharing one
    LLMManager across all of them so provider connections / response cache
    are reused instead of duplicated per agent.
    """

    def __init__(self, llm=None):
        if llm is None:
            from models.llm.manager import LLMManager
            llm = LLMManager()
        self._llm = llm
        self._classes = discover_agent_classes()
        self._instances: Dict[str, BaseAgent] = {}

    def names(self):
        return list(self._classes.keys())

    def get(self, name: str) -> BaseAgent:
        if name not in self._instances:
            if name not in self._classes:
                raise KeyError(f"No agent registered under '{name}'")
            self._instances[name] = self._classes[name](llm=self._llm)
        return self._instances[name]

    def all(self) -> Dict[str, BaseAgent]:
        for name in self._classes:
            self.get(name)
        return dict(self._instances)

    def capability_index(self):
        """Flat list of (agent_name, capability) across every agent, used by
        the MCP server to generate one tool per capability.
        """
        index = []
        for name, agent in self.all().items():
            for cap in agent.get_capabilities():
                index.append((name, agent, cap))
        return index
