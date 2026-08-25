"""DATORYX Agent Coordinator - Orchestrates all 20 specialized agents."""
from typing import Dict, Any, List, Optional, Callable
import asyncio
from datetime import datetime, timezone

from shared.types import Task, Priority
from agents.base import BaseAgent, AgentResult


class AgentCoordinator:
    """Central coordinator for all DATORYX agents with intelligent routing."""

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_map: Dict[str, str] = {}  # capability -> agent_name
        self._task_history: List[Dict[str, Any]] = []
        self._routing_rules: List[Callable] = []
        self._performance_metrics: Dict[str, Dict] = {}

    def register_agent(self, agent: BaseAgent):
        """Register an agent and index its capabilities."""
        self._agents[agent.name] = agent
        self._performance_metrics[agent.name] = {
            "tasks_completed": 0,
            "avg_success_rate": 1.0,
            "total_execution_time": 0.0
        }

        # Index capabilities
        for cap in agent.capabilities:
            self._agent_map[cap.name] = agent.name

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get agent by name."""
        return self._agents.get(name)

    def find_agent_for_task(self, task_type: str) -> Optional[BaseAgent]:
        """Find the best agent for a task type."""
        # Direct capability match
        if task_type in self._agent_map:
            return self._agents.get(self._agent_map[task_type])

        # Fuzzy matching
        for cap_name, agent_name in self._agent_map.items():
            if task_type.lower() in cap_name.lower() or cap_name.lower() in task_type.lower():
                return self._agents.get(agent_name)

        # Default to first available
        return next(iter(self._agents.values())) if self._agents else None

    async def execute_task(self, task: Dict[str, Any], 
                          agent_name: str = None) -> AgentResult:
        """Execute a task with the appropriate agent."""
        if agent_name:
            agent = self._agents.get(agent_name)
        else:
            task_type = task.get("type", task.get("tool", "unknown"))
            agent = self.find_agent_for_task(task_type)

        if not agent:
            return AgentResult(
                success=False,
                message=f"No agent found for task: {task}"
            )

        start_time = asyncio.get_event_loop().time()
        result = agent.execute(task)
        execution_time = asyncio.get_event_loop().time() - start_time

        # Update metrics
        metrics = self._performance_metrics[agent.name]
        metrics["tasks_completed"] += 1
        metrics["total_execution_time"] += execution_time
        metrics["avg_success_rate"] = (
            (metrics["avg_success_rate"] * (metrics["tasks_completed"] - 1) + 
             (1.0 if result.success else 0.0)) / metrics["tasks_completed"]
        )

        self._task_history.append({
            "agent": agent.name,
            "task": task,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        return result

    async def execute_pipeline(self, tasks: List[Dict[str, Any]],
                              parallel: bool = False) -> List[AgentResult]:
        """Execute a pipeline of tasks."""
        if parallel:
            return await asyncio.gather(*[
                self.execute_task(task) for task in tasks
            ])
        else:
            results = []
            for task in tasks:
                result = await self.execute_task(task)
                results.append(result)
                # Pass previous result as context
                if result.success and result.data:
                    task.setdefault("context", {})["previous_result"] = result.data
            return results

    async def collaborate(self, task: Dict[str, Any], 
                         agent_names: List[str]) -> Dict[str, AgentResult]:
        """Have multiple agents collaborate on a task."""
        results = {}
        for name in agent_names:
            agent = self._agents.get(name)
            if agent:
                results[name] = agent.execute(task)
        return results

    def get_all_agents(self) -> List[Dict[str, Any]]:
        """Get status of all agents."""
        return [agent.get_status() for agent in self._agents.values()]

    def get_capabilities(self) -> Dict[str, List[str]]:
        """Get all available capabilities mapped to agents."""
        caps = {}
        for agent in self._agents.values():
            caps[agent.name] = [c.name for c in agent.capabilities]
        return caps

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report for all agents."""
        return {
            "agents": self._performance_metrics,
            "total_tasks": len(self._task_history),
            "overall_success_rate": sum(
                m["avg_success_rate"] for m in self._performance_metrics.values()
            ) / max(len(self._performance_metrics), 1)
        }

    def get_task_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent task execution history."""
        return self._task_history[-limit:]
