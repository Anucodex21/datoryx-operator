"""DATORYX Automation Agent - Builds automation workflows and bots (LLM-backed)"""
from typing import Dict, Any, List

from ..base import BaseAgent, AgentCapability

SYSTEM_PROMPT = (
    "You are an automation engineer inside DATORYX. You design concrete, runnable workflow and bot "
    "specifications tailored to what's asked - not generic placeholders."
)


class AutomationAgent(BaseAgent):
    """Builds automation workflows and bots"""

    def __init__(self, llm: Any = None):
        super().__init__(name="Automation", description="Builds automation workflows and bots", llm=llm)

    def _register_capabilities(self):
        self.capabilities = [
            AgentCapability("design_workflow", "Design automation workflows", {"steps": "list"}),
            AgentCapability("build_bot", "Build automation bots", {"purpose": "str"}),
            AgentCapability("schedule_task", "Schedule automated tasks", {"task": "str", "schedule": "str"}),
            AgentCapability("integrate_systems", "Integrate external systems", {"system_a": "str", "system_b": "str"}),
            AgentCapability("monitor_automation", "Monitor automation health", {"workflow_id": "str"})
        ]

    def _register_tools(self):
        self.tools = {
            "design_workflow": self._design_workflow,
            "build_bot": self._build_bot,
            "schedule_task": self._schedule_task,
            "integrate_systems": self._integrate_systems,
            "monitor_automation": self._monitor_automation
        }

    def _design_workflow(self, steps: List[Dict],
                          triggers: List[str]) -> Dict:
        user_prompt = (
            f"Review and refine this automation workflow. Triggers: {triggers}. Proposed steps (JSON): {steps}\n\n"
            'Return JSON: {"parallel_steps": [<indices or names of steps that can run in parallel>], '
            '"conditional_steps": [<steps with conditions>], "estimated_runtime_seconds": <number>, '
            '"retry_policy": {"max_retries": <int>, "backoff": "..."}, "risks": ["..."]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "workflow_id": f"wf_{hash(str(steps)) % 10000}",
            "triggers": triggers,
            "steps": steps,
            "parallel_steps": result.get("parallel_steps", [s for s in steps if s.get("parallel", False)]),
            "conditional_steps": result.get("conditional_steps", [s for s in steps if "condition" in s]),
            "estimated_runtime": result.get("estimated_runtime_seconds", sum(s.get("duration", 1) for s in steps)),
            "retry_policy": result.get("retry_policy", {"max_retries": 3, "backoff": "exponential"}),
            "risks": result.get("risks", [])
        }

    def _build_bot(self, purpose: str, platform: str = "slack") -> Dict:
        user_prompt = (
            f"Design a bot for platform '{platform}' with this purpose: {purpose}\n\n"
            'Return JSON: {"framework": "...", "language": "...", "features": ["..."], '
            '"deployment": "...", "sample_commands": ["..."]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "purpose": purpose,
            "platform": platform,
            "framework": result.get("framework", ""),
            "language": result.get("language", ""),
            "features": result.get("features", []),
            "deployment": result.get("deployment", "Docker container"),
            "sample_commands": result.get("sample_commands", [])
        }

    def _schedule_task(self, task: str, schedule: str,
                        timezone: str = "UTC") -> Dict:
        cron_map = {"daily": "0 0 * * *", "hourly": "0 * * * *",
                    "weekly": "0 0 * * 0", "monthly": "0 0 1 * *"}
        cron = cron_map.get(schedule)
        if cron is None:
            user_prompt = (
                f"Convert this human schedule description into a standard 5-field cron expression: '{schedule}'\n\n"
                'Return JSON: {"cron": "<cron expression>", "interpretation": "<what it means>"}'
            )
            result = self._llm_json(SYSTEM_PROMPT, user_prompt)
            cron = result.get("cron", schedule)
        return {
            "task": task,
            "schedule": schedule,
            "cron": cron,
            "timezone": timezone,
            "next_run": "scheduled",
            "enabled": True
        }

    def _integrate_systems(self, system_a: str, system_b: str,
                            integration_type: str = "api") -> Dict:
        user_prompt = (
            f"Plan an integration between '{system_a}' and '{system_b}' using integration type "
            f"'{integration_type}'.\n\n"
            'Return JSON: {"protocol": "...", "authentication": "...", "data_mapping": "...", '
            '"error_handling": "...", "monitoring": "..."}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "integration_id": f"int_{system_a}_{system_b}",
            "type": integration_type,
            "protocol": result.get("protocol", ""),
            "authentication": result.get("authentication", "OAuth 2.0"),
            "data_mapping": result.get("data_mapping", f"Map fields from {system_a} to {system_b}"),
            "error_handling": result.get("error_handling", "Retry with exponential backoff"),
            "monitoring": result.get("monitoring", "Health checks every 30s")
        }

    def _monitor_automation(self, workflow_id: str) -> Dict:
        # No live telemetry system is wired up yet; be explicit that this is a placeholder
        # status rather than pretending it's a real health check.
        return {
            "workflow_id": workflow_id,
            "status": "unknown - no telemetry source connected",
            "note": "Wire this method up to the workflow_engine/scheduler event log for real health data."
        }
