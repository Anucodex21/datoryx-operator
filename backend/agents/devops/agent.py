"""DATORYX DevOps Agent - Manages CI/CD, infrastructure, and deployment pipelines (LLM-backed)"""
from typing import Dict, Any, List

from ..base import BaseAgent, AgentCapability

SYSTEM_PROMPT = (
    "You are a DevOps engineer inside DATORYX. You design concrete CI/CD pipelines, infrastructure-as-code "
    "plans, and deployment strategies grounded in current industry practice for the specific stack given."
)


class DevOpsAgent(BaseAgent):
    """Manages CI/CD, infrastructure, and deployment pipelines"""

    def __init__(self, llm: Any = None):
        super().__init__(name="DevOps", description="Manages CI/CD, infrastructure, and deployment pipelines", llm=llm)

    def _register_capabilities(self):
        self.capabilities = [
            AgentCapability("design_cicd", "Design CI/CD pipelines", {"repo": "str"}),
            AgentCapability("infrastructure_as_code", "Generate IaC templates", {"resources": "list"}),
            AgentCapability("deploy", "Plan deployments", {"artifact": "str"}),
            AgentCapability("monitor_infrastructure", "Monitor infrastructure", {"services": "list"}),
            AgentCapability("auto_scale", "Configure auto-scaling", {"service": "str", "metrics": "dict"})
        ]

    def _register_tools(self):
        self.tools = {
            "design_cicd": self._design_cicd,
            "infrastructure_as_code": self._infrastructure_as_code,
            "deploy": self._deploy,
            "monitor_infrastructure": self._monitor_infrastructure,
            "auto_scale": self._auto_scale
        }

    def _design_cicd(self, repo: str, language: str = "python",
                      tests: bool = True) -> Dict:
        user_prompt = (
            f"Design a CI/CD pipeline for repo '{repo}' written in {language}. Include tests: {tests}.\n\n"
            'Return JSON: {"stages": [{"name": "...", "steps": ["..."]}], "triggers": ["..."], '
            '"artifacts": ["..."], "rollback": "..."}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "pipeline_name": f"cicd_{repo}",
            "stages": result.get("stages", []),
            "triggers": result.get("triggers", ["push to main", "pull request"]),
            "artifacts": result.get("artifacts", []),
            "rollback": result.get("rollback", "Automatic on failure")
        }

    def _infrastructure_as_code(self, resources: List[Dict],
                                 provider: str = "aws") -> Dict:
        user_prompt = (
            f"Plan infrastructure-as-code for provider '{provider}' with these resources (JSON): {resources}\n\n"
            'Return JSON: {"tool": "...", "state_management": "...", "drift_detection": "...", '
            '"cost_estimate": "<rough monthly $ estimate with reasoning>"}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "provider": provider,
            "tool": result.get("tool", "Terraform"),
            "resources": resources,
            "state_management": result.get("state_management", "Remote backend with locking"),
            "drift_detection": result.get("drift_detection", "Enabled"),
            "cost_estimate": result.get("cost_estimate", "")
        }

    def _deploy(self, artifact: str, environment: str = "staging",
                strategy: str = "rolling") -> Dict:
        user_prompt = (
            f"Plan a {strategy} deployment of artifact '{artifact}' to environment '{environment}'.\n\n"
            'Return JSON: {"strategy_description": "...", "health_checks": ["..."], '
            '"rollback_time": "...", "estimated_duration": "..."}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "artifact": artifact,
            "environment": environment,
            "strategy": strategy,
            "strategy_description": result.get("strategy_description", ""),
            "health_checks": result.get("health_checks", ["/health", "/ready"]),
            "rollback_time": result.get("rollback_time", ""),
            "estimated_duration": result.get("estimated_duration", "")
        }

    def _monitor_infrastructure(self, services: List[str]) -> Dict:
        # No live monitoring backend is wired up - describe the monitoring plan honestly
        # rather than fabricating current metric values.
        user_prompt = (
            f"Recommend a monitoring plan for these services: {services}\n\n"
            'Return JSON: {"metrics": ["..."], "alerts": [{"metric": "...", "threshold": "...", "action": "..."}], '
            '"dashboards": ["..."], "log_aggregation": "..."}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "services_monitored": services,
            "metrics": result.get("metrics", []),
            "alerts": result.get("alerts", []),
            "dashboards": result.get("dashboards", []),
            "log_aggregation": result.get("log_aggregation", ""),
            "note": "Plan only - connect to core.monitoring.Monitoring for live status."
        }

    def _auto_scale(self, service: str, metrics: Dict[str, float],
                     min_replicas: int = 2, max_replicas: int = 20) -> Dict:
        # Replica sizing from a given metric is a simple, deterministic bound calculation.
        cpu = metrics.get("cpu", 50)
        replicas = min(max_replicas, max(min_replicas, int(cpu / 10)))
        user_prompt = (
            f"Recommend auto-scaling thresholds for service '{service}' given current metrics: {metrics}, "
            f"replica range [{min_replicas}, {max_replicas}].\n\n"
            'Return JSON: {"scale_up_threshold": "...", "scale_down_threshold": "...", "cooldown_period": "..."}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "service": service,
            "current_replicas": replicas,
            "min_replicas": min_replicas,
            "max_replicas": max_replicas,
            "scaling_metric": "CPU utilization",
            "scale_up_threshold": result.get("scale_up_threshold", "70%"),
            "scale_down_threshold": result.get("scale_down_threshold", "30%"),
            "cooldown_period": result.get("cooldown_period", "5 minutes")
        }
