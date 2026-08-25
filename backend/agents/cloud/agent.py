"""DATORYX Cloud Agent - Designs cloud architecture and manages cloud resources (LLM-backed)"""
from typing import Dict, Any, List

from ..base import BaseAgent, AgentCapability

SYSTEM_PROMPT = (
    "You are a cloud solutions architect inside DATORYX, deeply familiar with AWS, Azure, and GCP service "
    "catalogs. You give concrete, provider-specific recommendations, not generic advice."
)


class CloudAgent(BaseAgent):
    """Designs cloud architecture and manages cloud resources"""

    def __init__(self, llm: Any = None):
        super().__init__(name="Cloud", description="Designs cloud architecture and manages cloud resources", llm=llm)

    def _register_capabilities(self):
        self.capabilities = [
            AgentCapability("design_architecture", "Design cloud architectures", {"requirements": "dict"}),
            AgentCapability("cost_optimize", "Optimize cloud costs", {"resources": "list"}),
            AgentCapability("multi_cloud_strategy", "Plan multi-cloud strategies", {"services": "list"}),
            AgentCapability("disaster_recovery", "Design DR plans", {"rto": "int", "rpo": "int"}),
            AgentCapability("cloud_migration", "Plan cloud migrations", {"applications": "list"})
        ]

    def _register_tools(self):
        self.tools = {
            "design_architecture": self._design_cloud_architecture,
            "cost_optimize": self._cost_optimize,
            "multi_cloud_strategy": self._multi_cloud_strategy,
            "disaster_recovery": self._disaster_recovery,
            "cloud_migration": self._cloud_migration
        }

    def _design_cloud_architecture(self, requirements: Dict,
                                    provider: str = "aws") -> Dict:
        user_prompt = (
            f"Design a cloud architecture on '{provider}' for these requirements (JSON): {requirements}\n\n"
            'Return JSON: {"architecture": {"compute": "...", "storage": "...", "database": "...", '
            '"networking": "...", "security": "..."}, "pattern": "...", "availability": "...", '
            '"estimated_cost": "...", "compliance": ["..."]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "provider": provider,
            "architecture": result.get("architecture", {}),
            "pattern": result.get("pattern", requirements.get("pattern", "microservices")),
            "availability": result.get("availability", ""),
            "regions": requirements.get("regions", ["us-east-1"]),
            "estimated_cost": result.get("estimated_cost", ""),
            "compliance": result.get("compliance", [])
        }

    def _cost_optimize(self, resources: List[Dict],
                        savings_target: float = 0.2) -> Dict:
        # Aggregate current spend deterministically; let the LLM reason about which
        # specific resources to act on and why, grounded in the real utilization data.
        current_cost = sum(r.get("monthly_cost", 0) for r in resources)
        user_prompt = (
            f"Given these resources and their utilization (JSON): {resources}, current total monthly cost "
            f"${current_cost}, target savings {savings_target:.0%}.\n\n"
            'Return JSON: {"recommendations": ["<specific resource-by-resource recommendation>"], '
            '"realistic_savings_estimate": <dollar amount>}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "current_monthly_cost": current_cost,
            "potential_savings": result.get("realistic_savings_estimate", current_cost * savings_target),
            "recommendations": result.get("recommendations", []),
            "savings_target_met": len(result.get("recommendations", [])) >= 3
        }

    def _multi_cloud_strategy(self, services: List[str],
                               providers: List[str]) -> Dict:
        user_prompt = (
            f"Design a multi-cloud strategy across providers {providers} for services {services}.\n\n"
            'Return JSON: {"strategy": "...", "workload_distribution": {"primary": "...", "secondary": "...", '
            '"failover": "..."}, "data_sync": "...", "challenges": ["..."], "benefits": ["..."]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "providers": providers,
            "services": services,
            "strategy": result.get("strategy", ""),
            "workload_distribution": result.get("workload_distribution", {}),
            "data_sync": result.get("data_sync", ""),
            "challenges": result.get("challenges", []),
            "benefits": result.get("benefits", [])
        }

    def _disaster_recovery(self, rto: int = 4, rpo: int = 1) -> Dict:
        user_prompt = (
            f"Design a disaster recovery plan for RTO={rto} hours and RPO={rpo} hours.\n\n"
            'Return JSON: {"tier": "...", "backup_strategy": "...", "test_frequency": "...", '
            '"runbook": ["<ordered steps>"]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "rto_hours": rto,
            "rpo_hours": rpo,
            "tier": result.get("tier", ""),
            "backup_strategy": result.get("backup_strategy", ""),
            "test_frequency": result.get("test_frequency", ""),
            "runbook": result.get("runbook", [])
        }

    def _cloud_migration(self, applications: List[str],
                          target: str = "aws") -> Dict:
        user_prompt = (
            f"Plan a migration of these applications to '{target}': {applications}\n\n"
            'Return JSON: {"phases": [{"name": "...", "duration": "...", "tasks": ["..."]}], '
            '"total_duration": "...", "migration_strategy": "...", "risk_mitigation": ["..."]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "applications": applications,
            "target_platform": target,
            "phases": result.get("phases", []),
            "total_duration": result.get("total_duration", ""),
            "migration_strategy": result.get("migration_strategy", ""),
            "risk_mitigation": result.get("risk_mitigation", [])
        }
