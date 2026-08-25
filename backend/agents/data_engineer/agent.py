"""DATORYX DataEngineer Agent - Designs, builds, and maintains data pipelines and infrastructure (LLM-backed)"""
from typing import Dict, Any, List

from ..base import BaseAgent, AgentCapability

SYSTEM_PROMPT = (
    "You are a data engineer inside DATORYX. You design concrete, production-grade pipeline "
    "architectures and ingestion strategies tailored to the specific source/destination given."
)


class DataEngineerAgent(BaseAgent):
    """Designs, builds, and maintains data pipelines and infrastructure"""

    def __init__(self, llm: Any = None):
        super().__init__(
            name="DataEngineer",
            description="Designs, builds, and maintains data pipelines and infrastructure",
            llm=llm,
        )

    def _register_capabilities(self):
        self.capabilities = [
            AgentCapability("design_pipeline", "Design data pipelines", {"source": "str", "destination": "str"}),
            AgentCapability("validate_schema", "Validate data schemas", {"data": "dict", "schema": "dict"}),
            AgentCapability("build_ingestion", "Build data ingestion connectors", {"source_type": "str"}),
            AgentCapability("monitor_pipeline", "Monitor pipeline health", {"pipeline_id": "str"}),
            AgentCapability("optimize_storage", "Optimize data storage", {"table": "str"})
        ]

    def _register_tools(self):
        self.tools = {
            "design_pipeline": self._design_pipeline,
            "validate_schema": self._validate_schema,
            "build_ingestion": self._build_ingestion,
            "monitor_pipeline": self._monitor_pipeline,
            "optimize_storage": self._optimize_storage
        }

    def _design_pipeline(self, source: str, destination: str,
                          format: str = "parquet", schedule: str = "daily") -> Dict:
        user_prompt = (
            f"Design a data pipeline from source '{source}' to destination '{destination}', output format "
            f"'{format}', running '{schedule}'.\n\n"
            'Return JSON: {"extract": {"method": "...", "notes": "..."}, '
            '"transform": {"steps": ["..."]}, "load": {"mode": "...", "notes": "..."}, '
            '"orchestration": "...", "failure_handling": "..."}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "pipeline_id": f"pipe_{source}_{destination}",
            "architecture": {
                "extract": {"source": source, **result.get("extract", {"method": "batch"})},
                "transform": result.get("transform", {"steps": []}),
                "load": {"destination": destination, "format": format, **result.get("load", {})}
            },
            "schedule": schedule,
            "orchestration": result.get("orchestration", ""),
            "failure_handling": result.get("failure_handling", ""),
            "status": "designed"
        }

    def _validate_schema(self, data: Dict, expected_schema: Dict) -> Dict:
        # Structural validation is deterministic - avoid eval() and check types directly.
        type_map = {"str": str, "int": int, "float": (int, float), "bool": bool,
                    "list": list, "dict": dict}
        issues = []
        for key, expected_type in expected_schema.items():
            if key not in data:
                issues.append(f"Missing field: {key}")
                continue
            py_type = type_map.get(expected_type)
            if py_type and not isinstance(data[key], py_type):
                issues.append(f"Type mismatch for {key}: expected {expected_type}, got {type(data[key]).__name__}")
        return {"valid": len(issues) == 0, "issues": issues}

    def _build_ingestion(self, source_type: str, config: Dict) -> Dict:
        user_prompt = (
            f"Design an ingestion connector for source type '{source_type}' with config (JSON): {config}\n\n"
            'Return JSON: {"connector_description": "...", "retry_strategy": "...", '
            '"schema_evolution_handling": "..."}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "connector": result.get("connector_description", f"Connector for {source_type}"),
            "retry_strategy": result.get("retry_strategy", ""),
            "schema_evolution_handling": result.get("schema_evolution_handling", ""),
            "config": config,
            "status": "built"
        }

    def _monitor_pipeline(self, pipeline_id: str) -> Dict:
        # No live metrics store is wired up - report that honestly instead of fabricating numbers.
        return {
            "pipeline_id": pipeline_id,
            "health": "unknown - no telemetry source connected",
            "note": "Connect this to core.monitoring.Monitoring for real throughput/latency/error data."
        }

    def _optimize_storage(self, table: str, strategy: str = "partition") -> Dict:
        user_prompt = (
            f"Recommend a storage optimization for table '{table}' using strategy '{strategy}'.\n\n"
            'Return JSON: {"recommendation": "<concrete DDL or config change>", "expected_impact": "..."}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "optimization": result.get("recommendation", ""),
            "expected_impact": result.get("expected_impact", ""),
            "table": table
        }
