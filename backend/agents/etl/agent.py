"""DATORYX ETL Agent - Designs and executes Extract, Transform, Load processes (LLM-backed)"""
from typing import Dict, Any, List

from ..base import BaseAgent, AgentCapability

SYSTEM_PROMPT = (
    "You are an ETL engineer inside DATORYX. You design concrete extract/transform/load plans tailored "
    "to the sources and destinations given, and you are honest about what is a plan versus an actual "
    "execution result."
)


class ETLAgent(BaseAgent):
    """Designs and executes Extract, Transform, Load processes"""

    def __init__(self, llm: Any = None):
        super().__init__(
            name="ETL",
            description="Designs and executes Extract, Transform, Load processes",
            llm=llm,
        )

    def _register_capabilities(self):
        self.capabilities = [
            AgentCapability("design_etl", "Design ETL pipelines", {"sources": "list", "destination": "str"}),
            AgentCapability("extract_data", "Extract data from sources", {"source": "str"}),
            AgentCapability("transform_data", "Apply data transformations", {"data": "list", "rules": "list"}),
            AgentCapability("load_data", "Load data to targets", {"data": "list", "target": "str"}),
            AgentCapability("validate_etl", "Validate ETL results", {"source_count": "int", "target_count": "int"})
        ]

    def _register_tools(self):
        self.tools = {
            "design_etl": self._design_etl,
            "extract_data": self._extract_data,
            "transform_data": self._transform_data,
            "load_data": self._load_data,
            "validate_etl": self._validate_etl
        }

    def _design_etl(self, sources: List[str], destination: str,
                     transformations: List[str]) -> Dict:
        user_prompt = (
            f"Design an ETL pipeline pulling from sources {sources} into destination '{destination}', "
            f"applying transformations: {transformations}\n\n"
            'Return JSON: {"extract": [{"source": "...", "method": "full|incremental", "notes": "..."}], '
            '"load": {"mode": "...", "notes": "..."}, "orchestration": "..."}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "pipeline_name": f"etl_{'_'.join(sources)}_{destination}",
            "extract": result.get("extract", [{"source": s} for s in sources]),
            "transform": transformations,
            "load": result.get("load", {"destination": destination, "mode": "upsert"}),
            "orchestration": result.get("orchestration", "Airflow DAG with retry logic")
        }

    def _extract_data(self, source: str, query: str = None,
                       batch_size: int = 1000) -> Dict:
        # No real source connector is wired up yet - be explicit that this is a plan,
        # not fabricated execution telemetry (record counts, durations, checksums).
        return {
            "source": source,
            "query": query,
            "batch_size": batch_size,
            "status": "not executed - no source connector configured",
            "note": "Wire this to data_platform.DataPlatform or a real connector to get actual extraction results."
        }

    def _transform_data(self, data: List[Dict], rules: List[Dict]) -> Dict:
        # Applying explicit, well-defined rules is deterministic - execute them exactly.
        transformed = []
        for row in data:
            new_row = dict(row)
            for rule in rules:
                op = rule.get("operation")
                col = rule.get("column")
                if op == "uppercase" and col in new_row:
                    new_row[col] = str(new_row[col]).upper()
                elif op == "replace" and col in new_row:
                    new_row[col] = str(new_row[col]).replace(rule.get("old", ""), rule.get("new", ""))
                elif op == "cast" and col in new_row:
                    caster = {"int": int, "float": float, "str": str, "bool": bool}.get(rule.get("type"))
                    if caster:
                        new_row[col] = caster(new_row[col])
            transformed.append(new_row)
        return {"records_transformed": len(transformed), "sample": transformed[:3]}

    def _load_data(self, data: List[Dict], target: str, mode: str = "append") -> Dict:
        # No real target connector is wired up - report a plan, not a fabricated success record.
        return {
            "target": target,
            "records_to_load": len(data),
            "mode": mode,
            "status": "not executed - no target connector configured",
            "note": "Wire this to data_platform.DataPlatform or a real connector to actually persist data."
        }

    def _validate_etl(self, source_count: int, target_count: int,
                       checksums: Dict) -> Dict:
        # Count/checksum comparison is deterministic.
        match = source_count == target_count
        return {
            "source_records": source_count,
            "target_records": target_count,
            "match": match,
            "checksum_valid": checksums.get("source") == checksums.get("target"),
            "validation_status": "passed" if match else "failed"
        }
