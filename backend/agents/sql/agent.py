"""DATORYX SQL Agent - Writes, optimizes, and executes SQL queries (LLM-backed)"""
from typing import Dict, Any, List

from ..base import BaseAgent, AgentCapability

SYSTEM_PROMPT = (
    "You are a senior database engineer inside DATORYX, an autonomous data platform. "
    "You write correct, dialect-agnostic ANSI SQL unless a dialect is specified, "
    "and you reason carefully about indexes, cardinality, and query cost."
)


class SQLAgent(BaseAgent):
    """Writes, optimizes, and executes SQL queries"""

    def __init__(self, llm: Any = None):
        super().__init__(
            name="SQL",
            description="Writes, optimizes, and executes SQL queries",
            llm=llm,
        )

    def _register_capabilities(self):
        self.capabilities = [
            AgentCapability("write_query", "Generate SQL queries", {"intent": "str", "tables": "dict"}),
            AgentCapability("optimize_query", "Optimize SQL performance", {"query": "str"}),
            AgentCapability("explain_plan", "Analyze query execution plans", {"query": "str"}),
            AgentCapability("generate_schema", "Generate database schemas", {"entities": "dict"}),
            AgentCapability("migrate_data", "Plan data migrations", {"source": "str", "target": "str"})
        ]

    def _register_tools(self):
        self.tools = {
            "write_query": self._write_query,
            "optimize_query": self._optimize_query,
            "explain_plan": self._explain_plan,
            "generate_schema": self._generate_schema,
            "migrate_data": self._migrate_data
        }

    def _write_query(self, intent: str, tables: Dict[str, List[str]]) -> Dict:
        user_prompt = (
            f"Write a SQL query for this intent: {intent}\n"
            f"Available tables and columns (JSON): {tables}\n\n"
            'Return JSON: {"suggested_query": "<the SQL>", "explanation": "<why it satisfies the intent>", '
            '"assumptions": ["..."]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "intent": intent,
            "tables": tables,
            "suggested_query": result.get("suggested_query", result.get("raw_response", "")),
            "explanation": result.get("explanation", ""),
            "assumptions": result.get("assumptions", [])
        }

    def _optimize_query(self, query: str, table_size: int = 1000000) -> Dict:
        user_prompt = (
            f"Optimize this SQL query for a table with approximately {table_size} rows:\n\n{query}\n\n"
            'Return JSON: {"suggestions": ["..."], "optimized_query": "<rewritten SQL, or the same query if already optimal>", '
            '"estimated_cost": "low|medium|high", "reasoning": "<short explanation>"}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "original": query,
            "suggestions": result.get("suggestions", []),
            "optimized_query": result.get("optimized_query", query),
            "estimated_cost": result.get("estimated_cost", "unknown"),
            "reasoning": result.get("reasoning", "")
        }

    def _explain_plan(self, query: str) -> Dict:
        user_prompt = (
            f"Produce a plausible query execution plan analysis for this SQL query, as a typical "
            f"PostgreSQL-style EXPLAIN ANALYZE might report it:\n\n{query}\n\n"
            'Return JSON: {"plan": [{"step": "...", "cost": <number>, "rows": <number>}], '
            '"total_cost": <number>, "index_suggestions": ["..."]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "query": query,
            "plan": result.get("plan", []),
            "total_cost": result.get("total_cost", 0),
            "index_suggestions": result.get("index_suggestions", [])
        }

    def _generate_schema(self, entities: Dict[str, Dict[str, str]]) -> Dict:
        user_prompt = (
            f"Generate a normalized SQL DDL schema for these entities and their proposed columns (JSON): "
            f"{entities}\n\n"
            "Add sensible primary keys, foreign keys between related tables, NOT NULL constraints where "
            "appropriate, and helpful indexes.\n\n"
            'Return JSON: {"ddl": "<full CREATE TABLE statements, newline separated>", "tables": ["..."], '
            '"notes": ["<design decisions or trade-offs>"]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "ddl": result.get("ddl", ""),
            "tables": result.get("tables", list(entities.keys())),
            "notes": result.get("notes", [])
        }

    def _migrate_data(self, source_table: str, target_table: str,
                       transformations: List[str]) -> Dict:
        user_prompt = (
            f"Plan a data migration from table '{source_table}' to table '{target_table}', applying these "
            f"transformations: {transformations}\n\n"
            'Return JSON: {"steps": ["<ordered SQL or procedural steps>"], "rollback": "<rollback SQL>", '
            '"risks": ["..."]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "migration_id": f"mig_{source_table}_{target_table}",
            "steps": result.get("steps", []),
            "rollback": result.get("rollback", f"DROP TABLE {target_table}"),
            "risks": result.get("risks", [])
        }
