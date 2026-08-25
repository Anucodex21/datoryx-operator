"""DATORYX Testing Agent - Designs test strategies and generates test cases (LLM-backed)"""
from typing import Dict, Any, List

from ..base import BaseAgent, AgentCapability

SYSTEM_PROMPT = (
    "You are a QA engineer inside DATORYX. You design concrete, specific test strategies and test cases - "
    "not generic filler like 'Valid input' - and you reason about the actual requirement or system given."
)


class TestingAgent(BaseAgent):
    """Designs test strategies and generates test cases"""

    def __init__(self, llm: Any = None):
        super().__init__(name="Testing", description="Designs test strategies and generates test cases", llm=llm)

    def _register_capabilities(self):
        self.capabilities = [
            AgentCapability("design_test_strategy", "Design test strategies", {"app_type": "str"}),
            AgentCapability("generate_test_cases", "Generate test cases", {"requirement": "str"}),
            AgentCapability("performance_test", "Plan performance tests", {"endpoint": "str"}),
            AgentCapability("security_test", "Plan security tests", {"target": "str"}),
            AgentCapability("test_coverage", "Analyze test coverage", {"source_files": "list", "test_files": "list"})
        ]

    def _register_tools(self):
        self.tools = {
            "design_test_strategy": self._design_test_strategy,
            "generate_test_cases": self._generate_test_cases,
            "performance_test": self._performance_test,
            "security_test": self._security_test,
            "test_coverage": self._test_coverage
        }

    def _design_test_strategy(self, application_type: str,
                               criticality: str = "medium") -> Dict:
        user_prompt = (
            f"Design a test strategy for a '{application_type}' application with '{criticality}' "
            "criticality.\n\n"
            'Return JSON: {"test_levels": ["..."], "automation_target": "...", "environments": ["..."], '
            '"entry_criteria": "...", "exit_criteria": "..."}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "application_type": application_type,
            "criticality": criticality,
            "test_levels": result.get("test_levels", []),
            "automation_target": result.get("automation_target", "80%"),
            "environments": result.get("environments", ["dev", "staging", "prod-like"]),
            "entry_criteria": result.get("entry_criteria", ""),
            "exit_criteria": result.get("exit_criteria", "")
        }

    def _generate_test_cases(self, requirement: str,
                              boundary_values: List = None) -> List[Dict]:
        user_prompt = (
            f"Generate specific test cases for this requirement: {requirement}\n"
            + (f"Include boundary cases around these values: {boundary_values}\n" if boundary_values else "")
            + '\nReturn JSON: {"cases": [{"id": "TC001", "type": "positive|negative|boundary|edge", '
              '"description": "<specific, not generic>", "expected": "..."}]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return result.get("cases", [])

    def _performance_test(self, endpoint: str,
                           load_profile: Dict) -> Dict:
        user_prompt = (
            f"Plan a performance test for endpoint '{endpoint}' with load profile (JSON): {load_profile}\n\n"
            'Return JSON: {"scenarios": [{"name": "...", "users": <int>, "duration": "..."}], '
            '"metrics": ["..."], "sla": {"p95_latency": "...", "error_rate": "...", "availability": "..."}}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "endpoint": endpoint,
            "load_profile": load_profile,
            "scenarios": result.get("scenarios", []),
            "metrics": result.get("metrics", ["Response time", "Throughput", "Error rate"]),
            "sla": result.get("sla", {})
        }

    def _security_test(self, target: str,
                        test_types: List[str] = None) -> Dict:
        user_prompt = (
            f"Plan a security test engagement against '{target}'."
            + (f" Focus on these test types: {test_types}." if test_types else "")
            + '\n\nReturn JSON: {"test_types": ["..."], "tools": ["..."], "scope": "...", '
              '"methodology": "...", "deliverables": ["..."]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "target": target,
            "test_types": result.get("test_types", test_types or []),
            "tools": result.get("tools", []),
            "scope": result.get("scope", ""),
            "methodology": result.get("methodology", ""),
            "deliverables": result.get("deliverables", [])
        }

    def _test_coverage(self, source_files: List[str],
                        test_files: List[str]) -> Dict:
        # Without a real coverage tool wired up, be honest about that instead of
        # fabricating coverage percentages - but ask the LLM to reason about the
        # ratio of source to test files as a rough qualitative signal.
        user_prompt = (
            f"There are {len(source_files)} source files: {source_files} and {len(test_files)} test "
            f"files: {test_files}. No coverage tool is connected.\n\n"
            'Return JSON: {"qualitative_assessment": "<is test coverage likely adequate given this ratio "'
            '"and naming, and why>", "likely_uncovered_areas": ["..."], "recommendations": ["..."]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "source_files": len(source_files),
            "test_files": len(test_files),
            "qualitative_assessment": result.get("qualitative_assessment", ""),
            "likely_uncovered_areas": result.get("likely_uncovered_areas", []),
            "recommendations": result.get("recommendations", []),
            "note": "No coverage tool connected - this is a qualitative estimate, not measured coverage."
        }
