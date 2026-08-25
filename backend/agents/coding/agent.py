"""DATORYX Coding Agent - Writes, reviews, and optimizes code across languages (LLM-backed)"""
from typing import Dict, Any, List

from ..base import BaseAgent, AgentCapability

SYSTEM_PROMPT = (
    "You are a senior software engineer inside DATORYX. You write complete, working code - never TODO "
    "placeholders or stub bodies - and you review/refactor/debug real code carefully and specifically."
)


class CodingAgent(BaseAgent):
    """Writes, reviews, and optimizes code across languages"""

    def __init__(self, llm: Any = None):
        super().__init__(name="Coding", description="Writes, reviews, and optimizes code across languages", llm=llm)

    def _register_capabilities(self):
        self.capabilities = [
            AgentCapability("write_code", "Generate code from requirements", {"requirements": "str", "language": "str"}),
            AgentCapability("review_code", "Review code quality", {"code": "str"}),
            AgentCapability("refactor", "Refactor existing code", {"code": "str", "goals": "list"}),
            AgentCapability("generate_tests", "Generate unit tests", {"code": "str"}),
            AgentCapability("debug", "Debug code errors", {"error": "str", "code": "str"})
        ]

    def _register_tools(self):
        self.tools = {
            "write_code": self._write_code,
            "review_code": self._review_code,
            "refactor": self._refactor,
            "generate_tests": self._generate_tests,
            "debug": self._debug
        }

    def _write_code(self, requirements: str, language: str = "python",
                     framework: str = None) -> Dict:
        user_prompt = (
            f"Write complete, working {language} code" + (f" using {framework}" if framework else "") +
            f" that satisfies these requirements:\n{requirements}\n\n"
            'Return JSON: {"code": "<the complete code>", "estimated_lines": <int>, '
            '"notes": "<anything the caller should know>"}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt, max_tokens=2000)
        code = result.get("code", "")
        return {
            "language": language,
            "framework": framework,
            "code": code,
            "requirements": requirements,
            "estimated_lines": result.get("estimated_lines", len(code.splitlines())),
            "notes": result.get("notes", "")
        }

    def _review_code(self, code: str, language: str = "python") -> Dict:
        user_prompt = (
            f"Review this {language} code for correctness, security, and style issues:\n\n{code}\n\n"
            'Return JSON: {"issues": [{"type": "error|warning|suggestion", "message": "..."}], '
            '"score": <0-100 int>, "suggestions": ["..."]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        issues = result.get("issues", [])
        score = result.get("score")
        if score is None:
            score = max(0, 100 - len(issues) * 10)
        grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D"
        return {
            "issues": issues,
            "score": score,
            "grade": grade,
            "suggestions": result.get("suggestions", [])
        }

    def _refactor(self, code: str, goals: List[str]) -> Dict:
        user_prompt = (
            f"Refactor this code to achieve these goals: {goals}\n\n{code}\n\n"
            'Return JSON: {"refactored": "<the full refactored code>", '
            '"changes_made": ["<what actually changed and why>"]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt, max_tokens=2000)
        return {
            "original": code,
            "refactored": result.get("refactored", code),
            "changes_made": result.get("changes_made", goals)
        }

    def _generate_tests(self, code: str, language: str = "python") -> Dict:
        user_prompt = (
            f"Write unit tests for this {language} code, covering happy path, edge cases, and error "
            f"handling:\n\n{code}\n\n"
            'Return JSON: {"test_framework": "...", "tests": "<the complete test file content>", '
            '"test_categories": ["..."]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt, max_tokens=2000)
        return {
            "language": language,
            "test_framework": result.get("test_framework", "pytest" if language == "python" else "jest"),
            "tests": result.get("tests", ""),
            "test_categories": result.get("test_categories", ["unit", "edge_cases"])
        }

    def _debug(self, error_message: str, code: str,
               language: str = "python") -> Dict:
        user_prompt = (
            f"This {language} code raised the following error:\n{error_message}\n\nCode:\n{code}\n\n"
            'Return JSON: {"error_type": "...", "root_cause": "...", "suggested_fix": "...", '
            '"fixed_code": "<corrected code if a small fix suffices, else empty string>", '
            '"confidence": <0-1 float>}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt, max_tokens=1800)
        return {
            "error_type": result.get("error_type", "Unknown"),
            "root_cause": result.get("root_cause", ""),
            "suggested_fix": result.get("suggested_fix", ""),
            "fixed_code": result.get("fixed_code", ""),
            "confidence": result.get("confidence", 0.5)
        }
