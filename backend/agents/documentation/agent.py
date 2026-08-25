"""DATORYX Documentation Agent - Generates technical documentation and guides (LLM-backed)"""
from typing import Dict, Any, List

from ..base import BaseAgent, AgentCapability

SYSTEM_PROMPT = (
    "You are a technical writer inside DATORYX. You produce real, specific documentation content from "
    "the material given - not placeholder section headers."
)


class DocumentationAgent(BaseAgent):
    """Generates technical documentation and guides"""

    def __init__(self, llm: Any = None):
        super().__init__(name="Documentation", description="Generates technical documentation and guides", llm=llm)

    def _register_capabilities(self):
        self.capabilities = [
            AgentCapability("generate_api_docs", "Generate API documentation", {"endpoints": "list"}),
            AgentCapability("create_user_guide", "Create user guides", {"feature": "str"}),
            AgentCapability("code_documentation", "Document code", {"code": "str"}),
            AgentCapability("architecture_docs", "Document architecture", {"components": "list"}),
            AgentCapability("changelog", "Generate changelogs", {"commits": "list"})
        ]

    def _register_tools(self):
        self.tools = {
            "generate_api_docs": self._generate_api_docs,
            "create_user_guide": self._create_user_guide,
            "code_documentation": self._code_documentation,
            "architecture_docs": self._architecture_docs,
            "changelog": self._changelog
        }

    def _generate_api_docs(self, endpoints: List[Dict],
                            format: str = "openapi") -> Dict:
        user_prompt = (
            f"Generate {format} API documentation for these endpoints (JSON): {endpoints}\n\n"
            'Return JSON: {"spec_version": "...", "documentation": "<the actual doc content, markdown or '
            'YAML as appropriate>", "includes": ["..."]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "format": format,
            "endpoints": endpoints,
            "spec_version": result.get("spec_version", "3.0.0" if format == "openapi" else "2.0"),
            "documentation": result.get("documentation", ""),
            "includes": result.get("includes", []),
            "generated_file": f"api_spec.{format}",
            "interactive": format == "openapi"
        }

    def _create_user_guide(self, feature: str,
                            audience: str = "general") -> Dict:
        user_prompt = (
            f"Write a user guide for the feature '{feature}', aimed at a '{audience}' audience.\n\n"
            'Return JSON: {"sections": [{"heading": "...", "content": "<real content, several sentences>"}], '
            '"estimated_read_time": "..."}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "feature": feature,
            "audience": audience,
            "sections": result.get("sections", []),
            "format": "markdown",
            "estimated_read_time": result.get("estimated_read_time", "")
        }

    def _code_documentation(self, code: str,
                             style: str = "google") -> Dict:
        user_prompt = (
            f"Write {style}-style docstrings/comments for this code:\n\n{code}\n\n"
            'Return JSON: {"documented_code": "<the code with docstrings added>", '
            '"suggestions": ["..."]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "style": style,
            "documented_functions": code.count("def "),
            "documented_classes": code.count("class "),
            "documented_code": result.get("documented_code", code),
            "suggestions": result.get("suggestions", [])
        }

    def _architecture_docs(self, components: List[Dict],
                            interactions: List[Dict]) -> Dict:
        user_prompt = (
            f"Document this system architecture. Components (JSON): {components}\n"
            f"Interactions (JSON): {interactions}\n\n"
            'Return JSON: {"narrative": "<prose description of the architecture>", '
            '"technologies": ["..."], "decision_log": ["<key design decisions and why>"]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "diagram_type": "C4 Model",
            "components": components,
            "interactions": interactions,
            "views": ["Context", "Container", "Component", "Code"],
            "narrative": result.get("narrative", ""),
            "technologies": result.get("technologies", list({c.get("tech", "") for c in components})),
            "decision_log": result.get("decision_log", [])
        }

    def _changelog(self, commits: List[Dict],
                   version: str = "1.0.0") -> Dict:
        # Categorizing by conventional-commit prefix is deterministic.
        categories = {"feat": [], "fix": [], "docs": [], "refactor": [], "test": [], "chore": []}
        for commit in commits:
            msg = commit.get("message", "")
            for prefix in categories:
                if msg.startswith(prefix):
                    categories[prefix].append(msg)
                    break
        changes = {k: v for k, v in categories.items() if v}
        breaking = [m for m in sum(categories.values(), []) if "BREAKING" in m]

        summary = ""
        if changes:
            user_prompt = (
                f"Write a one-paragraph human-readable summary of this release ({version}) given these "
                f"categorized changes: {changes}\n\n"
                'Return JSON: {"summary": "..."}'
            )
            summary = self._llm_json(SYSTEM_PROMPT, user_prompt).get("summary", "")

        return {
            "version": version,
            "date": "today",
            "changes": changes,
            "summary": summary,
            "breaking_changes": breaking,
            "migration_guide": "See migration.md" if breaking else None
        }
