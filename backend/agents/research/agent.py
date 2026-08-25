"""DATORYX Research Agent - Conducts research, literature reviews, and knowledge synthesis (LLM-backed)"""
from typing import Dict, Any, List

from ..base import BaseAgent, AgentCapability

SYSTEM_PROMPT = (
    "You are a research analyst inside DATORYX. You draw on general domain knowledge to synthesize "
    "topics, but you are honest about the limits of that knowledge: you do not invent specific paper "
    "titles, authors, or citation counts you aren't confident about. When asked for citable sources, "
    "prefer describing the type of source needed over fabricating a specific one."
)


class ResearchAgent(BaseAgent):
    """Conducts research, literature reviews, and knowledge synthesis"""

    def __init__(self, llm: Any = None):
        super().__init__(name="Research", description="Conducts research, literature reviews, and knowledge synthesis", llm=llm)

    def _register_capabilities(self):
        self.capabilities = [
            AgentCapability("literature_review", "Conduct literature reviews", {"topic": "str"}),
            AgentCapability("synthesize_findings", "Synthesize research findings", {"findings": "list"}),
            AgentCapability("gap_analysis", "Identify research gaps", {"current": "dict", "desired": "dict"}),
            AgentCapability("trend_research", "Research domain trends", {"domain": "str"}),
            AgentCapability("cite_sources", "Format citations", {"sources": "list"})
        ]

    def _register_tools(self):
        self.tools = {
            "literature_review": self._literature_review,
            "synthesize_findings": self._synthesize_findings,
            "gap_analysis": self._gap_analysis,
            "trend_research": self._trend_research,
            "cite_sources": self._cite_sources
        }

    def _literature_review(self, topic: str, sources: int = 10) -> Dict:
        user_prompt = (
            f"Write a literature-review-style synthesis on: {topic}\n"
            f"Frame it as if surveying roughly {sources} sources, but do not invent specific paper titles, "
            "authors, or citation counts - describe the state of knowledge, major schools of thought, and "
            "open questions in prose instead.\n\n"
            'Return JSON: {"themes": ["..."], "synthesis": "<2-4 paragraph synthesis>", '
            '"open_questions": ["..."]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "topic": topic,
            "sources_reviewed": sources,
            "themes": result.get("themes", []),
            "synthesis": result.get("synthesis", ""),
            "open_questions": result.get("open_questions", [])
        }

    def _synthesize_findings(self, findings: List[Dict]) -> Dict:
        user_prompt = (
            f"Synthesize these research findings, grouped by theme (JSON): {findings}\n\n"
            'Return JSON: {"themes": ["..."], "consensus_areas": ["..."], "conflict_areas": ["..."], '
            '"overall_confidence": <0-1 float>}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "themes": result.get("themes", []),
            "consensus_areas": result.get("consensus_areas", []),
            "conflict_areas": result.get("conflict_areas", []),
            "overall_confidence": result.get("overall_confidence", 0.5)
        }

    def _gap_analysis(self, current_state: Dict,
                       desired_state: Dict) -> Dict:
        # Numeric gap sizing is deterministic; the LLM adds prioritization judgment.
        gaps = []
        for key in desired_state:
            current = current_state.get(key, 0)
            desired = desired_state.get(key, 0)
            if current < desired:
                gaps.append({"area": key, "current": current, "desired": desired, "gap": desired - current})

        if not gaps:
            return {"gaps": [], "total_gaps": 0, "critical_gaps": 0}

        user_prompt = (
            f"Prioritize these gaps between current and desired state: {gaps}\n\n"
            'Return JSON: {"gaps": [{"area": "...", "priority": "high|medium|low", "rationale": "..."}]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        prioritized = result.get("gaps", gaps)
        return {
            "gaps": prioritized,
            "total_gaps": len(prioritized),
            "critical_gaps": len([g for g in prioritized if g.get("priority") == "high"])
        }

    def _trend_research(self, domain: str, timeframe: str = "last_5_years") -> Dict:
        user_prompt = (
            f"Analyze trends in the domain of '{domain}' over the {timeframe}.\n\n"
            'Return JSON: {"emerging_trends": ["..."], "maturing_trends": ["..."], '
            '"declining_trends": ["..."], "outlook": "<1-2 sentence forward-looking assessment>"}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "domain": domain,
            "timeframe": timeframe,
            "emerging_trends": result.get("emerging_trends", []),
            "maturing_trends": result.get("maturing_trends", []),
            "declining_trends": result.get("declining_trends", []),
            "outlook": result.get("outlook", "")
        }

    def _cite_sources(self, sources: List[Dict],
                       style: str = "APA") -> List[str]:
        user_prompt = (
            f"Format these sources (JSON) in {style} citation style: {sources}\n\n"
            'Return JSON: {"citations": ["<formatted citation string>", ...]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return result.get("citations", [])
