"""DATORYX BusinessIntelligence Agent - Creates BI solutions, metrics, and strategic insights (LLM-backed)"""
from typing import Dict, Any, List

from ..base import BaseAgent, AgentCapability

SYSTEM_PROMPT = (
    "You are a business intelligence strategist inside DATORYX. You give concrete, situation-specific "
    "analysis grounded in what's actually given - never generic placeholders like 'Strong brand' unless "
    "the data actually supports it. Where you don't have real market data, say so rather than inventing "
    "specific figures (market size, market share percentages, etc.)."
)


class BusinessIntelligenceAgent(BaseAgent):
    """Creates BI solutions, metrics, and strategic insights"""

    def __init__(self, llm: Any = None):
        super().__init__(name="BusinessIntelligence", description="Creates BI solutions, metrics, and strategic insights", llm=llm)

    def _register_capabilities(self):
        self.capabilities = [
            AgentCapability("define_kpis", "Define business KPIs", {"objectives": "list"}),
            AgentCapability("build_scorecard", "Build balanced scorecards", {"kpis": "list"}),
            AgentCapability("swot_analysis", "Perform SWOT analysis", {"company_data": "dict"}),
            AgentCapability("market_analysis", "Analyze market landscape", {"industry": "str"}),
            AgentCapability("competitive_intelligence", "Analyze competitors", {"competitor": "str"})
        ]

    def _register_tools(self):
        self.tools = {
            "define_kpis": self._define_kpis,
            "build_scorecard": self._build_scorecard,
            "swot_analysis": self._swot_analysis,
            "market_analysis": self._market_analysis,
            "competitive_intelligence": self._competitive_intelligence
        }

    def _define_kpis(self, business_objectives: List[str],
                      department: str = "all") -> Dict:
        user_prompt = (
            f"Define concrete KPIs for department '{department}' given these business objectives: "
            f"{business_objectives}\n\n"
            'Return JSON: {"kpis": [{"name": "...", "formula": "...", "target": <number or string>, '
            '"frequency": "..."}]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        kpis = result.get("kpis", [])
        return {"department": department, "kpis": kpis, "total_kpis": len(kpis)}

    def _build_scorecard(self, kpis: List[Dict], period: str = "Q3 2026") -> Dict:
        # Achievement % per KPI is exact math given actual/target; only the qualitative
        # status thresholds are fixed conventions, not something to hallucinate.
        scorecard = {"period": period, "categories": []}
        total_score = 0
        for kpi in kpis:
            actual = kpi.get("actual", 0)
            target = kpi.get("target", 1)
            achievement = (actual / target * 100) if target != 0 else 0
            score = min(100, achievement)
            total_score += score
            scorecard["categories"].append({
                "kpi": kpi.get("name", "unnamed"),
                "target": target,
                "actual": actual,
                "achievement": f"{score:.1f}%",
                "status": "green" if score >= 90 else "yellow" if score >= 70 else "red"
            })
        scorecard["overall_score"] = total_score / len(kpis) if kpis else 0
        return scorecard

    def _swot_analysis(self, company_data: Dict) -> Dict:
        user_prompt = (
            f"Perform a SWOT analysis using this company data (JSON): {company_data}\n"
            "Only use what's given or can be reasonably inferred from it - don't invent specifics that "
            "aren't grounded in the input.\n\n"
            'Return JSON: {"strengths": ["..."], "weaknesses": ["..."], "opportunities": ["..."], '
            '"threats": ["..."], "strategic_recommendations": ["..."]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "strengths": result.get("strengths", company_data.get("strengths", [])),
            "weaknesses": result.get("weaknesses", company_data.get("weaknesses", [])),
            "opportunities": result.get("opportunities", company_data.get("opportunities", [])),
            "threats": result.get("threats", company_data.get("threats", [])),
            "strategic_recommendations": result.get("strategic_recommendations", [])
        }

    def _market_analysis(self, industry: str, competitors: List[str]) -> Dict:
        user_prompt = (
            f"Analyze the market landscape for the '{industry}' industry, with these known competitors: "
            f"{competitors}\n"
            "Do not invent specific market-size dollar figures or market-share percentages you aren't "
            "confident about - describe qualitatively instead where precise numbers aren't known.\n\n"
            'Return JSON: {"market_characterization": "...", "key_players": ["..."], "trends": ["..."], '
            '"barriers_to_entry": ["..."]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "industry": industry,
            "market_characterization": result.get("market_characterization", ""),
            "key_players": result.get("key_players", competitors),
            "trends": result.get("trends", []),
            "barriers_to_entry": result.get("barriers_to_entry", [])
        }

    def _competitive_intelligence(self, competitor: str,
                                   data_points: List[str]) -> Dict:
        user_prompt = (
            f"Analyze competitor '{competitor}' with respect to these aspects: {data_points}\n"
            "Base this on general, well-known public information about the company if you recognize it; "
            "if you don't have reliable information, say so rather than inventing specifics.\n\n"
            'Return JSON: {"strengths": ["..."], "weaknesses": ["..."], "recent_moves": ["..."], '
            '"threat_level": "low|medium|high", "recommended_response": "...", '
            '"confidence": "high|low - do we actually know this company"}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "competitor": competitor,
            "analyzed_aspects": data_points,
            "strengths": result.get("strengths", []),
            "weaknesses": result.get("weaknesses", []),
            "recent_moves": result.get("recent_moves", []),
            "threat_level": result.get("threat_level", "unknown"),
            "recommended_response": result.get("recommended_response", ""),
            "confidence": result.get("confidence", "low")
        }
