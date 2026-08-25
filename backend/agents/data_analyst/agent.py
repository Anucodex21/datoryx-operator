"""DATORYX DataAnalyst Agent - Analyzes data, creates reports, and identifies trends (LLM-backed)"""
from typing import Dict, Any, List

from ..base import BaseAgent, AgentCapability

SYSTEM_PROMPT = (
    "You are a data analyst inside DATORYX. You interpret already-computed numeric results honestly and "
    "write reports grounded in the actual data given, never inventing figures."
)


class DataAnalystAgent(BaseAgent):
    """Analyzes data, creates reports, and identifies trends"""

    def __init__(self, llm: Any = None):
        super().__init__(
            name="DataAnalyst",
            description="Analyzes data, creates reports, and identifies trends",
            llm=llm,
        )

    def _register_capabilities(self):
        self.capabilities = [
            AgentCapability("analyze_trends", "Analyze time series trends", {"time_series": "list"}),
            AgentCapability("segment_data", "Segment data by dimensions", {"data": "list", "segment_by": "str"}),
            AgentCapability("correlation_analysis", "Analyze variable correlations", {"variables": "dict"}),
            AgentCapability("create_report", "Generate analytical reports", {"title": "str", "sections": "list"}),
            AgentCapability("kpi_dashboard", "Build KPI dashboards", {"metrics": "dict", "targets": "dict"})
        ]

    def _register_tools(self):
        self.tools = {
            "analyze_trends": self._analyze_trends,
            "segment_data": self._segment_data,
            "correlation_analysis": self._correlation_analysis,
            "create_report": self._create_report,
            "kpi_dashboard": self._kpi_dashboard
        }

    def _analyze_trends(self, time_series: List[float], period: int = 7) -> Dict:
        if len(time_series) < period:
            return {"error": "Insufficient data"}

        # The moving average and % change are exact math - compute them directly.
        moving_avg = [sum(time_series[i:i + period]) / period for i in range(len(time_series) - period + 1)]
        trend = "increasing" if moving_avg[-1] > moving_avg[0] else "decreasing" if moving_avg[-1] < moving_avg[0] else "stable"
        change_pct = ((time_series[-1] - time_series[0]) / time_series[0] * 100) if time_series[0] != 0 else 0

        user_prompt = (
            f"A time series trend was computed as '{trend}' with a {change_pct:.2f}% change over the "
            f"period. Latest value: {time_series[-1]}.\n\n"
            'Return JSON: {"interpretation": "<1-2 sentence business-relevant interpretation>"}'
        )
        commentary = self._llm_json(SYSTEM_PROMPT, user_prompt)

        return {
            "trend": trend,
            "moving_average": moving_avg,
            "current_value": time_series[-1],
            "change_pct": change_pct,
            "interpretation": commentary.get("interpretation", "")
        }

    def _segment_data(self, data: List[Dict], segment_by: str) -> Dict:
        # Grouping is deterministic - no LLM required.
        segments = {}
        for row in data:
            key = row.get(segment_by, "unknown")
            segments.setdefault(key, []).append(row)
        return {"segments": segments, "count": len(segments)}

    def _correlation_analysis(self, variables: Dict[str, List[float]]) -> Dict:
        import statistics
        corr_matrix = {}
        cols = list(variables.keys())
        for col1 in cols:
            corr_matrix[col1] = {}
            for col2 in cols:
                if col1 == col2:
                    corr_matrix[col1][col2] = 1.0
                else:
                    x, y = variables[col1], variables[col2]
                    mx, my = statistics.mean(x), statistics.mean(y)
                    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
                    den = (sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y)) ** 0.5
                    corr_matrix[col1][col2] = num / den if den != 0 else 0

        strong_pairs = [(a, b, v) for a in cols for b in cols if a < b
                         for v in [corr_matrix[a][b]] if abs(v) > 0.6]
        interpretation = ""
        if strong_pairs:
            user_prompt = (
                f"These variable pairs show strong correlation: {strong_pairs}\n\n"
                'Return JSON: {"interpretation": "<1-2 sentences on what these correlations suggest>"}'
            )
            interpretation = self._llm_json(SYSTEM_PROMPT, user_prompt).get("interpretation", "")

        return {"correlation_matrix": corr_matrix, "interpretation": interpretation}

    def _create_report(self, title: str, sections: List[Dict]) -> Dict:
        user_prompt = (
            f"Write a markdown analytical report titled '{title}' from these sections (JSON, each has a "
            f"heading and supporting content/data): {sections}\n\n"
            "Return the full markdown document as plain text."
        )
        content = self._llm_text(SYSTEM_PROMPT, user_prompt, max_tokens=1600)
        return {
            "title": title,
            "sections": sections,
            "generated_at": "now",
            "format": "markdown",
            "content": content
        }

    def _kpi_dashboard(self, metrics: Dict[str, float], targets: Dict[str, float]) -> Dict:
        # Achievement percentages are exact math.
        dashboard = {}
        for metric, value in metrics.items():
            target = targets.get(metric, value)
            achievement = (value / target * 100) if target != 0 else 0
            dashboard[metric] = {
                "value": value,
                "target": target,
                "achievement_pct": achievement,
                "status": "on_track" if achievement >= 90 else "at_risk" if achievement >= 70 else "critical"
            }
        return {"kpis": dashboard}
