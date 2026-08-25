"""DATORYX Visualization Agent - Creates charts, dashboards, and data visualizations (LLM-backed)"""
from typing import Dict, Any, List

from ..base import BaseAgent, AgentCapability

SYSTEM_PROMPT = (
    "You are a data visualization designer inside DATORYX. You know Plotly/D3 conventions and data-viz "
    "best practices, and you give concrete, chart-specific configuration and advice - not generic tips."
)


class VisualizationAgent(BaseAgent):
    """Creates charts, dashboards, and data visualizations"""

    def __init__(self, llm: Any = None):
        super().__init__(name="Visualization", description="Creates charts, dashboards, and data visualizations", llm=llm)

    def _register_capabilities(self):
        self.capabilities = [
            AgentCapability("create_chart", "Create data visualizations", {"data": "list", "chart_type": "str"}),
            AgentCapability("create_dashboard", "Build interactive dashboards", {"widgets": "list"}),
            AgentCapability("recommend_viz", "Recommend visualization types", {"data_type": "str", "goal": "str"}),
            AgentCapability("generate_plotly", "Generate Plotly configurations", {"data": "list"}),
            AgentCapability("style_guide", "Create visualization style guides", {"brand_colors": "list"})
        ]

    def _register_tools(self):
        self.tools = {
            "create_chart": self._create_chart,
            "create_dashboard": self._create_dashboard,
            "recommend_viz": self._recommend_viz,
            "generate_plotly": self._generate_plotly,
            "style_guide": self._style_guide
        }

    def _create_chart(self, data: List[Dict], chart_type: str,
                       x_col: str, y_col: str) -> Dict:
        user_prompt = (
            f"Configure a '{chart_type}' chart plotting '{x_col}' vs '{y_col}'. Sample data (first rows): "
            f"{data[:5]}\n\n"
            'Return JSON: {"config": {<chart-library config appropriate to this chart type>}, '
            '"recommendations": ["<specific, non-generic tips for this data>"]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "chart_type": chart_type,
            "config": result.get("config", {"type": chart_type}),
            "data_preview": data[:5],
            "x": x_col,
            "y": y_col,
            "recommendations": result.get("recommendations", [])
        }

    def _create_dashboard(self, widgets: List[Dict], layout: str = "grid") -> Dict:
        user_prompt = (
            f"Plan a {layout}-layout dashboard containing these widgets (JSON): {widgets}\n\n"
            'Return JSON: {"theme": "...", "refresh_interval_seconds": <int>, "export_formats": ["..."], '
            '"layout_notes": "..."}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "layout": layout,
            "widgets": widgets,
            "responsive": True,
            "theme": result.get("theme", "light"),
            "refresh_interval": result.get("refresh_interval_seconds", 30),
            "export_formats": result.get("export_formats", ["png", "pdf", "html"]),
            "layout_notes": result.get("layout_notes", "")
        }

    def _recommend_viz(self, data_type: str, goal: str) -> Dict:
        user_prompt = (
            f"Recommend chart types for data of type '{data_type}' with the analytical goal '{goal}'.\n\n"
            'Return JSON: {"recommended_charts": ["..."], "best_practices": ["<specific to this case>"]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "data_type": data_type,
            "goal": goal,
            "recommended_charts": result.get("recommended_charts", []),
            "best_practices": result.get("best_practices", [])
        }

    def _generate_plotly(self, data: List[Dict], chart_type: str) -> Dict:
        user_prompt = (
            f"Generate a Plotly layout configuration for a '{chart_type}' chart of this data (sample): "
            f"{data[:5]}\n\n"
            'Return JSON: {"layout": {"title": {"text": "..."}, "xaxis": {"title": "..."}, '
            '"yaxis": {"title": "..."}, "template": "..."}, "hover_info": "..."}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "library": "plotly",
            "chart_type": chart_type,
            "data": data,
            "layout": result.get("layout", {"template": "plotly_white"}),
            "interactive": True,
            "hover_info": result.get("hover_info", "all")
        }

    def _style_guide(self, brand_colors: List[str] = None) -> Dict:
        colors = brand_colors or ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        user_prompt = (
            f"Build a data-visualization style guide around this palette: {colors}\n\n"
            'Return JSON: {"colors": {"primary": "...", "secondary": "...", "accent": "...", '
            '"error": "...", "success": "..."}, "typography": {"title_font": "...", "axis_font": "...", '
            '"annotation_font": "..."}, "accessibility": {"colorblind_friendly": true/false, '
            '"contrast_ratio": <float>, "alt_text_required": true/false}}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "colors": result.get("colors", {"primary": colors[0]}),
            "typography": result.get("typography", {}),
            "accessibility": result.get("accessibility", {"colorblind_friendly": True, "contrast_ratio": 4.5, "alt_text_required": True})
        }
