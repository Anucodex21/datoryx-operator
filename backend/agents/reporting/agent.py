"""DATORYX Reporting Agent - Generates automated reports and narratives from data (LLM-backed)"""
from datetime import datetime, timezone
from typing import Dict, Any, List

from ..base import BaseAgent, AgentCapability

SYSTEM_PROMPT = (
    "You are a business reporting analyst inside DATORYX. You write clear, data-grounded prose - "
    "no filler, no generic boilerplate. Every claim you make should trace back to the numbers you were given."
)


class ReportingAgent(BaseAgent):
    """Generates automated reports and narratives from data"""

    def __init__(self, llm: Any = None):
        super().__init__(name="Reporting", description="Generates automated reports and narratives from data", llm=llm)

    def _register_capabilities(self):
        self.capabilities = [
            AgentCapability("generate_report", "Generate structured reports", {"title": "str", "data": "dict"}),
            AgentCapability("narrative_insights", "Generate narrative insights", {"metrics": "dict"}),
            AgentCapability("automated_alerts", "Create threshold alerts", {"thresholds": "dict", "values": "dict"}),
            AgentCapability("executive_summary", "Create executive summaries", {"report_data": "dict"}),
            AgentCapability("scheduled_reports", "Schedule automated reports", {"config": "dict"})
        ]

    def _register_tools(self):
        self.tools = {
            "generate_report": self._generate_report,
            "narrative_insights": self._narrative_insights,
            "automated_alerts": self._automated_alerts,
            "executive_summary": self._executive_summary,
            "scheduled_reports": self._scheduled_reports
        }

    def _generate_report(self, title: str, data: Dict,
                          sections: List[str]) -> Dict:
        user_prompt = (
            f"Write a markdown report titled '{title}' with these sections, in order: {sections}.\n"
            f"Base every section on this data (JSON): {data}\n\n"
            "Write real content grounded in the data - not placeholder headings. If a section has no "
            "supporting data, say what's missing rather than inventing numbers.\n\n"
            "Return the full markdown document as plain text (no JSON wrapping)."
        )
        content = self._llm_text(SYSTEM_PROMPT, user_prompt, max_tokens=1800)
        return {
            "title": title,
            "format": "markdown",
            "content": content,
            "word_count": len(content.split()),
            "sections": sections,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

    def _narrative_insights(self, metrics: Dict[str, float],
                             context: str = "monthly") -> List[str]:
        user_prompt = (
            f"Write one narrative sentence per metric interpreting performance, for a {context} review.\n"
            f"Metrics (name -> value, values are ratios where 1.0 = 100%): {metrics}\n\n"
            'Return JSON: {"narratives": ["<one sentence per metric>"]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return result.get("narratives", [])

    def _automated_alerts(self, thresholds: Dict[str, tuple],
                           current_values: Dict[str, float]) -> List[Dict]:
        # Threshold breach detection is deterministic - no LLM needed to compare numbers.
        alerts = []
        for metric, (min_val, max_val) in thresholds.items():
            current = current_values.get(metric, 0)
            if current < min_val:
                alerts.append({"severity": "critical", "metric": metric,
                                "message": f"{metric} below threshold: {current} < {min_val}"})
            elif current > max_val:
                alerts.append({"severity": "warning", "metric": metric,
                                "message": f"{metric} above threshold: {current} > {max_val}"})

        if not alerts:
            return alerts

        # Let the LLM add a one-line human-readable summary of what's happening across the alerts.
        user_prompt = (
            f"These metric alerts just fired: {alerts}\n\n"
            'Return JSON: {"summary": "<one sentence overall assessment>", "recommended_action": "<one sentence>"}'
        )
        commentary = self._llm_json(SYSTEM_PROMPT, user_prompt)
        for alert in alerts:
            alert["overall_summary"] = commentary.get("summary", "")
            alert["recommended_action"] = commentary.get("recommended_action", "")
        return alerts

    def _executive_summary(self, report_data: Dict,
                            max_length: int = 500) -> Dict:
        user_prompt = (
            f"Write an executive summary (target under {max_length} characters) from this report data "
            f"(JSON): {report_data}\n\n"
            'Return JSON: {"summary": "<the summary text>"}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        summary = result.get("summary", result.get("raw_response", "")).strip()
        return {
            "summary": summary,
            "length": len(summary),
            "max_length": max_length,
            "audience": "executive"
        }

    def _scheduled_reports(self, report_config: Dict) -> Dict:
        return {
            "report_name": report_config.get("name", "Default Report"),
            "frequency": report_config.get("frequency", "daily"),
            "recipients": report_config.get("recipients", []),
            "format": report_config.get("format", "pdf"),
            "delivery_method": report_config.get("delivery", "email"),
            "next_run": "scheduled",
            "templates": report_config.get("templates", ["standard"])
        }
