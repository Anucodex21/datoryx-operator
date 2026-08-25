"""DATORYX DataScientist Agent - Builds statistical models, runs experiments, and derives insights (LLM-backed)"""
from typing import Dict, Any, List

from ..base import BaseAgent, AgentCapability

SYSTEM_PROMPT = (
    "You are a data scientist inside DATORYX. You are given exact, already-computed numeric results and "
    "asked to reason about modeling choices and insights - you never contradict or recompute the numbers "
    "you're given."
)


class DataScientistAgent(BaseAgent):
    """Builds statistical models, runs experiments, and derives insights"""

    def __init__(self, llm: Any = None):
        super().__init__(
            name="DataScientist",
            description="Builds statistical models, runs experiments, and derives insights",
            llm=llm,
        )

    def _register_capabilities(self):
        self.capabilities = [
            AgentCapability("explore_data", "Explore and summarize datasets", {"dataset": "dict"}),
            AgentCapability("hypothesis_test", "Run statistical hypothesis tests", {"group_a": "list", "group_b": "list"}),
            AgentCapability("build_model", "Build predictive models", {"data": "dict", "target": "str"}),
            AgentCapability("evaluate_model", "Evaluate model performance", {"predictions": "list", "actuals": "list"}),
            AgentCapability("generate_insights", "Generate data insights", {"results": "dict"})
        ]

    def _register_tools(self):
        self.tools = {
            "explore_data": self._explore_data,
            "hypothesis_test": self._hypothesis_test,
            "build_model": self._build_model,
            "evaluate_model": self._evaluate_model,
            "generate_insights": self._generate_insights
        }

    def _explore_data(self, dataset: Dict, analysis_type: str = "summary") -> Dict:
        import statistics
        numeric_cols = {k: v for k, v in dataset.items()
                         if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v)}
        summary = {}
        for col, values in numeric_cols.items():
            summary[col] = {
                "count": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0,
                "min": min(values), "max": max(values)
            }

        user_prompt = (
            f"Given this exact computed summary of a dataset: {summary}\n\n"
            'Return JSON: {"observations": ["<notable pattern, e.g. skew, high variance, outlier risk>"]}'
        )
        commentary = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {"analysis_type": analysis_type, "summary": summary, "observations": commentary.get("observations", [])}

    def _hypothesis_test(self, group_a: List[float], group_b: List[float],
                          test_type: str = "t_test") -> Dict:
        import statistics
        mean_a, mean_b = statistics.mean(group_a), statistics.mean(group_b)
        diff = abs(mean_a - mean_b)
        pooled_std = ((statistics.stdev(group_a) ** 2 + statistics.stdev(group_b) ** 2) / 2) ** 0.5 if len(group_a) > 1 and len(group_b) > 1 else 0
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        computed = {
            "test": test_type, "mean_a": mean_a, "mean_b": mean_b,
            "difference": diff, "cohens_d": cohens_d, "significant": cohens_d > 0.5
        }
        user_prompt = f"Interpret this {test_type} result: {computed}\n\nReturn JSON: {{\"interpretation\": \"...\"}}"
        computed["interpretation"] = self._llm_json(SYSTEM_PROMPT, user_prompt).get("interpretation", "")
        return computed

    def _build_model(self, data: Dict, target: str, algorithm: str = "linear") -> Dict:
        user_prompt = (
            f"A '{algorithm}' model is being built to predict '{target}' from features {[k for k in data if k != target]}.\n\n"
            'Return JSON: {"algorithm_description": "...", "recommended_hyperparameters": {...}, '
            '"caveats": ["<data-specific concerns, e.g. multicollinearity, class imbalance>"]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "algorithm": result.get("algorithm_description", algorithm),
            "target": target,
            "features": [k for k in data.keys() if k != target],
            "recommended_hyperparameters": result.get("recommended_hyperparameters", {}),
            "caveats": result.get("caveats", []),
            "status": "designed - not yet fit; connect to models.llm or an ML backend to actually train"
        }

    def _evaluate_model(self, predictions: List, actuals: List) -> Dict:
        n = len(predictions) or 1
        mae = sum(abs(p - a) for p, a in zip(predictions, actuals)) / n
        mse = sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / n
        return {"mae": mae, "mse": mse, "rmse": mse ** 0.5}

    def _generate_insights(self, analysis_results: Dict) -> List[str]:
        user_prompt = (
            f"Generate concrete, specific insights from this analysis (JSON): {analysis_results}\n\n"
            'Return JSON: {"insights": ["<specific, data-grounded insight>"]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return result.get("insights", [])
