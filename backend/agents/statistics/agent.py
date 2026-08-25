"""DATORYX Statistics Agent - Performs statistical analysis, tests, and inference (LLM-backed)"""
from typing import Dict, Any, List

from ..base import BaseAgent, AgentCapability

SYSTEM_PROMPT = (
    "You are a statistician inside DATORYX. You are given exact, already-computed statistical results "
    "and asked to interpret them plainly and correctly - you never recompute or contradict the numbers "
    "you're given, only explain what they mean."
)


class StatisticsAgent(BaseAgent):
    """Performs statistical analysis, tests, and inference"""

    def __init__(self, llm: Any = None):
        super().__init__(name="Statistics", description="Performs statistical analysis, tests, and inference", llm=llm)

    def _register_capabilities(self):
        self.capabilities = [
            AgentCapability("descriptive_stats", "Calculate descriptive statistics", {"data": "list"}),
            AgentCapability("inferential_stats", "Run inferential tests", {"sample": "list", "population_mean": "float"}),
            AgentCapability("regression", "Perform regression analysis", {"x": "list", "y": "list"}),
            AgentCapability("anova", "Run ANOVA tests", {"groups": "dict"}),
            AgentCapability("confidence_interval", "Calculate confidence intervals", {"data": "list"})
        ]

    def _register_tools(self):
        self.tools = {
            "descriptive_stats": self._descriptive_stats,
            "inferential_stats": self._inferential_stats,
            "regression": self._regression,
            "anova": self._anova,
            "confidence_interval": self._confidence_interval
        }

    # All numeric results below are computed with exact formulas - never delegated to the
    # LLM, which should not be trusted to do arithmetic. The LLM is used only to add a
    # plain-language interpretation on top of numbers that are already correct.

    def _interpret(self, computed: Dict[str, Any], context: str) -> str:
        user_prompt = (
            f"These statistical results were computed exactly: {computed}\n"
            f"Context: {context}\n\n"
            'Return JSON: {"interpretation": "<1-3 plain-language sentences explaining what this means>"}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return result.get("interpretation", "")

    def _descriptive_stats(self, data: List[float]) -> Dict:
        import statistics
        computed = {
            "n": len(data),
            "mean": statistics.mean(data),
            "median": statistics.median(data),
            "mode": statistics.multimode(data) if hasattr(statistics, "multimode") else (
                statistics.mode(data) if len(set(data)) < len(data) else None),
            "std": statistics.stdev(data) if len(data) > 1 else 0,
            "variance": statistics.variance(data) if len(data) > 1 else 0,
            "min": min(data),
            "max": max(data),
            "range": max(data) - min(data),
            "quartiles": {
                "q1": sorted(data)[len(data) // 4],
                "q3": sorted(data)[3 * len(data) // 4]
            }
        }
        computed["interpretation"] = self._interpret(computed, "descriptive statistics of a dataset")
        return computed

    def _inferential_stats(self, sample: List[float], population_mean: float) -> Dict:
        import statistics
        n = len(sample)
        sample_mean = statistics.mean(sample)
        sample_std = statistics.stdev(sample) if n > 1 else 0
        se = sample_std / (n ** 0.5) if n > 0 else 0
        t_stat = (sample_mean - population_mean) / se if se > 0 else 0
        computed = {
            "sample_mean": sample_mean,
            "population_mean": population_mean,
            "t_statistic": t_stat,
            "standard_error": se,
            "sample_size": n,
            "significant_at_05": abs(t_stat) > 1.96
        }
        computed["interpretation"] = self._interpret(computed, "one-sample t-test against a population mean")
        return computed

    def _regression(self, x: List[float], y: List[float]) -> Dict:
        import statistics
        n = len(x)
        mx, my = statistics.mean(x), statistics.mean(y)
        ss_xy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        ss_xx = sum((xi - mx) ** 2 for xi in x)
        slope = ss_xy / ss_xx if ss_xx != 0 else 0
        intercept = my - slope * mx
        y_pred = [slope * xi + intercept for xi in x]
        ss_res = sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred))
        ss_tot = sum((yi - my) ** 2 for yi in y)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        computed = {
            "slope": slope,
            "intercept": intercept,
            "equation": f"y = {slope:.4f}x + {intercept:.4f}",
            "r_squared": r2,
            "predictions": y_pred
        }
        interp = self._interpret(
            {k: v for k, v in computed.items() if k != "predictions"},
            "simple linear regression fit"
        )
        computed["interpretation"] = interp
        return computed

    def _anova(self, groups: Dict[str, List[float]]) -> Dict:
        import statistics
        all_data = [v for g in groups.values() for v in g]
        grand_mean = statistics.mean(all_data)
        ss_between = sum(len(g) * (statistics.mean(g) - grand_mean) ** 2 for g in groups.values())
        ss_within = sum(sum((v - statistics.mean(g)) ** 2 for v in g) for g in groups.values())
        df_between = len(groups) - 1
        df_within = len(all_data) - len(groups)
        ms_between = ss_between / df_between if df_between > 0 else 0
        ms_within = ss_within / df_within if df_within > 0 else 0
        f_stat = ms_between / ms_within if ms_within > 0 else 0
        computed = {
            "f_statistic": f_stat,
            "ss_between": ss_between,
            "ss_within": ss_within,
            "groups": list(groups.keys()),
            "significant": f_stat > 3.0
        }
        computed["interpretation"] = self._interpret(computed, "one-way ANOVA across groups")
        return computed

    def _confidence_interval(self, data: List[float], confidence: float = 0.95) -> Dict:
        import statistics
        n = len(data)
        mean = statistics.mean(data)
        std = statistics.stdev(data) if n > 1 else 0
        z = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
        margin = z * (std / (n ** 0.5)) if n > 0 else 0
        computed = {
            "mean": mean,
            "confidence_level": confidence,
            "lower_bound": mean - margin,
            "upper_bound": mean + margin,
            "margin_of_error": margin
        }
        computed["interpretation"] = self._interpret(computed, "confidence interval around a sample mean")
        return computed
