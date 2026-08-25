"""DATORYX Forecasting Agent - Builds time series forecasting models (LLM-backed)"""
from typing import Dict, Any, List

from ..base import BaseAgent, AgentCapability

SYSTEM_PROMPT = (
    "You are a time-series forecasting analyst inside DATORYX. Given a numeric series, you reason about "
    "trend, seasonality, and volatility the way a statistician would, and produce concrete forward "
    "projections with honest uncertainty bounds - not a flat extrapolation dressed up as a forecast."
)


class ForecastingAgent(BaseAgent):
    """Builds time series forecasting models"""

    def __init__(self, llm: Any = None):
        super().__init__(name="Forecasting", description="Builds time series forecasting models", llm=llm)

    def _register_capabilities(self):
        self.capabilities = [
            AgentCapability("arima_forecast", "ARIMA-style time series forecasting", {"series": "list"}),
            AgentCapability("exponential_smoothing", "Exponential smoothing", {"series": "list"}),
            AgentCapability("prophet_style", "Prophet-style forecasting", {"dates": "list", "values": "list"}),
            AgentCapability("ensemble_forecast", "Ensemble forecasting", {"series": "list", "models": "list"}),
            AgentCapability("evaluate_forecast", "Evaluate forecast accuracy", {"actuals": "list", "predictions": "list"})
        ]

    def _register_tools(self):
        self.tools = {
            "arima_forecast": self._arima_forecast,
            "exponential_smoothing": self._exponential_smoothing,
            "prophet_style": self._prophet_style,
            "ensemble_forecast": self._ensemble_forecast,
            "evaluate_forecast": self._evaluate_forecast
        }

    def _arima_forecast(self, series: List[float], p: int = 1, d: int = 1,
                         q: int = 1, horizon: int = 10) -> Dict:
        user_prompt = (
            f"Fit an ARIMA({p},{d}{q}) style forecast to this series: {series}\n"
            f"Produce {horizon} forward points with 90% confidence bounds.\n\n"
            'Return JSON: {"forecast": [<numbers>], "lower": [<numbers>], "upper": [<numbers>], '
            '"reasoning": "<trend/seasonality observations>"}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        forecast = result.get("forecast", [])
        return {
            "model": f"ARIMA({p},{d},{q})",
            "forecast": forecast,
            "horizon": horizon,
            "confidence_intervals": {
                "lower": result.get("lower", [f * 0.9 for f in forecast]),
                "upper": result.get("upper", [f * 1.1 for f in forecast])
            },
            "reasoning": result.get("reasoning", "")
        }

    def _exponential_smoothing(self, series: List[float], alpha: float = 0.3,
                                horizon: int = 10) -> Dict:
        # Smoothing itself is a well-defined deterministic recurrence - keep it exact.
        smoothed = [series[0]]
        for val in series[1:]:
            smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])

        user_prompt = (
            f"An exponentially smoothed series (alpha={alpha}) ends at {smoothed[-1]:.4f}, having started "
            f"at {smoothed[0]:.4f} across {len(smoothed)} points. Raw series: {series}\n"
            f"Project {horizon} points forward, accounting for any trend the smoothed series still shows "
            "(plain exponential smoothing has no trend component, so correct for that).\n\n"
            'Return JSON: {"forecast": [<numbers>], "trend_detected": true/false, "trend_direction": "up|down|flat"}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "alpha": alpha,
            "smoothed_values": smoothed,
            "forecast": result.get("forecast", [smoothed[-1]] * horizon),
            "horizon": horizon,
            "trend_detected": result.get("trend_detected", smoothed[-1] > smoothed[0])
        }

    def _prophet_style(self, dates: List[str], values: List[float],
                        horizon: int = 30) -> Dict:
        user_prompt = (
            f"Produce a Prophet-style decomposition (trend + seasonality) forecast for {horizon} periods "
            f"beyond this series.\nDates: {dates}\nValues: {values}\n\n"
            'Return JSON: {"components": ["..."], "changepoints": <int>, '
            '"forecast": {"dates": ["..."], "values": [<numbers>]}, "holidays": ["..."], '
            '"interval_width": <float>}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "components": result.get("components", ["trend"]),
            "changepoints": result.get("changepoints", 0),
            "forecast": result.get("forecast", {"dates": [], "values": []}),
            "holidays": result.get("holidays", []),
            "interval_width": result.get("interval_width", 0.8)
        }

    def _ensemble_forecast(self, series: List[float], models: List[str],
                            horizon: int = 10) -> Dict:
        forecasts = {}
        if "arima" in models:
            forecasts["arima"] = self._arima_forecast(series, horizon=horizon)["forecast"]
        if "exp_smooth" in models:
            forecasts["exp_smooth"] = self._exponential_smoothing(series, horizon=horizon)["forecast"]

        ensemble = []
        for i in range(horizon):
            vals = [forecasts[m][i] for m in forecasts if i < len(forecasts[m])]
            ensemble.append(sum(vals) / len(vals) if vals else 0)

        return {
            "individual_forecasts": forecasts,
            "ensemble": ensemble,
            "weights": {m: 1 / len(forecasts) for m in forecasts} if forecasts else {},
            "horizon": horizon
        }

    def _evaluate_forecast(self, actuals: List[float],
                            predictions: List[float]) -> Dict:
        # Accuracy metrics are exact math - no LLM needed here.
        n = len(actuals) or 1
        mae = sum(abs(a - p) for a, p in zip(actuals, predictions)) / n
        mse = sum((a - p) ** 2 for a, p in zip(actuals, predictions)) / n
        rmse = mse ** 0.5
        nonzero = [(a, p) for a, p in zip(actuals, predictions) if a != 0]
        mape = (sum(abs((a - p) / a) for a, p in nonzero) / len(nonzero) * 100) if nonzero else 0
        return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape}
