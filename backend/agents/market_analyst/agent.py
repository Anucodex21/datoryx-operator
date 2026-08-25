"""DATORYX Market Analyst Agent.

The 21st agent in the fleet - brought in from the separate AI-Market-
Analyzer project. Unlike most agents here, its structured capabilities do
real numeric computation (pandas-based technical indicators, not just an
LLM call) and then use the LLM to synthesize the numbers into a readable
read, the same news_ai/chart_ai/market_ai split AI-Market-Analyzer used
across its phases, condensed into one agent.

Accepts OHLCV data directly as a list of bar dicts (the caller's
responsibility to supply - e.g. from yfinance, a broker API, or a CSV) so
this agent has zero required network access and stays testable offline.
If `yfinance` is installed, `fetch_symbol` will also pull it live.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from agents.base import AgentCapability, BaseAgent
from market_data.indicators import momentum, trend, volatility


class MarketAnalystAgent(BaseAgent):
    """Technical + narrative market analysis: real indicator math, LLM-read summary."""

    def __init__(self, llm: Any = None):
        super().__init__(
            name="Market Analyst",
            description="Computes real technical indicators (RSI, MACD, SMA/EMA, ADX, volatility) "
                        "from OHLCV data and synthesizes them into a plain-English read.",
            llm=llm,
        )

    def _register_capabilities(self):
        self.capabilities = [
            AgentCapability(
                "analyze_ohlcv",
                "Compute technical indicators from OHLCV bars and summarize the technical picture",
                {"bars": "list", "symbol": "str"},
            ),
            AgentCapability(
                "fetch_symbol",
                "Fetch recent OHLCV data for a ticker (requires yfinance) and analyze it",
                {"symbol": "str", "period": "str"},
            ),
            AgentCapability(
                "market_regime",
                "Classify overall market regime (trending/ranging/volatile) from an index's OHLCV data",
                {"bars": "list"},
            ),
        ]

    def _register_tools(self):
        self.tools = {
            "analyze_ohlcv": self._analyze_ohlcv,
            "fetch_symbol": self._fetch_symbol,
            "market_regime": self._market_regime,
        }

    # ------------------------------------------------------------------

    def _bars_to_df(self, bars: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(bars)
        rename = {c: c.title() for c in df.columns if c.lower() in ("open", "high", "low", "close", "volume")}
        df = df.rename(columns=rename)
        required = {"Open", "High", "Low", "Close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"OHLCV bars missing required columns: {missing}")
        return df

    def _compute_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        close = df["Close"]
        out = {
            "last_close": round(float(close.iloc[-1]), 4),
            "sma_20": round(float(trend.sma(close, 20).iloc[-1]), 4) if len(close) >= 20 else None,
            "sma_50": round(float(trend.sma(close, 50).iloc[-1]), 4) if len(close) >= 50 else None,
            "rsi_14": round(float(momentum.rsi(close).iloc[-1]), 2) if len(close) >= 15 else None,
        }
        if len(close) >= 35:
            macd_df = momentum.macd(close)
            out["macd"] = round(float(macd_df["macd"].iloc[-1]), 4)
            out["macd_signal"] = round(float(macd_df["macd_signal"].iloc[-1]), 4)
            out["macd_hist"] = round(float(macd_df["macd_hist"].iloc[-1]), 4)
        try:
            adx_df = trend.adx(df)
            out["adx_14"] = round(float(adx_df.iloc[-1]), 2) if hasattr(adx_df, "iloc") else None
        except Exception:
            out["adx_14"] = None
        return out

    def _analyze_ohlcv(self, bars: List[Dict[str, Any]], symbol: str = "") -> Dict[str, Any]:
        df = self._bars_to_df(bars)
        indicators = self._compute_indicators(df)

        system_prompt = (
            "You are a market analyst. Given real computed technical indicators for a security, "
            "give a concise (3-5 sentence) plain-English read: trend direction, momentum state, "
            "and one thing to watch. Do not invent numbers not given to you. Do not give financial "
            "advice or a buy/sell recommendation - describe the technical picture only."
        )
        user_prompt = f"Symbol: {symbol or 'unspecified'}\nIndicators: {indicators}"
        summary = self._llm_text(system_prompt, user_prompt, temperature=0.3, max_tokens=400)

        return {"symbol": symbol, "indicators": indicators, "summary": summary, "bars_used": len(df)}

    def _fetch_symbol(self, symbol: str, period: str = "6mo") -> Dict[str, Any]:
        try:
            import yfinance as yf
        except ImportError:
            return {
                "error": "yfinance is not installed. Install it (`pip install yfinance`) or call "
                         "analyze_ohlcv directly with your own OHLCV bars instead.",
                "symbol": symbol,
            }
        data = yf.download(symbol, period=period, progress=False)
        if data.empty:
            return {"error": f"No data returned for '{symbol}'", "symbol": symbol}
        bars = data.reset_index().to_dict("records")
        return self._analyze_ohlcv(bars, symbol=symbol)

    def _market_regime(self, bars: List[Dict[str, Any]]) -> Dict[str, Any]:
        df = self._bars_to_df(bars)
        close = df["Close"]
        vol_df = volatility.add_volatility_features(df) if hasattr(volatility, "add_volatility_features") else df
        realized_vol = float(close.pct_change().rolling(20).std().iloc[-1] * (252 ** 0.5)) if len(close) >= 21 else None
        adx_series = trend.adx(df)
        adx_val = float(adx_series.iloc[-1]) if hasattr(adx_series, "iloc") and len(df) >= 15 else None

        if adx_val is not None and adx_val >= 25:
            regime = "trending"
        elif realized_vol is not None and realized_vol > 0.35:
            regime = "volatile"
        else:
            regime = "ranging"

        return {
            "regime": regime,
            "adx_14": round(adx_val, 2) if adx_val is not None else None,
            "annualized_realized_vol": round(realized_vol, 4) if realized_vol is not None else None,
        }
