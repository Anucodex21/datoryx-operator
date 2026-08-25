"""Momentum indicators: RSI, MACD, Stochastic oscillator."""
from __future__ import annotations

import pandas as pd

from market_data.indicators.trend import ema


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})


def stochastic(df: pd.DataFrame, window: int = 14, smooth_k: int = 3, smooth_d: int = 3,
                high_col: str = "High", low_col: str = "Low", close_col: str = "Close") -> pd.DataFrame:
    lowest_low = df[low_col].rolling(window).min()
    highest_high = df[high_col].rolling(window).max()
    raw_k = 100 * (df[close_col] - lowest_low) / (highest_high - lowest_low).replace(0, 1e-10)
    k = raw_k.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


def add_momentum_features(df: pd.DataFrame, close_col: str = "Close") -> pd.DataFrame:
    out = df.copy()
    out["rsi_14"] = rsi(out[close_col])
    macd_df = macd(out[close_col])
    out = out.join(macd_df)
    out["macd_positive"] = (out["macd"] > 0).astype(int)
    stoch_df = stochastic(out)
    out = out.join(stoch_df)
    return out
