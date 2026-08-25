"""Trend indicators: SMA/EMA (20/50/200 & 9/21/50) and ADX."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def add_moving_averages(df: pd.DataFrame, close_col: str = "Close") -> pd.DataFrame:
    out = df.copy()
    for w in (20, 50, 200):
        out[f"sma_{w}"] = sma(out[close_col], w)
    for s in (9, 21, 50):
        out[f"ema_{s}"] = ema(out[close_col], s)

    # Common derived signals: price relative to moving averages
    out["close_above_sma50"] = (out[close_col] > out["sma_50"]).astype(int)
    out["close_above_sma200"] = (out[close_col] > out["sma_200"]).astype(int)
    out["golden_cross"] = (out["sma_50"] > out["sma_200"]).astype(int)
    return out


def adx(df: pd.DataFrame, window: int = 14,
        high_col: str = "High", low_col: str = "Low", close_col: str = "Close") -> pd.Series:
    """Average Directional Index — trend strength (not direction)."""
    high, low, close = df[high_col], df[low_col], df[close_col]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / window, adjust=False, min_periods=window
    ).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / window, adjust=False, min_periods=window
    ).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_moving_averages(df)
    out["adx_14"] = adx(out)
    return out
