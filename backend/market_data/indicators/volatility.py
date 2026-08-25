"""Volatility indicators: ATR, Bollinger Bands, historical (realized) volatility."""
from __future__ import annotations

import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, window: int = 14,
        high_col: str = "High", low_col: str = "Low", close_col: str = "Close") -> pd.Series:
    high, low, close = df[high_col], df[low_col], df[close_col]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    pct_b = (series - lower) / (upper - lower).replace(0, 1e-10)
    bandwidth = (upper - lower) / mid.replace(0, 1e-10)
    return pd.DataFrame({
        "bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
        "bb_pct_b": pct_b, "bb_bandwidth": bandwidth,
    })


def historical_volatility(series: pd.Series, window: int = 20, trading_days: int = 252) -> pd.Series:
    """Annualized realized volatility from log returns."""
    log_returns = np.log(series / series.shift(1))
    return log_returns.rolling(window).std() * np.sqrt(trading_days)


def add_volatility_features(df: pd.DataFrame, close_col: str = "Close") -> pd.DataFrame:
    out = df.copy()
    out["atr_14"] = atr(out)
    out["atr_pct"] = out["atr_14"] / out[close_col]
    bb = bollinger_bands(out[close_col])
    out = out.join(bb)
    out["hist_vol_20"] = historical_volatility(out[close_col])
    return out
