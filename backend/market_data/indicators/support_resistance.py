"""
Price-structure features: rolling support/resistance levels, breakouts,
higher-highs/lower-lows, and gaps. Simpler and more robust than trying to
detect exact chart patterns numerically — pattern detection from pixels is
Phase 7 (chart_ai/), not this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_support_resistance_features(
    df: pd.DataFrame, window: int = 20,
    high_col: str = "High", low_col: str = "Low", close_col: str = "Close", open_col: str = "Open",
) -> pd.DataFrame:
    out = df.copy()

    out["resistance_20"] = out[high_col].rolling(window).max()
    out["support_20"] = out[low_col].rolling(window).min()

    out["breakout_up"] = (out[close_col] > out["resistance_20"].shift(1)).astype(int)
    out["breakout_down"] = (out[close_col] < out["support_20"].shift(1)).astype(int)

    # Higher-highs / lower-lows over a short lookback (simple swing structure)
    out["higher_high"] = (out[high_col] > out[high_col].shift(1).rolling(5).max()).astype(int)
    out["lower_low"] = (out[low_col] < out[low_col].shift(1).rolling(5).min()).astype(int)

    # Gaps: today's open vs. yesterday's close
    prev_close = out[close_col].shift(1)
    out["gap_pct"] = (out[open_col] - prev_close) / prev_close
    out["gap_up"] = (out["gap_pct"] > 0.01).astype(int)
    out["gap_down"] = (out["gap_pct"] < -0.01).astype(int)

    return out
