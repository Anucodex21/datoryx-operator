"""Volume indicators: volume moving average, spikes, OBV, price/volume divergence."""
from __future__ import annotations

import numpy as np
import pandas as pd


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def add_volume_features(df: pd.DataFrame, close_col: str = "Close", volume_col: str = "Volume") -> pd.DataFrame:
    out = df.copy()
    out["vol_sma_20"] = out[volume_col].rolling(20).mean()
    out["vol_ratio"] = out[volume_col] / out["vol_sma_20"].replace(0, np.nan)
    out["vol_spike"] = (out["vol_ratio"] > 2.0).astype(int)

    out["obv"] = on_balance_volume(out[close_col], out[volume_col])
    out["obv_slope_5"] = out["obv"].diff(5)

    # Price/volume divergence: price up but volume trending down, or vice versa
    price_dir = np.sign(out[close_col].diff(5))
    volume_dir = np.sign(out["obv_slope_5"])
    out["price_volume_divergence"] = (price_dir != volume_dir).astype(int)

    return out
