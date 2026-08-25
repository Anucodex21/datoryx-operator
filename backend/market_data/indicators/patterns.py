"""
NOT IMPLEMENTED YET.

Candlestick pattern detection (doji, engulfing, hammer, etc.) computed
numerically from OHLC data, as distinct from the CNN-based visual chart
pattern recognition that belongs in chart_ai/ (Phase 7).

Left as a stub deliberately: MVP1 ships without it. When you're ready,
implement pure functions here, one per pattern, each returning a 0/1
Series aligned to the DataFrame index, e.g.:

    def is_bullish_engulfing(df: pd.DataFrame) -> pd.Series: ...
    def is_doji(df: pd.DataFrame, body_pct_threshold: float = 0.1) -> pd.Series: ...

then wire them into features/technical_features.py alongside the other
indicator modules.
"""
