from __future__ import annotations

import numpy as np
import pandas as pd

FIELDS = ["open", "high", "low", "close", "adj_close", "volume", "vwap", "transactions"]


def bars_long_to_panel(bars: pd.DataFrame, tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    if bars.empty:
        raise ValueError("bars is empty")
    bars = bars.copy()
    bars["date"] = pd.to_datetime(bars["date"]).dt.normalize()
    if tickers is None:
        tickers = sorted(bars["ticker"].dropna().unique().tolist())
    dates = pd.DatetimeIndex(sorted(bars["date"].unique()), name="date")
    panel: dict[str, pd.DataFrame] = {}
    for field in FIELDS:
        src = "close" if field == "adj_close" and "adj_close" not in bars.columns else field
        if src not in bars.columns:
            continue
        wide = bars.pivot_table(index="date", columns="ticker", values=src, aggfunc="last")
        wide = wide.reindex(index=dates, columns=tickers).astype(float)
        panel[field] = wide
    return panel


def compute_forward_returns(adj_close: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    return adj_close.shift(-horizon) / adj_close - 1.0


def finite_ratio(x) -> float:
    arr = np.asarray(x, dtype=float)
    return float(np.isfinite(arr).mean()) if arr.size else 0.0
