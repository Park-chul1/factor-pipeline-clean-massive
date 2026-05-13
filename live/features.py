from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from factor_pipeline.price_volume_factors import build_price_volume_factors
from factor_pipeline.preprocess import preprocess_factor
from live.config import LiveConfig


def _history_to_daily_like_panel(history: pd.DataFrame) -> dict[str, pd.DataFrame]:
    bars = history.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    bars = bars.dropna(subset=["timestamp", "ticker"])
    tickers = sorted(bars["ticker"].astype(str).unique())
    times = pd.DatetimeIndex(sorted(bars["timestamp"].unique()), name="timestamp")
    panel: dict[str, pd.DataFrame] = {}
    for field in ["open", "high", "low", "close", "volume", "vwap", "transactions"]:
        if field not in bars.columns:
            continue
        wide = bars.pivot_table(index="timestamp", columns="ticker", values=field, aggfunc="last")
        panel[field] = wide.reindex(index=times, columns=tickers).astype(float)
    panel["adj_close"] = panel["close"]
    return panel


def build_latest_exposures(
    recent_history: pd.DataFrame,
    config: LiveConfig,
) -> tuple[pd.DataFrame, list[str], list[str], pd.Series]:
    if recent_history.empty:
        return pd.DataFrame(), [], [], pd.Series(dtype=float)

    panel = _history_to_daily_like_panel(recent_history)
    raw = build_price_volume_factors(panel)
    latest_ts = max(next(iter(raw.values())).index)
    exposures: dict[str, pd.Series] = {}
    diagnostics = []

    for name, df in raw.items():
        z = preprocess_factor(df, min_names=config.MIN_FACTOR_NAMES)
        row = z.loc[latest_ts]
        finite_fraction = float(np.isfinite(row.to_numpy(dtype=float)).mean()) if len(row) else 0.0
        diagnostics.append({"factor": name, "finite_fraction": finite_fraction})
        if finite_fraction > 0:
            exposures[name] = row

    if not exposures:
        return pd.DataFrame(), [], [], pd.Series(dtype=float)

    X = pd.DataFrame(exposures)
    finite_by_ticker = X.notna().mean(axis=1)
    keep = finite_by_ticker >= config.MIN_EXPOSURE_COVERAGE
    X = X.loc[keep]
    X = X.dropna(axis=1, how="all")
    X = X.dropna(axis=0, how="all")
    if X.empty:
        return pd.DataFrame(), [], [], finite_by_ticker

    # After dropping poor-coverage names, zero is the neutral z-scored exposure.
    X = X.fillna(0.0)
    logging.info(
        "latest exposures timestamp=%s tickers=%s factors=%s finite_fraction=%.4f",
        latest_ts,
        len(X.index),
        len(X.columns),
        float(np.isfinite(X.to_numpy(dtype=float)).mean()),
    )
    return X, X.index.astype(str).tolist(), X.columns.astype(str).tolist(), finite_by_ticker
