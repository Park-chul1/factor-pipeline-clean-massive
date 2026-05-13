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


def build_tradable_mask(
    panel: dict[str, pd.DataFrame],
    require_volume: bool = True,
) -> pd.DataFrame:
    """Return a date x ticker mask for names tradable on each date.

    The mask is intentionally derived from the historical daily bars already
    loaded into the pipeline, so it adds only one boolean T x N array and does
    not require per-ticker state objects.
    """
    if "adj_close" in panel:
        price = panel["adj_close"]
    elif "close" in panel:
        price = panel["close"]
    else:
        raise KeyError("panel must contain adj_close or close")

    price_arr = price.to_numpy(dtype=float)
    mask = pd.DataFrame(
        np.isfinite(price_arr) & (price_arr > 0),
        index=price.index,
        columns=price.columns,
    )

    if require_volume and "volume" in panel:
        volume = panel["volume"].reindex_like(price)
        volume_arr = volume.to_numpy(dtype=float)
        volume_mask = pd.DataFrame(
            np.isfinite(volume_arr) & (volume_arr > 0),
            index=price.index,
            columns=price.columns,
        )
        mask &= volume_mask

    return mask.astype(bool)


def compute_forward_returns(
    adj_close: pd.DataFrame,
    horizon: int = 1,
    max_abs_return: float | None = None,
) -> pd.DataFrame:
    returns = adj_close.shift(-horizon) / adj_close - 1.0
    if max_abs_return is not None:
        if max_abs_return <= 0:
            raise ValueError("max_abs_return must be positive")
        returns = returns.mask(returns.abs() > max_abs_return)
    return returns


def finite_ratio(x) -> float:
    arr = np.asarray(x, dtype=float)
    return float(np.isfinite(arr).mean()) if arr.size else 0.0
