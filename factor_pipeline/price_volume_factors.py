from __future__ import annotations

import numpy as np
import pandas as pd


def _ret(close: pd.DataFrame, n: int) -> pd.DataFrame:
    return close / close.shift(n) - 1.0


def _logret(close: pd.DataFrame) -> pd.DataFrame:
    return np.log(close).diff()


def build_price_volume_factors(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    close = panel["adj_close"]
    high = panel.get("high", close)
    low = panel.get("low", close)
    open_ = panel.get("open", close)
    vol = panel["volume"].replace(0, np.nan)
    dollar = (close * vol).replace(0, np.nan)
    lr = _logret(close)

    f: dict[str, pd.DataFrame] = {}
    for n in [1, 5, 10, 21, 63, 126, 252]:
        f[f"mom_{n}"] = _ret(close, n)
    f["rev_1"] = -_ret(close, 1)
    f["rev_5"] = -_ret(close, 5)
    f["rev_21"] = -_ret(close, 21)
    for n in [10, 20, 63, 126]:
        f[f"volatility_{n}"] = lr.rolling(n, min_periods=max(3, n // 2)).std()
    f["parkinson_vol_20"] = (np.log(high / low) ** 2).rolling(20, min_periods=10).mean() / (4 * np.log(2))
    f["intraday_return"] = close / open_ - 1.0
    f["overnight_return"] = open_ / close.shift(1) - 1.0
    f["hl_range_20"] = ((high - low) / close).rolling(20, min_periods=10).mean()
    f["close_to_high_20"] = close / high.rolling(20, min_periods=10).max() - 1.0
    f["close_to_low_20"] = close / low.rolling(20, min_periods=10).min() - 1.0
    f["dist_52w_high"] = close / high.rolling(252, min_periods=126).max() - 1.0
    f["dist_52w_low"] = close / low.rolling(252, min_periods=126).min() - 1.0
    for n in [20, 60]:
        f[f"log_dollar_vol_{n}"] = np.log(dollar.rolling(n, min_periods=max(5, n // 2)).mean())
        f[f"volume_z_{n}"] = (vol - vol.rolling(n, min_periods=max(5, n // 2)).mean()) / vol.rolling(n, min_periods=max(5, n // 2)).std()
    f["volume_mom_20_60"] = vol.rolling(20, min_periods=10).mean() / vol.rolling(60, min_periods=30).mean() - 1.0
    f["amihud_20"] = lr.abs().rolling(20, min_periods=10).mean() / dollar.rolling(20, min_periods=10).mean()
    f["price_skew_63"] = lr.rolling(63, min_periods=30).skew()
    f["price_kurt_63"] = lr.rolling(63, min_periods=30).kurt()
    return f
