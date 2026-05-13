from __future__ import annotations

import numpy as np
import pandas as pd

from live.config import LiveConfig


def compute_turnover(current: pd.Series, target: pd.Series) -> float:
    idx = current.index.union(target.index)
    return float(0.5 * (target.reindex(idx, fill_value=0.0) - current.reindex(idx, fill_value=0.0)).abs().sum())


def enforce_exposure_limits(weights: pd.Series, config: LiveConfig) -> pd.Series:
    w = weights.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if w.empty:
        return w
    w = w.clip(-config.MAX_POSITION_WEIGHT, config.MAX_POSITION_WEIGHT)
    gross = float(w.abs().sum())
    if gross > config.MAX_GROSS_EXPOSURE and gross > 0:
        w *= config.MAX_GROSS_EXPOSURE / gross
    net = float(w.sum())
    if abs(net) > config.MAX_NET_EXPOSURE and gross > 0:
        shift = (net - np.sign(net) * config.MAX_NET_EXPOSURE) / len(w)
        w = w - shift
        w = w.clip(-config.MAX_POSITION_WEIGHT, config.MAX_POSITION_WEIGHT)
        gross = float(w.abs().sum())
        if gross > config.MAX_GROSS_EXPOSURE and gross > 0:
            w *= config.MAX_GROSS_EXPOSURE / gross
    return w[abs(w) > 1e-12]


def apply_min_holding_period(
    target: pd.Series,
    current: pd.Series,
    holding_bars: dict[str, int],
    config: LiveConfig,
) -> pd.Series:
    adjusted = target.copy()
    for ticker, current_weight in current.items():
        if abs(current_weight) <= 1e-12:
            continue
        held = holding_bars.get(ticker, config.MIN_HOLDING_PERIOD_BARS)
        target_weight = float(adjusted.get(ticker, 0.0))
        direction_change = target_weight == 0.0 or np.sign(target_weight) != np.sign(current_weight)
        if held < config.MIN_HOLDING_PERIOD_BARS and direction_change:
            adjusted.loc[ticker] = current_weight
    return adjusted


def cap_turnover(current: pd.Series, target: pd.Series, config: LiveConfig) -> pd.Series:
    turnover = compute_turnover(current, target)
    if turnover <= config.MAX_TURNOVER_PER_REBALANCE or turnover <= 0:
        return target
    idx = current.index.union(target.index)
    cur = current.reindex(idx, fill_value=0.0)
    tgt = target.reindex(idx, fill_value=0.0)
    scale = config.MAX_TURNOVER_PER_REBALANCE / turnover
    return (cur + (tgt - cur) * scale).loc[lambda s: s.abs() > 1e-12]


def construct_target_portfolio(
    candidates: pd.DataFrame,
    current_positions: pd.Series,
    config: LiveConfig,
    holding_bars: dict[str, int] | None = None,
) -> pd.Series:
    holding_bars = holding_bars or {}
    if candidates.empty:
        active = pd.DataFrame(columns=["ticker", "alpha", "z_alpha", "passes_signal"])
    else:
        active = candidates[candidates["passes_signal"]].copy()
    if not config.LONG_SHORT:
        active = active[active["alpha"] > 0]
    if active.empty:
        base = pd.Series(dtype=float)
    else:
        score = active.set_index("ticker")["z_alpha"].astype(float)
        if not config.LONG_SHORT:
            score = score.clip(lower=0)
            denom = float(score.sum())
            base = score / denom * min(config.MAX_GROSS_EXPOSURE, 1.0) if denom > 0 else pd.Series(dtype=float)
        else:
            longs = score[score > 0]
            shorts = score[score < 0]
            pieces = []
            if not longs.empty:
                pieces.append(longs / longs.sum() * (config.MAX_GROSS_EXPOSURE / 2.0))
            if not shorts.empty:
                pieces.append((-shorts / (-shorts).sum()) * -(config.MAX_GROSS_EXPOSURE / 2.0))
            base = pd.concat(pieces) if pieces else pd.Series(dtype=float)
    target = enforce_exposure_limits(base, config)
    target = apply_min_holding_period(target, current_positions, holding_bars, config)
    target = enforce_exposure_limits(target, config)
    target = cap_turnover(current_positions, target, config)
    return enforce_exposure_limits(target, config).sort_index()
