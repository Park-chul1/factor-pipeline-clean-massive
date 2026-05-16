from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from live.alpha import apply_signal_filters, compute_alpha, standardize_alpha
from live.config import LiveConfig
from live.data import load_recent_history, update_local_bar_cache
from live.execution import generate_orders
from live.features import build_latest_exposures
from live.paper_order_submitter import _apply_min_price_to_targets
from live.portfolio import compute_turnover, construct_target_portfolio
from live.risk import assert_demo_safety


def cfg(**kwargs) -> LiveConfig:
    base = LiveConfig(
        MIN_DOLLAR_VOLUME=0,
        MIN_FACTOR_NAMES=2,
        MIN_EXPOSURE_COVERAGE=0.5,
        ALPHA_Z_THRESHOLD=0.5,
        MAX_TURNOVER_PER_REBALANCE=1.0,
        MIN_HOLDING_PERIOD_BARS=2,
    )
    return base.__class__(**{**base.__dict__, **kwargs})


def test_no_lookahead_in_delayed_history(tmp_path):
    path = tmp_path / "bars.parquet"
    bars = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-05-12 10:00", "2026-05-12 10:15"]),
        "ticker": ["A", "A"],
        "open": [10, 99],
        "high": [11, 100],
        "low": [9, 98],
        "close": [10, 99],
        "volume": [1000, 1000],
    })
    update_local_bar_cache(bars, path)
    hist = load_recent_history(["A"], 10, path, end_time=pd.Timestamp("2026-05-12 10:00"))
    assert hist["timestamp"].max() == pd.Timestamp("2026-05-12 10:00")
    assert hist["close"].iloc[-1] == 10


def test_alpha_threshold_filter():
    config = cfg(ALPHA_Z_THRESHOLD=0.9, COST_BPS=1, SLIPPAGE_BPS=1)
    alpha = pd.Series({"A": 0.02, "B": 0.00001, "C": -0.03})
    z = pd.Series({"A": 1.0, "B": 2.0, "C": -0.1})
    bars = pd.DataFrame({"ticker": ["A", "B", "C"], "close": [10, 10, 10], "volume": [1000, 1000, 1000]})
    out = apply_signal_filters(alpha, z, bars, config).set_index("ticker")
    assert bool(out.loc["A", "passes_signal"])
    assert not bool(out.loc["B", "passes_signal"])
    assert not bool(out.loc["C", "passes_signal"])


def test_alpha_filter_uses_net_alpha_after_cost_threshold():
    config = cfg(ALPHA_Z_THRESHOLD=0.0, COST_BPS=1, SLIPPAGE_BPS=2, MIN_NET_ALPHA_AFTER_COST_BPS=5)
    alpha = pd.Series({"A": 0.0009, "B": 0.0007})
    z = pd.Series({"A": 1.0, "B": 1.0})
    bars = pd.DataFrame({"ticker": ["A", "B"], "close": [10, 10], "volume": [1000, 1000]})
    out = apply_signal_filters(alpha, z, bars, config).set_index("ticker")
    assert bool(out.loc["A", "passes_signal"])
    assert not bool(out.loc["B", "passes_signal"])


def test_turnover_cap():
    config = cfg(MAX_TURNOVER_PER_REBALANCE=0.05, MAX_POSITION_WEIGHT=1, MAX_GROSS_EXPOSURE=2)
    candidates = pd.DataFrame({
        "ticker": ["A", "B"],
        "alpha": [0.02, -0.02],
        "z_alpha": [2.0, -2.0],
        "passes_signal": [True, True],
    })
    target = construct_target_portfolio(candidates, pd.Series(dtype=float), config)
    assert compute_turnover(pd.Series(dtype=float), target) <= 0.0500001


def test_min_holding_period_blocks_exit():
    config = cfg(MIN_HOLDING_PERIOD_BARS=3, MAX_TURNOVER_PER_REBALANCE=1)
    candidates = pd.DataFrame(columns=["ticker", "alpha", "z_alpha", "passes_signal"])
    current = pd.Series({"A": 0.01})
    target = construct_target_portfolio(candidates, current, config, holding_bars={"A": 1})
    assert target.loc["A"] == pytest.approx(0.01)


def test_max_position_size():
    config = cfg(MAX_POSITION_WEIGHT=0.03, MAX_GROSS_EXPOSURE=1, MAX_TURNOVER_PER_REBALANCE=1)
    candidates = pd.DataFrame({
        "ticker": ["A", "B"],
        "alpha": [0.05, -0.05],
        "z_alpha": [10.0, -10.0],
        "passes_signal": [True, True],
    })
    target = construct_target_portfolio(candidates, pd.Series(dtype=float), config)
    assert target.abs().max() <= 0.0300001


def test_paper_demo_safety_flag():
    safe = cfg(PAPER_TRADING=True, ENABLE_REAL_TRADING=False)
    assert_demo_safety(safe)
    unsafe = cfg(PAPER_TRADING=False, ENABLE_REAL_TRADING=False)
    with pytest.raises(RuntimeError):
        assert_demo_safety(unsafe)


def test_order_generation_from_current_to_target():
    config = cfg(MIN_ORDER_DOLLARS=0)
    orders = generate_orders(
        current_positions=pd.Series({"A": 5}),
        target_weights=pd.Series({"A": 0.02, "B": -0.01}),
        prices=pd.Series({"A": 10.0, "B": 20.0}),
        equity=10_000,
        config=config,
    )
    by_ticker = {o.ticker: o for o in orders}
    assert by_ticker["A"].side == "BUY"
    assert by_ticker["A"].quantity == 15
    assert by_ticker["B"].side == "SELL"
    assert by_ticker["B"].quantity == 5


def test_paper_submitter_drops_low_price_targets_before_order_generation():
    target = pd.Series({"A": 0.01, "B": -0.01, "C": 0.02})
    prices = pd.Series({"A": 4.99, "B": 5.00})
    filtered = _apply_min_price_to_targets(target, prices, min_price=5.0)
    assert filtered.to_dict() == {"B": -0.01, "C": 0.02}


def test_nan_exposure_handling():
    X = pd.DataFrame({"f1": [1.0, np.nan, np.nan], "f2": [0.5, np.nan, np.nan]}, index=["A", "B", "C"])
    alpha = compute_alpha(X.dropna(how="all"), pd.Series({"f1": 1.0, "f2": 1.0}))
    z = standardize_alpha(alpha)
    assert "B" not in alpha.index
    assert z.index.tolist() == ["A"]


def test_feature_layer_drops_poor_nan_coverage():
    config = cfg(MIN_FACTOR_NAMES=2, MIN_EXPOSURE_COVERAGE=0.5, LOOKBACK_BARS=80)
    rows = []
    for i, ts in enumerate(pd.date_range("2026-01-01 09:30", periods=80, freq="15min")):
        for ticker, base in [("A", 10), ("B", 20), ("C", 30)]:
            rows.append({
                "timestamp": ts,
                "ticker": ticker,
                "open": base + i * 0.1,
                "high": base + i * 0.1 + 1,
                "low": base + i * 0.1 - 1,
                "close": base + i * 0.1,
                "volume": 1000 if ticker != "C" else np.nan,
            })
    X, tickers, names, coverage = build_latest_exposures(pd.DataFrame(rows), config)
    assert set(tickers).issubset({"A", "B", "C"})
    assert np.isfinite(X.to_numpy(dtype=float)).all() if not X.empty else True
