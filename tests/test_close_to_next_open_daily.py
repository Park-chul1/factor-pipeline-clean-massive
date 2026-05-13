from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from backtests.close_to_next_open_daily import (
    DailyBacktestConfig,
    REQUIRED_REPORT_FILES,
    _rank_weights_one_day,
    compute_alpha_and_contributions,
    run_close_to_next_open_backtest,
    write_daily_report,
)


def _write_fixture(tmp_path):
    input_dir = tmp_path / "processed"
    input_dir.mkdir()
    dates = pd.bdate_range("2024-01-02", periods=6)
    tickers = [f"T{i:02d}" for i in range(10)]
    factors = ["momentum", "quality", "liquidity"]
    X = np.zeros((len(dates), len(tickers), len(factors)), dtype=float)
    for t in range(len(dates)):
        X[t, :, 0] = np.linspace(-1, 1, len(tickers))
        X[t, :, 1] = np.linspace(1, -1, len(tickers)) * 0.2
        X[t, :, 2] = 0.1
    f = np.tile(np.array([0.02, -0.01, 0.005]), (len(dates), 1))
    f[0] = np.nan
    r = np.tile(np.linspace(-0.01, 0.01, len(tickers)), (len(dates), 1))
    tradable = np.ones((len(dates), len(tickers)), dtype=bool)
    np.save(input_dir / "X.npy", X)
    np.save(input_dir / "factor_returns.npy", f)
    np.save(input_dir / "r.npy", r)
    np.save(input_dir / "tradable_mask.npy", tradable)
    pd.DataFrame({"date": dates}).to_csv(input_dir / "dates.csv", index=False)
    pd.DataFrame({"ticker": tickers}).to_csv(input_dir / "tickers.csv", index=False)
    pd.DataFrame({"factor": factors}).to_csv(input_dir / "factor_names.csv", index=False)

    rows = []
    for t, date in enumerate(dates):
        for i, ticker in enumerate(tickers):
            price = 20 + i + t
            rows.append({
                "date": date,
                "ticker": ticker,
                "open": price,
                "close": price + 0.5,
                "volume": 1_000_000 + i,
            })
    bars_path = tmp_path / "bars.parquet"
    pd.DataFrame(rows).to_parquet(bars_path, index=False)
    return input_dir, bars_path, dates, tickers, factors, X, f


def _cfg(tmp_path) -> DailyBacktestConfig:
    input_dir, bars_path, *_ = _write_fixture(tmp_path)
    return DailyBacktestConfig(
        input_dir=input_dir,
        out_dir=tmp_path / "bt",
        daily_bars_path=bars_path,
        reports_dir=tmp_path / "reports",
        min_names_per_side=1,
        max_position_weight=1.0,
        gross_exposure=2.0,
        forecast_method="latest",
        min_periods=1,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )


def test_timing_and_execution_uses_next_open(tmp_path):
    cfg = _cfg(tmp_path)
    result = run_close_to_next_open_backtest(cfg)
    daily = result.daily.dropna(subset=["execution_date"])
    assert (pd.to_datetime(daily["signal_date"]) < pd.to_datetime(daily["execution_date"])).all()
    first_order = result.orders.dropna(subset=["execution_price"]).iloc[0]
    bars = pd.read_parquet(cfg.daily_bars_path)
    expected_open = bars[(pd.to_datetime(bars["date"]).dt.normalize() == pd.Timestamp(first_order["execution_date"]).normalize()) & (bars["ticker"] == first_order["ticker"])]["open"].iloc[0]
    assert first_order["execution_price"] == pytest.approx(expected_open)


def test_oracle_forecast_is_rejected(tmp_path):
    cfg = _cfg(tmp_path)
    bad = cfg.__class__(**{**cfg.__dict__, "forecast_method": "oracle"})
    with pytest.raises(ValueError):
        run_close_to_next_open_backtest(bad)


def test_portfolio_quantiles_and_exposure(tmp_path):
    cfg = _cfg(tmp_path)
    result = run_close_to_next_open_backtest(cfg)
    ranked = result.alpha_rankings[pd.to_datetime(result.alpha_rankings["signal_date"]).dt.normalize().eq(pd.Timestamp("2024-01-04"))]
    longs = ranked[ranked["target_weight"] > 0]
    shorts = ranked[ranked["target_weight"] < 0]
    assert not longs.empty
    assert not shorts.empty
    assert longs["alpha_score"].min() >= ranked["alpha_score"].quantile(cfg.long_quantile) - 1e-12
    assert shorts["alpha_score"].max() <= ranked["alpha_score"].quantile(cfg.short_quantile) + 1e-12
    latest = result.daily[pd.to_datetime(result.daily["signal_date"]).dt.normalize().eq(pd.Timestamp("2024-01-04"))].iloc[0]
    assert latest["gross_exposure"] == pytest.approx(2.0)
    assert latest["net_exposure"] == pytest.approx(0.0)


def test_cost_adjusted_alpha_threshold_blocks_weak_signals(tmp_path):
    cfg = DailyBacktestConfig(
        min_names_per_side=1,
        max_position_weight=1.0,
        gross_exposure=2.0,
        transaction_cost_bps=1.0,
        slippage_bps=2.0,
        min_net_alpha_after_cost_bps=5.0,
    )
    alpha = np.array([0.0007, 0.0009, -0.0007, -0.0009])
    target, info = _rank_weights_one_day(alpha, np.ones_like(alpha, dtype=bool), np.full_like(alpha, 1_000_000.0), cfg)
    assert target[1] > 0
    assert target[3] < 0
    assert target[0] == 0
    assert target[2] == 0
    assert info["cost_adjusted_alpha_threshold"] == pytest.approx(0.0008)
    assert info["filtered_alpha_threshold"] == 2


def test_reports_are_generated_and_reasons_present(tmp_path):
    cfg = _cfg(tmp_path)
    result = run_close_to_next_open_backtest(cfg)
    report_dir = write_daily_report(result, cfg.reports_dir, signal_date=pd.Timestamp("2024-01-03"))
    for name in REQUIRED_REPORT_FILES:
        assert (report_dir / name).exists(), name
    assert (report_dir / "factor_dashboard.png").exists()
    orders = pd.read_csv(report_dir / "orders.csv")
    assert orders["reason_text"].fillna("").str.len().gt(0).all()
    metadata = json.loads((report_dir / "pipeline_metadata.json").read_text())
    assert metadata["signal_date"] < metadata["execution_date"]


def test_factor_contributions_sum_to_alpha(tmp_path):
    input_dir, _bars_path, _dates, _tickers, _factors, X, f = _write_fixture(tmp_path)
    f_pred = np.vstack([np.full(3, np.nan), f[:-1]])
    alpha, contrib = compute_alpha_and_contributions(X, f_pred)
    mask = np.isfinite(alpha)
    assert np.allclose(contrib.sum(axis=2)[mask], alpha[mask])


def test_residual_heatmap_handles_many_tickers(tmp_path):
    cfg = _cfg(tmp_path)
    result = run_close_to_next_open_backtest(cfg)
    report_dir = write_daily_report(result, cfg.reports_dir, signal_date=pd.Timestamp("2024-01-03"))
    assert (report_dir / "residual_heatmap.png").stat().st_size > 0


def test_kalman_diagnostic_columns_are_reported(tmp_path):
    cfg = _cfg(tmp_path)
    result = run_close_to_next_open_backtest(cfg)
    cols = set(result.factor_forecast.columns)
    assert {"kalman_f_hat", "kalman_gain", "state_covariance_diag", "process_noise_q", "measurement_noise_r", "final_f_hat"}.issubset(cols)
