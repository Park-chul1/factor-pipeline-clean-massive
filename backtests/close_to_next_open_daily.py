from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from factor_pipeline.signals import predict_factor_returns


REQUIRED_REPORT_FILES = [
    "summary.md",
    "orders.csv",
    "target_positions.csv",
    "current_positions.csv",
    "factor_returns.csv",
    "factor_forecast.csv",
    "alpha_rankings.csv",
    "quantile_summary.csv",
    "residuals.csv",
    "residual_heatmap.png",
    "outliers.csv",
    "execution_quality.csv",
    "risk_summary.json",
    "pipeline_metadata.json",
]


@dataclass(frozen=True)
class DailyBacktestConfig:
    input_dir: Path = Path("data/processed_clean")
    out_dir: Path = Path("data/close_to_next_open_backtest")
    daily_bars_path: Path | None = None
    reports_dir: Path = Path("reports/daily")
    rebalance_frequency: str = "daily"
    holding_period_days: int = 1
    long_quantile: float = 0.9
    short_quantile: float = 0.1
    max_positions_long: int | None = None
    max_positions_short: int | None = None
    gross_exposure: float = 2.0
    max_position_weight: float = 0.05
    min_alpha_threshold: float = 0.0
    min_dollar_volume: float = 0.0
    turnover_cap: float | None = None
    transaction_cost_bps: float = 1.0
    slippage_bps: float = 2.0
    min_net_alpha_after_cost_bps: float = 5.0
    execution_price_mode: str = "open"
    allow_short: bool = True
    forecast_method: str = "ewma"
    lookback: int = 20
    ewma_halflife: float = 20.0
    min_periods: int = 5
    min_names_per_side: int = 5
    residual_outlier_z_threshold: float = 3.0
    initial_equity: float = 100_000.0
    max_abs_return: float | None = 1.0
    covariance_warning_threshold: float = 1e6

    def validate(self) -> None:
        if self.rebalance_frequency != "daily":
            raise NotImplementedError("Only daily rebalance_frequency is implemented")
        if self.holding_period_days < 1:
            raise ValueError("holding_period_days must be >= 1")
        if not (0 < self.short_quantile < 0.5):
            raise ValueError("short_quantile must be in (0, 0.5)")
        if not (0.5 < self.long_quantile < 1):
            raise ValueError("long_quantile must be in (0.5, 1)")
        if self.gross_exposure <= 0:
            raise ValueError("gross_exposure must be positive")
        if self.max_position_weight <= 0:
            raise ValueError("max_position_weight must be positive")
        if self.execution_price_mode not in {"open", "vwap_proxy", "open_plus_slippage"}:
            raise ValueError("execution_price_mode must be open, vwap_proxy, or open_plus_slippage")
        if self.forecast_method == "oracle":
            raise ValueError("oracle forecast is prohibited for the canonical daily backtest")
        if self.transaction_cost_bps < 0 or self.slippage_bps < 0:
            raise ValueError("transaction_cost_bps and slippage_bps must be non-negative")
        if self.min_net_alpha_after_cost_bps < 0:
            raise ValueError("min_net_alpha_after_cost_bps must be non-negative")


@dataclass
class DailyBacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    orders: pd.DataFrame
    target_positions: pd.DataFrame
    current_positions: pd.DataFrame
    factor_forecast: pd.DataFrame
    factor_returns: pd.DataFrame
    alpha_rankings: pd.DataFrame
    quantile_summary: pd.DataFrame
    residuals: pd.DataFrame
    outliers: pd.DataFrame
    execution_quality: pd.DataFrame
    risk_summary: dict[str, Any]
    metadata: dict[str, Any]
    warnings: list[str]


def config_from_dict(raw: dict[str, Any]) -> DailyBacktestConfig:
    names = {f.name for f in fields(DailyBacktestConfig)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"Unknown close-to-next-open config keys: {sorted(unknown)}")
    path_fields = {"input_dir", "out_dir", "daily_bars_path", "reports_dir"}
    clean = dict(raw)
    for key in path_fields & set(clean):
        clean[key] = None if clean[key] in {None, ""} else Path(clean[key])
    cfg = DailyBacktestConfig(**clean)
    cfg.validate()
    return cfg


def load_processed_inputs(input_dir: Path) -> dict[str, Any]:
    X = np.load(input_dir / "X.npy")
    f = np.load(input_dir / "factor_returns.npy")
    r = np.load(input_dir / "r.npy") if (input_dir / "r.npy").exists() else None
    tradable = np.load(input_dir / "tradable_mask.npy") if (input_dir / "tradable_mask.npy").exists() else np.ones(X.shape[:2], dtype=bool)
    dates = pd.to_datetime(pd.read_csv(input_dir / "dates.csv")["date"]).dt.normalize()
    tickers_df = pd.read_csv(input_dir / "tickers.csv")
    ticker_col = "ticker" if "ticker" in tickers_df.columns else tickers_df.columns[0]
    tickers = tickers_df[ticker_col].astype(str).tolist()[: X.shape[1]]
    factor_df = pd.read_csv(input_dir / "factor_names.csv")
    factor_col = "factor" if "factor" in factor_df.columns else factor_df.columns[0]
    factor_names = factor_df[factor_col].astype(str).tolist()[: X.shape[2]]
    return {"X": X, "f": f, "r": r, "tradable": tradable, "dates": pd.DatetimeIndex(dates), "tickers": tickers, "factor_names": factor_names}


def load_daily_price_panels(path: Path | None, dates: pd.DatetimeIndex, tickers: list[str]) -> dict[str, pd.DataFrame]:
    if path is None or not path.exists():
        return {}
    bars = pd.read_parquet(path)
    if "date" not in bars.columns or "ticker" not in bars.columns:
        raise ValueError("daily_bars_path must contain date and ticker columns")
    bars = bars.copy()
    bars["date"] = pd.to_datetime(bars["date"]).dt.normalize()
    panels: dict[str, pd.DataFrame] = {}
    for col in ["open", "close", "vwap", "volume"]:
        if col in bars.columns:
            panels[col] = (
                bars.pivot_table(index="date", columns="ticker", values=col, aggfunc="last")
                .reindex(index=dates, columns=tickers)
                .astype(float)
            )
    return panels


def daily_bars_date_range(path: Path | None) -> tuple[str | None, str | None]:
    if path is None or not path.exists():
        return None, None
    dates = pd.to_datetime(pd.read_parquet(path, columns=["date"])["date"]).dt.normalize()
    if dates.empty:
        return None, None
    return dates.min().date().isoformat(), dates.max().date().isoformat()


def check_timing(dates: pd.DatetimeIndex, f: np.ndarray, f_pred: np.ndarray, method: str) -> list[str]:
    warnings: list[str] = []
    if not dates.is_monotonic_increasing:
        raise ValueError("dates must be monotonic increasing")
    if method == "oracle":
        raise ValueError("oracle forecast uses same-date factor returns and is not allowed")
    if f.shape != f_pred.shape:
        raise ValueError("factor return and forecast shapes differ")
    if len(dates) > 1 and not (dates[1:] > dates[:-1]).all():
        raise ValueError("dates must be strictly increasing")
    return warnings


def compute_alpha_and_contributions(X: np.ndarray, f_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_safe = np.where(np.isfinite(X), X, 0.0)
    f_safe = np.where(np.isfinite(f_pred), f_pred, 0.0)
    contrib = X_safe * f_safe[:, None, :]
    alpha = np.nansum(contrib, axis=2)
    alpha[~np.isfinite(f_pred).any(axis=1), :] = np.nan
    return alpha, contrib


def _rank_weights_one_day(alpha: np.ndarray, tradable: np.ndarray, dollar_volume: np.ndarray | None, cfg: DailyBacktestConfig) -> tuple[np.ndarray, dict[str, Any]]:
    base_valid = np.isfinite(alpha) & tradable
    valid = base_valid.copy()
    if dollar_volume is not None:
        valid &= np.isfinite(dollar_volume) & (dollar_volume >= cfg.min_dollar_volume)
    trading_cost = (cfg.transaction_cost_bps + cfg.slippage_bps) / 10_000.0
    min_net_after_cost = cfg.min_net_alpha_after_cost_bps / 10_000.0
    cost_adjusted_threshold = max(cfg.min_alpha_threshold, trading_cost + min_net_after_cost)
    passes_cost_threshold = (alpha >= cost_adjusted_threshold) | (alpha <= -cost_adjusted_threshold)
    valid &= passes_cost_threshold
    w = np.zeros_like(alpha, dtype=float)
    vals = alpha[valid]
    info: dict[str, Any] = {
        "long_threshold_alpha": np.nan,
        "short_threshold_alpha": np.nan,
        "cost_adjusted_alpha_threshold": cost_adjusted_threshold,
        "trading_cost_return": trading_cost,
        "min_net_alpha_after_cost": min_net_after_cost,
        "filtered_missing_or_untradable": int((~base_valid).sum()),
        "filtered_liquidity": 0,
        "filtered_alpha_threshold": int((base_valid & ~passes_cost_threshold).sum()),
    }
    if dollar_volume is not None:
        info["filtered_liquidity"] = int((base_valid & ~(np.isfinite(dollar_volume) & (dollar_volume >= cfg.min_dollar_volume))).sum())
    if vals.size < 2 * cfg.min_names_per_side:
        return w, info
    short_cut = float(np.nanquantile(vals, cfg.short_quantile))
    long_cut = float(np.nanquantile(vals, cfg.long_quantile))
    info["long_threshold_alpha"] = long_cut
    info["short_threshold_alpha"] = short_cut
    long_idx = np.where(valid & (alpha >= long_cut))[0]
    short_idx = np.where(valid & (alpha <= short_cut))[0] if cfg.allow_short else np.array([], dtype=int)
    long_idx = long_idx[np.argsort(alpha[long_idx])[::-1]]
    short_idx = short_idx[np.argsort(alpha[short_idx])]
    if cfg.max_positions_long:
        long_idx = long_idx[: cfg.max_positions_long]
    if cfg.max_positions_short:
        short_idx = short_idx[: cfg.max_positions_short]
    if len(long_idx) < cfg.min_names_per_side:
        long_idx = np.array([], dtype=int)
    if cfg.allow_short and len(short_idx) < cfg.min_names_per_side:
        short_idx = np.array([], dtype=int)
    long_leg = cfg.gross_exposure / 2.0 if cfg.allow_short and len(short_idx) else cfg.gross_exposure
    short_leg = cfg.gross_exposure / 2.0
    if len(long_idx):
        w[long_idx] = min(long_leg / len(long_idx), cfg.max_position_weight)
    if cfg.allow_short and len(short_idx):
        w[short_idx] = -min(short_leg / len(short_idx), cfg.max_position_weight)
    gross = float(np.abs(w).sum())
    if gross > cfg.gross_exposure and gross > 0:
        w *= cfg.gross_exposure / gross
    return w, info


def build_daily_weights(alpha: np.ndarray, tradable: np.ndarray, dollar_volume: np.ndarray | None, cfg: DailyBacktestConfig) -> tuple[np.ndarray, pd.DataFrame]:
    weights = np.zeros_like(alpha, dtype=float)
    rows: list[dict[str, Any]] = []
    prev = np.zeros(alpha.shape[1], dtype=float)
    for t in range(alpha.shape[0]):
        dv = None if dollar_volume is None else dollar_volume[t]
        target, info = _rank_weights_one_day(alpha[t], tradable[t], dv, cfg)
        if cfg.turnover_cap is not None:
            turnover = 0.5 * np.abs(target - prev).sum()
            if turnover > cfg.turnover_cap and turnover > 0:
                target = prev + (target - prev) * (cfg.turnover_cap / turnover)
        weights[t] = target
        info.update({"row": t, "final_number_long": int((target > 0).sum()), "final_number_short": int((target < 0).sum())})
        rows.append(info)
        prev = target
    return weights, pd.DataFrame(rows)


def compute_execution_returns(weights: np.ndarray, dates: pd.DatetimeIndex, panels: dict[str, pd.DataFrame], saved_r: np.ndarray | None, cfg: DailyBacktestConfig) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    warnings: list[str] = []
    T, N = weights.shape
    realized = np.full((T, N), np.nan)
    signal_price = np.full((T, N), np.nan)
    execution_price = np.full((T, N), np.nan)
    exit_price = np.full((T, N), np.nan)
    if {"open", "close"}.issubset(panels):
        open_arr = panels["open"].to_numpy(dtype=float)
        close_arr = panels["close"].to_numpy(dtype=float)
        signal_price = close_arr.copy()
        for t in range(T - cfg.holding_period_days):
            exec_t = t + 1
            exit_t = min(exec_t + cfg.holding_period_days, T - 1)
            execution_price[t] = open_arr[exec_t]
            exit_price[t] = open_arr[exit_t] if exit_t != exec_t else close_arr[exec_t]
            realized[t] = exit_price[t] / execution_price[t] - 1.0
    elif saved_r is not None:
        warnings.append("daily_bars_path was not provided/found; realized returns fall back to saved r.npy, so execution_price is unavailable.")
        realized = saved_r.copy()
    else:
        warnings.append("No daily bars or r.npy available; PnL cannot be computed.")
    if cfg.max_abs_return is not None:
        realized = np.where(np.abs(realized) <= cfg.max_abs_return, realized, np.nan)
    prices = pd.DataFrame({
        "signal_date": np.repeat(dates.to_numpy(), N),
        "execution_date": np.repeat(np.r_[dates.to_numpy()[1:], np.datetime64("NaT")], N),
        "exit_date": np.repeat(np.r_[dates.to_numpy()[1 + cfg.holding_period_days:], np.full(min(1 + cfg.holding_period_days, T), np.datetime64("NaT"))], N)[: T * N],
        "ticker_index": np.tile(np.arange(N), T),
        "signal_price": signal_price.reshape(-1),
        "execution_price": execution_price.reshape(-1),
        "exit_price": exit_price.reshape(-1),
    })
    return realized, prices, warnings


def portfolio_pnl(weights: np.ndarray, realized: np.ndarray, cfg: DailyBacktestConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gross_turnover = np.full(weights.shape[0], np.nan)
    ret = np.full(weights.shape[0], np.nan)
    cost = np.full(weights.shape[0], 0.0)
    prev = np.zeros(weights.shape[1], dtype=float)
    cost_rate = (cfg.transaction_cost_bps + cfg.slippage_bps) / 10_000.0
    for t in range(weights.shape[0]):
        gross_turnover[t] = 0.5 * float(np.abs(weights[t] - prev).sum())
        active = np.abs(weights[t]) > 0
        if active.any():
            r = np.where(np.isfinite(realized[t]), realized[t], 0.0)
            ret[t] = float(np.sum(weights[t] * r))
            cost[t] = gross_turnover[t] * cost_rate
            ret[t] -= cost[t]
        prev = weights[t]
    return ret, gross_turnover, cost


def _safe_git_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _factor_frame(dates: pd.DatetimeIndex, factor_names: list[str], f: np.ndarray, f_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    prev = np.vstack([np.full((1, f_pred.shape[1]), np.nan), f_pred[:-1]])
    for t, date in enumerate(dates):
        hist = f[: t + 1]
        finite = np.isfinite(hist)
        counts = finite.sum(axis=0) if hist.size else np.zeros(len(factor_names), dtype=int)
        sums = np.where(finite, hist, 0.0).sum(axis=0) if hist.size else np.zeros(len(factor_names))
        mean = np.divide(sums, counts, out=np.full(len(factor_names), np.nan), where=counts > 0)
        centered = np.where(finite, hist - mean, 0.0) if hist.size else np.zeros((0, len(factor_names)))
        var = np.divide((centered * centered).sum(axis=0), counts, out=np.full(len(factor_names), np.nan), where=counts > 0)
        std = np.sqrt(var)
        z = (f[t] - mean) / np.where(std > 1e-12, std, np.nan)
        for k, name in enumerate(factor_names):
            rows.append({
                "date": date,
                "factor": name,
                "latest_estimated_f_t": f[t, k],
                "rolling_mean_f": mean[k],
                "kalman_forecast_f_hat": np.nan,
                "previous_day_f_hat": prev[t, k],
                "day_over_day_change": f_pred[t, k] - prev[t, k] if np.isfinite(prev[t, k]) else np.nan,
                "zscore_factor_return": z[k],
                "kalman_f_hat": np.nan,
                "kalman_gain": np.nan,
                "state_covariance_diag": np.nan,
                "process_noise_q": np.nan,
                "measurement_noise_r": np.nan,
                "kalman_weight": 0.0,
                "rolling_weight": 1.0,
                "final_f_hat": f_pred[t, k],
            })
    return pd.DataFrame(rows)


def build_alpha_rankings(
    dates: pd.DatetimeIndex,
    tickers: list[str],
    factor_names: list[str],
    alpha: np.ndarray,
    contrib: np.ndarray,
    X: np.ndarray,
    f_pred: np.ndarray,
    weights: np.ndarray,
    previous_weights: np.ndarray,
    dollar_volume: np.ndarray | None,
    cfg: DailyBacktestConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for t, date in enumerate(dates):
        score = alpha[t]
        finite = np.isfinite(score)
        ranks = pd.Series(score).rank(ascending=False, method="first")
        buckets = pd.qcut(pd.Series(score[finite]), 10, labels=False, duplicates="drop") if finite.sum() >= 10 else pd.Series(dtype=float)
        bucket_map = dict(zip(np.where(finite)[0], (buckets.astype(float) + 1).tolist())) if len(buckets) else {}
        for i, ticker in enumerate(tickers):
            c = contrib[t, i]
            order = np.argsort(np.abs(c))[::-1]
            top_pos = [factor_names[j] for j in order if c[j] > 0][:3]
            top_neg = [factor_names[j] for j in order if c[j] < 0][:3]
            side = ""
            delta = weights[t, i] - previous_weights[t, i]
            if delta > 1e-12:
                side = "BUY" if previous_weights[t, i] >= 0 else "COVER"
            elif delta < -1e-12:
                side = "SHORT" if weights[t, i] < 0 else "SELL"
            bucket = bucket_map.get(i, np.nan)
            liquidity_flag = bool(dollar_volume is None or (np.isfinite(dollar_volume[t, i]) and dollar_volume[t, i] >= cfg.min_dollar_volume))
            risk_flag = bool(abs(weights[t, i]) <= cfg.max_position_weight + 1e-12)
            trading_cost_bps = cfg.transaction_cost_bps + cfg.slippage_bps
            directional_alpha = -score[i] if weights[t, i] < 0 else score[i]
            net_alpha_after_cost_bps = (directional_alpha * 10_000.0) - trading_cost_bps if np.isfinite(score[i]) else np.nan
            passes_cost_threshold = bool(np.isfinite(net_alpha_after_cost_bps) and net_alpha_after_cost_bps >= cfg.min_net_alpha_after_cost_bps)
            reason = ""
            if side:
                if weights[t, i] > 0:
                    direction = "top decile"
                    main = ", ".join(top_pos)
                elif weights[t, i] < 0:
                    direction = "bottom decile"
                    main = ", ".join(top_neg)
                else:
                    direction = "cost-adjusted no-position band"
                    main = ", ".join(top_pos or top_neg)
                main = main or "the cross-sectional factor mix"
                reason = (
                    f"{side} because predicted alpha is in the {direction}. "
                    f"Net alpha after estimated {trading_cost_bps:.1f} bps trading cost is {net_alpha_after_cost_bps:.2f} bps "
                    f"versus threshold {cfg.min_net_alpha_after_cost_bps:.2f} bps. "
                    f"Main contributors were {main}. Liquidity filter {'passed' if liquidity_flag else 'failed'}."
                )
            rows.append({
                "signal_date": date,
                "ticker": ticker,
                "company_name": "",
                "side": side,
                "current_weight": previous_weights[t, i],
                "target_weight": weights[t, i],
                "order_weight_delta": delta,
                "estimated_order_value": delta * cfg.initial_equity,
                "alpha_score": score[i],
                "trading_cost_bps": trading_cost_bps,
                "net_alpha_after_cost_bps": net_alpha_after_cost_bps,
                "min_net_alpha_after_cost_bps": cfg.min_net_alpha_after_cost_bps,
                "passes_cost_threshold": passes_cost_threshold,
                "alpha_rank": ranks.iloc[i] if finite[i] else np.nan,
                "quantile_bucket": bucket,
                "top_positive_factor_contributors": ";".join(top_pos),
                "top_negative_factor_contributors": ";".join(top_neg),
                "factor_exposures": json.dumps({factor_names[k]: float(X[t, i, k]) for k in range(len(factor_names)) if np.isfinite(X[t, i, k])}),
                "factor_return_forecast_used": json.dumps({factor_names[k]: float(f_pred[t, k]) for k in range(len(factor_names)) if np.isfinite(f_pred[t, k])}),
                "factor_contributions": json.dumps({factor_names[k]: float(c[k]) for k in range(len(factor_names)) if np.isfinite(c[k])}),
                "residual_score": np.nan,
                "outlier_flag": False,
                "liquidity_flag": liquidity_flag,
                "risk_flag": risk_flag,
                "reason_text": reason,
            })
    return pd.DataFrame(rows)


def compute_residuals(dates: pd.DatetimeIndex, tickers: list[str], X: np.ndarray, f: np.ndarray, r: np.ndarray | None, z_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if r is None:
        return pd.DataFrame(columns=["date", "ticker", "residual", "residual_zscore"]), pd.DataFrame(columns=["ticker", "date", "residual", "residual_zscore", "side_implication", "excluded_from_trading", "reason"])
    pred = np.einsum("tnk,tk->tn", np.where(np.isfinite(X), X, 0.0), np.where(np.isfinite(f), f, 0.0))
    resid = r - pred
    mean = np.nanmean(resid, axis=1)[:, None]
    sd = np.nanstd(resid, axis=1)[:, None]
    z = (resid - mean) / np.where(sd > 1e-12, sd, np.nan)
    rows = []
    out = []
    for t, date in enumerate(dates):
        for i, ticker in enumerate(tickers):
            rows.append({"date": date, "ticker": ticker, "residual": resid[t, i], "residual_zscore": z[t, i]})
            if np.isfinite(z[t, i]) and abs(z[t, i]) >= z_threshold:
                out.append({
                    "ticker": ticker,
                    "date": date,
                    "residual": resid[t, i],
                    "residual_zscore": z[t, i],
                    "side_implication": "positive_outlier" if z[t, i] > 0 else "negative_outlier",
                    "excluded_from_trading": False,
                    "reason": "Reported only; outlier policy does not trade residual outliers automatically.",
                })
    return pd.DataFrame(rows), pd.DataFrame(out)


def write_residual_heatmap(residuals: pd.DataFrame, path: Path, max_tickers: int = 60, lookback: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        if residuals.empty:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.text(0.5, 0.5, "No residual data", ha="center", va="center")
            ax.axis("off")
        else:
            zwide = residuals.pivot(index="ticker", columns="date", values="residual_zscore")
            score = zwide.abs().max(axis=1).sort_values(ascending=False)
            zwide = zwide.loc[score.head(max_tickers).index, zwide.columns[-lookback:]]
            fig, ax = plt.subplots(figsize=(12, max(4, min(14, len(zwide) * 0.18))))
            im = ax.imshow(zwide.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
            ax.set_yticks(range(len(zwide.index)))
            ax.set_yticklabels(zwide.index, fontsize=6)
            ax.set_xticks(range(len(zwide.columns)))
            ax.set_xticklabels([pd.Timestamp(x).date().isoformat() for x in zwide.columns], rotation=90, fontsize=6)
            fig.colorbar(im, ax=ax, label="Residual z-score")
            fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception:
        # 1x1 transparent PNG fallback keeps report generation deterministic on minimal environments.
        path.write_bytes(bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a49444154789c636000000200015d0b2a0b0000000049454e44ae426082"))


def build_quantile_summary(dates: pd.DatetimeIndex, alpha: np.ndarray, realized: np.ndarray | None, residuals: pd.DataFrame, weights: np.ndarray, dollar_volume: np.ndarray | None, cfg: DailyBacktestConfig, filter_info: pd.DataFrame) -> pd.DataFrame:
    rows = []
    resid_map = residuals.set_index(["date", "ticker"]) if not residuals.empty else None
    for t, date in enumerate(dates):
        s = pd.Series(alpha[t])
        finite = s.replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty:
            continue
        buckets = pd.qcut(finite, 10, labels=False, duplicates="drop") + 1 if len(finite) >= 10 else pd.Series(1, index=finite.index)
        for bucket in sorted(buckets.dropna().unique()):
            idx = buckets[buckets == bucket].index.to_numpy(dtype=int)
            rows.append({
                "signal_date": date,
                "bucket": int(bucket),
                "alpha_min": float(np.nanmin(alpha[t, idx])),
                "alpha_max": float(np.nanmax(alpha[t, idx])),
                "number_of_stocks": int(len(idx)),
                "average_alpha": float(np.nanmean(alpha[t, idx])),
                "average_realized_next_day_return": float(np.nanmean(realized[t, idx])) if realized is not None and np.isfinite(realized[t, idx]).any() else np.nan,
                "average_residual": np.nan,
                "long_selected_count": int((weights[t, idx] > 0).sum()),
                "short_selected_count": int((weights[t, idx] < 0).sum()),
                "average_liquidity": float(np.nanmean(dollar_volume[t, idx])) if dollar_volume is not None else np.nan,
                "average_volatility": np.nan,
                "long_threshold_alpha": filter_info.loc[t, "long_threshold_alpha"] if t in filter_info.index else np.nan,
                "short_threshold_alpha": filter_info.loc[t, "short_threshold_alpha"] if t in filter_info.index else np.nan,
                "filtered_out_liquidity": filter_info.loc[t, "filtered_liquidity"] if t in filter_info.index else np.nan,
                "filtered_out_missing_data": filter_info.loc[t, "filtered_missing_or_untradable"] if t in filter_info.index else np.nan,
                "filtered_out_alpha_threshold": filter_info.loc[t, "filtered_alpha_threshold"] if t in filter_info.index else np.nan,
                "cost_adjusted_alpha_threshold": filter_info.loc[t, "cost_adjusted_alpha_threshold"] if t in filter_info.index else np.nan,
                "trading_cost_return": filter_info.loc[t, "trading_cost_return"] if t in filter_info.index else np.nan,
                "min_net_alpha_after_cost": filter_info.loc[t, "min_net_alpha_after_cost"] if t in filter_info.index else np.nan,
                "final_number_long": filter_info.loc[t, "final_number_long"] if t in filter_info.index else np.nan,
                "final_number_short": filter_info.loc[t, "final_number_short"] if t in filter_info.index else np.nan,
            })
    return pd.DataFrame(rows)


def run_close_to_next_open_backtest(cfg: DailyBacktestConfig) -> DailyBacktestResult:
    cfg.validate()
    data = load_processed_inputs(cfg.input_dir)
    X, f, saved_r, tradable = data["X"], data["f"], data["r"], data["tradable"]
    dates, tickers, factor_names = data["dates"], data["tickers"], data["factor_names"]
    f_pred = predict_factor_returns(f, method=cfg.forecast_method, lookback=cfg.lookback, ewma_halflife=cfg.ewma_halflife, min_periods=cfg.min_periods)
    warnings = check_timing(dates, f, f_pred, cfg.forecast_method)
    panels = load_daily_price_panels(cfg.daily_bars_path, dates, tickers)
    bars_start, bars_end = daily_bars_date_range(cfg.daily_bars_path)
    dollar_volume = None
    if {"close", "volume"}.issubset(panels):
        dollar_volume = panels["close"].to_numpy(dtype=float) * panels["volume"].to_numpy(dtype=float)
    alpha, contrib = compute_alpha_and_contributions(X, f_pred)
    weights, filter_info = build_daily_weights(alpha, tradable, dollar_volume, cfg)
    previous_weights = np.vstack([np.zeros((1, weights.shape[1])), weights[:-1]])
    realized, price_frame, price_warnings = compute_execution_returns(weights, dates, panels, saved_r, cfg)
    warnings.extend(price_warnings)
    returns, turnover, costs = portfolio_pnl(weights, realized, cfg)
    equity = np.cumprod(1.0 + np.where(np.isfinite(returns), returns, 0.0))
    daily = pd.DataFrame({
        "signal_date": dates,
        "execution_date": pd.Series(dates).shift(-1).to_numpy(),
        "signal_timestamp": [pd.Timestamp(d).replace(hour=16).isoformat() for d in dates],
        "execution_timestamp": [pd.Timestamp(d).replace(hour=9, minute=30).isoformat() if pd.notna(d) else None for d in pd.Series(dates).shift(-1)],
        "portfolio_return": returns,
        "cumulative_return": equity - 1.0,
        "turnover": turnover,
        "transaction_cost": costs,
        "gross_exposure": np.abs(weights).sum(axis=1),
        "net_exposure": weights.sum(axis=1),
        "long_exposure": np.clip(weights, 0, None).sum(axis=1),
        "short_exposure": np.clip(weights, None, 0).sum(axis=1),
        "number_long": (weights > 0).sum(axis=1),
        "number_short": (weights < 0).sum(axis=1),
    })
    alpha_rankings = build_alpha_rankings(dates, tickers, factor_names, alpha, contrib, X, f_pred, weights, previous_weights, dollar_volume, cfg)
    factor_report = _factor_frame(dates, factor_names, f, f_pred)
    residuals, outliers = compute_residuals(dates, tickers, X, f, saved_r, cfg.residual_outlier_z_threshold)
    quantile = build_quantile_summary(dates, alpha, realized, residuals, weights, dollar_volume, cfg, filter_info.set_index("row"))
    order_rows = alpha_rankings[alpha_rankings["side"].ne("")].copy()
    prices = price_frame.copy()
    prices["ticker"] = [tickers[i] for i in prices["ticker_index"]]
    price_lookup = prices.set_index(["signal_date", "ticker"])
    for col in ["signal_price", "execution_price", "exit_price", "execution_date", "exit_date"]:
        order_rows[col] = [price_lookup.loc[(pd.Timestamp(r.signal_date), r.ticker), col] if (pd.Timestamp(r.signal_date), r.ticker) in price_lookup.index else np.nan for r in order_rows.itertuples()]
    order_rows["holding_period_days"] = cfg.holding_period_days
    execution_quality = order_rows.rename(columns={"order_weight_delta": "planned_weight_delta", "execution_price": "planned_price"}).copy()
    execution_quality["date"] = execution_quality["execution_date"]
    execution_quality["planned_quantity"] = np.where(execution_quality["planned_price"].astype(float).gt(0), (execution_quality["planned_weight_delta"].abs() * cfg.initial_equity / execution_quality["planned_price"]).fillna(0).astype(int), 0)
    execution_quality["filled_quantity"] = execution_quality["planned_quantity"]
    execution_quality["fill_ratio"] = 1.0
    execution_quality["average_fill_price"] = execution_quality["planned_price"]
    execution_quality["arrival_price"] = execution_quality["planned_price"]
    execution_quality["close_price"] = execution_quality["exit_price"]
    execution_quality["slippage_bps"] = cfg.slippage_bps
    execution_quality["transaction_cost"] = execution_quality["planned_weight_delta"].abs() * cfg.initial_equity * (cfg.transaction_cost_bps / 10_000.0)
    execution_quality["order_status"] = "filled"
    execution_quality["rejected_reason"] = ""
    execution_quality = execution_quality[["date", "ticker", "side", "planned_quantity", "filled_quantity", "fill_ratio", "planned_price", "average_fill_price", "arrival_price", "close_price", "slippage_bps", "transaction_cost", "order_status", "rejected_reason"]]
    target_positions = alpha_rankings[[
        "signal_date", "ticker", "target_weight", "current_weight", "order_weight_delta",
        "alpha_score", "trading_cost_bps", "net_alpha_after_cost_bps",
        "min_net_alpha_after_cost_bps", "passes_cost_threshold",
        "alpha_rank", "quantile_bucket", "reason_text",
    ]].copy()
    current_positions = target_positions.rename(columns={"current_weight": "weight"})[["signal_date", "ticker", "weight"]]
    top_abs = pd.Series(weights[-1], index=tickers).abs().sort_values(ascending=False).head(10)
    risk_summary = {
        "gross_exposure": float(np.abs(weights[-1]).sum()) if len(weights) else 0.0,
        "net_exposure": float(weights[-1].sum()) if len(weights) else 0.0,
        "long_exposure": float(np.clip(weights[-1], 0, None).sum()) if len(weights) else 0.0,
        "short_exposure": float(np.clip(weights[-1], None, 0).sum()) if len(weights) else 0.0,
        "number_long": int((weights[-1] > 0).sum()) if len(weights) else 0,
        "number_short": int((weights[-1] < 0).sum()) if len(weights) else 0,
        "max_position_weight": float(np.nanmax(np.abs(weights[-1]))) if len(weights) else 0.0,
        "top_10_position_weights": {k: float(weights[-1, tickers.index(k)]) for k in top_abs.index},
        "sector_exposures": {},
        "beta_estimate": None,
        "expected_turnover": float(turnover[-1]) if len(turnover) else 0.0,
        "realized_turnover": float(turnover[-1]) if len(turnover) else 0.0,
        "cash": None,
        "leverage": float(np.abs(weights[-1]).sum()) if len(weights) else 0.0,
        "margin_usage": None,
        "liquidity_warnings": [],
    }
    finite_by_factor = np.isfinite(X).mean(axis=(0, 1))
    metadata = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": _safe_git_hash(),
        "data_start_date": dates[0].date().isoformat() if len(dates) else None,
        "data_end_date": dates[-1].date().isoformat() if len(dates) else None,
        "signal_date": dates[-2].date().isoformat() if len(dates) > 1 else None,
        "execution_date": dates[-1].date().isoformat() if len(dates) > 1 else None,
        "universe_size_raw": len(tickers),
        "universe_size_after_filters": int(np.isfinite(alpha[-2]).sum()) if len(alpha) > 1 else 0,
        "factor_names": factor_names,
        "model_config": {"forecast_method": cfg.forecast_method, "lookback": cfg.lookback, "ewma_halflife": cfg.ewma_halflife},
        "backtest_config": asdict(cfg) | {"input_dir": str(cfg.input_dir), "out_dir": str(cfg.out_dir), "daily_bars_path": str(cfg.daily_bars_path) if cfg.daily_bars_path else None, "reports_dir": str(cfg.reports_dir)},
        "data_source": str(cfg.daily_bars_path) if cfg.daily_bars_path else str(cfg.input_dir),
        "input_files": {
            "input_dir": str(cfg.input_dir),
            "daily_bars_path": str(cfg.daily_bars_path) if cfg.daily_bars_path else None,
            "processed_dates_start": dates[0].date().isoformat() if len(dates) else None,
            "processed_dates_end": dates[-1].date().isoformat() if len(dates) else None,
            "daily_bars_start": bars_start,
            "daily_bars_end": bars_end,
        },
        "warnings": warnings,
        "data_quality": {
            "missing_factor_coverage": {name: float(1.0 - finite_by_factor[i]) for i, name in enumerate(factor_names)},
            "nan_counts_by_factor": {name: int(np.isnan(X[:, :, i]).sum()) for i, name in enumerate(factor_names)},
            "finite_fraction_by_factor": {name: float(finite_by_factor[i]) for i, name in enumerate(factor_names)},
            "excluded_tickers_latest": int((~tradable[-1]).sum()) if len(tradable) else 0,
            "point_in_time_universe_warning": "Warn: historical backtest uses the saved pipeline universe; verify it is point-in-time before relying on results.",
        },
    }
    return DailyBacktestResult(
        daily=daily,
        trades=order_rows,
        orders=order_rows,
        target_positions=target_positions,
        current_positions=current_positions,
        factor_forecast=factor_report,
        factor_returns=factor_report,
        alpha_rankings=alpha_rankings,
        quantile_summary=quantile,
        residuals=residuals,
        outliers=outliers,
        execution_quality=execution_quality,
        risk_summary=risk_summary,
        metadata=metadata,
        warnings=warnings,
    )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def write_daily_report(result: DailyBacktestResult, report_dir: Path, signal_date: pd.Timestamp | None = None) -> Path:
    if signal_date is None:
        signal_date = pd.Timestamp(result.metadata.get("signal_date") or result.daily["signal_date"].dropna().iloc[-1])
    out = report_dir / pd.Timestamp(signal_date).date().isoformat()
    out.mkdir(parents=True, exist_ok=True)
    latest_orders = result.orders[pd.to_datetime(result.orders["signal_date"]).dt.normalize().eq(pd.Timestamp(signal_date).normalize())] if not result.orders.empty else result.orders
    latest_targets = result.target_positions[pd.to_datetime(result.target_positions["signal_date"]).dt.normalize().eq(pd.Timestamp(signal_date).normalize())] if not result.target_positions.empty else result.target_positions
    latest_alpha = result.alpha_rankings[pd.to_datetime(result.alpha_rankings["signal_date"]).dt.normalize().eq(pd.Timestamp(signal_date).normalize())] if not result.alpha_rankings.empty else result.alpha_rankings
    latest_quantile = result.quantile_summary[pd.to_datetime(result.quantile_summary["signal_date"]).dt.normalize().eq(pd.Timestamp(signal_date).normalize())] if not result.quantile_summary.empty and "signal_date" in result.quantile_summary else result.quantile_summary
    latest_orders.to_csv(out / "orders.csv", index=False)
    latest_targets.to_csv(out / "target_positions.csv", index=False)
    result.current_positions.to_csv(out / "current_positions.csv", index=False)
    result.factor_returns.to_csv(out / "factor_returns.csv", index=False)
    result.factor_forecast.to_csv(out / "factor_forecast.csv", index=False)
    latest_alpha.to_csv(out / "alpha_rankings.csv", index=False)
    latest_quantile.to_csv(out / "quantile_summary.csv", index=False)
    result.residuals.to_csv(out / "residuals.csv", index=False)
    result.outliers.to_csv(out / "outliers.csv", index=False)
    result.execution_quality.to_csv(out / "execution_quality.csv", index=False)
    metadata = dict(result.metadata)
    row = result.daily[pd.to_datetime(result.daily["signal_date"]).dt.normalize().eq(pd.Timestamp(signal_date).normalize())]
    metadata["signal_date"] = pd.Timestamp(signal_date).date().isoformat()
    if not row.empty and pd.notna(row.iloc[0]["execution_date"]):
        metadata["execution_date"] = pd.Timestamp(row.iloc[0]["execution_date"]).date().isoformat()
    metadata["report_directory"] = str(out)
    (out / "risk_summary.json").write_text(json.dumps(result.risk_summary, indent=2, default=_json_default), encoding="utf-8")
    (out / "pipeline_metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")
    write_residual_heatmap(result.residuals, out / "residual_heatmap.png")
    # Placeholder dashboard uses the same robust image path until richer plotting is needed.
    write_residual_heatmap(result.residuals, out / "factor_dashboard.png")
    buys = latest_orders[latest_orders["side"].isin(["BUY", "COVER"])].sort_values("alpha_score", ascending=False).head(10) if not latest_orders.empty else pd.DataFrame()
    sells = latest_orders[latest_orders["side"].isin(["SELL", "SHORT"])].sort_values("alpha_score", ascending=True).head(10) if not latest_orders.empty else pd.DataFrame()
    def md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return ""
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for row in df.itertuples(index=False):
            lines.append("| " + " | ".join("" if pd.isna(x) else str(x) for x in row) + " |")
        return "\n".join(lines)

    qmd = md_table(latest_quantile[["bucket", "number_of_stocks", "average_alpha", "long_selected_count", "short_selected_count"]]) if not latest_quantile.empty else "No quantile data."
    summary = [
        f"# Daily Factor Report - {pd.Timestamp(signal_date).date().isoformat()}",
        "",
        "## Model Decision",
        f"Generated close-based signals for execution on the next trading day. Proposed orders: {len(latest_orders)}. Expected turnover: {result.risk_summary.get('expected_turnover', 0):.4f}.",
        f"Orders require predicted alpha net of estimated trading cost to exceed {metadata.get('backtest_config', {}).get('min_net_alpha_after_cost_bps', 0):.2f} bps.",
        "",
        "## Buy Candidates",
        md_table(buys[["ticker", "side", "target_weight", "alpha_score", "reason_text"]]) if not buys.empty else "No buy candidates.",
        "",
        "## Sell / Short Candidates",
        md_table(sells[["ticker", "side", "target_weight", "alpha_score", "reason_text"]]) if not sells.empty else "No sell/short candidates.",
        "",
        "## Quantile Summary",
        qmd,
        "",
        "## Risk / Exposure",
        f"Gross: {result.risk_summary.get('gross_exposure', 0):.4f}; Net: {result.risk_summary.get('net_exposure', 0):.4f}; Long names: {result.risk_summary.get('number_long', 0)}; Short names: {result.risk_summary.get('number_short', 0)}.",
        "",
        "## Data Quality",
        f"Warnings: {', '.join(result.warnings) if result.warnings else 'none'}. Missing and finite factor coverage are recorded in pipeline_metadata.json.",
        "",
        "## Timing Convention",
        "Signals use data available at or before signal_date close. Orders are for the next trading day open. No intraday alpha recomputation is part of this workflow.",
    ]
    (out / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return out
