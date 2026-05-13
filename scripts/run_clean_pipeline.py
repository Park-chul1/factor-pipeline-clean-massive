from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factor_pipeline.config import get_api_key
from factor_pipeline.massive_client import MassiveClient, download_nasdaq_tickers, download_grouped_daily_range, download_financials
from factor_pipeline.panel import bars_long_to_panel, build_tradable_mask, compute_forward_returns
from factor_pipeline.price_volume_factors import build_price_volume_factors
from factor_pipeline.fundamental_factors import flatten_financials, build_ttm_financials, fundamentals_to_daily, build_fundamental_factors
from factor_pipeline.preprocess import apply_universe_mask, build_exposure_tensor
from factor_pipeline.estimation import estimate_factor_returns
from factor_pipeline.diagnostics import array_summary, factor_diagnostics, save_json


def parse_args():
    p = argparse.ArgumentParser(description="Clean Massive NASDAQ price+fundamental factor pipeline")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--exchange", default="XNAS")
    p.add_argument("--ticker-status", default="all", choices=["all", "active", "inactive"],
                   help="all includes inactive metadata when available; active reproduces active-only runs")
    p.add_argument("--max-tickers", type=int, default=None)
    p.add_argument("--universe-rank-by", default="dollar_volume", choices=["ticker", "dollar_volume"],
                   help="ticker = alphabetical; dollar_volume = top names by median close*volume")
    p.add_argument("--universe-rank-window-days", type=int, default=126,
                   help="Use the first or last N trading days in the requested range to rank tickers by dollar volume")
    p.add_argument("--universe-rank-anchor", default="end", choices=["start", "end"],
                   help="Choose whether dollar-volume ranking is anchored at the range start or end")
    p.add_argument("--out-dir", default="data/processed_clean")
    p.add_argument("--cache-dir", default="data/cache_clean")
    p.add_argument("--api-cache-dir", default=None,
                   help="Raw API response cache directory; defaults to <cache-dir>/api_responses")
    p.add_argument("--min-names", type=int, default=30)
    p.add_argument("--ridge", default="1e-4",
                   help="Fixed ridge lambda, or 'auto' to select lambda date-by-date with GCV")
    p.add_argument("--ridge-grid", default=None,
                   help="Comma-separated lambda grid for --ridge auto, e.g. 1e-6,1e-5,1e-4,1e-3")
    p.add_argument("--ridge-solver", default="qr", choices=["qr", "normal"],
                   help="qr solves the augmented ridge least-squares system")
    p.add_argument("--estimation-workers", type=int, default=1,
                   help="Parallel worker threads for date-by-date factor return estimation")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument(
        "--max-abs-forward-return",
        type=float,
        default=1.0,
        help="Drop forward returns whose absolute value exceeds this threshold; use <=0 to disable",
    )
    p.add_argument("--financial-timeframe", default="quarterly", choices=["ttm", "quarterly", "annual"],
                   help="quarterly builds historical point-in-time TTM flows; ttm keeps vendor TTM rows")
    p.add_argument("--financial-limit", type=int, default=100)
    p.add_argument("--financial-workers", type=int, default=1,
                   help="Parallel workers for per-ticker financial downloads")
    p.add_argument("--financial-lookback-days", type=int, default=550,
                   help="Extra report-period history before --start used to seed quarterly TTM values")
    p.add_argument("--financial-lag-days", type=int, default=60, help="Used only when filing_date is missing")
    p.add_argument("--ttm-min-quarters", type=int, default=4,
                   help="Minimum quarterly rows required to build a TTM flow value")
    p.add_argument("--min-factor-coverage", type=float, default=0.02, help="Drop factors with lower processed finite coverage")
    p.add_argument("--max-factor-corr", type=float, default=0.999,
                   help="Drop later factors whose abs correlation with an earlier kept factor is at least this value; use 0 to disable")
    p.add_argument("--corr-min-overlap", type=int, default=100,
                   help="Minimum finite pair observations required before applying the factor correlation filter")
    p.add_argument("--no-fill-missing-exposures", action="store_true", help="Keep NaNs in X after preprocessing instead of neutral 0 fill")
    p.add_argument("--sleep", type=float, default=0.15)
    p.add_argument("--no-cache", action="store_true", help="Disable parquet dataset caches")
    p.add_argument("--no-api-cache", action="store_true", help="Disable raw API response cache")
    p.add_argument("--run-diagnostics", action="store_true", help="Run post-pipeline ridge diagnostics")
    p.add_argument("--diagnostics-lambdas", default="0,1e-8,1e-6,1e-4,1e-2,1e-1",
                   help="Comma-separated lambda values for diagnostics")
    p.add_argument("--diagnostics-output-dir", default=None,
                   help="Output directory for diagnostics. Default: <out-dir>/diagnostics")
    return p.parse_args()

def ticker_status_to_active(status: str) -> bool | None:
    return {"all": None, "active": True, "inactive": False}[status]


def download_tickers_for_status(client: MassiveClient, exchange: str, status: str) -> pd.DataFrame:
    if status != "all":
        return download_nasdaq_tickers(
            client,
            exchange=exchange,
            active=ticker_status_to_active(status),
        )

    frames = [
        download_nasdaq_tickers(client, exchange=exchange, active=True),
        download_nasdaq_tickers(client, exchange=exchange, active=False),
    ]
    frames = [df for df in frames if not df.empty]
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("ticker", keep="first")
        .sort_values("ticker")
        .reset_index(drop=True)
    )


def read_or_build(path: Path, use_cache: bool, builder):
    if use_cache and path.exists():
        return pd.read_parquet(path)
    df = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def parse_ridge(value: str) -> float | str:
    return "auto" if str(value).lower() == "auto" else float(value)


def parse_ridge_grid(value: str | None) -> list[float] | None:
    if not value:
        return None
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True); cache_dir.mkdir(parents=True, exist_ok=True)
    use_cache = not args.no_cache
    api_cache_dir = Path(args.api_cache_dir) if args.api_cache_dir else cache_dir
    api_cache_namespace = "" if args.api_cache_dir else "api_responses"
    api_cache_display = api_cache_dir if not api_cache_namespace else api_cache_dir / api_cache_namespace
    client = MassiveClient(
        get_api_key(),
        sleep_sec=args.sleep,
        cache_dir=api_cache_dir,
        use_cache=not args.no_api_cache,
        cache_namespace=api_cache_namespace,
    )

    tickers_path = cache_dir / f"tickers_{args.exchange}_{args.ticker_status}.parquet"
    tickers_df = read_or_build(
        tickers_path,
        use_cache,
        lambda: download_tickers_for_status(client, args.exchange, args.ticker_status),
    )
    candidate_tickers = tickers_df["ticker"].dropna().astype(str).sort_values().tolist()

    bars_path = cache_dir / f"grouped_daily_{args.start}_{args.end}.parquet"
    bars = read_or_build(bars_path, use_cache, lambda: download_grouped_daily_range(client, args.start, args.end))
    bars = bars[bars["ticker"].isin(candidate_tickers)].copy()

    if args.max_tickers and args.universe_rank_by == "dollar_volume":
        rank_bars = bars.copy()
        rank_bars["date"] = pd.to_datetime(rank_bars["date"]).dt.normalize()
        rank_dates = sorted(rank_bars["date"].dropna().unique())
        if args.universe_rank_window_days and len(rank_dates) > args.universe_rank_window_days:
            if args.universe_rank_anchor == "start":
                rank_dates = rank_dates[: args.universe_rank_window_days]
            else:
                rank_dates = rank_dates[-args.universe_rank_window_days:]
            rank_bars = rank_bars[rank_bars["date"].isin(rank_dates)].copy()

        rank_bars["dollar_volume"] = rank_bars["close"].astype(float) * rank_bars["volume"].astype(float)
        ranked = (
            rank_bars.groupby("ticker")["dollar_volume"]
            .median()
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .sort_values(ascending=False)
        )
        tickers = ranked.head(args.max_tickers).index.astype(str).tolist()
    else:
        tickers = candidate_tickers
        if args.max_tickers:
            tickers = tickers[: args.max_tickers]

    tickers_df = tickers_df[tickers_df["ticker"].isin(tickers)].copy()
    bars = bars[bars["ticker"].isin(tickers)].copy()
    panel = bars_long_to_panel(bars, tickers=tickers)
    dates = panel["adj_close"].index
    tradable_mask_df = build_tradable_mask(panel)
    tradable_mask = tradable_mask_df.to_numpy(dtype=bool)

    financial_report_start = (
        pd.Timestamp(args.start) - pd.Timedelta(days=args.financial_lookback_days)
    ).date().isoformat()
    fin_path = cache_dir / (
        f"financials_{args.financial_timeframe}_{financial_report_start}_{args.end}_"
        f"{args.financial_limit}_{len(tickers)}.parquet"
    )
    def fetch_financials_one(i: int, t: str) -> pd.DataFrame:
        print(f"financials {i}/{len(tickers)} {t}", flush=True)
        try:
            return download_financials(
                client,
                t,
                timeframe=args.financial_timeframe,
                limit=args.financial_limit,
                period_of_report_date_gte=financial_report_start,
                period_of_report_date_lte=args.end,
            )
        except Exception as e:
            print(f"WARN financials failed {t}: {e}", flush=True)
            return pd.DataFrame()

    def build_financials():
        frames = []
        if args.financial_workers <= 1:
            for i, t in enumerate(tickers, 1):
                df = fetch_financials_one(i, t)
                if not df.empty:
                    frames.append(df)
        else:
            with ThreadPoolExecutor(max_workers=args.financial_workers) as executor:
                futures = [
                    executor.submit(fetch_financials_one, i, t)
                    for i, t in enumerate(tickers, 1)
                ]
                for future in as_completed(futures):
                    df = future.result()
                    if not df.empty:
                        frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    fin_raw = read_or_build(fin_path, use_cache, build_financials)
    fin_flat = flatten_financials(fin_raw, lag_days=args.financial_lag_days) if not fin_raw.empty else pd.DataFrame()
    if not fin_flat.empty:
        fin_flat.to_parquet(out_dir / "financials_flat.parquet", index=False)
    if args.financial_timeframe == "quarterly" and not fin_flat.empty:
        fin_model = build_ttm_financials(fin_flat, min_quarters=args.ttm_min_quarters)
        if not fin_model.empty:
            fin_model.to_parquet(out_dir / "financials_ttm.parquet", index=False)
    else:
        fin_model = fin_flat
    fund_daily = fundamentals_to_daily(fin_model, dates, tickers) if not fin_model.empty else {}

    pv_factors = build_price_volume_factors(panel)
    fundamental_factors = build_fundamental_factors(fund_daily, panel["adj_close"]) if fund_daily else {}
    factors = apply_universe_mask({**pv_factors, **fundamental_factors}, tradable_mask_df)

    X, factor_names, preprocess_diag = build_exposure_tensor(
        factors,
        min_names=args.min_names,
        min_factor_coverage=args.min_factor_coverage,
        fill_missing=not args.no_fill_missing_exposures,
        max_factor_corr=args.max_factor_corr,
        corr_min_overlap=args.corr_min_overlap,
    )
    max_abs_forward_return = args.max_abs_forward_return if args.max_abs_forward_return > 0 else None
    r_df = compute_forward_returns(
        panel["adj_close"],
        horizon=args.horizon,
        max_abs_return=max_abs_forward_return,
    ).where(tradable_mask_df)
    r = r_df.to_numpy(dtype=float)
    ridge_value = parse_ridge(args.ridge)
    f, ridge_diag = estimate_factor_returns(
        X,
        r,
        min_names=args.min_names,
        ridge=ridge_value,
        universe_mask=tradable_mask,
        ridge_grid=parse_ridge_grid(args.ridge_grid),
        ridge_selection="gcv" if ridge_value == "auto" else "fixed",
        solver=args.ridge_solver,
        n_jobs=args.estimation_workers,
        return_diagnostics=True,
    )

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "r.npy", r)
    np.save(out_dir / "factor_returns.npy", f)
    np.save(out_dir / "tradable_mask.npy", tradable_mask)
    tickers_df.to_csv(out_dir / "tickers.csv", index=False)
    pd.DataFrame({"date": dates}).to_csv(out_dir / "dates.csv", index=False)
    pd.DataFrame({"factor": factor_names}).to_csv(out_dir / "factor_names.csv", index=False)
    factor_diagnostics(factors).to_csv(out_dir / "factor_diagnostics_raw.csv", index=False)
    preprocess_diag.to_csv(out_dir / "factor_diagnostics_preprocessed.csv", index=False)
    ridge_diag.insert(0, "date", dates.to_numpy())
    ridge_diag.to_csv(out_dir / "ridge_diagnostics.csv", index=False)

    selected = ridge_diag["selected_ridge"].replace([np.inf, -np.inf], np.nan).dropna()
    ridge_summary = {
        "mode": "gcv" if ridge_value == "auto" else "fixed",
        "solver": args.ridge_solver,
        "fixed_lambda": None if ridge_value == "auto" else float(ridge_value),
        "grid": parse_ridge_grid(args.ridge_grid),
        "finite_dates": int(selected.size),
        "median_selected_lambda": float(selected.median()) if not selected.empty else None,
        "min_selected_lambda": float(selected.min()) if not selected.empty else None,
        "max_selected_lambda": float(selected.max()) if not selected.empty else None,
    }

    summary = {
        "data_policy": {
            "prices": "Massive grouped daily bars adjusted=true; daily OHLCV only available after market date.",
            "fundamentals": "Massive financials endpoint; quarterly rows are converted to point-in-time TTM flow fields before daily forward-fill. Each filing is available from filing_date if provided, otherwise end_date + financial_lag_days.",
            "look_ahead_bias_control": "Fundamentals are point-in-time merged by available_date then forward-filled. Future filings are never backfilled into earlier trading dates. Returns use adj_close[t+h]/adj_close[t]-1 and outlier forward returns are dropped when max_abs_forward_return is set.",
            "universe": f"Massive ticker metadata exchange={args.exchange}, ticker_status={args.ticker_status}; selected by universe_rank_by={args.universe_rank_by}, max_tickers={args.max_tickers}.",
            "tradable_mask": "date x ticker mask from finite positive adjusted close and positive volume; applied before factor preprocessing and during regression.",
            "api_cache": f"Raw Massive API JSON responses are cached under {api_cache_display} unless --no-api-cache is set.",
        },
        "params": vars(args),
        "n_dates": len(dates),
        "n_tickers": len(tickers),
        "n_financial_rows_raw": int(len(fin_raw)),
        "n_financial_rows_flat": int(len(fin_flat)),
        "n_financial_rows_model": int(len(fin_model)),
        "n_factors_raw": len(factors),
        "n_factors_kept": len(factor_names),
        "n_factors": len(factor_names),
        "max_abs_forward_return": max_abs_forward_return,
        "price_volume_factors": list(pv_factors.keys()),
        "fundamental_factors": list(fundamental_factors.keys()),
        "X": array_summary("X", X),
        "r": array_summary("r", r),
        "factor_returns": array_summary("factor_returns", f),
        "ridge": ridge_summary,
        "estimation_workers": args.estimation_workers,
        "tradable_mask": {
            "shape": list(tradable_mask.shape),
            "true_count": int(tradable_mask.sum()),
            "true_ratio": float(tradable_mask.mean()) if tradable_mask.size else 0.0,
        },
        "factor_filtering": {
            "min_factor_coverage": args.min_factor_coverage,
            "max_factor_corr": args.max_factor_corr,
            "corr_min_overlap": args.corr_min_overlap,
            "fill_missing_exposures_after_zscore": not args.no_fill_missing_exposures,
            "kept_factors": factor_names,
            "dropped_factors": preprocess_diag.loc[~preprocess_diag["kept"], "factor"].tolist() if not preprocess_diag.empty else [],
            "high_corr_dropped_factors": preprocess_diag.loc[preprocess_diag["drop_reason"].eq("high_corr"), "factor"].tolist() if not preprocess_diag.empty else [],
        },
    }
    save_json(summary, out_dir / "pipeline_summary.json")
    print(f"done: {out_dir}")
    print(f"X={X.shape}, r={r.shape}, factor_returns={f.shape}, factors={len(factor_names)}")

    # Optional: run post-pipeline ridge diagnostics
    if args.run_diagnostics:
        from factor_pipeline.diagnostics_ridge import run_estimation_diagnostics
        diagnostics_out = args.diagnostics_output_dir or str(out_dir / "diagnostics")
        lambdas = [float(x.strip()) for x in args.diagnostics_lambdas.split(",")]
        run_estimation_diagnostics(
            X=X,
            r=r,
            valid_mask=tradable_mask,
            lambdas=lambdas,
            factor_names=factor_names,
            output_dir=diagnostics_out,
            min_names=args.min_names,
        )


if __name__ == "__main__":
    main()
