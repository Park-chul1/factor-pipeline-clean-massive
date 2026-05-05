from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factor_pipeline.config import get_api_key
from factor_pipeline.massive_client import MassiveClient, download_nasdaq_tickers, download_grouped_daily_range, download_financials
from factor_pipeline.panel import bars_long_to_panel, compute_forward_returns
from factor_pipeline.price_volume_factors import build_price_volume_factors
from factor_pipeline.fundamental_factors import flatten_financials, fundamentals_to_daily, build_fundamental_factors
from factor_pipeline.preprocess import build_exposure_tensor
from factor_pipeline.estimation import estimate_factor_returns
from factor_pipeline.diagnostics import array_summary, factor_diagnostics, save_json


def parse_args():
    p = argparse.ArgumentParser(description="Clean Massive NASDAQ price+fundamental factor pipeline")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--exchange", default="XNAS")
    p.add_argument("--max-tickers", type=int, default=None)
    p.add_argument("--universe-rank-by", default="dollar_volume", choices=["ticker", "dollar_volume"],
                   help="ticker = alphabetical; dollar_volume = top names by median close*volume")
    p.add_argument("--universe-rank-window-days", type=int, default=126,
                   help="Use the last N trading days in the requested range to rank tickers by dollar volume")
    p.add_argument("--out-dir", default="data/processed_clean")
    p.add_argument("--cache-dir", default="data/cache_clean")
    p.add_argument("--min-names", type=int, default=30)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--financial-timeframe", default="ttm", choices=["ttm", "quarterly", "annual"])
    p.add_argument("--financial-limit", type=int, default=50)
    p.add_argument("--financial-lag-days", type=int, default=60, help="Used only when filing_date is missing")
    p.add_argument("--min-factor-coverage", type=float, default=0.02, help="Drop factors with lower processed finite coverage")
    p.add_argument("--no-fill-missing-exposures", action="store_true", help="Keep NaNs in X after preprocessing instead of neutral 0 fill")
    p.add_argument("--sleep", type=float, default=0.15)
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def read_or_build(path: Path, use_cache: bool, builder):
    if use_cache and path.exists():
        return pd.read_parquet(path)
    df = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True); cache_dir.mkdir(parents=True, exist_ok=True)
    use_cache = not args.no_cache
    client = MassiveClient(get_api_key(), sleep_sec=args.sleep)

    tickers_path = cache_dir / f"tickers_{args.exchange}.parquet"
    tickers_df = read_or_build(tickers_path, use_cache, lambda: download_nasdaq_tickers(client, exchange=args.exchange))
    candidate_tickers = tickers_df["ticker"].dropna().astype(str).sort_values().tolist()

    bars_path = cache_dir / f"grouped_daily_{args.start}_{args.end}.parquet"
    bars = read_or_build(bars_path, use_cache, lambda: download_grouped_daily_range(client, args.start, args.end))
    bars = bars[bars["ticker"].isin(candidate_tickers)].copy()

    if args.max_tickers and args.universe_rank_by == "dollar_volume":
        rank_bars = bars.copy()
        rank_bars["date"] = pd.to_datetime(rank_bars["date"]).dt.normalize()
        rank_dates = sorted(rank_bars["date"].dropna().unique())
        if args.universe_rank_window_days and len(rank_dates) > args.universe_rank_window_days:
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

    fin_path = cache_dir / f"financials_{args.financial_timeframe}_{args.financial_limit}_{len(tickers)}.parquet"
    def build_financials():
        frames = []
        for i, t in enumerate(tickers, 1):
            print(f"financials {i}/{len(tickers)} {t}", flush=True)
            try:
                df = download_financials(client, t, timeframe=args.financial_timeframe, limit=args.financial_limit)
            except Exception as e:
                print(f"WARN financials failed {t}: {e}", flush=True)
                continue
            if not df.empty:
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    fin_raw = read_or_build(fin_path, use_cache, build_financials)
    fin_flat = flatten_financials(fin_raw, lag_days=args.financial_lag_days) if not fin_raw.empty else pd.DataFrame()
    if not fin_flat.empty:
        fin_flat.to_parquet(out_dir / "financials_flat.parquet", index=False)
    fund_daily = fundamentals_to_daily(fin_flat, dates, tickers) if not fin_flat.empty else {}

    pv_factors = build_price_volume_factors(panel)
    fundamental_factors = build_fundamental_factors(fund_daily, panel["adj_close"]) if fund_daily else {}
    factors = {**pv_factors, **fundamental_factors}

    X, factor_names, preprocess_diag = build_exposure_tensor(
        factors,
        min_names=args.min_names,
        min_factor_coverage=args.min_factor_coverage,
        fill_missing=not args.no_fill_missing_exposures,
    )
    r_df = compute_forward_returns(panel["adj_close"], horizon=args.horizon)
    r = r_df.to_numpy(dtype=float)
    f = estimate_factor_returns(X, r, min_names=args.min_names, ridge=args.ridge)

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "r.npy", r)
    np.save(out_dir / "factor_returns.npy", f)
    tickers_df.to_csv(out_dir / "tickers.csv", index=False)
    pd.DataFrame({"date": dates}).to_csv(out_dir / "dates.csv", index=False)
    pd.DataFrame({"factor": factor_names}).to_csv(out_dir / "factor_names.csv", index=False)
    factor_diagnostics(factors).to_csv(out_dir / "factor_diagnostics_raw.csv", index=False)
    preprocess_diag.to_csv(out_dir / "factor_diagnostics_preprocessed.csv", index=False)

    summary = {
        "data_policy": {
            "prices": "Massive grouped daily bars adjusted=true; daily OHLCV only available after market date.",
            "fundamentals": "Massive financials endpoint; each filing is available from filing_date if provided, otherwise end_date + financial_lag_days.",
            "look_ahead_bias_control": "Fundamentals are point-in-time merged by available_date then forward-filled. Future filings are never backfilled into earlier trading dates. Returns use adj_close[t+h]/adj_close[t]-1.",
            "universe": f"Massive ticker metadata exchange={args.exchange}; selected by universe_rank_by={args.universe_rank_by}, max_tickers={args.max_tickers}.",
        },
        "params": vars(args),
        "n_dates": len(dates),
        "n_tickers": len(tickers),
        "n_factors_raw": len(factors),
        "n_factors_kept": len(factor_names),
        "n_factors": len(factor_names),
        "price_volume_factors": list(pv_factors.keys()),
        "fundamental_factors": list(fundamental_factors.keys()),
        "X": array_summary("X", X),
        "r": array_summary("r", r),
        "factor_returns": array_summary("factor_returns", f),
        "factor_filtering": {
            "min_factor_coverage": args.min_factor_coverage,
            "fill_missing_exposures_after_zscore": not args.no_fill_missing_exposures,
            "kept_factors": factor_names,
            "dropped_factors": preprocess_diag.loc[~preprocess_diag["kept"], "factor"].tolist() if not preprocess_diag.empty else [],
        },
    }
    save_json(summary, out_dir / "pipeline_summary.json")
    print(f"done: {out_dir}")
    print(f"X={X.shape}, r={r.shape}, factor_returns={f.shape}, factors={len(factor_names)}")


if __name__ == "__main__":
    main()
