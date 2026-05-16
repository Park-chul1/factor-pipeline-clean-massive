from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factor_pipeline.config import get_api_key
from factor_pipeline.massive_client import MassiveClient, download_grouped_daily_range


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Massive grouped daily bars and append them to a parquet cache")
    p.add_argument("--base-bars", default=None, help="Optional existing grouped daily parquet to append/update")
    p.add_argument("--out-bars", required=True, help="Combined grouped daily parquet output")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--cache-dir", default="data/api_cache")
    p.add_argument("--sleep", type=float, default=0.02)
    p.add_argument("--no-api-cache", action="store_true")
    p.add_argument("--tickers-csv", default=None, help="Optional ticker universe filter")
    p.add_argument("--active-only", action="store_true", help="If tickers-csv has active, keep active rows only")
    p.add_argument("--unadjusted", action="store_true", help="Fetch raw, unadjusted OHLCV bars for execution and price/liquidity filters")
    return p.parse_args()


def load_ticker_filter(path: str | None, active_only: bool) -> set[str] | None:
    if path is None:
        return None
    tickers = pd.read_csv(path)
    if "ticker" not in tickers.columns:
        raise ValueError(f"{path} must contain a ticker column")
    if active_only and "active" in tickers.columns:
        tickers = tickers[tickers["active"].fillna(False).astype(bool)]
    return set(tickers["ticker"].dropna().astype(str))


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_bars) if args.base_bars else None
    out_path = Path(args.out_bars)
    client = MassiveClient(
        get_api_key(),
        sleep_sec=args.sleep,
        cache_dir=Path(args.cache_dir),
        use_cache=not args.no_api_cache,
    )

    adjusted = not args.unadjusted
    print(f"fetching Massive grouped daily {args.start_date}..{args.end_date} adjusted={adjusted}", flush=True)
    fetched = download_grouped_daily_range(client, args.start_date, args.end_date, adjusted=adjusted)
    ticker_filter = load_ticker_filter(args.tickers_csv, args.active_only)
    if ticker_filter is not None and not fetched.empty:
        fetched = fetched[fetched["ticker"].astype(str).isin(ticker_filter)].copy()
    print(
        f"fetched rows={len(fetched)} tickers={fetched['ticker'].nunique() if not fetched.empty else 0}",
        flush=True,
    )

    base = pd.read_parquet(base_path) if base_path is not None and base_path.exists() else pd.DataFrame()
    combined = pd.concat([base, fetched], ignore_index=True) if not fetched.empty else base
    if combined.empty:
        raise RuntimeError("No bars available from base or fetch; refusing to write empty cache")
    combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
    combined = (
        combined.drop_duplicates(["date", "ticker"], keep="last")
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    print(
        f"wrote {out_path}: rows={len(combined)} "
        f"dates={combined['date'].min().date()}..{combined['date'].max().date()} "
        f"tickers={combined['ticker'].nunique()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
