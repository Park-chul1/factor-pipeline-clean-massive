from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from ib_insync import IB, Stock
except ImportError as exc:
    raise ImportError(
        "ib_insync is required to run this script. Install it with `pip install ib_insync`."
    ) from exc


def parse_args():
    p = argparse.ArgumentParser(
        description="Fetch delayed IBKR daily bars and append them to a grouped daily parquet cache"
    )
    p.add_argument("--host", default="172.30.1.41")
    p.add_argument("--port", type=int, default=7497)
    p.add_argument("--client-id", type=int, default=41)
    p.add_argument("--market-data-type", type=int, default=3, help="3 means delayed market data")
    p.add_argument("--tickers-csv", required=True, help="CSV with a ticker column and optional active column")
    p.add_argument("--base-bars", required=True, help="Existing grouped daily parquet to extend")
    p.add_argument("--checkpoint", required=True, help="Parquet file for fetched IBKR rows")
    p.add_argument("--status-csv", required=True, help="CSV file with per-symbol fetch status")
    p.add_argument("--out-bars", required=True, help="Combined grouped daily parquet output")
    p.add_argument("--start-date", default=None, help="First date to keep; defaults to base max date + 1 day")
    p.add_argument("--end-date", default=None, help="Last date to keep; useful for pinning an after-close run")
    p.add_argument("--duration", default="3 M", help="IBKR historical duration, e.g. '3 M'")
    p.add_argument("--bar-size", default="1 day")
    p.add_argument("--what-to-show", default="TRADES", choices=["TRADES", "ADJUSTED_LAST"])
    p.add_argument("--active-only", action="store_true", help="Fetch only rows where tickers.csv active is true")
    p.add_argument("--limit", type=int, default=None, help="Optional symbol limit for smoke tests")
    p.add_argument("--sleep", type=float, default=0.05, help="Pause between requests")
    p.add_argument("--refresh", action="store_true", help="Refetch symbols already present in status-csv")
    return p.parse_args()


def load_symbols(path: Path, active_only: bool, limit: int | None) -> list[str]:
    tickers = pd.read_csv(path)
    if "ticker" not in tickers.columns:
        raise ValueError(f"{path} must contain a ticker column")
    if active_only and "active" in tickers.columns:
        tickers = tickers[tickers["active"].fillna(False).astype(bool)]
    symbols = tickers["ticker"].dropna().astype(str).drop_duplicates().sort_values().tolist()
    return symbols[:limit] if limit is not None else symbols


def bar_to_row(symbol: str, bar) -> dict:
    return {
        "date": pd.Timestamp(bar.date).normalize(),
        "ticker": symbol,
        "open": float(bar.open),
        "high": float(bar.high),
        "low": float(bar.low),
        "close": float(bar.close),
        "volume": float(bar.volume),
        "vwap": float(bar.average) if bar.average is not None else float("nan"),
        "transactions": float(bar.barCount) if bar.barCount is not None else float("nan"),
    }


def read_parquet_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def write_outputs(
    rows: list[dict],
    status_rows: list[dict],
    checkpoint_path: Path,
    status_path: Path,
) -> None:
    if rows:
        new_rows = pd.DataFrame(rows)
        old_rows = read_parquet_or_empty(checkpoint_path)
        combined = pd.concat([old_rows, new_rows], ignore_index=True) if not old_rows.empty else new_rows
        combined = (
            combined.drop_duplicates(["date", "ticker"], keep="last")
            .sort_values(["date", "ticker"])
            .reset_index(drop=True)
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(checkpoint_path, index=False)

    if status_rows:
        new_status = pd.DataFrame(status_rows)
        old_status = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
        combined_status = (
            pd.concat([old_status, new_status], ignore_index=True)
            if not old_status.empty
            else new_status
        )
        combined_status = combined_status.drop_duplicates("ticker", keep="last").sort_values("ticker")
        status_path.parent.mkdir(parents=True, exist_ok=True)
        combined_status.to_csv(status_path, index=False)


def combine_with_base(
    base_path: Path,
    checkpoint_path: Path,
    out_path: Path,
    end_date: pd.Timestamp | None = None,
) -> None:
    base = pd.read_parquet(base_path)
    fetched = read_parquet_or_empty(checkpoint_path)
    if fetched.empty:
        combined = base
    else:
        combined = pd.concat([base, fetched], ignore_index=True)
        combined = combined.drop_duplicates(["date", "ticker"], keep="last")
    if end_date is not None:
        combined = combined[pd.to_datetime(combined["date"]).dt.normalize() <= end_date].copy()
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    print(
        f"wrote {out_path}: rows={len(combined)}, dates={combined['date'].min()}..{combined['date'].max()}, "
        f"tickers={combined['ticker'].nunique()}"
    )


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_bars)
    checkpoint_path = Path(args.checkpoint)
    status_path = Path(args.status_csv)
    out_path = Path(args.out_bars)

    base = pd.read_parquet(base_path, columns=["date"])
    base_max_date = pd.to_datetime(base["date"]).max().normalize()
    start_date = (
        pd.Timestamp(args.start_date).normalize()
        if args.start_date
        else base_max_date + pd.Timedelta(days=1)
    )
    end_date = pd.Timestamp(args.end_date).normalize() if args.end_date else None
    if end_date is not None and end_date < start_date:
        raise ValueError("--end-date must be on or after --start-date")
    print(
        f"base max date={base_max_date.date()}, fetching rows on/after {start_date.date()}"
        + (f" and on/before {end_date.date()}" if end_date is not None else "")
    )

    symbols = load_symbols(Path(args.tickers_csv), args.active_only, args.limit)
    done: set[str] = set()
    if status_path.exists() and not args.refresh:
        done = set(pd.read_csv(status_path)["ticker"].astype(str))
    todo = [symbol for symbol in symbols if symbol not in done]
    print(f"symbols total={len(symbols)}, already_done={len(done)}, todo={len(todo)}")

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id, timeout=10)
    ib.reqMarketDataType(args.market_data_type)

    rows: list[dict] = []
    status_rows: list[dict] = []
    started = time.time()
    try:
        for i, symbol in enumerate(todo, 1):
            status = {
                "ticker": symbol,
                "status": "ok",
                "n_bars": 0,
                "n_kept": 0,
                "last_date": "",
                "error": "",
            }
            try:
                bars = ib.reqHistoricalData(
                    Stock(symbol, "SMART", "USD"),
                    endDateTime="",
                    durationStr=args.duration,
                    barSizeSetting=args.bar_size,
                    whatToShow=args.what_to_show,
                    useRTH=True,
                    formatDate=1,
                    keepUpToDate=False,
                )
                kept = []
                for bar in bars:
                    bar_date = pd.Timestamp(bar.date).normalize()
                    if bar_date < start_date:
                        continue
                    if end_date is not None and bar_date > end_date:
                        continue
                    kept.append(bar_to_row(symbol, bar))
                rows.extend(kept)
                status["n_bars"] = len(bars)
                status["n_kept"] = len(kept)
                if bars:
                    status["last_date"] = str(pd.Timestamp(bars[-1].date).date())
                if not kept:
                    status["status"] = "no_new_rows"
            except Exception as exc:
                status["status"] = "error"
                status["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            status_rows.append(status)

            if args.sleep > 0:
                ib.sleep(args.sleep)
            if i % 25 == 0:
                write_outputs(rows, status_rows, checkpoint_path, status_path)
                rows.clear()
                status_rows.clear()
                elapsed = time.time() - started
                print(f"progress {i}/{len(todo)} elapsed={elapsed:.1f}s", flush=True)
    finally:
        write_outputs(rows, status_rows, checkpoint_path, status_path)
        ib.disconnect()

    combine_with_base(base_path, checkpoint_path, out_path, end_date=end_date)


if __name__ == "__main__":
    main()
