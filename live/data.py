from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import logging
import os
import time

import pandas as pd

from factor_pipeline.config import get_api_key
from factor_pipeline.massive_client import MassiveClient
from live.config import LiveConfig

BAR_COLUMNS = ["timestamp", "ticker", "open", "high", "low", "close", "volume", "vwap", "transactions"]


def pandas_bar_interval(interval: str) -> str:
    if interval.endswith("m") and interval[:-1].isdigit():
        return f"{interval[:-1]}min"
    return interval


def ibkr_bar_size(interval: str) -> str:
    if interval.endswith("m") and interval[:-1].isdigit():
        n = int(interval[:-1])
        return f"{n} mins" if n != 1 else "1 min"
    if interval.endswith("min") and interval[:-3].isdigit():
        n = int(interval[:-3])
        return f"{n} mins" if n != 1 else "1 min"
    return interval


def naive_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        return ts.tz_convert(None)
    return ts


@dataclass
class FetchResult:
    bars: pd.DataFrame
    latency_seconds: float
    symbols_requested: int
    symbols_updated: int
    asof_time: pd.Timestamp


class DelayedBarProvider:
    def fetch_latest_delayed_bars(self, universe: list[str], asof_time: pd.Timestamp) -> FetchResult:
        raise NotImplementedError


def latest_cached_timestamp_by_ticker(cache_path: Path, universe: list[str]) -> pd.Series:
    if not cache_path.exists():
        return pd.Series(dtype="datetime64[ns]")
    df = normalize_bars(pd.read_parquet(cache_path, columns=["timestamp", "ticker"]))
    if universe:
        df = df[df["ticker"].isin(set(universe))]
    if df.empty:
        return pd.Series(dtype="datetime64[ns]")
    return df.groupby("ticker")["timestamp"].max()


class MassiveDelayedProvider(DelayedBarProvider):
    """Massive/Polygon-style aggregate provider.

    The code first asks for the latest complete interval at or before the delayed
    as-of. Per-symbol calls are used only because intraday grouped bars are not
    exposed by the existing project client. The provider is isolated so a batch
    endpoint can replace this without changing portfolio or execution code.
    """

    def __init__(self, config: LiveConfig):
        self.config = config
        self.client = MassiveClient(
            api_key=get_api_key(),
            cache_dir=Path("data/massive_api_cache"),
            use_cache=True,
            sleep_sec=config.MASSIVE_REQUEST_SLEEP_SECONDS,
            max_retries=3,
            retry_sleep_sec=1.0,
        )
        self._cursor = 0

    def _multiplier(self) -> int:
        return int(self.config.BAR_INTERVAL[:-1]) if self.config.BAR_INTERVAL.endswith("m") else 15

    def _rotate(self, symbols: list[str], limit: int) -> list[str]:
        if limit <= 0 or len(symbols) <= limit:
            return symbols
        start = self._cursor % len(symbols)
        ordered = symbols[start:] + symbols[:start]
        self._cursor = (start + limit) % len(symbols)
        return ordered[:limit]

    def _fetch_many(self, tasks: list[tuple[str, pd.Timestamp, pd.Timestamp, int]]) -> list[pd.DataFrame]:
        if not tasks:
            return []
        workers = max(1, int(self.config.LIVE_FETCH_WORKERS))
        if workers == 1:
            out = []
            for ticker, start, end, limit in tasks:
                try:
                    out.append(self._fetch_symbol_range(ticker, start, end, limit=limit))
                except Exception as exc:
                    logging.warning("Massive fetch failed ticker=%s error=%s", ticker, exc)
            return [df for df in out if not df.empty]

        frames: list[pd.DataFrame] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._fetch_symbol_range, ticker, start, end, limit): ticker
                for ticker, start, end, limit in tasks
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    df = future.result()
                    if not df.empty:
                        frames.append(df)
                except Exception as exc:
                    logging.warning("Massive fetch failed ticker=%s error=%s", ticker, exc)
        return frames

    def _rows_from_results(self, ticker: str, results: list[dict], asof: pd.Timestamp) -> list[dict]:
        rows = []
        for bar in results:
            ts = pd.to_datetime(bar["t"], unit="ms", utc=True).tz_convert(None)
            if ts > asof:
                continue
            rows.append({
                "timestamp": ts,
                "ticker": ticker,
                "open": bar.get("o"),
                "high": bar.get("h"),
                "low": bar.get("l"),
                "close": bar.get("c"),
                "volume": bar.get("v"),
                "vwap": bar.get("vw"),
                "transactions": bar.get("n"),
            })
        return rows

    def _fetch_symbol_range(
        self,
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        limit: int = 5000,
    ) -> pd.DataFrame:
        data = self.client.get(
            f"/v2/aggs/ticker/{ticker}/range/{self._multiplier()}/minute/{start.date()}/{end.date()}",
            {"adjusted": "true", "sort": "asc", "limit": limit},
        )
        rows = self._rows_from_results(ticker, data.get("results", []) or [], end)
        return normalize_bars(pd.DataFrame(rows))

    def fetch_latest_delayed_bars(self, universe: list[str], asof_time: pd.Timestamp) -> FetchResult:
        started = time.perf_counter()
        asof = naive_timestamp(asof_time).floor(pandas_bar_interval(self.config.BAR_INTERVAL))
        cached_latest = latest_cached_timestamp_by_ticker(self.config.CACHE_PATH, universe)
        missing = [
            ticker for ticker in universe
            if ticker not in cached_latest.index or pd.Timestamp(cached_latest.loc[ticker]) < asof
        ]
        todo = self._rotate(missing, self.config.LIVE_FETCH_BATCH_SIZE)
        tasks = [(ticker, asof.normalize(), asof, 5000) for ticker in todo]
        frames = []
        for df in self._fetch_many(tasks):
            latest = df[df["timestamp"] == df["timestamp"].max()].copy()
            frames.append(latest)
        bars = normalize_bars(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame(columns=BAR_COLUMNS)
        return FetchResult(
            bars=bars,
            latency_seconds=time.perf_counter() - started,
            symbols_requested=len(todo),
            symbols_updated=int(bars["ticker"].nunique()) if not bars.empty else 0,
            asof_time=asof,
        )

    def prefetch_history(self, universe: list[str], asof_time: pd.Timestamp | None = None, limit: int | None = None) -> FetchResult:
        started = time.perf_counter()
        asof_raw = naive_timestamp(asof_time) if asof_time is not None else pd.Timestamp.utcnow().tz_convert(None)
        asof = asof_raw.floor(pandas_bar_interval(self.config.BAR_INTERVAL))
        start = asof - pd.Timedelta(days=self.config.MASSIVE_PREFETCH_DAYS)
        symbols = universe[: limit if limit is not None else len(universe)]
        tasks = [(ticker, start, asof, 5000) for ticker in symbols]
        frames = self._fetch_many(tasks)
        bars = normalize_bars(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame(columns=BAR_COLUMNS)
        return FetchResult(bars, time.perf_counter() - started, len(symbols), int(bars["ticker"].nunique()) if not bars.empty else 0, asof)


class IBKRDelayedProvider(DelayedBarProvider):
    """IBKR delayed historical 15-minute bar provider.

    IBKR historical data is contract-oriented, so there is no true grouped
    endpoint equivalent here. The provider skips symbols whose cache is already
    current for the delayed as-of and caps per-loop requests to avoid hammering
    TWS/Gateway. Use prefetch_history before the live loop to warm the cache.
    """

    def __init__(self, config: LiveConfig):
        try:
            from ib_insync import IB, Stock
        except ImportError as exc:
            raise ImportError("ib_insync is required for DATA_PROVIDER=ibkr_delayed") from exc

        self.config = config
        self.IB = IB
        self.Stock = Stock
        self.ib = IB()
        self.ib.connect(
            config.IBKR_HOST,
            config.IBKR_PORT,
            clientId=config.IBKR_DATA_CLIENT_ID,
            timeout=10,
        )
        self.ib.reqMarketDataType(config.IBKR_MARKET_DATA_TYPE)
        self._cursor = 0

    def _stock(self, ticker: str):
        return self.Stock(ticker, "SMART", "USD")

    def _bar_to_row(self, ticker: str, bar) -> dict:
        return {
            "timestamp": pd.Timestamp(bar.date).tz_localize(None) if pd.Timestamp(bar.date).tzinfo is None else pd.Timestamp(bar.date).tz_convert(None),
            "ticker": ticker,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
            "vwap": float(bar.average) if bar.average is not None else float("nan"),
            "transactions": float(bar.barCount) if bar.barCount is not None else float("nan"),
        }

    def _fetch_symbol(self, ticker: str, asof: pd.Timestamp, duration: str | None = None) -> pd.DataFrame:
        end = asof.strftime("%Y%m%d %H:%M:%S")
        bars = self.ib.reqHistoricalData(
            self._stock(ticker),
            endDateTime=end,
            durationStr=duration or self.config.IBKR_HISTORY_DURATION,
            barSizeSetting=ibkr_bar_size(self.config.BAR_INTERVAL),
            whatToShow=self.config.IBKR_WHAT_TO_SHOW,
            useRTH=self.config.IBKR_USE_RTH,
            formatDate=1,
            keepUpToDate=False,
        )
        if self.config.IBKR_REQUEST_SLEEP_SECONDS > 0:
            self.ib.sleep(self.config.IBKR_REQUEST_SLEEP_SECONDS)
        rows = [self._bar_to_row(ticker, bar) for bar in bars]
        out = normalize_bars(pd.DataFrame(rows))
        return out[out["timestamp"] <= asof] if not out.empty else out

    def _rotate(self, symbols: list[str], limit: int) -> list[str]:
        if limit <= 0 or len(symbols) <= limit:
            return symbols
        start = self._cursor % len(symbols)
        ordered = symbols[start:] + symbols[:start]
        self._cursor = (start + limit) % len(symbols)
        return ordered[:limit]

    def fetch_latest_delayed_bars(self, universe: list[str], asof_time: pd.Timestamp) -> FetchResult:
        started = time.perf_counter()
        asof = naive_timestamp(asof_time).floor(pandas_bar_interval(self.config.BAR_INTERVAL))
        cached_latest = latest_cached_timestamp_by_ticker(self.config.CACHE_PATH, universe)
        missing = [
            ticker for ticker in universe
            if ticker not in cached_latest.index or pd.Timestamp(cached_latest.loc[ticker]) < asof
        ]
        todo = self._rotate(missing, self.config.LIVE_FETCH_BATCH_SIZE)
        frames = []
        for ticker in todo:
            try:
                df = self._fetch_symbol(ticker, asof)
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                logging.warning("IBKR delayed fetch failed ticker=%s error=%s", ticker, exc)
        bars = normalize_bars(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame(columns=BAR_COLUMNS)
        return FetchResult(
            bars=bars,
            latency_seconds=time.perf_counter() - started,
            symbols_requested=len(todo),
            symbols_updated=int(bars["ticker"].nunique()) if not bars.empty else 0,
            asof_time=asof,
        )

    def prefetch_history(self, universe: list[str], asof_time: pd.Timestamp | None = None, limit: int | None = None) -> FetchResult:
        started = time.perf_counter()
        asof_raw = naive_timestamp(asof_time) if asof_time is not None else pd.Timestamp.utcnow().tz_convert(None)
        asof = asof_raw.floor(pandas_bar_interval(self.config.BAR_INTERVAL))
        symbols = universe[: limit if limit is not None else len(universe)]
        frames = []
        for i, ticker in enumerate(symbols, 1):
            try:
                df = self._fetch_symbol(ticker, asof, duration=self.config.IBKR_HISTORY_DURATION)
                if not df.empty:
                    frames.append(df)
                if i % 25 == 0:
                    logging.info("IBKR prefetch progress %s/%s", i, len(symbols))
            except Exception as exc:
                logging.warning("IBKR prefetch failed ticker=%s error=%s", ticker, exc)
        bars = normalize_bars(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame(columns=BAR_COLUMNS)
        return FetchResult(bars, time.perf_counter() - started, len(symbols), int(bars["ticker"].nunique()) if not bars.empty else 0, asof)

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()


class CachedOnlyProvider(DelayedBarProvider):
    def __init__(self, config: LiveConfig):
        self.config = config

    def fetch_latest_delayed_bars(self, universe: list[str], asof_time: pd.Timestamp) -> FetchResult:
        started = time.perf_counter()
        asof = naive_timestamp(asof_time).floor(pandas_bar_interval(self.config.BAR_INTERVAL))
        hist = load_recent_history(universe, self.config.LOOKBACK_BARS, self.config.CACHE_PATH, end_time=asof)
        if hist.empty:
            bars = pd.DataFrame(columns=BAR_COLUMNS)
        else:
            latest_ts = hist["timestamp"].max()
            bars = hist[hist["timestamp"] == latest_ts].copy()
        return FetchResult(bars, time.perf_counter() - started, len(universe), bars["ticker"].nunique() if not bars.empty else 0, asof)


def make_provider(config: LiveConfig) -> DelayedBarProvider:
    if config.DATA_PROVIDER.lower() in {"ibkr", "ibkr_delayed"}:
        return IBKRDelayedProvider(config)
    if config.DATA_PROVIDER.lower() == "massive" and os.getenv("MASSIVE_API_KEY"):
        return MassiveDelayedProvider(config)
    return CachedOnlyProvider(config)


def normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)
    out = bars.copy()
    if "timestamp" not in out.columns and "date" in out.columns:
        out["timestamp"] = out["date"]
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    out["ticker"] = out["ticker"].astype(str)
    for col in [c for c in BAR_COLUMNS if c not in {"timestamp", "ticker"}]:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[BAR_COLUMNS].dropna(subset=["timestamp", "ticker"])
    return out.drop_duplicates(["timestamp", "ticker"], keep="last").sort_values(["timestamp", "ticker"]).reset_index(drop=True)


def update_local_bar_cache(bars: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    bars = normalize_bars(bars)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        old = normalize_bars(pd.read_parquet(cache_path))
        combined = pd.concat([old, bars], ignore_index=True) if not bars.empty else old
    else:
        combined = bars
    combined = combined.drop_duplicates(["timestamp", "ticker"], keep="last").sort_values(["timestamp", "ticker"]).reset_index(drop=True)
    combined.to_parquet(cache_path, index=False)
    return combined


def load_recent_history(
    universe: list[str],
    lookback: int,
    cache_path: Path,
    end_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame(columns=BAR_COLUMNS)
    df = normalize_bars(pd.read_parquet(cache_path))
    if universe:
        df = df[df["ticker"].isin(set(universe))]
    if end_time is not None:
        end = naive_timestamp(end_time)
        df = df[df["timestamp"] <= end]
    if df.empty:
        return df
    last_times = sorted(df["timestamp"].dropna().unique())[-lookback:]
    return df[df["timestamp"].isin(last_times)].sort_values(["timestamp", "ticker"]).reset_index(drop=True)


def load_universe(config: LiveConfig) -> list[str]:
    path = config.UNIVERSE_PATH
    if path.exists():
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        if config.LIVE_ACTIVE_ONLY and "active" in df.columns:
            df = df[df["active"].fillna(False).astype(bool)].copy()
        if config.LIVE_ACTIVE_ONLY and "delisted_utc" in df.columns:
            df = df[df["delisted_utc"].isna()].copy()
        col = "ticker" if "ticker" in df.columns else df.columns[0]
        tickers = df[col].dropna().astype(str).drop_duplicates().sort_values().tolist()
    else:
        tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

    limit = config.UNIVERSE_SIZE_LIMIT

    if config.UNIVERSE_RANK_BY == "dollar_volume":
        ranked = rank_universe_by_dollar_volume(tickers, config)
        if ranked:
            return ranked[:limit]
    elif config.UNIVERSE_RANK_BY not in {"ticker", "alphabetical"}:
        raise ValueError(f"unknown UNIVERSE_RANK_BY={config.UNIVERSE_RANK_BY}")
    return tickers[:limit]


def rank_universe_by_dollar_volume(tickers: list[str], config: LiveConfig) -> list[str]:
    bars_path = config.UNIVERSE_BARS_PATH
    if not bars_path.exists():
        logging.warning("universe dollar-volume ranking cache missing: %s", bars_path)
        return []

    try:
        bars = pd.read_parquet(bars_path, columns=["date", "ticker", "close", "volume"])
    except Exception as exc:
        logging.warning("failed to read universe bars for ranking path=%s error=%s", bars_path, exc)
        return []

    if bars.empty:
        return []

    ticker_set = set(tickers)
    bars = bars[bars["ticker"].astype(str).isin(ticker_set)].copy()
    if bars.empty:
        return []

    bars["date"] = pd.to_datetime(bars["date"], errors="coerce")
    bars["close"] = pd.to_numeric(bars["close"], errors="coerce")
    bars["volume"] = pd.to_numeric(bars["volume"], errors="coerce")
    dates = sorted(bars["date"].dropna().unique())
    if config.UNIVERSE_RANK_LOOKBACK_BARS > 0 and len(dates) > config.UNIVERSE_RANK_LOOKBACK_BARS:
        dates = dates[-config.UNIVERSE_RANK_LOOKBACK_BARS :]
        bars = bars[bars["date"].isin(dates)].copy()

    bars["dollar_volume"] = bars["close"] * bars["volume"]
    ranked = (
        bars.groupby("ticker")["dollar_volume"]
        .median()
        .replace([float("inf"), float("-inf")], pd.NA)
        .dropna()
        .sort_values(ascending=False)
    )
    return ranked.index.astype(str).tolist()
