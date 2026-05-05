from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import time
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://api.massive.com"


@dataclass
class MassiveClient:
    api_key: str
    sleep_sec: float = 0.15
    timeout: int = 30

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = dict(params or {})
        params["apiKey"] = self.api_key
        url = path if path.startswith("http") else BASE_URL + path
        r = requests.get(url, params=params, timeout=self.timeout)
        if self.sleep_sec:
            time.sleep(self.sleep_sec)
        if r.status_code != 200:
            raise RuntimeError(f"Massive API error {r.status_code}: {r.text[:1000]}")
        return r.json()

    def get_all_pages(self, path: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        data = self.get(path, params)
        out.extend(data.get("results", []) or [])
        next_url = data.get("next_url")
        while next_url:
            data = self.get(next_url, {})
            out.extend(data.get("results", []) or [])
            next_url = data.get("next_url")
        return out


def download_nasdaq_tickers(client: MassiveClient, exchange: str = "XNAS", active: bool | None = True) -> pd.DataFrame:
    params: dict[str, Any] = {
        "market": "stocks",
        "exchange": exchange,
        "limit": 1000,
        "sort": "ticker",
    }
    if active is not None:
        params["active"] = str(active).lower()
    rows = client.get_all_pages("/v3/reference/tickers", params)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Keep common stocks where type info is available. Preserve ETFs/ADRs only if type field is missing.
    if "type" in df.columns:
        common_like = df["type"].isna() | df["type"].isin(["CS", "Common Stock", "COMMON"])
        df = df.loc[common_like].copy()
    return df.sort_values("ticker").reset_index(drop=True)


def download_grouped_daily(client: MassiveClient, d: date | str) -> pd.DataFrame:
    ds = pd.Timestamp(d).date().isoformat()
    data = client.get(f"/v2/aggs/grouped/locale/us/market/stocks/{ds}", {"adjusted": "true"})
    rows = data.get("results", []) or []
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume", "vwap", "transactions"])
    df = pd.DataFrame(rows).rename(columns={"T": "ticker", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap", "n": "transactions"})
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(None).dt.normalize()
    keep = [c for c in ["date", "ticker", "open", "high", "low", "close", "volume", "vwap", "transactions"] if c in df.columns]
    return df[keep]


def download_grouped_daily_range(client: MassiveClient, start: str, end: str) -> pd.DataFrame:
    frames = []
    for d in pd.date_range(start, end, freq="B"):
        day = download_grouped_daily(client, d)
        if not day.empty:
            frames.append(day)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def download_financials(client: MassiveClient, ticker: str, timeframe: str = "ttm", limit: int = 120) -> pd.DataFrame:
    params = {"ticker": ticker, "timeframe": timeframe, "limit": limit, "sort": "filing_date"}
    rows = client.get_all_pages("/vX/reference/financials", params)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ticker"] = ticker
    return df
