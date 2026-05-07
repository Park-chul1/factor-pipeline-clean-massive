from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit

import pandas as pd
import requests

BASE_URL = "https://api.massive.com"


@dataclass
class MassiveClient:
    api_key: str
    sleep_sec: float = 0.15
    timeout: int = 30
    cache_dir: Path | str | None = None
    use_cache: bool = True
    cache_namespace: str = "api_responses"

    def __post_init__(self) -> None:
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)

    def _cache_payload(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        split = urlsplit(url)
        clean_url = urlunsplit((split.scheme, split.netloc, split.path, "", ""))
        items: list[tuple[str, str]] = []
        for key, value in parse_qsl(split.query, keep_blank_values=True):
            if key.lower() != "apikey":
                items.append((key, value))
        for key, value in params.items():
            if key.lower() != "apikey":
                items.append((key, str(value)))
        return {"url": clean_url, "params": sorted(items)}

    def _cache_path(self, url: str, params: dict[str, Any]) -> Path | None:
        if self.cache_dir is None or not self.use_cache:
            return None
        payload = self._cache_payload(url, params)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        return Path(self.cache_dir) / self.cache_namespace / digest[:2] / f"{digest}.json"

    def _read_cache(self, path: Path | None) -> dict[str, Any] | None:
        if path is None or not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("response") if isinstance(data, dict) and "response" in data else data

    def _write_cache(self, path: Path | None, url: str, params: dict[str, Any], response: dict[str, Any]) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "request": self._cache_payload(url, params),
            "response": response,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":"), default=str), encoding="utf-8")
        tmp.replace(path)

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = dict(params or {})
        params["apiKey"] = self.api_key
        url = path if path.startswith("http") else BASE_URL + path
        cache_path = self._cache_path(url, params)
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached

        r = requests.get(url, params=params, timeout=self.timeout)
        if self.sleep_sec:
            time.sleep(self.sleep_sec)
        if r.status_code != 200:
            raise RuntimeError(f"Massive API error {r.status_code}: {r.text[:1000]}")
        data = r.json()
        self._write_cache(cache_path, url, params, data)
        return data

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


def download_nasdaq_tickers(client: MassiveClient, exchange: str = "XNAS", active: bool | None = None) -> pd.DataFrame:
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


def _iso_date(value: date | str | None) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).date().isoformat()


def download_financials(
    client: MassiveClient,
    ticker: str,
    timeframe: str = "quarterly",
    limit: int = 120,
    filing_date_gte: date | str | None = None,
    filing_date_lte: date | str | None = None,
    period_of_report_date_gte: date | str | None = None,
    period_of_report_date_lte: date | str | None = None,
    sort: str = "filing_date",
    order: str = "asc",
) -> pd.DataFrame:
    params = {
        "ticker": ticker,
        "timeframe": timeframe,
        "limit": limit,
        "sort": sort,
        "order": order,
    }
    optional_dates = {
        "filing_date.gte": filing_date_gte,
        "filing_date.lte": filing_date_lte,
        "period_of_report_date.gte": period_of_report_date_gte,
        "period_of_report_date.lte": period_of_report_date_lte,
    }
    for name, value in optional_dates.items():
        iso = _iso_date(value)
        if iso is not None:
            params[name] = iso
    rows = client.get_all_pages("/vX/reference/financials", params)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ticker"] = ticker
    return df
