from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class PipelineConfig:
    start: str = "2019-01-01"
    end: str = "2024-12-31"
    exchange: str = "XNAS"
    max_tickers: int | None = None
    out_dir: Path = Path("data/processed")
    cache_dir: Path = Path("data/cache")
    min_names: int = 30
    ridge: float = 1e-4
    forward_horizon: int = 1
    financial_timeframe: str = "quarterly"
    financial_limit: int = 100
    financial_workers: int = 1
    financial_lookback_days: int = 550
    financial_lag_days: int = 60
    ttm_min_quarters: int = 4
    max_factor_corr: float = 0.999
    corr_min_overlap: int = 100
    request_sleep_sec: float = 0.15
    use_cache: bool = True


def get_api_key() -> str:
    key = os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")
    if not key:
        raise RuntimeError(
            "Set MASSIVE_API_KEY first, e.g. export MASSIVE_API_KEY='...'"
        )
    return key
