from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd


def array_summary(name: str, arr) -> dict:
    x = np.asarray(arr, dtype=float)
    finite = np.isfinite(x)
    return {
        "name": name,
        "shape": list(x.shape),
        "finite_ratio": float(finite.mean()) if x.size else 0.0,
        "nan_count": int(np.isnan(x).sum()),
        "posinf_count": int(np.isposinf(x).sum()),
        "neginf_count": int(np.isneginf(x).sum()),
        "mean": float(np.nanmean(x)) if finite.any() else None,
        "std": float(np.nanstd(x)) if finite.any() else None,
    }


def factor_diagnostics(factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in factors.items():
        arr = df.to_numpy(dtype=float)
        fin = np.isfinite(arr)
        rows.append({
            "factor": name,
            "shape": str(list(arr.shape)),
            "finite_ratio": float(fin.mean()) if arr.size else 0.0,
            "mean": float(np.nanmean(arr)) if fin.any() else np.nan,
            "std": float(np.nanstd(arr)) if fin.any() else np.nan,
        })
    return pd.DataFrame(rows).sort_values("factor")


def save_json(obj: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
