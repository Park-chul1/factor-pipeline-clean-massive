from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_cross_section(
    df: pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
    min_names: int = 30,
) -> pd.DataFrame:
    """Clip each date's cross-section to robust quantiles.

    Missing values are not imputed here. If too few names are available on a
    date, that date is marked unavailable for this factor.
    """
    out = df.copy().astype(float)
    for idx, row in out.iterrows():
        vals = row[np.isfinite(row)]
        if vals.size < min_names:
            out.loc[idx] = np.nan
            continue
        lo, hi = vals.quantile(lower), vals.quantile(upper)
        out.loc[idx] = row.clip(lo, hi)
    return out


def zscore_cross_section(df: pd.DataFrame, min_names: int = 30) -> pd.DataFrame:
    """Convert each date's cross-section to mean 0, std 1."""
    out = df.copy().astype(float)
    for idx, row in out.iterrows():
        vals = row[np.isfinite(row)]
        if vals.size < min_names:
            out.loc[idx] = np.nan
            continue
        mu = vals.mean()
        sd = vals.std(ddof=0)
        if not np.isfinite(sd) or sd <= 1e-12:
            out.loc[idx] = np.nan
        else:
            out.loc[idx] = (row - mu) / sd
    return out


def preprocess_factor(
    df: pd.DataFrame,
    min_names: int = 30,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Winsorize then cross-sectionally z-score a raw factor panel."""
    return zscore_cross_section(
        winsorize_cross_section(df, lower=lower, upper=upper, min_names=min_names),
        min_names=min_names,
    )


def preprocess_factor_dict(
    factors: dict[str, pd.DataFrame],
    min_names: int = 30,
    min_factor_coverage: float = 0.02,
    lower: float = 0.01,
    upper: float = 0.99,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Preprocess all factors and drop unusable ones.

    A factor is kept only if enough entries are finite after point-in-time
    construction, winsorization, and z-scoring. The threshold is intentionally
    configurable because tiny smoke tests have much less coverage than a full
    NASDAQ run.
    """
    processed: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    for name, raw in factors.items():
        raw_arr = raw.to_numpy(dtype=float)
        raw_fin = np.isfinite(raw_arr)
        z = preprocess_factor(raw, min_names=min_names, lower=lower, upper=upper)
        z_arr = z.to_numpy(dtype=float)
        z_fin = np.isfinite(z_arr)
        finite_ratio = float(z_fin.mean()) if z_arr.size else 0.0
        finite_dates = int(np.isfinite(z_arr).any(axis=1).sum()) if z_arr.ndim == 2 else 0
        kept = finite_ratio >= min_factor_coverage and finite_dates > 0
        rows.append({
            "factor": name,
            "kept": bool(kept),
            "raw_finite_ratio": float(raw_fin.mean()) if raw_arr.size else 0.0,
            "processed_finite_ratio": finite_ratio,
            "finite_dates": finite_dates,
            "min_factor_coverage": min_factor_coverage,
        })
        if kept:
            processed[name] = z
    diag = pd.DataFrame(rows).sort_values(["kept", "processed_finite_ratio", "factor"], ascending=[False, False, True])
    return processed, diag


def build_exposure_tensor(
    factors: dict[str, pd.DataFrame],
    min_names: int = 30,
    min_factor_coverage: float = 0.02,
    fill_missing: bool = True,
    lower: float = 0.01,
    upper: float = 0.99,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Build X[T,N,K] from raw factor panels.

    Processing policy:
    1. raw factor values keep natural NaNs from insufficient history or missing
       financials;
    2. each date/factor is winsorized cross-sectionally;
    3. each date/factor is z-scored cross-sectionally;
    4. sparse factors are dropped by processed finite coverage;
    5. remaining NaNs are optionally filled with 0, the cross-sectional neutral
       exposure after z-scoring. This keeps regressions usable without leaking
       future data.
    """
    processed, diag = preprocess_factor_dict(
        factors,
        min_names=min_names,
        min_factor_coverage=min_factor_coverage,
        lower=lower,
        upper=upper,
    )
    names = list(processed.keys())
    if not names:
        # Preserve expected T,N from first raw factor if possible.
        if factors:
            first = next(iter(factors.values()))
            return np.empty((len(first.index), len(first.columns), 0), dtype=float), [], diag
        return np.empty((0, 0, 0), dtype=float), [], diag

    arrs = []
    for name in names:
        a = processed[name].to_numpy(dtype=float)
        if fill_missing:
            a = np.where(np.isfinite(a), a, 0.0)
        arrs.append(a)
    X = np.stack(arrs, axis=2)  # T x N x K
    return X, names, diag
