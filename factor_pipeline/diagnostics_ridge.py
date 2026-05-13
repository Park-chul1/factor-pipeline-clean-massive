"""Ridge regression diagnostics: compare lambdas, inspect stability, detect ill-conditioning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_condition_number(A: np.ndarray, rcond: float = 1e-15) -> tuple[float, int]:
    """Compute condition number safely, return (cond, rank)."""
    if A.size == 0:
        return np.nan, 0
    try:
        _, s, _ = np.linalg.svd(A, full_matrices=False)
        rank = np.sum(s > rcond * s[0]) if s.size > 0 else 0
        cond = s[0] / s[-1] if s.size > 0 and s[-1] > 0 else np.nan
        return float(cond) if np.isfinite(cond) else np.inf, int(rank)
    except np.linalg.LinAlgError:
        return np.inf, 0


def _effective_dof(singular_values: np.ndarray, ridge_lambda: float) -> float:
    """Effective degrees of freedom for ridge: sum(s^2 / (s^2 + lambda))."""
    s2 = singular_values ** 2
    if ridge_lambda <= 0:
        return float(np.sum(s2 > 0))
    return float(np.sum(s2 / (s2 + ridge_lambda)))


def _solve_ridge(X: np.ndarray, r: np.ndarray, ridge_lambda: float, svd_cached: tuple | None = None) -> tuple[np.ndarray, dict]:
    """
    Solve ridge regression and return coefficients and diagnostics.

    Args:
        X: Design matrix [n_names, n_factors]
        r: Response vector [n_names]
        ridge_lambda: Ridge regularization strength
        svd_cached: Pre-computed (U, s, Vt) from SVD to reuse across lambdas

    Returns: (f, diag_dict)
    """
    n_names, n_factors = X.shape
    diag = {}

    # Check input validity
    if n_names < 1 or n_factors < 1:
        return np.full(n_factors, np.nan), {"n_names": n_names, "error": "Invalid shape"}

    # Compute or reuse SVD
    if svd_cached is None:
        try:
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            return np.full(n_factors, np.nan), {"n_names": n_names, "error": "SVD failed"}
    else:
        U, s, Vt = svd_cached

    diag["n_names"] = n_names
    diag["n_factors"] = n_factors
    diag["rank_X"] = int(np.sum(s > 1e-15 * s[0]) if s.size > 0 else 0)
    diag["singular_min"] = float(s[-1]) if s.size > 0 else np.nan
    diag["singular_max"] = float(s[0]) if s.size > 0 else np.nan

    cond_X, _ = _safe_condition_number(X)
    diag["cond_X"] = cond_X

    # Condition number of X.T @ X
    try:
        XtX = X.T @ X
        cond_XtX, _ = _safe_condition_number(XtX)
        diag["cond_XtX"] = cond_XtX
    except Exception:
        diag["cond_XtX"] = np.nan

    # Effective degrees of freedom
    diag["effective_df"] = _effective_dof(s, ridge_lambda)

    # Solve ridge using SVD: X = U * S * V^T
    # ridge: (X'X + lambda I)^{-1} X'r = V * (S^2 + lambda I)^{-1} * S * U'r
    s2 = s ** 2
    Uty = U.T @ r

    if ridge_lambda > 0:
        # (s^2 + lambda) in denominator
        f = Vt.T @ (Uty * s / (s2 + ridge_lambda))
    else:
        # OLS: lambda = 0
        f = Vt.T @ (Uty / s)

    # Factor stats
    diag["mean_abs_f"] = float(np.mean(np.abs(f)))
    diag["max_abs_f"] = float(np.max(np.abs(f)))
    diag["norm_f"] = float(np.linalg.norm(f))
    diag["f_is_finite"] = bool(np.all(np.isfinite(f)))

    # Return stats
    diag["return_norm"] = float(np.linalg.norm(r))

    # Fitted return and residuals
    fitted = X @ f
    residual = r - fitted

    diag["fitted_norm"] = float(np.linalg.norm(fitted))
    diag["residual_norm"] = float(np.linalg.norm(residual))
    diag["explained_norm_ratio"] = diag["fitted_norm"] / max(diag["return_norm"], 1e-12)
    diag["residual_norm_ratio"] = diag["residual_norm"] / max(diag["return_norm"], 1e-12)
    diag["fitted_std"] = float(np.std(fitted))
    diag["residual_std"] = float(np.std(residual))

    # R-squared metrics
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((r - np.mean(r)) ** 2))
    diag["centered_R2"] = 1.0 - ss_res / max(ss_tot, 1e-12) if ss_tot > 0 else np.nan
    diag["uncentered_R2"] = 1.0 - (diag["residual_norm"] ** 2) / max(diag["return_norm"] ** 2, 1e-12)

    diag["day_valid"] = bool(np.all(np.isfinite(f)) and diag["n_names"] >= 1)

    return f, diag


def _compute_date_diagnostics(
    X: np.ndarray,
    r: np.ndarray,
    valid_mask: np.ndarray,
    lambdas: list[float],
    min_names: int = 30,
    require_full_rank_without_ridge: bool = True,
) -> tuple[dict[int, np.ndarray], list[dict]]:
    """
    Estimate factor returns for each lambda on each date.

    Optimization: Compute SVD once per date, reuse for all lambdas.

    Returns: (factor_returns_by_lambda, date_diagnostics_list)
    """
    T, N, K = X.shape
    factor_returns_dict = {i: np.full((T, K), np.nan) for i in range(len(lambdas))}
    date_diags = []

    for t in range(T):
        mask = np.asarray(valid_mask[t], dtype=bool)
        mask &= np.isfinite(r[t])
        mask &= np.any(np.isfinite(X[t]), axis=1)
        idx_valid = np.where(mask)[0]
        n_names = len(idx_valid)

        # If there are no valid rows, preserve the date-level record for every lambda.
        if n_names == 0:
            for i, lam in enumerate(lambdas):
                date_diags.append(
                    {
                        "date_idx": t,
                        "lambda_idx": i,
                        "lambda": float(lam),
                        "n_names": 0,
                        "n_factors": K,
                        "rank_X": np.nan,
                        "singular_min": np.nan,
                        "singular_max": np.nan,
                        "cond_X": np.nan,
                        "cond_XtX": np.nan,
                        "effective_df": np.nan,
                        "mean_abs_f": np.nan,
                        "max_abs_f": np.nan,
                        "norm_f": np.nan,
                        "f_is_finite": False,
                        "return_norm": np.nan,
                        "fitted_norm": np.nan,
                        "residual_norm": np.nan,
                        "explained_norm_ratio": np.nan,
                        "residual_norm_ratio": np.nan,
                        "fitted_std": np.nan,
                        "residual_std": np.nan,
                        "centered_R2": np.nan,
                        "uncentered_R2": np.nan,
                        "day_valid": False,
                    }
                )
            continue

        X_t = np.where(np.isfinite(X[t, idx_valid, :]), X[t, idx_valid, :], 0.0)
        r_t = r[t, idx_valid].astype(float)
        r_t = r_t - np.nanmean(r_t)

        # Compute SVD ONCE for this date, reuse for all lambdas
        try:
            U, s, Vt = np.linalg.svd(X_t, full_matrices=False)
            svd_cached = (U, s, Vt)
        except np.linalg.LinAlgError:
            svd_cached = None

        for i, lam in enumerate(lambdas):
            required = min_names
            if lam == 0.0 and require_full_rank_without_ridge:
                required = max(min_names, K + 2)
            if n_names < required:
                date_diags.append(
                    {
                        "date_idx": t,
                        "lambda_idx": i,
                        "lambda": float(lam),
                        "n_names": n_names,
                        "n_factors": K,
                        "rank_X": np.nan,
                        "singular_min": np.nan,
                        "singular_max": np.nan,
                        "cond_X": np.nan,
                        "cond_XtX": np.nan,
                        "effective_df": np.nan,
                        "mean_abs_f": np.nan,
                        "max_abs_f": np.nan,
                        "norm_f": np.nan,
                        "f_is_finite": False,
                        "return_norm": float(np.linalg.norm(r_t)),
                        "fitted_norm": np.nan,
                        "residual_norm": np.nan,
                        "explained_norm_ratio": np.nan,
                        "residual_norm_ratio": np.nan,
                        "fitted_std": np.nan,
                        "residual_std": np.nan,
                        "centered_R2": np.nan,
                        "uncentered_R2": np.nan,
                        "day_valid": False,
                    }
                )
                continue

            # Pass cached SVD to avoid recomputing
            f_t, diag_t = _solve_ridge(X_t, r_t, lam, svd_cached=svd_cached)
            factor_returns_dict[i][t, :] = f_t
            diag_t["date_idx"] = t
            diag_t["lambda"] = float(lam)
            diag_t["lambda_idx"] = i
            date_diags.append(diag_t)

    return factor_returns_dict, date_diags


def _compute_factor_diagnostics(
    factor_returns_dict: dict[int, np.ndarray],
    lambdas: list[float],
    factor_names: list[str] | None = None,
) -> list[dict]:
    """Compute per-factor statistics over time for each lambda."""
    factor_diags = []

    for lambda_idx, lam in enumerate(lambdas):
        f_arr = factor_returns_dict[lambda_idx]  # Shape: [T, K]
        T, K = f_arr.shape

        for k in range(K):
            f_k = f_arr[:, k]
            finite_mask = np.isfinite(f_k)
            f_k_finite = f_k[finite_mask]

            diag_k = {
                "factor_idx": k,
                "factor_name": factor_names[k] if factor_names is not None else f"factor_{k}",
                "lambda_idx": lambda_idx,
                "lambda": float(lam),
                "finite_frac": float(np.sum(finite_mask) / T) if T > 0 else 0.0,
            }

            if len(f_k_finite) > 0:
                diag_k["mean"] = float(np.mean(f_k_finite))
                diag_k["std"] = float(np.std(f_k_finite))
                diag_k["median"] = float(np.median(f_k_finite))
                diag_k["mad"] = float(np.median(np.abs(f_k_finite - np.median(f_k_finite))))
                diag_k["min"] = float(np.min(f_k_finite))
                diag_k["max"] = float(np.max(f_k_finite))
                diag_k["mean_abs"] = float(np.mean(np.abs(f_k_finite)))
                diag_k["max_abs"] = float(np.max(np.abs(f_k_finite)))

                # Percentiles
                for q in [1, 5, 25, 75, 95, 99]:
                    diag_k[f"q{q:02d}"] = float(np.percentile(f_k_finite, q))
            else:
                diag_k.update({
                    "mean": np.nan, "std": np.nan, "median": np.nan, "mad": np.nan,
                    "min": np.nan, "max": np.nan, "mean_abs": np.nan, "max_abs": np.nan
                })
                for q in [1, 5, 25, 75, 95, 99]:
                    diag_k[f"q{q:02d}"] = np.nan

            factor_diags.append(diag_k)

    return factor_diags


def _compute_lambda_summary(date_diags: list[dict], factor_returns_dict: dict[int, np.ndarray]) -> list[dict]:
    """Compute summary statistics for each lambda."""
    df_date = pd.DataFrame(date_diags)
    lambdas_unique = sorted(df_date["lambda"].unique())
    summaries = []

    for lam in lambdas_unique:
        subset = df_date[df_date["lambda"] == lam]

        valid_days = subset[subset["day_valid"]].shape[0]
        n_names_vals = subset["n_names"].dropna()
        norm_f_vals = subset["norm_f"].dropna()
        max_abs_f_vals = subset["max_abs_f"].dropna()
        residual_ratio = subset["residual_norm_ratio"].dropna()
        explained_ratio = subset["explained_norm_ratio"].dropna()
        r2_centered = subset["centered_R2"].dropna()
        r2_uncentered = subset["uncentered_R2"].dropna()
        eff_df = subset["effective_df"].dropna()
        cond_x = subset["cond_X"].dropna()

        summary = {
            "lambda": float(lam),
            "valid_days": int(valid_days),
            "avg_n_names": float(n_names_vals.mean()) if len(n_names_vals) > 0 else np.nan,
            "median_n_names": float(n_names_vals.median()) if len(n_names_vals) > 0 else np.nan,
            "mean_norm_f": float(norm_f_vals.mean()) if len(norm_f_vals) > 0 else np.nan,
            "median_norm_f": float(norm_f_vals.median()) if len(norm_f_vals) > 0 else np.nan,
            "q95_norm_f": float(norm_f_vals.quantile(0.95)) if len(norm_f_vals) > 0 else np.nan,
            "q99_norm_f": float(norm_f_vals.quantile(0.99)) if len(norm_f_vals) > 0 else np.nan,
            "max_norm_f": float(norm_f_vals.max()) if len(norm_f_vals) > 0 else np.nan,
            "mean_max_abs_f": float(max_abs_f_vals.mean()) if len(max_abs_f_vals) > 0 else np.nan,
            "q95_max_abs_f": float(max_abs_f_vals.quantile(0.95)) if len(max_abs_f_vals) > 0 else np.nan,
            "q99_max_abs_f": float(max_abs_f_vals.quantile(0.99)) if len(max_abs_f_vals) > 0 else np.nan,
            "max_max_abs_f": float(max_abs_f_vals.max()) if len(max_abs_f_vals) > 0 else np.nan,
            "mean_residual_norm_ratio": float(residual_ratio.mean()) if len(residual_ratio) > 0 else np.nan,
            "median_residual_norm_ratio": float(residual_ratio.median()) if len(residual_ratio) > 0 else np.nan,
            "mean_explained_norm_ratio": float(explained_ratio.mean()) if len(explained_ratio) > 0 else np.nan,
            "median_explained_norm_ratio": float(explained_ratio.median()) if len(explained_ratio) > 0 else np.nan,
            "mean_centered_R2": float(r2_centered.mean()) if len(r2_centered) > 0 else np.nan,
            "median_centered_R2": float(r2_centered.median()) if len(r2_centered) > 0 else np.nan,
            "mean_uncentered_R2": float(r2_uncentered.mean()) if len(r2_uncentered) > 0 else np.nan,
            "median_uncentered_R2": float(r2_uncentered.median()) if len(r2_uncentered) > 0 else np.nan,
            "mean_effective_df": float(eff_df.mean()) if len(eff_df) > 0 else np.nan,
            "median_effective_df": float(eff_df.median()) if len(eff_df) > 0 else np.nan,
            "min_effective_df": float(eff_df.min()) if len(eff_df) > 0 else np.nan,
            "max_effective_df": float(eff_df.max()) if len(eff_df) > 0 else np.nan,
            "mean_cond_X": float(cond_x.mean()) if len(cond_x) > 0 else np.nan,
            "median_cond_X": float(cond_x.median()) if len(cond_x) > 0 else np.nan,
            "q95_cond_X": float(cond_x.quantile(0.95)) if len(cond_x) > 0 else np.nan,
            "q99_cond_X": float(cond_x.quantile(0.99)) if len(cond_x) > 0 else np.nan,
            "max_cond_X": float(cond_x.max()) if len(cond_x) > 0 else np.nan,
        }
        summaries.append(summary)

    return summaries


def _add_warning_flags(
    date_diags: list[dict],
    high_cond_X_threshold: float = 1e6,
    low_effective_df_frac: float = 0.5,
    low_explained_ratio_threshold: float = 0.05,
    bad_R2_threshold: float = -0.1,
) -> list[dict]:
    """Add warning boolean flags to date diagnostics."""
    df = pd.DataFrame(date_diags)

    # Per-lambda thresholds for high_f_norm and high_max_abs_f
    high_f_norm_thresholds = {}
    high_max_abs_f_thresholds = {}

    for lam in df["lambda"].unique():
        subset = df[df["lambda"] == lam]
        norm_vals = subset["norm_f"].dropna()
        max_abs_vals = subset["max_abs_f"].dropna()

        high_f_norm_thresholds[lam] = float(norm_vals.quantile(0.99)) if len(norm_vals) > 0 else np.inf
        high_max_abs_f_thresholds[lam] = float(max_abs_vals.quantile(0.99)) if len(max_abs_vals) > 0 else np.inf

    # Add warning columns
    df["warn_high_cond_X"] = df["cond_X"] > high_cond_X_threshold
    df["warn_high_f_norm"] = df.apply(
        lambda row: row["norm_f"] > high_f_norm_thresholds.get(row["lambda"], np.inf), axis=1
    )
    df["warn_high_max_abs_f"] = df.apply(
        lambda row: row["max_abs_f"] > high_max_abs_f_thresholds.get(row["lambda"], np.inf), axis=1
    )
    df["warn_low_effective_df"] = df["effective_df"] < (low_effective_df_frac * df["n_factors"])
    df["warn_low_explained_norm_ratio"] = df["explained_norm_ratio"] < low_explained_ratio_threshold
    df["warn_bad_R2"] = df["centered_R2"] < bad_R2_threshold

    return df.to_dict("records")


def _find_suspicious_days(
    date_diags: list[dict],
    n_top: int = 20,
) -> pd.DataFrame:
    """Find top suspicious dates per lambda and reason."""
    df = pd.DataFrame(date_diags)
    results = []

    for lam in df["lambda"].unique():
        subset = df[df["lambda"] == lam].copy()

        reasons = [
            ("largest_norm_f", "norm_f", True),
            ("largest_max_abs_f", "max_abs_f", True),
            ("largest_cond_X", "cond_X", True),
            ("worst_centered_R2", "centered_R2", False),
            ("largest_residual_norm_ratio", "residual_norm_ratio", True),
        ]

        for reason_name, col, largest in reasons:
            if col not in subset.columns:
                continue
            if largest:
                top_rows = subset.nlargest(n_top, col)[["date_idx", "lambda", col]].copy()
            else:
                top_rows = subset.nsmallest(n_top, col)[["date_idx", "lambda", col]].copy()
            top_rows["reason"] = reason_name
            top_rows["value"] = top_rows[col]
            results.append(top_rows[["date_idx", "lambda", "reason", "value"]])

    if results:
        return pd.concat(results, ignore_index=True).sort_values(["lambda", "reason", "value"], ascending=[True, True, False])
    return pd.DataFrame()


def run_estimation_diagnostics(
    X: np.ndarray,
    r: np.ndarray,
    valid_mask: np.ndarray,
    lambdas: list[float] | None = None,
    factor_names: list[str] | None = None,
    output_dir: str | Path = "data/processed/diagnostics",
    min_names: int = 30,
) -> dict[str, Any]:
    """
    Run comprehensive ridge diagnostics on factor estimation.

    Args:
        X: Shape [T, N, K] exposure tensor
        r: Shape [T, N] forward returns
        valid_mask: Shape [T, N] bool, indicates valid (tradable) stocks
        lambdas: List of lambda values to test. Default: [0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
        factor_names: List of K factor names
        output_dir: Output directory for diagnostics
        min_names: Minimum number of stocks per date

    Returns:
        Dictionary with paths and summary objects
    """
    if lambdas is None:
        lambdas = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1]

    lambdas = list(lambdas)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running estimation diagnostics with {len(lambdas)} lambdas...")
    print(f"Lambdas: {lambdas}")

    # Compute factor returns and date diagnostics
    factor_returns_dict, date_diags = _compute_date_diagnostics(X, r, valid_mask, lambdas, min_names=min_names)

    # Add warning flags
    date_diags = _add_warning_flags(date_diags)

    # Compute factor and lambda summaries
    factor_diags = _compute_factor_diagnostics(factor_returns_dict, lambdas, factor_names)
    lambda_summaries = _compute_lambda_summary(date_diags, factor_returns_dict)

    # Find suspicious days
    suspicious_df = _find_suspicious_days(date_diags, n_top=20)

    # Save outputs
    print(f"Saving diagnostics to {output_dir}...")

    # NPZ: factor returns by lambda
    np.savez(
        output_dir / "factor_returns_by_lambda.npz",
        **{f"lambda_{i}": factor_returns_dict[i] for i in range(len(lambdas))}
    )

    # CSV: lambda grid
    pd.DataFrame({"lambda_idx": range(len(lambdas)), "lambda": lambdas}).to_csv(
        output_dir / "lambda_grid.csv", index=False
    )

    # CSV: date diagnostics
    pd.DataFrame(date_diags).to_csv(output_dir / "date_diagnostics.csv", index=False)

    # CSV: factor diagnostics
    pd.DataFrame(factor_diags).to_csv(output_dir / "factor_diagnostics.csv", index=False)

    # CSV: lambda summary
    pd.DataFrame(lambda_summaries).to_csv(output_dir / "lambda_summary.csv", index=False)

    # CSV: suspicious days
    if not suspicious_df.empty:
        suspicious_df.to_csv(output_dir / "suspicious_days.csv", index=False)

    # Summary JSON
    default_lambda_idx = 0
    summary_json = {
        "default_lambda_idx": default_lambda_idx,
        "default_lambda": float(lambdas[default_lambda_idx]),
        "available_lambdas": lambdas,
        "n_lambdas": len(lambdas),
        "n_dates": X.shape[0],
        "n_stocks_avg": float(np.mean([np.sum(valid_mask[t]) for t in range(X.shape[0])])),
        "n_factors": X.shape[2],
        "output_dir": str(output_dir),
        "date_diagnostics_path": str(output_dir / "date_diagnostics.csv"),
        "factor_diagnostics_path": str(output_dir / "factor_diagnostics.csv"),
        "lambda_summary_path": str(output_dir / "lambda_summary.csv"),
        "suspicious_days_path": str(output_dir / "suspicious_days.csv"),
    }

    with open(output_dir / "diagnostics_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("RIDGE DIAGNOSTICS SUMMARY")
    print("=" * 80)

    if lambda_summaries:
        df_summary = pd.DataFrame(lambda_summaries)
        print(f"\nDefault lambda: {lambdas[default_lambda_idx]}")

        for _, row in df_summary.iterrows():
            print(f"\n  Lambda = {row['lambda']:.2e}")
            print(f"    Valid days:              {int(row['valid_days'])}")
            print(f"    Median condition(X):    {row['median_cond_X']:.2e}")
            print(f"    Q99 |f| norm:           {row['q99_norm_f']:.4f}")
            print(f"    Median effective df:    {row['median_effective_df']:.1f}")
            print(f"    Median centered R²:     {row['median_centered_R2']:.4f}")

    print("=" * 80 + "\n")

    return {
        "output_dir": output_dir,
        "factor_returns_by_lambda": factor_returns_dict,
        "date_diagnostics": date_diags,
        "factor_diagnostics": factor_diags,
        "lambda_summaries": lambda_summaries,
        "suspicious_days": suspicious_df,
        "summary": summary_json,
    }
