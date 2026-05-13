import numpy as np
import pandas as pd
from pathlib import Path

from factor_pipeline.diagnostics_ridge import run_estimation_diagnostics


def test_run_estimation_diagnostics_creates_output_files(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    T, N, K = 4, 12, 3
    X = rng.normal(size=(T, N, K))
    beta = rng.normal(size=K)
    r = np.einsum("tnk,k->tn", X, beta) + rng.normal(scale=0.01, size=(T, N))
    valid_mask = np.ones((T, N), dtype=bool)

    out_dir = tmp_path / "ridge_diag"
    result = run_estimation_diagnostics(
        X=X,
        r=r,
        valid_mask=valid_mask,
        lambdas=[0.0, 1e-6, 1e-2],
        factor_names=[f"f{i}" for i in range(K)],
        output_dir=out_dir,
        min_names=5,
    )

    assert (out_dir / "factor_returns_by_lambda.npz").exists()
    assert (out_dir / "lambda_grid.csv").exists()
    assert (out_dir / "date_diagnostics.csv").exists()
    assert (out_dir / "factor_diagnostics.csv").exists()
    assert (out_dir / "lambda_summary.csv").exists()
    assert (out_dir / "diagnostics_summary.json").exists()
    assert isinstance(result["lambda_summaries"], list)
    assert len(result["lambda_summaries"]) == 3


def test_zero_lambda_matches_ols(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    T, N, K = 3, 10, 4
    X = rng.normal(size=(T, N, K))
    beta = rng.normal(size=K)
    r = np.einsum("tnk,k->tn", X, beta) + rng.normal(scale=0.01, size=(T, N))
    valid_mask = np.ones((T, N), dtype=bool)

    out_dir = tmp_path / "ridge_ols"
    result = run_estimation_diagnostics(
        X=X,
        r=r,
        valid_mask=valid_mask,
        lambdas=[0.0],
        factor_names=[f"f{i}" for i in range(K)],
        output_dir=out_dir,
        min_names=1,
    )

    npz_path = out_dir / "factor_returns_by_lambda.npz"
    with np.load(npz_path) as npz:
        f_ridge = npz["lambda_0"]

    # Compute OLS explicitly for each date using the same cross-sectional centering as the pipeline.
    f_ols = np.full((T, K), np.nan)
    for t in range(T):
        Xt = X[t, :, :]
        y = r[t] - np.nanmean(r[t])
        ft, _, _, _ = np.linalg.lstsq(Xt, y, rcond=None)
        f_ols[t] = ft

    assert np.allclose(f_ridge, f_ols, atol=1e-8, equal_nan=True)
    assert all(np.isfinite(f_ridge).ravel())


def test_effective_df_decreases_as_lambda_increases(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    T, N, K = 5, 15, 3
    X = rng.normal(size=(T, N, K))
    beta = rng.normal(size=K)
    r = np.einsum("tnk,k->tn", X, beta) + rng.normal(scale=0.01, size=(T, N))
    valid_mask = np.ones((T, N), dtype=bool)

    out_dir = tmp_path / "ridge_df"
    run_estimation_diagnostics(
        X=X,
        r=r,
        valid_mask=valid_mask,
        lambdas=[0.0, 1e-6, 1e-2, 1e-1],
        factor_names=[f"f{i}" for i in range(K)],
        output_dir=out_dir,
        min_names=5,
    )

    df = pd.read_csv(out_dir / "lambda_summary.csv")
    effective_df = df["median_effective_df"].to_numpy()
    assert np.all(np.diff(effective_df[np.isfinite(effective_df)]) <= 1e-8)


def test_insufficient_names_produces_invalid_day_flags(tmp_path: Path) -> None:
    rng = np.random.default_rng(13)
    T, N, K = 2, 4, 2
    X = rng.normal(size=(T, N, K))
    beta = rng.normal(size=K)
    r = np.einsum("tnk,k->tn", X, beta) + rng.normal(scale=0.01, size=(T, N))
    valid_mask = np.ones((T, N), dtype=bool)

    out_dir = tmp_path / "ridge_invalid"
    run_estimation_diagnostics(
        X=X,
        r=r,
        valid_mask=valid_mask,
        lambdas=[1e-2, 1e-1],
        factor_names=[f"f{i}" for i in range(K)],
        output_dir=out_dir,
        min_names=10,
    )

    df = pd.read_csv(out_dir / "date_diagnostics.csv")
    assert not df["day_valid"].any()
    assert (df["n_names"] < 10).all()
