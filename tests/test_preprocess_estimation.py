import numpy as np
import pandas as pd

from factor_pipeline.preprocess import build_exposure_tensor
from factor_pipeline.estimation import estimate_factor_returns


def test_build_exposure_tensor_filters_and_fills_missing():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    tickers = list("ABCDE")
    good = pd.DataFrame(np.arange(25, dtype=float).reshape(5, 5), index=dates, columns=tickers)
    sparse = pd.DataFrame(np.nan, index=dates, columns=tickers)
    factors = {"good": good, "sparse": sparse}

    X, names, diag = build_exposure_tensor(
        factors,
        min_names=3,
        min_factor_coverage=0.1,
        fill_missing=True,
    )

    assert names == ["good"]
    assert X.shape == (5, 5, 1)
    assert np.isfinite(X).all()
    assert diag.set_index("factor").loc["sparse", "kept"] == False


def test_build_exposure_tensor_applies_universe_mask_before_zscore():
    dates = pd.date_range("2024-01-01", periods=1, freq="B")
    tickers = ["A", "B", "DEAD"]
    raw = pd.DataFrame([[1.0, 2.0, 100.0]], index=dates, columns=tickers)
    universe_mask = pd.DataFrame([[True, True, False]], index=dates, columns=tickers)

    X, names, _ = build_exposure_tensor(
        {"value": raw},
        min_names=2,
        min_factor_coverage=0.1,
        fill_missing=True,
        universe_mask=universe_mask,
    )

    assert names == ["value"]
    assert X.shape == (1, 3, 1)
    np.testing.assert_allclose(X[0, :, 0], [-1.0, 1.0, 0.0])


def test_build_exposure_tensor_drops_nearly_duplicate_factors():
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    tickers = list("ABCDE")
    base = pd.DataFrame(
        np.arange(20, dtype=float).reshape(4, 5),
        index=dates,
        columns=tickers,
    )
    factors = {
        "mom_1": base,
        "rev_1_like": -base,
        "size": pd.DataFrame(
            np.tile([0.0, 2.0, 1.0, 4.0, 3.0], (4, 1)),
            index=dates,
            columns=tickers,
        ),
    }

    X, names, diag = build_exposure_tensor(
        factors,
        min_names=3,
        min_factor_coverage=0.1,
        fill_missing=True,
        max_factor_corr=0.999,
        corr_min_overlap=1,
    )

    assert names == ["mom_1", "size"]
    assert X.shape == (4, 5, 2)
    row = diag.set_index("factor").loc["rev_1_like"]
    assert row["kept"] == False
    assert row["drop_reason"] == "high_corr"
    assert row["corr_with"] == "mom_1"
    assert abs(row["corr_value"] + 1.0) < 1e-12


def test_ridge_estimation_works_when_factor_count_exceeds_names():
    rng = np.random.default_rng(7)
    T, N, K = 4, 8, 12
    X = rng.normal(size=(T, N, K))
    beta = rng.normal(size=K)
    r = np.einsum("tnk,k->tn", X, beta) + rng.normal(scale=0.01, size=(T, N))

    f = estimate_factor_returns(X, r, min_names=5, ridge=1e-2)

    assert f.shape == (T, K)
    assert np.isfinite(f).all()


def test_estimation_respects_universe_mask():
    X = np.ones((1, 3, 1), dtype=float)
    r = np.array([[0.01, 0.02, 0.03]], dtype=float)
    universe_mask = np.array([[True, True, False]])

    f = estimate_factor_returns(
        X,
        r,
        min_names=3,
        ridge=1e-2,
        universe_mask=universe_mask,
    )

    assert np.isnan(f[0, 0])
