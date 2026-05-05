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


def test_ridge_estimation_works_when_factor_count_exceeds_names():
    rng = np.random.default_rng(7)
    T, N, K = 4, 8, 12
    X = rng.normal(size=(T, N, K))
    beta = rng.normal(size=K)
    r = np.einsum("tnk,k->tn", X, beta) + rng.normal(scale=0.01, size=(T, N))

    f = estimate_factor_returns(X, r, min_names=5, ridge=1e-2)

    assert f.shape == (T, K)
    assert np.isfinite(f).all()
