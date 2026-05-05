import pandas as pd

from factor_pipeline.price_volume_factors import build_price_volume_factors
from factor_pipeline.panel import compute_forward_returns


def test_price_volume_factor_names_exist():
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    close = pd.DataFrame({"A": range(100, 400)}, index=idx, dtype=float)
    panel = {"adj_close": close, "close": close, "open": close*0.99, "high": close*1.01, "low": close*0.98, "volume": close*1000}
    f = build_price_volume_factors(panel)
    assert "mom_252" in f
    assert "amihud_20" in f
    assert "dist_52w_high" in f


def test_forward_returns_are_future_returns():
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    close = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=idx)
    r = compute_forward_returns(close, horizon=1)
    assert abs(r.iloc[0, 0] - 0.10) < 1e-12
    assert abs(r.iloc[1, 0] - 0.10) < 1e-12
    assert pd.isna(r.iloc[2, 0])
