import numpy as np

from scripts.plot_residual_heatmap import _signed_sqrt_abs_residual, plot_heatmap


def test_signed_sqrt_abs_residual_keeps_sign_and_uses_root_magnitude():
    residual = np.array([-4.0, -0.25, 0.0, 0.25, 9.0, np.nan])
    transformed = _signed_sqrt_abs_residual(residual)

    assert np.allclose(transformed[:5], np.array([-2.0, -0.5, 0.0, 0.5, 3.0]))
    assert np.isnan(transformed[5])


def test_plot_heatmap_writes_signed_sqrt_abs_residual_image(tmp_path):
    residual = np.array([[-4.0, 0.0, 9.0], [np.nan, 0.25, -0.25]])
    output_path = tmp_path / "heatmap.png"

    order = plot_heatmap(
        sq_err=residual**2,
        residual=residual,
        dates=["2024-01-02", "2024-01-03"],
        tickers=["AAA", "BBB", "CCC"],
        output_path=output_path,
        clip_percentile=100.0,
        sort_tickers_by="none",
    )

    assert order.tolist() == [0, 1, 2]
    assert output_path.stat().st_size > 0
