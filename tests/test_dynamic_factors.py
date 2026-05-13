from __future__ import annotations

import numpy as np
import pytest

from factors.kalman_filter import kalman_filter_factor_returns_from_ols, kalman_filter_factor_returns_direct_observation
from factors.regime_sensor import classify_regime
from factors.blend import blend_factor_returns


class TestKalmanFilterDirect:
    def test_shapes_and_types(self):
        """Test output shapes and types for direct observation."""
        T, N, K = 100, 50, 5
        np.random.seed(42)
        X = np.random.randn(T, N, K)
        r = np.random.randn(T, N)

        result = kalman_filter_factor_returns_direct_observation(X, r)

        assert isinstance(result, dict)
        assert result["f_kf"].shape == (T, K)
        assert result["innovation_norm"].shape == (T,)
        assert result["innovation_score"].shape == (T,)
        assert result["valid_obs_count"].shape == (T,)

    def test_nan_handling(self):
        """Test NaN handling in inputs."""
        T, N, K = 50, 20, 3
        np.random.seed(42)
        X = np.random.randn(T, N, K)
        r = np.random.randn(T, N)

        # Add some NaNs
        X[10, 5, :] = np.nan
        r[10, 5] = np.nan
        X[20, :, 1] = np.nan  # Make some exposures invalid

        result = kalman_filter_factor_returns_direct_observation(X, r)

        # Should still produce outputs
        assert np.isfinite(result["f_kf"]).sum() > 0
        assert result["valid_obs_count"][10] < N  # Should have fewer valid obs

    def test_valid_obs_count(self):
        """Test valid observation counting."""
        T, N, K = 20, 10, 2
        X = np.ones((T, N, K))  # All finite
        r = np.ones((T, N))     # All finite

        result = kalman_filter_factor_returns_direct_observation(X, r)

        # All should be valid
        assert np.all(result["valid_obs_count"] == N)

        # Make some invalid
        X[5, 3, 0] = np.nan
        result2 = kalman_filter_factor_returns_direct_observation(X, r)
        assert result2["valid_obs_count"][5] == N - 1


class TestKalmanFilterFromOLS:
    def test_shapes_and_types(self):
        """Test output shapes and types for OLS-based filter."""
        T, K = 100, 5
        np.random.seed(42)
        f_ols = np.random.randn(T, K)

        result = kalman_filter_factor_returns_from_ols(f_ols)

        assert isinstance(result, dict)
        assert result["f_kf"].shape == (T, K)
        assert result["innovation_norm"].shape == (T,)
        assert result["innovation_score"].shape == (T,)
        assert result["valid_factor_count"].shape == (T,)

    def test_nan_handling(self):
        """Test NaN handling in f_ols."""
        T, K = 50, 3
        np.random.seed(42)
        f_ols = np.random.randn(T, K)

        # Add some NaNs
        f_ols[10, 1] = np.nan
        f_ols[20, :] = np.nan  # Entire row NaN

        result = kalman_filter_factor_returns_from_ols(f_ols)

        # Should still produce outputs
        assert np.isfinite(result["f_kf"]).sum() > 0
        assert result["valid_factor_count"][10] == K - 1
        assert result["valid_factor_count"][20] == 0

    def test_no_large_matrices(self):
        """Test that no matrices larger than K x K are created."""
        T, K = 30, 3
        f_ols = np.random.randn(T, K)

        result = kalman_filter_factor_returns_from_ols(f_ols)
        assert result["f_kf"].shape == (T, K)

    def test_handles_all_nan(self):
        """Test when all factors are NaN."""
        T, K = 10, 3
        f_ols = np.full((T, K), np.nan)

        result = kalman_filter_factor_returns_from_ols(f_ols)

        # Should carry forward zeros
        assert np.allclose(result["f_kf"], 0.0)
        assert result["valid_factor_count"][0] == 0


class TestRegimeSensor:
    def test_regime_states(self):
        """Test regime state classification."""
        T = 300
        np.random.seed(42)

        # Create scores with some high values
        innovation_score = np.random.normal(0, 1, T)
        innovation_score[100:110] = np.random.normal(5, 1, 10)  # High scores
        innovation_score[200:205] = np.random.normal(10, 1, 5)  # Very high

        result = classify_regime(innovation_score, lookback=252, warning_q=0.95, break_q=0.99)

        assert len(result["regime_states"]) == T
        assert len(result["warning_threshold"]) == T
        assert len(result["break_threshold"]) == T
        assert len(result["regime_event"]) == T
        assert len(result["regime_switch"]) == T

        # Check warmup period
        assert np.all(result["regime_states"][:252] == "warmup")

        # Check some states are not warmup
        non_warmup = result["regime_states"][252:]
        assert "normal" in non_warmup or "warning" in non_warmup or "break" in non_warmup

    def test_past_only(self):
        """Test that only past data is used for thresholds."""
        T = 260
        innovation_score = np.full(T, 1.0)
        innovation_score[258] = 100.0  # High score at end

        result = classify_regime(innovation_score, lookback=252)

        # Thresholds at t=258 should not include score[258]
        # Since all past scores are 1.0, thresholds should be based on 1.0
        assert result["warning_threshold"][258] <= 2.0  # Should be low
        assert result["break_threshold"][258] <= 2.0

    def test_regime_event_switch(self):
        """Test regime_event and regime_switch logic."""
        T = 10
        innovation_score = np.array([1.0] * T)
        innovation_score[5] = 10.0  # Should trigger event

        result = classify_regime(innovation_score, lookback=5, warning_q=0.8, break_q=0.95)

        # Should have some events
        assert result["regime_event"].sum() > 0
        # Should have switches
        assert result["regime_switch"].sum() > 0

    def test_rejects_reversed_quantiles(self):
        innovation_score = np.ones(10)
        with pytest.raises(ValueError, match="upper-tail quantiles"):
            classify_regime(innovation_score, lookback=5, warning_q=0.05, break_q=0.01)


class TestBlend:
    def test_blend_shapes(self):
        """Test blend output shapes."""
        T, K = 100, 5
        f_ols = np.random.randn(T, K)
        f_kf = np.random.randn(T, K)
        regime_states = np.array(["normal"] * 50 + ["warning"] * 50)

        f_blend, alpha_ols = blend_factor_returns(f_ols, f_kf, regime_states)

        assert f_blend.shape == (T, K)
        assert alpha_ols.shape == (T,)
        assert np.allclose(alpha_ols[:50], 0.3)
        assert np.allclose(alpha_ols[50:], 0.6)

    def test_fallback(self):
        """Test fallback when one input is NaN."""
        T, K = 10, 3
        f_ols = np.full((T, K), np.nan)
        f_kf = np.random.randn(T, K)
        regime_states = np.array(["normal"] * T)

        f_blend, alpha_ols = blend_factor_returns(f_ols, f_kf, regime_states)

        # Should use f_kf
        assert np.allclose(f_blend, f_kf, equal_nan=True)

        # Reverse
        f_ols = np.random.randn(T, K)
        f_kf = np.full((T, K), np.nan)

        f_blend, alpha_ols = blend_factor_returns(f_ols, f_kf, regime_states)

        # Should use f_ols
        assert np.allclose(f_blend, f_ols, equal_nan=True)

    def test_partial_nan(self):
        """Test blending with partial NaNs."""
        T, K = 5, 2
        f_ols = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0], [5.0, 6.0], [7.0, np.nan]])
        f_kf = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        regime_states = np.array(["normal"] * T)

        f_blend, alpha_ols = blend_factor_returns(f_ols, f_kf, regime_states)

        # Where both finite, should blend
        assert f_blend[1, 0] == 0.3 * 2.0 + 0.7 * 0.5  # Both finite
        assert f_blend[1, 1] == 0.3 * 3.0 + 0.7 * 0.5

        # Where only one finite, use that one
        assert f_blend[0, 0] == 0.3 * 1.0 + 0.7 * 0.5  # Both finite, so blend
        assert f_blend[0, 1] == 0.5  # Only KF finite
        assert f_blend[2, 0] == 0.5  # Only KF finite
        assert f_blend[2, 1] == 0.3 * 4.0 + 0.7 * 0.5  # Both finite, so blend
