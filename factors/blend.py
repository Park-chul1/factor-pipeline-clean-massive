from __future__ import annotations

import numpy as np


def blend_factor_returns(
    f_ols: np.ndarray,
    f_kf: np.ndarray,
    regime_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend OLS and Kalman factor returns based on regime.

    Blend rule:
    - normal: alpha_ols = 0.3
    - warning: alpha_ols = 0.6
    - break: alpha_ols = 0.8
    - warmup/unknown: alpha_ols = 0.5

    Formula: f_blend[t] = alpha_ols[t] * f_ols[t] + (1 - alpha_ols[t]) * f_kf[t]

    Fallback:
    - If only f_ols[t] is finite, use f_ols[t]
    - If only f_kf[t] is finite, use f_kf[t]
    - If neither, NaN

    Args:
        f_ols: OLS factor returns, shape (T, K)
        f_kf: Kalman factor returns, shape (T, K)
        regime_states: Regime states, shape (T,)

    Returns:
        tuple: (f_blend, alpha_ols)
        - f_blend: Blended factor returns, shape (T, K)
        - alpha_ols: OLS weights, shape (T,)
    """
    T, K = f_ols.shape
    if f_kf.shape != (T, K):
        raise ValueError(f"f_kf shape {f_kf.shape} must match f_ols shape {(T, K)}")
    if len(regime_states) != T:
        raise ValueError(f"regime_states length {len(regime_states)} must match T={T}")

    f_blend = np.full((T, K), np.nan, dtype=float)
    alpha_ols = np.full(T, np.nan, dtype=float)

    alpha_map = {
        "normal": 0.3,
        "warning": 0.6,
        "break": 0.8,
        "warmup": 0.5,
        "unknown": 0.5,
    }

    for t in range(T):
        state = regime_states[t]
        alpha = alpha_map.get(state, 0.5)
        alpha_ols[t] = alpha

        f_ols_t = f_ols[t]
        f_kf_t = f_kf[t]

        # Check which are finite
        ols_finite = np.isfinite(f_ols_t)
        kf_finite = np.isfinite(f_kf_t)

        if np.any(ols_finite) and np.any(kf_finite):
            # Both have some finite, blend where both finite
            both_finite = ols_finite & kf_finite
            f_blend[t] = np.where(
                both_finite,
                alpha * f_ols_t + (1 - alpha) * f_kf_t,
                np.where(ols_finite, f_ols_t, f_kf_t)
            )
        elif np.any(ols_finite):
            f_blend[t] = f_ols_t
        elif np.any(kf_finite):
            f_blend[t] = f_kf_t
        # else: remains NaN

    return f_blend, alpha_ols