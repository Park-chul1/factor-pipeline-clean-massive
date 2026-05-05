from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_factor_returns(
    X: np.ndarray,
    r: np.ndarray,
    min_names: int = 30,
    ridge: float = 1e-4,
    require_full_rank_without_ridge: bool = True,
) -> np.ndarray:
    """Estimate daily cross-sectional factor returns.

    With ridge > 0, the system is allowed to be underdetermined (N < K) because
    the ridge penalty makes the normal equations numerically solvable. With
    ridge == 0, require more observations than factors to avoid unstable OLS.
    """
    T, N, K = X.shape
    f = np.full((T, K), np.nan, dtype=float)
    if K == 0:
        return f

    for t in range(T):
        Xt = X[t]
        rt = r[t]
        mask = np.isfinite(rt) & np.any(np.isfinite(Xt), axis=1)
        n_obs = int(mask.sum())
        if ridge > 0:
            required = min_names
        else:
            required = max(min_names, K + 2) if require_full_rank_without_ridge else min_names
        if n_obs < required:
            continue

        A = np.where(np.isfinite(Xt[mask]), Xt[mask], 0.0)
        y = rt[mask]
        # Center y cross-sectionally so intercept-like market return does not
        # get forced into style factors. A true intercept can be added later.
        y = y - np.nanmean(y)

        if ridge > 0:
            G = A.T @ A + ridge * np.eye(K)
            b = A.T @ y
            try:
                f[t] = np.linalg.solve(G, b)
            except np.linalg.LinAlgError:
                f[t] = np.linalg.lstsq(G, b, rcond=None)[0]
        else:
            try:
                f[t] = np.linalg.lstsq(A, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
    return f
