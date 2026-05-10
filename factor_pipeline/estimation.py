from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_RIDGE_GRID = (
    1e-8,
    3e-8,
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    3e-1,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
    3000.0,
    10000.0,
    30000.0,
    100000.0,
)


def _solve_ridge_qr(A: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """Solve ridge regression through an augmented least-squares QR system."""
    n_factors = A.shape[1]
    if ridge > 0:
        A_aug = np.vstack([A, np.sqrt(ridge) * np.eye(n_factors)])
        y_aug = np.concatenate([y, np.zeros(n_factors, dtype=float)])
    else:
        A_aug = A
        y_aug = y

    q, r = np.linalg.qr(A_aug, mode="reduced")
    qty = q.T @ y_aug
    try:
        return np.linalg.solve(r, qty)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(r, qty, rcond=None)[0]


def _solve_ridge_normal(A: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    n_factors = A.shape[1]
    if ridge > 0:
        G = A.T @ A + ridge * np.eye(n_factors)
        b = A.T @ y
        try:
            return np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(G, b, rcond=None)[0]
    return np.linalg.lstsq(A, y, rcond=None)[0]


def _select_ridge_gcv(A: np.ndarray, y: np.ndarray, ridge_grid: np.ndarray) -> float:
    """Pick lambda by generalized cross-validation for one cross-section."""
    ridge_grid = np.asarray(ridge_grid, dtype=float)
    ridge_grid = ridge_grid[np.isfinite(ridge_grid) & (ridge_grid >= 0)]
    if ridge_grid.size == 0:
        raise ValueError("ridge_grid must contain at least one non-negative finite value")

    try:
        _, singular_values, vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return float(np.nanmedian(ridge_grid))

    # A @ V = U * S, so this equals S * U.T @ y.
    s_uty = (A @ vt.T).T @ y
    s2 = singular_values ** 2
    y_norm2 = float(y @ y)
    best_score = np.inf
    best_lambda = float(ridge_grid[0])

    for lam in ridge_grid:
        beta = vt.T @ (s_uty / (s2 + lam))
        resid = y - A @ beta
        rss = float(resid @ resid)
        df = float(np.sum(s2 / (s2 + lam)))
        denom = max(A.shape[0] - df, 1e-12)
        score = (rss / A.shape[0]) / ((denom / A.shape[0]) ** 2)
        if not np.isfinite(score):
            score = y_norm2
        if score < best_score:
            best_score = score
            best_lambda = float(lam)

    return best_lambda


def estimate_factor_returns(
    X: np.ndarray,
    r: np.ndarray,
    min_names: int = 30,
    ridge: float | str = 1e-4,
    require_full_rank_without_ridge: bool = True,
    universe_mask: np.ndarray | pd.DataFrame | None = None,
    ridge_grid: list[float] | tuple[float, ...] | np.ndarray | None = None,
    ridge_selection: str = "fixed",
    solver: str = "qr",
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, pd.DataFrame]:
    """Estimate daily cross-sectional factor returns.

    With ridge > 0, the system is allowed to be underdetermined (N < K) because
    the ridge penalty makes the normal equations numerically solvable. With
    ridge == 0, require more observations than factors to avoid unstable OLS.
    """
    T, N, K = X.shape
    mask_arr = None
    if universe_mask is not None:
        mask_arr = np.asarray(universe_mask, dtype=bool)
        if mask_arr.shape != (T, N):
            raise ValueError(f"universe_mask shape {mask_arr.shape} must equal {(T, N)}")

    ridge_selection = "gcv" if str(ridge).lower() == "auto" else ridge_selection.lower()
    solver = solver.lower()
    if ridge_selection not in {"fixed", "gcv"}:
        raise ValueError("ridge_selection must be 'fixed' or 'gcv'")
    if solver not in {"qr", "normal"}:
        raise ValueError("solver must be 'qr' or 'normal'")

    fixed_ridge = np.nan if str(ridge).lower() == "auto" else float(ridge)
    grid = np.asarray(ridge_grid if ridge_grid is not None else DEFAULT_RIDGE_GRID, dtype=float)

    f = np.full((T, K), np.nan, dtype=float)
    selected_ridge = np.full(T, np.nan, dtype=float)
    n_observations = np.zeros(T, dtype=int)
    if K == 0:
        diag = pd.DataFrame({"selected_ridge": selected_ridge, "n_observations": n_observations})
        return (f, diag) if return_diagnostics else f

    for t in range(T):
        Xt = X[t]
        rt = r[t]
        mask = np.isfinite(rt) & np.any(np.isfinite(Xt), axis=1)
        if mask_arr is not None:
            mask &= mask_arr[t]
        n_obs = int(mask.sum())
        n_observations[t] = n_obs
        ridge_t = _select_ridge_gcv(
            np.where(np.isfinite(Xt[mask]), Xt[mask], 0.0),
            rt[mask] - np.nanmean(rt[mask]),
            grid,
        ) if ridge_selection == "gcv" and n_obs > 0 else fixed_ridge
        selected_ridge[t] = ridge_t

        if ridge_t > 0:
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

        try:
            if solver == "qr":
                f[t] = _solve_ridge_qr(A, y, ridge_t)
            else:
                f[t] = _solve_ridge_normal(A, y, ridge_t)
        except np.linalg.LinAlgError:
            continue

    diag = pd.DataFrame({
        "selected_ridge": selected_ridge,
        "n_observations": n_observations,
    })
    return (f, diag) if return_diagnostics else f
