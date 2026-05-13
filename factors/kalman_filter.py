from __future__ import annotations

import numpy as np


def kalman_filter_factor_returns_direct_observation(
    X: np.ndarray,
    r: np.ndarray,
    q_scale: float = 1e-4,
    r_scale: float = 1e-2,
    p0_scale: float = 1.0,
) -> dict:
    """Estimate factor returns using Kalman filter with direct stock-level observations.

    WARNING: This implementation forms an N x N covariance matrix and is not recommended
    for large universes. Use kalman_filter_factor_returns_from_ols instead.

    State-space model:
        f_t = A f_{t-1} + w_t
        r_t = X_t f_t + v_t

    Where:
    - A = identity matrix
    - Q = q_scale * I (process noise covariance)
    - R = r_scale * I (measurement noise covariance)
    - P0 = p0_scale * I (initial state covariance)

    Args:
        X: Factor exposure tensor, shape (T, N, K)
        r: Forward returns, shape (T, N)
        q_scale: Process noise scale
        r_scale: Measurement noise scale
        p0_scale: Initial state covariance scale

    Returns:
        dict with keys:
        - f_kf: Kalman-filtered factor returns, shape (T, K)
        - innovation_norm: Innovation norm, shape (T,)
        - innovation_score: Innovation score, shape (T,)
        - valid_obs_count: Number of valid observations per date, shape (T,)
    """
    T, N, K = X.shape
    if r.shape != (T, N):
        raise ValueError(f"r shape {r.shape} must match (T, N) = {(T, N)}")

    # Initialize
    f_kf = np.full((T, K), np.nan, dtype=float)
    innovation_norm = np.full(T, np.nan, dtype=float)
    innovation_score = np.full(T, np.nan, dtype=float)
    valid_obs_count = np.zeros(T, dtype=int)

    # State transition: f_t = I * f_{t-1} + w_t
    A = np.eye(K)
    Q = q_scale * np.eye(K)
    R_scale = r_scale  # Measurement noise scale

    # Initial state
    f_prev = np.zeros(K)
    P_prev = p0_scale * np.eye(K)

    for t in range(T):
        Xt = X[t]
        rt = r[t]

        # Find valid observations: finite r and all finite X
        valid_mask = np.isfinite(rt) & np.all(np.isfinite(Xt), axis=1)
        valid_obs_count[t] = int(valid_mask.sum())

        if valid_obs_count[t] < K + 5:
            # Too few observations, carry forward previous state
            f_kf[t] = f_prev
            continue

        # Extract valid data
        X_valid = Xt[valid_mask]  # shape (n_valid, K)
        r_valid = rt[valid_mask]  # shape (n_valid,)

        n_valid = valid_obs_count[t]

        # Predict
        f_pred = A @ f_prev
        P_pred = A @ P_prev @ A.T + Q

        # Measurement noise covariance: R_scale * I_{n_valid}
        R = R_scale * np.eye(n_valid)

        # Innovation
        y_pred = X_valid @ f_pred
        y = r_valid - y_pred  # innovation vector, shape (n_valid,)

        # Innovation covariance
        H = X_valid  # measurement matrix
        S = H @ P_pred @ H.T + R

        # Add jitter if needed
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S += 1e-8 * np.eye(n_valid)
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                # Skip if still fails
                f_kf[t] = f_prev
                continue

        # Kalman gain
        K_gain = P_pred @ H.T @ S_inv

        # Update
        f_update = f_pred + K_gain @ y
        P_update = P_pred - K_gain @ H @ P_pred

        # Store
        f_kf[t] = f_update

        # Innovation norm: ||y|| / sqrt(n_valid)
        innovation_norm[t] = np.linalg.norm(y) / np.sqrt(n_valid)

        # Innovation score: y.T @ S_inv @ y
        innovation_score[t] = y @ S_inv @ y

        # Update for next iteration
        f_prev = f_update
        P_prev = P_update

    return {
        "f_kf": f_kf,
        "innovation_norm": innovation_norm,
        "innovation_score": innovation_score,
        "valid_obs_count": valid_obs_count,
    }


def kalman_filter_factor_returns_from_ols(
    f_ols: np.ndarray,
    q_scale: float = 1e-5,
    r_scale: float = 1e-3,
    p0_scale: float = 1.0,
) -> dict:
    """Factor-level Kalman smoother using OLS factor returns as observations.

    This is a factor-level Kalman smoother. The cross-sectional compression from N stocks
    to K factor returns is already done by OLS. This avoids the expensive N x N innovation
    covariance from direct-observation Kalman filtering.

    State-space model:
        f_t = A f_{t-1} + w_t
        f_ols[t] = I f_t + eta_t

    Where:
    - A = identity matrix
    - Q = q_scale * I (process noise covariance)
    - R = r_scale * I (measurement noise covariance)
    - P0 = p0_scale * I (initial state covariance)

    Args:
        f_ols: OLS factor returns, shape (T, K)
        q_scale: Process noise scale
        r_scale: Measurement noise scale
        p0_scale: Initial state covariance scale

    Returns:
        dict with keys:
        - f_kf: Kalman-filtered factor returns, shape (T, K)
        - innovation_norm: Innovation norm, shape (T,)
        - innovation_score: Innovation score, shape (T,)
        - valid_factor_count: Number of valid factor observations per date, shape (T,)
    """
    T, K = f_ols.shape

    # Initialize
    f_kf = np.full((T, K), np.nan, dtype=float)
    innovation_norm = np.full(T, np.nan, dtype=float)
    innovation_score = np.full(T, np.nan, dtype=float)
    valid_factor_count = np.zeros(T, dtype=int)

    # State transition: f_t = I * f_{t-1} + w_t
    A = np.eye(K)
    Q = q_scale * np.eye(K)
    R_scale = r_scale  # Measurement noise scale

    # Initial state
    f_prev = np.zeros(K)
    P_prev = p0_scale * np.eye(K)

    for t in range(T):
        f_ols_t = f_ols[t]

        # Find valid factor observations: finite f_ols
        valid_mask = np.isfinite(f_ols_t)
        valid_factor_count[t] = int(valid_mask.sum())

        if valid_factor_count[t] < 1:
            # No valid observations, carry forward previous state
            f_kf[t] = f_prev
            continue

        # Extract valid data
        valid_indices = np.where(valid_mask)[0]
        z = f_ols_t[valid_indices]  # shape (n_valid,)

        n_valid = valid_factor_count[t]

        # Predict
        f_pred = A @ f_prev
        P_pred = A @ P_prev @ A.T + Q

        # Measurement matrix: I for valid factors
        H = np.eye(K)[valid_indices]  # shape (n_valid, K)

        # Measurement noise covariance: R_scale * I_{n_valid}
        R_t = R_scale * np.eye(n_valid)

        # Innovation
        y = z - H @ f_pred  # shape (n_valid,)

        # Innovation covariance
        S = H @ P_pred @ H.T + R_t  # shape (n_valid, n_valid), but n_valid <= K

        # Assert no large matrices
        assert S.shape[0] <= K, f"S shape {S.shape} exceeds K={K}"

        # Use solve instead of inv
        try:
            sol_y = np.linalg.solve(S, y)
            sol_HP = np.linalg.solve(S, H @ P_pred)
            K_gain = sol_HP.T  # shape (K, n_valid)
        except np.linalg.LinAlgError:
            # Add jitter
            S_jitter = S + 1e-8 * np.eye(n_valid)
            try:
                sol_y = np.linalg.solve(S_jitter, y)
                sol_HP = np.linalg.solve(S_jitter, H @ P_pred)
                K_gain = sol_HP.T
            except np.linalg.LinAlgError:
                # Skip if still fails
                f_kf[t] = f_prev
                continue

        # Update
        f_update = f_pred + K_gain @ y
        P_update = P_pred - K_gain @ H @ P_pred

        # Store
        f_kf[t] = f_update

        # Innovation norm: ||y|| / sqrt(n_valid)
        innovation_norm[t] = np.linalg.norm(y) / np.sqrt(n_valid)

        # Innovation score: y.T @ solve(S, y)
        innovation_score[t] = y @ sol_y

        # Update for next iteration
        f_prev = f_update
        P_prev = P_update

    return {
        "f_kf": f_kf,
        "innovation_norm": innovation_norm,
        "innovation_score": innovation_score,
        "valid_factor_count": valid_factor_count,
    }


# Backward compatibility alias
def kalman_filter_factor_returns(
    X: np.ndarray | None = None,
    r: np.ndarray | None = None,
    f_ols: np.ndarray | None = None,
    q_scale: float = 1e-4,
    r_scale: float = 1e-2,
    p0_scale: float = 1.0,
) -> dict:
    """Backward compatibility wrapper.

    If f_ols is provided, uses factor-level Kalman.
    Otherwise, uses direct observation (deprecated).
    """
    if f_ols is not None:
        print("WARNING: Using deprecated kalman_filter_factor_returns with f_ols. Use kalman_filter_factor_returns_from_ols directly.")
        return kalman_filter_factor_returns_from_ols(f_ols, q_scale, r_scale, p0_scale)
    elif X is not None and r is not None:
        print("WARNING: Using direct-observation Kalman filter. This forms N x N matrices and is not recommended for large universes.")
        return kalman_filter_factor_returns_direct_observation(X, r, q_scale, r_scale, p0_scale)
    else:
        raise ValueError("Must provide either f_ols or both X and r")