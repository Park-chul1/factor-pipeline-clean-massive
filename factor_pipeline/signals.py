from __future__ import annotations

import numpy as np


def predict_factor_returns(
    f: np.ndarray,
    method: str = "latest",
    lookback: int = 20,
    ewma_halflife: float = 20.0,
    min_periods: int = 5,
) -> np.ndarray:
    """Build point-in-time predictions of factor returns.

    f[t] is the realized factor return estimated from X[t] and forward return r[t].
    Therefore f[t] is not available when forming a position at date t.
    This function predicts f_pred[t] using only f[:t].

    Supported methods:
    - latest: f_pred[t] = most recent finite f before t
    - rolling: f_pred[t] = mean of f over the previous lookback rows
    - ewma: exponentially weighted mean of past f, using ewma_halflife
    - zero: no expected factor return, useful sanity check
    - oracle: f_pred[t] = f[t], intentionally look-ahead, diagnostic upper bound only
    """
    f = np.asarray(f, dtype=float)
    if f.ndim != 2:
        raise ValueError(f"f must be 2-D [T,K], got shape {f.shape}")
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if min_periods <= 0:
        raise ValueError("min_periods must be positive")
    if ewma_halflife <= 0:
        raise ValueError("ewma_halflife must be positive")

    method = method.lower()
    T, K = f.shape
    pred = np.full((T, K), np.nan, dtype=float)

    if method == "oracle":
        return f.copy()
    if method == "zero":
        pred[:] = 0.0
        return pred
    if method not in {"latest", "rolling", "ewma"}:
        raise ValueError(f"unknown method: {method}")

    for t in range(T):
        hist = f[:t]
        if hist.size == 0 or not np.isfinite(hist).any():
            continue

        if method == "latest":
            valid_rows = np.where(np.isfinite(hist).any(axis=1))[0]
            if valid_rows.size == 0:
                continue
            last = hist[valid_rows[-1]].copy()
            pred[t] = np.where(np.isfinite(last), last, 0.0)

        elif method == "rolling":
            window = hist[-lookback:]
            counts = np.isfinite(window).sum(axis=0)
            values = np.full(K, np.nan, dtype=float)
            ok = counts >= min_periods
            if ok.any():
                values[ok] = np.nanmean(window[:, ok], axis=0)
            pred[t] = np.where(np.isfinite(values), values, 0.0)

        elif method == "ewma":
            window = hist[-lookback:]
            age = np.arange(len(window) - 1, -1, -1, dtype=float)
            weights = 0.5 ** (age / ewma_halflife)
            values = np.full(K, np.nan, dtype=float)
            for k in range(K):
                x = window[:, k]
                m = np.isfinite(x)
                if int(m.sum()) >= min_periods:
                    wk = weights[m]
                    values[k] = float(np.sum(wk * x[m]) / np.sum(wk))
            pred[t] = np.where(np.isfinite(values), values, 0.0)

    return pred


def make_scores(X: np.ndarray, f_pred: np.ndarray, tradable_mask: np.ndarray | None = None) -> np.ndarray:
    """Compute stock-level scores score[t,n] = X[t,n,:] @ f_pred[t,:]."""
    X = np.asarray(X, dtype=float)
    f_pred = np.asarray(f_pred, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"X must be 3-D [T,N,K], got shape {X.shape}")
    if f_pred.ndim != 2:
        raise ValueError(f"f_pred must be 2-D [T,K], got shape {f_pred.shape}")
    T, N, K = X.shape
    if f_pred.shape != (T, K):
        raise ValueError(f"f_pred shape {f_pred.shape} must equal {(T, K)}")

    X_safe = np.where(np.isfinite(X), X, 0.0)
    f_safe = np.where(np.isfinite(f_pred), f_pred, 0.0)
    scores = np.einsum("tnk,tk->tn", X_safe, f_safe)

    valid_pred = np.isfinite(f_pred).any(axis=1)
    scores[~valid_pred, :] = np.nan

    if tradable_mask is not None:
        mask = np.asarray(tradable_mask, dtype=bool)
        if mask.shape != (T, N):
            raise ValueError(f"tradable_mask shape {mask.shape} must equal {(T, N)}")
        scores = np.where(mask, scores, np.nan)

    return scores


def make_quantile_long_short_weights(
    score: np.ndarray,
    tradable: np.ndarray | None = None,
    quantile: float = 0.10,
    gross: float = 2.0,
    min_names_per_side: int = 5,
) -> np.ndarray:
    """Equal-weight top/bottom quantile long-short portfolio for one date.

    With gross=2.0, the long leg sums to +1 and the short leg sums to -1.
    """
    score = np.asarray(score, dtype=float)
    if score.ndim != 1:
        raise ValueError(f"score must be 1-D, got shape {score.shape}")
    if not (0 < quantile < 0.5):
        raise ValueError("quantile must be between 0 and 0.5")
    if gross <= 0:
        raise ValueError("gross must be positive")

    valid = np.isfinite(score)
    if tradable is not None:
        tmask = np.asarray(tradable, dtype=bool)
        if tmask.shape != score.shape:
            raise ValueError(f"tradable shape {tmask.shape} must equal {score.shape}")
        valid &= tmask

    w = np.zeros_like(score, dtype=float)
    vals = score[valid]
    min_total = 2 * min_names_per_side
    if vals.size < min_total:
        return w

    short_cut = np.nanquantile(vals, quantile)
    long_cut = np.nanquantile(vals, 1.0 - quantile)
    long_mask = valid & (score >= long_cut)
    short_mask = valid & (score <= short_cut)

    n_long = int(long_mask.sum())
    n_short = int(short_mask.sum())
    if n_long < min_names_per_side or n_short < min_names_per_side:
        return w

    leg = gross / 2.0
    w[long_mask] = leg / n_long
    w[short_mask] = -leg / n_short
    return w


def make_positions_from_scores(
    scores: np.ndarray,
    tradable_mask: np.ndarray | None = None,
    quantile: float = 0.10,
    gross: float = 2.0,
    min_names_per_side: int = 5,
) -> np.ndarray:
    """Build a T x N weight matrix from stock-level scores."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D [T,N], got shape {scores.shape}")
    T, N = scores.shape
    if tradable_mask is not None:
        mask = np.asarray(tradable_mask, dtype=bool)
        if mask.shape != (T, N):
            raise ValueError(f"tradable_mask shape {mask.shape} must equal {(T, N)}")
    else:
        mask = None

    weights = np.zeros((T, N), dtype=float)
    for t in range(T):
        weights[t] = make_quantile_long_short_weights(
            scores[t],
            tradable=None if mask is None else mask[t],
            quantile=quantile,
            gross=gross,
            min_names_per_side=min_names_per_side,
        )
    return weights
