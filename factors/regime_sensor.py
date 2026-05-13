from __future__ import annotations

import numpy as np


def classify_regime(
    innovation_score: np.ndarray,
    lookback: int = 252,
    warning_q: float = 0.95,
    break_q: float = 0.99,
) -> dict:
    """Classify regime based on innovation scores using rolling quantiles.

    Args:
        innovation_score: Innovation scores, shape (T,)
        lookback: Lookback window for quantile calculation
        warning_q: Warning quantile threshold
        break_q: Break quantile threshold

    Returns:
        dict with keys:
        - regime_states: Regime states, shape (T,), values in {"warmup", "unknown", "normal", "warning", "break"}
        - warning_threshold: Rolling warning thresholds, shape (T,)
        - break_threshold: Rolling break thresholds, shape (T,)
        - regime_event: Boolean array indicating regime events, shape (T,)
        - regime_switch: Boolean array indicating regime switches, shape (T,)
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if not (0.0 < warning_q < break_q < 1.0):
        raise ValueError("expected upper-tail quantiles with 0 < warning_q < break_q < 1")

    T = len(innovation_score)
    regime_states = np.full(T, "unknown", dtype=object)
    warning_threshold = np.full(T, np.nan, dtype=float)
    break_threshold = np.full(T, np.nan, dtype=float)
    regime_event = np.zeros(T, dtype=bool)
    regime_switch = np.zeros(T, dtype=bool)

    for t in range(T):
        if t < lookback:
            regime_states[t] = "warmup"
            continue

        # Use only past scores: innovation_score[t-lookback:t]
        past_scores = innovation_score[t-lookback:t]
        finite_past = past_scores[np.isfinite(past_scores)]

        if len(finite_past) < lookback // 2:
            regime_states[t] = "unknown"
            continue

        # Compute quantiles
        warn_th = np.quantile(finite_past, warning_q)
        break_th = np.quantile(finite_past, break_q)

        warning_threshold[t] = warn_th
        break_threshold[t] = break_th

        score = innovation_score[t]
        if not np.isfinite(score):
            regime_states[t] = "unknown"
            continue

        if score > break_th:
            state = "break"
        elif score > warn_th:
            state = "warning"
        else:
            state = "normal"

        regime_states[t] = state

        # Regime event: warning or break
        regime_event[t] = state in {"warning", "break"}

        # Regime switch: state changed from previous
        if t > 0:
            regime_switch[t] = regime_states[t] != regime_states[t-1]

    return {
        "regime_states": regime_states,
        "warning_threshold": warning_threshold,
        "break_threshold": break_threshold,
        "regime_event": regime_event,
        "regime_switch": regime_switch,
    }
