from __future__ import annotations

import numpy as np
import pandas as pd


def dynamic_factor_diagnostics(
    dates: np.ndarray,
    tickers: list[str],
    factor_names: list[str],
    f_ols: np.ndarray,
    f_kf: np.ndarray,
    f_blend: np.ndarray,
    innovation_score: np.ndarray,
    innovation_norm: np.ndarray,
    regime_states: np.ndarray,
    warning_threshold: np.ndarray,
    break_threshold: np.ndarray,
    valid_obs_count: np.ndarray,  # Now valid_factor_count for new Kalman
    alpha_ols: np.ndarray,
    q_scale: float,
    r_scale: float,
    lookback: int,
    warning_q: float,
    break_q: float,
) -> dict:
    """Generate diagnostics summary for dynamic factor estimation."""

    T, K = f_ols.shape
    N = len(tickers)

    # Basic stats
    finite_ols = np.isfinite(f_ols)
    finite_kf = np.isfinite(f_kf)
    finite_blend = np.isfinite(f_blend)
    finite_innov = np.isfinite(innovation_score)

    coverage_ols = finite_ols.sum() / finite_ols.size
    coverage_kf = finite_kf.sum() / finite_kf.size
    coverage_blend = finite_blend.sum() / finite_blend.size
    coverage_innov = finite_innov.sum() / len(innovation_score)

    # Valid obs stats (now valid_factor_count for factor-level Kalman)
    mean_valid_obs = valid_obs_count.mean()
    min_valid_obs = valid_obs_count.min()
    # For factor-level Kalman, skip when no valid factors; for direct obs, this was K+5
    skipped_dates = (valid_obs_count == 0).sum()  # Updated for factor-level Kalman

    # Regime counts
    unique_states, counts = np.unique(regime_states, return_counts=True)
    state_counts = dict(zip(unique_states, counts))

    # Alpha stats
    finite_alpha = alpha_ols[np.isfinite(alpha_ols)]
    mean_alpha = finite_alpha.mean() if len(finite_alpha) > 0 else np.nan

    # Innovation spikes
    if finite_innov.sum() > 0:
        top_indices = np.argsort(innovation_score[finite_innov])[-5:][::-1]
        top_scores = innovation_score[finite_innov][top_indices]
        top_dates = dates[finite_innov][top_indices]
        top_warn = warning_threshold[finite_innov][top_indices]
        top_break = break_threshold[finite_innov][top_indices]
        top_states = regime_states[finite_innov][top_indices]

        top_spikes = []
        for i in range(len(top_indices)):
            top_spikes.append({
                "date": str(top_dates[i]),
                "score": float(top_scores[i]),
                "warn_th": float(top_warn[i]),
                "break_th": float(top_break[i]),
                "state": str(top_states[i]),
            })
    else:
        top_spikes = []

    # Recent regime events
    recent_events = []
    event_mask = (regime_states == "warning") | (regime_states == "break")
    if event_mask.sum() > 0:
        event_indices = np.where(event_mask)[0][-5:]  # Last 5 events
        for idx in event_indices:
            recent_events.append({
                "date": str(dates[idx]),
                "state": str(regime_states[idx]),
                "score": float(innovation_score[idx]),
                "warn_th": float(warning_threshold[idx]),
                "break_th": float(break_threshold[idx]),
            })

    # Warnings
    warnings = []
    if skipped_dates > T * 0.1:
        warnings.append(f"High skipped dates: {skipped_dates}/{T} ({100*skipped_dates/T:.1f}%)")
    if coverage_innov < 0.8:
        warnings.append(f"Low innovation coverage: {coverage_innov:.1%}")
    if state_counts.get("break", 0) > T * 0.05:
        warnings.append(f"Frequent breaks: {state_counts.get('break', 0)}/{T} ({100*state_counts.get('break', 0)/T:.1f}%)")
    if coverage_blend < coverage_ols * 0.9:
        warnings.append(f"Blend coverage {coverage_blend:.1%} much lower than OLS {coverage_ols:.1%}")

    return {
        "dates": T,
        "assets": N,
        "factors": K,
        "kalman": {
            "q_scale": q_scale,
            "r_scale": r_scale,
            "mean_valid_obs": float(mean_valid_obs),
            "min_valid_obs": int(min_valid_obs),
            "skipped_dates": int(skipped_dates),
        },
        "regime": {
            "lookback": lookback,
            "warning_q": warning_q,
            "break_q": break_q,
            "normal_days": int(state_counts.get("normal", 0)),
            "warning_days": int(state_counts.get("warning", 0)),
            "break_days": int(state_counts.get("break", 0)),
            "unknown_warmup_days": int(state_counts.get("unknown", 0) + state_counts.get("warmup", 0)),
        },
        "blend": {
            "mean_alpha_ols": float(mean_alpha),
            "normal_alpha_ols": 0.3,
            "warning_alpha_ols": 0.6,
            "break_alpha_ols": 0.8,
        },
        "top_innovation_spikes": top_spikes,
        "recent_regime_events": recent_events,
        "warnings": warnings,
        "saved_files": [
            "data/processed/factor_returns_ols.csv",
            "data/processed/factor_returns_kalman.csv",
            "data/processed/factor_returns_blend.csv",
            "data/processed/innovation_scores.csv",
            "data/processed/innovation_norm.csv",
            "data/processed/regime_states.csv",
            "data/processed/regime_thresholds.csv",
            "data/processed/blend_alpha_ols.csv",
            "data/processed/dynamic_factor_summary.json",
        ],
    }


def print_dynamic_summary(diag: dict) -> None:
    """Print clean terminal diagnostics summary."""
    print("=" * 50)
    print("Dynamic Factor Estimation")
    print("=" * 50)
    print(f"Dates: {diag['dates']}")
    print(f"Assets: {diag['assets']}")
    print(f"Factors: {diag['factors']}")
    print()

    print("Kalman Filter")
    print("-" * 20)
    kf = diag["kalman"]
    print(f"q_scale: {kf['q_scale']:.0e}")
    print(f"r_scale: {kf['r_scale']:.0e}")
    print(f"mean valid obs/date: {kf['mean_valid_obs']:.1f}")
    print(f"min valid obs/date: {kf['min_valid_obs']}")
    print(f"skipped dates: {kf['skipped_dates']}")
    print()

    print("Innovation Sensor")
    print("-" * 20)
    reg = diag["regime"]
    print(f"lookback: {reg['lookback']}")
    print(f"warning quantile: {reg['warning_q']:.2f}")
    print(f"break quantile: {reg['break_q']:.2f}")
    print(f"normal days: {reg['normal_days']}")
    print(f"warning days: {reg['warning_days']}")
    print(f"break days: {reg['break_days']}")
    print(f"unknown/warmup days: {reg['unknown_warmup_days']}")
    print()

    if diag["top_innovation_spikes"]:
        print("Top innovation spikes:")
        print("date          score       warn_th     break_th    state")
        for spike in diag["top_innovation_spikes"]:
            print(f"{spike['date']}    {spike['score']:>8.2f}    {spike['warn_th']:>8.2f}    {spike['break_th']:>8.2f}    {spike['state']}")
        print()

    if diag["recent_regime_events"]:
        print("Recent regime events:")
        print("date          state       score       warn_th     break_th")
        for event in diag["recent_regime_events"]:
            print(f"{event['date']}    {event['state']:<8}    {event['score']:>8.2f}    {event['warn_th']:>8.2f}    {event['break_th']:>8.2f}")
        print()

    print("Blend Weights")
    print("-" * 20)
    blend = diag["blend"]
    print(f"mean alpha_ols: {blend['mean_alpha_ols']:.2f}")
    print(f"normal alpha_ols: {blend['normal_alpha_ols']}")
    print(f"warning alpha_ols: {blend['warning_alpha_ols']}")
    print(f"break alpha_ols: {blend['break_alpha_ols']}")
    print()

    print("Saved:")
    for f in diag["saved_files"]:
        print(f"  {f}")

    if diag["warnings"]:
        print()
        print("Warnings:")
        for w in diag["warnings"]:
            print(f"  {w}")