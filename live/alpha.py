from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from factor_pipeline.signals import predict_factor_returns
from live.config import LiveConfig


def load_or_compute_f_pred(config: LiveConfig, factor_names: list[str]) -> pd.Series:
    if not factor_names:
        return pd.Series(dtype=float)
    path = Path(config.FACTOR_PRED_PATH)
    if not path.exists():
        return pd.Series(0.0, index=factor_names)

    f = np.load(path)
    if f.ndim == 1:
        pred = f
    else:
        pred_matrix = predict_factor_returns(f, method="ewma", lookback=min(20, max(1, f.shape[0])), min_periods=1)
        pred = pred_matrix[-1]

    names = factor_names
    names_path = Path(config.FACTOR_NAMES_PATH)
    if names_path.exists():
        loaded = pd.read_csv(names_path)
        col = "factor" if "factor" in loaded.columns else loaded.columns[0]
        all_names = loaded[col].astype(str).tolist()
        mapping = {name: float(pred[i]) for i, name in enumerate(all_names[: len(pred)])}
        return pd.Series({name: mapping.get(name, 0.0) for name in names}, dtype=float)
    return pd.Series({name: float(pred[i]) if i < len(pred) and np.isfinite(pred[i]) else 0.0 for i, name in enumerate(names)})


def compute_alpha(X_now: pd.DataFrame, f_pred: pd.Series) -> pd.Series:
    if X_now.empty:
        return pd.Series(dtype=float)
    aligned = f_pred.reindex(X_now.columns).fillna(0.0)
    return pd.Series(X_now.to_numpy(dtype=float) @ aligned.to_numpy(dtype=float), index=X_now.index, name="alpha")


def standardize_alpha(alpha: pd.Series) -> pd.Series:
    vals = alpha.replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return pd.Series(np.nan, index=alpha.index, name="z_alpha")
    sd = float(vals.std(ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.nan, index=alpha.index, name="z_alpha")
    return ((alpha - float(vals.mean())) / sd).rename("z_alpha")


def estimated_cost_return(config: LiveConfig) -> float:
    return (config.COST_BPS + config.SLIPPAGE_BPS) / 10_000.0


def apply_signal_filters(
    alpha: pd.Series,
    z_alpha: pd.Series,
    latest_bars: pd.DataFrame,
    config: LiveConfig,
) -> pd.DataFrame:
    if alpha.empty:
        return pd.DataFrame(columns=["ticker", "alpha", "z_alpha", "dollar_volume", "passes_signal"])
    bars = latest_bars.copy()
    bars["dollar_volume"] = pd.to_numeric(bars["close"], errors="coerce") * pd.to_numeric(bars["volume"], errors="coerce")
    liquidity = bars.drop_duplicates("ticker", keep="last").set_index("ticker")["dollar_volume"]
    out = pd.DataFrame({"ticker": alpha.index, "alpha": alpha.values, "z_alpha": z_alpha.reindex(alpha.index).values})
    out["dollar_volume"] = out["ticker"].map(liquidity)
    cost = estimated_cost_return(config)
    min_net_after_cost = config.MIN_NET_ALPHA_AFTER_COST_BPS / 10_000.0
    out["passes_signal"] = (
        out["alpha"].abs().sub(cost).gt(min_net_after_cost)
        & out["z_alpha"].abs().gt(config.ALPHA_Z_THRESHOLD)
        & out["dollar_volume"].ge(config.MIN_DOLLAR_VOLUME)
    )
    return out
