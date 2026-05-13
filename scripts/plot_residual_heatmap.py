from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot squared residual diagnostics from saved factor pipeline outputs"
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing factor pipeline outputs")
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=99.5,
        help="Percentile used as the heatmap colorbar vmax",
    )
    parser.add_argument(
        "--sort-tickers-by",
        default="coverage",
        choices=["none", "coverage", "mean_sq_err", "residual_similarity"],
        help="Optional ticker sorting method for the heatmap",
    )
    parser.add_argument(
        "--bad-color",
        default="lightgray",
        help="Matplotlib color used for NaN/invalid heatmap cells",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.05,
        help="Finite residual fraction threshold used for low-coverage ticker diagnostics",
    )
    parser.add_argument(
        "--filter-low-coverage",
        action="store_true",
        help="Also apply coverage filtering to the main heatmap; cleaned heatmap is always saved",
    )
    parser.add_argument(
        "--extreme-clip-percentile",
        type=float,
        default=95.0,
        help="More aggressive clipping percentile for the separate cleaned heatmap",
    )
    parser.add_argument("--spike-days", type=int, default=20, help="Number of highest daily MSE dates to analyze")
    parser.add_argument(
        "--spike-top-tickers",
        type=int,
        default=10,
        help="Top contributing tickers saved per daily MSE spike date",
    )
    parser.add_argument("--top-tickers", type=int, default=50, help="Number of tickers in the MSE bar plot")
    parser.add_argument("--top-events", type=int, default=100, help="Number of largest residual events to save")
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(16.0, 8.0),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches for the heatmap",
    )
    return parser.parse_args()


def _find_existing(input_dir: Path, candidates: list[str], required: bool = True) -> Path | None:
    for name in candidates:
        path = input_dir / name
        if path.exists():
            return path
    if required:
        joined = ", ".join(candidates)
        raise FileNotFoundError(f"Missing required pipeline output in {input_dir}: expected one of {joined}")
    return None


def _read_single_column_csv(path: Path, preferred_column: str | None = None) -> list[str]:
    df = pd.read_csv(path)
    if df.empty and len(df.columns) == 1:
        return []
    if preferred_column and preferred_column in df.columns:
        series = df[preferred_column]
    else:
        series = df.iloc[:, 0]
    labels = []
    prefix = preferred_column or series.name or "label"
    for idx, value in series.items():
        if pd.isna(value) or str(value).strip() == "":
            labels.append(f"missing_{prefix}_{idx}")
        else:
            labels.append(str(value))
    return labels


def _validate_shapes(
    X: np.ndarray,
    r: np.ndarray,
    factor_returns: np.ndarray,
    dates: list[str],
    tickers: list[str],
    factor_names: list[str] | None,
) -> None:
    if X.ndim != 3:
        raise ValueError(f"X must have shape [T, N, K], got {X.shape}")
    T, N, K = X.shape
    if r.shape != (T, N):
        raise ValueError(f"r must have shape [T, N] = {(T, N)}, got {r.shape}")
    if factor_returns.shape != (T, K):
        raise ValueError(
            f"factor_returns must have shape [T, K] = {(T, K)}, got {factor_returns.shape}"
        )
    if len(dates) != T:
        raise ValueError(f"dates length must equal T={T}, got {len(dates)}")
    if len(tickers) != N:
        raise ValueError(f"tickers length must equal N={N}, got {len(tickers)}")
    if factor_names is not None and len(factor_names) != K:
        raise ValueError(f"factor_names length must equal K={K}, got {len(factor_names)}")


def load_pipeline_outputs(input_dir: str | Path) -> dict:
    input_dir = Path(input_dir)
    X_path = _find_existing(input_dir, ["X.npy", "exposures.npy"])
    r_path = _find_existing(input_dir, ["r.npy", "returns.npy", "forward_returns.npy"])
    f_path = _find_existing(input_dir, ["factor_returns.npy", "f.npy"])
    dates_path = _find_existing(input_dir, ["dates.csv", "date.csv"])
    tickers_path = _find_existing(input_dir, ["tickers.csv", "ticker.csv", "symbols.csv"])
    factor_names_path = _find_existing(input_dir, ["factor_names.csv", "factors.csv"], required=False)

    X = np.load(X_path)
    r = np.load(r_path)
    factor_returns = np.load(f_path)
    dates = _read_single_column_csv(dates_path, "date")
    tickers = _read_single_column_csv(tickers_path, "ticker")
    factor_names = None
    if factor_names_path is not None:
        factor_names = _read_single_column_csv(factor_names_path, "factor")

    _validate_shapes(X, r, factor_returns, dates, tickers, factor_names)
    return {
        "X": X,
        "r": r,
        "factor_returns": factor_returns,
        "dates": dates,
        "tickers": tickers,
        "factor_names": factor_names,
        "paths": {
            "X": str(X_path),
            "r": str(r_path),
            "factor_returns": str(f_path),
            "dates": str(dates_path),
            "tickers": str(tickers_path),
            "factor_names": str(factor_names_path) if factor_names_path else None,
        },
    }


def compute_residuals(X: np.ndarray, r: np.ndarray, factor_returns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_hat = np.einsum("tnk,tk->tn", X, factor_returns)
    residual = r - r_hat
    sq_err = residual**2
    return r_hat, residual, sq_err


def _finite_percentile(values: np.ndarray, percentile: float) -> float | None:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return None
    return float(np.percentile(finite_values, percentile))


def _finite_stat(values: np.ndarray, fn) -> float | None:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return None
    return float(fn(finite_values))


def _nanmean_axis(values: np.ndarray, axis: int) -> np.ndarray:
    finite = np.isfinite(values)
    counts = finite.sum(axis=axis)
    sums = np.where(finite, values, 0.0).sum(axis=axis)
    out = np.full(counts.shape, np.nan, dtype=float)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def compute_ticker_coverage(residual: np.ndarray) -> np.ndarray:
    return np.isfinite(residual).mean(axis=0)


def compute_valid_count(residual: np.ndarray) -> np.ndarray:
    return np.isfinite(residual).sum(axis=1)


def _ticker_order(
    sq_err: np.ndarray,
    residual: np.ndarray,
    sort_tickers_by: str,
    ticker_coverage: np.ndarray,
) -> np.ndarray:
    if sort_tickers_by == "coverage":
        ticker_mse = _nanmean_axis(sq_err, axis=0)
        return np.lexsort((-np.nan_to_num(ticker_mse, nan=-np.inf), -ticker_coverage))
    if sort_tickers_by == "mean_sq_err":
        ticker_mse = _nanmean_axis(sq_err, axis=0)
        return np.argsort(np.nan_to_num(ticker_mse, nan=-np.inf))
    if sort_tickers_by == "residual_similarity":
        return _residual_similarity_order(residual, ticker_coverage)
    return np.arange(sq_err.shape[1])


def _residual_similarity_order(residual: np.ndarray, ticker_coverage: np.ndarray) -> np.ndarray:
    finite = np.isfinite(residual)
    enough = finite.sum(axis=0) >= 2
    if not enough.any():
        return np.argsort(-ticker_coverage)

    work = residual[:, enough].astype(float, copy=True)
    means = _nanmean_axis(work, axis=0)
    work = work - means
    work[~np.isfinite(work)] = 0.0
    norms = np.sqrt((work**2).sum(axis=0))
    usable = norms > 0
    full_order_source = np.flatnonzero(enough)
    if usable.sum() < 2:
        ordered = full_order_source[np.argsort(-ticker_coverage[full_order_source])]
    else:
        normalized = work[:, usable] / norms[usable]
        try:
            _, _, vh = np.linalg.svd(normalized, full_matrices=False)
            coords = vh[: min(3, vh.shape[0])].T
            keys = [coords[:, i] for i in range(coords.shape[1] - 1, -1, -1)]
            ordered_usable = full_order_source[usable][np.lexsort(keys)]
        except np.linalg.LinAlgError:
            ordered_usable = full_order_source[usable][np.argsort(-ticker_coverage[full_order_source[usable]])]
        unusable = full_order_source[~usable]
        ordered = np.concatenate([ordered_usable, unusable])

    missing = np.setdiff1d(np.arange(residual.shape[1]), ordered, assume_unique=False)
    if missing.size:
        missing = missing[np.argsort(-ticker_coverage[missing])]
        ordered = np.concatenate([ordered, missing])
    return ordered


def _subsample_positions(length: int, max_labels: int) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=int)
    count = min(length, max_labels)
    return np.unique(np.linspace(0, length - 1, count, dtype=int))


def plot_heatmap(
    sq_err: np.ndarray,
    residual: np.ndarray,
    dates: list[str],
    tickers: list[str],
    output_path: Path,
    clip_percentile: float = 99.5,
    sort_tickers_by: str = "coverage",
    ticker_coverage: np.ndarray | None = None,
    figsize: tuple[float, float] = (16.0, 8.0),
    bad_color: str = "lightgray",
    title: str = "Squared Residual Heatmap",
) -> np.ndarray:
    if ticker_coverage is None:
        ticker_coverage = compute_ticker_coverage(residual)
    ticker_order = _ticker_order(sq_err, residual, sort_tickers_by, ticker_coverage)

    sorted_sq_err = sq_err[:, ticker_order]
    sorted_tickers = [tickers[i] for i in ticker_order]
    heatmap_data = np.ma.masked_invalid(sorted_sq_err.T)
    vmax = _finite_percentile(sorted_sq_err, clip_percentile)
    if vmax is not None and vmax <= 0:
        vmax = None

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(bad_color)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        heatmap_data,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=0,
        vmax=vmax,
        cmap=cmap,
    )
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("ticker")

    x_positions = _subsample_positions(len(dates), 12)
    y_positions = _subsample_positions(len(sorted_tickers), 30)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([dates[i] for i in x_positions], rotation=45, ha="right")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([sorted_tickers[i] for i in y_positions])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("squared residual")
    ax.text(
        0.0,
        -0.16,
        "All heatmap rows are retained; only axis labels are subsampled.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return ticker_order


def plot_daily_mse(sq_err: np.ndarray, dates: list[str], output_path: Path) -> np.ndarray:
    daily_mse = _nanmean_axis(sq_err, axis=1)
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(np.arange(len(dates)), daily_mse, linewidth=1.5)
    ax.set_title("Daily Cross-sectional MSE")
    ax.set_xlabel("date")
    ax.set_ylabel("daily cross-sectional MSE")
    x_positions = _subsample_positions(len(dates), 12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([dates[i] for i in x_positions], rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return daily_mse


def save_daily_mse(
    dates: list[str],
    daily_mse: np.ndarray,
    valid_count: np.ndarray,
    output_path: Path,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "date": dates,
            "daily_mse": daily_mse,
            "valid_count": valid_count,
        }
    )
    df.to_csv(output_path, index=False)
    return df


def plot_valid_count(dates: list[str], valid_count: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(np.arange(len(dates)), valid_count, linewidth=1.5)
    ax.set_title("Daily Valid Residual Count")
    ax.set_xlabel("date")
    ax.set_ylabel("finite residual count")
    x_positions = _subsample_positions(len(dates), 12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([dates[i] for i in x_positions], rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_ticker_coverage(
    tickers: list[str],
    ticker_coverage: np.ndarray,
    ticker_mse: np.ndarray,
    coverage_threshold: float,
    output_path: Path,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "ticker": tickers,
            "finite_residual_fraction": ticker_coverage,
            "mean_squared_residual": ticker_mse,
            "below_coverage_threshold": ticker_coverage < coverage_threshold,
        }
    )
    df.to_csv(output_path, index=False)
    return df


def plot_coverage_histogram(
    ticker_coverage: np.ndarray,
    coverage_threshold: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(ticker_coverage, bins=50, range=(0.0, 1.0), color="#4477aa", edgecolor="white")
    ax.axvline(coverage_threshold, color="#cc3311", linestyle="--", linewidth=1.5)
    ax.set_title("Ticker Finite Residual Coverage")
    ax.set_xlabel("finite residual fraction")
    ax.set_ylabel("ticker count")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_ticker_mse(
    sq_err: np.ndarray,
    tickers: list[str],
    output_path: Path,
    top_tickers: int = 50,
) -> np.ndarray:
    ticker_mse = _nanmean_axis(sq_err, axis=0)
    finite_mse = np.nan_to_num(ticker_mse, nan=-np.inf)
    top_count = min(max(top_tickers, 0), len(tickers))
    order = np.argsort(finite_mse)[::-1][:top_count]

    fig_height = max(5.0, min(14.0, 0.25 * max(top_count, 1)))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.bar(np.arange(top_count), ticker_mse[order])
    ax.set_title("Top Tickers by Mean Squared Residual")
    ax.set_xlabel("ticker")
    ax.set_ylabel("mean squared residual")
    ax.set_xticks(np.arange(top_count))
    ax.set_xticklabels([tickers[i] for i in order], rotation=90)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return ticker_mse


def save_top_events(
    dates: list[str],
    tickers: list[str],
    residual: np.ndarray,
    sq_err: np.ndarray,
    r: np.ndarray,
    r_hat: np.ndarray,
    output_path: Path,
    top_events: int = 100,
) -> pd.DataFrame:
    finite_mask = np.isfinite(sq_err)
    finite_flat = np.flatnonzero(finite_mask.ravel())
    if finite_flat.size == 0 or top_events <= 0:
        df = pd.DataFrame(
            columns=["date", "ticker", "residual", "squared_residual", "actual_return", "predicted_return"]
        )
        df.to_csv(output_path, index=False)
        return df

    count = min(top_events, finite_flat.size)
    finite_values = sq_err.ravel()[finite_flat]
    local_idx = np.argpartition(finite_values, -count)[-count:]
    event_flat = finite_flat[local_idx]
    event_flat = event_flat[np.argsort(sq_err.ravel()[event_flat])[::-1]]
    t_idx, n_idx = np.unravel_index(event_flat, sq_err.shape)

    df = pd.DataFrame(
        {
            "date": [dates[i] for i in t_idx],
            "ticker": [tickers[i] for i in n_idx],
            "residual": residual[t_idx, n_idx],
            "squared_residual": sq_err[t_idx, n_idx],
            "actual_return": r[t_idx, n_idx],
            "predicted_return": r_hat[t_idx, n_idx],
        }
    )
    df.to_csv(output_path, index=False)
    return df


def save_daily_mse_spikes(
    dates: list[str],
    tickers: list[str],
    residual: np.ndarray,
    sq_err: np.ndarray,
    r: np.ndarray,
    r_hat: np.ndarray,
    daily_mse: np.ndarray,
    valid_count: np.ndarray,
    output_path: Path,
    spike_days: int = 20,
    spike_top_tickers: int = 10,
) -> pd.DataFrame:
    finite_days = np.flatnonzero(np.isfinite(daily_mse))
    columns = [
        "date",
        "daily_mse",
        "valid_count",
        "rank_within_date",
        "ticker",
        "residual",
        "squared_residual",
        "actual_return",
        "predicted_return",
        "contribution_share",
        "max_residual_ticker",
        "max_squared_residual",
        "residual_concentration_ratio",
    ]
    if finite_days.size == 0 or spike_days <= 0 or spike_top_tickers <= 0:
        df = pd.DataFrame(columns=columns)
        df.to_csv(output_path, index=False)
        return df

    day_count = min(spike_days, finite_days.size)
    spike_idx = finite_days[np.argsort(daily_mse[finite_days])[::-1][:day_count]]
    rows = []
    for t in spike_idx:
        day_sq = sq_err[t]
        finite_tickers = np.flatnonzero(np.isfinite(day_sq))
        if finite_tickers.size == 0:
            continue
        day_values = day_sq[finite_tickers]
        total_sq = float(day_values.sum())
        top_count = min(spike_top_tickers, finite_tickers.size)
        top_local = np.argsort(day_values)[::-1][:top_count]
        top_indices = finite_tickers[top_local]
        max_idx = int(top_indices[0])
        top_sum = float(day_sq[top_indices].sum())
        concentration = top_sum / total_sq if total_sq > 0 else np.nan
        for rank, n in enumerate(top_indices, start=1):
            contribution = float(day_sq[n] / total_sq) if total_sq > 0 else np.nan
            rows.append(
                {
                    "date": dates[t],
                    "daily_mse": daily_mse[t],
                    "valid_count": valid_count[t],
                    "rank_within_date": rank,
                    "ticker": tickers[n],
                    "residual": residual[t, n],
                    "squared_residual": sq_err[t, n],
                    "actual_return": r[t, n],
                    "predicted_return": r_hat[t, n],
                    "contribution_share": contribution,
                    "max_residual_ticker": tickers[max_idx],
                    "max_squared_residual": sq_err[t, max_idx],
                    "residual_concentration_ratio": concentration,
                }
            )
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_path, index=False)
    return df


def save_summary(
    sq_err: np.ndarray,
    residual: np.ndarray,
    daily_mse: np.ndarray,
    valid_count: np.ndarray,
    ticker_mse: np.ndarray,
    ticker_coverage: np.ndarray,
    dates: list[str],
    tickers: list[str],
    K: int,
    coverage_threshold: float,
    factor_names: list[str] | None,
    output_path: Path,
) -> dict:
    finite_residual = np.isfinite(residual)
    finite_sq_err = sq_err[np.isfinite(sq_err)]

    top_ticker = None
    if np.isfinite(ticker_mse).any():
        top_ticker = tickers[int(np.nanargmax(ticker_mse))]

    top_date = None
    if np.isfinite(daily_mse).any():
        top_date = dates[int(np.nanargmax(daily_mse))]

    T, N = sq_err.shape
    summary = {
        "T": int(T),
        "N": int(N),
        "K": int(K),
        "finite_fraction_of_residual": float(finite_residual.mean()) if residual.size else 0.0,
        "mean_squared_residual": _finite_stat(sq_err, np.mean),
        "median_squared_residual": _finite_stat(sq_err, np.median),
        "p95_squared_residual": _finite_percentile(sq_err, 95.0),
        "p99_squared_residual": _finite_percentile(sq_err, 99.0),
        "p99_5_squared_residual": _finite_percentile(sq_err, 99.5),
        "max_squared_residual": float(finite_sq_err.max()) if finite_sq_err.size else None,
        "top_ticker_by_mean_squared_residual": top_ticker,
        "top_date_by_daily_mse": top_date,
        "min_daily_valid_count": int(valid_count.min()) if valid_count.size else None,
        "median_daily_valid_count": float(np.median(valid_count)) if valid_count.size else None,
        "max_daily_valid_count": int(valid_count.max()) if valid_count.size else None,
        "coverage_threshold": float(coverage_threshold),
        "low_coverage_ticker_count": int((ticker_coverage < coverage_threshold).sum()),
        "median_ticker_coverage": _finite_stat(ticker_coverage, np.median),
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if not 0 < args.clip_percentile <= 100:
        raise ValueError(f"--clip-percentile must be in (0, 100], got {args.clip_percentile}")
    if not 0 < args.extreme_clip_percentile <= 100:
        raise ValueError(
            f"--extreme-clip-percentile must be in (0, 100], got {args.extreme_clip_percentile}"
        )
    if not 0 <= args.coverage_threshold <= 1:
        raise ValueError(f"--coverage-threshold must be in [0, 1], got {args.coverage_threshold}")
    if args.figsize[0] <= 0 or args.figsize[1] <= 0:
        raise ValueError(f"--figsize values must be positive, got {args.figsize}")

    input_dir = Path(args.input_dir)
    out_dir = input_dir / "diagnostics" / "residuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = load_pipeline_outputs(input_dir)
    X = outputs["X"]
    r = outputs["r"]
    factor_returns = outputs["factor_returns"]
    dates = outputs["dates"]
    tickers = outputs["tickers"]
    factor_names = outputs["factor_names"]

    r_hat, residual, sq_err = compute_residuals(X, r, factor_returns)
    ticker_coverage = compute_ticker_coverage(residual)
    valid_count = compute_valid_count(residual)
    heatmap_mask = ticker_coverage >= args.coverage_threshold if args.filter_low_coverage else np.ones_like(ticker_coverage, dtype=bool)
    if not heatmap_mask.any():
        raise ValueError(
            f"No tickers meet --coverage-threshold {args.coverage_threshold}; lower the threshold or omit --filter-low-coverage"
        )
    plot_heatmap(
        sq_err=sq_err[:, heatmap_mask],
        residual=residual[:, heatmap_mask],
        dates=dates,
        tickers=[ticker for ticker, keep in zip(tickers, heatmap_mask) if keep],
        output_path=out_dir / "squared_residual_heatmap.png",
        clip_percentile=args.clip_percentile,
        sort_tickers_by=args.sort_tickers_by,
        ticker_coverage=ticker_coverage[heatmap_mask],
        figsize=tuple(args.figsize),
        bad_color=args.bad_color,
    )
    daily_mse = plot_daily_mse(sq_err, dates, out_dir / "daily_mse.png")
    save_daily_mse(dates, daily_mse, valid_count, out_dir / "daily_mse.csv")
    plot_valid_count(dates, valid_count, out_dir / "daily_valid_count.png")
    ticker_mse = plot_ticker_mse(sq_err, tickers, out_dir / "ticker_mse_top.png", args.top_tickers)
    save_ticker_coverage(
        tickers=tickers,
        ticker_coverage=ticker_coverage,
        ticker_mse=ticker_mse,
        coverage_threshold=args.coverage_threshold,
        output_path=out_dir / "ticker_coverage.csv",
    )
    plot_coverage_histogram(
        ticker_coverage=ticker_coverage,
        coverage_threshold=args.coverage_threshold,
        output_path=out_dir / "ticker_coverage_histogram.png",
    )
    coverage_mask = ticker_coverage >= args.coverage_threshold
    if coverage_mask.any():
        coverage_tickers = [ticker for ticker, keep in zip(tickers, coverage_mask) if keep]
        plot_heatmap(
            sq_err=sq_err[:, coverage_mask],
            residual=residual[:, coverage_mask],
            dates=dates,
            tickers=coverage_tickers,
            output_path=out_dir / "squared_residual_heatmap_coverage_filtered.png",
            clip_percentile=args.clip_percentile,
            sort_tickers_by=args.sort_tickers_by,
            ticker_coverage=ticker_coverage[coverage_mask],
            figsize=tuple(args.figsize),
            bad_color=args.bad_color,
            title=f"Squared Residual Heatmap, Coverage >= {args.coverage_threshold:g}",
        )
        plot_heatmap(
            sq_err=sq_err[:, coverage_mask],
            residual=residual[:, coverage_mask],
            dates=dates,
            tickers=coverage_tickers,
            output_path=out_dir / "squared_residual_heatmap_extreme_clipped.png",
            clip_percentile=args.extreme_clip_percentile,
            sort_tickers_by=args.sort_tickers_by,
            ticker_coverage=ticker_coverage[coverage_mask],
            figsize=tuple(args.figsize),
            bad_color=args.bad_color,
            title=f"Squared Residual Heatmap, Coverage Filtered, p{args.extreme_clip_percentile:g} Clip",
        )
    save_top_events(
        dates=dates,
        tickers=tickers,
        residual=residual,
        sq_err=sq_err,
        r=r,
        r_hat=r_hat,
        output_path=out_dir / "top_residual_events.csv",
        top_events=args.top_events,
    )
    save_daily_mse_spikes(
        dates=dates,
        tickers=tickers,
        residual=residual,
        sq_err=sq_err,
        r=r,
        r_hat=r_hat,
        daily_mse=daily_mse,
        valid_count=valid_count,
        output_path=out_dir / "daily_mse_spikes.csv",
        spike_days=args.spike_days,
        spike_top_tickers=args.spike_top_tickers,
    )
    save_summary(
        sq_err=sq_err,
        residual=residual,
        daily_mse=daily_mse,
        valid_count=valid_count,
        ticker_mse=ticker_mse,
        ticker_coverage=ticker_coverage,
        dates=dates,
        tickers=tickers,
        K=X.shape[2],
        coverage_threshold=args.coverage_threshold,
        factor_names=factor_names,
        output_path=out_dir / "residual_summary.json",
    )

    print(f"saved residual diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
