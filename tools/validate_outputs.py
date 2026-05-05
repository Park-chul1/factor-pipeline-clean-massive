from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def ratio(x: np.ndarray) -> float:
    return float(np.isfinite(x).mean()) if x.size else 0.0


def main() -> int:
    p = argparse.ArgumentParser(description="Validate generated factor pipeline outputs")
    p.add_argument("out_dir", help="Output directory, e.g. data/test_run3")
    p.add_argument("--min-x-finite", type=float, default=0.95, help="Expected after neutral fill; lower if --no-fill-missing-exposures was used")
    p.add_argument("--min-r-finite", type=float, default=0.50)
    p.add_argument("--min-finite-factor-returns", type=float, default=0.10)
    args = p.parse_args()

    out = Path(args.out_dir)
    required = ["X.npy", "r.npy", "factor_returns.npy", "factor_names.csv", "pipeline_summary.json", "factor_diagnostics_preprocessed.csv"]
    missing = [name for name in required if not (out / name).exists()]
    if missing:
        print("FAIL missing files:", missing)
        return 1

    X = np.load(out / "X.npy")
    r = np.load(out / "r.npy")
    f = np.load(out / "factor_returns.npy")
    names = pd.read_csv(out / "factor_names.csv")
    diag = pd.read_csv(out / "factor_diagnostics_preprocessed.csv")

    print("X shape:", X.shape, "finite:", ratio(X))
    print("r shape:", r.shape, "finite:", ratio(r))
    print("factor_returns shape:", f.shape, "finite:", ratio(f))
    print("kept factors:", len(names), "dropped:", int((~diag["kept"].astype(bool)).sum()) if not diag.empty else 0)

    ok = True
    if X.ndim != 3 or r.ndim != 2 or f.ndim != 2:
        print("FAIL invalid array rank")
        ok = False
    if X.shape[:2] != r.shape:
        print("FAIL X and r shape mismatch")
        ok = False
    if X.shape[2] != len(names) or f.shape[1] != len(names):
        print("FAIL factor_names length mismatch")
        ok = False
    if ratio(X) < args.min_x_finite:
        print(f"FAIL X finite ratio below threshold {args.min_x_finite}")
        ok = False
    if ratio(r) < args.min_r_finite:
        print(f"FAIL r finite ratio below threshold {args.min_r_finite}")
        ok = False
    if ratio(f) < args.min_finite_factor_returns:
        print(f"FAIL factor return finite ratio below threshold {args.min_finite_factor_returns}")
        ok = False

    # Show weakest kept factors for quick debugging.
    if not diag.empty:
        cols = ["factor", "kept", "raw_finite_ratio", "processed_finite_ratio", "finite_dates"]
        print("\nLowest processed coverage:")
        print(diag[cols].sort_values("processed_finite_ratio").head(10).to_string(index=False))

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
