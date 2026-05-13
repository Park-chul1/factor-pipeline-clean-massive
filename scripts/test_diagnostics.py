#!/usr/bin/env python3
"""Quick test: run ridge diagnostics on existing pipeline output and display results."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factor_pipeline.diagnostics_ridge import run_estimation_diagnostics


def main():
    # Try to load existing pipeline outputs
    out_dir = Path("data/processed_clean")
    
    if not (out_dir / "X.npy").exists():
        print(f"Error: Pipeline output not found in {out_dir}")
        print("Run: python scripts/run_clean_pipeline.py first")
        sys.exit(1)
    
    print(f"Loading pipeline outputs from {out_dir}...")
    X = np.load(out_dir / "X.npy")
    r = np.load(out_dir / "r.npy")
    tradable_mask = np.load(out_dir / "tradable_mask.npy")
    
    factor_names = None
    if (out_dir / "factor_names.csv").exists():
        factor_names = pd.read_csv(out_dir / "factor_names.csv")["factor"].tolist()
    
    print(f"X shape: {X.shape} (dates, stocks, factors)")
    print(f"r shape: {r.shape}")
    print(f"tradable_mask shape: {tradable_mask.shape}")
    print(f"factors: {len(factor_names) if factor_names else 'Unknown'}")
    print()
    
    # Run diagnostics with a reasonable lambda grid
    lambdas = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
    diagnostics_dir = out_dir / "diagnostics"
    
    print(f"Running ridge diagnostics with lambdas: {lambdas}")
    result = run_estimation_diagnostics(
        X=X,
        r=r,
        valid_mask=tradable_mask,
        lambdas=lambdas,
        factor_names=factor_names,
        output_dir=diagnostics_dir,
        min_names=30,
    )
    
    print()
    print("=" * 80)
    print("OUTPUTS SAVED TO:")
    print("=" * 80)
    for key in ["date_diagnostics_path", "factor_diagnostics_path", "lambda_summary_path", "suspicious_days_path"]:
        if key in result["summary"]:
            print(f"  {result['summary'][key]}")
    print("=" * 80)
    print()
    
    # Display lambda summary
    if result["lambda_summaries"]:
        print("LAMBDA SUMMARY (key metrics per lambda):")
        df = pd.DataFrame(result["lambda_summaries"])
        display_cols = [
            "lambda", "valid_days", "mean_centered_R2", "median_centered_R2",
            "median_cond_X", "q99_cond_X", "mean_norm_f", "q99_norm_f"
        ]
        display_cols = [c for c in display_cols if c in df.columns]
        print(df[display_cols].to_string(index=False))
        print()
    
    # Display date diagnostics summary
    if result["date_diagnostics"]:
        df_date = pd.DataFrame(result["date_diagnostics"])
        print("DATE DIAGNOSTICS (counts by lambda):")
        print(df_date.groupby("lambda_idx")[["day_valid", "n_names"]].agg({
            "day_valid": "sum",
            "n_names": "mean"
        }).round(2).to_string())
        print()
    
    # Display suspicious days
    if not result["suspicious_days"].empty:
        print("TOP SUSPICIOUS DAYS (per lambda, per reason):")
        df_susp = result["suspicious_days"]
        for lam in sorted(df_susp["lambda"].unique()):
            print(f"\n  Lambda = {lam}:")
            subset = df_susp[df_susp["lambda"] == lam]
            for reason in sorted(subset["reason"].unique()):
                reason_subset = subset[subset["reason"] == reason].head(3)
                print(f"    {reason}:")
                for _, row in reason_subset.iterrows():
                    print(f"      date_idx={int(row['date_idx'])}, value={row['value']:.4f}")


if __name__ == "__main__":
    main()
