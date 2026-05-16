#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a NumPy .npy array file to CSV."
    )
    parser.add_argument("input", help="Path to the .npy file")
    parser.add_argument(
        "output",
        nargs="?",
        help="Path to the output .csv file. Defaults to same name as input with .csv extension.",
    )
    parser.add_argument(
        "--allow-pickle",
        action="store_true",
        help="Allow loading object arrays pickled in the .npy file.",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="CSV delimiter to use. Default is ','.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_suffix(".csv")

    arr = np.load(input_path, allow_pickle=args.allow_pickle)

    if arr.ndim == 0:
        df = pd.DataFrame([arr.item()])
    elif arr.ndim == 1:
        df = pd.DataFrame(arr)
    else:
        df = pd.DataFrame(arr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, sep=args.delimiter)
    print(f"Saved CSV to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
