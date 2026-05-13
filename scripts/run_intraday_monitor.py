from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description="Intraday monitor: order/risk/connectivity checks only, no signal recomputation")
    p.add_argument("--report-dir", default="reports/daily")
    args = p.parse_args()
    print("intraday monitor policy: no alpha recomputation and no delayed-data rebalancing")
    print(f"watching reports under: {args.report_dir}")


if __name__ == "__main__":
    main()
