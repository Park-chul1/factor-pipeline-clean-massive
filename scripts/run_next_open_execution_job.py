from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live.daily_jobs import run_next_open_execution_job


def main() -> None:
    p = argparse.ArgumentParser(description="Run next-open execution from an existing after-close order plan")
    p.add_argument("--date", required=True, help="Execution date YYYY-MM-DD")
    p.add_argument("--config", default=None)
    args = p.parse_args()
    report_dir = run_next_open_execution_job(args.date, args.config)
    print(f"execution artifacts updated: {report_dir}")


if __name__ == "__main__":
    main()
