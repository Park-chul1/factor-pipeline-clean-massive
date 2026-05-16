from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.close_to_next_open_daily import config_from_dict, run_close_to_next_open_backtest, write_daily_report
from factor_pipeline.simple_config import load_config_dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run daily close-signal to next-open execution backtest")
    p.add_argument("--config", default=None, help="JSON or simple key: value YAML config")
    p.add_argument("--input-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--daily-bars-path", default=None)
    p.add_argument("--execution-bars-path", default=None)
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_config_dict(args.config)
    if args.input_dir:
        raw["input_dir"] = args.input_dir
    if args.out_dir:
        raw["out_dir"] = args.out_dir
    if args.daily_bars_path:
        raw["daily_bars_path"] = args.daily_bars_path
    if args.execution_bars_path:
        raw["execution_bars_path"] = args.execution_bars_path
    if args.start_date:
        raw["start_date"] = args.start_date
    if args.end_date:
        raw["end_date"] = args.end_date
    cfg = config_from_dict(raw)
    result = run_close_to_next_open_backtest(cfg)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    result.daily.to_csv(cfg.out_dir / "daily_returns.csv", index=False)
    result.orders.to_csv(cfg.out_dir / "orders.csv", index=False)
    result.target_positions.to_csv(cfg.out_dir / "target_positions.csv", index=False)
    result.alpha_rankings.to_csv(cfg.out_dir / "alpha_rankings.csv", index=False)
    result.quantile_summary.to_csv(cfg.out_dir / "quantile_summary.csv", index=False)
    result.execution_quality.to_csv(cfg.out_dir / "execution_quality.csv", index=False)
    report_dir = write_daily_report(result, cfg.reports_dir)
    print(f"done: {cfg.out_dir}")
    print(f"daily report: {report_dir}")
    if result.warnings:
        print("warnings:")
        for warning in result.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
