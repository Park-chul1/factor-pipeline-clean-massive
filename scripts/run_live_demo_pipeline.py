from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import time
import traceback

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live.alpha import apply_signal_filters, compute_alpha, load_or_compute_f_pred, standardize_alpha
from live.broker import make_broker
from live.config import load_config
from live.data import load_recent_history, load_universe, make_provider, update_local_bar_cache
from live.execution import current_weights_from_positions, generate_orders, latest_prices_from_bars
from live.features import build_latest_exposures
from live.logging_utils import append_csv
from live.portfolio import compute_turnover, construct_target_portfolio
from live.risk import apply_risk_checks, assert_demo_safety


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run delayed-data live demo signal and paper execution pipeline")
    p.add_argument("--config", default=None, help="Optional JSON config path")
    p.add_argument("--once", action="store_true", help="Run one iteration and exit")
    p.add_argument("--max-loops", type=int, default=None, help="Run at most N loops, then exit")
    p.add_argument("--poll-interval", type=int, default=None, help="Override polling interval seconds")
    p.add_argument("--dry-run", action="store_true", help="Log orders without filling simulated broker")
    p.add_argument("--data-provider", choices=["massive", "cache", "ibkr_delayed"], default=None, help="Market data provider; default config uses massive")
    p.add_argument("--ibkr-paper", action="store_true", help="Use IBKR paper broker for order routing only")
    p.add_argument("--ibkr-host", default=None, help="TWS/Gateway host override")
    p.add_argument("--ibkr-port", type=int, default=None, help="TWS/Gateway paper port override, usually 7497")
    p.add_argument("--ibkr-account", default=None, help="Optional paper account filter")
    p.add_argument("--data-client-id", type=int, default=None, help="IBKR client id for market data")
    p.add_argument("--broker-client-id", type=int, default=None, help="IBKR client id for paper order routing")
    p.add_argument("--universe-limit", type=int, default=None, help="Override universe size limit")
    p.add_argument("--all-universe", action="store_true", help="Use the full active universe instead of truncating")
    p.add_argument("--fetch-batch-size", type=int, default=None, help="Max symbols to refresh per loop")
    p.add_argument("--fetch-workers", type=int, default=None, help="Parallel fetch workers for provider calls")
    p.add_argument("--cache-path", default=None, help="Live 15-minute bar parquet cache path")
    p.add_argument("--prefetch-history", action="store_true", help="Warm local 15-minute bar cache before trading loop")
    p.add_argument("--prefetch-only", action="store_true", help="Prefetch cache and exit without computing or trading")
    p.add_argument("--prefetch-limit", type=int, default=None, help="Optional cap for prefetch symbols")
    return p.parse_args()


def override_config(config, args: argparse.Namespace):
    updates = {}
    if args.dry_run:
        updates["DRY_RUN"] = True
    if args.data_provider is not None:
        updates["DATA_PROVIDER"] = args.data_provider
    if args.ibkr_paper:
        updates.update({"BROKER": "ibkr_paper", "PAPER_TRADING": True, "ENABLE_REAL_TRADING": False})
    if args.ibkr_host is not None:
        updates["IBKR_HOST"] = args.ibkr_host
    if args.ibkr_port is not None:
        updates["IBKR_PORT"] = args.ibkr_port
    if args.ibkr_account is not None:
        updates["IBKR_ACCOUNT"] = args.ibkr_account
    if args.data_client_id is not None:
        updates["IBKR_DATA_CLIENT_ID"] = args.data_client_id
    if args.broker_client_id is not None:
        updates["IBKR_BROKER_CLIENT_ID"] = args.broker_client_id
    if args.universe_limit is not None:
        updates["UNIVERSE_SIZE_LIMIT"] = args.universe_limit
    if args.all_universe:
        updates["UNIVERSE_SIZE_LIMIT"] = None
    if args.fetch_batch_size is not None:
        updates["LIVE_FETCH_BATCH_SIZE"] = args.fetch_batch_size
    if args.fetch_workers is not None:
        updates["LIVE_FETCH_WORKERS"] = args.fetch_workers
    if args.cache_path is not None:
        updates["CACHE_PATH"] = Path(args.cache_path)
    if args.poll_interval is not None:
        updates["POLL_INTERVAL_SECONDS"] = args.poll_interval
    if not updates:
        return config
    cfg = config.__class__(**{**config.__dict__, **updates})
    cfg.validate()
    return cfg


def market_is_open() -> bool:
    return True


def run_once(config, provider, broker, universe: list[str], holding_bars: dict[str, int]) -> None:
    loop_started = time.perf_counter()
    now = pd.Timestamp.utcnow().tz_convert(None)
    delayed_asof = now - pd.Timedelta(minutes=config.DELAY_MINUTES)

    fetch = provider.fetch_latest_delayed_bars(universe, delayed_asof)
    if not fetch.bars.empty:
        update_local_bar_cache(fetch.bars, config.CACHE_PATH)
    recent_history = load_recent_history(universe, config.LOOKBACK_BARS, config.CACHE_PATH, end_time=fetch.asof_time)

    compute_started = time.perf_counter()
    X_now, tickers, factor_names, coverage = build_latest_exposures(recent_history, config)
    f_pred = load_or_compute_f_pred(config, factor_names)
    alpha = compute_alpha(X_now, f_pred)
    z_alpha = standardize_alpha(alpha)
    latest_bars = recent_history[recent_history["timestamp"] == recent_history["timestamp"].max()] if not recent_history.empty else fetch.bars
    candidates = apply_signal_filters(alpha, z_alpha, latest_bars, config)

    account = broker.get_account_summary()
    equity = float(account.get("equity", config.PAPER_STARTING_EQUITY))
    prices = latest_prices_from_bars(latest_bars)
    current_shares = broker.get_positions()
    price_symbols = sorted(set(candidates["ticker"].astype(str).tolist() if not candidates.empty else []) | set(current_shares.index.astype(str).tolist()))
    if config.DATA_PROVIDER.lower() in {"ibkr", "ibkr_delayed"} and hasattr(broker, "get_prices") and price_symbols:
        live_prices = broker.get_prices(price_symbols)
        prices = live_prices if not live_prices.empty else pd.Series(dtype=float)
    current_weights = current_weights_from_positions(current_shares, prices, equity)
    target_weights = construct_target_portfolio(candidates, current_weights, config, holding_bars)
    turnover = compute_turnover(current_weights, target_weights)
    orders = generate_orders(current_shares, target_weights, prices, equity, config)
    safe_orders, risk_rows = apply_risk_checks(orders, config, equity, prices)
    compute_latency = time.perf_counter() - compute_started

    for row in risk_rows:
        row["timestamp"] = now.isoformat()
        append_csv(config.LOG_DIR / "risk_checks.csv", row)
    for order in safe_orders:
        append_csv(config.LOG_DIR / "orders.csv", {
            "timestamp": now.isoformat(),
            "ticker": order.ticker,
            "side": order.side,
            "quantity": order.quantity,
            "limit_price": order.limit_price,
            "dry_run": order.dry_run,
            "paper_trading": config.PAPER_TRADING,
        })
        if config.PAPER_TRADING:
            try:
                order_id = broker.place_order(order)
                append_csv(config.LOG_DIR / "order_results.csv", {
                    "timestamp": now.isoformat(),
                    "ticker": order.ticker,
                    "side": order.side,
                    "quantity": order.quantity,
                    "limit_price": order.limit_price,
                    "status": "accepted",
                    "order_id": order_id,
                    "error": "",
                })
            except Exception as exc:
                append_csv(config.LOG_DIR / "order_rejections.csv", {
                    "timestamp": now.isoformat(),
                    "ticker": order.ticker,
                    "side": order.side,
                    "quantity": order.quantity,
                    "limit_price": order.limit_price,
                    "status": "rejected",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                print(
                    f"order rejected ticker={order.ticker} side={order.side} "
                    f"qty={order.quantity}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
        else:
            raise RuntimeError("Refusing to place orders unless PAPER_TRADING is true in demo pipeline")

    gross = float(target_weights.abs().sum()) if not target_weights.empty else 0.0
    net = float(target_weights.sum()) if not target_weights.empty else 0.0
    event = {
        "timestamp": now.isoformat(),
        "delayed_asof": delayed_asof.isoformat(),
        "fetch_asof": fetch.asof_time.isoformat(),
        "api_latency": fetch.latency_seconds,
        "compute_latency": compute_latency,
        "number_of_symbols": len(universe),
        "number_valid_exposures": len(tickers),
        "alpha_mean": float(alpha.mean()) if not alpha.empty else None,
        "alpha_std": float(alpha.std(ddof=0)) if not alpha.empty else None,
        "number_candidates": int(candidates["passes_signal"].sum()) if not candidates.empty else 0,
        "estimated_turnover": turnover,
        "number_orders": len(safe_orders),
        "expected_gross_exposure": gross,
        "expected_net_exposure": net,
    }
    append_csv(config.LOG_DIR / "live_signal_events.csv", event)
    append_csv(config.LOG_DIR / "pipeline_latency.csv", {
        "timestamp": now.isoformat(),
        "api_latency": fetch.latency_seconds,
        "compute_latency": compute_latency,
        "total_latency": time.perf_counter() - loop_started,
    })

    for ticker in list(holding_bars):
        holding_bars[ticker] += 1
        if ticker not in target_weights.index:
            del holding_bars[ticker]
    for ticker in target_weights.index:
        holding_bars[ticker] = holding_bars.get(ticker, 0) + 1

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] asof={fetch.asof_time} "
        f"symbols={len(universe)} valid={len(tickers)} candidates={event['number_candidates']} "
        f"orders={len(safe_orders)} turnover={turnover:.4f} latency={time.perf_counter() - loop_started:.2f}s",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if not args.prefetch_only and os.getenv("ALLOW_LEGACY_INTRADAY_REBALANCE") != "1":
        raise RuntimeError(
            "run_live_demo_pipeline.py is the legacy intraday delayed-data rebalance loop and is disabled by default. "
            "Use scripts/run_after_close_job.py and scripts/run_next_open_execution_job.py for the canonical daily "
            "close-to-next-open workflow. Set ALLOW_LEGACY_INTRADAY_REBALANCE=1 only for explicit legacy diagnostics."
        )
    config = override_config(load_config(args.config), args)
    assert_demo_safety(config)
    provider = make_provider(config)
    universe = load_universe(config)

    if args.prefetch_history:
        if not hasattr(provider, "prefetch_history"):
            raise RuntimeError("Selected data provider does not support --prefetch-history")
        now = pd.Timestamp.utcnow().tz_convert(None) - pd.Timedelta(minutes=config.DELAY_MINUTES)
        fetch = provider.prefetch_history(universe, asof_time=now, limit=args.prefetch_limit)
        if not fetch.bars.empty:
            update_local_bar_cache(fetch.bars, config.CACHE_PATH)
        print(
            f"prefetch done asof={fetch.asof_time} requested={fetch.symbols_requested} "
            f"updated={fetch.symbols_updated} rows={len(fetch.bars)} latency={fetch.latency_seconds:.2f}s",
            flush=True,
        )
        if args.prefetch_only:
            if hasattr(provider, "disconnect"):
                provider.disconnect()
            return

    broker = make_broker(config)
    holding_bars: dict[str, int] = {}

    loops = 0
    try:
        try:
            while market_is_open():
                run_once(config, provider, broker, universe, holding_bars)
                loops += 1
                if args.once:
                    break
                if args.max_loops is not None and loops >= args.max_loops:
                    break
                time.sleep(config.POLL_INTERVAL_SECONDS)
        except Exception as exc:
            config.LOG_DIR.mkdir(parents=True, exist_ok=True)
            err_path = config.LOG_DIR / "live_runtime_errors.log"
            with err_path.open("a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.now().isoformat()}] {type(exc).__name__}: {exc}\n")
                f.write(traceback.format_exc())
            print(f"runtime error logged to {err_path}: {type(exc).__name__}: {exc}", flush=True)
            raise
    finally:
        if hasattr(provider, "disconnect"):
            provider.disconnect()


if __name__ == "__main__":
    main()
