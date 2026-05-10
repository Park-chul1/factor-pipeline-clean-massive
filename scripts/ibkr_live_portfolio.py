from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from ib_insync import IB, MarketOrder, Stock
except ImportError as exc:
    raise ImportError(
        "ib_insync is required to run this script. Install it with `pip install ib_insync`."
    ) from exc


def parse_args():
    p = argparse.ArgumentParser(
        description="Live IBKR portfolio rebalancer using saved pipeline target weights"
    )
    p.add_argument("--host", default="127.0.0.1", help="TWS/Gateway host")
    p.add_argument("--port", type=int, default=7497, help="TWS demo socket port")
    p.add_argument("--client-id", type=int, default=1, help="IB API client ID")
    p.add_argument("--account", default=None, help="Optional IB account to filter account values")
    p.add_argument("--load-pipeline", default=None, help="Directory containing saved pipeline outputs with tickers.csv and weights.npy or X.npy/factor_returns.npy for fallback calculation")
    p.add_argument("--date-index", type=int, default=-1, help="Index of the date to load target weights from weights.npy or compute from pipeline outputs")
    p.add_argument("--method", default="ewma", choices=["latest", "rolling", "ewma", "zero", "oracle"], help="Factor return prediction method when computing weights from saved pipeline outputs")
    p.add_argument("--lookback", type=int, default=20, help="Lookback window for factor return prediction")
    p.add_argument("--ewma-halflife", type=float, default=20.0, help="EWMA halflife for factor return prediction")
    p.add_argument("--min-periods", type=int, default=5, help="Minimum positive observations for factor return prediction")
    p.add_argument("--quantile", type=float, default=0.10, help="Quantile for long/short portfolio weights when computing from scores")
    p.add_argument("--gross", type=float, default=2.0, help="Gross exposure for long/short weight construction")
    p.add_argument("--notional", type=float, default=100000.0, help="Notional size for target weights when estimating order sizes")
    p.add_argument("--top-n", type=int, default=50, help="Limit live trading to top N target positions by notional size")
    p.add_argument("--max-symbol-pct", type=float, default=0.02, help="Maximum equity fraction for any single symbol")
    p.add_argument("--max-side-pct", type=float, default=0.60, help="Maximum total long or short exposure as a fraction of equity")
    p.add_argument("--dry-run", action="store_true", help="Do not send live orders; only print order plan")
    p.add_argument("--auto-rebalance", action="store_true", help="Place live orders to rebalance current positions toward target weights")
    p.add_argument("--stream-interval", type=float, default=30.0, help="Seconds between live market data refreshes")
    p.add_argument("--stream-duration-minutes", type=float, default=10.0, help="How many minutes to stream live market data")
    p.add_argument("--rebalance-interval-minutes", type=float, default=0.0, help="If >0, repeat rebalance every N minutes while streaming")
    p.add_argument("--market-data-type", type=int, default=1, help="IB market data type: 1 realtime, 2 frozen, 3 delayed, 4 delayed frozen")
    p.add_argument("--symbols", nargs="*", default=[], help="Additional symbols to stream for market data")
    p.add_argument("--pnl-log", default="data/ibkr_daily_pnl.csv", help="CSV file path for daily PnL and equity snapshots")
    return p.parse_args()


def connect_ibkr(host: str, port: int, client_id: int, timeout: float = 10.0) -> IB:
    ib = IB()
    ib.connect(host, port, clientId=client_id, timeout=timeout)
    return ib


def stock_contract(symbol: str, exchange: str = "SMART", currency: str = "USD") -> Stock:
    return Stock(symbol, exchange, currency)


def set_market_data_type(ib: IB, market_data_type: int = 1) -> None:
    if hasattr(ib, "reqMarketDataType"):
        try:
            ib.reqMarketDataType(market_data_type)
        except Exception as exc:
            print(f"WARN: failed to set market data type {market_data_type}: {exc}")


def account_values(ib: IB, account: str | None = None) -> list:
    values = ib.accountValues()
    if account is not None:
        values = [v for v in values if v.account == account]
    return values


def summary_equity(values: list) -> dict[str, float | str]:
    out = {}
    for v in values:
        if v.tag == "EquityWithLoanValue":
            out["equity"] = float(v.value)
        if v.tag == "CashBalance":
            out["cash"] = float(v.value)
        if v.tag == "AvailableFunds":
            out["available_funds"] = float(v.value)
    return out


def positions_by_symbol(ib: IB, account: str | None = None) -> dict[str, int]:
    out: dict[str, int] = {}
    for pos in ib.positions():
        if account is not None and pos.account != account:
            continue
        symbol = getattr(pos.contract, "symbol", None)
        if symbol is None:
            continue
        out[symbol] = int(pos.position)
    return out


def get_latest_prices(ib: IB, symbols: list[str], timeout: float = 2.0, market_data_type: int = 1) -> dict[str, float | None]:
    set_market_data_type(ib, market_data_type)
    prices: dict[str, float | None] = {}
    tickers = []
    for symbol in symbols:
        contract = stock_contract(symbol)
        ticker = ib.reqMktData(contract, snapshot=True)
        tickers.append((symbol, ticker))

    elapsed = 0.0
    while elapsed < timeout:
        all_updated = True
        for _, ticker in tickers:
            if not math.isfinite(getattr(ticker, 'last', float('nan'))):
                all_updated = False
                break
        if all_updated:
            break
        ib.sleep(0.05)
        elapsed += 0.05

    for symbol, ticker in tickers:
        price = getattr(ticker, "last", float("nan"))
        if not math.isfinite(price):
            price = getattr(ticker, "close", float("nan"))
        prices[symbol] = float(price) if math.isfinite(price) else None
    for _, ticker in tickers:
        ib.cancelMktData(ticker)
    return prices


def load_pipeline_target_weights(
    path: Path,
    date_index: int = -1,
    method: str = "ewma",
    lookback: int = 20,
    ewma_halflife: float = 20.0,
    min_periods: int = 5,
    quantile: float = 0.10,
    gross: float = 2.0,
) -> tuple[list[str], np.ndarray]:
    tickers_path = path / "tickers.csv"
    if not tickers_path.exists():
        raise FileNotFoundError(f"Pipeline output missing tickers.csv in {path}")
    tickers = pd.read_csv(tickers_path, usecols=["ticker"]).squeeze("columns").astype(str).tolist()

    weights_path = path / "weights.npy"
    positions_path = path / "positions.npy"
    scores_path = path / "scores.npy"

    if weights_path.exists() or positions_path.exists() or scores_path.exists():
        if weights_path.exists():
            weights = np.load(weights_path)
        elif positions_path.exists():
            weights = np.load(positions_path)
        else:
            scores = np.load(scores_path)
            if scores.ndim != 2:
                raise ValueError(f"scores.npy must be 2-D [T,N], got {scores.shape}")
            if date_index < 0:
                date_index = scores.shape[0] + date_index
            if not (0 <= date_index < scores.shape[0]):
                raise IndexError(f"date_index {date_index} out of range for scores shape {scores.shape}")
            return tickers, scores[date_index]

        if weights.ndim != 2:
            raise ValueError(f"weight array must be 2-D [T,N], got {weights.shape}")
        if date_index < 0:
            date_index = weights.shape[0] + date_index
        if not (0 <= date_index < weights.shape[0]):
            raise IndexError(f"date_index {date_index} out of range for weight array shape {weights.shape}")
        return tickers, weights[date_index]

    X_path = path / "X.npy"
    f_path = path / "factor_returns.npy"
    if not X_path.exists() or not f_path.exists():
        raise FileNotFoundError(
            f"Pipeline output missing tickers.csv and no weights.npy/positions.npy/scores.npy, nor X.npy/factor_returns.npy in {path}"
        )

    X = np.load(X_path)
    f = np.load(f_path)
    tradable_mask_path = path / "tradable_mask.npy"
    tradable_mask = np.load(tradable_mask_path) if tradable_mask_path.exists() else None

    from factor_pipeline.signals import (
        make_quantile_long_short_weights,
        make_scores,
        predict_factor_returns,
    )

    f_pred = predict_factor_returns(
        f,
        method=method,
        lookback=lookback,
        ewma_halflife=ewma_halflife,
        min_periods=min_periods,
    )

    if date_index < 0:
        date_index = f_pred.shape[0] + date_index
    if not (0 <= date_index < f_pred.shape[0]):
        raise IndexError(f"date_index {date_index} out of range for factor predictions shape {f_pred.shape}")

    scores = make_scores(X, f_pred, tradable_mask=tradable_mask)
    weights = make_quantile_long_short_weights(
        scores[date_index],
        tradable_mask[date_index] if tradable_mask is not None else None,
        quantile=quantile,
        gross=gross,
    )
    return tickers, weights


def compute_target_dollars(weights: np.ndarray, total_equity: float, gross: float = 2.0) -> np.ndarray:
    if gross <= 0:
        raise ValueError("gross must be positive")
    return weights * (total_equity / gross)


def apply_risk_limits(
    target_dollars: dict[str, float],
    equity: float,
    max_symbol_pct: float = 0.02,
    max_side_pct: float = 0.60,
    max_symbols: int = 50,
) -> dict[str, float]:
    capped: dict[str, float] = {}
    max_symbol = equity * max_symbol_pct
    for symbol, dollar in target_dollars.items():
        capped[symbol] = math.copysign(min(abs(dollar), max_symbol), dollar)

    longs = {s: d for s, d in capped.items() if d > 0}
    shorts = {s: d for s, d in capped.items() if d < 0}
    max_side = equity * max_side_pct

    long_total = sum(longs.values())
    short_total = sum(-d for d in shorts.values())
    if long_total > max_side and long_total > 0:
        scale = max_side / long_total
        longs = {s: d * scale for s, d in longs.items()}
    if short_total > max_side and short_total > 0:
        scale = max_side / short_total
        shorts = {s: d * scale for s, d in shorts.items()}

    merged = {**longs, **shorts}
    sorted_items = sorted(merged.items(), key=lambda item: abs(item[1]), reverse=True)
    return dict(sorted_items[:max_symbols])


def target_shares_from_dollars(target_dollars: dict[str, float], prices: dict[str, float | None]) -> dict[str, int]:
    target: dict[str, int] = {}
    for symbol, dollar in target_dollars.items():
        price = prices.get(symbol)
        if price is None or price <= 0.0 or not math.isfinite(price):
            continue
        qty = int(math.floor(abs(dollar) / price))
        if qty <= 0:
            continue
        target[symbol] = qty if dollar > 0 else -qty
    return target


def order_plan(target_shares: dict[str, int], current_shares: dict[str, int]) -> dict[str, int]:
    plan: dict[str, int] = {}
    symbols = set(target_shares) | set(current_shares)
    for symbol in symbols:
        target_qty = target_shares.get(symbol, 0)
        current_qty = current_shares.get(symbol, 0)
        diff = target_qty - current_qty
        if diff != 0:
            plan[symbol] = diff
    return plan


def place_market_order(ib: IB, symbol: str, quantity: int) -> None:
    contract = stock_contract(symbol)
    action = "BUY" if quantity > 0 else "SELL"
    order = MarketOrder(action, abs(quantity))
    trade = ib.placeOrder(contract, order)
    ib.sleep(1)
    print(f"Order: {action} {abs(quantity)} {symbol} -> status={trade.orderStatus.status}")


def write_daily_pnl(path: Path, account: str | None, equity: float, cash: float | None) -> None:
    row = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "account": account or "all",
        "equity": equity,
        "cash": cash if cash is not None else "",
        "timestamp": datetime.now().isoformat(),
    }
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_live_stream(ib: IB, symbols: list[str], interval: float, duration_minutes: float, market_data_type: int = 1) -> None:
    if not symbols:
        print("No symbols provided for live streaming")
        return
    set_market_data_type(ib, market_data_type)
    print(f"Starting live market data stream for {len(symbols)} symbols (market_data_type={market_data_type})")
    contracts = [stock_contract(symbol) for symbol in symbols]
    tickers = [ib.reqMktData(contract, snapshot=False) for contract in contracts]
    try:
        loops = max(1, int(duration_minutes * 60 / max(interval, 1.0)))
        for i in range(loops):
            ib.sleep(interval)
            print(f"--- market snapshot {i + 1}/{loops} ---")
            for symbol, ticker in zip(symbols, tickers):
                last = getattr(ticker, "last", float("nan"))
                bid = getattr(ticker, "bid", float("nan"))
                ask = getattr(ticker, "ask", float("nan"))
                print(
                    f"{symbol}: last={last if math.isfinite(last) else 'NA'}"
                    f", bid={bid if math.isfinite(bid) else 'NA'}"
                    f", ask={ask if math.isfinite(ask) else 'NA'}"
                )
    finally:
        for ticker in tickers:
            ib.cancelMktData(ticker)


def rebalance_portfolio(
    ib: IB,
    tickers: list[str],
    weights: np.ndarray,
    equity: float,
    account: str | None,
    max_symbol_pct: float,
    max_side_pct: float,
    top_n: int,
    dry_run: bool,
    auto_rebalance: bool,
    market_data_type: int = 1,
) -> tuple[list[str], dict[str, int], dict[str, int], dict[str, int]]:
    target_dollars_raw = compute_target_dollars(weights, equity)
    target_weights = {
        symbol: float(target_dollars_raw[i])
        for i, symbol in enumerate(tickers)
        if np.isfinite(target_dollars_raw[i]) and target_dollars_raw[i] != 0.0
    }
    limited = apply_risk_limits(target_weights, equity, max_symbol_pct, max_side_pct, top_n)
    print(f"Selected {len(limited)} symbols after risk limits")

    live_symbols = list(limited.keys())
    prices = get_latest_prices(ib, live_symbols, market_data_type=market_data_type)
    print("--- live prices ---")
    for symbol in live_symbols:
        print(f"{symbol}: {prices.get(symbol)}")

    target_shares = target_shares_from_dollars(limited, prices)
    current_shares = positions_by_symbol(ib, account=account)
    plan = order_plan(target_shares, current_shares)
    print(f"Planned orders for {len(plan)} symbols")
    for symbol, diff in plan.items():
        print(f"{symbol}: current={current_shares.get(symbol,0)}, target={target_shares.get(symbol,0)}, order={diff}")

    if auto_rebalance:
        if dry_run:
            print("dry-run mode: no live orders will be placed")
        for symbol, diff in plan.items():
            if diff == 0:
                continue
            if dry_run:
                print(f"DRY-RUN: would place {diff} shares for {symbol}")
            else:
                place_market_order(ib, symbol, diff)
    return live_symbols, current_shares, target_shares, plan


def main():
    args = parse_args()
    ib = connect_ibkr(args.host, args.port, args.client_id)
    print("connected:", ib.isConnected())
    print("managedAccounts:", ib.managedAccounts())

    values = account_values(ib, account=args.account)
    summary = summary_equity(values)
    equity = summary.get("equity")
    cash = summary.get("cash")
    print("--- account summary ---")
    print(values)
    print(f"equity={equity}, cash={cash}")

    stream_symbols: list[str] = []
    if args.load_pipeline:
        if equity is None:
            raise RuntimeError("Unable to read equity from IB account values")

        pipeline_path = Path(args.load_pipeline)
        tickers, weights = load_pipeline_target_weights(
            pipeline_path,
            args.date_index,
            method=args.method,
            lookback=args.lookback,
            ewma_halflife=args.ewma_halflife,
            min_periods=args.min_periods,
            quantile=args.quantile,
            gross=args.gross,
        )
        print(f"Loaded {len(tickers)} tickers from {pipeline_path}")

        stream_symbols, current_shares, target_shares, plan = rebalance_portfolio(
            ib=ib,
            tickers=tickers,
            weights=weights,
            equity=equity,
            account=args.account,
            max_symbol_pct=args.max_symbol_pct,
            max_side_pct=args.max_side_pct,
            top_n=args.top_n,
            dry_run=args.dry_run,
            auto_rebalance=args.auto_rebalance,
            market_data_type=args.market_data_type,
        )
        if args.symbols:
            stream_symbols = sorted(set(stream_symbols) | set(args.symbols))
    elif args.symbols:
        stream_symbols = args.symbols
        prices = get_latest_prices(ib, stream_symbols)
        print("--- live prices ---")
        for symbol, price in prices.items():
            print(f"{symbol}: {price}")

    if args.pnl_log and equity is not None:
        write_daily_pnl(Path(args.pnl_log), args.account, equity, cash)
        print(f"Appended daily PnL snapshot to {args.pnl_log}")

    if stream_symbols:
        run_live_stream(
            ib,
            stream_symbols,
            args.stream_interval,
            args.stream_duration_minutes,
            market_data_type=args.market_data_type,
        )

    ib.disconnect()
    print("disconnected")


if __name__ == "__main__":
    main()
