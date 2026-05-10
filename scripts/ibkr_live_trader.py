from __future__ import annotations

import argparse
import math
import sys
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
    p = argparse.ArgumentParser(description="IBKR live trading helper for paper trading and pipeline target positions")
    p.add_argument("--host", default="127.0.0.1", help="TWS/Gateway host")
    p.add_argument("--port", type=int, default=7497, help="TWS demo socket port")
    p.add_argument("--client-id", type=int, default=1, help="IB API client ID")
    p.add_argument("--account", default=None, help="Optional IB account to filter account values")
    p.add_argument("--symbols", nargs="*", default=["AAPL"], help="Symbols to fetch live market data for")
    p.add_argument("--load-pipeline", default=None, help="Directory containing saved pipeline outputs with tickers.csv and weights.npy or X.npy/factor_returns.npy for fallback calculation")
    p.add_argument("--date-index", type=int, default=-1, help="Index of the date to load target weights from weights.npy or compute from pipeline outputs")
    p.add_argument("--method", default="ewma", choices=["latest", "rolling", "ewma", "zero", "oracle"], help="Factor return prediction method when computing weights from saved pipeline outputs")
    p.add_argument("--lookback", type=int, default=20, help="Lookback window for factor return prediction")
    p.add_argument("--ewma-halflife", type=float, default=20.0, help="EWMA halflife for factor return prediction")
    p.add_argument("--min-periods", type=int, default=5, help="Minimum positive observations for factor return prediction")
    p.add_argument("--quantile", type=float, default=0.10, help="Quantile for long/short portfolio weights when computing from scores")
    p.add_argument("--gross", type=float, default=2.0, help="Gross exposure for long/short weight construction")
    p.add_argument("--notional", type=float, default=100000.0, help="Notional size for target weights when estimating order sizes")
    p.add_argument("--market-data-type", type=int, default=1, help="IB market data type: 1 realtime, 2 frozen, 3 delayed, 4 delayed frozen")
    p.add_argument("--dry-run", action="store_true", help="Do not send live orders; only show what would be traded")
    p.add_argument("--place-sample-order", action="store_true", help="Place a sample market order for the first symbol")
    p.add_argument("--sample-qty", type=int, default=1, help="Quantity for the sample market order")
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


def get_snapshot_price(ib: IB, symbol: str, timeout: float = 5.0) -> float | None:
    contract = stock_contract(symbol)
    ticker = ib.reqMktData(contract, snapshot=True)
    elapsed = 0.0
    while elapsed < timeout:
        price = getattr(ticker, "last", float('nan'))
        if np.isfinite(price):
            break
        ib.sleep(0.05)
        elapsed += 0.05
    price = getattr(ticker, "last", float('nan'))
    if not np.isfinite(price):
        price = getattr(ticker, "close", float('nan'))
    ib.cancelMktData(ticker)
    return float(price) if np.isfinite(price) else None


def fetch_prices(ib: IB, symbols: list[str], market_data_type: int = 1) -> dict[str, float | None]:
    set_market_data_type(ib, market_data_type)
    prices: dict[str, float | None] = {}
    for symbol in symbols:
        try:
            prices[symbol] = get_snapshot_price(ib, symbol)
        except Exception:
            prices[symbol] = None
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


def print_account_summary(ib: IB, account: str | None = None) -> None:
    accounts = ib.managedAccounts()
    print("managedAccounts:", accounts)
    values = ib.accountValues()
    if account is not None:
        values = [v for v in values if v.account == account]
    print(f"accountValues ({len(values)})")
    for v in values[:50]:
        print(v)
    positions = ib.positions()
    print(f"positions ({len(positions)})")
    for pos in positions[:50]:
        print(pos)


def place_market_order(ib: IB, symbol: str, action: str, quantity: int) -> None:
    contract = stock_contract(symbol)
    order = MarketOrder(action, quantity)
    trade = ib.placeOrder(contract, order)
    ib.sleep(1)
    print(f"Placed {action} {quantity} {symbol}, order status={trade.orderStatus.status}")
    return


def main():
    args = parse_args()
    ib = connect_ibkr(args.host, args.port, args.client_id)
    print("connected:", ib.isConnected())
    print("managedAccounts:", ib.managedAccounts())

    print("--- account summary ---")
    print_account_summary(ib, account=args.account)

    if args.symbols:
        print("--- market data ---")
        prices = fetch_prices(ib, args.symbols, market_data_type=args.market_data_type)
        for symbol, price in prices.items():
            print(f"{symbol}: {price}")

    if args.load_pipeline:
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
        target_dollars = compute_target_dollars(weights, args.notional)
        nonzero = np.where(np.isfinite(target_dollars) & (target_dollars != 0.0))[0]
        print(f"Nonzero target positions: {len(nonzero)}")
        for idx in nonzero[:20]:
            print(f"{tickers[idx]} -> target ${target_dollars[idx]:.2f}")
        if len(nonzero) > 20:
            print("...")

    if args.place_sample_order:
        if args.dry_run:
            print("dry run: sample order not sent")
        else:
            symbol = args.symbols[0] if args.symbols else "AAPL"
            action = "BUY" if args.sample_qty > 0 else "SELL"
            place_market_order(ib, symbol, action, abs(args.sample_qty))

    ib.disconnect()
    print("disconnected")


if __name__ == "__main__":
    main()
