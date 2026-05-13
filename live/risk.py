from __future__ import annotations

import pandas as pd

from live.broker import Order
from live.config import LiveConfig


def assert_demo_safety(config: LiveConfig) -> None:
    config.validate()
    if not config.PAPER_TRADING and not config.ENABLE_REAL_TRADING:
        raise RuntimeError("Neither paper trading nor explicit real trading is enabled")


def apply_risk_checks(orders: list[Order], config: LiveConfig, equity: float, prices: pd.Series) -> tuple[list[Order], list[dict]]:
    checked: list[Order] = []
    rows: list[dict] = []
    for order in orders:
        notional = abs(order.quantity * order.limit_price)
        max_notional = equity * config.MAX_POSITION_WEIGHT
        ok = order.quantity > 0 and order.limit_price > 0 and notional <= max_notional * 1.10
        reason = "" if ok else "size_or_price_limit"
        rows.append({
            "ticker": order.ticker,
            "side": order.side,
            "quantity": order.quantity,
            "limit_price": order.limit_price,
            "notional": notional,
            "ok": ok,
            "reason": reason,
        })
        if ok:
            checked.append(order)
    return checked, rows
