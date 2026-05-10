from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from ib_insync import IB
except ImportError as exc:
    raise ImportError(
        "ib_insync is required to run this script. Install it with `pip install ib_insync`."
    ) from exc


def parse_args():
    p = argparse.ArgumentParser(description="Connect to IBKR TWS demo account and show account info")
    p.add_argument("--host", default="127.0.0.1", help="TWS/Gateway host")
    p.add_argument("--port", type=int, default=7497, help="TWS demo socket port")
    p.add_argument("--client-id", type=int, default=1, help="IB API client ID")
    p.add_argument("--timeout", type=float, default=10.0, help="Connection timeout in seconds")
    p.add_argument("--account", default=None, help="Optional IB account identifier to filter results")
    return p.parse_args()


def main():
    args = parse_args()
    print("IBKR TWS demo connect script")
    print("Please make sure TWS is running, logged in to a demo account, and API access is enabled.")

    ib = IB()
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=args.timeout)
    except Exception as exc:
        print(f"Failed to connect to TWS at {args.host}:{args.port} with clientId={args.client_id}")
        print("Check that TWS is running, API is enabled, and the port is correct.")
        print("Error:", exc)
        return

    print("connected:", ib.isConnected())
    print("managedAccounts:", ib.managedAccounts())

    if args.account:
        print(f"Filtering account values for {args.account}")
        values = [v for v in ib.accountValues() if v.account == args.account]
    else:
        values = ib.accountValues()

    print(f"accountValues ({len(values)})")
    for value in values[:50]:
        print(value)

    positions = ib.positions()
    print(f"positions ({len(positions)})")
    for pos in positions[:50]:
        print(pos)

    ib.disconnect()
    print("disconnected")


if __name__ == "__main__":
    main()
