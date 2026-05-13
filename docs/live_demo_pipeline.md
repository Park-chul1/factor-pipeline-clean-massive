# Live Demo Pipeline

This is a delayed-data signal and paper-execution pipeline. It is intentionally safe by default:

- `PAPER_TRADING = true`
- `ENABLE_REAL_TRADING = false`
- real trading also requires `LIVE_TRADING_CONFIRM=I_UNDERSTAND_REAL_MONEY_RISK`

Run one demo iteration:

```bash
python scripts/run_live_demo_pipeline.py --once --dry-run
```

Run with Massive delayed market data and the currently reachable IBKR paper gateway:

```bash
python scripts/run_live_demo_pipeline.py \
  --data-provider massive \
  --ibkr-paper \
  --ibkr-host 172.30.1.41 \
  --ibkr-port 7497 \
  --ibkr-account DUN189935 \
  --broker-client-id 72 \
  --cache-path data/live_cache/massive_15m_bars.parquet \
  --prefetch-history \
  --universe-limit 100 \
  --fetch-batch-size 25 \
  --fetch-workers 8
```

Use `--prefetch-only` to warm the local 15-minute Massive cache without computing signals or sending paper orders. IBKR is used only for paper account state and order routing unless `--data-provider ibkr_delayed` is explicitly set.

Use the full active universe:

```bash
python scripts/run_live_demo_pipeline.py \
  --data-provider massive \
  --cache-path data/live_cache/massive_15m_bars.parquet \
  --prefetch-history \
  --prefetch-only \
  --all-universe \
  --fetch-batch-size 0 \
  --fetch-workers 16
```

`--fetch-batch-size 0` means refresh every missing/outdated symbol in the loop. That is expensive for full active NASDAQ because Massive intraday bars are currently fetched per ticker. Use `--fetch-workers` to parallelize those calls, keeping the value low enough to avoid provider rate limits.

The pipeline fetches the latest available delayed bar, appends it to `data/live_cache/bars.parquet`, builds only the latest exposure row from recent cached history, computes `alpha = X_now @ f_pred`, applies alpha, liquidity, turnover, holding-period, and exposure controls, then sends orders only to the simulated paper broker.

Logs are appended to:

- `logs/live_signal_events.csv`
- `logs/orders.csv`
- `logs/risk_checks.csv`
- `logs/pipeline_latency.csv`

If `MASSIVE_API_KEY` is unavailable, the data layer falls back to cached bars only. The provider abstraction is isolated in `live/data.py` so a true batch/grouped intraday endpoint can replace per-symbol fetches without changing signal, portfolio, or execution code.
