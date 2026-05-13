# Clean Massive NASDAQ Factor Pipeline

A production-grade, point-in-time cross-sectional factor model pipeline built from Massive.io / Polygon-compatible market data. Combines academic factor theory with practical machine learning and risk management.

## Overview

The pipeline implements a **daily cross-sectional linear factor model** with ridge-regularized parameter estimation, signal generation, backtesting, and automated IBKR live trading.

```text
Massive API
├─ Ticker metadata & OHLCV data
└─ Financial statements
    ↓
Price/Volume Factors (9) + Fundamental Factors (~15)
    ↓
Cross-sectional Exposure Matrix X[T, N, K]
    ↓
Ridge Regression: f[T, K] = (X'X + λI)⁻¹ X'r
    ↓
Factor Return Predictions (EWMA/Rolling/Latest)
    ↓
Signal Generation: scores = X ⊙ f_pred
    ↓
Position Sizing & Risk Limits → Live Trading / Backtest
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your Massive API key:

```bash
export MASSIVE_API_KEY="your_key"
```

## Quick Start

Minimal smoke test:

```bash
python scripts/run_clean_pipeline.py \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --max-tickers 30 \
  --out-dir data/test_clean
```

Full NASDAQ backtest (2023-2026):

```bash
python scripts/run_clean_pipeline.py \
  --start 2023-01-01 \
  --end 2026-01-01 \
  --ticker-status all \
  --out-dir data/nasdaq_full
```

Run backtest:

```bash
python scripts/run_backtest.py \
  --input-dir data/nasdaq_full \
  --method ewma \
  --lookback 120 \
  --quantile 0.10 \
  --ewma-halflife 20
```

## Canonical Daily Trading Workflow

The production convention is now a daily close-based factor strategy:

```text
close[t] data -> after-close signal and target weights -> next trading day open execution
```

This is not an intraday strategy. Signals are computed once after the market close using data available through `signal_date`. The next market session only loads the saved order plan, submits/monitors orders, and records fills. Intraday jobs must not recompute alpha or rebalance from delayed 15-minute bars.

### Daily Data Update

Use Massive grouped daily bars as the default market data source. To append a new completed trading day to an existing parquet cache:

```bash
python scripts/fetch_massive_grouped_daily.py \
  --base-bars data/cache_daily_YYYYMMDD/grouped_daily_2023-01-01_YYYY-MM-DD.parquet \
  --out-bars data/cache_daily_NEXT/grouped_daily_2023-01-01_NEXT-DATE.parquet \
  --start-date NEXT-DATE \
  --end-date NEXT-DATE
```

Then rebuild the factor tensors through the new date. For full-universe runs, date-by-date factor regressions can be parallelized:

```bash
python scripts/run_clean_pipeline.py \
  --start 2023-01-01 \
  --end YYYY-MM-DD \
  --ticker-status all \
  --out-dir data/processed_daily_YYYYMMDD \
  --cache-dir data/cache_daily_YYYYMMDD \
  --min-names 30 \
  --ridge auto \
  --ridge-solver qr \
  --estimation-workers 6 \
  --financial-timeframe quarterly \
  --financial-limit 100
```

### Backtest

The close-to-next-open backtest uses `signal_date = t`, `signal_price = close[t]`, `execution_date = next trading day`, and `execution_price = open[t+1]` by default:

```bash
python scripts/run_close_to_next_open_backtest.py \
  --config configs/daily_factor_backtest.yaml
```

Backtest PnL subtracts turnover-based transaction cost and slippage. The default order eligibility rule requires predicted alpha net of estimated trading cost to exceed the configured threshold:

```yaml
transaction_cost_bps: 1.0
slippage_bps: 2.0
min_net_alpha_after_cost_bps: 5.0
```

With these defaults, a new long needs alpha of at least 8 bps and a new short needs alpha of at most -8 bps.

### After-Close Signal Generation

After the market close, generate the daily report and next-day order plan:

```bash
python scripts/run_after_close_job.py \
  --date YYYY-MM-DD \
  --config configs/daily_live.yaml
```

This writes:

```text
reports/daily/YYYY-MM-DD/summary.md
reports/daily/YYYY-MM-DD/orders.csv
reports/daily/YYYY-MM-DD/target_positions.csv
reports/daily/YYYY-MM-DD/alpha_rankings.csv
reports/daily/YYYY-MM-DD/factor_returns.csv
reports/daily/YYYY-MM-DD/factor_forecast.csv
reports/daily/YYYY-MM-DD/quantile_summary.csv
reports/daily/YYYY-MM-DD/residuals.csv
reports/daily/YYYY-MM-DD/residual_heatmap.png
reports/daily/YYYY-MM-DD/execution_quality.csv
reports/daily/YYYY-MM-DD/risk_summary.json
reports/daily/YYYY-MM-DD/pipeline_metadata.json
```

Each proposed order includes factor contribution attribution and cost-adjusted alpha fields such as `net_alpha_after_cost_bps` and `passes_cost_threshold`.

### Paper Execution

For IBKR paper trading, submit a saved daily report. The default host is `172.30.1.41`, paper port `7497`.

Dry run:

```bash
python scripts/submit_daily_paper_orders.py \
  --report-dir reports/daily/YYYY-MM-DD \
  --dry-run
```

Actual paper submit:

```bash
python scripts/submit_daily_paper_orders.py \
  --report-dir reports/daily/YYYY-MM-DD \
  --submit-mode burst
```

The submitter re-checks the same cost-adjusted threshold before generating broker orders, so stale reports cannot bypass the alpha-after-cost rule. It records order status and estimated costs in `paper_order_results.csv`.

### Intraday Monitoring

During market hours, use intraday monitoring only for order status, fills, exposure, cash/margin, connectivity, and abnormal conditions:

```bash
python scripts/run_intraday_monitor.py --report-dir reports/daily
```

Legacy intraday delayed-data rebalancing is disabled by default and requires `ALLOW_LEGACY_INTRADAY_REBALANCE=1` for explicit diagnostics.

## Pipeline Outputs

```text
X.npy                         # T x N x K exposure tensor
r.npy                         # T x N forward returns
factor_returns.npy            # T x K realized cross-sectional factor returns
factor_names.csv              # K factor names
tickers.csv                   # N ticker symbols
dates.csv                     # T trading dates
tradable_mask.npy             # T x N liquidity/volume mask
factor_diagnostics_*.csv      # Preprocessing diagnostics
financials_flat.parquet       # Vendor filing rows
financials_ttm.parquet        # Point-in-time TTM fields
pipeline_summary.json         # Metadata and parameters
```

Residual diagnostics can be generated from any pipeline output directory with `python scripts/plot_residual_heatmap.py --input-dir data/nasdaq_full --clip-percentile 99.5 --sort-tickers-by coverage --top-tickers 50 --top-events 100`. The script writes `diagnostics/residuals/` under the input directory, including percentile-clipped squared residual heatmaps, daily cross-sectional MSE and valid-count plots, ticker coverage diagnostics, largest residual events and daily MSE spike CSVs, and `residual_summary.json`. Heatmap data keeps all ticker rows unless `--filter-low-coverage` is passed; axis labels are subsampled only to keep the image readable.

## Data Preprocessing & Bias Controls

### Point-in-Time Data Availability

**Prices:** Daily bar available only after that trading date.

**Financials:** Filing availability strictly point-in-time:
- Filing date used if available
- Fallback: end_date + lag (default 60 calendar days)
- Forward-fill only; no backfilling into past dates

**Returns:** Forward return (t+1 to t+h), not lagged return.

### Factor Exposure Processing

For each date:

1. Apply tradable_mask (liquidity, volume, listing status)
2. Winsorize at 1st and 99th percentiles
3. Z-score cross-sectionally
4. Drop factors with < min_factor_coverage finite observations
5. Drop factors with |corr| ≥ max_factor_corr with earlier factors
6. Fill remaining NaNs with 0.0 (neutral exposure)

### Survivorship Bias Reduction

- Full NASDAQ universe with `--ticker-status all` includes inactive/delisted names
- `tradable_mask[t, n]` ensures only liquid, listed stocks enter regressions
- No look-ahead bias in financial statement availability

## Factor Set

### Price/Volume (9 factors)

Momentum, realized volatility, Parkinson volatility, intraday/overnight return, 52-week distance, liquidity, Amihud proxy, skew, kurtosis, volume momentum.

### Fundamental (~15 factors)

**Valuation:** earnings_yield, book_to_market, sales_to_price, cashflow_to_price, fcf_yield  
**Profitability:** roe, roa, gross_margin, operating_margin, net_margin  
**Financial Health:** debt_to_equity, current_ratio, cash_to_assets, asset_turnover  
**Growth:** revenue_growth_yoy, net_income_growth_yoy

Total: **~45 factors** across price/volume and fundamental domains.

## Cross-Sectional Ridge Regression

### Model

For each trading date t:

```
r_t = X_t f_t + ε_t
```

- r_t ∈ ℝ^N: forward returns
- X_t ∈ ℝ^{N×K}: winsorized, z-scored exposures
- f_t ∈ ℝ^K: factor returns
- ε_t: idiosyncratic residuals

### Ridge-Regularized OLS Estimator

```
f_t = (X_t'X_t + λI)^(-1) X_t'r_t
```

where:
- **λ** (ridge parameter): Controls regularization strength
- Default: `--ridge 1e-4` balances fit and stability
- Prevents overfitting when K > N or X'X is ill-conditioned

### Why Ridge Regression

- **Stability:** X'X can be singular or ill-conditioned on small universes
- **Generalization:** Reduces variance of estimates
- **Academic grounding:** Standard in cross-sectional factor estimation (Fama, Macbeth; Blitz, Hanauer, Vidojevic)

## Signal Generation & Daily Rebalancing

### Factor Return Prediction Methods

Predict f_t+1 using only history up to t:

- **Latest:** Most recent non-NaN factor return
- **Rolling:** Mean over past lookback window
- **EWMA:** Exponentially weighted mean (halflife parameter)
- **Oracle:** f_t+1 = f_t (look-ahead bias diagnostic only)

### Daily Score Computation

For each stock on date t:

```
alpha[t, n] = X[t, n, :] @ f_pred[t+1, :]
```

The saved reports also include per-factor contribution attribution:

```text
contribution[k] = X[t, n, k] * f_pred[t+1, k]
```

### Position Construction

Long/short quantile portfolio:

- Long: Top 10% scores (or specified quantile)
- Short: Bottom 10%
- Equal-weight within each leg
- Gross exposure: 2.0 (long +1.0, short -1.0)

### Daily Rebalancing

Backtest and live trading rebalance every trading day:

1. After close[t], update daily OHLCV/fundamental data.
2. Build exposures X_t using only information available through close[t].
3. Estimate historical factor returns from past data.
4. Forecast f_hat[t+1] and compute alpha_t = X_t @ f_hat[t+1].
5. Apply liquidity, risk, turnover, and net-alpha-after-cost thresholds.
6. Save the next-day order plan and daily report.
7. On t+1, execute near the open without recomputing alpha intraday.

## Backtesting Framework

### Implementation

- **`factor_pipeline/backtest.py`**: Daily rebalancing backtest engine
- **`factor_pipeline/signals.py`**: Factor prediction and signal generation
- **`scripts/run_backtest.py`**: Batch backtest runner

### Key Metrics

- **Daily returns:** Position weights × forward returns
- **Turnover:** Sum of absolute position changes
- **Sharpe ratio:** Mean return / volatility (annualized)
- **Calmar ratio:** Return / max drawdown
- **Sortino ratio:** Excess return / downside deviation

### Example Results (2023-01-01 to 2026-03-01 NASDAQ)

**EWMA (halflife=20, lookback=120, quantile=0.1):**
- Sharpe: 4.88
- Annualized return: ~45%
- Max drawdown: ~12%
- Gross leverage: 2.0
- Average turnover: ~15% per day

## IBKR Paper Trading and Legacy Helpers

### Daily Paper Execution

Canonical paper execution uses `scripts/submit_daily_paper_orders.py` against a saved daily report. It does not recompute signals from intraday data.

```bash
python scripts/submit_daily_paper_orders.py \
  --report-dir reports/daily/YYYY-MM-DD \
  --submit-mode burst
```

### Legacy Portfolio Helper

**Script:** `scripts/ibkr_live_portfolio.py`

Connects to Interactive Brokers via TWS/Gateway API:

1. Load pipeline predictions (or compute on-the-fly)
2. Fetch current account equity
3. Compute target positions (notional per symbol)
4. Apply risk limits:
   - Max **2% equity per symbol**
   - Max **60% total long** or **short** exposure
   - Top **50 symbols** by notional size
5. Calculate share deltas (target - current)
6. Place market or limit orders

This legacy helper is blocked from auto-rebalancing by default under the daily close-to-next-open convention. Use it only for diagnostics unless `ALLOW_LEGACY_INTRADAY_REBALANCE=1` is explicitly set.

### Market Data & Delayed Execution

**Data Subscription Issue:**
- IBKR real-time data requires paid subscription
- Workaround: Use `--market-data-type 3` (Delayed) instead
- Delayed data: **~15 minutes behind** real-time
- Do not use delayed intraday data to recompute alpha or rebalance this strategy

**Market Hours Constraint:**
- Market Orders only execute during US regular hours (09:30-16:00 ET)
- Submitted orders wait until market open
- Use `GTC` (Good-Till-Cancelled) for overnight orders

### Example Commands

**Dry run (no actual orders):**

```bash
python scripts/ibkr_live_portfolio.py \
  --host 127.0.0.1 \
  --port 7497 \
  --load-pipeline data/nasdaq_full \
  --method ewma \
  --lookback 120 \
  --market-data-type 3 \
  --auto-rebalance \
  --dry-run
```

**Legacy live helper, explicitly disabled for normal trading:**

```bash
ALLOW_LEGACY_INTRADAY_REBALANCE=1 \
python scripts/ibkr_live_portfolio.py \
  --host 172.30.1.41 \
  --port 7497 \
  --load-pipeline data/nasdaq_full \
  --auto-rebalance \
  --market-data-type 3 \
  --top-n 50 \
  --max-symbol-pct 0.02 \
  --max-side-pct 0.60
```

## Caching Strategy

API responses cached locally to avoid redundant downloads:

```text
<cache-dir>/                    # Parquet datasets (tickers, bars, financials)
<cache-dir>/api_responses/**    # Raw JSON responses keyed by endpoint + params
```

Use `--no-cache` to refresh parquet datasets only. Disable API cache with `--no-api-cache`.

## Tests

```bash
pytest -q
```

The daily workflow tests cover close-to-next-open timing, cost-adjusted signal thresholds, report generation, live/paper order generation, factor estimation, and ridge diagnostics.

## Project Structure

```
factor_pipeline/
├── backtest.py          # Backtesting engine
├── signals.py           # Factor prediction & scoring
├── estimation.py        # Ridge regression estimator
├── preprocessing.py     # Exposure tensor construction
├── fundamental_factors.py
├── price_volume_factors.py
├── massive_client.py
├── config.py
└── ...

scripts/
├── run_clean_pipeline.py       # Build factor pipeline
├── run_backtest.py             # Backtest execution
├── ibkr_live_portfolio.py      # Live rebalancing
├── ibkr_live_trader.py         # Market data helper
└── connect_ibkr_demo.py        # Connection diagnostics

tests/                          # Unit tests
tools/
└── validate_outputs.py         # Output validation

data/
└── run_nasdaq_all_.../ → X.npy, r.npy, etc.
```

## References

- Fama, E. F., & MacBeth, J. D. (1973). "Risk, return, and equilibrium: Empirical tests." *Journal of Political Economy*, 81(3), 607-636.
- Blitz, D., Hanauer, M. X., Vidojevic, M., & Hanauer, M. X. (2019). "Five concerns with factor investing." *Journal of Portfolio Management*, 45(4).
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. (Ridge Regression)


By default the runner requests `--ticker-status all`, so inactive metadata is included when the data vendor returns it. Use `--ticker-status active` only when you explicitly want active-only behavior.

For a full NASDAQ backtest from 2023-01-01 to 2026-01-01, run with `--start 2023-01-01 --end 2026-01-01 --ticker-status all` and omit `--max-tickers` if you want the broadest universe. If you want a start-date style top-500 universe, use `--max-tickers 500 --universe-rank-by dollar_volume --universe-rank-anchor start`.

The pipeline also saves `tradable_mask.npy`, a dense boolean `date x ticker` matrix derived from finite positive adjusted close and positive volume. This mask is applied before cross-sectional factor preprocessing and again during factor-return estimation, so dead/not-yet-listed/non-trading names do not enter z-scores or regressions.

This is a practical starting point, but it is not yet a fully historical point-in-time exchange-membership universe. For production-grade research, add delisting-return handling and a vendor-backed historical membership filter.

## Factor set

### Price/volume factors

Momentum, realized volatility, Parkinson volatility, intraday/overnight return, 52-week high/low distance, liquidity, Amihud proxy, skew/kurtosis, volume momentum.

`rev_*` factors are intentionally not included because they are exact sign-flipped duplicates of short-horizon `mom_*` factors.

### Fundamental factors

Value, quality, profitability, leverage, liquidity, and growth factors, including:

```text
earnings_yield
book_to_market
sales_to_price
cashflow_to_price
fcf_yield
roe
roa
gross_margin
operating_margin
net_margin
debt_to_equity
current_ratio
cash_to_assets
asset_turnover
revenue_growth_yoy
net_income_growth_yoy
```

Some factors require share-count fields from the financial payload. If Massive does not provide the relevant field for a company/period, the factor is left as NaN and excluded date-by-date during regression.

## Regression model

For each date:

```text
r_t = X_t f_t + epsilon_t
```

The estimator uses cross-sectional ridge-regularized OLS:

```text
f_t = (X_t'X_t + lambda I)^(-1) X_t'r_t
```

Rows with missing returns or missing factor exposures are dropped date-by-date.

## Tests

```bash
pytest -q
```

## Preprocessing, filtering, and validation policy

This version performs the factor research preprocessing inside the pipeline before writing `X.npy`.

### Factor preprocessing

For every raw factor panel `factor[date, ticker]`:

1. Keep natural missing values from insufficient price history, missing OHLCV, missing financials, or point-in-time availability rules.
2. Apply `tradable_mask[date, ticker]` so inactive, dead, not-yet-listed, or non-trading names do not enter cross-sectional statistics.
3. Winsorize each trading date cross-sectionally at the 1st and 99th percentiles.
4. Z-score each trading date cross-sectionally, so each factor is expressed as a relative exposure across stocks on that date.
5. Drop factors whose processed finite coverage is below `--min-factor-coverage`.
6. Drop later factors whose absolute correlation with an earlier kept factor is at least `--max-factor-corr` over at least `--corr-min-overlap` finite observations. Use `--max-factor-corr 0` to disable this filter.
7. Fill remaining missing standardized exposures with `0.0` by default. After z-scoring, zero means neutral exposure. This prevents one missing fundamental field from removing the entire stock from every cross-sectional regression. Use `--no-fill-missing-exposures` to keep NaNs instead.

The preprocessed factor diagnostics are saved to:

```text
factor_diagnostics_preprocessed.csv
```

### Regression policy

Daily factor returns are estimated by cross-sectional ridge regression. With `--ridge > 0`, the model can still produce stable factor returns when the number of factors is larger than the number of stocks in a tiny smoke test. For production research, use a large universe and keep `--min-names` reasonably high.

### Output validation

After a run, validate the generated arrays with:

```bash
python tools/validate_outputs.py data/test_run3
```

For a tiny smoke test, use relaxed thresholds if necessary:

```bash
python tools/validate_outputs.py data/test_run3 \
  --min-x-finite 0.95 \
  --min-r-finite 0.50 \
  --min-finite-factor-returns 0.10
```

### Recommended smoke test

```bash
python scripts/run_clean_pipeline.py \
  --start 2024-01-01 \
  --end 2024-03-01 \
  --max-tickers 50 \
  --min-names 10 \
  --financial-limit 100 \
  --financial-timeframe quarterly \
  --min-factor-coverage 0.02 \
  --ridge 1e-4 \
  --out-dir data/test_run_clean \
  --no-cache

python tools/validate_outputs.py data/test_run_clean
```

For a more realistic run, increase `--max-tickers` or remove it, and use `--min-names 30` or higher.


## Mathematical Framework

This project implements a cross-sectional factor model based on linear algebra and statistical estimation.

### Linear Factor Model

For each date t, we model stock returns as:

r_t = X_t f_t + ε_t

where:
- r_t ∈ ℝ^N: forward returns across N stocks
- X_t ∈ ℝ^{N×K}: factor exposure matrix
- f_t ∈ ℝ^K: factor returns
- ε_t: idiosyncratic noise

### Ordinary Least Squares (OLS)

Factor returns are estimated via:

f_t = (X_tᵀ X_t)⁻¹ X_tᵀ r_t

This corresponds to the least-squares solution minimizing:

||r_t - X_t f_t||²

### Projection Interpretation

OLS can be interpreted geometrically as projecting r_t onto the column space of X_t:

X_t f_t = Proj_{Col(X_t)}(r_t)

The residual ε_t is orthogonal to the factor space:

ε_t ⟂ Col(X_t)

### Regularization (Ridge)

To ensure numerical stability and mitigate multicollinearity:

f_t = (X_tᵀ X_t + λI)⁻¹ X_tᵀ r_t

### Cross-sectional Standardization

Each factor is normalized per date:

z = (x - μ) / σ

This ensures comparability across factors and prevents scale dominance.

### Information Coefficient (IC)

We evaluate factor quality using cross-sectional correlation:

IC_t = corr(X_{:,k}, r_t)

Typically using Spearman rank correlation.

---

This framework connects linear algebra (projection, subspaces) with statistical learning (estimation, noise decomposition).
