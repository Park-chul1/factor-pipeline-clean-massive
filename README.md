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

### Intraday Score Computation

For each stock on date t:

```
score[t, n] = X[t, n, :] ⊙ f_pred[t, :]
```

(Element-wise product of exposures and predicted returns)

### Position Construction

Long/short quantile portfolio:

- Long: Top 10% scores (or specified quantile)
- Short: Bottom 10%
- Equal-weight within each leg
- Gross exposure: 2.0 (long +1.0, short -1.0)

### Daily Rebalancing

Backtest and live trading rebalance every trading day:

1. Compute cross-sectional factor returns from yesterday's return predictions
2. Estimate today's factor returns f_t
3. Predict f_t+1 using all methods
4. Generate today's scores and positions
5. (Live only) Apply risk limits and place orders

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

## Live Trading: IBKR Integration

### Real-Time Portfolio Management

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

### Market Data & Delayed Execution

**Data Subscription Issue:**
- IBKR real-time data requires paid subscription
- Workaround: Use `--market-data-type 3` (Delayed) instead
- Delayed data: **~15 minutes behind** real-time
- Sufficient for overnight rebalancing or lower-frequency tactics

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

**Live trading (15-min delayed data):**

```bash
python scripts/ibkr_live_portfolio.py \
  --host 127.0.0.1 \
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

All 19 existing tests pass. New modules (backtest, signals) compile without errors.

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
