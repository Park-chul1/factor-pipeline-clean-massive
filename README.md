# Clean Massive NASDAQ Factor Pipeline

A clean, point-in-time style factor-model pipeline built from scratch for Massive.io / Polygon-compatible market data.

It builds:

```text
Massive ticker metadata
+ Massive grouped daily OHLCV
+ Massive financial statements
→ price/volume factors
→ fundamental factors
→ X[T, N, K]
→ forward returns r[T, N]
→ cross-sectional factor returns f[T, K]
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your key locally:

```bash
export MASSIVE_API_KEY="your_key"
```

## Smoke test

Start small before downloading all NASDAQ names:

```bash
python scripts/run_clean_pipeline.py \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --max-tickers 30 \
  --out-dir data/test_clean
```

Full-ish run:

```bash
python scripts/run_clean_pipeline.py \
  --start 2019-01-01 \
  --end 2024-12-31 \
  --out-dir data/processed_clean
```

## Outputs

```text
X.npy                         # T x N x K exposure tensor, winsorized/z-scored by date
r.npy                         # T x N forward returns
factor_returns.npy            # T x K cross-sectional factor returns
tradable_mask.npy             # T x N bool mask aligned to dates.csv and tickers.csv
factor_names.csv
factor_diagnostics_raw.csv
tickers.csv
dates.csv
financials_flat.parquet         # flattened vendor filing rows
financials_ttm.parquet          # quarterly rows converted to point-in-time TTM fields
pipeline_summary.json
```

## Caching

The runner uses two cache layers:

```text
<cache-dir>/*.parquet              # built datasets such as tickers, grouped bars, financial rows
<cache-dir>/api_responses/**/*.json # raw Massive API responses keyed by endpoint + params
```

`--no-cache` disables the parquet dataset cache only. The raw API response cache stays on, so rebuilding outputs does not have to re-download the same API responses. Use `--no-api-cache` only when you intentionally want fresh API responses.

You can choose a separate API cache location with:

```bash
--api-cache-dir data/massive_api_cache
```

## Data policy and look-ahead bias controls

### Prices

Daily OHLCV comes from Massive grouped daily bars:

```text
/v2/aggs/grouped/locale/us/market/stocks/{date}?adjusted=true
```

The pipeline treats a daily bar as available only after that trading date. Price/volume factors at date `t` use only data up to and including date `t`.

### Fundamentals

Financial statements come from:

```text
/vX/reference/financials
```

By default the pipeline downloads historical `quarterly` financial rows using report-period filters. It asks for extra history before the backtest start via `--financial-lookback-days` so the first trading dates can already have enough prior quarters to form TTM values.

For every financial statement row, the pipeline creates an `available_date`:

1. If `filing_date` exists, use `filing_date`.
2. If `filing_date` is missing, use `end_date + financial_lag_days`.

Default fallback lag is 60 calendar days. This is conservative enough for most quarterly filings, but you can adjust it:

```bash
--financial-lag-days 75
```

Fundamental data is then merged onto the daily trading calendar by ticker using point-in-time forward fill:

```text
Only statements with available_date <= trading_date are visible.
```

The pipeline never backfills a later filing into an earlier trading date.

When `--financial-timeframe quarterly` is used, flow fields such as revenue, net income, operating cash flow, and capex are converted to TTM by summing the latest four quarterly rows known at each filing date. Balance-sheet fields such as assets, equity, debt, and cash use the latest reported quarter. The resulting TTM event table is saved as:

```text
financials_ttm.parquet
```

### Returns

The target return is forward return:

```text
r[t, n] = adj_close[t + horizon, n] / adj_close[t, n] - 1
```

So `X[t]` explains or predicts the future return after `t`, not the past return ending at `t`.

### Universe

Ticker universe is built from Massive ticker metadata with:

```text
market=stocks
exchange=XNAS
```

By default the runner requests `--ticker-status all`, so inactive metadata is included when the data vendor returns it. Use `--ticker-status active` only when you explicitly want active-only behavior.

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
