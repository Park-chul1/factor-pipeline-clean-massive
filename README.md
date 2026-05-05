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
factor_names.csv
factor_diagnostics_raw.csv
tickers.csv
dates.csv
financials_flat.parquet
pipeline_summary.json
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

If the `type` field is available, the pipeline keeps common-stock-like rows. This is a practical starting point, but it is not yet a fully historical point-in-time membership universe. For production-grade research, add delisting handling and a historical tradability filter.

## Factor set

### Price/volume factors

Momentum, reversal, realized volatility, Parkinson volatility, intraday/overnight return, 52-week high/low distance, liquidity, Amihud proxy, skew/kurtosis, volume momentum.

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
2. Winsorize each trading date cross-sectionally at the 1st and 99th percentiles.
3. Z-score each trading date cross-sectionally, so each factor is expressed as a relative exposure across stocks on that date.
4. Drop factors whose processed finite coverage is below `--min-factor-coverage`.
5. Fill remaining missing standardized exposures with `0.0` by default. After z-scoring, zero means neutral exposure. This prevents one missing fundamental field from removing the entire stock from every cross-sectional regression. Use `--no-fill-missing-exposures` to keep NaNs instead.

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
  --financial-limit 50 \
  --min-factor-coverage 0.02 \
  --ridge 1e-4 \
  --out-dir data/test_run_clean \
  --no-cache

python tools/validate_outputs.py data/test_run_clean
```

For a more realistic run, increase `--max-tickers` or remove it, and use `--min-names 30` or higher.
