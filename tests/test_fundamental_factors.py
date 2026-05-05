import numpy as np
import pandas as pd

from factor_pipeline.fundamental_factors import flatten_financials, fundamentals_to_daily, build_fundamental_factors


def sample_record(filing_date="2024-05-10"):
    return {
        "ticker": "AAPL",
        "tickers": ["AAPL"],
        "start_date": "2024-01-01",
        "end_date": "2024-03-31",
        "filing_date": filing_date,
        "timeframe": "ttm",
        "financials": {
            "income_statement": {
                "revenues": {"value": 1000},
                "gross_profit": {"value": 400},
                "operating_income_loss": {"value": 250},
                "net_income_loss": {"value": 200},
                "diluted_average_shares": {"value": 100},
            },
            "balance_sheet": {
                "assets": {"value": 2000},
                "equity": {"value": 1000},
                "long_term_debt": {"value": 300},
                "cash_and_cash_equivalents": {"value": 50},
                "current_assets": {"value": 600},
                "current_liabilities": {"value": 300},
            },
            "cash_flow_statement": {
                "net_cash_flow_from_operating_activities": {"value": 220},
                "capital_expenditure": {"value": -20},
            },
        },
    }


def test_available_date_uses_filing_date():
    flat = flatten_financials(pd.DataFrame([sample_record()]), lag_days=60)
    assert flat.loc[0, "available_date"] == pd.Timestamp("2024-05-10")
    assert flat.loc[0, "availability_source"] == "filing_date"


def test_available_date_fallback_uses_end_date_plus_lag():
    flat = flatten_financials(pd.DataFrame([sample_record(filing_date=None)]), lag_days=60)
    assert flat.loc[0, "available_date"] == pd.Timestamp("2024-05-30")
    assert flat.loc[0, "availability_source"] == "end_date_plus_lag"


def test_point_in_time_daily_forward_fill_no_backfill():
    flat = flatten_financials(pd.DataFrame([sample_record()]))
    dates = pd.date_range("2024-05-01", "2024-05-15", freq="B")
    daily = fundamentals_to_daily(flat, dates, ["AAPL"])
    before = daily["revenues"].loc[pd.Timestamp("2024-05-09"), "AAPL"]
    after = daily["revenues"].loc[pd.Timestamp("2024-05-10"), "AAPL"]
    assert np.isnan(before)
    assert after == 1000


def test_basic_fundamental_factor_values():
    flat = flatten_financials(pd.DataFrame([sample_record()]))
    dates = pd.date_range("2024-05-10", "2024-05-14", freq="B")
    daily = fundamentals_to_daily(flat, dates, ["AAPL"])
    close = pd.DataFrame({"AAPL": [10.0, 11.0, 12.0]}, index=dates)
    factors = build_fundamental_factors(daily, close)
    assert factors["earnings_yield"].iloc[0, 0] == 200 / (10 * 100)
    assert factors["gross_margin"].iloc[0, 0] == 400 / 1000
