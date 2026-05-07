import numpy as np
import pandas as pd

from factor_pipeline.fundamental_factors import flatten_financials, build_ttm_financials, fundamentals_to_daily, build_fundamental_factors


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


def quarterly_record(end_date, filing_date, revenues, net_income, assets):
    return {
        "ticker": "AAPL",
        "tickers": ["AAPL"],
        "start_date": str(pd.Timestamp(end_date) - pd.offsets.QuarterEnd(startingMonth=12) + pd.Timedelta(days=1)),
        "end_date": end_date,
        "filing_date": filing_date,
        "timeframe": "quarterly",
        "financials": {
            "income_statement": {
                "revenues": {"value": revenues},
                "net_income_loss": {"value": net_income},
                "diluted_average_shares": {"value": 100},
            },
            "balance_sheet": {
                "assets": {"value": assets},
                "equity": {"value": assets / 2},
            },
            "cash_flow_statement": {
                "net_cash_flow_from_operating_activities": {"value": net_income + 10},
                "capital_expenditure": {"value": -5},
            },
        },
    }


def test_available_date_uses_filing_date():
    flat = flatten_financials(pd.DataFrame([sample_record()]), lag_days=60)
    assert flat.loc[0, "available_date"] == pd.Timestamp("2024-05-10")
    assert flat.loc[0, "availability_source"] == "filing_date"


def test_flatten_prefers_requested_ticker_over_payload_tickers_list():
    row = sample_record()
    row["ticker"] = "GOOGL"
    row["tickers"] = ["GOOG", "GOOGL"]

    flat = flatten_financials(pd.DataFrame([row]))

    assert flat.loc[0, "ticker"] == "GOOGL"


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


def test_build_ttm_financials_sums_latest_four_known_quarters():
    raw = pd.DataFrame([
        quarterly_record("2023-03-31", "2023-05-01", 100, 10, 1000),
        quarterly_record("2023-06-30", "2023-08-01", 200, 20, 1100),
        quarterly_record("2023-09-30", "2023-11-01", 300, 30, 1200),
        quarterly_record("2023-12-31", "2024-02-01", 400, 40, 1300),
        quarterly_record("2024-03-31", "2024-05-01", 500, 50, 1400),
    ])
    flat = flatten_financials(raw)

    ttm = build_ttm_financials(flat)

    assert np.isnan(ttm.loc[2, "revenues"])
    assert ttm.loc[3, "revenues"] == 1000
    assert ttm.loc[3, "net_income"] == 100
    assert ttm.loc[3, "assets"] == 1300
    assert ttm.loc[3, "free_cash_flow"] == 140 - 20
    assert ttm.loc[4, "revenues"] == 1400


def test_ttm_fundamentals_forward_fill_from_filing_date():
    raw = pd.DataFrame([
        quarterly_record("2023-03-31", "2023-05-01", 100, 10, 1000),
        quarterly_record("2023-06-30", "2023-08-01", 200, 20, 1100),
        quarterly_record("2023-09-30", "2023-11-01", 300, 30, 1200),
        quarterly_record("2023-12-31", "2024-02-01", 400, 40, 1300),
    ])
    ttm = build_ttm_financials(flatten_financials(raw))
    dates = pd.date_range("2024-01-29", "2024-02-05", freq="B")

    daily = fundamentals_to_daily(ttm, dates, ["AAPL"])

    assert np.isnan(daily["revenues"].loc[pd.Timestamp("2024-01-31"), "AAPL"])
    assert daily["revenues"].loc[pd.Timestamp("2024-02-01"), "AAPL"] == 1000
