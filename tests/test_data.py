"""Tests for data loading and cleaning (uses synthetic DataFrames — no xlsx needed)."""

import pandas as pd
import pytest
from probuyer_xai.data import clean


def _make_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "invoice_no": ["536365", "C536366", "536367", "536368"],
            "stock_code": ["85123A", "85123A", "71053", "84406B"],
            "description": ["item1", "item1", "item2", "item3"],
            "quantity": [6, -1, 3, 12],
            "invoice_date": pd.to_datetime(
                ["2010-12-01 08:26", "2010-12-01 09:00", "2010-12-01 10:00", "2010-12-02 11:00"]
            ),
            "unit_price": [2.55, 2.55, 3.39, 1.65],
            "customer_id": ["17850", "17850", "13047", "13047"],
            "country": ["UK", "UK", "UK", "UK"],
        }
    )


def test_standardised_columns_present():
    raw = _make_raw()
    clean_df, _ = clean(raw)
    expected = {
        "invoice_no", "stock_code", "description", "quantity",
        "invoice_date", "unit_price", "customer_id", "country",
    }
    assert expected.issubset(set(clean_df.columns))


def test_amount_calculation():
    raw = _make_raw()
    _, normal = clean(raw)
    row = normal[normal["invoice_no"] == "536365"].iloc[0]
    assert abs(row["amount"] - 6 * 2.55) < 0.01


def test_cancellation_flag_c_prefix():
    raw = _make_raw()
    clean_df, _ = clean(raw)
    assert clean_df.loc[clean_df["invoice_no"] == "C536366", "is_cancelled"].all()


def test_cancellation_flag_negative_qty():
    raw = _make_raw()
    clean_df, _ = clean(raw)
    # negative qty row should also be cancelled
    neg_row = clean_df[clean_df["quantity"] < 0]
    assert neg_row["is_cancelled"].all()


def test_normal_subset_excludes_cancellations():
    raw = _make_raw()
    _, normal = clean(raw)
    assert (normal["quantity"] > 0).all()
    assert (normal["unit_price"] > 0).all()
    assert not normal["is_cancelled"].any()


def test_no_null_customer_id():
    raw = _make_raw()
    raw.loc[0, "customer_id"] = None  # type: ignore[call-overload]
    clean_df, _ = clean(raw)
    assert clean_df["customer_id"].notna().all()
