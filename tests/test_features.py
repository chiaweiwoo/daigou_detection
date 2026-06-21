"""Tests for feature engineering (synthetic data only)."""

import numpy as np
import pandas as pd
import pytest
from probuyer_xai.features import build_features
from probuyer_xai.config import MODEL_FEATURES


def _make_transactions():
    """Minimal synthetic transaction data."""
    clean_df = pd.DataFrame(
        {
            "invoice_no": ["A1", "A2", "B1", "B2", "B3", "B3"],
            "stock_code": ["S1", "S1", "S2", "S3", "S2", "S4"],
            "description": ["item1", "item1", "item2", "item3", "item2", "item4"],
            "quantity": [10, 5, 100, 200, 150, 3],
            "invoice_date": pd.to_datetime(
                ["2010-12-01", "2010-12-05", "2010-12-01", "2010-12-02", "2010-12-10", "2010-12-10"]
            ),
            "unit_price": [2.0, 2.0, 1.5, 1.0, 1.5, 0.5],
            "amount": [20.0, 10.0, 150.0, 200.0, 225.0, 1.5],
            "customer_id": ["C1", "C1", "C2", "C2", "C2", "C2"],
            "country": ["UK", "UK", "UK", "UK", "UK", "UK"],
            "is_cancelled": [False, False, False, False, False, False],
        }
    )
    normal_df = clean_df[~clean_df["is_cancelled"]].copy()
    return clean_df, normal_df


def test_one_row_per_customer():
    clean_df, normal_df = _make_transactions()
    features, _ = build_features(clean_df, normal_df)
    assert features.index.nunique() == features.shape[0]
    assert set(features.index) == {"C1", "C2"}


def test_no_infinite_values():
    clean_df, normal_df = _make_transactions()
    features, _ = build_features(clean_df, normal_df)
    numeric_cols = features.select_dtypes(include="number").columns
    assert not np.isinf(features[numeric_cols].values).any()


def test_bulk_invoice_count_nonnegative():
    clean_df, normal_df = _make_transactions()
    features, _ = build_features(clean_df, normal_df)
    assert (features["bulk_invoice_count"] >= 0).all()


def test_ratios_in_range():
    clean_df, normal_df = _make_transactions()
    features, _ = build_features(clean_df, normal_df)
    ratio_cols = [c for c in MODEL_FEATURES if "ratio" in c or "share" in c]
    for col in ratio_cols:
        if col in features.columns:
            assert features[col].between(0, 1).all(), f"{col} out of [0,1]"


def test_no_nan_in_model_features():
    clean_df, normal_df = _make_transactions()
    features, _ = build_features(clean_df, normal_df)
    for col in MODEL_FEATURES:
        if col in features.columns:
            assert features[col].notna().all(), f"NaN found in {col}"


def test_metadata_contains_bulk_threshold():
    clean_df, normal_df = _make_transactions()
    _, meta = build_features(clean_df, normal_df)
    assert "bulk_threshold_qty" in meta
    assert meta["bulk_threshold_qty"] > 0
