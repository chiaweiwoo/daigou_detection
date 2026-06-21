"""Tests for business rule engine (no external files required)."""

import json
import numpy as np
import pandas as pd
import pytest
from probuyer_xai.rules import apply_rules


_SAMPLE_RULES = {
    "version": "v1",
    "description": "Test rules",
    "rules": [
        {
            "rule_id": "R001",
            "name": "Extreme total quantity",
            "feature": "total_quantity",
            "operator": ">=",
            "threshold_source": "p99",
            "risk_reason": "Very high total quantity.",
        },
        {
            "rule_id": "R003",
            "name": "High bulk invoice ratio",
            "feature": "bulk_invoice_ratio",
            "operator": ">=",
            "threshold_source": "p95",
            "risk_reason": "Large share of bulk invoices.",
        },
    ],
}


def _make_features(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(n)],
            "total_quantity": rng.integers(1, 5000, n),
            "bulk_invoice_ratio": rng.uniform(0, 1, n),
        }
    ).set_index("customer_id")


def test_rules_load_from_dict():
    assert "rules" in _SAMPLE_RULES
    assert len(_SAMPLE_RULES["rules"]) == 2


def test_apply_rules_returns_expected_columns():
    features = _make_features()
    result = apply_rules(features, _SAMPLE_RULES)
    assert "customer_id" in result.columns
    assert "rule_hits" in result.columns
    assert "rule_count" in result.columns


def test_at_least_one_rule_triggers():
    features = _make_features(100)
    result = apply_rules(features, _SAMPLE_RULES)
    assert result["rule_count"].max() > 0, "No rule triggered on synthetic data"


def test_rule_hits_are_lists():
    features = _make_features()
    result = apply_rules(features, _SAMPLE_RULES)
    for hits in result["rule_hits"]:
        assert isinstance(hits, list)


def test_rule_count_matches_hits_length():
    features = _make_features()
    result = apply_rules(features, _SAMPLE_RULES)
    for _, row in result.iterrows():
        assert len(row["rule_hits"]) == row["rule_count"]


def test_synthetic_extreme_triggers_r001():
    """A customer with max total_quantity should hit R001."""
    features = _make_features(100)
    # Force one customer to have very high total_quantity
    features.iloc[0, features.columns.get_loc("total_quantity")] = 999_999
    result = apply_rules(features, _SAMPLE_RULES)
    top_cid = features.index[0]
    top_row = result[result["customer_id"] == top_cid].iloc[0]
    assert "R001" in top_row["rule_hits"]
