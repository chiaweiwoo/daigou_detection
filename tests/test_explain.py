"""Tests for deterministic explanation evidence layer."""

import numpy as np
import pandas as pd
import pytest
from probuyer_xai.explain import build_evidence, evidence_for_customer, _confidence


def _make_inputs(n: int = 20):
    rng = np.random.default_rng(0)
    cids = [f"C{i}" for i in range(n)]

    features = pd.DataFrame(
        {
            "total_quantity": rng.integers(10, 5000, n),
            "total_spend": rng.uniform(50, 50000, n),
            "num_invoices": rng.integers(1, 200, n),
            "active_days": rng.integers(1, 365, n),
            "avg_quantity_per_invoice": rng.uniform(1, 500, n),
            "max_quantity_per_invoice": rng.uniform(10, 2000, n),
            "avg_spend_per_invoice": rng.uniform(5, 2000, n),
            "max_spend_per_invoice": rng.uniform(20, 5000, n),
            "unique_skus": rng.integers(1, 100, n),
            "unique_descriptions": rng.integers(1, 100, n),
            "repeat_sku_ratio": rng.uniform(0, 1, n),
            "top_sku_quantity_share": rng.uniform(0, 1, n),
            "top_sku_spend_share": rng.uniform(0, 1, n),
            "quantity_per_active_day": rng.uniform(0.1, 50, n),
            "spend_per_active_day": rng.uniform(0.5, 200, n),
            "bulk_invoice_count": rng.integers(0, 50, n),
            "bulk_invoice_ratio": rng.uniform(0, 1, n),
            "cancelled_invoice_count": rng.integers(0, 10, n),
            "cancelled_quantity_abs": rng.uniform(0, 500, n),
            "cancelled_amount_abs": rng.uniform(0, 2000, n),
            "cancellation_ratio": rng.uniform(0, 0.5, n),
            "first_purchase_date": pd.to_datetime(["2010-12-01"] * n),
            "last_purchase_date": pd.to_datetime(["2011-11-30"] * n),
        },
        index=cids,
    )

    scores = pd.DataFrame(
        {
            "customer_id": cids,
            "anomaly_score": rng.uniform(0, 1, n),
            "risk_percentile": rng.uniform(0, 100, n),
            "risk_band": (["High"] * 2 + ["Medium"] * 4 + ["Low"] * (n - 6)),
        }
    )

    rule_hits = pd.DataFrame(
        {
            "customer_id": cids,
            "rule_hits": [["R001", "R002"] if i < 2 else [] for i in range(n)],
            "rule_reasons": [
                ["High qty", "Large basket"] if i < 2 else [] for i in range(n)
            ],
            "rule_count": [2 if i < 2 else 0 for i in range(n)],
        }
    )

    return features, scores, rule_hits


def test_evidence_has_required_fields():
    features, scores, rule_hits = _make_inputs()
    evidences = build_evidence(features, scores, rule_hits)
    required = {
        "customer_id", "risk_band", "risk_percentile", "anomaly_type",
        "confidence", "top_reasons", "rule_hits", "recommended_action",
    }
    for ev in evidences:
        assert required.issubset(set(ev.keys())), f"Missing fields in {ev.get('customer_id')}"


def test_confidence_high_requires_high_band_and_two_rules():
    assert _confidence("High", 2) == "High"
    assert _confidence("High", 1) == "Medium"
    assert _confidence("Medium", 1) == "Medium"
    assert _confidence("Low", 0) == "Low"


def test_anomaly_type_is_valid_string():
    features, scores, rule_hits = _make_inputs()
    valid_types = {
        "broad_wholesale_buyer",
        "single_product_bulk_buyer",
        "high_frequency_buyer",
        "return_or_cancellation_anomaly",
        "potential_stockout_risk",
    }
    evidences = build_evidence(features, scores, rule_hits)
    for ev in evidences:
        assert ev["anomaly_type"] in valid_types, f"Invalid type: {ev['anomaly_type']}"


def test_evidence_for_customer_lookup():
    features, scores, rule_hits = _make_inputs()
    evidences = build_evidence(features, scores, rule_hits)
    ev = evidence_for_customer("C0", evidences)
    assert ev is not None
    assert ev["customer_id"] == "C0"


def test_evidence_for_missing_customer_returns_none():
    features, scores, rule_hits = _make_inputs()
    evidences = build_evidence(features, scores, rule_hits)
    assert evidence_for_customer("DOES_NOT_EXIST", evidences) is None
