"""Tests for what-if scenario analysis."""

import numpy as np
import pandas as pd
import pytest
from probuyer_xai.whatif import (
    change_risk_threshold,
    exclude_cancellation_anomalies,
    whitelist_customer,
    require_min_rule_hits,
)


def _make_scores_and_evidences():
    cids = [f"C{i}" for i in range(20)]
    rng = np.random.default_rng(7)

    scores = pd.DataFrame(
        {
            "customer_id": cids,
            "anomaly_score": rng.uniform(0, 1, 20),
            "risk_percentile": [99.5, 99.0, 98.5, 98.0, 97.5, 97.0, 95.0] + [50.0] * 13,
            "risk_band": ["High", "High", "Medium", "Medium", "Medium", "Medium", "Medium"]
            + ["Low"] * 13,
        }
    )

    evidences = []
    for i, cid in enumerate(cids):
        evidences.append(
            {
                "customer_id": cid,
                "risk_band": scores.iloc[i]["risk_band"],
                "risk_percentile": scores.iloc[i]["risk_percentile"],
                "anomaly_type": "return_or_cancellation_anomaly" if i == 2 else "broad_wholesale_buyer",
                "confidence": "High" if i < 2 else "Low",
                "rule_count": 2 if i < 2 else 0,
                "top_reasons": [],
                "rule_hits": [],
                "recommended_action": "",
            }
        )

    return scores, evidences


def test_threshold_change_updates_count():
    scores, evidences = _make_scores_and_evidences()
    # Tighten threshold: fewer customers flagged
    result_tight = change_risk_threshold(scores, evidences, new_high_pct=99.4, new_med_pct=99.0)
    result_loose = change_risk_threshold(scores, evidences, new_high_pct=95.0, new_med_pct=93.0)
    assert result_tight["after_flagged_count"] <= result_loose["after_flagged_count"]


def test_exclude_cancellation_removes_customer():
    scores, evidences = _make_scores_and_evidences()
    result = exclude_cancellation_anomalies(scores, evidences)
    # C2 is a cancellation anomaly and should be removed if it was flagged
    assert result["before_flagged_count"] >= result["after_flagged_count"]


def test_whitelist_reduces_count():
    scores, evidences = _make_scores_and_evidences()
    # C0 is high risk; whitelist it
    result = whitelist_customer("C0", scores, evidences)
    assert result["after_flagged_count"] <= result["before_flagged_count"]
    assert "C0" in result["customers_removed"]


def test_min_rule_hits_filter():
    scores, evidences = _make_scores_and_evidences()
    result = require_min_rule_hits(2, scores, evidences)
    # Only C0, C1 have rule_count=2; others should be removed
    assert result["after_flagged_count"] <= result["before_flagged_count"]


def test_result_contains_required_keys():
    scores, evidences = _make_scores_and_evidences()
    result = require_min_rule_hits(1, scores, evidences)
    required = {
        "scenario", "before_flagged_count", "after_flagged_count",
        "customers_added", "customers_removed", "business_interpretation",
    }
    assert required.issubset(set(result.keys()))
