"""Deterministic what-if scenario analysis.

All functions operate on in-memory DataFrames.
The LLM only explains the output; it never computes it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from probuyer_xai.config import RISK_HIGH_PCT, RISK_MED_PCT


def _flagged_ids(
    scores: pd.DataFrame,
    evidences: list[dict],
    min_rule_hits: int = 0,
    exclude_cancellation: bool = False,
    whitelist: set[str] | None = None,
    high_pct: float = RISK_HIGH_PCT,
    med_pct: float = RISK_MED_PCT,
) -> set[str]:
    """Return the set of flagged customer IDs under given parameters."""
    ev_map = {e["customer_id"]: e for e in evidences}

    flagged = set()
    for _, row in scores.iterrows():
        cid = str(row["customer_id"])
        pct = float(row["risk_percentile"])

        if pct < med_pct:
            continue
        if whitelist and cid in whitelist:
            continue

        ev = ev_map.get(cid, {})
        if exclude_cancellation and ev.get("anomaly_type") == "return_or_cancellation_anomaly":
            continue
        if ev.get("rule_count", 0) < min_rule_hits:
            continue

        flagged.add(cid)

    return flagged


def _base_flagged(scores: pd.DataFrame, evidences: list[dict]) -> set[str]:
    return _flagged_ids(scores, evidences)


def _result(
    scenario: str,
    before: set[str],
    after: set[str],
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "before_flagged_count": len(before),
        "after_flagged_count": len(after),
        "customers_added": sorted(after - before),
        "customers_removed": sorted(before - after),
        "business_interpretation": (
            f"Changing to '{scenario}' {'reduces' if len(after) < len(before) else 'increases'} "
            f"the flagged customer count from {len(before)} to {len(after)}."
        ),
    }


def change_risk_threshold(
    scores: pd.DataFrame,
    evidences: list[dict],
    new_high_pct: float,
    new_med_pct: float | None = None,
) -> dict[str, Any]:
    if new_med_pct is None:
        new_med_pct = new_high_pct - 2.0
    before = _base_flagged(scores, evidences)
    after = _flagged_ids(scores, evidences, high_pct=new_high_pct, med_pct=new_med_pct)
    return _result(
        f"Risk threshold changed to high>={new_high_pct}%, medium>={new_med_pct}%",
        before,
        after,
    )


def change_bulk_threshold(
    scores: pd.DataFrame,
    evidences: list[dict],
    features: pd.DataFrame,
    new_pct: float,
) -> dict[str, Any]:
    """Recompute bulk_invoice_ratio at a different percentile and re-flag."""
    inv_col = "max_quantity_per_invoice"
    new_threshold = float(np.percentile(features[inv_col].dropna().values, new_pct))

    # Re-derive bulk_invoice_ratio (approximate: flag customers whose
    # max invoice qty >= new threshold as having a bulk-like profile)
    adj_features = features.copy()
    adj_features["bulk_invoice_ratio"] = (
        features["max_quantity_per_invoice"] >= new_threshold
    ).astype(float)

    before = _base_flagged(scores, evidences)
    # Simple proxy: customers with max_qty >= new_threshold are now "bulk"
    adj_high = adj_features[adj_features["bulk_invoice_ratio"] >= 0.5].index
    after = before | set(str(cid) for cid in adj_high if str(cid) in
                          {e["customer_id"] for e in evidences if e.get("risk_band") in ("High", "Medium")})
    after = {str(cid) for cid in adj_high} & {e["customer_id"] for e in evidences if
                                                 float(scores[scores["customer_id"] == cid]["risk_percentile"].iloc[0])
                                                 >= RISK_MED_PCT
                                                 if len(scores[scores["customer_id"] == cid]) > 0}
    return _result(f"Bulk threshold changed to p{new_pct:.0f}", before, after | before)


def exclude_cancellation_anomalies(
    scores: pd.DataFrame,
    evidences: list[dict],
) -> dict[str, Any]:
    before = _base_flagged(scores, evidences)
    after = _flagged_ids(scores, evidences, exclude_cancellation=True)
    return _result("Exclude cancellation/return anomalies", before, after)


def whitelist_customer(
    customer_id: str,
    scores: pd.DataFrame,
    evidences: list[dict],
) -> dict[str, Any]:
    before = _base_flagged(scores, evidences)
    after = _flagged_ids(scores, evidences, whitelist={str(customer_id)})
    return _result(f"Whitelist customer {customer_id}", before, after)


def require_min_rule_hits(
    n: int,
    scores: pd.DataFrame,
    evidences: list[dict],
) -> dict[str, Any]:
    before = _base_flagged(scores, evidences)
    after = _flagged_ids(scores, evidences, min_rule_hits=n)
    return _result(f"Require at least {n} rule hit(s)", before, after)
