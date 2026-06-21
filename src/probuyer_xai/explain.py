"""Deterministic structured explanation evidence per customer.

The LLM is NOT involved here. This layer converts DS output into
structured JSON that the LLM can then narrate.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


# --- Anomaly type classification ------------------------------------------

def _classify_anomaly_type(row: pd.Series) -> str:
    """Map customer feature values to an anomaly type label."""
    high_qty = row.get("total_quantity", 0) > row.get("_p90_total_quantity", float("inf"))
    high_spend = row.get("total_spend", 0) > row.get("_p90_total_spend", float("inf"))
    high_sku_conc = row.get("top_sku_quantity_share", 0) >= 0.7
    high_cancel = row.get("cancellation_ratio", 0) > row.get("_p90_cancellation_ratio", float("inf"))
    high_max_inv_qty = row.get("max_quantity_per_invoice", 0) > row.get("_p90_max_quantity_per_invoice", float("inf"))
    high_freq = row.get("num_invoices", 0) > row.get("_p90_num_invoices", float("inf"))

    if high_cancel:
        return "return_or_cancellation_anomaly"
    if high_sku_conc and (high_qty or high_max_inv_qty):
        return "single_product_bulk_buyer"
    if high_qty and high_spend and not high_sku_conc:
        return "broad_wholesale_buyer"
    if high_freq:
        return "high_frequency_buyer"
    if high_max_inv_qty or high_sku_conc:
        return "potential_stockout_risk"
    return "broad_wholesale_buyer"


def _confidence(risk_band: str, rule_count: int) -> str:
    if risk_band == "High" and rule_count >= 2:
        return "High"
    if risk_band in ("High", "Medium") and rule_count >= 1:
        return "Medium"
    return "Low"


def _top_reasons(row: pd.Series, rule_reasons: list[str]) -> list[str]:
    reasons = list(rule_reasons) if rule_reasons else []
    # Add a generic reason if none from rules
    if not reasons:
        reasons.append("Anomaly score is high relative to the customer population.")
    return reasons[:5]


def _recommended_action(risk_band: str, anomaly_type: str) -> str:
    if risk_band == "High":
        return (
            "Flag for business review. Verify customer intent before granting rebate "
            "or priority-stock access."
        )
    if risk_band == "Medium":
        return "Monitor purchase frequency and quantity. Review if pattern escalates."
    return "No immediate action required. Continue standard monitoring."


# --- Main evidence builder ------------------------------------------------

def build_evidence(
    features: pd.DataFrame,
    scores: pd.DataFrame,
    rule_hits: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build structured explanation evidence for all customers.

    Returns a list of evidence dicts (one per customer).
    """
    # Compute p90 thresholds for anomaly type classification
    p90_cols = [
        "total_quantity",
        "total_spend",
        "top_sku_quantity_share",
        "cancellation_ratio",
        "max_quantity_per_invoice",
        "num_invoices",
    ]
    p90 = {col: float(features[col].quantile(0.9)) for col in p90_cols if col in features.columns}

    # Merge all data
    merged = (
        features.reset_index()
        .rename(columns={"index": "customer_id"})
        .merge(scores[["customer_id", "anomaly_score", "risk_percentile", "risk_band"]], on="customer_id", how="left")
        .merge(rule_hits[["customer_id", "rule_hits", "rule_reasons", "rule_count"]], on="customer_id", how="left")
    )
    merged["rule_hits"] = merged["rule_hits"].apply(lambda x: x if isinstance(x, list) else [])
    merged["rule_reasons"] = merged["rule_reasons"].apply(lambda x: x if isinstance(x, list) else [])
    merged["rule_count"] = merged["rule_count"].fillna(0).astype(int)

    # Attach p90 columns for anomaly type logic
    for col, val in p90.items():
        merged[f"_p90_{col}"] = val

    evidences = []
    for _, row in merged.iterrows():
        anomaly_type = _classify_anomaly_type(row)
        confidence = _confidence(row["risk_band"], row["rule_count"])
        evidence = {
            "customer_id": str(row["customer_id"]),
            "risk_band": row["risk_band"],
            "risk_percentile": round(float(row["risk_percentile"]), 2),
            "anomaly_score": round(float(row["anomaly_score"]), 4),
            "anomaly_type": anomaly_type,
            "confidence": confidence,
            "top_reasons": _top_reasons(row, row["rule_reasons"]),
            "rule_hits": row["rule_hits"],
            "rule_count": int(row["rule_count"]),
            "key_metrics": {
                "total_quantity": float(row.get("total_quantity", 0)),
                "total_spend": round(float(row.get("total_spend", 0)), 2),
                "num_invoices": int(row.get("num_invoices", 0)),
                "max_quantity_per_invoice": float(row.get("max_quantity_per_invoice", 0)),
                "bulk_invoice_ratio": round(float(row.get("bulk_invoice_ratio", 0)), 4),
                "top_sku_quantity_share": round(float(row.get("top_sku_quantity_share", 0)), 4),
                "cancellation_ratio": round(float(row.get("cancellation_ratio", 0)), 4),
            },
            "recommended_action": _recommended_action(row["risk_band"], anomaly_type),
        }
        evidences.append(evidence)

    return evidences


def evidence_for_customer(
    customer_id: str,
    all_evidence: list[dict],
) -> dict[str, Any] | None:
    for ev in all_evidence:
        if ev["customer_id"] == str(customer_id):
            return ev
    return None
