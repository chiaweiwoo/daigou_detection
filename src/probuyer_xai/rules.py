"""Business rule engine: load rules JSON, compute thresholds, apply to customers."""

from __future__ import annotations

import json
import operator
from pathlib import Path

import numpy as np
import pandas as pd

from probuyer_xai.config import PROBUYER_RULES, CUSTOMER_RULE_HITS, DATA_PROCESSED

_OPERATORS = {
    ">=": operator.ge,
    ">": operator.gt,
    "<=": operator.le,
    "<": operator.lt,
    "==": operator.eq,
}

_PCT_MAP = {
    "p99": 99,
    "p98": 98,
    "p97": 97,
    "p95": 95,
    "p90": 90,
}


def load_rules(path: Path = PROBUYER_RULES) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_rules(
    features: pd.DataFrame,
    rules_doc: dict,
) -> pd.DataFrame:
    """Evaluate each rule against customer features.

    Returns DataFrame with columns:
        customer_id, rule_hits (list of rule_ids), rule_count,
        rule_reasons (list of reason strings).
    """
    rules = rules_doc["rules"]
    hit_records: dict[str, list] = {cid: [] for cid in features.index}
    reason_records: dict[str, list] = {cid: [] for cid in features.index}

    for rule in rules:
        feature = rule["feature"]
        op_fn = _OPERATORS[rule["operator"]]
        pct = _PCT_MAP[rule["threshold_source"]]
        threshold = float(np.percentile(features[feature].dropna().values, pct))

        hits = features[op_fn(features[feature], threshold)].index
        for cid in hits:
            hit_records[cid].append(rule["rule_id"])
            reason_records[cid].append(rule["risk_reason"])

    result = pd.DataFrame(
        {
            "customer_id": list(hit_records.keys()),
            "rule_hits": list(hit_records.values()),
            "rule_reasons": list(reason_records.values()),
        }
    )
    result["rule_count"] = result["rule_hits"].apply(len)
    return result


def save_rule_hits(rule_hits: pd.DataFrame) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    rule_hits.to_parquet(CUSTOMER_RULE_HITS, index=False)
    n_flagged = (rule_hits["rule_count"] > 0).sum()
    print(f"Rule hits saved -> {CUSTOMER_RULE_HITS}")
    print(f"Customers with at least 1 rule hit: {n_flagged:,}")


def load_rule_hits() -> pd.DataFrame:
    return pd.read_parquet(CUSTOMER_RULE_HITS)
