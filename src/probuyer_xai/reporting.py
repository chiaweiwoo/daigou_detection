"""Generate case studies and LLM example reports."""

from __future__ import annotations

from typing import Any

from probuyer_xai.config import REPORT_CASES, REPORT_LLM, REPORTS_DIR
from probuyer_xai.llm import summarise_case


def _pick_cases(evidences: list[dict]) -> list[dict]:
    """Pick 3 contrasting customers for case studies."""
    # Case 1: highest-confidence broad wholesale buyer
    broad = [
        e for e in evidences
        if e["anomaly_type"] == "broad_wholesale_buyer" and e["confidence"] == "High"
    ]
    if not broad:
        broad = [e for e in evidences if e["risk_band"] == "High"]
    case1 = max(broad, key=lambda e: e["risk_percentile"]) if broad else None

    # Case 2: single-product bulk buyer
    single = [e for e in evidences if e["anomaly_type"] == "single_product_bulk_buyer"]
    if not single:
        single = [e for e in evidences if e["risk_band"] in ("High", "Medium")
                  and e.get("key_metrics", {}).get("top_sku_quantity_share", 0) >= 0.6]
    case2 = max(single, key=lambda e: e["risk_percentile"]) if single else None

    # Case 3: high anomaly but lower probuyer confidence (cancellation/return)
    cancel = [e for e in evidences if e["anomaly_type"] == "return_or_cancellation_anomaly"]
    if not cancel:
        cancel = [e for e in evidences
                  if e["risk_band"] in ("High", "Medium") and e["confidence"] == "Low"]
    case3 = max(cancel, key=lambda e: e["risk_percentile"]) if cancel else None

    # Fallback: just pick top anomalies if cases are missing
    by_score = sorted(evidences, key=lambda e: e["risk_percentile"], reverse=True)
    chosen = []
    used_ids = set()
    for c in [case1, case2, case3]:
        if c and c["customer_id"] not in used_ids:
            chosen.append(c)
            used_ids.add(c["customer_id"])
    for e in by_score:
        if len(chosen) >= 3:
            break
        if e["customer_id"] not in used_ids:
            chosen.append(e)
            used_ids.add(e["customer_id"])
    return chosen[:3]


def _fmt_case(case: dict, idx: int, llm_text: str) -> str:
    m = case.get("key_metrics", {})
    hits = ", ".join(case.get("rule_hits", [])) or "None"
    lines = [
        f"## Case {idx}: {case['anomaly_type'].replace('_', ' ').title()}",
        "",
        f"**Customer ID:** {case['customer_id']}",
        f"**Risk Band:** {case['risk_band']} | **Percentile:** {case['risk_percentile']:.1f}",
        f"**Confidence:** {case['confidence']}",
        f"**Anomaly Type:** {case['anomaly_type']}",
        "",
        "### Key metrics",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total quantity | {m.get('total_quantity', 'N/A'):,.0f} |",
        f"| Total spend | {m.get('total_spend', 'N/A'):,.2f} |",
        f"| Invoices | {m.get('num_invoices', 'N/A')} |",
        f"| Max invoice qty | {m.get('max_quantity_per_invoice', 'N/A'):,.0f} |",
        f"| Bulk invoice ratio | {m.get('bulk_invoice_ratio', 'N/A'):.2%} |",
        f"| Top SKU qty share | {m.get('top_sku_quantity_share', 'N/A'):.2%} |",
        f"| Cancellation ratio | {m.get('cancellation_ratio', 'N/A'):.2%} |",
        "",
        f"**Rule hits:** {hits}",
        "",
        "### Business reasons",
    ]
    for r in case.get("top_reasons", []):
        lines.append(f"- {r}")
    lines += [
        "",
        f"**Recommended action:** {case.get('recommended_action', '')}",
        "",
        "### LLM explanation",
        llm_text,
        "",
    ]
    return "\n".join(lines)


def generate_case_studies(evidences: list[dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cases = _pick_cases(evidences)

    case_blocks = []
    llm_blocks = []

    for i, case in enumerate(cases, start=1):
        llm_text = summarise_case(case)
        case_blocks.append(_fmt_case(case, i, llm_text))
        llm_blocks.append(
            f"### Case {i} — Customer {case['customer_id']}\n\n{llm_text}\n"
        )

    report_text = "# Case Studies\n\n" + "\n---\n\n".join(case_blocks)
    REPORT_CASES.write_text(report_text, encoding="utf-8")
    print(f"Case studies -> {REPORT_CASES}")

    llm_text = (
        "# LLM Explanation Examples\n\n"
        "These are examples of LLM-generated explanations from structured DS evidence.\n\n"
        + "\n---\n\n".join(llm_blocks)
    )
    REPORT_LLM.write_text(llm_text, encoding="utf-8")
    print(f"LLM examples -> {REPORT_LLM}")
