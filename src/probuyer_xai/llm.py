"""LLM wrapper: DeepSeek via raw requests, with deterministic mock fallback."""

from __future__ import annotations

import json
from typing import Any

import requests

from probuyer_xai.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

_SYSTEM_PROMPT = """You are a retail risk analyst assistant. You explain customer risk evidence
to business reviewers in clear, non-accusatory business language.

Rules you must follow:
- Only reference values from the provided JSON evidence. Never invent numbers.
- Do not accuse or confirm guilt. Use language like "flagged for review" or "warrants attention."
- Use "probuyer-like" or "wholesale-like" — never "confirmed daigou."
- Always acknowledge uncertainty. This is a flag for review, not a verdict.
- Keep responses concise (3–5 sentences unless asked for more).
- Use plain business English. Avoid jargon."""


def _mock_explanation(evidence: dict[str, Any], task: str = "customer") -> str:
    """Generate deterministic mock explanation when LLM is unavailable."""
    cid = evidence.get("customer_id", "unknown")
    band = evidence.get("risk_band", "Unknown")
    pct = evidence.get("risk_percentile", 0)
    atype = evidence.get("anomaly_type", "unknown").replace("_", " ")
    reasons = evidence.get("top_reasons", [])
    action = evidence.get("recommended_action", "")
    metrics = evidence.get("key_metrics", {})

    if task == "whatif":
        before = evidence.get("before_flagged_count", "?")
        after = evidence.get("after_flagged_count", "?")
        scenario = evidence.get("scenario", "policy change")
        return (
            f"[Mock LLM] Under the scenario '{scenario}', the flagged customer count "
            f"changes from {before} to {after}. "
            "This reflects a shift in how strictly the detection criteria are applied. "
            "Review the added and removed customers to assess whether the change aligns "
            "with business risk appetite."
        )

    if task == "summary":
        return (
            f"[Mock LLM] This case study concerns Customer {cid}, who exhibits "
            f"{atype} behaviour ({band} risk, {pct:.1f}th percentile). "
            f"Key signals include: {'; '.join(reasons[:2]) if reasons else 'elevated anomaly score'}. "
            f"{action}"
        )

    return (
        f"[Mock LLM] Customer {cid} is flagged at {band} risk "
        f"(anomaly percentile: {pct:.1f}). "
        f"The detected pattern is consistent with {atype} behaviour. "
        f"Key signals: {'; '.join(reasons[:2]) if reasons else 'elevated anomaly score relative to the customer population'}. "
        f"Total quantity purchased: {metrics.get('total_quantity', 'N/A'):,.0f}, "
        f"spend: {metrics.get('total_spend', 'N/A'):,.2f}. "
        f"{action}"
    )


def _call_llm(prompt: str) -> str:
    """Call DeepSeek via raw requests. Returns text or raises."""
    url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 400,
        "temperature": 0.3,
    }
    resp = requests.post(url, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _has_key() -> bool:
    return bool(DEEPSEEK_API_KEY and DEEPSEEK_API_KEY.strip())


def explain_customer(evidence: dict[str, Any]) -> str:
    """Generate a plain-English explanation of why a customer is flagged."""
    if not _has_key():
        return _mock_explanation(evidence, task="customer")

    prompt = (
        "Explain to a business reviewer why this retail customer has been flagged "
        "for probuyer-like risk. Use the evidence below only.\n\n"
        f"Evidence:\n{json.dumps(evidence, indent=2)}"
    )
    try:
        return _call_llm(prompt)
    except Exception as exc:
        return _mock_explanation(evidence, task="customer") + f"\n[LLM unavailable: {exc}]"


def summarise_case(evidence: dict[str, Any]) -> str:
    """Summarise a case study for the reports."""
    if not _has_key():
        return _mock_explanation(evidence, task="summary")

    prompt = (
        "Write a short case study summary (3–5 sentences) for a portfolio audience. "
        "Describe the customer's buying pattern, the risk signals, and the recommended action. "
        "Use business-friendly language. Do not accuse — say 'warrants review'.\n\n"
        f"Evidence:\n{json.dumps(evidence, indent=2)}"
    )
    try:
        return _call_llm(prompt)
    except Exception as exc:
        return _mock_explanation(evidence, task="summary") + f"\n[LLM unavailable: {exc}]"


def explain_whatif(whatif_result: dict[str, Any]) -> str:
    """Explain the business impact of a what-if scenario."""
    if not _has_key():
        return _mock_explanation(whatif_result, task="whatif")

    prompt = (
        "Explain the business impact of this what-if policy scenario to a retail risk manager. "
        "Comment on whether the change seems reasonable.\n\n"
        f"What-if result:\n{json.dumps(whatif_result, indent=2)}"
    )
    try:
        return _call_llm(prompt)
    except Exception as exc:
        return _mock_explanation(whatif_result, task="whatif") + f"\n[LLM unavailable: {exc}]"


def monthly_portfolio_summary(evidences: list[dict[str, Any]]) -> str:
    """Generate a monthly portfolio-style summary for all flagged customers."""
    high = [e for e in evidences if e.get("risk_band") == "High"]
    med = [e for e in evidences if e.get("risk_band") == "Medium"]

    summary_input = {
        "total_customers_analysed": len(evidences),
        "high_risk_count": len(high),
        "medium_risk_count": len(med),
        "anomaly_type_breakdown": {},
    }
    for e in evidences:
        at = e.get("anomaly_type", "unknown")
        summary_input["anomaly_type_breakdown"][at] = (
            summary_input["anomaly_type_breakdown"].get(at, 0) + 1
        )

    if not _has_key():
        return (
            f"[Mock LLM] Monthly portfolio summary: {len(high)} high-risk and "
            f"{len(med)} medium-risk customers were identified out of "
            f"{len(evidences)} analysed. "
            "The most common pattern is broad wholesale-like buying. "
            "Review high-risk customers before the next rebate cycle."
        )

    prompt = (
        "Write a concise monthly portfolio-style summary for a retail risk team. "
        "Cover total flagged, breakdown by risk band and anomaly type, and a "
        "recommended priority for business review.\n\n"
        f"Summary input:\n{json.dumps(summary_input, indent=2)}"
    )
    try:
        return _call_llm(prompt)
    except Exception as exc:
        return (
            f"[Mock LLM] Monthly summary unavailable. {len(high)} high-risk, "
            f"{len(med)} medium-risk customers flagged. [LLM unavailable: {exc}]"
        )
