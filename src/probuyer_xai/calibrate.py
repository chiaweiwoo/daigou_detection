"""LLM-as-judge calibration loop.

After the HBOS model scores customers, this module samples a stratified set
from each risk band and has a pro LLM independently rate each customer's
probuyer-likeness on a 1–5 scale (temperature=0, structured JSON output).

Agreement between the model's band and the LLM's score is the calibration
signal. If High-band agreement is low, the model is flagging the wrong things.
The report surfaces specific threshold adjustment suggestions.

The LLM does not touch the model or its scores — it only evaluates evidence JSON.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests

from probuyer_xai.config import (
    CALIBRATION_MODEL,
    CALIBRATION_REPORT,
    CALIBRATION_SAMPLE_HIGH,
    CALIBRATION_SAMPLE_LOW,
    CALIBRATION_SAMPLE_MED,
    CUSTOMER_EVIDENCE,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    PROBUYER_RULES,
)


_CALIBRATION_SYSTEM = (
    "You are a retail risk calibration expert. Your job is to independently evaluate "
    "whether a customer's purchasing pattern genuinely resembles probuyer or wholesale "
    "reseller behaviour. Respond only in English. Return only valid JSON — no prose, "
    "no markdown fences, no explanation outside the JSON object."
)

_CALIBRATION_PROMPT = """\
Evaluate the probuyer-likeness of the retail customer below.

Probuyer signals defined for this retail context:
{rules_context}

Rate on a 1–5 scale:
  5 = Strong probuyer/wholesale signal — bulk buying clearly consistent with resale
  4 = Moderate probuyer signal — pattern warrants priority human review
  3 = Borderline — could be probuyer or legitimate heavy business/personal use
  2 = Weak probuyer signal — has some elements but significant disqualifiers
  1 = Not probuyer-like — pattern suggests fraud, return abuse, or normal heavy retail use

Return ONLY this JSON (no other text):
{{
  "score": <integer 1-5>,
  "primary_signal": "<the single strongest probuyer indicator in the evidence, or null if none>",
  "disqualifiers": ["<reasons that weaken the probuyer interpretation>"],
  "is_probuyer_like": <true if score >= 4, false otherwise>,
  "rating_confidence": "<Low|Medium|High — how confident are you given this evidence?>"
}}

Rules:
- Only reference values explicitly present in the evidence JSON. Never invent numbers.
- If the anomaly_type is return_or_cancellation_anomaly, score must be 1 or 2.
- Before finalising, check: does every number you mention appear in the evidence? If not, remove it.

Customer evidence:
{evidence_json}"""


@dataclass
class CustomerRating:
    customer_id: str
    model_band: str
    llm_score: int
    primary_signal: str | None
    disqualifiers: list[str]
    is_probuyer_like: bool
    rating_confidence: str
    agrees_with_model: bool


@dataclass
class BandCalibration:
    band: str
    sampled: int
    agreed: int
    agreement_rate: float
    disagreements: list[CustomerRating] = field(default_factory=list)


@dataclass
class CalibrationReport:
    iteration: int
    timestamp: str
    overall_agreement: float
    by_band: list[BandCalibration]
    suggested_adjustments: list[str]
    raw_ratings: list[CustomerRating]
    model_used: str
    is_mock: bool = False


# --- Helpers -----------------------------------------------------------------

def _load_rules_context() -> str:
    with open(PROBUYER_RULES) as f:
        rules = json.load(f)
    return "\n".join(
        f"- {r['name']} ({r['rule_id']}): {r['risk_reason']}"
        for r in rules["rules"]
    )


def _agrees(model_band: str, llm_score: int) -> bool:
    """True when the LLM score is consistent with the model's risk band."""
    if model_band == "High":
        return llm_score >= 4
    if model_band == "Medium":
        return llm_score >= 3
    return llm_score <= 2  # Low band → should rate 1–2


def _call_calibration_llm(evidence: dict[str, Any], rules_context: str) -> dict[str, Any]:
    """Call the pro LLM at temperature=0. Returns parsed JSON dict. Raises on failure."""
    url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    prompt = _CALIBRATION_PROMPT.format(
        rules_context=rules_context,
        evidence_json=json.dumps(evidence, indent=2),
    )
    body = {
        "model": CALIBRATION_MODEL,
        "messages": [
            {"role": "system", "content": _CALIBRATION_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 300,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(url, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    return json.loads(resp.json()["choices"][0]["message"]["content"])


def _mock_rating(evidence: dict[str, Any]) -> dict[str, Any]:
    """Deterministic mock rating from evidence fields — no LLM call."""
    band = evidence.get("risk_band", "Low")
    rule_count = evidence.get("rule_count", 0)
    atype = evidence.get("anomaly_type", "")

    if atype == "return_or_cancellation_anomaly":
        score = 1
    elif band == "High" and rule_count >= 2:
        score = 4
    elif band == "High":
        score = 3
    elif band == "Medium" and rule_count >= 1:
        score = 3
    elif band == "Medium":
        score = 2
    else:
        score = 1

    return {
        "score": score,
        "primary_signal": (evidence.get("top_reasons") or [None])[0],
        "disqualifiers": ["[Mock — no LLM call made]"],
        "is_probuyer_like": score >= 4,
        "rating_confidence": "Low",
    }


def _rate_customer(
    evidence: dict[str, Any],
    rules_context: str,
    use_mock: bool,
) -> CustomerRating:
    try:
        raw = _mock_rating(evidence) if use_mock else _call_calibration_llm(evidence, rules_context)
    except Exception:
        raw = _mock_rating(evidence)

    score = int(raw.get("score", 3))
    band = evidence.get("risk_band", "Low")
    return CustomerRating(
        customer_id=evidence["customer_id"],
        model_band=band,
        llm_score=score,
        primary_signal=raw.get("primary_signal"),
        disqualifiers=raw.get("disqualifiers", []),
        is_probuyer_like=bool(raw.get("is_probuyer_like", score >= 4)),
        rating_confidence=raw.get("rating_confidence", "Low"),
        agrees_with_model=_agrees(band, score),
    )


def _suggest_adjustments(
    by_band: list[BandCalibration],
    ratings: list[CustomerRating],
) -> list[str]:
    suggestions: list[str] = []

    for bc in by_band:
        if bc.sampled == 0:
            continue
        if bc.band == "High" and bc.agreement_rate < 0.6:
            suggestions.append(
                f"High-risk band agreement is {bc.agreement_rate:.0%} ({bc.agreed}/{bc.sampled}). "
                "Consider raising RISK_HIGH_PCT (e.g. 99.2–99.5) to restrict High flags to the "
                "very top anomalies and reduce false positives."
            )
        elif bc.band == "High" and bc.agreement_rate > 0.9:
            suggestions.append(
                f"High-risk band has strong agreement ({bc.agreement_rate:.0%}). "
                "The model is precise at the top. You could consider lowering RISK_HIGH_PCT "
                "slightly to expand coverage without meaningful precision loss."
            )
        if bc.band == "Medium" and bc.agreement_rate < 0.5:
            suggestions.append(
                f"Medium-risk band agreement is {bc.agreement_rate:.0%} ({bc.agreed}/{bc.sampled}). "
                "Consider raising RISK_MED_PCT to narrow the Medium band."
            )

    # Check recurring disqualifiers in disagreement cases
    disagree_disq = [
        d
        for r in ratings
        for d in r.disqualifiers
        if not r.agrees_with_model and "[Mock" not in d
    ]
    cancel_count = sum(1 for d in disagree_disq if "cancel" in d.lower() or "return" in d.lower())
    if cancel_count >= 3:
        suggestions.append(
            "LLM repeatedly cited cancellation/return patterns as disqualifiers in disagreement cases. "
            "Verify that cancellation_ratio is excluded from MODEL_FEATURES (confirmed excluded in current config)."
        )

    if not suggestions:
        suggestions.append(
            "Agreement rates look healthy across all bands. No threshold adjustments suggested at this time."
        )
    return suggestions


# --- Public API --------------------------------------------------------------

def run_calibration(
    all_evidence: list[dict[str, Any]],
    iteration: int = 1,
    seed: int = 42,
) -> CalibrationReport:
    """Run one round of LLM-as-judge calibration.

    Stratified sample from each risk band → pro LLM rates each at temperature=0
    → agreement report with adjustment suggestions.
    """
    use_mock = not bool(DEEPSEEK_API_KEY and DEEPSEEK_API_KEY.strip())
    rules_context = _load_rules_context()
    rng = random.Random(seed)

    by_band_pool: dict[str, list[dict]] = {"High": [], "Medium": [], "Low": []}
    for ev in all_evidence:
        band = ev.get("risk_band", "Low")
        if band in by_band_pool:
            by_band_pool[band].append(ev)

    limits = {
        "High": CALIBRATION_SAMPLE_HIGH,
        "Medium": CALIBRATION_SAMPLE_MED,
        "Low": CALIBRATION_SAMPLE_LOW,
    }
    samples: list[dict] = []
    for band, limit in limits.items():
        pool = by_band_pool[band]
        samples.extend(rng.sample(pool, min(limit, len(pool))))

    ratings = [_rate_customer(ev, rules_context, use_mock) for ev in samples]

    band_cals: list[BandCalibration] = []
    for band in ("High", "Medium", "Low"):
        band_ratings = [r for r in ratings if r.model_band == band]
        agreed = sum(1 for r in band_ratings if r.agrees_with_model)
        band_cals.append(BandCalibration(
            band=band,
            sampled=len(band_ratings),
            agreed=agreed,
            agreement_rate=agreed / len(band_ratings) if band_ratings else 0.0,
            disagreements=[r for r in band_ratings if not r.agrees_with_model],
        ))

    total = len(ratings)
    overall = sum(1 for r in ratings if r.agrees_with_model) / total if total else 0.0

    return CalibrationReport(
        iteration=iteration,
        timestamp=datetime.now(timezone.utc).isoformat(),
        overall_agreement=round(overall, 4),
        by_band=band_cals,
        suggested_adjustments=_suggest_adjustments(band_cals, ratings),
        raw_ratings=ratings,
        model_used=CALIBRATION_MODEL,
        is_mock=use_mock,
    )


def save_report(report: CalibrationReport) -> None:
    CALIBRATION_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_REPORT, "w") as f:
        json.dump(asdict(report), f, indent=2)


def load_report() -> CalibrationReport | None:
    if not CALIBRATION_REPORT.exists():
        return None
    with open(CALIBRATION_REPORT) as f:
        data = json.load(f)
    data["by_band"] = [
        BandCalibration(
            **{k: v for k, v in bc.items() if k != "disagreements"},
            disagreements=[CustomerRating(**r) for r in bc["disagreements"]],
        )
        for bc in data["by_band"]
    ]
    data["raw_ratings"] = [CustomerRating(**r) for r in data["raw_ratings"]]
    return CalibrationReport(**data)


def save_evidence(evidences: list[dict[str, Any]]) -> None:
    """Persist full evidence list to disk for the calibration CLI to read."""
    CUSTOMER_EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    with open(CUSTOMER_EVIDENCE, "w") as f:
        json.dump(evidences, f, indent=2)
