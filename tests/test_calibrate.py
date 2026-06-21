"""Tests for calibrate.py — all synthetic, no LLM calls, no external data."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from probuyer_xai.calibrate import (
    BandCalibration,
    CalibrationReport,
    CustomerRating,
    _agrees,
    _mock_rating,
    _rate_customer,
    _suggest_adjustments,
    load_report,
    run_calibration,
    save_report,
)


# --- Fixtures ----------------------------------------------------------------

def _ev(customer_id: str, risk_band: str, rule_count: int = 1, atype: str = "broad_wholesale_buyer") -> dict:
    return {
        "customer_id": customer_id,
        "risk_band": risk_band,
        "risk_percentile": 99.5 if risk_band == "High" else 97.0 if risk_band == "Medium" else 50.0,
        "anomaly_score": 0.9 if risk_band == "High" else 0.5 if risk_band == "Medium" else 0.1,
        "anomaly_type": atype,
        "confidence": "High" if risk_band == "High" else "Medium",
        "top_reasons": ["High total quantity"],
        "rule_hits": ["R001"] if rule_count >= 1 else [],
        "rule_count": rule_count,
        "key_metrics": {
            "total_quantity": 50000.0,
            "total_spend": 25000.0,
            "num_invoices": 20,
            "max_quantity_per_invoice": 5000.0,
            "bulk_invoice_ratio": 0.8,
            "top_sku_quantity_share": 0.4,
            "cancellation_ratio": 0.05,
        },
        "recommended_action": "Flag for review.",
    }


SAMPLE_EVIDENCES = (
    [_ev(str(i), "High", rule_count=2) for i in range(20)]
    + [_ev(str(i + 20), "Medium", rule_count=1) for i in range(15)]
    + [_ev(str(i + 35), "Low", rule_count=0) for i in range(10)]
)


# --- _agrees -----------------------------------------------------------------

def test_agrees_high_band_needs_score_4():
    assert _agrees("High", 4) is True
    assert _agrees("High", 5) is True
    assert _agrees("High", 3) is False


def test_agrees_medium_band_needs_score_3():
    assert _agrees("Medium", 3) is True
    assert _agrees("Medium", 4) is True
    assert _agrees("Medium", 2) is False


def test_agrees_low_band_needs_score_le_2():
    assert _agrees("Low", 1) is True
    assert _agrees("Low", 2) is True
    assert _agrees("Low", 3) is False


# --- _mock_rating ------------------------------------------------------------

def test_mock_rating_cancellation_scores_1():
    ev = _ev("99", "High", rule_count=2, atype="return_or_cancellation_anomaly")
    result = _mock_rating(ev)
    assert result["score"] == 1
    assert result["is_probuyer_like"] is False


def test_mock_rating_high_band_multiple_rules_scores_4():
    ev = _ev("1", "High", rule_count=2)
    result = _mock_rating(ev)
    assert result["score"] == 4
    assert result["is_probuyer_like"] is True


def test_mock_rating_low_band_scores_1():
    ev = _ev("50", "Low", rule_count=0)
    result = _mock_rating(ev)
    assert result["score"] == 1


def test_mock_rating_returns_required_keys():
    ev = _ev("1", "High", rule_count=2)
    result = _mock_rating(ev)
    for key in ("score", "primary_signal", "disqualifiers", "is_probuyer_like", "rating_confidence"):
        assert key in result


# --- _rate_customer ----------------------------------------------------------

def test_rate_customer_mock_returns_customer_rating():
    ev = _ev("42", "High", rule_count=2)
    rating = _rate_customer(ev, rules_context="", use_mock=True)
    assert isinstance(rating, CustomerRating)
    assert rating.customer_id == "42"
    assert rating.model_band == "High"
    assert 1 <= rating.llm_score <= 5


def test_rate_customer_cancellation_disagrees_with_high():
    ev = _ev("77", "High", rule_count=2, atype="return_or_cancellation_anomaly")
    rating = _rate_customer(ev, rules_context="", use_mock=True)
    assert rating.llm_score == 1
    assert rating.agrees_with_model is False


# --- run_calibration ---------------------------------------------------------

def test_run_calibration_mock_returns_report():
    with patch("probuyer_xai.calibrate.DEEPSEEK_API_KEY", ""):
        report = run_calibration(SAMPLE_EVIDENCES, iteration=1)
    assert isinstance(report, CalibrationReport)
    assert report.is_mock is True
    assert report.iteration == 1
    assert 0.0 <= report.overall_agreement <= 1.0


def test_run_calibration_has_all_bands():
    with patch("probuyer_xai.calibrate.DEEPSEEK_API_KEY", ""):
        report = run_calibration(SAMPLE_EVIDENCES, iteration=1)
    bands = {bc.band for bc in report.by_band}
    assert bands == {"High", "Medium", "Low"}


def test_run_calibration_samples_within_limits():
    with patch("probuyer_xai.calibrate.DEEPSEEK_API_KEY", ""):
        with patch("probuyer_xai.calibrate.CALIBRATION_SAMPLE_HIGH", 5):
            with patch("probuyer_xai.calibrate.CALIBRATION_SAMPLE_MED", 3):
                with patch("probuyer_xai.calibrate.CALIBRATION_SAMPLE_LOW", 2):
                    report = run_calibration(SAMPLE_EVIDENCES, iteration=1)
    for bc in report.by_band:
        limit = {"High": 5, "Medium": 3, "Low": 2}[bc.band]
        assert bc.sampled <= limit


def test_run_calibration_has_suggestions():
    with patch("probuyer_xai.calibrate.DEEPSEEK_API_KEY", ""):
        report = run_calibration(SAMPLE_EVIDENCES, iteration=1)
    assert len(report.suggested_adjustments) >= 1
    assert all(isinstance(s, str) for s in report.suggested_adjustments)


# --- _suggest_adjustments ----------------------------------------------------

def test_suggest_low_high_agreement_recommends_raising_threshold():
    bc_high = BandCalibration(band="High", sampled=10, agreed=4, agreement_rate=0.4, disagreements=[])
    bc_med = BandCalibration(band="Medium", sampled=10, agreed=7, agreement_rate=0.7, disagreements=[])
    bc_low = BandCalibration(band="Low", sampled=5, agreed=4, agreement_rate=0.8, disagreements=[])
    suggestions = _suggest_adjustments([bc_high, bc_med, bc_low], [])
    assert any("RISK_HIGH_PCT" in s for s in suggestions)


def test_suggest_high_agreement_no_threshold_warning():
    bc_high = BandCalibration(band="High", sampled=10, agreed=10, agreement_rate=1.0, disagreements=[])
    bc_med = BandCalibration(band="Medium", sampled=10, agreed=9, agreement_rate=0.9, disagreements=[])
    bc_low = BandCalibration(band="Low", sampled=5, agreed=5, agreement_rate=1.0, disagreements=[])
    suggestions = _suggest_adjustments([bc_high, bc_med, bc_low], [])
    assert not any("raising RISK_HIGH_PCT" in s for s in suggestions)


# --- save/load round-trip ----------------------------------------------------

def test_save_load_round_trip():
    with patch("probuyer_xai.calibrate.DEEPSEEK_API_KEY", ""):
        report = run_calibration(SAMPLE_EVIDENCES, iteration=2)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "calibration_report.json"
        with patch("probuyer_xai.calibrate.CALIBRATION_REPORT", path):
            save_report(report)
            loaded = load_report()

    assert loaded is not None
    assert loaded.iteration == report.iteration
    assert loaded.overall_agreement == report.overall_agreement
    assert len(loaded.by_band) == len(report.by_band)
    assert len(loaded.raw_ratings) == len(report.raw_ratings)
