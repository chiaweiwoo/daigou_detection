"""HBOS anomaly detection model: training, scoring, and persistence."""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from pyod.models.hbos import HBOS
from sklearn.preprocessing import RobustScaler

from probuyer_xai.config import (
    MODEL_FEATURES,
    RISK_HIGH_PCT,
    RISK_MED_PCT,
    HBOS_MODEL,
    FEATURE_SCALER,
    MODEL_METADATA,
    CUSTOMER_SCORES,
    MODELS_DIR,
    DATA_PROCESSED,
)


def _assign_risk_band(pct_values: np.ndarray) -> list[str]:
    """Assign risk bands from a numpy array of percentile values."""
    bands = []
    for p in pct_values:
        if p >= RISK_HIGH_PCT:
            bands.append("High")
        elif p >= RISK_MED_PCT:
            bands.append("Medium")
        else:
            bands.append("Low")
    return bands


def _compute_risk_percentile(raw_scores: np.ndarray) -> np.ndarray:
    """Percentile rank: fraction of scores <= each score, scaled to 0–100."""
    n = len(raw_scores)
    return np.array([float(np.sum(raw_scores <= s) / n * 100) for s in raw_scores])


def train(features: pd.DataFrame) -> tuple[HBOS, RobustScaler, pd.DataFrame]:
    """Train HBOS on feature matrix.

    Returns:
        model: fitted HBOS instance.
        scaler: fitted RobustScaler.
        scores: DataFrame with customer_id, anomaly_score, risk_percentile, risk_band.
    """
    X = features[MODEL_FEATURES].copy()

    # log1p transform (all model features are non-negative)
    X_log = np.log1p(X)

    # Robust scale
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_log)

    # Fit HBOS
    model = HBOS()
    model.fit(X_scaled)

    raw_scores = model.decision_scores_  # higher = more anomalous
    pct = _compute_risk_percentile(raw_scores)

    scores = pd.DataFrame(
        {
            "customer_id": features.index,
            "anomaly_score": raw_scores,
            "risk_percentile": pct,
            "risk_band": _assign_risk_band(pct),
        }
    )

    return model, scaler, scores


def score(
    features: pd.DataFrame,
    model: HBOS,
    scaler: RobustScaler,
) -> pd.DataFrame:
    """Score new customers using a pre-fitted model."""
    X = features[MODEL_FEATURES].copy()
    X_scaled = scaler.transform(np.log1p(X))
    raw_scores = model.decision_function(X_scaled)
    pct = _compute_risk_percentile(raw_scores)

    return pd.DataFrame(
        {
            "customer_id": features.index,
            "anomaly_score": raw_scores,
            "risk_percentile": pct,
            "risk_band": _assign_risk_band(pct),
        }
    )


def save_model(
    model: HBOS,
    scaler: RobustScaler,
    scores: pd.DataFrame,
    extra_meta: dict | None = None,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, HBOS_MODEL)
    joblib.dump(scaler, FEATURE_SCALER)
    scores.to_parquet(CUSTOMER_SCORES, index=False)

    meta = {
        "trained_at": datetime.utcnow().isoformat(),
        "model_class": "HBOS",
        "feature_list": MODEL_FEATURES,
        "risk_high_pct": RISK_HIGH_PCT,
        "risk_med_pct": RISK_MED_PCT,
        "n_customers": len(scores),
        "high_risk_count": int((scores["risk_band"] == "High").sum()),
        "medium_risk_count": int((scores["risk_band"] == "Medium").sum()),
        "contamination": "not set (unsupervised)",
    }
    if extra_meta:
        meta.update(extra_meta)

    MODEL_METADATA.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Model saved -> {HBOS_MODEL}")
    print(f"Scaler saved -> {FEATURE_SCALER}")
    print(f"Scores saved -> {CUSTOMER_SCORES}")
    print(f"Metadata saved -> {MODEL_METADATA}")


def load_model() -> tuple[HBOS, RobustScaler]:
    model = joblib.load(HBOS_MODEL)
    scaler = joblib.load(FEATURE_SCALER)
    return model, scaler


def load_scores() -> pd.DataFrame:
    return pd.read_parquet(CUSTOMER_SCORES)
