"""Train HBOS model and save artifacts."""

from probuyer_xai.features import load_features
from probuyer_xai.model import train, save_model

if __name__ == "__main__":
    features = load_features()
    print(f"Training HBOS on {len(features):,} customers, {len(features.columns)} features ...")
    model, scaler, scores = train(features)
    save_model(model, scaler, scores)
    top5 = scores.nlargest(5, "anomaly_score")[["customer_id", "anomaly_score", "risk_percentile", "risk_band"]]
    print("\nTop 5 anomalies:")
    print(top5.to_string(index=False))
    print("\nPhase 3 complete.")
