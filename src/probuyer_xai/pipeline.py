"""Orchestrate the full pipeline: download -> clean -> features -> model -> rules -> explain -> cases."""

from __future__ import annotations

import sys

from probuyer_xai.config import RAW_XLSX


def run_all(skip_download: bool = False) -> bool:
    """Run all pipeline steps. Returns True on success."""
    print("=" * 60)
    print("Retail Probuyer Risk Detection Pipeline")
    print("=" * 60)

    # Step 1: Download data
    if not skip_download:
        print("\n[1/7] Downloading data ...")
        from probuyer_xai.data import download_raw
        ok = download_raw()
        if not ok:
            print()
            print("=" * 60)
            print("PIPELINE STOPPED — Dataset missing")
            print("=" * 60)
            print("Please download the UCI Online Retail dataset manually:")
            print("  URL: https://archive.ics.uci.edu/ml/datasets/Online+Retail")
            print(f"  Save to: {RAW_XLSX}")
            print("Then re-run: uv run python scripts/06_run_all.py")
            print("=" * 60)
            return False
    else:
        print("\n[1/7] Skipping download (dataset present).")

    # Step 2: Clean data
    print("\n[2/7] Cleaning data ...")
    from probuyer_xai.data import load_raw, clean, save_processed
    raw_df = load_raw()
    txn_clean, txn_normal = clean(raw_df)
    save_processed(txn_clean, txn_normal)
    print(f"  {len(txn_clean):,} clean rows, {len(txn_normal):,} normal rows")

    # Step 3: Feature engineering
    print("\n[3/7] Building customer features ...")
    from probuyer_xai.features import build_features, save_features
    features, meta = build_features(txn_clean, txn_normal)
    save_features(features, meta)
    print(f"  {len(features):,} customers, bulk threshold: {meta['bulk_threshold_qty']:.0f} units")

    # Step 4: Train HBOS
    print("\n[4/7] Training HBOS model ...")
    from probuyer_xai.model import train, save_model
    model, scaler, scores = train(features)
    save_model(model, scaler, scores)
    high_n = (scores["risk_band"] == "High").sum()
    med_n = (scores["risk_band"] == "Medium").sum()
    print(f"  High risk: {high_n}, Medium risk: {med_n}")

    # Step 5: Apply rules
    print("\n[5/7] Applying business rules ...")
    from probuyer_xai.rules import load_rules, apply_rules, save_rule_hits
    rules_doc = load_rules()
    rule_hits = apply_rules(features, rules_doc)
    save_rule_hits(rule_hits)

    # Step 6: Build explanation evidence
    print("\n[6/7] Building explanation evidence ...")
    from probuyer_xai.explain import build_evidence
    evidences = build_evidence(features, scores, rule_hits)
    print(f"  Evidence built for {len(evidences):,} customers")

    # Step 7: Generate case studies
    print("\n[7/7] Generating case studies ...")
    from probuyer_xai.reporting import generate_case_studies
    generate_case_studies(evidences)

    print()
    print("=" * 60)
    print("Pipeline complete!")
    print(f"  Customers analysed: {len(features):,}")
    print(f"  High risk: {high_n} | Medium risk: {med_n}")
    print(f"  Run dashboard: uv run streamlit run app/dashboard.py")
    print("=" * 60)
    return True
