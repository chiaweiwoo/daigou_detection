"""Orchestrate the full pipeline: download -> clean -> features -> model -> rules -> explain -> cases -> calibrate."""

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
        print("\n[1/8] Downloading data ...")
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
        print("\n[1/8] Skipping download (dataset present).")

    # Step 2: Clean data
    print("\n[2/8] Cleaning data ...")
    from probuyer_xai.data import load_raw, clean, save_processed
    raw_df = load_raw()
    txn_clean, txn_normal = clean(raw_df)
    save_processed(txn_clean, txn_normal)
    print(f"  {len(txn_clean):,} clean rows, {len(txn_normal):,} normal rows")

    # Step 3: Feature engineering
    print("\n[3/8] Building customer features ...")
    from probuyer_xai.features import build_features, save_features
    features, meta = build_features(txn_clean, txn_normal)
    save_features(features, meta)
    print(f"  {len(features):,} customers, bulk threshold: {meta['bulk_threshold_qty']:.0f} units")

    # Step 4: Train HBOS
    print("\n[4/8] Training HBOS model ...")
    from probuyer_xai.model import train, save_model
    model, scaler, scores = train(features)
    save_model(model, scaler, scores)
    high_n = (scores["risk_band"] == "High").sum()
    med_n = (scores["risk_band"] == "Medium").sum()
    print(f"  High risk: {high_n}, Medium risk: {med_n}")

    # Step 5: Apply rules
    print("\n[5/8] Applying business rules ...")
    from probuyer_xai.rules import load_rules, apply_rules, save_rule_hits
    rules_doc = load_rules()
    rule_hits = apply_rules(features, rules_doc)
    save_rule_hits(rule_hits)

    # Step 6: Build explanation evidence
    print("\n[6/8] Building explanation evidence ...")
    from probuyer_xai.explain import build_evidence
    from probuyer_xai.calibrate import save_evidence
    evidences = build_evidence(features, scores, rule_hits)
    save_evidence(evidences)
    flagged = [e for e in evidences if e["risk_band"] in ("High", "Medium")]
    print(f"  Evidence built for {len(evidences):,} customers ({len(flagged)} flagged)")

    # Step 7: Generate case studies
    print("\n[7/8] Generating case studies ...")
    from probuyer_xai.reporting import generate_case_studies
    generate_case_studies(evidences)

    # Step 8: LLM-as-judge calibration
    print("\n[8/8] Running LLM calibration ...")
    from probuyer_xai.calibrate import load_report, run_calibration, save_report
    existing = load_report()
    iteration = (existing.iteration + 1) if existing else 1
    report = run_calibration(evidences, iteration=iteration)
    save_report(report)
    mode = "mock" if report.is_mock else report.model_used
    print(f"  Overall agreement: {report.overall_agreement:.1%} [{mode}]")
    for bc in report.by_band:
        print(f"    {bc.band:6s}: {bc.agreed}/{bc.sampled} ({bc.agreement_rate:.0%})")
    for s in report.suggested_adjustments:
        print(f"  > {s}")

    print()
    print("=" * 60)
    print("Pipeline complete!")
    print(f"  Customers analysed: {len(features):,}")
    print(f"  High risk: {high_n} | Medium risk: {med_n}")
    print(f"  Calibration: iteration {iteration}, {report.overall_agreement:.1%} agreement")
    print(f"  Run dashboard: uv run streamlit run app/dashboard.py")
    print("=" * 60)
    return True
