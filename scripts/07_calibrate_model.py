"""Run LLM-as-judge calibration on current model outputs.

Samples customers from each risk band, has a pro LLM independently rate
their probuyer-likeness, and prints an agreement report with suggestions.

Usage:
    uv run python scripts/07_calibrate_model.py

Requires:
    - Pipeline to have run first (scripts/06_run_all.py)
    - DEEPSEEK_API_KEY in .env (or mock mode runs automatically)
    - CALIBRATION_MODEL in .env to override the pro model (default: deepseek-chat)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probuyer_xai.calibrate import load_report, run_calibration, save_report
from probuyer_xai.config import CALIBRATION_REPORT, CUSTOMER_EVIDENCE


def main() -> None:
    if not CUSTOMER_EVIDENCE.exists():
        print(f"Evidence file not found: {CUSTOMER_EVIDENCE}")
        print("Run the pipeline first: uv run python scripts/06_run_all.py")
        sys.exit(1)

    with open(CUSTOMER_EVIDENCE) as f:
        all_evidence = json.load(f)

    existing = load_report()
    iteration = (existing.iteration + 1) if existing else 1

    print(f"Running calibration iteration {iteration} on {len(all_evidence):,} customers ...")
    report = run_calibration(all_evidence, iteration=iteration)
    save_report(report)

    mode = "mock" if report.is_mock else report.model_used
    print(f"\nCalibration complete [{mode}]")
    print(f"Overall agreement: {report.overall_agreement:.1%}")
    print()
    for bc in report.by_band:
        bar = "█" * bc.agreed + "░" * (bc.sampled - bc.agreed)
        print(f"  {bc.band:6s}  {bar}  {bc.agreed}/{bc.sampled} ({bc.agreement_rate:.0%})")

    print("\nSuggested adjustments:")
    for s in report.suggested_adjustments:
        print(f"  • {s}")

    if not report.is_mock:
        print("\nDisagreement cases:")
        for bc in report.by_band:
            for r in bc.disagreements:
                print(
                    f"  Customer {r.customer_id} | model={r.model_band} | "
                    f"llm_score={r.llm_score} | {r.primary_signal or 'no primary signal'}"
                )
                for d in r.disqualifiers[:2]:
                    print(f"    disqualifier: {d}")

    print(f"\nFull report: {CALIBRATION_REPORT}")


if __name__ == "__main__":
    main()
