"""Generate case studies from existing processed artifacts."""

from probuyer_xai.features import load_features
from probuyer_xai.model import load_scores
from probuyer_xai.rules import load_rule_hits
from probuyer_xai.explain import build_evidence
from probuyer_xai.reporting import generate_case_studies

if __name__ == "__main__":
    features = load_features()
    scores = load_scores()
    rule_hits = load_rule_hits()
    evidences = build_evidence(features, scores, rule_hits)
    generate_case_studies(evidences)
    print("Phase 5 complete.")
