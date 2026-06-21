"""Apply business rules to customer features and save rule hits."""

from probuyer_xai.features import load_features
from probuyer_xai.rules import load_rules, apply_rules, save_rule_hits

if __name__ == "__main__":
    features = load_features()
    rules_doc = load_rules()
    print(f"Applying {len(rules_doc['rules'])} rules to {len(features):,} customers ...")
    rule_hits = apply_rules(features, rules_doc)
    save_rule_hits(rule_hits)
    print("\nRule hit distribution:")
    print(rule_hits["rule_count"].value_counts().sort_index().to_string())
    print("\nPhase 4 complete.")
