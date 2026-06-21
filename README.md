# Retail Probuyer Risk Detection + LLM Explainability

> This project demonstrates how traditional anomaly detection can be enhanced with LLM explainability for business review workflows. It is not a production daigou detection system and does not use real company data.

## Project summary

Identify customers showing **wholesale-like or probuyer-like buying patterns** in retail transaction data using unsupervised anomaly detection (HBOS). The model flags unusual customers; saved business rules explain *why* they are flagged; an LLM translates structured evidence into plain-English summaries a business analyst can act on.

## Dataset

**UCI Online Retail** — public transactional dataset from a UK-based online retailer (2010–2011). Contains ~540k transactions across ~4k customers.

The dataset is downloaded automatically on first run. If the download fails, manual placement instructions are printed.

## Why HBOS

- No labels required (no ground-truth daigou labels exist)
- Naturally handles extreme values in quantity and spend features
- Fast and interpretable: one histogram bin per feature
- Output is a continuous anomaly score that maps cleanly to percentile-based risk bands

## Why LLM

The DS layer produces a risk score and structured rule-hit evidence. A business reviewer needs plain language, not a JSON blob. The LLM bridges that gap — it translates evidence into readable summaries, without making any risk decisions itself.

## Installation

```bash
# Requires uv (https://docs.astral.sh/uv/)
uv python pin 3.11
uv sync
cp .env.example .env   # optional: fill in DEEPSEEK_API_KEY for live LLM
```

## How to run

```bash
# Build all artifacts (download → clean → features → model → rules → explain → cases)
uv run python scripts/06_run_all.py

# Launch dashboard
uv run streamlit run app/dashboard.py

# Run tests
uv run pytest
```

## Dashboard

Four pages:
- **Overview** — score distribution, risk band counts, anomaly type breakdown, top-20 flagged customers
- **Customer Investigation** — per-customer deep dive with LLM explanation
- **What-if Simulator** — policy threshold tuning with before/after impact
- **Case Studies** — three contrasting customer profiles

The dashboard works fully **without a DeepSeek API key** (mock explanations are shown).

## Example customer explanation

> Customer 14646 is flagged because their total purchase quantity and spend are far above the normal customer population. The pattern is consistent with wholesale-like buying rather than ordinary consumer shopping. This does not prove abuse, but it suggests the customer should be reviewed before receiving rebate or priority stock privileges.

## Limitations

- Based on public data; no real daigou labels exist
- HBOS is unsupervised — no precision/recall metrics possible
- LLM explanations depend on DeepSeek API availability
- Dashboard is for showcase only; no production auth

## Future work

- Add supervised layer if labels become available
- Integrate customer service history or return-reason data
- Add time-series features (seasonal buying, purchase acceleration)
- Support multi-retailer datasets

---

*Built with: Python 3.11 · pandas · scikit-learn · PyOD (HBOS) · Streamlit · Plotly · DeepSeek*
