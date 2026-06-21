# Retail Probuyer Risk Detection + LLM Explainability

Unsupervised anomaly detection (HBOS) on retail transactions to identify wholesale-like
buying patterns, with LLM-generated explanations and a Streamlit dashboard.

Built on UCI Online Retail data (public, 2010–2011 UK retailer). Portfolio project —
not a production system.

## Quickstart

```bash
uv python pin 3.11
uv sync
cp .env.example .env        # optional: add DEEPSEEK_API_KEY for live LLM

uv run python scripts/06_run_all.py     # full pipeline (download → model → calibrate)
uv run streamlit run app/dashboard.py  # launch dashboard
uv run pytest                           # run tests
```

Works without a DeepSeek key — mock explanations are shown instead.

## Technical walkthrough

See [WALKTHROUGH.md](WALKTHROUGH.md) for a full description of the architecture,
every module, design decisions, and known limitations.

---

*Python 3.11 · pandas · PyOD (HBOS) · scikit-learn · Streamlit · Plotly · DeepSeek*
