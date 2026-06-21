# TECHNICAL_DESIGN.md — Technical Design

## Data pipeline

```
UCI Online Retail .xlsx
  → scripts/01_download_data.py    (auto-download or manual instruction)
  → data/raw/Online Retail.xlsx
  → scripts/02_prepare_data.py     (clean, standardise, split)
  → data/processed/transactions_clean.parquet    (all rows, cancellation flag added)
  → data/processed/transactions_normal.parquet   (qty>0, price>0, has customer_id)
```

Column standardisation map:
`InvoiceNo→invoice_no, StockCode→stock_code, Description→description, Quantity→quantity, InvoiceDate→invoice_date, UnitPrice→unit_price, CustomerID→customer_id, Country→country`

Cancellation detection: `invoice_no.str.startswith("C")` OR `quantity < 0`.

## Feature engineering (`src/probuyer_xai/features.py`)

Aggregated to one row per `customer_id` from `transactions_normal`.

Cancellation features are computed separately from `transactions_clean` (cancelled rows).

Bulk invoice threshold: 95th percentile of invoice-level total quantity. Stored in `data/processed/feature_metadata.json`.

SKU concentration: `top_sku_quantity_share = max SKU quantity / total quantity`.

`repeat_sku_ratio`: fraction of SKUs purchased more than once.

## HBOS model design (`src/probuyer_xai/model.py`)

Preprocessing:
1. `log1p` transform on all model features (handles right-skewed distributions)
2. `RobustScaler` (handles outliers in scale)
3. `HBOS` from `pyod`

Risk band assignment:
- `risk_percentile = percentile rank of anomaly_score` within the customer population
- `risk_band`: Top 1% → High, Top 1–3% → Medium, else → Low

Saved artifacts: `hbos_model.joblib`, `feature_scaler.joblib`, `model_metadata.json`

## Rule extraction (`src/probuyer_xai/rules.py`)

Rules in `rules/probuyer_rules.json` define feature + operator + percentile source.

At rule-application time:
1. Load rules JSON
2. Compute per-feature thresholds from customer feature distribution
3. Evaluate each rule condition per customer
4. Output `customer_rule_hits.parquet` with columns: `customer_id, rule_hits (list), rule_count`

Rules run independently of the model — no retraining needed.

## LLM explanation design (`src/probuyer_xai/llm.py`)

- Raw `requests.post` to `{DEEPSEEK_BASE_URL}/chat/completions`
- Model: `DEEPSEEK_MODEL` from env (default: `deepseek-v4-flash`)
- Auth: `Authorization: Bearer {DEEPSEEK_API_KEY}`
- Input: structured explanation evidence JSON
- Output: plain English business text

Mock mode: if `DEEPSEEK_API_KEY` is empty or the HTTP call fails, generate deterministic text from the evidence JSON template.

Prompt guardrails built into system prompt:
- Only reference provided JSON values
- Never invent numbers
- Never accuse; say "flagged for review"
- Use "probuyer-like" not "daigou"
- State uncertainty

## What-if logic (`src/probuyer_xai/whatif.py`)

All functions operate on in-memory DataFrames (loaded from parquet).

Supported scenarios:
1. `change_risk_threshold(new_pct)` — re-assign High/Medium bands
2. `change_bulk_threshold(new_pct)` — recompute bulk invoice ratio
3. `exclude_cancellation_anomalies()` — remove customers whose primary anomaly type is `return_or_cancellation_anomaly`
4. `whitelist_customer(customer_id)` — remove specific customer from flagged list
5. `require_min_rule_hits(n)` — filter flagged list to customers with ≥n rule hits

Each returns: `{scenario, before_flagged_count, after_flagged_count, customers_added, customers_removed, business_interpretation}`

LLM explains the JSON output but does not compute it.

## Dashboard design (`app/dashboard.py`)

Four Streamlit pages via `st.sidebar.radio`:

| Page | Key content |
|---|---|
| Overview | KPI cards, score distribution histogram, anomaly type donut, top-20 table |
| Customer Investigation | Customer selector, risk card, metric table, rule hits, LLM explanation, sample transactions |
| What-if Simulator | Sliders/toggles, before/after count, diff table, LLM interpretation |
| Case Studies | Three customer deep-dives loaded from reports/case_studies.md |

Dashboard reads saved parquets, rules JSON, and model metadata — no retraining on startup.
Loads `.env` so LLM calls work if key is present; falls back to mock otherwise.

## Testing plan

All tests use synthetic DataFrames — no dataset or model dependency.

| Test file | What it covers |
|---|---|
| `test_data.py` | Column standardisation, amount calc, cancellation flag |
| `test_features.py` | One row/customer, no inf values, ratios in [0,1] |
| `test_rules.py` | Rules JSON loads, rule application columns, synthetic trigger |
| `test_explain.py` | Required evidence fields, confidence logic, anomaly type logic |
| `test_whatif.py` | Threshold change, whitelist, min-rule-hits filter |
