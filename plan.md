# PLAN.md — Retail Probuyer Risk Detection + LLM Explainability

## 1. Project Summary

Build an end-to-end DS + LLM project that detects **probuyer-like retail customers** using public transaction data.

The business idea is inspired by daigou / professional buyers:

* They buy in unusually large quantities.
* They may repeatedly buy the same products or categories.
* They can bring sales, so they are not automatically bad.
* But they may create stockout risk, rebate abuse risk, or unfair access to limited products.
* The system should flag them for business review, not automatically punish them.

This project demonstrates a traditional anomaly detection workflow enhanced by LLM explainability.

Core idea:

```text
Retail transactions
→ customer-level feature engineering
→ HBOS anomaly detection
→ saved business rules
→ structured evidence
→ LLM explanation
→ dashboard showcase
```

The LLM must not decide who is risky.
The DS/rule layer decides.
The LLM only explains, summarizes, and answers simple what-if questions using structured evidence.

---

## 2. Locked Defaults

These defaults should be used unless technically impossible.

### Dataset

Use **UCI Online Retail**.

Reason:

* Customer-level transaction data
* Invoice/order ID
* Product/item ID
* Quantity
* Price
* Date
* Customer ID
* Country
* Many customers behave like wholesalers, which fits the probuyer-risk story

Dataset fallback:

* Try automatic download first.
* If download fails, ask user to manually place the file here:

```text
data/raw/Online Retail.xlsx
```

Do not change dataset unless explicitly instructed.

### Environment

Use:

```text
Python 3.11
uv
pandas
numpy
scikit-learn
pyod
plotly
streamlit
pydantic
python-dotenv
joblib
openpyxl
```

### LLM

Use DeepSeek via environment variables.

Expected `.env`:

```text
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=
DEEPSEEK_MODEL=deepseek-v4-flash
```

If API key is missing, the app should still run using a deterministic mock explanation mode.

### Main anomaly algorithm

Use **HBOS** as the primary anomaly detection model.

Why:

* No labels required
* Good for extreme retail behavior
* Simple and explainable
* Similar to the historical business approach
* Suitable for quantity/spend/frequency anomalies

Also include simple percentile rules as supporting evidence.

### Dashboard

Use Streamlit.

The dashboard is for showcase only, not production.

---

## 3. Non-Negotiable Project Principles

1. Do not overclaim.

   * Say “probuyer-like risk” or “wholesale-like behavior.”
   * Do not say “confirmed daigou.”

2. LLM does not make the risk decision.

   * Risk score, risk band, rule hits, and confidence come from Python logic.

3. Rules must be saved in repo.

   * Store final rules as JSON.
   * Dashboard should load saved rules.

4. Model must be reproducible.

   * Training script should save model artifacts.
   * Scoring script should produce stable outputs.

5. Project must be portfolio-friendly.

   * Clear README.
   * Clear dashboard.
   * Clear case studies.
   * Clear explanation of why HBOS was used.

6. AI coding agent should not ask unnecessary questions.

   * Make reasonable defaults.
   * Pause only for missing API key or missing dataset.

---

## 4. Repository Structure

Create this structure:

```text
retail-probuyer-xai/
├── AGENTS.md
├── CLAUDE.md
├── PLAN.md
├── PRD.md
├── TECHNICAL_DESIGN.md
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/
│   ├── hbos_model.joblib
│   ├── feature_scaler.joblib
│   └── model_metadata.json
├── rules/
│   └── probuyer_rules.json
├── reports/
│   ├── data_understanding.md
│   ├── model_summary.md
│   ├── case_studies.md
│   └── llm_examples.md
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_anomaly_detection.ipynb
│   └── 04_case_studies.ipynb
├── src/
│   └── probuyer_xai/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── model.py
│       ├── rules.py
│       ├── explain.py
│       ├── llm.py
│       ├── whatif.py
│       ├── reporting.py
│       └── pipeline.py
├── app/
│   └── dashboard.py
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_prepare_data.py
│   ├── 03_train_model.py
│   ├── 04_generate_rules.py
│   ├── 05_generate_case_studies.py
│   └── 06_run_all.py
└── tests/
    ├── test_data.py
    ├── test_features.py
    ├── test_rules.py
    ├── test_explain.py
    └── test_whatif.py
```

---

## 5. Documentation Files to Create First

Before implementation, create these docs.

### 5.1 AGENTS.md

Purpose:

Tell AI coding agents how to work.

Must include:

* Do not ask unnecessary questions.
* Use uv.
* Keep DS layer deterministic.
* Do not let LLM decide risk.
* Use mock LLM mode when API key is missing.
* Run tests after implementation.
* Keep files small and modular.
* Use type hints where reasonable.
* Prefer readable code over clever code.

### 5.2 CLAUDE.md

Purpose:

Claude Code-specific guide.

Must include:

* Project objective
* Setup commands
* Run commands
* Test commands
* Folder explanation
* Implementation order
* Known assumptions
* What to avoid

### 5.3 PRD.md

Purpose:

Business/product explanation.

Must include:

* Problem statement
* Target user
* User stories
* In-scope features
* Out-of-scope features
* Acceptance criteria
* Limitations

### 5.4 TECHNICAL_DESIGN.md

Purpose:

Technical design.

Must include:

* Data pipeline
* Feature engineering
* HBOS model design
* Rule extraction
* LLM explanation design
* What-if logic
* Dashboard design
* Testing plan

### 5.5 README.md

Purpose:

Portfolio-facing file.

Initially keep it simple. Final polish after implementation.

Must include:

* Project summary
* Dataset
* How to run
* What the dashboard shows
* Why HBOS
* Why LLM
* Limitations
* Future work

---

## 6. Implementation Phases

## Phase 0 — Project Setup

Create:

```text
pyproject.toml
.env.example
.gitignore
src/probuyer_xai/
scripts/
app/
tests/
```

Use uv.

Expected commands:

```bash
uv sync
uv run python scripts/06_run_all.py
uv run streamlit run app/dashboard.py
uv run pytest
```

Acceptance criteria:

* Project installs.
* Package imports work.
* Empty dashboard can start.
* Tests can run.

---

## Phase 1 — Data Download and Loading

Create:

```text
scripts/01_download_data.py
src/probuyer_xai/data.py
```

Behavior:

1. Try to download UCI Online Retail automatically.
2. Save raw file to:

```text
data/raw/Online Retail.xlsx
```

3. If download fails, print clear instruction:

```text
Please download UCI Online Retail manually and place it at:
data/raw/Online Retail.xlsx
```

4. Load Excel file with pandas.
5. Standardize column names.

Expected standardized columns:

```text
invoice_no
stock_code
description
quantity
invoice_date
unit_price
customer_id
country
```

Acceptance criteria:

* Raw data loads.
* Column names are standardized.
* A small sample can be printed.

---

## Phase 2 — Data Cleaning

Create:

```text
scripts/02_prepare_data.py
src/probuyer_xai/data.py
```

Cleaning logic:

1. Remove rows without `customer_id`.
2. Parse `invoice_date`.
3. Create `amount = quantity * unit_price`.
4. Mark cancelled invoices:

   * invoice number starts with `C`
   * or quantity < 0
5. Keep cancellation rows for cancellation features.
6. Create normal purchase subset:

   * quantity > 0
   * unit_price > 0
   * customer_id not null
7. Save outputs:

```text
data/processed/transactions_clean.parquet
data/processed/transactions_normal.parquet
```

Also create data quality report:

```text
reports/data_understanding.md
```

Report should include:

* row count
* customer count
* invoice count
* date range
* missing customer rows
* cancellation count
* top countries
* quantity distribution
* amount distribution

Acceptance criteria:

* Cleaned files are saved.
* Report is generated.
* No silent data loss.

---

## Phase 3 — Feature Engineering

Create:

```text
src/probuyer_xai/features.py
```

Generate one row per customer.

Required features:

```text
customer_id
total_quantity
total_spend
num_invoices
active_days
avg_quantity_per_invoice
max_quantity_per_invoice
avg_spend_per_invoice
max_spend_per_invoice
unique_skus
unique_descriptions
repeat_sku_ratio
top_sku_quantity_share
top_sku_spend_share
quantity_per_active_day
spend_per_active_day
bulk_invoice_count
bulk_invoice_ratio
cancelled_invoice_count
cancelled_quantity_abs
cancelled_amount_abs
cancellation_ratio
first_purchase_date
last_purchase_date
```

Bulk invoice logic:

* Compute invoice-level quantity.
* Define a bulk invoice as an invoice whose quantity is above the 95th percentile of invoice quantity.
* Save threshold in metadata.

Output:

```text
data/processed/customer_features.parquet
data/processed/feature_metadata.json
```

Acceptance criteria:

* One row per customer.
* No infinite values.
* No unexpected nulls in model features.
* Feature metadata saved.

---

## Phase 4 — HBOS Model Training

Create:

```text
src/probuyer_xai/model.py
scripts/03_train_model.py
```

Model features:

```text
total_quantity
total_spend
num_invoices
active_days
avg_quantity_per_invoice
max_quantity_per_invoice
avg_spend_per_invoice
max_spend_per_invoice
unique_skus
repeat_sku_ratio
top_sku_quantity_share
top_sku_spend_share
quantity_per_active_day
spend_per_active_day
bulk_invoice_count
bulk_invoice_ratio
cancellation_ratio
```

Preprocessing:

* log1p transform skewed positive features
* robust scaling
* fit HBOS

Model output:

```text
customer_id
anomaly_score
risk_percentile
risk_band
```

Risk bands:

```text
Top 1% = High
Top 1–3% = Medium
Others = Low
```

Save:

```text
models/hbos_model.joblib
models/feature_scaler.joblib
models/model_metadata.json
data/processed/customer_scores.parquet
```

Acceptance criteria:

* HBOS model trains.
* Customer scores are saved.
* Top anomalies can be printed.
* Model metadata includes feature list, date, thresholds, and contamination assumption.

---

## Phase 5 — Business Rule Generation

Create:

```text
src/probuyer_xai/rules.py
scripts/04_generate_rules.py
```

Rules should be saved to:

```text
rules/probuyer_rules.json
```

Use percentile-based rules.

Example saved rules:

```json
{
  "version": "v1",
  "description": "Simple probuyer-like risk rules based on customer-level batch purchasing features.",
  "rules": [
    {
      "rule_id": "R001",
      "name": "Extreme total quantity",
      "feature": "total_quantity",
      "operator": ">=",
      "threshold_source": "p99",
      "risk_reason": "Customer purchased unusually high total quantity."
    },
    {
      "rule_id": "R002",
      "name": "Extreme max invoice quantity",
      "feature": "max_quantity_per_invoice",
      "operator": ">=",
      "threshold_source": "p99",
      "risk_reason": "Customer had at least one unusually large basket."
    },
    {
      "rule_id": "R003",
      "name": "High bulk invoice ratio",
      "feature": "bulk_invoice_ratio",
      "operator": ">=",
      "threshold_source": "p95",
      "risk_reason": "Large share of customer invoices are bulk-like."
    },
    {
      "rule_id": "R004",
      "name": "High SKU concentration",
      "feature": "top_sku_quantity_share",
      "operator": ">=",
      "threshold_source": "p95",
      "risk_reason": "Large share of quantity comes from one SKU."
    },
    {
      "rule_id": "R005",
      "name": "High cancellation ratio",
      "feature": "cancellation_ratio",
      "operator": ">=",
      "threshold_source": "p95",
      "risk_reason": "Customer has unusually high cancellation or return-like behavior."
    }
  ]
}
```

Apply rules to customers.

Output:

```text
data/processed/customer_rule_hits.parquet
```

Acceptance criteria:

* Rules are saved as JSON.
* Rule hits are reproducible.
* Each high-risk customer has a list of triggered rules.
* Rules can run without retraining model.

---

## Phase 6 — Explanation Layer

Create:

```text
src/probuyer_xai/explain.py
```

This layer creates structured explanation evidence.

For each customer, generate:

```json
{
  "customer_id": "12345",
  "risk_band": "High",
  "risk_percentile": 99.7,
  "anomaly_type": "broad_wholesale_buyer",
  "confidence": "High",
  "top_reasons": [],
  "rule_hits": [],
  "recommended_action": ""
}
```

Anomaly type logic:

```text
broad_wholesale_buyer:
  high spend + high quantity + diverse SKU mix

single_product_bulk_buyer:
  high quantity + high top_sku_quantity_share

high_frequency_buyer:
  high num_invoices + high active_days or quantity_per_active_day

return_or_cancellation_anomaly:
  high cancellation_ratio or cancelled amount

potential_stockout_risk:
  high max invoice quantity or high top SKU concentration
```

Confidence logic:

```text
High:
  high risk band + at least 2 probuyer-related rule hits

Medium:
  medium/high risk band + at least 1 probuyer-related rule hit

Low:
  anomaly score high but evidence suggests operational anomaly or weak probuyer pattern
```

Acceptance criteria:

* Explanation evidence is structured JSON.
* Explanation does not require LLM.
* Output can be used by dashboard and LLM.

---

## Phase 7 — LLM Integration

Create:

```text
src/probuyer_xai/llm.py
```

DeepSeek wrapper:

* Read `.env`
* Use model from `DEEPSEEK_MODEL`
* Use mock mode if key missing
* Do not crash dashboard if LLM is unavailable

LLM tasks:

1. Explain customer risk
2. Summarize case study
3. Explain what-if result
4. Generate monthly portfolio-style summary

Prompt rules:

* Only use provided JSON evidence.
* Do not invent numbers.
* Do not accuse customer.
* Use business-friendly language.
* Say “probuyer-like” or “wholesale-like,” not “confirmed daigou.”
* Mention uncertainty.

Example output style:

```text
Customer 14646 is flagged because their total purchase quantity and spend are far above the normal customer population. The pattern is consistent with wholesale-like buying rather than ordinary consumer shopping. This does not prove abuse, but it suggests the customer should be reviewed before receiving rebate or priority stock privileges.
```

Acceptance criteria:

* LLM can generate explanation from structured evidence.
* Mock mode works without API key.
* No raw chain-of-thought or hidden reasoning is requested.
* No hallucinated metrics.

---

## Phase 8 — What-if Analysis

Create:

```text
src/probuyer_xai/whatif.py
```

Supported what-if functions:

1. Change high-risk percentile threshold.
2. Change bulk invoice threshold.
3. Exclude cancellation-related anomalies.
4. Whitelist a customer.
5. Require at least N rule hits.

Output:

```json
{
  "scenario": "Require at least 2 rule hits",
  "before_flagged_count": 130,
  "after_flagged_count": 78,
  "customers_added": [],
  "customers_removed": [],
  "business_interpretation": ""
}
```

LLM may explain the result, but Python computes the result.

Acceptance criteria:

* What-if logic is deterministic.
* Results are returned as JSON.
* LLM only explains the JSON.

---

## Phase 9 — Case Studies

Create:

```text
scripts/05_generate_case_studies.py
src/probuyer_xai/reporting.py
```

Pick 3 customer cases:

1. Highest-confidence broad wholesale/probuyer-like buyer
2. Strong single-product bulk buyer
3. High anomaly but lower probuyer confidence, such as cancellation/return anomaly

For each case, output:

```text
customer_id
risk_score
risk_band
anomaly_type
rule_hits
key metrics
confidence
business interpretation
recommended action
LLM explanation
```

Save:

```text
reports/case_studies.md
reports/llm_examples.md
```

Acceptance criteria:

* Three cases show different behavior patterns.
* Each case has DS evidence and LLM explanation.
* Confidence is clearly stated.
* No customer is called confirmed daigou.

---

## Phase 10 — Dashboard

Create:

```text
app/dashboard.py
```

Dashboard pages:

### Page 1 — Overview

Show:

* total customers
* high-risk customer count
* medium-risk customer count
* score distribution
* anomaly type breakdown
* top 20 flagged customers

### Page 2 — Customer Investigation

Show:

* customer selector
* risk band
* risk percentile
* anomaly type
* confidence
* key metrics
* rule hits
* LLM explanation
* sample transactions

### Page 3 — What-if Simulator

Controls:

* high-risk percentile threshold
* minimum rule hits
* exclude cancellation anomaly toggle
* customer whitelist input

Show:

* before flagged count
* after flagged count
* customers added
* customers removed
* explanation

### Page 4 — Case Studies

Show the three generated case studies.

Acceptance criteria:

* Dashboard runs locally.
* Dashboard works without DeepSeek key.
* Dashboard can showcase project in under five minutes.
* No production auth needed.

---

## Phase 11 — Full Pipeline Script

Create:

```text
scripts/06_run_all.py
```

Run steps:

```text
1. download data
2. clean data
3. build features
4. train HBOS
5. generate rules
6. generate explanations
7. generate case studies
8. print completion summary
```

Command:

```bash
uv run python scripts/06_run_all.py
```

Acceptance criteria:

* One command builds all processed artifacts.
* Clear logs are printed.
* If dataset is missing, script gives manual instruction.
* If LLM key is missing, script continues in mock mode.

---

## 12. Testing Plan

Create tests for:

### Data

* standardized columns exist
* amount calculation works
* cancellation detection works

### Features

* one row per customer
* no infinite values
* bulk invoice count is non-negative
* ratios are between 0 and 1

### Rules

* rules load from JSON
* rule application returns expected columns
* at least one rule can trigger on synthetic test row

### Explanation

* explanation JSON contains required fields
* confidence logic works
* anomaly type logic works

### What-if

* threshold change updates flagged count
* whitelist removes selected customer
* minimum rule hits filter works

Command:

```bash
uv run pytest
```

---

## 13. Final README Requirements

After implementation, update README with:

1. Project title
2. Business problem
3. Why this dataset
4. Why HBOS
5. Why LLM
6. How to install
7. How to run pipeline
8. How to run dashboard
9. Main screenshots
10. Example customer explanation
11. Limitations
12. Future work

README positioning:

```text
This project demonstrates how traditional anomaly detection can be enhanced with LLM explainability for business review workflows. It is not a production daigou detection system and does not use real company data.
```

---

## 14. Development Order for AI Coding Agent

Follow this exact order:

1. Create repo structure.
2. Create `pyproject.toml`, `.env.example`, `.gitignore`.
3. Create documentation files:

   * `PLAN.md`
   * `AGENTS.md`
   * `CLAUDE.md`
   * `PRD.md`
   * `TECHNICAL_DESIGN.md`
   * initial `README.md`
4. Implement data loading.
5. Implement data cleaning.
6. Implement feature engineering.
7. Implement HBOS training.
8. Implement saved rules.
9. Implement explanation evidence.
10. Implement LLM wrapper with mock fallback.
11. Implement what-if functions.
12. Implement case study generation.
13. Implement Streamlit dashboard.
14. Add tests.
15. Run pipeline.
16. Run tests.
17. Polish README.

Do not build dashboard before the data pipeline works.

---

## 15. Done Definition

Project is complete when:

* `uv sync` works
* `uv run python scripts/06_run_all.py` works
* `uv run pytest` works
* `uv run streamlit run app/dashboard.py` works
* processed customer features are generated
* HBOS model artifacts are saved
* rules are saved in `rules/probuyer_rules.json`
* three case studies are generated
* dashboard displays results
* LLM explanation works or mock explanation works
* README explains the project clearly

---

## 16. Coding Agent Behavior

The coding agent should proceed autonomously.

Do not ask the user about:

* package choice
* folder structure
* dashboard library
* risk band thresholds
* first version feature list
* first version rule list
* README structure
* testing approach

Only pause when:

1. DeepSeek API key is needed for real LLM output.
2. Dataset cannot be downloaded and must be manually placed in `data/raw`.
3. A command fails in a way that cannot be resolved from error logs.

When blocked, provide:

* what failed
* exact file needed
* exact command to retry
* exact `.env` variable needed

---

## 17. Expected Portfolio Story

Final project story:

> I built a retail probuyer-risk detection project using public transaction data. Since there is no ground-truth daigou label, I used HBOS to identify rare customer buying patterns, then converted the anomaly output into transparent business rules. The LLM does not make the risk decision; it explains structured DS evidence in business language and supports what-if analysis. The result is a small but realistic example of combining traditional anomaly detection with LLM-based explainability for business review workflows.
