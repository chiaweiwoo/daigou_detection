# Technical Walkthrough — Retail Probuyer Risk Detection

This document is the single source of truth for understanding the codebase. It is written
for someone new to the project who wants to know what every part does, why it was built
that way, and where to find it.

---

## What this project does

It identifies retail customers whose purchasing behaviour looks wholesale-like or
probuyer-like — think: someone buying 5,000 units of one item across many invoices,
consistent with resale rather than personal use. No ground-truth labels exist for this,
so the detection is unsupervised. The result is a risk score per customer, plain-English
explanations a business reviewer can act on, and a dashboard to explore the findings.

This is a portfolio project using public data (UCI Online Retail, 2010–2011 UK retailer).
It is not a production system and does not use real company data or real daigou labels.

---

## System architecture

```
Raw transactions (UCI xlsx)
        │
        ▼
┌─────────────────────────────────┐
│  DATA LAYER                     │
│  data.py  →  features.py        │  Clean, standardise, build customer features
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  SCORING LAYER                  │
│  model.py  +  rules.py          │  HBOS anomaly score + business rule hits
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  EXPLANATION LAYER              │
│  explain.py  →  llm.py          │  Deterministic evidence → LLM narrates it
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  CALIBRATION LAYER              │
│  calibrate.py                   │  Pro LLM validates model agreement per-band
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  INTERFACE                      │
│  app/dashboard.py               │  Streamlit, 5 pages
└─────────────────────────────────┘
```

**The core principle:** the DS/rule layer decides risk. The LLM only explains.
Risk scores, percentile thresholds, and rule hits are computed deterministically
before the LLM is ever called.

---

## How to run

```bash
# One-time setup
uv python pin 3.11
uv sync
cp .env.example .env          # fill in DEEPSEEK_API_KEY for live LLM (optional)

# Full pipeline: download → clean → features → model → rules → explain → cases → calibrate
uv run python scripts/06_run_all.py

# Dashboard
uv run streamlit run app/dashboard.py

# Tests (no dataset or API key needed)
uv run pytest

# Re-run calibration only (without retraining)
uv run python scripts/07_calibrate_model.py
```

Everything works without a DeepSeek key — mock explanations replace live LLM calls.

---

## Repository layout

```
daigou_detection/
├── src/probuyer_xai/       Core library
│   ├── config.py           Central config: all paths, constants, env vars
│   ├── data.py             Download, load, clean raw transactions
│   ├── features.py         Customer-level feature engineering
│   ├── model.py            HBOS training, anomaly scoring, risk bands
│   ├── rules.py            Business rule application engine
│   ├── explain.py          Deterministic evidence builder (no LLM)
│   ├── llm.py              LLM wrapper with mock fallback
│   ├── whatif.py           Policy scenario analysis (deterministic)
│   ├── reporting.py        Case study generation
│   ├── calibrate.py        LLM-as-judge calibration loop
│   └── pipeline.py         Orchestrate all 8 steps
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_prepare_data.py
│   ├── 03_train_model.py
│   ├── 04_generate_rules.py
│   ├── 05_generate_case_studies.py
│   ├── 06_run_all.py       Entry point — runs everything
│   └── 07_calibrate_model.py  Standalone calibration re-run
├── app/dashboard.py        Streamlit, 5 pages
├── rules/probuyer_rules.json   Version-controlled rule definitions
├── calibration/history.md      Experiment log (calibration iterations)
├── tests/                  Synthetic pytest tests (no external deps)
├── data/                   Gitignored — raw xlsx + processed parquets
├── models/                 Gitignored — HBOS joblib artifacts
└── reports/                Gitignored — regenerated markdown reports
```

---

## Module deep-dives

### `src/probuyer_xai/config.py`

Single source of truth for every path, constant, and env var. Nothing is hardcoded
elsewhere — all magic numbers live here.

Key constants:
- `ROOT` — resolves the repo root from `__file__` using `parents[2]`
- `MODEL_FEATURES` — the 16 features the HBOS model trains on (see why `cancellation_ratio`
  is excluded in the comment; including it caused the model to flag fraud/return-abusers
  as probuyers)
- `RISK_HIGH_PCT = 99.0`, `RISK_MED_PCT = 97.0` — percentile cutoffs for risk bands;
  these values were validated by the calibration loop (see `calibration/history.md`)
- `CALIBRATION_MODEL = "deepseek-chat"` — the pro model (DeepSeek-V3) used as an
  independent judge; kept separate from `DEEPSEEK_MODEL` (the flash model used for
  explanations)

---

### `src/probuyer_xai/data.py`

**Download** (`download_raw`): fetches the UCI Online Retail xlsx to `data/raw/`.
On failure, cleans up the partial file and prints the exact manual-placement path.
Does not raise — returns `False` so the pipeline can handle it gracefully.

**Load** (`load_raw`): reads the xlsx with explicit `dtype` overrides. This matters:
`StockCode` contains both integers and alphanumeric codes (`"85123A"`). Without
`dtype={"StockCode": str}`, pandas infers a mixed type that pyarrow refuses to
serialize to parquet.

**Clean** (`clean`): standardises column names to snake_case, drops rows missing
a `customer_id`, parses invoice dates, computes `amount = quantity * unit_price`,
and adds `is_cancelled = invoice_no.startswith("C") OR quantity < 0`.

Returns two DataFrames:
- `transactions_clean` — all rows including cancellations (406,829 rows)
- `transactions_normal` — positive qty, positive price, has customer_id (397,884 rows)

The two are kept separate because cancellation features need the full set, but
volume/spend features should only be computed on normal purchases.

---

### `src/probuyer_xai/features.py`

Aggregates 23 features to one row per customer. Built from `transactions_normal`
for volume/spend, from `transactions_clean` for cancellation context.

**Invoice-level first, then customer-level.** Most per-invoice metrics
(avg basket size, max basket size) are computed by grouping by
`(customer_id, invoice_no)` first, then aggregating to customer level. This prevents
multi-line invoices from inflating counts.

**Feature groups:**

| Group | Features | What they capture |
|---|---|---|
| Volume | `total_quantity`, `total_spend`, `num_invoices` | Raw scale of buying |
| Basket size | `avg_quantity_per_invoice`, `max_quantity_per_invoice`, `avg_spend_per_invoice`, `max_spend_per_invoice` | How large individual orders are |
| Activity | `active_days`, `quantity_per_active_day`, `spend_per_active_day` | Buying intensity over time |
| SKU diversity | `unique_skus`, `repeat_sku_ratio`, `top_sku_quantity_share`, `top_sku_spend_share` | How concentrated buying is |
| Bulk invoices | `bulk_invoice_count`, `bulk_invoice_ratio` | Share of orders above the bulk threshold |
| Cancellations | `cancellation_ratio` | Kept in the feature matrix for context display, excluded from model |

**Bulk threshold:** the 95th percentile of invoice-level total quantity across all
invoices in the dataset. Stored in `data/processed/feature_metadata.json` so the
dashboard can display it without retraining. At run time: ~778 units per invoice.

**`repeat_sku_ratio`:** fraction of unique SKUs a customer bought more than once,
clipped to [0,1]. Probuyers who restock for resale show high values here.

**`top_sku_quantity_share`:** `max_sku_quantity / total_quantity`. A value near 1.0
means almost all purchases are one SKU — classic single-product bulk buyer pattern.

**Why `cancellation_ratio` is excluded from `MODEL_FEATURES`:** daigou buyers keep
what they buy; high cancellation is a fraud or speculative-ordering signal, not a
probuyer signal. Including it caused the model to rank high-cancellation customers
as High-risk probuyers, which was wrong. The feature is retained in the DataFrame
for downstream display and anomaly type classification but is not fed to HBOS.

---

### `src/probuyer_xai/model.py`

**Preprocessing chain:**
1. `log1p` transform on all 16 model features — handles the right-skewed distributions
   typical of retail quantity/spend data (a few customers have 100× the median)
2. `RobustScaler` — centres on the median and scales by IQR, making the transform
   resistant to the outliers we're deliberately trying to flag
3. `HBOS` from `pyod` — Histogram-Based Outlier Score; builds one histogram per feature,
   scores each data point by the inverse of the density at its bin

**Risk band assignment:**

```python
risk_percentile = np.sum(raw_scores <= s) / n * 100   # empirical CDF, 0–100
risk_band: >= RISK_HIGH_PCT → "High", >= RISK_MED_PCT → "Medium", else → "Low"
```

A critical bug that was fixed: `_assign_risk_band` originally returned a pandas
Series indexed by `customer_id`. Assigning it to `scores` (integer-indexed) caused
pandas to silently fill everything with NaN — resulting in 0 High and 0 Medium
customers. Fixed by operating on `np.ndarray` and returning a plain `list[str]`.

**Saved artifacts:** `models/hbos_model.joblib`, `models/feature_scaler.joblib`,
`data/processed/customer_scores.parquet` (customer_id, anomaly_score, risk_percentile,
risk_band), `models/model_metadata.json`.

**Results on UCI data:** 4,338 customers scored. 44 High-risk (top 1%), 87 Medium-risk
(top 1–3%), 4,207 Low-risk.

---

### `rules/probuyer_rules.json` + `src/probuyer_xai/rules.py`

Rules are stored as data, not code. This means you can add or modify a rule without
touching Python.

```json
{
  "rule_id": "R001",
  "feature": "total_quantity",
  "operator": ">=",
  "threshold_source": "p99"
}
```

Five rules:
- **R001** `total_quantity >= p99` — extreme total volume
- **R002** `max_quantity_per_invoice >= p99` — at least one enormous order
- **R003** `bulk_invoice_ratio >= p95` — most orders are bulk-sized
- **R004** `top_sku_quantity_share >= p95` — buying is highly concentrated on one SKU
- **R005** `repeat_sku_ratio >= p95` — repeatedly buying the same SKUs (restocking pattern)

R005 was originally `cancellation_ratio >= p95`, which was wrong for the same reason
the feature was removed from the model — it flagged return-abusers as probuyers.
Changed to `repeat_sku_ratio`, a genuine resale signal.

`apply_rules` in `rules.py` computes each rule's threshold from the live customer
feature distribution, then evaluates all rules against all customers. Output:
`data/processed/customer_rule_hits.parquet` with `rule_hits` (list of rule IDs),
`rule_reasons` (list of explanatory strings), `rule_count`.

Rule hits from 574 customers at `p95`/`p99` thresholds.

---

### `src/probuyer_xai/explain.py`

Builds structured evidence JSON per customer. No LLM involved here — this is fully
deterministic. The evidence JSON is what the LLM receives as input.

**Anomaly type classification** (`_classify_anomaly_type`): uses p90 thresholds
to categorise each customer into one of five types:
- `return_or_cancellation_anomaly` — checked first; always overrides others
- `single_product_bulk_buyer` — high SKU concentration + high volume/max-invoice
- `broad_wholesale_buyer` — high quantity and spend, distributed SKUs
- `high_frequency_buyer` — unusually many invoices
- `potential_stockout_risk` — large single orders or high SKU concentration without
  the volume to qualify for the above

**Confidence scoring** (`_confidence`):

```python
if anomaly_type == "return_or_cancellation_anomaly":  → "Low"  (always)
if risk_band == "High" and rule_count >= 2:           → "High"
if risk_band in ("High", "Medium") and rule_count >= 1: → "Medium"
else:                                                 → "Low"
```

Confidence requires both model score (risk band) and rule corroboration. A
High-band customer with zero rule hits gets only Medium confidence.
A cancellation anomaly gets Low regardless of model score.

**Recommended action:** cancellation anomalies are directed to fraud/care team, not
the probuyer review queue. This was a deliberate design decision to avoid routing
fraud cases through a probuyer workflow.

**A parquet edge case that was fixed:** `rule_hits` is stored as a list column.
When read back from parquet, list columns arrive as numpy arrays, not Python lists.
`isinstance(x, list)` fails silently. Fixed: `lambda x: list(x) if hasattr(x, "__iter__") and not isinstance(x, str) else []`.

---

### `src/probuyer_xai/llm.py`

Wraps DeepSeek via raw `requests.post` to the OpenAI-compatible `/chat/completions`
endpoint. The system prompt (`_SYSTEM_PROMPT`) is the governing document for LLM
behaviour and was written to close four specific audit gaps:

1. **Confidence-aware output** — if `confidence = "Low"` or `anomaly_type = "return_or_cancellation_anomaly"`,
   the LLM must lead with a caveat that this is NOT a probuyer pattern
2. **Language constraint** — always respond in English (data descriptions can be
   in any language)
3. **Scope disclaimer** — output is advisory; cannot replace human review
4. **Self-review step** — before responding, the LLM checks whether it cited any
   number not present in the evidence JSON

**Mock fallback** (`_mock_explanation`): deterministic template-based text built
from the evidence JSON. Used when `DEEPSEEK_API_KEY` is empty or the API call fails.
Three task variants: `"customer"`, `"summary"`, `"whatif"`.

**Four public functions:**
- `explain_customer(evidence)` — per-customer risk explanation
- `summarise_case(evidence)` — case study narrative
- `explain_whatif(whatif_result)` — what-if scenario interpretation
- `monthly_portfolio_summary(evidences)` — portfolio-level summary

`_call_llm` uses `max_tokens=400, temperature=0.3` for explanations.

---

### `src/probuyer_xai/calibrate.py`

The LLM-as-judge calibration loop. This is where the project goes beyond naive
hyperparameter tuning to use LLM domain knowledge as a model validation signal.

**Problem it solves:** HBOS is unsupervised — there is no ground truth to tell
you whether the model is flagging the right customers. The calibration loop uses
a pro LLM (DeepSeek-V3) as an independent judge to rate a sample of customers
on probuyer-likeness (1–5 scale, `temperature=0`), then measures agreement between
the LLM's rating and the model's risk band.

**Why `temperature=0`:** at zero temperature, the model is deterministic. A single
rating per customer is sufficient — no need to average multiple calls.

**Why a separate, stronger model for calibration:** the explanation model
(`deepseek-v4-flash`) is optimised for speed and cost. The calibration model
(`deepseek-chat`, DeepSeek-V3) is more capable and acts as an independent judge.
Using the same model for both explanation and calibration would reduce independence.

**Agreement definition:**
```
High band   → LLM score ≥ 4 to agree
Medium band → LLM score ≥ 3 to agree
Low band    → LLM score ≤ 2 to agree
```

**Calibration report** includes: per-band agreement rate, disagreement cases with
LLM reasoning, and specific threshold adjustment suggestions.

**Calibration history** is tracked in `calibration/history.md` (version-controlled,
not regenerated by the pipeline). It records what was found, what was changed, and why.

---

### `calibration/history.md`

The experiment log. See it for the full story of the two calibration iterations run
on this dataset. Summary:

- **Iteration 1** (RISK_MED_PCT=97.0): High 93%, Medium 90%, Low 20%. Precise at
  the top; some real signal leaking into the Low band.
- **Iteration 2** (RISK_MED_PCT=95.0): High 93%, Medium 60%, Low 60%. Expanding
  the Medium band overshot — the p95–p97 cohort is too noisy.
- **Decision:** kept 97.0. At 90% LLM agreement, Medium is an actionable reviewer
  queue. At 60%, it becomes a "maybe watch" tier with too many false positives.

---

### `src/probuyer_xai/whatif.py`

Five deterministic policy scenario functions. They operate in memory on the scored
customer DataFrames — no model retraining needed.

| Function | What it does |
|---|---|
| `change_risk_threshold(new_high, new_med)` | Re-assigns bands at new percentile cutoffs |
| `change_bulk_threshold(new_pct)` | Recomputes bulk features at a new invoice-quantity percentile |
| `exclude_cancellation_anomalies(...)` | Removes customers whose anomaly type is `return_or_cancellation_anomaly` |
| `whitelist_customer(customer_id, ...)` | Removes a specific customer from the flagged list |
| `require_min_rule_hits(n, ...)` | Filters flagged list to customers with ≥ n rule hits |

Each returns a dict: `{scenario, before_flagged_count, after_flagged_count, customers_added, customers_removed}`.
The LLM receives this dict and explains the business impact — it does not compute it.

---

### `src/probuyer_xai/reporting.py`

Generates `reports/case_studies.md` and `reports/llm_examples.md`.

`_pick_cases` selects three contrasting customers from the evidence list:
1. The highest-risk broad wholesale buyer (highest anomaly score with `broad_wholesale_buyer` type)
2. The top single-product bulk buyer (`single_product_bulk_buyer` type)
3. A cancellation anomaly (to demonstrate the fraud/return distinction)

The LLM writes the narrative for each case via `summarise_case`.
Reports are gitignored — they're regenerated on every pipeline run.

---

### `app/dashboard.py`

Five Streamlit pages via `st.sidebar.radio`. All loaders use `@st.cache_data` to
avoid re-reading parquets on every interaction. Artifacts are checked at startup
(`_check_artifacts`); if missing, a clear instruction to run the pipeline is shown
and `st.stop()` is called.

| Page | Key components |
|---|---|
| Overview | KPI metrics (total, High, Medium), anomaly score histogram (Plotly), anomaly type donut, top-20 flagged table |
| Customer Investigation | Customer dropdown (sorted by anomaly score), risk metrics, key metrics table, rule hits, LLM explanation, sample transactions |
| What-if Simulator | Scenario radio, sliders/inputs, before/after metrics, LLM interpretation |
| Case Studies | Renders `reports/case_studies.md` directly |
| Model Calibration | Agreement bar chart, LLM score distribution histogram, adjustment suggestions, disagreement case table |

The Calibration page loads `data/processed/calibration_report.json` via
`calibrate.load_report()`. If no report exists, it shows an instruction to run
`scripts/07_calibrate_model.py`.

---

### `src/probuyer_xai/pipeline.py` + `scripts/06_run_all.py`

`pipeline.run_all()` orchestrates all 8 steps with lazy imports (each step's module
is imported inside the step block, not at module load time). This means the pipeline
starts fast and fails at the right step if a dependency is missing.

```
[1/8] Download data
[2/8] Clean + save parquets
[3/8] Build customer features
[4/8] Train HBOS, assign risk bands
[5/8] Apply business rules
[6/8] Build explanation evidence  → saves customer_evidence.json
[7/8] Generate case studies
[8/8] LLM-as-judge calibration    → saves calibration_report.json
```

Step 1 fails gracefully: if the download fails, it prints the exact manual path and
exits with a clear message rather than raising an exception mid-pipeline.

`scripts/06_run_all.py` is just a four-line wrapper that imports `run_all` and calls it.

---

## Tests

44 tests across 6 files. All synthetic — no UCI dataset, no LLM key, no network.

| File | Covers |
|---|---|
| `test_data.py` | Column standardisation, amount calculation, cancellation detection |
| `test_features.py` | One row per customer, no inf/NaN in model features, ratios in [0,1] |
| `test_rules.py` | Rules JSON loads correctly, rule application columns, synthetic rule trigger |
| `test_explain.py` | Required evidence fields, confidence logic, anomaly type classification |
| `test_whatif.py` | Threshold change, whitelist, min-rule-hits filter |
| `test_calibrate.py` | Agreement logic, mock rating, report save/load round-trip, suggestion generation |

---

## Key design decisions

**1. LLM explains, DS decides.** The LLM never sees raw features or model scores
before a human-readable evidence JSON is built. The risk band in that JSON is a
string ("High", "Medium", "Low") computed by percentile cutoffs — not something the
LLM can influence.

**2. Cancellation ≠ probuyer.** Daigou buyers buy in bulk and keep the goods.
High cancellation is a fraud or speculative-ordering pattern. This distinction is
enforced at three layers: `MODEL_FEATURES` (feature excluded), `rules/probuyer_rules.json`
(R005 uses `repeat_sku_ratio`, not `cancellation_ratio`), and `explain.py`
(cancellation anomalies get `confidence="Low"` and are routed to fraud/care team).

**3. Percentile thresholds, not magic constants.** Rule thresholds (p95, p99) are
computed from the live feature distribution each time rules are applied. If the
customer population shifts, thresholds shift with it. The only hardcoded percentiles
are `RISK_HIGH_PCT` and `RISK_MED_PCT` in config, and those were validated by
calibration (see `calibration/history.md`).

**4. Mock mode everywhere.** The LLM is optional. Every LLM call has a
deterministic fallback. The pipeline, dashboard, and tests all work without a
DeepSeek key.

**5. Rules as versioned JSON.** `rules/probuyer_rules.json` is committed to git.
You can see when a rule changed and why. The rule engine reads and applies the JSON
at runtime — no retraining needed to change a rule.

**6. Calibration as model validation.** Because HBOS is unsupervised, there is no
precision/recall to compute. Calibration fills that gap: a pro LLM acts as a domain
expert rating whether flagged customers actually look like probuyers. Two iterations
converged on `RISK_MED_PCT=97.0` as the operating point (90% LLM agreement on the
Medium band vs 60% at p95).

---

## Known limitations

- No real daigou labels — validation is LLM-assisted, not ground-truth-based
- UCI dataset is B2B-heavy; many "flagged" customers are legitimate wholesale buyers
- HBOS has no notion of time — seasonal buying spikes aren't distinguished from
  persistent wholesale behaviour
- Dashboard is a showcase only; no auth, no production deployment
- LLM calibration uses the same data the model trained on (no held-out set)
