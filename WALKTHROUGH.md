# Technical Walkthrough — Retail Probuyer Risk Detection

This document is written for a senior engineer reading the codebase cold. It explains
what every module does line by line in terms of logic, not just intent. It covers the
decisions made, the trade-offs considered, and the things deliberately left out.

---

## 1. What this project is actually doing

The core problem: a retailer suspects that some customers are professional bulk buyers
(daigou / probuyers) who drain stock before ordinary consumers can buy. There is no
ground-truth label. You cannot train a classifier. You can only observe the purchasing
behaviour and flag customers whose pattern is statistically rare.

The solution is a two-layer system:

**Layer 1 — Anomaly detection (deterministic):** HBOS scores each customer purely from
their transactional features. The DS layer assigns a risk band. Five percentile-based
business rules independently check for specific extreme behaviours. These two signals —
model score and rule hits — are merged into a structured evidence dict per customer.
Nothing probabilistic or LLM-driven happens here.

**Layer 2 — Explanation (generative):** The evidence dict is serialised to JSON and sent
to DeepSeek. The LLM narrates it in business English. It does not score, rank, or decide
anything. If the API is unavailable, a deterministic mock template fills in.

Everything in layer 1 is reproducible from the same data. Layer 2 is cosmetic.

---

## 2. Repository layout and what each file owns

```
daigou_detection/
├── src/probuyer_xai/
│   ├── config.py        all paths, feature lists, thresholds, env loading
│   ├── data.py          download, load xlsx, clean, save parquets
│   ├── features.py      customer-level feature engineering (23 features)
│   ├── model.py         HBOS training, scoring, artifact persistence
│   ├── rules.py         rule engine: load JSON, compute thresholds, apply
│   ├── explain.py       deterministic evidence dict per customer (no LLM)
│   ├── llm.py           DeepSeek wrapper + mock fallback
│   ├── whatif.py        5 policy scenario functions (pure Python)
│   ├── reporting.py     case study selection and report generation
│   └── pipeline.py      orchestrates all 7 steps in sequence
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_prepare_data.py
│   ├── 03_train_model.py
│   ├── 04_generate_rules.py
│   ├── 05_generate_case_studies.py
│   └── 06_run_all.py    calls pipeline.run_all()
├── app/
│   └── dashboard.py     4-page Streamlit app
├── rules/
│   └── probuyer_rules.json   version-controlled rule definitions
├── data/
│   ├── raw/             Online Retail.xlsx (gitignored)
│   └── processed/       parquets: clean, normal, features, scores, rule_hits
├── models/              hbos_model.joblib, feature_scaler.joblib, metadata.json
└── tests/               28 pytest tests, all synthetic
```

---

## 3. Config (`src/probuyer_xai/config.py`)

Everything that would otherwise be a magic string or number lives here. This is the only
file that calls `load_dotenv()`, so `.env` is read exactly once at import time. All other
modules import constants from here — they never call `os.getenv` directly.

Key constants a senior should know:

- `ROOT` is resolved via `Path(__file__).resolve().parents[2]` — two levels up from
  `src/probuyer_xai/`, landing at the repo root. This makes paths work regardless of
  where you invoke the script from.

- `MODEL_FEATURES` is a hard list of 17 feature names. This list is the single source of
  truth for what goes into HBOS. If you add a feature to `features.py` but not here, it
  will not be used by the model. If you add it here but not in `features.py`, training
  will fail with a KeyError.

- `RISK_HIGH_PCT = 99.0` and `RISK_MED_PCT = 97.0` are the only thresholds that control
  the High/Medium/Low band assignment. They are percentile cutoffs on the anomaly score
  distribution. Top 1% = High, 1–3% = Medium, rest = Low.

- LLM credentials (`DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`) are read
  from env. If `DEEPSEEK_API_KEY` is empty, the LLM layer falls back to mock mode. The
  model name `deepseek-v4-flash` is kept exactly as specified — it is not validated.

---

## 4. Data layer (`src/probuyer_xai/data.py`)

### Download

`download_raw()` tries `urllib.request.urlretrieve` against the UCI ML repository URL.
It is intentionally simple — no retry logic, no progress bar. If it fails (timeout,
firewall, the UCI server being down), it returns `False` and the caller is responsible
for printing the manual instruction. It cleans up a partial file if the download failed
mid-stream.

### Loading

`load_raw()` reads the xlsx with `pd.read_excel`. Three columns are forced to `str` at
read time: `CustomerID`, `InvoiceNo`, and `StockCode`. This matters because:

- `CustomerID` in the raw file is stored as a float (e.g. 17850.0). Pandas would read
  it as float64, and `.astype(str)` would produce `"17850.0"`. Forcing `dtype=str` at
  read time gives `"17850"`.
- `StockCode` has both numeric codes (e.g. `71053`) and alphanumeric codes (e.g.
  `85123A`). If pandas infers the type, it reads numeric-looking rows as int and then
  pyarrow refuses to serialise the mixed-type column to parquet. The explicit `dtype=str`
  prevents this bug — we hit it once and fixed it here.
- `InvoiceNo` starts with `C` for cancellations but is otherwise numeric. Same issue.

After reading, `.str.strip()` is applied to all three to remove any trailing whitespace
from the xlsx.

### Cleaning

`clean()` produces two DataFrames from the same input:

**`transactions_clean`** — everything with a `customer_id`, with `amount` and
`is_cancelled` added. This is the complete dataset including returns and cancellations.
It is kept because cancellation features are computed from this frame, not from
`transactions_normal`.

**`transactions_normal`** — the subset where `is_cancelled` is False, `quantity > 0`,
and `unit_price > 0`. This is what flows into feature engineering for purchasing behaviour.

The `is_cancelled` flag is true if `invoice_no` starts with `"C"` OR `quantity < 0`.
Both conditions exist in the data: some cancellation rows use the `C` prefix, some use
negative quantities without the prefix. We need both checks to catch all of them.

The reason we keep the cancelled rows in a separate frame rather than just dropping them
is that the cancellation ratio is a meaningful signal — a customer who cancels a lot of
orders is either testing availability or is a return-prone bulk buyer. We compute that
feature from `transactions_clean` by cross-referencing the customer's normal purchase
count.

Real pipeline output: 540,455 raw rows → 406,829 with customer_id → 397,884 normal
purchases → 4,371 customers (4,338 after dropping those with NaN in model features).

---

## 5. Feature engineering (`src/probuyer_xai/features.py`)

This is the most logic-dense file. It takes the two clean DataFrames and produces one
row per customer with 23 columns. 17 of those go into the HBOS model. The other 6 are
kept for explanation and reporting.

### What each feature actually measures

**Volume features** (go into model):
- `total_quantity` — sum of all item quantities across all normal invoices
- `total_spend` — sum of `amount = quantity * unit_price`
- `num_invoices` — count of distinct invoice numbers (not row count)
- `active_days` — `(last_invoice_date - first_invoice_date).days + 1`, minimum 1

The `+1` in active_days prevents division by zero for customers who placed all their
orders on a single day. The `max(..., 1)` is the safety net.

**Per-invoice stats** (go into model):
- `avg_quantity_per_invoice`, `max_quantity_per_invoice`
- `avg_spend_per_invoice`, `max_spend_per_invoice`

These are computed at the invoice level first, then aggregated. The correct way to
compute these is to group by `(customer_id, invoice_no)` first, then sum quantity/amount
per invoice, then take the mean/max across invoices. If you grouped by `customer_id` and
took the mean of line-level quantities, you'd get the average line quantity, which is a
different number entirely (one invoice can have many line items). The code does this
correctly via `inv_qty_by_cust`.

**SKU diversity features** (go into model):
- `unique_skus` — number of distinct `stock_code` values
- `repeat_sku_ratio` — fraction of SKUs bought more than once:
  `(number of SKUs with purchase count > 1) / unique_skus`.
  Implemented by counting `(customer_id, stock_code)` rows (each row is a line item, so
  >1 row = bought on multiple invoices or multiple times on one invoice), then dividing
  by `unique_skus`. Clipped to [0, 1]. Filled to 0 for customers with no repeat SKUs
  rather than NaN.

**SKU concentration features** (go into model):
- `top_sku_quantity_share` — what fraction of the customer's total quantity came from
  their single highest-volume SKU. Computed as `max SKU quantity / total_quantity`.
  A customer who bought 10,000 units of one SKU and 200 units of everything else has a
  share close to 1.0.
- `top_sku_spend_share` — same logic but for spend. The two can differ if the top-volume
  SKU is cheap and the top-spend SKU is expensive.

**Rate features** (go into model):
- `quantity_per_active_day` = `total_quantity / active_days`
- `spend_per_active_day` = `total_spend / active_days`

These normalise volume by time window. A customer who bought 5,000 units over 2 years
is different from one who bought 5,000 units in 3 days.

**Bulk invoice features** (go into model):
- Bulk threshold is the **95th percentile of invoice-level total quantity** across all
  invoices in the dataset, not just one customer's invoices. This gives a dataset-wide
  reference point. If the p95 invoice-quantity is 778 units, any invoice above that is
  "bulk". This threshold is stored in `feature_metadata.json` so rules can reference it
  later.
- `bulk_invoice_count` — number of the customer's invoices that exceed the threshold.
  Customers with no bulk invoices get 0 (via `reindex(..., fill_value=0)`).
- `bulk_invoice_ratio` = `bulk_invoice_count / num_invoices`. Clipped to [0, 1].

**Cancellation features** (NOT in model, kept for explanation):
- `cancelled_invoice_count`, `cancelled_quantity_abs`, `cancelled_amount_abs` — pulled
  from `transactions_clean` by filtering `is_cancelled == True` and grouping by
  `customer_id`. Customers with no cancellations get 0 via `fillna(0)`.
- `cancellation_ratio` = `cancelled_invoice_count / num_invoices`. Note the denominator
  is the normal purchase invoice count, not a total. A customer with 10 normal invoices
  and 4 returns has ratio 0.4.

`cancellation_ratio` is in `MODEL_FEATURES` even though the raw cancellation counts are
not. The ratio is scale-invariant; the raw counts are not.

**Date bookmarks** (not in model, kept for context):
- `first_purchase_date`, `last_purchase_date` — used in dashboard's transaction table.

### Safety nets

After assembly, two things happen:
1. `replace([inf, -inf], np.nan)` — rate features can produce inf if `active_days` or
   `num_invoices` somehow ends up at zero. The `max(..., 1)` in active_days should
   prevent this, but it's defensive.
2. `dropna(subset=MODEL_FEATURES)` — drops the 33 customers who ended up with NaN in
   any model feature. In practice these are customers in `transactions_normal` but not in
   the groups used to compute some features (edge case in the `cancellation_ratio`
   join). Logged to stdout.

---

## 6. HBOS model (`src/probuyer_xai/model.py`)

### Why HBOS

HBOS (Histogram-Based Outlier Score) works by building a histogram for each feature
independently, then scoring each data point by summing the negative log-densities of the
bins it falls into. Points in sparse bins (rare combinations) get high scores.

Three reasons it was chosen over alternatives:

1. **No label dependency.** There are no ground-truth daigou labels. HBOS is fully
   unsupervised.
2. **Handles extreme right skew.** Retail purchasing features (total quantity, spend)
   follow power-law distributions. HBOS captures this: a customer who bought 80,000 units
   falls into a very sparse bin and gets a high score without any tuning.
3. **Feature independence assumption.** HBOS scores features independently and sums the
   log-densities. This is a simplification (features are correlated), but for the flagging
   use case it is acceptable. A more correct model would be Isolation Forest or LOF, but
   HBOS is simpler to explain to a business audience.

### Preprocessing pipeline

1. **`log1p` transform.** All model features are non-negative and right-skewed. Taking
   `log(1 + x)` compresses the scale so that the difference between 100 and 1,000 units
   is treated comparably to the difference between 1 and 10. Without this, the dense
   low-value region would dominate the histograms.

2. **`RobustScaler`.** Subtracts the median and divides by the IQR. Unlike `StandardScaler`,
   it is not pulled by extreme outliers (which are exactly what we're looking for).
   The scaler is fitted here and saved separately — it must be applied consistently to
   any new data scored against this model.

3. **`HBOS()`.** Default hyperparameters from pyod. No contamination fraction set —
   contamination controls what fraction of the training set is assumed to be anomalous,
   and we do not know this. Risk bands are assigned post-hoc via percentile cutoffs.

### Risk band assignment

`_compute_risk_percentile()` computes for each customer the fraction of all scores that
are <= their score, scaled to 0–100. This is the empirical CDF of the anomaly score
distribution. The top customer has percentile 100.0.

`_assign_risk_band()` takes those percentile values as a numpy array and returns a list
of strings. **Why a plain list and not a pandas Series with an index?** There was a bug
in the first implementation where `_assign_risk_band` returned a Series indexed by
`customer_id`, but `scores` was indexed by integer position. When pandas tried to assign
it, the index mismatch silently filled everything with NaN, producing 0 High and 0 Medium
customers. The fix was to operate on numpy arrays, bypassing index alignment entirely.

### Artifacts saved

- `hbos_model.joblib` — the fitted HBOS instance
- `feature_scaler.joblib` — the fitted RobustScaler
- `customer_scores.parquet` — one row per customer with `customer_id`, `anomaly_score`,
  `risk_percentile`, `risk_band`
- `model_metadata.json` — training date, feature list, thresholds, counts. Useful for
  auditing whether the loaded model matches the current feature set.

---

## 7. Business rules (`rules/probuyer_rules.json` + `src/probuyer_xai/rules.py`)

### Why rules exist alongside the model

The HBOS score is a single number. It tells you how anomalous a customer is but not
which dimension is anomalous. A customer could score high because they buy a lot
(quantity), buy one thing (SKU concentration), cancel a lot (returns), or some
combination. The business rules make the specific reason explicit and version-controllable.

The rules are stored in `rules/probuyer_rules.json` and checked into git. This means:
- The rule logic is auditable and reviewable without touching Python code.
- Rules can be rerun against existing features without retraining the model.
- A business analyst can propose a rule change as a JSON diff.

### How the rule engine works

`apply_rules()` iterates over the 5 rules in the JSON. For each rule:

1. Look up the feature column (e.g. `total_quantity`).
2. Resolve `threshold_source` (e.g. `"p99"`) to an integer percentile via `_PCT_MAP`.
3. Compute `np.percentile(features[feature].dropna().values, pct)` — the threshold is
   derived fresh from the current feature distribution, not hardcoded. This means
   thresholds auto-adjust if you retrain on a different dataset slice.
4. Apply the operator (`>=`, `>`, etc.) via Python's `operator` module to the full
   feature column. Returns the index of matching customers.
5. Append the rule ID and reason string to each matching customer's hit record.

Output: one row per customer with `rule_hits` (list of rule IDs), `rule_reasons` (list
of human-readable strings), `rule_count` (integer). Customers with zero hits have empty
lists.

### The 5 rules and their threshold levels

| Rule | Feature | Threshold | Why this level |
|------|---------|-----------|----------------|
| R001 | `total_quantity` | p99 | Only the most extreme total buyers — p95 would flag 200+ customers |
| R002 | `max_quantity_per_invoice` | p99 | One enormous basket is a strong signal |
| R003 | `bulk_invoice_ratio` | p95 | p99 for a ratio feature is too restrictive |
| R004 | `top_sku_quantity_share` | p95 | SKU concentration is common in B2B, p99 was too tight |
| R005 | `cancellation_ratio` | p95 | High cancellations are common; p95 catches genuine anomalies |

---

## 8. Explanation layer (`src/probuyer_xai/explain.py`)

This layer runs entirely in Python. No LLM, no randomness. It produces a structured
`dict` per customer that the LLM then narrates.

### Anomaly type classification (`_classify_anomaly_type`)

Uses p90 thresholds (computed from the live feature distribution) as internal reference
points. The classification logic has a **priority order**:

1. `return_or_cancellation_anomaly` — checked first. If cancellation ratio is above p90,
   this type is assigned regardless of other signals. This is intentional: cancellation
   anomalies are a distinct pattern and should be clearly separated from pure bulk buyers.
2. `single_product_bulk_buyer` — high SKU concentration (>=0.7) combined with high
   quantity OR high max invoice quantity.
3. `broad_wholesale_buyer` — high quantity AND high spend but NOT high SKU concentration.
   This is the classic B2B pattern: many different SKUs in large volumes.
4. `high_frequency_buyer` — many invoices relative to the population.
5. `potential_stockout_risk` — high max invoice or high SKU concentration alone.
6. Default fallback: `broad_wholesale_buyer`.

The 0.7 hardcoded threshold for high SKU concentration in the anomaly type logic is
separate from the R004 rule threshold (p95 of `top_sku_quantity_share`). The p95
threshold controls the rule hit; 0.7 is a human-readable "mostly one SKU" cutoff for
labelling the anomaly type. They serve different purposes.

### Confidence scoring (`_confidence`)

| Condition | Confidence |
|-----------|------------|
| High band + ≥2 rule hits | High |
| High or Medium band + ≥1 rule hit | Medium |
| Anything else | Low |

A customer can have a High risk band from the model but Low confidence if no business
rule independently confirms the pattern. This is intentional: the model may flag a
customer for reasons that aren't cleanly captured by the 5 rules. Low confidence with
High band means "worth a look, but weaker evidence."

### Evidence dict structure

```json
{
  "customer_id": "15749",
  "risk_band": "High",
  "risk_percentile": 100.0,
  "anomaly_score": 0.8421,
  "anomaly_type": "broad_wholesale_buyer",
  "confidence": "High",
  "top_reasons": ["Customer purchased unusually high total quantity.", ...],
  "rule_hits": ["R001", "R002", "R003"],
  "rule_count": 3,
  "key_metrics": {
    "total_quantity": 18028.0,
    "total_spend": 27549.65,
    "num_invoices": 3,
    "max_quantity_per_invoice": 9014.0,
    "bulk_invoice_ratio": 1.0,
    "top_sku_quantity_share": 0.5,
    "cancellation_ratio": 0.667
  },
  "recommended_action": "Flag for business review. ..."
}
```

This dict is the interface between the DS layer and the LLM. The LLM is only ever given
this dict — never the raw feature matrix or the anomaly score directly.

---

## 9. LLM layer (`src/probuyer_xai/llm.py`)

### Transport

`_call_llm()` makes a raw `requests.post` to `{DEEPSEEK_BASE_URL}/chat/completions` —
the OpenAI-compatible endpoint. Body is the standard chat completions format with a
`system` and `user` message. Response is `resp.json()["choices"][0]["message"]["content"]`.

Key parameters: `max_tokens=400`, `temperature=0.3`. Low temperature keeps the output
consistent across runs (important for reproducibility of case study reports).

### System prompt design

The system prompt has six explicit constraints:
1. Only reference provided JSON values — prevent hallucination.
2. No accusation — say "flagged for review," not "guilty of daigou."
3. "Probuyer-like" / "wholesale-like" language only — never "confirmed daigou."
4. Always acknowledge uncertainty.
5. 3–5 sentences unless asked for more.
6. Plain business English.

These constraints exist because an LLM given a customer's anomaly score and no guardrails
will often write with false certainty ("this customer is definitely a daigou reseller").
The system prompt forces hedge language. The prompt injection risk is low because the
input is structured JSON from our own pipeline, not user-supplied text.

### Mock fallback

`_mock_explanation()` is called when `DEEPSEEK_API_KEY` is empty or the HTTP call raises
any exception. It produces a templated string from the evidence dict's values — no API
call, no randomness. The dashboard never crashes because of a missing key.

The three mock task types are `"customer"` (individual explanation), `"summary"` (case
study), and `"whatif"` (policy scenario). Each has its own template that pulls different
fields from the evidence dict.

### Four LLM functions

- `explain_customer(evidence)` — explain why one customer is flagged. Used in dashboard
  Customer Investigation page on demand.
- `summarise_case(evidence)` — write a case study summary. Used by `reporting.py` at
  pipeline run time.
- `explain_whatif(whatif_result)` — narrate the result of a policy simulation. Used in
  dashboard What-if Simulator page.
- `monthly_portfolio_summary(evidences)` — takes the full evidence list, computes the
  breakdown by band and anomaly type in Python, sends the summary stats to the LLM. Not
  currently wired into the dashboard but available as a library function.

---

## 10. What-if analysis (`src/probuyer_xai/whatif.py`)

### Core abstraction

All 5 scenario functions call the same private function `_flagged_ids()`, which walks
the scores DataFrame and the evidence list, applying filters in this order:

1. Percentile threshold: skip if `risk_percentile < med_pct`.
2. Whitelist: skip if customer is in the whitelist set.
3. Anomaly type filter: skip if `exclude_cancellation=True` and anomaly type is
   `return_or_cancellation_anomaly`.
4. Min rule hits: skip if `rule_count < min_rule_hits`.

Each public function calls `_base_flagged()` (defaults, no filters) to get the "before"
set, then calls `_flagged_ids()` with adjusted parameters to get the "after" set. The
`_result()` helper computes the symmetric difference: `customers_added = after - before`,
`customers_removed = before - after`.

This design means every scenario has identical cost (one pass over scores + evidence)
and the result is always a precise diff — not an approximation.

### Note on `change_bulk_threshold`

This function is more complex than the others because the bulk threshold affects a
feature that was computed at training time. A proper re-threshold would require
re-running `features.py` with a new `_BULK_QTY_PCT`. The current implementation
approximates this: it re-derives `bulk_invoice_ratio` using `max_quantity_per_invoice`
as a proxy, then intersects with customers already at Medium+ risk. This is a
deliberate simplification noted in the code comment. For a full re-threshold scenario,
you'd re-run the pipeline from step 3.

---

## 11. Case studies (`src/probuyer_xai/reporting.py`)

### Selection logic

`_pick_cases()` tries to find three structurally different customers:

**Case 1:** `anomaly_type == "broad_wholesale_buyer"` AND `confidence == "High"`.
Fallback: any High-band customer. Then take the one with the highest risk percentile.

**Case 2:** `anomaly_type == "single_product_bulk_buyer"`. Fallback: High or Medium band
AND `top_sku_quantity_share >= 0.6`. Then take the highest risk percentile.

**Case 3:** `anomaly_type == "return_or_cancellation_anomaly"`. Fallback: any High or
Medium band customer with Low confidence (anomaly detected but weak probuyer pattern).

If the same customer would appear in two cases, the fallback loop picks the next best
unselected customer from the ranked-by-percentile list. The `used_ids` set prevents
duplicates.

### Why three specific types

The three cases demonstrate that "anomaly" does not mean "probuyer." Case 3 is
deliberately a customer who has a high anomaly score but a different underlying pattern
(return abuse, speculative ordering). This is a key part of the portfolio story: the
system distinguishes between types of outliers and assigns confidence accordingly.

---

## 12. Pipeline orchestration (`src/probuyer_xai/pipeline.py`)

`run_all()` imports each module inside the function body, not at module level. This is
intentional: it means `import probuyer_xai.pipeline` does not import pandas, pyod,
joblib, etc. at import time. Only the step that is currently running needs its imports.
This has a minor effect on startup time and a meaningful effect on making tests fast —
test files that import from `pipeline.py` don't drag in the full ML stack.

The pipeline stops cleanly at step 1 if the dataset is missing (returns `False`, prints
exact instructions). All other steps are expected to succeed — there is no partial
recovery beyond that point.

---

## 13. Dashboard (`app/dashboard.py`)

### Data loading

All four data loaders (`_load_features`, `_load_scores`, `_load_rule_hits`,
`_load_normal`) are decorated with `@st.cache_data`. This means Streamlit only reads
from disk on the first render per session. Subsequent page navigations use the cached
result. The evidence list is NOT cached because `_build_evidences()` calls
`build_evidence()` from `explain.py`, which does CPU-bound work each call. It's fast
enough (~0.5s) that caching it would complicate invalidation.

### Artifact guard

`_check_artifacts()` verifies the three required parquets exist before rendering any
page. If any is missing, the entire app stops with a clear message and the exact command
to run. This prevents confusing KeyError or FileNotFoundError messages that Streamlit
would otherwise surface as a red stack trace.

### Page 1 — Overview

Reads scores directly, computes High/Medium counts, plots a Plotly histogram coloured
by risk band. Calls `_build_evidences()` to get anomaly types for the donut chart.
Top-20 table merges `scores` with `rule_hits` on `customer_id` to show rule hit count
alongside the anomaly score.

### Page 2 — Customer Investigation

Selector is pre-filtered to High and Medium band customers only — you cannot select a
Low-risk customer for investigation. The LLM explanation is called on-demand when the
page renders for a given customer. It uses `st.spinner` to block the UI while waiting.
Sample transactions are pulled from `transactions_normal.parquet` filtered by
`customer_id`.

### Page 3 — What-if Simulator

Uses `st.radio` to select the scenario type, then renders scenario-specific controls
(sliders, text input) below. The "Run scenario" button triggers the Python computation
and LLM explanation. The split-column layout (controls left, results right) means the
user can see controls and results simultaneously.

A subtle UX decision: the `wl_cid` variable (customer ID to whitelist) is only defined
in the `elif scenario == "Whitelist a customer"` branch, but the `st.button` block
references it. Streamlit re-renders from top to bottom each interaction, so by the time
the button block runs, the variable either exists from the branch above or is not
referenced (because the scenario is different). This is fine in Streamlit's execution
model but would be a bug in linear Python.

### Page 4 — Case Studies

Reads `reports/case_studies.md` and renders it with `st.markdown()`. The markdown file
is generated by `reporting.py` and contains both the structured evidence table and the
LLM explanation inline. If the file doesn't exist (pipeline not run), it shows an error
with the exact command.

---

## 14. Tests (`tests/`)

28 tests, all using synthetic DataFrames. No test touches the disk, the network, or the
LLM API. Each test file corresponds to one source module.

**test_data.py (6 tests):** Constructs a 4-row synthetic DataFrame with one cancellation
(C-prefix) and one negative-quantity row. Tests that: columns are standardised, amount
is correct, the C-prefix and negative-qty rows are both flagged as cancelled, and
customers with null `customer_id` are dropped.

**test_features.py (6 tests):** Constructs a 6-row DataFrame with 2 customers. Tests
one-row-per-customer, no infinities, non-negative bulk count, ratios in [0,1], no NaN
in model features, and that metadata contains the bulk threshold.

**test_rules.py (6 tests):** Uses inline rule definitions (not the file on disk) and a
50-row synthetic feature DataFrame. Tests that the rule engine returns expected columns,
that at least one rule triggers on realistic data, that hits are lists, and that a
deliberately extreme value triggers R001.

**test_explain.py (5 tests):** Builds synthetic features/scores/rule_hits for 20
customers. Tests required dict fields, the confidence scoring function directly, anomaly
type validity, customer lookup, and None return for a missing customer.

**test_whatif.py (5 tests):** Builds 20 synthetic customers with known risk percentiles
and rule counts. Tests that tightening the threshold reduces the count, cancellation
exclusion removes the right customers, whitelisting a specific customer removes it from
the result and lists it in `customers_removed`, and that the result dict has all required
keys.

---

## 15. What we deliberately did not build

**No ground-truth labels.** The UCI dataset has no daigou labels and no way to validate
precision/recall. HBOS is unsupervised. The "accuracy" of this system cannot be measured
in the traditional ML sense. It is an anomaly detection and business review tool, not a
classifier.

**No Jupyter notebooks.** They would overlap with the scripts pipeline and become stale.
The `reports/*.md` files carry the data narrative and are generated fresh on each run.

**No real-time scoring.** The pipeline is batch. New customers would need to re-run
from step 3 (features) through step 6 (evidence). The model artifacts are saved and
reusable, but there is no API endpoint.

**No supervised rule tuning.** The 5 rules use fixed percentile levels (p99, p95).
In production you would calibrate these against confirmed flagged customers. We have none.

**No multi-retailer or multi-tenant support.** One dataset, one model, one rule set.

**No auth on the dashboard.** It's a showcase tool, not a production app.

---

## 16. Key design invariant

**The DS/rule layer decides risk. The LLM only explains.**

Every function that makes a risk decision — `_assign_risk_band`, `_classify_anomaly_type`,
`_confidence`, `apply_rules`, `_flagged_ids` — is in Python. None of them call the LLM.
The LLM receives a completed evidence dict. It can only produce text from it.

This means the system is:
- **Reproducible** — same data, same risk output, every time.
- **Auditable** — you can inspect the exact evidence dict that the LLM was given.
- **Resilient** — if DeepSeek is down, the risk decision is still made correctly. Only
  the explanation text changes (to mock mode).
- **Not misleading** — the LLM cannot produce a different risk band than the one
  computed by Python. It cannot say a Low-risk customer is High-risk.
