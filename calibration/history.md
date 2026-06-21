# Calibration History

This log records each calibration iteration: what the LLM-as-judge found, what
we decided to change, and why. It's the experiment notebook for the modelling
layer — not auto-generated, never overwritten by the pipeline.

---

## Iteration 1 — 2026-06-21

**Config at time of run**
- `RISK_HIGH_PCT = 99.0` (top 1% → High)
- `RISK_MED_PCT = 97.0` (top 1–3% → Medium)
- `CALIBRATION_MODEL = deepseek-chat` (DeepSeek-V3)

**Results**
| Band   | Sampled | Agreed | Agreement |
|--------|---------|--------|-----------|
| High   | 15      | 14     | 93%       |
| Medium | 10      | 9      | 90%       |
| Low    | 5       | 1      | 20%       |
| **Overall** | **30** | **24** | **80%** |

**What the LLM told us**

High and Medium bands are precise. The LLM independently confirmed 93% of High-risk
and 90% of Medium-risk customers as genuinely probuyer-like — strong validation that
the HBOS model is catching the right signal at the top.

The Low-band problem: 4 out of 5 sampled Low-band customers were rated 3–4 by the
LLM, meaning they look probuyer-like but sit just below the Medium percentile cutoff.
The 97th percentile threshold is conservative — there is real signal being left in the
Low band that the model isn't promoting to Medium.

The LLM also repeatedly cited cancellation/return patterns as disqualifiers in
disagreement cases. This is correct behaviour — those customers are caught by HBOS
on other features but the LLM correctly identifies them as non-probuyer. No action
needed on cancellation handling (already excluded from MODEL_FEATURES).

**Decision**

Lower `RISK_MED_PCT` from 97.0 to 95.0. This expands the Medium band from roughly
top 1–3% to top 1–5%, promoting borderline customers the LLM was flagging as
overlooked. High band stays at 99.0 (precision is already excellent there).

---

## Iteration 2 — 2026-06-21

**Config at time of run**
- `RISK_HIGH_PCT = 99.0` (unchanged)
- `RISK_MED_PCT = 95.0` (lowered from 97.0)
- `CALIBRATION_MODEL = deepseek-chat`

**Results**
| Band   | Sampled | Agreed | Agreement |
|--------|---------|--------|-----------|
| High   | 15      | 14     | 93%       |
| Medium | 10      | 6      | 60%       |
| Low    | 5       | 3      | 60%       |
| **Overall** | **30** | **23** | **76.7%** |

Medium band: 87 → 173 customers (86 new customers from the p95–p97 range).

**What the LLM told us**

High stayed at 93% — the top of the distribution is still clean and unaffected by
the threshold change. The High band is well-calibrated.

Medium dropped from 90% → 60%. The extra 86 customers pulled in from the p95–p97
range are noisier — only 60% pass the LLM's probuyer test. Lowering the threshold
to 95.0 overshot: it captured some real signal (we know this because Low improved)
but mixed it with customers who are just moderately heavy buyers, not wholesale-like.

Low improved from 20% → 60%. This confirms some genuine probuyer signal was promoted
from the Low band — the adjustment was directionally correct, just too aggressive.

**Decision**

The p95 cut is too wide. The right boundary is between 97.0 (too conservative, 90%
precision) and 95.0 (too loose, 60% precision). Rolled `RISK_MED_PCT` back to 97.0
and accepted the tradeoff: the Medium band is smaller but more trustworthy.

The business interpretation: treat Medium as a genuine "review this" tier. At 95.0
it would become a "maybe watch" tier — less actionable. 97.0 with 90% LLM agreement
is the right operating point for a reviewer's queue.

The Low-band signal (customers the LLM rates 3–4 who don't reach Medium) is a known
limitation of unsupervised percentile thresholds. A future improvement would be to
use the LLM's per-customer score as a soft label to train a supervised re-ranker.

---

## Standing configuration (post calibration)

| Parameter       | Value | Rationale |
|-----------------|-------|-----------|
| `RISK_HIGH_PCT` | 99.0  | 93% LLM agreement — don't change |
| `RISK_MED_PCT`  | 97.0  | 90% LLM agreement — sweet spot for actionability |
| `CALIBRATION_MODEL` | deepseek-chat | DeepSeek-V3; independent of the explanation model |
