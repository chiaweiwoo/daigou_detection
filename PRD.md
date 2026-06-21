# PRD.md — Product Requirements

## Problem statement

Retailers face risk from professional bulk buyers (daigou / probuyers) who:
- Create stockout risk for ordinary consumers
- May abuse rebate or loyalty programs
- Exploit limited-availability product launches

No ground-truth label exists for these buyers. We need a risk-flagging system based on observable purchasing behavior, with clear business justification that a human reviewer can act on.

## Target user

A retail analyst or business reviewer who needs to:
- Identify customers showing wholesale-like buying patterns
- Understand *why* a customer is flagged
- Explore "what if" policy changes before applying them

## User stories

1. As an analyst, I want to see a list of high-risk customers ranked by anomaly score so I can prioritise my review queue.
2. As an analyst, I want a plain-English explanation of why a specific customer is flagged so I can justify action.
3. As a policy manager, I want to simulate the effect of tightening or loosening the flagging threshold before applying changes.
4. As a portfolio reviewer, I want to see three representative case studies showing different probuyer-like patterns.

## In-scope

- Automated feature engineering from transactional data
- HBOS anomaly scoring (unsupervised)
- Percentile-based business rules (saved as JSON)
- Structured explanation evidence per customer
- LLM explanation in business language (with mock fallback)
- What-if scenario analysis (deterministic, LLM explains result)
- Streamlit showcase dashboard

## Out of scope

- Real-time scoring
- Production authentication / user management
- Ground-truth labels or supervised models
- Email / alerting integration
- Multi-tenant or multi-retailer support

## Acceptance criteria

- Pipeline runs end-to-end with one command
- Dashboard works without a DeepSeek API key
- Risk decisions are never made by the LLM
- Rules are saved as version-controlled JSON
- Model artifacts are reproducible from the same data

## Limitations

- No real company data; public UCI dataset only
- Wholesale buyers in this dataset are legitimate B2B customers, not actual daigou
- HBOS is unsupervised; no precision/recall metrics possible without labels
- LLM explanations depend on third-party API availability
