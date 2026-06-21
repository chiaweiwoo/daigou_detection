# AGENTS.md — Instructions for AI Coding Agents

## Core rules

1. Do not ask unnecessary questions. Make reasonable defaults and proceed.
2. Pause only for: missing DeepSeek API key (real output), dataset not downloadable, or an unresolvable command failure.
3. When blocked, provide: what failed, exact file needed, exact command to retry, exact `.env` variable.

## Environment

- Use `uv` for all Python operations (`uv sync`, `uv run pytest`, `uv run python ...`).
- Python version: 3.11 (pinned via `uv python pin 3.11`).
- Never use `pip` directly.

## Architecture principles

- The **DS/rule layer decides risk**. The LLM only explains structured evidence.
- Keep the DS layer **deterministic and reproducible**. No randomness in risk decisions.
- Anomaly scores come from HBOS; risk bands come from fixed percentile thresholds.
- Business rules are saved as JSON and can run without retraining.

## LLM

- Use mock explanation mode when `DEEPSEEK_API_KEY` is missing or the call fails.
- Never crash the dashboard because the LLM is unavailable.
- Never hallucinate metrics. Only use values from the provided evidence JSON.
- Say "probuyer-like" or "wholesale-like," never "confirmed daigou."

## Code style

- Type hints where reasonable.
- Small, single-purpose functions.
- Readable over clever.
- No unnecessary abstractions.

## Testing

- Tests must run without the UCI dataset or a live LLM.
- Use small synthetic DataFrames.
- Run: `uv run pytest`

## Implementation order

Follow the phase order in CLAUDE.md. Do not build the dashboard before the data pipeline works.
