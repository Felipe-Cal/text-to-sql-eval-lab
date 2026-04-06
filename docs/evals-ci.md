# Evals-as-CI (GitHub Actions)

Every push and pull request to `main` runs the eval suite automatically via GitHub Actions. The pipeline blocks merges if scores drop below defined thresholds — preventing prompt regressions the same way unit tests prevent code regressions.

## Why eval-as-CI?

Prompt engineering and LLM behaviour are surprisingly fragile. A small change to the system prompt, the schema string format, or an example in the few-shot set can silently drop `result_match` by 10–20%. Without automated eval gates, these regressions only show up in production.

Evals-as-CI treats the eval harness as a test suite: green = safe to merge, red = regression detected, investigate before merging.

## How it works

```
PR opened / push to main
        │
        ▼
GitHub Actions: .github/workflows/eval.yml
        │
        ├── Install dependencies (pip install -e ".[dev]")
        ├── Seed DuckDB database (python src/utils/db.py)
        └── python scripts/ci_eval.py
                │
                ├── Runs full eval: 15 questions, zero_shot strategy
                ├── Reads scores from Inspect AI results
                ├── Compares each scorer against its threshold
                │
                ├── All pass → exits 0 → ✅ CI green, PR can merge
                └── Any fail → exits 1 → ❌ CI red, merge blocked
```

## Thresholds

Set conservatively below the known `zero_shot` baseline — loose enough to tolerate natural LLM variance (~5–10%), tight enough to catch real regressions:

| Scorer | Threshold | Observed baseline | Buffer |
|---|---|---|---|
| `syntax_valid` | 0.90 | 1.00 | 10% |
| `execution_ok` | 0.80 | 0.933 | 13% |
| `result_match` | 0.60 | 0.733 | 17% |
| `semantic_judge` | 0.73 | 0.867 | 14% |

**Why different buffers?** `syntax_valid` is very stable (almost always 1.0) so a tight threshold is safe. `result_match` has more variance because exact row comparison is stricter — the 17% buffer avoids false alarms on questions where the LLM produces semantically correct but slightly different SQL.

## Setup

Add your API keys as GitHub Actions secrets:

- `OPENAI_API_KEY` — required (used for model calls and `semantic_judge`)
- `ANTHROPIC_API_KEY` — optional (only needed if the CI eval uses Anthropic models)

The workflow file is at `.github/workflows/eval.yml`. Eval logs are uploaded as workflow artifacts and retained for 14 days. To explore them locally:

```bash
# Download the artifact from the Actions tab, then:
python -m inspect_ai view start --log-dir ./logs
```

## Running the gate locally

Run `ci_eval.py` locally before pushing to check whether your changes pass:

```bash
# Default: zero_shot strategy, default thresholds
python scripts/ci_eval.py

# Custom strategy — useful when developing a new strategy
python scripts/ci_eval.py --strategy few_shot_dynamic

# Custom thresholds — tighten when you know your changes improved quality
python scripts/ci_eval.py --strategy few_shot_dynamic --result-match 0.90

# Exits 0 on pass, 1 on failure
# Safe to use in any shell pipeline: python scripts/ci_eval.py && echo "ready to push"
```

## What CI does NOT check

- **`tool_use` strategy** — agentic runs take too long and cost too much for per-commit CI. Run `benchmark_agent.py` manually before merging changes to `agent/tools.py`.
- **RAG strategies** — require embedding API calls per question × 15 questions. Run `run_eval.py --strategy rag_hybrid` manually before merging changes to `schema_retriever.py`.
- **Model regressions across providers** — CI only tests `DEFAULT_MODEL`. Changes to Anthropic or Gemini model behaviour won't be caught.

For changes to areas not covered by CI, run the relevant benchmark script manually and review the output before merging.
