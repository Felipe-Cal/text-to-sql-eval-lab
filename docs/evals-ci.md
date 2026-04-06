# Evals-as-CI (GitHub Actions)

Every push and pull request to `main` runs the eval suite automatically via GitHub Actions. The pipeline blocks merges if scores drop below defined thresholds — preventing prompt regressions the same way unit tests prevent code regressions.

## How it works

```
PR opened / push to main
        │
        ▼
GitHub Actions: eval.yml
        │
        ├── Install dependencies
        ├── Seed DuckDB
        └── python scripts/ci_eval.py
                │
                ├── Runs full eval (15 questions, zero_shot)
                ├── Compares each scorer against threshold
                │
                ├── All pass → ✅ CI green, PR can merge
                └── Any fail → ❌ CI red, merge blocked
```

## Thresholds

Set conservatively below the known zero_shot baseline to catch real regressions without being brittle to natural LLM variance:

| Scorer | Threshold | Baseline |
|---|---|---|
| `syntax_valid` | 0.90 | 1.00 |
| `execution_ok` | 0.80 | 0.933 |
| `result_match` | 0.60 | 0.733 |
| `semantic_judge` | 0.73 | 0.867 |

## Setup

Add your API keys as GitHub Actions secrets in your repo settings:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY` (optional, only needed if testing Anthropic models)

The workflow file is at `.github/workflows/eval.yml`. Eval logs are uploaded as artifacts and retained for 14 days — download them from the Actions tab and explore with:

```bash
python -m inspect_ai view start --log-dir ./logs
```

## Running the gate locally

```bash
# Default thresholds, zero_shot
python scripts/ci_eval.py

# Custom strategy and thresholds
python scripts/ci_eval.py --strategy few_shot_dynamic --result-match 0.90

# Exits 0 on pass, 1 on failure — safe to use in any shell script or CI system
```
