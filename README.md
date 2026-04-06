# text-to-sql-eval-lab

An evaluation lab for benchmarking LLMs on the text-to-SQL task. Given a natural language question and a database schema, models are asked to generate a SQL query. The lab measures syntax validity, execution correctness, result accuracy, and semantic correctness via an LLM-as-judge — against a golden dataset.

The project ships as a **FastAPI service** so the agent and eval suite can be consumed by any frontend, CI pipeline, or external tool via HTTP, not just as a local script.

## How it works

```
Natural language question + schema
            │
            ▼
      LLM (via LiteLLM)
            │
     Self-correction loop (up to 3 retries on SQL error)
            │
            ▼
      Generated SQL
            │
     ┌──────┼──────┬──────────┐
     ▼      ▼      ▼          ▼
 syntax  execute  result   semantic
  valid    ok     match     judge
```

1. A **question** from the golden dataset is sent to the model along with the database schema.
2. The model returns a SQL query.
3. Four **scorers** evaluate the query independently, in increasing order of sophistication:
   - `syntax_valid` — parses the SQL with [sqlglot](https://github.com/tobymao/sqlglot); no database needed.
   - `execution_ok` — runs the query against the real DuckDB database; checks it doesn't error.
   - `result_match` — executes and compares returned rows to golden expected rows (1.0 = exact, 0.5 = right shape, 0.0 = wrong).
   - `semantic_judge` — sends question, generated SQL, actual results, and expected results to a judge LLM for a structured verdict.
4. **Self-correction loop**: on SQL error, the agent feeds the error back into its conversation and retries up to 3 times.

The eval harness is [Inspect AI](https://inspect.aisi.org.uk/), which handles parallelism, logging, and the `inspect view` log explorer.

---

## Dataset

- **Golden suite:** 15 hand-crafted Q&A pairs in `datasets/golden/questions.json`.
- **Synthetic flywheel:** `scripts/generate_synthetic.py` generates and validates synthetic questions, split into `tuning.json` (80%) and `holdout_test.json` (20%).

**Schema**

```
customers(id, name, email, country, signup_date)
products(id, name, category, price)
orders(id, customer_id, order_date, status)       -- status: completed | pending | cancelled
order_items(id, order_id, product_id, quantity, unit_price)
```

The schema is wrapped inside a **50-table enterprise data warehouse** simulation (46 decoy tables from HR, logistics, finance, marketing) to stress-test schema linking via the `rag` strategy.

**Difficulty breakdown**

| Difficulty | Questions | Patterns tested |
|---|---|---|
| Easy | q01–q05 | COUNT, DISTINCT, simple filters, ORDER BY |
| Medium | q06–q10 | JOINs, aggregations, subqueries, date functions |
| Hard | q11–q15 | Anti-joins (NOT IN), HAVING, time-series, multi-table JOINs |

---

## Setup

**Requirements:** Python 3.11+

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill in API keys
```

Key `.env` variables:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `DEFAULT_MODEL` | LiteLLM model string (e.g. `openai/gpt-4o-mini`) |
| `DATABASE_PATH` | Path to DuckDB file |
| `JUDGE_MODEL` | Model used by the semantic_judge scorer |
| `EMBEDDING_MODEL` | Model used for few_shot_dynamic and RAG strategies |
| `HARD_MODEL` | Escalation model for hard questions in the router (optional) |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Langfuse observability (optional) |

The database is seeded automatically on first run. To reseed manually: `python src/utils/db.py`.

---

## Quick start

```bash
# Run evals (default: gpt-4o-mini, zero_shot, all 15 questions)
python scripts/run_eval.py

# Recommended strategy
python scripts/run_eval.py --strategy few_shot_dynamic

# Gemini alias is accepted and normalized to the Inspect API prefix
python scripts/run_eval.py --model gemini/gemini-2.0-flash --strategy few_shot_dynamic

# Start the API server
uvicorn src.api.main:app --reload

# Run tests
pytest
```

---

## Going deeper

| Topic | Doc |
|---|---|
| Prompt strategies (zero_shot, few_shot_dynamic, rag, etc.) | [docs/strategies.md](docs/strategies.md) |
| FastAPI endpoints (`/query`, `/evals/run`) | [docs/api.md](docs/api.md) |
| Fine-tuning Llama 3.1 8B with LoRA | [docs/fine-tuning.md](docs/fine-tuning.md) |
| Model router (difficulty classification) | [docs/routing.md](docs/routing.md) |
| DSPy prompt optimization | [docs/dspy.md](docs/dspy.md) |
| Evals-as-CI (GitHub Actions gate) | [docs/evals-ci.md](docs/evals-ci.md) |
| Guardrails (input/output, adversarial tests) | [docs/guardrails.md](docs/guardrails.md) |
| Experimental findings and benchmark results | [docs/findings.md](docs/findings.md) |

---

## Project structure

```
text-to-sql-eval-lab/
├── .github/workflows/eval.yml       # CI eval gate
├── datasets/
│   ├── ecommerce.duckdb             # DuckDB database (auto-created)
│   ├── golden/questions.json        # 15 golden Q&A pairs
│   ├── synthetic/                   # tuning.json + holdout_test.json
│   └── dspy_optimized_prompt.json   # output of optimize_prompt.py
├── docs/                            # detailed documentation
├── notebooks/finetune_llama.ipynb   # Colab LoRA fine-tuning notebook
├── scripts/
│   ├── run_eval.py                  # CLI eval runner
│   ├── generate_synthetic.py        # synthetic data flywheel
│   ├── optimize_prompt.py           # DSPy BootstrapFewShot optimizer
│   ├── ci_eval.py                   # CI gate script
│   ├── benchmark_routing.py         # routing vs. fixed-strategy comparison
│   └── prepare_finetune_data.py     # data prep for LoRA fine-tuning
├── src/
│   ├── api/                         # FastAPI service
│   ├── agent/                       # LLM call, strategies, few-shot, RAG, router
│   ├── evals/                       # Inspect AI tasks and scorers
│   ├── guardrails/                  # input + output guardrails
│   └── utils/db.py                  # DuckDB connection and seeding
└── tests/                           # pytest — guardrails + adversarial
```

---

## Key dependencies

| Package | Role |
|---|---|
| [fastapi](https://fastapi.tiangolo.com) | HTTP service layer |
| [inspect-ai](https://inspect.ai) | Eval harness — tasks, solvers, scorers, logging |
| [litellm](https://docs.litellm.ai) | Unified interface to OpenAI, Anthropic, and other providers |
| [duckdb](https://duckdb.org) | In-process SQL engine |
| [sqlglot](https://github.com/tobymao/sqlglot) | SQL parser for syntax validation and output guardrails |
| [dspy-ai](https://dspy.ai) | Prompt optimization |
| [langfuse](https://langfuse.com) | Observability and tracing (optional) |
