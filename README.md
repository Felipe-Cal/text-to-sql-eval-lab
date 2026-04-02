# text-to-sql-eval-lab

An evaluation lab for benchmarking LLMs on the text-to-SQL task. Given a natural language question and a database schema, models are asked to generate a SQL query. The lab measures syntax validity, execution correctness, result accuracy, and semantic correctness via an LLM-as-judge — against a golden dataset.

## How it works

```
Natural language question + schema
            │
            ▼
      LLM (via LiteLLM)
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
   - `result_match` — executes the query and compares the returned rows to the golden expected rows (1.0 = exact match, 0.5 = right shape but wrong values, 0.0 = wrong).
   - `semantic_judge` — sends the question, generated SQL, actual results, and expected results to a judge LLM; parses a structured verdict (`correct` / `partial` / `incorrect`). Catches false negatives from `result_match` caused by different column aliases, equivalent SQL written differently, or minor float precision gaps.

The eval harness is [Inspect AI](https://inspect.aisi.org.uk/), which handles parallelism, logging, and the `inspect view` log explorer.

## Dataset

15 hand-crafted question/answer pairs in [`datasets/golden/questions.json`](datasets/golden/questions.json) against a toy e-commerce DuckDB database.

**Schema**

```
customers(id, name, email, country, signup_date)
products(id, name, category, price)
orders(id, customer_id, order_date, status)       -- status: completed | pending | cancelled
order_items(id, order_id, product_id, quantity, unit_price)
```

**Difficulty breakdown**

| Difficulty | Questions | Patterns tested |
|---|---|---|
| Easy | q01–q05 | COUNT, DISTINCT, simple filters, ORDER BY |
| Medium | q06–q10 | JOINs, aggregations, subqueries, date functions |
| Hard | q11–q15 | Anti-joins (NOT IN), HAVING, time-series, multi-table JOINs |

## Setup

**Requirements:** Python 3.11+

```bash
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

```ini
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEFAULT_MODEL=openai/gpt-4o-mini
DATABASE_PATH=./datasets/ecommerce.duckdb

# LLM-as-judge scorer — can be a different (stronger) model than the one being evaluated
JUDGE_MODEL=openai/gpt-4o-mini

# Embedding model for few_shot_dynamic strategy
EMBEDDING_MODEL=openai/text-embedding-3-small
```

The database is seeded automatically on first run. To reseed manually:

```bash
python src/utils/db.py
```

## Prompt strategies

Four strategies are available, controlling what gets prepended to the prompt before the schema and question:

| Strategy | Description | Extra cost |
|---|---|---|
| `zero_shot` | Schema + question only (default) | none |
| `few_shot_static` | Prepends 3 fixed examples (easy/medium/hard) | none |
| `few_shot_dynamic` | Embeds the question, picks 3 most similar golden examples | 1 embedding call |
| `chain_of_thought` | Instructs the model to reason step-by-step before writing SQL | none (larger output) |

## Running evaluations

```bash
# Default model (gpt-4o-mini), zero_shot strategy, all 15 questions
python scripts/run_eval.py

# Specific model
python scripts/run_eval.py --model anthropic/claude-haiku-4-5

# Filter by difficulty
python scripts/run_eval.py --model openai/gpt-4o --difficulty hard

# Compare multiple models side by side
python scripts/run_eval.py --models openai/gpt-4o-mini anthropic/claude-haiku-4-5

# Run a specific prompt strategy
python scripts/run_eval.py --strategy few_shot_dynamic

# Compare all strategies side by side
python scripts/run_eval.py --strategies zero_shot few_shot_static few_shot_dynamic chain_of_thought
```

Any model supported by [LiteLLM](https://docs.litellm.ai/docs/providers) works — pass the `provider/model-name` string. The judge model is controlled separately via `JUDGE_MODEL` and can be set to a stronger model (e.g. `openai/gpt-4o`) for stricter evaluation without changing the model under test.

## Experimental findings

Full run: `openai/gpt-4o-mini`, 15 questions, all strategies.

| Strategy | result_match | semantic_judge | time |
|---|---|---|---|
| `zero_shot` | 0.733 | 0.867 | 0:48 |
| `few_shot_static` | 0.767 | 0.767 | 0:48 |
| `few_shot_dynamic` | **0.933** | **0.933** | 0:52 |
| `chain_of_thought` | 0.667 | 0.800 | 1:16 |

**Key findings:**

- **`few_shot_dynamic` is the recommended strategy** — +27% result_match over zero_shot, only +4s latency, and the only strategy where result_match and semantic_judge are fully aligned (no false negatives in either direction).

- **`few_shot_static` is inconsistent** — it helped `claude-haiku` significantly but *hurt* `gpt-4o-mini`'s semantic score vs zero_shot. Static examples can bias smaller models toward a specific SQL style, trading semantic correctness for syntactic similarity.

- **Chain-of-thought underperformed and is not recommended for this model/task** — −10% result_match vs zero_shot, +58% latency. Root cause: CoT focuses the model on reasoning about joins and filters but causes *column selection drift* — 4 of 5 failures used `SELECT *` or returned extra columns not asked for in the question. A fifth failure showed CoT reasoning leading to a join cardinality error (joining `order_items` for revenue inflated `COUNT(orders)`). CoT benefits larger models more; `gpt-4o-mini` lacks the reasoning capacity to reliably apply it.

- **LLM judge calibration matters** — the initial `semantic_judge` produced false partial verdicts on `q02`, `q10`, and `q13` due to Python datetime reprs being passed as-is to the judge, and the judge using world knowledge to infer "missing" rows. Fixed by pre-normalizing actual result rows (ISO dates, rounded floats) and adding explicit grounding rules to the judge prompt. After the fix, `few_shot_dynamic` shows perfect alignment: `result_match = semantic_judge = 0.933`.

> **Tip:** cases where `result_match = 0` but `semantic_judge = 1` are false negatives in the deterministic scorer — the model was semantically correct but returned rows in a different format. Cases where `result_match > semantic_judge` indicate potential judge calibration issues worth investigating.

Logs are saved to `./logs/`. Explore them with:

```bash
inspect view
```

## Project structure

```
text-to-sql-eval-lab/
├── datasets/
│   ├── ecommerce.duckdb             # DuckDB database (auto-created)
│   └── golden/
│       └── questions.json           # 15 golden Q&A pairs
├── scripts/
│   └── run_eval.py                  # CLI entrypoint — supports --model(s), --strategy/strategies, --difficulty
├── src/
│   ├── agent/
│   │   ├── agent.py                 # LLM call, prompt strategies, SQL extraction (Langfuse traced)
│   │   └── few_shot.py              # Static and dynamic (embedding-based) example selection
│   ├── evals/
│   │   ├── tasks.py                 # Inspect AI Task, Dataset, Solver
│   │   └── scorers.py               # syntax_valid, execution_ok, result_match, semantic_judge
│   └── utils/
│       └── db.py                    # DuckDB connection, seeding, schema string
└── pyproject.toml
```

## Key dependencies

| Package | Role |
|---|---|
| [inspect-ai](https://inspect.ai) | Eval harness — tasks, solvers, scorers, logging |
| [litellm](https://docs.litellm.ai) | Unified interface to OpenAI, Anthropic, and other providers |
| [duckdb](https://duckdb.org) | In-process SQL engine for the test database |
| [sqlglot](https://github.com/tobymao/sqlglot) | SQL parser for syntax validation |
| [langfuse](https://langfuse.com) | Observability and tracing (optional) |
