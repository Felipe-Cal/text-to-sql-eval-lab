# text-to-sql-eval-lab

An evaluation lab for benchmarking LLMs on the text-to-SQL task. Given a natural language question and a database schema, models are asked to generate a SQL query. The lab measures syntax validity, execution correctness, and result accuracy against a golden dataset.

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
     ┌──────┼──────┐
     ▼      ▼      ▼
 syntax  execute  result
  valid    ok     match
```

1. A **question** from the golden dataset is sent to the model along with the database schema.
2. The model returns a SQL query.
3. Three **scorers** evaluate the query independently:
   - `syntax_valid` — parses the SQL with [sqlglot](https://github.com/tobymao/sqlglot); no database needed.
   - `execution_ok` — runs the query against the real DuckDB database; checks it doesn't error.
   - `result_match` — executes the query and compares the returned rows to the golden expected rows (1.0 = exact match, 0.5 = right shape but wrong values, 0.0 = wrong).

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
```

The database is seeded automatically on first run. To reseed manually:

```bash
python src/utils/db.py
```

## Running evaluations

```bash
# Default model (gpt-4o-mini), all 15 questions
python scripts/run_eval.py

# Specific model
python scripts/run_eval.py --model anthropic/claude-haiku-4-5

# Filter by difficulty
python scripts/run_eval.py --model openai/gpt-4o --difficulty hard

# Compare multiple models side by side
python scripts/run_eval.py --models openai/gpt-4o-mini anthropic/claude-haiku-4-5
```

Any model supported by [LiteLLM](https://docs.litellm.ai/docs/providers) works — pass the `provider/model-name` string.

**Example output**

```
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ model             ┃ syntax_valid/accuracy┃ execution_ok/accuracy┃ result_match/accuracy┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ openai/gpt-4o-mini│                 1.000│                1.000│                 0.733│
└───────────────────┴──────────────────────┴─────────────────────┴──────────────────────┘
```

Logs are saved to `./logs/`. Explore them with:

```bash
inspect view
```

## Project structure

```
text-to-sql-eval-lab/
├── datasets/
│   ├── ecommerce.duckdb          # DuckDB database (auto-created)
│   └── golden/
│       └── questions.json        # 15 golden Q&A pairs
├── scripts/
│   └── run_eval.py               # CLI entrypoint
├── src/
│   ├── agent/
│   │   └── agent.py              # LLM call + SQL extraction
│   ├── evals/
│   │   ├── tasks.py              # Inspect AI Task, Dataset, Solver
│   │   └── scorers.py            # syntax_valid, execution_ok, result_match
│   └── utils/
│       └── db.py                 # DuckDB connection, seeding, schema string
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
