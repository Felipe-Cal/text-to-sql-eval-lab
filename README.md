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
   - `result_match` — executes the query and compares the returned rows to the golden expected rows (1.0 = exact match, 0.5 = right shape but wrong values, 0.0 = wrong).
   - `semantic_judge` — sends the question, generated SQL, actual results, and expected results to a judge LLM; parses a structured verdict (`correct` / `partial` / `incorrect`). Catches false negatives from `result_match` caused by different column aliases, equivalent SQL written differently, or minor float precision gaps.
4. **Self-Correction loop**: If a generated query throws a syntax or execution error in DuckDB, the agent automatically feeds the error back into its conversation history and retries up to 3 times.
5. **Constraint tracking**: Custom Inspect metrics natively track the `cost`, `latency`, `tokens`, and average `attempts` required for the run, tracking production viability alongside raw accuracy.

The eval harness is [Inspect AI](https://inspect.aisi.org.uk/), which handles parallelism, logging, and the `inspect view` log explorer.

---

## Dataset

1. **Golden Suite:** 15 hand-crafted question/answer pairs in `datasets/golden/questions.json` against a toy e-commerce DuckDB database.
2. **Synthetic Data Flywheel:** The script `scripts/generate_synthetic.py` uses an LLM to look at the schema, invent difficult questions, and *verify* them by executing them against the DuckDB instance. Validated outputs are split into `tuning.json` (80%) and `holdout_test.json` (20%) to test for generalization and prevent prompt overfitting.

**Schema**

```
customers(id, name, email, country, signup_date)
products(id, name, category, price)
orders(id, customer_id, order_date, status)       -- status: completed | pending | cancelled
order_items(id, order_id, product_id, quantity, unit_price)
```

The schema is intentionally wrapped inside a **50-table enterprise data warehouse** simulation (46 decoy tables from unrelated domains like HR, logistics, finance, marketing). This is used by the RAG strategy to stress-test schema linking under realistic noise conditions.

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

# Embedding model for few_shot_dynamic and RAG strategies
EMBEDDING_MODEL=openai/text-embedding-3-small
```

The database is seeded automatically on first run. To reseed manually:

```bash
python src/utils/db.py
```

---

## Prompt strategies

Five strategies are available, controlling what context is prepended to the prompt before the schema and question:

| Strategy | Description | Extra cost |
|---|---|---|
| `zero_shot` | Schema + question only (default) | none |
| `few_shot_static` | Prepends 3 fixed examples (easy/medium/hard) | none |
| `few_shot_dynamic` | Embeds the question, picks 3 most similar golden examples by cosine similarity | 1 embedding call |
| `chain_of_thought` | Instructs the model to reason step-by-step before writing SQL | none (larger output) |
| `rag` | Embeds the question and retrieves the top-K most semantically relevant table definitions from the 50-table DWH, rather than sending the full schema | 1 embedding call |

### RAG schema linking (`rag` strategy)

Rather than passing all 50 table definitions to the model (which balloons the prompt and confuses smaller models), the `rag` strategy:

1. Embeds the natural language question using the configured embedding model.
2. Pre-computes and **caches** embeddings for all 50 table definitions (only once per process).
3. Ranks tables by **cosine similarity** and injects only the top-K (default: 5) into the prompt.
4. Tracks `retrieval_recall` as an eval metric — what % of the tables required by the golden SQL were actually retrieved.

This simulates a real enterprise use case where the schema is too large to fit in the prompt.

---

## Running evaluations

All commands below use **15** golden questions unless you pass `--difficulty` (easy = 5, medium = 5, hard = 5). Logs are written to **`./logs/`**.

### Quick start (single run)

```bash
# Default: openai/gpt-4o-mini, zero_shot, all difficulties
python scripts/run_eval.py
```

### One model, all strategies (full strategy sweep)

Runs four strategies back-to-back and prints one summary table.

```bash
python scripts/run_eval.py --strategies zero_shot few_shot_static few_shot_dynamic chain_of_thought
```

### Full benchmark: models × strategies (matrix)

Runs every combination (e.g. 2 models × 4 strategies = **8** eval runs). Expect on the order of **~30–45 minutes** depending on API latency and models.

```bash
python scripts/run_eval.py \
  --models openai/gpt-4o-mini anthropic/claude-haiku-4-5 \
  --strategies zero_shot few_shot_static few_shot_dynamic chain_of_thought
```

### Other useful options

```bash
# Single model and strategy
python scripts/run_eval.py --model anthropic/claude-haiku-4-5 --strategy few_shot_dynamic

# Only hard questions (q11–q15)
python scripts/run_eval.py --model openai/gpt-4o-mini --difficulty hard

# Compare models only (default strategy: zero_shot)
python scripts/run_eval.py --models openai/gpt-4o-mini anthropic/claude-haiku-4-5
```

### Inspect AI CLI (advanced)

Equivalent to what `run_eval.py` wraps, with full Inspect flags:

```bash
inspect eval src/evals/tasks.py --model openai/gpt-4o-mini
```

Any model supported by [LiteLLM](https://docs.litellm.ai/docs/providers) works — pass the `provider/model-name` string. The judge model is controlled separately via `JUDGE_MODEL` and can be set to a stronger model (e.g. `openai/gpt-4o`) for stricter evaluation without changing the model under test.

---

## FastAPI service

The agent and eval suite are exposed as an HTTP service so they can be integrated with frontends, CI pipelines, Slack bots, or any other system.

### Start the server

```bash
uvicorn src.api.main:app --reload
```

The server starts at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the auto-generated interactive Swagger UI where you can try every endpoint from the browser.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/query` | Run the text-to-SQL agent on a question |
| `POST` | `/evals/run` | Start an eval run in the background; returns a `job_id` |
| `GET` | `/evals/{job_id}` | Poll an eval job's status and results |

---

### `POST /query` — run the agent

Runs the text-to-SQL agent on a natural language question and returns the generated SQL plus full execution metadata.

**Request body:**

```json
{
  "question": "What are the top 3 customers by total spend?",
  "model": "openai/gpt-4o-mini",
  "strategy": "few_shot_dynamic"
}
```

- `model` — optional; defaults to `DEFAULT_MODEL` env var
- `strategy` — optional; one of `zero_shot`, `few_shot_static`, `few_shot_dynamic`, `chain_of_thought`, `rag`; defaults to `zero_shot`

**Response:**

```json
{
  "question": "What are the top 3 customers by total spend?",
  "sql": "SELECT c.name, SUM(oi.quantity * oi.unit_price) AS total_spend ...",
  "model": "openai/gpt-4o-mini",
  "strategy": "few_shot_dynamic",
  "reasoning": null,
  "prompt_tokens": 412,
  "completion_tokens": 58,
  "cost": 0.00008,
  "latency": 1.23,
  "attempts": 1,
  "trace_id": "abc123"
}
```

- `reasoning` is non-null only for `chain_of_thought` strategy (contains the model's step-by-step reasoning).
- `attempts` shows how many tries the self-correction loop needed (1 = first try succeeded).
- `trace_id` links to the Langfuse trace for this call (if Langfuse is configured).

---

### `POST /evals/run` — start an eval job

Eval runs are long (minutes), so this endpoint returns immediately with a `job_id` and runs the evaluation in the background. Poll `GET /evals/{job_id}` to check progress.

**Request body:**

```json
{
  "model": "openai/gpt-4o-mini",
  "strategy": "zero_shot",
  "difficulty": "hard",
  "judge_model": "openai/gpt-4o"
}
```

- All fields are optional. `difficulty` filters to `easy`, `medium`, or `hard`; omit for all 15 questions.
- `judge_model` overrides the `JUDGE_MODEL` env var for this run only.

**Response (202 Accepted):**

```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "running",
  "started_at": "2026-04-03T10:00:00Z",
  "finished_at": null,
  "results": null,
  "error": null
}
```

---

### `GET /evals/{job_id}` — poll eval results

**Response when running:**

```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "running",
  "started_at": "2026-04-03T10:00:00Z",
  "finished_at": null,
  "results": null,
  "error": null
}
```

**Response when completed:**

```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "completed",
  "started_at": "2026-04-03T10:00:00Z",
  "finished_at": "2026-04-03T10:02:35Z",
  "results": {
    "total_samples": 15,
    "scores": {
      "syntax_valid":  { "accuracy": 1.0 },
      "execution_ok":  { "accuracy": 0.933 },
      "result_match":  { "accuracy": 0.933 },
      "semantic_judge":{ "accuracy": 0.933 },
      "avg_attempts":  { "mean": 1.07 },
      "avg_cost":      { "mean": 0.000082 },
      "avg_latency":   { "mean": 1.41 },
      "avg_total_tokens": { "mean": 487 },
      "retrieval_recall": { "mean": 1.0 }
    }
  },
  "error": null
}
```

### Design notes

- **`/query` is non-blocking** — the LiteLLM call runs in a thread pool via `asyncio.to_thread`, so FastAPI's event loop is never blocked while waiting for the LLM.
- **`/evals/run` is fire-and-forget** — eval jobs run in the background; you poll for results. This is necessary because a full 15-question run with judge scoring takes 1–3 minutes.
- **In-memory job store** — jobs are stored in a Python dict keyed by UUID. Fine for local use; swap for Redis/database in production.

---

## DSPy prompt optimization

Rather than hand-crafting prompts, [DSPy](https://dspy.ai) treats prompts as learnable programs. The optimizer automatically finds the best few-shot demonstrations from your training data by trying candidates and keeping the ones that actually improve the metric — no manual prompt tuning required.

### How it works

```
Synthetic tuning dataset (tuning.json)
            │
            ▼
     DSPy BootstrapFewShot
      ├── Tries candidate demonstrations
      ├── Executes predicted SQL against DuckDB
      └── Keeps demos that maximize result_match
            │
            ▼
  Compiled prompt saved to datasets/dspy_optimized_prompt.json
```

1. A `TextToSQL` **Signature** declares the task as typed I/O fields: `schema + question → sql`.
2. `SQLGenerator` wraps it in `dspy.ChainOfThought`, which automatically adds a reasoning step before the SQL output — no manual CoT prompt needed.
3. The **`db_metric`** function is the optimization target: it executes the predicted SQL against DuckDB and returns `1` if the rows exactly match the expected result (using the same `_normalize_rows` helper as the eval scorers), `0` otherwise.
4. **`BootstrapFewShot`** tries many few-shot demo combinations from the synthetic tuning set, evaluates each one with `db_metric`, and keeps the top 4 bootstrapped + 4 labeled demos that maximize real execution accuracy.
5. The compiled module (with its optimized demonstrations baked in) is saved as JSON and can be loaded for inference without re-running the optimizer.

### Inputs

DSPy optimization consumes the **synthetic tuning dataset** generated by `scripts/generate_synthetic.py` (`datasets/synthetic/tuning.json`). It also calls the RAG schema retriever to pre-fetch relevant table definitions for each training question — the same schema context the model will see at inference time.

### Run the optimizer

```bash
python scripts/optimize_prompt.py
```

This will:
1. Load `datasets/synthetic/tuning.json`
2. Fetch RAG schemas for each training question (cached after first call)
3. Run `BootstrapFewShot` optimization
4. Save the compiled prompt to `datasets/dspy_optimized_prompt.json`

Expect the run to take a few minutes — the optimizer makes one LLM + one DuckDB execution call per candidate demo tried.

### DSPy vs manual prompt strategies

| Approach | How examples are chosen | Optimized for |
|---|---|---|
| `few_shot_static` | Hand-picked by the developer | Developer intuition |
| `few_shot_dynamic` | Nearest neighbor by embedding similarity to the question | Semantic similarity |
| DSPy `BootstrapFewShot` | Automatically selected by trying candidates and measuring real execution accuracy | Actual task metric (result_match) |

The key difference: DSPy demos are chosen because they were **empirically proven to improve SQL execution accuracy**, not because a human thought they looked like good examples or because they were semantically similar.

### Key DSPy concepts used

| Concept | What it does |
|---|---|
| `dspy.Signature` | Declares the task as typed I/O: `schema + question → sql` |
| `dspy.ChainOfThought` | Adds an automatic reasoning step before the output field |
| `dspy.Module` | Wraps the pipeline into a composable, optimizable program |
| `BootstrapFewShot` | Optimizer that bootstraps few-shot demos from a training set using a metric |
| `db_metric` | Custom metric: executes SQL against DuckDB, compares rows to expected |
| `.save()` / `.load()` | Serializes the compiled (optimized) prompt for reuse |

---

## Guardrails and tests (pytest)

Implementation lives under [`src/guardrails/`](src/guardrails/): deterministic checks with **no extra LLM calls**. Call `check_input` before the LLM and `check_output` on generated SQL before execution when you wire a production path; this repo ships the **library plus tests** so behavior stays regression-tested.

```bash
pytest
```

### Input guardrails ([`src/guardrails/input.py`](src/guardrails/input.py))

Run **before** the model sees the question (`check_input`). Two independent checks; the first failure wins:

| Check | What it blocks |
|--------|----------------|
| **SQL injection** | Raw SQL fragments in the user string: DDL/DML keywords, comment tricks (`--`, `/* */`), `UNION` injection, stacked `;` queries, etc. |
| **Prompt injection** | Phrases that try to override instructions ("ignore the above", "new instructions", jailbreak-style patterns). |

Tests in [`tests/test_input_guardrails.py`](tests/test_input_guardrails.py) cover legitimate golden-style questions (must pass), injection strings (must fail), and edge cases (empty input, mixed case).

### Output guardrails ([`src/guardrails/output.py`](src/guardrails/output.py))

Run **after** the model returns SQL, **before** execution (`check_output`):

| Check | What it does |
|--------|----------------|
| **SELECT-only** | Uses **sqlglot AST**: root statement must be read-only; blocks DDL (e.g. `DROP`, `CREATE`) and DML (`INSERT`, `UPDATE`, `DELETE`). |
| **Schema scope** | Every referenced table must be in the allowlist (`customers`, `products`, `orders`, `order_items` by default). Stops queries against arbitrary tables (e.g. catalog tables). |

Tests in [`tests/test_output_guardrails.py`](tests/test_output_guardrails.py) cover valid `SELECT`s (including joins, CTEs, trailing `;`), and ensure DDL/DML / unknown tables are rejected.

### Adversarial tests ([`tests/test_guardrails_adversarial.py`](tests/test_guardrails_adversarial.py))

Goes beyond happy paths: **evasion attempts** (obfuscated keywords, leetspeak, Unicode lookalikes, prompt-injection phrasing). Some cases are marked **`xfail`** — they document **known gaps** of regex/keyword input filtering (not silent failures). The file's docstring explains the finding: **output** guardrails (AST-based) are harder to evade than **input** (regex-based); **defense in depth** means input bypasses may still be caught at output.

```bash
pytest tests/test_input_guardrails.py -v
pytest tests/test_output_guardrails.py -v
pytest tests/test_guardrails_adversarial.py -v
```

---

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

Logs are saved to `./logs/`. Explore them in the browser:

```bash
# Recommended (uses your active Python; works well with Conda)
python -m inspect_ai view start --log-dir ./logs
```

If `inspect view` alone shows empty rows or SSL errors, your shell may be picking a different `inspect` binary — use the command above or pass `--log-dir ./logs` explicitly.

Open the URL printed in the terminal (default [http://127.0.0.1:7575](http://127.0.0.1:7575)).

---

## Project structure

```
text-to-sql-eval-lab/
├── datasets/
│   ├── ecommerce.duckdb             # DuckDB database (auto-created)
│   └── golden/
│       └── questions.json           # 15 golden Q&A pairs
├── scripts/
│   ├── run_eval.py                  # CLI entrypoint — supports --model(s), --strategy/strategies, --difficulty
│   ├── generate_synthetic.py        # Data flywheel generating validated tuning + holdout datasets
│   └── optimize_prompt.py           # DSPy BootstrapFewShot optimizer — compiles few-shot demos from tuning.json
├── tests/                           # pytest — input/output/adversarial guardrails
├── src/
│   ├── api/
│   │   ├── main.py                  # FastAPI app, router wiring, /health
│   │   └── routes/
│   │       ├── agent.py             # POST /query — run agent, return SQL + metadata
│   │       └── evals.py             # POST /evals/run, GET /evals/{job_id}
│   ├── guardrails/
│   │   ├── input.py                 # pre-LLM: SQL + prompt injection checks
│   │   └── output.py                # post-LLM: SELECT-only + schema allowlist (AST)
│   ├── agent/
│   │   ├── agent.py                 # LLM call, prompt strategies, self-correction loop (Langfuse traced)
│   │   ├── few_shot.py              # Static and dynamic (embedding-based) example selection
│   │   └── schema_retriever.py      # RAG schema linking: embeds question, retrieves top-K tables
│   ├── evals/
│   │   ├── tasks.py                 # Inspect AI Task, Dataset, Solver
│   │   └── scorers.py               # syntax_valid, execution_ok, result_match, semantic_judge, custom metrics
│   └── utils/
│       └── db.py                    # DuckDB connection, seeding, schema string
└── pyproject.toml
```

---

## Key dependencies

| Package | Role |
|---|---|
| [fastapi](https://fastapi.tiangolo.com) | HTTP service layer — exposes agent and evals as REST endpoints |
| [uvicorn](https://www.uvicorn.org) | ASGI server that runs the FastAPI app |
| [inspect-ai](https://inspect.ai) | Eval harness — tasks, solvers, scorers, logging |
| [litellm](https://docs.litellm.ai) | Unified interface to OpenAI, Anthropic, and other providers |
| [duckdb](https://duckdb.org) | In-process SQL engine for the test database |
| [sqlglot](https://github.com/tobymao/sqlglot) | SQL parser for syntax validation and output guardrails |
| [langfuse](https://langfuse.com) | Observability and tracing (optional) |
| [pydantic](https://docs.pydantic.dev) | Request/response validation for the FastAPI layer |
| [dspy-ai](https://dspy.ai) | Prompt optimization — compiles few-shot demos from training data using a real execution metric |
