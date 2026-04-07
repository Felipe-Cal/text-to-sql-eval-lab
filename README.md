# text-to-sql-eval-lab

An evaluation lab for benchmarking LLMs on the text-to-SQL task. Given a natural language question and a database schema, models generate a SQL query. The lab measures syntax validity, execution correctness, result accuracy, and semantic correctness via an LLM-as-judge — against a golden dataset.

The project ships as a **FastAPI service** so the agent and eval suite can be consumed by any frontend, CI pipeline, or external tool via HTTP, not just as a local script.

## How it works

```
Natural language question + schema
            │
            ▼
     LLM (via LiteLLM)
            │
     Self-correction loop (LangGraph StateGraph, up to 3 retries on SQL error)
            │
            ▼
      Generated SQL
            │
     ┌──────┼──────┬──────────┐
     ▼      ▼      ▼          ▼
 syntax  execute  result   semantic
  valid    ok     match     judge
```

1. A **question** from the golden dataset is sent to the model along with the database schema (or a retrieved subset of it, for RAG strategies).
2. The model returns a SQL query.
3. Four **scorers** evaluate the query independently:
   - `syntax_valid` — parses the SQL with [sqlglot](https://github.com/tobymao/sqlglot); no database needed.
   - `execution_ok` — runs the query against the real DuckDB database; checks it doesn't error.
   - `result_match` — executes and compares returned rows to golden expected rows (1.0 = exact, 0.5 = right shape wrong values, 0.0 = wrong).
   - `semantic_judge` — sends question, generated SQL, actual results, and expected results to a judge LLM for a structured verdict (uses Instructor + Pydantic for type-safe output).
4. **Self-correction loop**: on SQL error, the agent feeds the error trace back and retries — up to 3 times, orchestrated by a LangGraph StateGraph.

The eval harness is [Inspect AI](https://inspect.aisi.org.uk/), which handles parallelism, logging, and the `inspect view` log explorer.

---

## Dataset

- **Golden suite:** 15 hand-crafted Q&A pairs in `datasets/golden/questions.json`.
- **Policy/hybrid questions:** 10 questions for tool-use evaluation in `datasets/golden/policy_questions.json` (5 KB-only, 5 hybrid data+policy).
- **Synthetic flywheel:** `scripts/generate_synthetic.py` generates and validates synthetic questions, split into `tuning.json` (80%) and `holdout_test.json` (20%).

**Schema**

```
customers(id, name, email, country, signup_date)
products(id, name, category, price)
orders(id, customer_id, order_date, status)       -- status: completed | pending | cancelled
order_items(id, order_id, product_id, quantity, unit_price)
```

The schema is wrapped inside a **50-table enterprise data warehouse** simulation (46 decoy tables from HR, logistics, finance, marketing) to stress-test schema linking via the RAG strategies.

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
| `GEMINI_API_KEY` | Google Gemini API key — get one at aistudio.google.com/apikey (optional) |
| `DEFAULT_MODEL` | LiteLLM model string (e.g. `openai/gpt-4o-mini`, `gemini/gemini-2.0-flash`) |
| `DATABASE_PATH` | Path to DuckDB file |
| `JUDGE_MODEL` | Model used by the semantic_judge scorer |
| `EMBEDDING_MODEL` | Model used for few_shot_dynamic and RAG dense strategies |
| `HARD_MODEL` | Escalation model for hard questions in the router (optional) |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Langfuse observability (optional) |
| `VLLM_API_BASE` | vLLM inference server URL — routes all completions there when set (optional) |
| `VLLM_MODEL` | Model name served by vLLM (e.g. `meta-llama/Meta-Llama-3.1-8B-Instruct`) |
| `VLLM_EMBEDDING_MODEL` | Embedding model served by vLLM (optional) |

The database is seeded automatically on first run. To reseed manually: `python src/utils/db.py`.

---

## Quick start

```bash
# Run evals (default: gpt-4o-mini, zero_shot, all 15 questions)
python scripts/run_eval.py

# Recommended strategy for SQL quality
python scripts/run_eval.py --strategy few_shot_dynamic

# RAG strategy with hybrid search
python scripts/run_eval.py --strategy rag_hybrid

# Start the API server
uvicorn src.api.main:app --reload

# Run tests
pytest
```

---

## Going deeper

| Topic | Doc |
|---|---|
| **Evaluation pipeline** (scorers, metrics, DeepEval, how to run and interpret results) | [docs/evaluation.md](docs/evaluation.md) |
| All 9 prompt strategies (zero_shot → tool_use), RAG, LangGraph, DeepEval | [docs/strategies.md](docs/strategies.md) |
| Tool-use agent (agentic SQL + KB search, policy/hybrid questions) | [docs/tool-use.md](docs/tool-use.md) |
| RAG infrastructure (chunking, vector stores, Qdrant hybrid search, benchmarks) | [docs/rag.md](docs/rag.md) |
| FastAPI endpoints (`/query`, `/evals/run`, response schemas) | [docs/api.md](docs/api.md) |
| Fine-tuning Llama 3.1 8B with LoRA (Colab, Ollama, results) | [docs/fine-tuning.md](docs/fine-tuning.md) |
| Model router (difficulty classification, rule-based + embedding k-NN) | [docs/routing.md](docs/routing.md) |
| DSPy prompt optimization (BootstrapFewShot, compiled prompts) | [docs/dspy.md](docs/dspy.md) |
| Evals-as-CI (GitHub Actions gate, thresholds, local gate script) | [docs/evals-ci.md](docs/evals-ci.md) |
| Guardrails (input regex, output AST, read-only DB, adversarial tests) | [docs/guardrails.md](docs/guardrails.md) |
| Experimental findings and benchmark results | [docs/findings.md](docs/findings.md) |

---

## Project structure

```
text-to-sql-eval-lab/
├── .github/workflows/eval.yml        # CI eval gate
├── datasets/
│   ├── ecommerce.duckdb              # DuckDB database (auto-created on first run)
│   ├── docs/ecommerce_kb.md          # Knowledge base for tool_use + RAG search
│   ├── golden/
│   │   ├── questions.json            # 15 golden SQL Q&A pairs
│   │   └── policy_questions.json     # 10 policy/hybrid questions for tool_use eval
│   ├── synthetic/                    # tuning.json + holdout_test.json
│   └── dspy_optimized_prompt.json    # output of optimize_prompt.py
├── docs/                             # detailed documentation (see "Going deeper" above)
├── notebooks/finetune_llama.ipynb    # Colab LoRA fine-tuning notebook
├── scripts/
│   ├── run_eval.py                   # CLI eval runner
│   ├── generate_synthetic.py         # synthetic data flywheel
│   ├── optimize_prompt.py            # DSPy BootstrapFewShot optimizer
│   ├── ci_eval.py                    # CI gate script (exits 0/1)
│   ├── benchmark_routing.py          # routing vs. fixed-strategy comparison
│   ├── benchmark_rag.py              # KB document retrieval: chunker × store × top-K
│   ├── benchmark_schema_retrieval.py # schema linking: backend × embedding library × strategy
│   ├── benchmark_agent.py            # zero_shot vs few_shot_dynamic vs tool_use
│   └── prepare_finetune_data.py      # data prep for LoRA fine-tuning
├── src/
│   ├── api/                          # FastAPI service
│   │   ├── main.py                   # app entry point, CORS, router wiring
│   │   └── routes/
│   │       ├── agent.py              # POST /query
│   │       └── evals.py              # POST /evals/run  GET /evals/{job_id}
│   ├── agent/
│   │   ├── agent.py                  # generate_sql — LangGraph loop, all strategies
│   │   ├── few_shot.py               # static + dynamic (embedding cosine similarity) examples
│   │   ├── schema_retriever.py       # Qdrant hybrid search — dense + sparse + RRF
│   │   ├── router.py                 # difficulty classifier: rule-based + embedding k-NN
│   │   └── tools.py                  # query_database, search_knowledge_base, get_schema
│   ├── rag/
│   │   ├── chunker.py                # FixedSizeChunker, SentenceChunker, SchemaChunker
│   │   ├── vector_store.py           # InMemoryStore + ChromaDBStore (shared interface)
│   │   └── retriever.py              # DocumentRetriever (chunker + store)
│   ├── evals/
│   │   ├── tasks.py                  # Inspect AI Task, Dataset, Solver
│   │   └── scorers.py                # all scorers + metrics (syntax, execution, result, judge, DeepEval)
│   ├── guardrails/
│   │   ├── input.py                  # pre-LLM: SQL injection + prompt injection checks (regex)
│   │   └── output.py                 # post-LLM: SELECT-only + schema allowlist (sqlglot AST)
│   └── utils/db.py                   # DuckDB connection (read_only=True for queries), seeding
└── tests/                            # pytest — guardrails, adversarial injection tests
```

---

## Key dependencies

| Package | Role |
|---|---|
| [fastapi](https://fastapi.tiangolo.com) | HTTP service layer |
| [inspect-ai](https://inspect.ai) | Eval harness — tasks, solvers, scorers, logging |
| [litellm](https://docs.litellm.ai) | Unified interface to OpenAI, Anthropic, Gemini, Ollama, and others |
| [langgraph](https://langchain-ai.github.io/langgraph/) | StateGraph for the self-correction retry loop in `generate_sql` |
| [instructor](https://python.useinstructor.com) | Structured outputs via Pydantic — used in `semantic_judge` |
| [qdrant-client](https://qdrant.tech) | Vector DB with native hybrid search (dense + sparse RRF) |
| [fastembed](https://qdrant.github.io/fastembed/) | Local embedding models (SPLADE sparse + BGE/Nomic dense) — zero API cost |
| [chromadb](https://www.trychroma.com) | Persistent HNSW vector store — used in KB retrieval |
| [duckdb](https://duckdb.org) | In-process SQL engine |
| [sqlglot](https://github.com/tobymao/sqlglot) | SQL parser for syntax validation and output guardrails |
| [dspy-ai](https://dspy.ai) | Prompt optimization via BootstrapFewShot |
| [deepeval](https://deepeval.com/) | Diagnostic LLM eval metrics (Faithfulness, Relevancy, G-Eval) |
| [langfuse](https://langfuse.com) | Observability and tracing (optional) |
