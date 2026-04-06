# Prompt strategies

Nine strategies are available, covering everything from a minimal zero-shot call to a full agentic loop:

| Strategy | Description | Extra cost |
|---|---|---|
| `zero_shot` | Schema + question only. No examples, no reasoning steps. The simplest possible prompt. | none |
| `few_shot_static` | Prepends 3 hand-picked examples (one easy, one medium, one hard) to every prompt regardless of the question. | none |
| `few_shot_dynamic` | Embeds the question and picks the 3 most semantically similar examples from the golden dataset using cosine similarity. | 1 embedding call |
| `chain_of_thought` | System prompt instructs the model to write `Reasoning:` before `SQL:`. The reasoning is parsed and returned as `AgentResult.reasoning`. | none (larger output) |
| `rag_dense` | Embeds the question and retrieves the top-K most relevant table definitions from the 50-table schema using Qdrant cosine similarity. Only the retrieved tables are sent to the model — not all 50. | 1 embedding call |
| `rag_sparse` | Retrieves table definitions using BM25 keyword matching (SPLADE via FastEmbed). Runs locally — no API call, no cost. Best for exact table/column name matches. | none (local compute) |
| `rag_hybrid` | Fuses dense and sparse retrieval results using Reciprocal Rank Fusion (RRF) in Qdrant. Best of both: semantic understanding + exact keyword matching. **Recommended for RAG.** | 1 embedding call |
| `routed` | Classifies question difficulty (easy / medium / hard) using rule-based patterns, falling back to embedding k-NN when confidence is low. Routes to the optimal model + strategy per difficulty. | 0–1 embedding calls |
| `tool_use` | Agentic: the LLM is given tools (`query_database`, `search_knowledge_base`, `get_schema`) and decides which to call. Handles SQL, knowledge base, and hybrid questions in a single interface. | multiple LLM round-trips |

**Recommended for pure SQL tasks:** `few_shot_dynamic` — +40% result_match over zero_shot at +4s latency.

**Recommended for general assistant (SQL + policies):** `tool_use` — handles KB-only, SQL, and hybrid questions. See [tool-use.md](tool-use.md).

---

## RAG strategies — schema linking with Qdrant

All three RAG strategies solve the same problem: the full 50-table schema (4 real + 46 decoy tables) is too large to pass to a model every time. Smaller models get confused by irrelevant tables and produce wrong SQL. Instead, the question is used to retrieve only the relevant tables.

### Why this matters

Without RAG: `SYSTEM: Here are all 50 tables: hr_employees, hr_payroll, ... [3,000 tokens]`

With RAG (top-5): `SYSTEM: Relevant tables for your question: orders, order_items, customers, products`

This reduces prompt size by ~90%, eliminates distraction from unrelated tables, and dramatically improves accuracy on smaller models.

### The three RAG strategies

**`rag_dense`** — semantic similarity via embeddings
- Embeds both the question and every table definition
- Retrieves the K tables with highest cosine similarity to the question
- Works well when the question paraphrases what the table does ("order history" → `orders`)
- Weakness: fails on exact keyword matches if the question uses different phrasing

**`rag_sparse`** — keyword matching via BM25/SPLADE
- Scores tables using token frequency (BM25) with a SPLADE neural weighting
- Runs locally via FastEmbed — zero API cost, zero latency from network calls
- Works well when the question contains the exact table or column name
- Weakness: misses synonyms and semantic paraphrases

**`rag_hybrid`** — Reciprocal Rank Fusion (RRF)
- Runs both dense and sparse searches independently
- Fuses the two ranked lists using RRF: `score = Σ 1/(rank + 60)` per result
- Gets the best of both: semantic understanding and exact keyword matching
- Benchmark result: perfect 1.000 recall@5 across all 15 golden questions

```python
# Internals: schema_retriever.py → retrieve_schema(question, top_k=5, retrieval_type="hybrid")
```

### Benchmark results

From `scripts/benchmark_schema_retrieval.py` across 15 golden questions, 50 tables:

| Strategy | Embedding | top_k | Recall@K | Avg query time |
|---|---|---|---|---|
| sparse (BM25/SPLADE) | local | 3–5 | **1.000** | 27–39ms |
| hybrid (RRF) | text-embedding-3-large | 3 | **1.000** | 329ms |
| hybrid (RRF) | bge-small (local) | 5 | **1.000** | 64ms |
| dense | text-embedding-3-large | 5 | **1.000** | 348ms |
| dense | text-embedding-3-small | 3–5 | 0.900 | 241ms |

For schema tables with exact names (like `order_items`), BM25 is surprisingly effective. For production use with a noisier, real-world schema, hybrid is the safest choice.

---

## LangGraph self-correction loop

Regardless of which prompt strategy is used, the core SQL generation in `src/agent/agent.py` runs inside a **LangGraph StateGraph** — a deterministic state machine with explicit transitions:

```
                ┌─────────────────┐
  question ──▶  │  generate_node  │  ◀──── error + SQL trace (on retry)
                └────────┬────────┘
                         │ candidate SQL
                         ▼
                ┌─────────────────┐
                │  execute_node   │  runs SQL against DuckDB (read_only=True)
                └────────┬────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
           success               SQL error
              │                     │
              ▼                     ▼
          return result      retry (up to max_retries=3)
```

**What makes this agentic:** on a SQL error, the graph passes the full error message and the failed SQL back to `generate_node`. The LLM reads the DuckDB error trace verbatim and produces a corrected query. This is the same self-correction pattern used in production agents at scale.

**Why LangGraph instead of a while loop?** The StateGraph makes every state transition explicit, typed, and auditable. State is immutable at each node — `generate_node` cannot accidentally read `execute_node`'s internal variables. The graph can be visualised, logged, and unit-tested at the node level.

---

## DeepEval diagnostic metrics (experimental)

In addition to the main scorers, `src/evals/scorers.py` includes three **DeepEval** diagnostic metrics:

| Metric | What it measures |
|---|---|
| `faithfulness_score` | Whether the generated SQL is grounded in the retrieved schema — detects hallucinated table/column names |
| `answer_relevancy_score` | Alignment between the natural language question and the SQL's intent |
| `sql_quality_geval` | Custom rubric: rewards efficient JOINs, correct aggregations, clean aliasing, avoids `SELECT *` |

These run as LLM-as-judge calls and are **diagnostic — not blocking**. They are useful for understanding failure modes (e.g. "the model got the SQL right but hallucinated a CTE alias") rather than as a go/no-go gate. Use them during development to diagnose regressions; rely on `result_match` and `semantic_judge` for CI thresholds.

---

## Running a strategy sweep

```bash
# Single strategy (recommended starting point)
python scripts/run_eval.py --strategy few_shot_dynamic

# Compare multiple strategies on one model
python scripts/run_eval.py --strategies zero_shot few_shot_static few_shot_dynamic chain_of_thought

# Full matrix: models × strategies
python scripts/run_eval.py \
  --models openai/gpt-4o-mini anthropic/claude-haiku-4-5 \
  --strategies zero_shot few_shot_dynamic rag_hybrid

# RAG strategies specifically
python scripts/run_eval.py --strategies rag_dense rag_sparse rag_hybrid
```
