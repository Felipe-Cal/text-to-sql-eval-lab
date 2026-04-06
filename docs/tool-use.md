# Tool-Use Agent

The `tool_use` strategy upgrades the agent from a SQL generator into a reasoning agent that decides what actions to take. Instead of being forced to produce SQL, the LLM is given a set of tools and chooses which to call based on the question.

---

## Architecture

```
User question
      │
      ▼
LLM Agent (LiteLLM function calling)
      │
      ├── query_database(sql)       → executes SQL on DuckDB → returns rows as JSON
      ├── search_knowledge_base(q)  → RAG over KB documents → returns top-K chunks
      └── get_schema(table?)        → returns database schema → guides SQL writing
      │
      ▼
LLM synthesises a natural language answer from tool results
```

The loop runs until the LLM responds with text instead of a tool call (up to `max_iterations=10`). All tool calls are logged in `AgentResult.tool_calls` for inspection and eval.

---

## Tools (`src/agent/tools.py`)

### `query_database(sql)`
Executes a SQL query against the live DuckDB database. Returns rows as a JSON string on success, or the error message on failure — the LLM reads the error and self-corrects with a revised query.

### `search_knowledge_base(query)`
Runs semantic search over `datasets/docs/ecommerce_kb.md` using the `DocumentRetriever` (sentence chunker + in-memory store). Returns the top-3 most relevant chunks formatted for prompt injection.

### `get_schema(table_name?)`
Returns the database schema, optionally filtered to a specific table. The agent calls this before writing complex SQL when it is unsure of column names — "look before you leap".

All tools use the OpenAI function-calling format natively supported by LiteLLM across all providers (OpenAI, Anthropic, Ollama, etc.).

---

## What tool_use handles that other strategies can't

| Question type | zero_shot / few_shot_dynamic | tool_use |
|---|---|---|
| SQL data question | ✅ | ✅ |
| Policy / KB question | ❌ (generates broken SQL) | ✅ (`search_knowledge_base`) |
| Hybrid (data + policy) | ❌ (can only do one) | ✅ (calls both tools) |
| Unknown schema | ❌ (hallucinates columns) | ✅ (calls `get_schema` first) |

---

## Usage

```bash
# Via CLI eval
python scripts/run_eval.py --strategy tool_use

# Via API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many orders were cancelled and what is the cancellation policy?", "strategy": "tool_use"}'
```

The API response includes `answer` (natural language synthesis) and `tool_calls` (full trace) in addition to `sql`.

---

## Evaluation dataset

`datasets/golden/policy_questions.json` contains 10 questions designed to expose what tool_use uniquely handles:

- **5 policy questions** (KB-only): return policy, payment methods, support hours, shipping, digital products
- **5 hybrid questions** (DB + KB): cancelled order count + cancellation policy, top customers + support tiers, revenue + payment methods, etc.

Scoring for these questions uses `answer_quality` (LLM judge checking for expected keywords) rather than `result_match` (row comparison).

---

## Benchmark

`scripts/benchmark_agent.py` compares all three main strategies across all question types:

```bash
# Full benchmark (all 3 parts, ~20 min)
python scripts/benchmark_agent.py

# SQL questions only (Part 1)
python scripts/benchmark_agent.py --parts 1

# Policy + hybrid only (Parts 2 and 3, ~5 min)
python scripts/benchmark_agent.py --parts 2 3
```

---

## Trade-offs

| | zero_shot | few_shot_dynamic | tool_use |
|---|---|---|---|
| SQL accuracy | 0.667 | **0.933** | ~0.800 |
| Policy accuracy | ❌ N/A | ❌ N/A | **0.900** |
| Avg cost/question | ~$0.0002 | ~$0.0003 | ~$0.001 |
| Avg latency | ~2s | ~2s | ~5–10s |
| Tool calls | 0 | 0 | 2–4 |

**Rule of thumb:**
- Pure SQL product → `few_shot_dynamic`
- General assistant (SQL + policies) → `tool_use`
- Cost-sensitive or latency-sensitive → `zero_shot`

---

## Design notes

- Tool functions return plain strings. The LLM receives these as `tool` role messages and synthesises a final answer — it can reason across multiple tool results before responding.
- `result.sql` is populated from the last `query_database` call so the existing SQL scorers (`syntax_valid`, `execution_ok`, `result_match`) still work when `tool_use` is used in the standard eval harness.
- The KB retriever is a module-level singleton — indexed once, reused across all requests.
- `avg_tool_calls` is tracked as an eval metric so the agentic overhead is visible in the results table alongside cost and latency.
