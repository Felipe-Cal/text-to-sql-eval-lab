# Tool-Use Agent

The `tool_use` strategy upgrades the agent from a SQL generator into a reasoning agent that decides what actions to take. Instead of being forced to produce SQL, the LLM is given a set of tools and chooses which to call based on the question.

This allows the same agent to handle questions that don't have SQL answers at all — policy questions, FAQ lookups, and hybrid questions that require both data and knowledge.

---

## Architecture

```
User question
      │
      ▼
LLM Agent (LiteLLM function calling)
      │
      ├── get_schema(table?)           → returns DB schema → used before writing SQL
      ├── query_database(sql)          → executes SQL on DuckDB → returns rows as JSON
      └── search_knowledge_base(query) → RAG over KB documents → returns top-3 chunks
      │
      ▼
LLM synthesises a natural language answer from all tool results
```

The loop runs until the LLM responds with text rather than a tool call (up to `max_iterations=10`). The agent naturally discovers the pattern: **look at schema first → write SQL → check KB if needed**. It is never told to do this — it infers it from the tool descriptions.

---

## Tools (`src/agent/tools.py`)

### `query_database(sql: str) → str`

Executes a SQL query against the live DuckDB database (read-only connection). Returns the result rows as a JSON string on success. On error, returns the DuckDB error message — the LLM reads the error and self-corrects with a revised query, just like a human developer would.

### `search_knowledge_base(query: str) → str`

Runs semantic search over `datasets/docs/ecommerce_kb.md` — a 6 KB knowledge base covering returns, refunds, shipping, payment methods, account management, and customer support policies.

Internally uses a `DocumentRetriever` (sentence chunker + InMemoryStore), indexed once as a module-level singleton. Returns the top-3 most relevant chunks formatted for prompt injection.

### `get_schema(table_name: str = None) → str`

Returns the database schema. If `table_name` is provided, returns only that table's definition. If omitted, returns all four core tables (`customers`, `products`, `orders`, `order_items`).

The agent calls this before writing complex queries when it needs to verify column names — "look before you leap" behaviour that reduces hallucinated columns.

**All tools use the OpenAI function-calling format,** which LiteLLM supports natively across all providers — OpenAI, Anthropic, Ollama, Gemini. No provider-specific code.

---

## What `tool_use` handles that other strategies can't

| Question type | zero_shot / few_shot | tool_use |
|---|---|---|
| SQL data question | ✅ Works well | ✅ Works well |
| Policy / KB question (no SQL) | ❌ Generates broken SQL | ✅ Calls `search_knowledge_base` |
| Hybrid (data + policy) | ❌ Can only do one thing | ✅ Calls both tools, synthesises answer |
| Unknown schema / new table | ❌ Hallucinates column names | ✅ Calls `get_schema` first |

Examples:
- **"What is the return policy?"** → no SQL needed → agent calls `search_knowledge_base`
- **"How many cancelled orders are there and what is the cancellation policy?"** → agent calls `query_database` for the count AND `search_knowledge_base` for the policy, then merges both into one answer

---

## How tool calls are logged

Every tool invocation is recorded in `AgentResult.tool_calls` as an ordered list:

```python
[
    {"tool": "get_schema",             "input": {"table_name": "orders"},        "output": "orders(id, ...)"},
    {"tool": "query_database",         "input": {"sql": "SELECT COUNT(*)..."},   "output": "[{\"count\": 2}]"},
    {"tool": "search_knowledge_base",  "input": {"query": "cancellation policy"}, "output": "Cancelled orders..."},
]
```

This trace is:
- Returned in the `/query` API response under `tool_calls`
- Stored in `state.metadata["tool_calls"]` in the Inspect AI eval harness
- Counted by the `avg_tool_calls` scorer in eval results — so agentic overhead is visible alongside cost and latency

---

## Usage

```bash
# Via CLI eval runner
python scripts/run_eval.py --strategy tool_use

# Via direct Inspect eval (SQL questions only)
python scripts/run_eval.py --strategy tool_use --dataset golden

# Via API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many orders were cancelled and what is the cancellation policy?",
    "strategy": "tool_use"
  }'
```

---

## Evaluation dataset

`datasets/golden/policy_questions.json` contains 10 questions designed specifically to expose what `tool_use` uniquely handles:

**5 pure KB questions (p01–p05):** questions with no SQL answer
- "What is the return window for purchased products?"
- "What payment methods do you accept?"
- "How long does a refund take to process?"
- "Are digital products refundable?"
- "What are your customer support hours?"

**5 hybrid questions (h01–h05):** require both DB data and KB policy
- "How many orders were cancelled and what is your cancellation policy?"
- "Who are the top 3 customers by spend and what support tier do they qualify for?"
- "What is the total revenue from completed orders and how can customers pay?"
- "Which customers never completed an order — what does our re-engagement policy say?"
- "What are the most popular product categories and are any non-refundable?"

Scoring for these questions uses `answer_quality` (LLM judge checking for expected keywords in the answer) rather than `result_match` (which compares SQL row outputs).

---

## Benchmark

`scripts/benchmark_agent.py` compares all three main strategies across all question types:

```bash
# Full benchmark — all 3 parts
python scripts/benchmark_agent.py

# SQL questions only (Part 1) — compares zero_shot vs few_shot_dynamic vs tool_use on 15 golden questions
python scripts/benchmark_agent.py --parts 1

# Policy + hybrid only (Parts 2 and 3)
python scripts/benchmark_agent.py --parts 2 3
```

---

## Trade-offs

| | `zero_shot` | `few_shot_dynamic` | `tool_use` |
|---|---|---|---|
| SQL accuracy (result_match) | 0.667 | **0.933** | ~0.800 |
| Policy question accuracy | ❌ N/A | ❌ N/A | **~0.900** |
| Avg cost per question | ~$0.0002 | ~$0.0003 | ~$0.001 |
| Avg latency per question | ~2s | ~2s | ~5–10s |
| Avg tool calls | 0 | 0 | 2–4 |

**Why `tool_use` has lower SQL accuracy than `few_shot_dynamic`:** the tool-use agent uses multi-turn calling rather than a direct SQL prompt, so it doesn't benefit from the same in-context examples in the same way. It also sometimes calls `get_schema` unnecessarily on simple questions, adding latency.

**Rule of thumb:**
- Pure SQL product → `few_shot_dynamic`
- General assistant (SQL + policies) → `tool_use`
- Cost-sensitive → `zero_shot`

---

## Design notes

- **Tools return strings, not exceptions.** Error messages from DuckDB are returned as strings so the LLM receives them as `tool` role messages, reads the error, and produces a corrected query. This mirrors how a developer uses a REPL.
- **`result.sql` is populated from the last `query_database` call** — so the existing SQL scorers (`syntax_valid`, `execution_ok`, `result_match`) continue to work when `tool_use` is run in the standard eval harness.
- **KB retriever is a module-level singleton** in `tools.py` — indexed on first call, reused across all requests. Re-indexing on every request would waste ~500ms.
- **`avg_tool_calls` scorer** makes agentic overhead visible in eval results alongside cost and latency.
