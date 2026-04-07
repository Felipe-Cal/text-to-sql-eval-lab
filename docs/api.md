# FastAPI service

The agent and eval suite are exposed as an HTTP service so they can be consumed by frontends, CI pipelines, Slack bots, or any external system — not just from the CLI.

## Start the server

```bash
uvicorn src.api.main:app --reload
```

The server starts at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the auto-generated interactive Swagger UI where you can try every endpoint directly in the browser.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `POST` | `/query` | Run the text-to-SQL agent on a question |
| `POST` | `/query/stream` | Same as `/query` but streams progress as SSE events |
| `POST` | `/evals/run` | Start an eval run in the background; returns a `job_id` immediately (202 Accepted) |
| `GET` | `/evals/{job_id}` | Poll an eval job's status and retrieve results when complete |

---

## `POST /query` — run the agent

Runs the text-to-SQL agent on a natural language question and returns the generated SQL, execution results, and full metadata.

**Request body:**

```json
{
  "question": "What are the top 3 customers by total spend?",
  "model": "openai/gpt-4o-mini",
  "strategy": "few_shot_dynamic"
}
```

| Field | Required | Default | Description |
|---|---|---|---|
| `question` | ✅ | — | Natural language question |
| `model` | ❌ | `DEFAULT_MODEL` env var | LiteLLM model string, e.g. `openai/gpt-4o-mini`, `anthropic/claude-haiku-4-5` |
| `strategy` | ❌ | `zero_shot` | One of the nine strategies — see [strategies.md](strategies.md) |

**Response:**

```json
{
  "question": "What are the top 3 customers by total spend?",
  "sql": "SELECT c.name, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_spend FROM customers c JOIN orders o ON o.customer_id = c.id JOIN order_items oi ON oi.order_id = o.id GROUP BY c.name ORDER BY total_spend DESC LIMIT 3",
  "data": [
    {"name": "Alice Martin", "total_spend": 599.92},
    {"name": "Bob Schmidt",  "total_spend": 384.92},
    {"name": "David Chen",   "total_spend": 114.98}
  ],
  "model": "openai/gpt-4o-mini",
  "strategy": "few_shot_dynamic",
  "reasoning": null,
  "prompt_tokens": 412,
  "completion_tokens": 58,
  "cost": 0.00008,
  "latency": 1.23,
  "attempts": 1,
  "trace_id": "abc123",
  "routed_difficulty": null,
  "router_method": null,
  "answer": null,
  "tool_calls": []
}
```

**Field reference:**

| Field | Description |
|---|---|
| `sql` | The final generated SQL query |
| `data` | Rows returned by executing `sql` against DuckDB (up to 1,000 rows, as a list of dicts) |
| `reasoning` | Non-null only for `chain_of_thought` — contains the model's step-by-step reasoning text |
| `attempts` | How many times the self-correction loop ran (1 = first attempt succeeded; up to 4) |
| `cost` | Estimated USD cost for the LLM call(s), based on token counts |
| `trace_id` | Links to the Langfuse trace for this call (if `LANGFUSE_*` env vars are configured) |
| `routed_difficulty` | Non-null only for `routed` strategy — `"easy"`, `"medium"`, or `"hard"` |
| `router_method` | Non-null only for `routed` — `"rule_based"` or `"embedding_knn"` |
| `answer` | Non-null only for `tool_use` — the model's natural language synthesis of tool results |
| `tool_calls` | Non-null only for `tool_use` — ordered list of every tool invocation (name + arguments + result) |

**`tool_use` response example:**

When `strategy: "tool_use"`, the model calls tools before answering. All calls are logged:

```json
{
  "sql": "SELECT COUNT(*) FROM orders WHERE status = 'cancelled'",
  "answer": "There are 2 cancelled orders. Per our policy, cancelled orders are not eligible for refunds after 48 hours.",
  "tool_calls": [
    {
      "tool": "get_schema",
      "input": {"table_name": "orders"},
      "output": "orders(id, customer_id, order_date DATE, status VARCHAR)"
    },
    {
      "tool": "query_database",
      "input": {"sql": "SELECT COUNT(*) FROM orders WHERE status = 'cancelled'"},
      "output": "[{\"count_star()\": 2}]"
    },
    {
      "tool": "search_knowledge_base",
      "input": {"query": "cancellation policy refund"},
      "output": "Cancelled orders are not eligible for refunds after 48 hours..."
    }
  ]
}
```

Notice the agent's natural `get_schema → query_database → search_knowledge_base` discovery pattern — it checked the schema before writing SQL, then fetched policy information to complete the answer.

---

## `POST /query/stream` — stream agent progress

Same request body as `/query`. Returns a `text/event-stream` response where each line is an SSE event. Events arrive in real time as the LLM generates tokens, tools are called, and retries happen — so a frontend can show a live typing indicator, tool call logs, and error recovery without waiting for the full response.

**Event format:**

Every event is `data: <JSON>\n\n`. The JSON object always has a `"type"` field:

| Event type | Fields | When it fires |
|---|---|---|
| `start` | `strategy`, `model` | Always first — confirms which strategy and model are running |
| `sql_token` | `content` | Once per LLM output chunk for token-streamable strategies (zero_shot, few_shot_*, chain_of_thought, rag_*) |
| `tool_call` | `tool`, `input` | `tool_use` only — fired before each tool execution |
| `tool_result` | `tool`, `output`, `success` | `tool_use` only — fired after each tool returns |
| `retry` | `attempt`, `error` | When the generated SQL fails execution and the agent retries |
| `done` | `sql`, `cost`, `latency`, `attempts`, `prompt_tokens`, `completion_tokens` | Always last — contains the final SQL and full metadata |
| `error` | `message` | If an unrecoverable exception occurs |

**Example — zero_shot (token streaming):**

```
data: {"type": "start", "strategy": "zero_shot", "model": "openai/gpt-4o-mini"}

data: {"type": "sql_token", "content": "SELECT"}
data: {"type": "sql_token", "content": " COUNT"}
data: {"type": "sql_token", "content": "(*)"}
data: {"type": "sql_token", "content": " FROM customers"}

data: {"type": "done", "sql": "SELECT COUNT(*) FROM customers", "cost": 0.00004,
       "latency": 0.94, "attempts": 1, "prompt_tokens": 310, "completion_tokens": 8}
```

**Example — tool_use (tool call events):**

```
data: {"type": "start", "strategy": "tool_use", "model": "openai/gpt-4o-mini"}

data: {"type": "tool_call", "tool": "get_schema", "input": {"table_name": "orders"}}
data: {"type": "tool_result", "tool": "get_schema", "output": "orders(id, customer_id, order_date, status)", "success": true}

data: {"type": "tool_call", "tool": "query_database", "input": {"sql": "SELECT COUNT(*) FROM orders WHERE status = 'cancelled'"}}
data: {"type": "tool_result", "tool": "query_database", "output": "[{\"count_star()\": 2}]", "success": true}

data: {"type": "done", "sql": "SELECT COUNT(*) FROM orders WHERE status = 'cancelled'",
       "cost": 0.00012, "latency": 3.1, "attempts": 2, "prompt_tokens": 580, "completion_tokens": 62}
```

**Example — retry (SQL error recovered):**

```
data: {"type": "start", "strategy": "zero_shot", "model": "openai/gpt-4o-mini"}

data: {"type": "sql_token", "content": "SELECT total FROM ordrs"}

data: {"type": "retry", "attempt": 2, "error": "Table with name ordrs does not exist!"}

data: {"type": "sql_token", "content": "SELECT total FROM orders"}

data: {"type": "done", "sql": "SELECT total FROM orders", "cost": 0.00008,
       "latency": 2.4, "attempts": 2, "prompt_tokens": 420, "completion_tokens": 16}
```

**Consuming the stream in JavaScript:**

```javascript
const response = await fetch('/query/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question: 'How many customers are there?', strategy: 'zero_shot' }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  for (const line of decoder.decode(value).split('\n')) {
    if (!line.startsWith('data: ')) continue;
    const event = JSON.parse(line.slice(6));
    if (event.type === 'sql_token') process.stdout.write(event.content);
    if (event.type === 'done') console.log('\nFinal SQL:', event.sql);
  }
}
```

**Strategy compatibility:**

| Strategy | sql_token events | tool_call/tool_result | Notes |
|---|---|---|---|
| `zero_shot`, `few_shot_*`, `chain_of_thought`, `rag_*` | ✅ | ❌ | Full token streaming |
| `tool_use` | ❌ | ✅ | Tool events stream in real time; no raw SQL tokens |
| `dspy` | ❌ | ❌ | Completes fully, emits start + done only |
| `routed` | Depends | Depends | Resolves to a real strategy first, then that strategy's events stream |

---

## `POST /evals/run` — start an eval job

Eval runs take 1–3 minutes (LLM calls + judge scoring for 15 questions). This endpoint returns immediately with a `job_id` and runs the evaluation in a background thread. Poll `/evals/{job_id}` to check status.

**Request body:**

```json
{
  "model": "openai/gpt-4o-mini",
  "strategy": "few_shot_dynamic",
  "difficulty": "hard",
  "judge_model": "openai/gpt-4o"
}
```

| Field | Required | Default | Description |
|---|---|---|---|
| `model` | ❌ | `DEFAULT_MODEL` | Model to run the agent with |
| `strategy` | ❌ | `zero_shot` | Prompt strategy |
| `difficulty` | ❌ | all | Filter to `"easy"`, `"medium"`, or `"hard"` questions only |
| `judge_model` | ❌ | `JUDGE_MODEL` env var | Model used for `semantic_judge` scoring — override per-run without changing `.env` |

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

## `GET /evals/{job_id}` — poll eval results

**Response when complete:**

```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "completed",
  "started_at": "2026-04-03T10:00:00Z",
  "finished_at": "2026-04-03T10:02:35Z",
  "results": {
    "total_samples": 15,
    "scores": {
      "syntax_valid":        { "accuracy": 1.000 },
      "execution_ok":        { "accuracy": 0.933 },
      "result_match":        { "accuracy": 0.933 },
      "semantic_judge":      { "accuracy": 0.933 },
      "avg_attempts":        { "mean": 1.07 },
      "avg_cost":            { "mean": 0.000082 },
      "avg_latency":         { "mean": 1.41 },
      "avg_total_tokens":    { "mean": 487 },
      "avg_tool_calls":      { "mean": 0.0 },
      "retrieval_recall":    { "mean": 1.0 }
    }
  },
  "error": null
}
```

**Scorer reference:**

| Scorer | What it measures | Range |
|---|---|---|
| `syntax_valid` | Can sqlglot parse the SQL? (no DB needed) | 0 or 1 |
| `execution_ok` | Does the SQL run against DuckDB without error? | 0 or 1 |
| `result_match` | Do the returned rows match the golden expected rows? 1.0 = exact, 0.5 = right shape wrong values, 0.0 = wrong | 0, 0.5, or 1 |
| `semantic_judge` | LLM judge (via Instructor + Pydantic) rates the SQL as correct / partial / incorrect | 0, 0.5, or 1 |
| `avg_attempts` | Mean number of self-correction retries per question (1 = first try succeeded) | ≥ 1 |
| `avg_cost` | Mean USD cost per question across all LLM calls | float |
| `avg_latency` | Mean wall-clock seconds per question | float |
| `avg_total_tokens` | Mean prompt + completion tokens per question | int |
| `avg_tool_calls` | Mean number of tool calls per question (0 for non-agentic strategies) | ≥ 0 |
| `retrieval_recall` | For RAG strategies: fraction of required tables that appeared in top-K retrieved tables | 0–1 |

**Status values:** `"running"` → `"completed"` or `"failed"`. If `"failed"`, the `error` field contains the traceback.

---

## Design notes

- **`/query` is non-blocking** — the LiteLLM call runs in a thread pool via `asyncio.to_thread`. FastAPI's async event loop is never blocked waiting for LLM responses.
- **`/query/stream` uses structured SSE** — `agenerate_sql_stream` is an async generator that yields event dicts; the route is a thin serializer. All strategies emit a `start` event first and a `done` event last, so clients don't need to special-case any strategy. The retry loop is built directly into the streaming path — if the generated SQL fails execution, the agent emits a `retry` event, appends the error to the conversation, and streams a corrected attempt (up to `max_retries=3`). Token usage in streaming responses may be 0 for some providers that don't include usage in stream chunks.
- **`/evals/run` is fire-and-forget** — the eval job runs in a `BackgroundTask` thread; you poll for results. A full 15-question run with judge scoring takes 1–3 minutes.
- **In-memory job store** — eval jobs are stored in a Python dict keyed by UUID. Fine for local use. For production, swap with Redis or a database so jobs survive server restarts.
- **CORS** — all origins are allowed (`*`). Suitable for local development; restrict in production.
