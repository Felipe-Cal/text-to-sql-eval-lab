# FastAPI service

The agent and eval suite are exposed as an HTTP service so they can be integrated with frontends, CI pipelines, Slack bots, or any other system.

## Start the server

```bash
uvicorn src.api.main:app --reload
```

The server starts at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the auto-generated interactive Swagger UI.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/query` | Run the text-to-SQL agent on a question |
| `POST` | `/evals/run` | Start an eval run in the background; returns a `job_id` |
| `GET` | `/evals/{job_id}` | Poll an eval job's status and results |

---

## `POST /query` — run the agent

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
- `strategy` — optional; one of `zero_shot`, `few_shot_static`, `few_shot_dynamic`, `chain_of_thought`, `rag`, `routed`; defaults to `zero_shot`

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

When using `strategy: "routed"`, the response also includes:

```json
{
  "routed_difficulty": "hard",
  "router_method": "rule_based"
}
```

---

## `POST /evals/run` — start an eval job

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

## `GET /evals/{job_id}` — poll eval results

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
      "syntax_valid":     { "accuracy": 1.0 },
      "execution_ok":     { "accuracy": 0.933 },
      "result_match":     { "accuracy": 0.933 },
      "semantic_judge":   { "accuracy": 0.933 },
      "avg_attempts":     { "mean": 1.07 },
      "avg_cost":         { "mean": 0.000082 },
      "avg_latency":      { "mean": 1.41 },
      "avg_total_tokens": { "mean": 487 },
      "retrieval_recall": { "mean": 1.0 }
    }
  },
  "error": null
}
```

---

## Design notes

- **`/query` is non-blocking** — the LiteLLM call runs in a thread pool via `asyncio.to_thread`, so FastAPI's event loop is never blocked.
- **`/evals/run` is fire-and-forget** — eval jobs run in the background; you poll for results. This is necessary because a full 15-question run with judge scoring takes 1–3 minutes.
- **In-memory job store** — jobs are stored in a Python dict keyed by UUID. Fine for local use; swap for Redis/database in production.
