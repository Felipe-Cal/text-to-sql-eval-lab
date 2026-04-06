# Model routing

Rather than using the same model and strategy for every question, the router classifies each question's difficulty and dispatches it to the most cost-effective model + strategy combination. Easy questions get a cheap, fast response. Hard questions get escalated.

## Why routing matters

A question like "How many customers are there?" requires a simple `COUNT(*)` — no examples needed, any model gets it right. A question like "Which customers have never placed a completed order?" requires a `NOT IN` anti-join pattern — few-shot examples and possibly a stronger model make the difference. Routing applies the right tool to each problem instead of overpaying on every question.

## Architecture

```
Question comes in
      │
      ▼
Stage 1: Rule-based classifier (free, ~0ms)
  Scores the question against keyword pattern lists.
  Assigns confidence based on how many patterns matched.
      │
      ├── Confidence ≥ 0.30 → use this result directly
      │
      └── Confidence < 0.30 → Stage 2: Embedding k-NN (1 API call)
                                  Embed the question.
                                  Find K=5 nearest golden examples.
                                  Take majority vote of their difficulty labels.
      │
      ▼
Routing decision → resolved to (model, strategy) pair
```

### Stage 1: Rule-based patterns

The rule-based classifier scores questions against keyword lists for each difficulty tier:

| Difficulty | Patterns that signal it |
|---|---|
| **easy** | single table keywords (`COUNT`, `DISTINCT`, `ORDER BY`, `WHERE`, `LIMIT`), questions about a single entity ("how many", "list all", "which ... are from") |
| **medium** | multi-table signals (`JOIN`, `GROUP BY`, aggregate functions like `AVG`, `SUM`, `ROUND`), date functions (`YEAR`, `MONTH`), subquery markers |
| **hard** | anti-join patterns (`NOT IN`, `NOT EXISTS`, `never`), `HAVING`, window functions, multi-table aggregation ("month-by-month", "by country", "top N ... by total") |

The classifier returns a `(difficulty, confidence)` tuple. Confidence is based on how many patterns matched and whether any contradictory signals appeared. The threshold is **0.30** — below that, the rule-based result is too uncertain and the embedding k-NN fallback is triggered.

### Stage 2: Embedding k-NN fallback

When rules don't fire confidently (e.g. ambiguous questions, unusual phrasing), the question is embedded and compared against the 15 golden questions' embeddings. The 5 nearest neighbors vote on difficulty by majority. The golden embeddings are computed once and cached via `@lru_cache` — the k-NN lookup itself is just a cosine similarity scan over 15 vectors.

## Routing table

| Difficulty | Model | Strategy | Rationale |
|---|---|---|---|
| easy | `DEFAULT_MODEL` | `zero_shot` | Simple queries need no examples — saves token cost |
| medium | `DEFAULT_MODEL` | `few_shot_dynamic` | JOINs and aggregations benefit from 3 similar examples |
| hard | `HARD_MODEL` | `few_shot_dynamic` | Escalate to stronger model; examples are critical |

Set `HARD_MODEL` in `.env` to escalate hard questions to a more capable model (e.g. `openai/gpt-4o`). If `HARD_MODEL` is not set, the router still picks the right strategy — it just uses `DEFAULT_MODEL` for all tiers.

## Usage

```bash
# Via CLI eval
python scripts/run_eval.py --strategy routed

# Via API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which customers have never placed a completed order?", "strategy": "routed"}'
```

The response includes `routed_difficulty` and `router_method` so you can see exactly how the decision was made:

```json
{
  "sql": "SELECT name FROM customers WHERE id NOT IN ...",
  "strategy": "few_shot_dynamic",
  "routed_difficulty": "hard",
  "router_method": "rule_based"
}
```

`router_method` is either `"rule_based"` (fast, no API call) or `"embedding_knn"` (1 embedding API call, fired when rule confidence < 0.30).

## Benchmark routing

Compare routing against fixed baselines:

```bash
python scripts/benchmark_routing.py
```

Three configurations run side-by-side across all 15 questions:

| Configuration | Model | Strategy |
|---|---|---|
| `baseline_cheap` | gpt-4o-mini | always zero_shot |
| `baseline_best` | gpt-4o-mini | always few_shot_dynamic |
| `routed` | gpt-4o-mini (+ HARD_MODEL for hard questions) | per-difficulty |

**Expected result:** `routed` quality ≈ `baseline_best`, `routed` cost < `baseline_best` (because easy questions use zero_shot instead of incurring embedding + extra example tokens).

## Implementation notes

- **`@lru_cache` on golden embeddings** — the 15 golden question embeddings are computed once per process and cached. k-NN lookup is effectively free after first call.
- **`routed` is a meta-strategy** — it resolves to a real strategy (e.g. `few_shot_dynamic`) before the LLM call. The resolved strategy name is visible in `AgentResult.strategy`.
- **`AgentResult.routed_difficulty` and `router_method`** are always populated when `routed` is used, whether the rule-based or embedding path fired. This is important for observability — you can see how the router is behaving across your eval runs.
- Source: `src/agent/router.py`
