# Model routing

Rather than using the same model and strategy for every question, the router classifies each question's difficulty and dispatches it to the most appropriate model + strategy — saving cost on easy questions while ensuring hard ones get the best treatment.

## How it works

```
Question comes in
      │
      ▼
Stage 1: Rule-based classifier (free, instant)
  Score question against keyword patterns
  (e.g. "never", "NOT IN", "month-by-month" → hard)
      │
      ├── Confidence ≥ 0.30 → use rule-based result
      └── Confidence < 0.30 → Stage 2: Embedding k-NN
                                  Embed question, find K nearest
                                  golden examples, majority vote
      │
      ▼
Routing table
  easy   → DEFAULT_MODEL + zero_shot
  medium → DEFAULT_MODEL + few_shot_dynamic
  hard   → HARD_MODEL    + few_shot_dynamic
```

## Routing table

| Difficulty | Model | Strategy | Rationale |
|---|---|---|---|
| easy | `DEFAULT_MODEL` | `zero_shot` | Simple queries need no examples |
| medium | `DEFAULT_MODEL` | `few_shot_dynamic` | JOINs + aggregations benefit from examples |
| hard | `HARD_MODEL` | `few_shot_dynamic` | Escalate to stronger model if configured |

Set `HARD_MODEL` in `.env` to escalate hard questions to a stronger model (e.g. `openai/gpt-4o`). Without it, the router still routes to different strategies on the same model.

## Using the routed strategy

```bash
# Via CLI eval
python scripts/run_eval.py --strategy routed

# Via API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which customers have never placed a completed order?", "strategy": "routed"}'
```

The response includes `routed_difficulty` and `router_method` so you can see how the decision was made:

```json
{
  "sql": "SELECT name FROM customers WHERE ...",
  "strategy": "few_shot_dynamic",
  "routed_difficulty": "hard",
  "router_method": "rule_based"
}
```

## Benchmark routing

Compare routing against fixed baselines to measure the quality/cost tradeoff:

```bash
python scripts/benchmark_routing.py
```

This runs three configurations side-by-side:

| Configuration | What it uses |
|---|---|
| `baseline_cheap` | always `gpt-4o-mini` + `zero_shot` |
| `baseline_best` | always `gpt-4o-mini` + `few_shot_dynamic` |
| `routed` | router picks strategy (and model) per question |

The goal: `routed` quality ≈ `baseline_best`, `routed` cost < `baseline_best`.

## Implementation notes

- Golden embeddings are cached via `@lru_cache` — not re-embedded on every request.
- Two-stage design keeps latency low: the rule-based stage is free and handles most questions; the embedding k-NN fallback fires only when confidence is low.
- `AgentResult.routed_difficulty` and `router_method` record how the decision was made for observability.
