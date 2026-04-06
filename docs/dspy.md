# DSPy prompt optimization

Rather than hand-crafting prompts, [DSPy](https://dspy.ai) treats prompts as learnable programs. The optimizer automatically finds the best few-shot demonstrations from your training data by trying candidates and measuring whether they actually improve SQL execution accuracy — no manual prompt tuning required.

## Core idea

`few_shot_static` and `few_shot_dynamic` both pick examples based on heuristics (hand-picked or embedding similarity). DSPy replaces this with **empirical search**: it tries many combinations of demos from the training set, executes the predicted SQL against DuckDB, and keeps the ones that produce correct rows. The examples are chosen because they **measurably help**, not because they look similar.

## How it works

```
Synthetic tuning dataset (datasets/synthetic/tuning.json — 40 questions)
            │
            ▼
     DSPy BootstrapFewShot
      ├── For each candidate demo combination:
      │     run SQLGenerator on training questions
      │     execute predicted SQL against DuckDB
      │     compute db_metric (1 if rows match, 0 otherwise)
      └── Keep top 4 bootstrapped + 4 labeled demos
            │
            ▼
  Compiled prompt saved to datasets/dspy_optimized_prompt.json
```

### Key components

**`TextToSQL` Signature** declares the task as typed I/O:
```python
class TextToSQL(dspy.Signature):
    """Generate a DuckDB SQL query for the given question and schema."""
    schema: str = dspy.InputField()
    question: str = dspy.InputField()
    sql: str = dspy.OutputField()
```

**`SQLGenerator` module** wraps the Signature in `dspy.ChainOfThought`, which automatically injects a reasoning step before the SQL output:
```python
class SQLGenerator(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought(TextToSQL)
```

**`db_metric`** is the optimization target — it executes the predicted SQL against DuckDB and compares normalized rows to expected:
```python
def db_metric(example, prediction, trace=None):
    try:
        rows = execute_query(prediction.sql)
        return normalize(rows) == normalize(example.expected_rows)
    except:
        return False
```

**`BootstrapFewShot`** tries combinations of training examples as few-shot demos, evaluates each with `db_metric`, and selects the subset that maximizes actual execution accuracy.

## Prompt strategy comparison

| Approach | How examples are chosen | Optimized for |
|---|---|---|
| `few_shot_static` | Hand-picked by the developer | Developer intuition |
| `few_shot_dynamic` | K-nearest by embedding similarity to the question | Semantic similarity |
| `dspy` / `BootstrapFewShot` | Tried empirically; kept only if they improve real execution accuracy | Actual metric (`result_match`) |

The key difference: DSPy demos are chosen because they were **empirically proven to improve SQL accuracy**, not because a human thought they looked right or because they were semantically similar.

## Run the optimizer

```bash
python scripts/optimize_prompt.py
```

This will:
1. Load `datasets/synthetic/tuning.json` (40 synthetic Q&A pairs)
2. Pre-fetch RAG schemas for each training question (cached after first call, so demos match the inference-time prompt format exactly)
3. Run `BootstrapFewShot` — expect a few minutes, one LLM call + one DuckDB execution per candidate
4. Save the compiled prompt to `datasets/dspy_optimized_prompt.json`

## Use the optimized prompt

```bash
python scripts/run_eval.py --strategy dspy
```

The `dspy` strategy in `agent.py` loads `datasets/dspy_optimized_prompt.json` at startup. Inference is fast — the compiled demos are just static text in the prompt; no optimizer runs at query time.

## DSPy vs. other strategies — when to use which

- **`zero_shot`**: no examples, cheapest. Good for a baseline or cost-constrained deployment.
- **`few_shot_dynamic`**: 3 embedding-similar examples, one extra API call. Best for SQL quality in production.
- **`dspy`**: empirically optimized examples, no extra call at inference time. Best when you have a training set and want to maximize accuracy without paying for embedding calls per query.
- **`dspy` re-optimization**: run `optimize_prompt.py` whenever your training data changes significantly (new schema, new question types). The compiled JSON is checked in so it can be used immediately without re-optimizing.

## Key DSPy concepts

| Concept | What it does in this project |
|---|---|
| `dspy.Signature` | Declares `schema + question → sql` as typed I/O — the task definition |
| `dspy.ChainOfThought` | Wraps the Signature and auto-adds a reasoning step before `sql` output |
| `dspy.Module` | Composes Signatures into a callable, serializable program |
| `BootstrapFewShot` | Optimizer: bootstraps demos from training set using `db_metric`, selects best subset |
| `db_metric` | Custom metric: executes SQL against DuckDB, normalizes rows, returns 1 or 0 |
| `.save()` / `.load()` | Serializes the compiled (demo-enriched) prompt to JSON for reuse |

## Implementation note on token tracking

DSPy manages its own LLM calls internally, which means token counts and costs are not automatically tracked by LiteLLM's usage tracking. The `dspy` strategy in `agent.py` extracts token usage from DSPy's interaction history manually after each call. If token counts show as 0 in eval results for the `dspy` strategy, this is expected — it means DSPy's history was empty for that call (can happen if the compiled module's LM handle differs from the active LiteLLM session).
