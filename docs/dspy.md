# DSPy prompt optimization

Rather than hand-crafting prompts, [DSPy](https://dspy.ai) treats prompts as learnable programs. The optimizer automatically finds the best few-shot demonstrations from your training data by trying candidates and keeping the ones that actually improve the metric — no manual prompt tuning required.

## How it works

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
3. The **`db_metric`** function is the optimization target: it executes the predicted SQL against DuckDB and returns `1` if the rows exactly match the expected result, `0` otherwise.
4. **`BootstrapFewShot`** tries many few-shot demo combinations from the synthetic tuning set, evaluates each one with `db_metric`, and keeps the top 4 bootstrapped + 4 labeled demos that maximize real execution accuracy.
5. The compiled module (with its optimized demonstrations baked in) is saved as JSON and can be loaded for inference without re-running the optimizer.

## Run the optimizer

```bash
python scripts/optimize_prompt.py
```

This will:
1. Load `datasets/synthetic/tuning.json`
2. Fetch RAG schemas for each training question (cached after first call)
3. Run `BootstrapFewShot` optimization
4. Save the compiled prompt to `datasets/dspy_optimized_prompt.json`

Expect a few minutes — the optimizer makes one LLM call + one DuckDB execution per candidate demo tried.

## DSPy vs manual prompt strategies

| Approach | How examples are chosen | Optimized for |
|---|---|---|
| `few_shot_static` | Hand-picked by the developer | Developer intuition |
| `few_shot_dynamic` | Nearest neighbor by embedding similarity to the question | Semantic similarity |
| DSPy `BootstrapFewShot` | Automatically selected by trying candidates and measuring real execution accuracy | Actual task metric (result_match) |

The key difference: DSPy demos are chosen because they were **empirically proven to improve SQL execution accuracy**, not because a human thought they looked like good examples or because they were semantically similar.

## Key DSPy concepts used

| Concept | What it does |
|---|---|
| `dspy.Signature` | Declares the task as typed I/O: `schema + question → sql` |
| `dspy.ChainOfThought` | Adds an automatic reasoning step before the output field |
| `dspy.Module` | Wraps the pipeline into a composable, optimizable program |
| `BootstrapFewShot` | Optimizer that bootstraps few-shot demos from a training set using a metric |
| `db_metric` | Custom metric: executes SQL against DuckDB, compares rows to expected |
| `.save()` / `.load()` | Serializes the compiled (optimized) prompt for reuse |
