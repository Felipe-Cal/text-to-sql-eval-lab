# Prompt strategies

Five strategies are available, controlling what context is prepended to the prompt before the schema and question:

| Strategy | Description | Extra cost |
|---|---|---|
| `zero_shot` | Schema + question only (default) | none |
| `few_shot_static` | Prepends 3 fixed examples (easy/medium/hard) | none |
| `few_shot_dynamic` | Embeds the question, picks 3 most similar golden examples by cosine similarity | 1 embedding call |
| `chain_of_thought` | Instructs the model to reason step-by-step before writing SQL | none (larger output) |
| `rag` | Embeds the question and retrieves the top-K most semantically relevant table definitions from the 50-table DWH, rather than sending the full schema | 1 embedding call |
| `routed` | Classifies question difficulty (rule-based → embedding k-NN), then resolves to best model + strategy | see [routing.md](routing.md) |

**Recommended strategy: `few_shot_dynamic`** — +40% result_match vs zero_shot, only +4s latency.

## RAG schema linking (`rag` strategy)

Rather than passing all 50 table definitions to the model (which balloons the prompt and confuses smaller models), the `rag` strategy:

1. Embeds the natural language question using the configured embedding model.
2. Pre-computes and **caches** embeddings for all 50 table definitions (only once per process).
3. Ranks tables by **cosine similarity** and injects only the top-K (default: 5) into the prompt.
4. Tracks `retrieval_recall` as an eval metric — what % of the tables required by the golden SQL were actually retrieved.

This simulates a real enterprise use case where the schema is too large to fit in the prompt.

## Running a strategy sweep

```bash
# One model, all strategies
python scripts/run_eval.py --strategies zero_shot few_shot_static few_shot_dynamic chain_of_thought

# Single strategy
python scripts/run_eval.py --strategy few_shot_dynamic

# Full matrix: models × strategies
python scripts/run_eval.py \
  --models openai/gpt-4o-mini anthropic/claude-haiku-4-5 \
  --strategies zero_shot few_shot_static few_shot_dynamic chain_of_thought
```
