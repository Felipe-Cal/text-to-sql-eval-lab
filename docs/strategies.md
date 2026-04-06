# Prompt strategies

Seven strategies are available:

| Strategy | Description | Extra cost |
|---|---|---|
| `zero_shot` | Schema + question only (default) | none |
| `few_shot_static` | Prepends 3 fixed examples (easy/medium/hard) | none |
| `few_shot_dynamic` | Embeds the question, picks 3 most similar golden examples by cosine similarity | 1 embedding call |
| `chain_of_thought` | Instructs the model to reason step-by-step before writing SQL | none (larger output) |
| `rag` | Embeds the question and retrieves the top-K most semantically relevant table definitions from the 50-table DWH, rather than sending the full schema | 1 embedding call |
| `routed` | Classifies question difficulty (rule-based → embedding k-NN), then resolves to best model + strategy | see [routing.md](routing.md) |
| `tool_use` | Agentic: the LLM is given tools (`query_database`, `search_knowledge_base`, `get_schema`) and decides which to call. Handles SQL, policy, and hybrid questions in a single interface. | multiple LLM round-trips |

**Recommended strategy for pure SQL:** `few_shot_dynamic` — +40% result_match vs zero_shot, only +4s latency.

**Recommended strategy for general assistant (SQL + policies):** `tool_use` — see [tool-use.md](tool-use.md).

## RAG schema linking (`rag` strategy)

Rather than passing all 50 table definitions to the model (which balloons the prompt and confuses smaller models), the `rag` strategy:

1. Embeds the natural language question using the configured embedding model.
2. Pre-computes and **caches** embeddings for all 50 table definitions (only once per process).
3. Ranks tables by **cosine similarity** and injects only the top-K (default: 5) into the prompt.
4. Tracks `retrieval_recall` as an eval metric — what % of the tables required by the golden SQL were actually retrieved.

This simulates a real enterprise use case where the schema is too large to fit in the prompt.

## LangGraph Orchestration & Self-Correction

Regardless of the prompt strategy used, the core SQL generation loop (`generate_sql` in `src/agent/agent.py`) is powered by fundamentally agentic **LangGraph** orchestration. 

We utilize a **Directed Acyclic Graph (DAG)** to orchestrate a deterministic self-correction state machine:
1. **`generate_node`**: Calls the selected model and outputs a candidate SQL query.
2. **`execute_node`**: Runs the query safely against DuckDB in `read_only=True` mode.
3. **Execution Routing**: If DuckDB throws an error (e.g. "Column not found"), the LangGraph conditional edge routes execution back to the `generate_node`. It passes the SQL execution trace verbatim to the LLM so it can learn and self-correct on its next attempt (up to `max_retries`).

This strict State Machine approach guarantees idempotency and isolates generation from logic loops—skills heavily utilized in Staff-level Agent designs.

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
