# Prompt strategies

Seven strategies are available:

| Strategy | Description | Extra cost |
|---|---|---|
| `zero_shot` | Schema + question only (default) | none |
| `few_shot_static` | Prepends 3 fixed examples (easy/medium/hard) | none |
| `few_shot_dynamic` | Embeds the question, picks 3 most similar golden examples by cosine similarity | 1 embedding call |
| `chain_of_thought` | Instructs the model to reason step-by-step before writing SQL | none (larger output) |
| `rag_dense` | Semantic retrieval using Qdrant (cosine similarity on embeddings) | 1 embedding call |
| `rag_sparse` | Keyword retrieval using Qdrant (BM25 token matching) | none (local compute) |
| `rag_hybrid` | Fusion of dense and sparse retrieval (Reciprocal Rank Fusion) | 1 embedding call |
| `routed` | Classifies question difficulty (rule-based → embedding k-NN), then resolves to best model + strategy | see [routing.md](routing.md) |
| `tool_use` | Agentic: the LLM is given tools (`query_database`, `search_knowledge_base`, `get_schema`) and decides which to call. Handles SQL, policy, and hybrid questions in a single interface. | multiple LLM round-trips |

**Recommended strategy for pure SQL:** `few_shot_dynamic` — +40% result_match vs zero_shot, only +4s latency.

**Recommended strategy for general assistant (SQL + policies):** `tool_use` — see [tool-use.md](tool-use.md).

## Advanced RAG schema linking (Qdrant Hybrid Search)

Rather than passing all 50 table definitions to the model (which balloons the prompt and confuses smaller models), we utilize an advanced RAG pipeline powered by **Qdrant**:

1. **`rag_dense`**: Standard semantic embedding search. Good for finding tables by meaning (e.g. "order history" -> "orders").
2. **`rag_sparse`**: Lexical search using **BM25** (via FastEmbed). Crucial for exact keyword matches (e.g. searching for a specific column name like `sku_id_v2`).
3. **`rag_hybrid` (Staff Recommendation)**: Fuses both search results using **Reciprocal Rank Fusion (RRF)**. This is the state-of-the-art approach for robust retrieval.

This simulates a real enterprise use case where the schema is too large to fit in the prompt.

## LangGraph Orchestration & Self-Correction

Regardless of the prompt strategy used, the core SQL generation loop (`generate_sql` in `src/agent/agent.py`) is powered by fundamentally agentic **LangGraph** orchestration. 

We utilize a **Directed Acyclic Graph (DAG)** to orchestrate a deterministic self-correction state machine:
1. **`generate_node`**: Calls the selected model and outputs a candidate SQL query.
2. **`execute_node`**: Runs the query safely against DuckDB in `read_only=True` mode.
3. **Execution Routing**: If DuckDB throws an error (e.g. "Column not found"), the LangGraph conditional edge routes execution back to the `generate_node`. It passes the SQL execution trace verbatim to the LLM so it can learn and self-correct on its next attempt (up to `max_retries`).

This strict State Machine approach guarantees idempotency and isolates generation from logic loops—skills heavily utilized in Staff-level Agent designs.

## Formal Diagnostic Evaluation (DeepEval)

Beyond the binary `result_match` scorer, we utilize **DeepEval** to provide high-fidelity diagnostics for every SQL generation. These metrics use **LLM-as-a-Judge** with specialized reasoning to pinpoint failure modes:

*   **Faithfulness**: Quantifies how well the generated SQL is grounded in the retrieved schema. Detects if the model "hallucinated" a table or column name (Scale 0-1).
*   **Answer Relevancy**: Measures the alignment between the user's natural language question and the generated SQL's intent.
*   **SQL Quality (G-Eval)**: A custom rubric-based metric that scores the "Staff-level" quality of the SQL, rewarding efficient JOINs, window functions, and clean aliasing.

These metrics are essential for establishing a production "Refusal & Hallucination" baseline before deployment.

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
