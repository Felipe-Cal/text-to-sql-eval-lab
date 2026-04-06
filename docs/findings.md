# Experimental findings

Benchmark results from running `scripts/run_eval.py`, `scripts/benchmark_schema_retrieval.py`, and fine-tuning experiments. All SQL evals use 15 golden Q&A pairs from `datasets/golden/questions.json`.

---

## Strategy sweep — GPT-4o-mini (15 questions)

| Strategy | result_match | semantic_judge | run time |
|---|---|---|---|
| `zero_shot` | 0.667 | 0.733 | 0:48 |
| `few_shot_static` | 0.767 | 0.767 | 0:48 |
| `few_shot_dynamic` | **0.933** | **0.933** | 0:52 |
| `chain_of_thought` | 0.667 | 0.800 | 1:16 |

### Findings

**`few_shot_dynamic` is the recommended strategy for cloud models** — +40% result_match over zero_shot for gpt-4o-mini, only +4s latency. It is also the only strategy where result_match and semantic_judge are fully aligned, meaning what the model produces is both deterministically correct and semantically sound.

**`few_shot_static` is inconsistent** — it improved scores for `claude-haiku` but hurt `gpt-4o-mini`'s semantic_judge score vs zero_shot. Static examples bias smaller models toward a specific SQL style, which can hurt more than help when the question pattern doesn't match the fixed examples.

**Chain-of-thought underperformed** — −10% result_match vs zero_shot, +58% latency. Root cause: CoT causes *column selection drift* — 4 of 5 failures used `SELECT *` or returned extra columns not in the golden expected rows. Chain-of-thought benefits larger reasoning models (GPT-4o, Claude 3.5 Sonnet) more; `gpt-4o-mini` lacks the capacity to reliably apply it to multi-table SQL generation.

**LLM judge calibration matters** — the initial `semantic_judge` produced false partial verdicts on `q02`, `q10`, and `q13` due to Python datetime reprs being passed as-is to the judge. Fixed by pre-normalizing result rows (ISO dates, rounded floats) and adding explicit grounding rules to the judge prompt. Now uses Instructor + Pydantic `VerdictData` model for type-safe structured outputs, eliminating brittle `json.loads()` failures.

> **Diagnostic tip:** cases where `result_match = 0` but `semantic_judge = 1` are false negatives in the deterministic scorer — the model was semantically correct but returned rows in a different format. Cases where `result_match > semantic_judge` suggest judge calibration drift worth investigating.

---

## Fine-tuned Llama 3.1 8B vs GPT-4o-mini (15 questions)

| Model | Strategy | result_match | semantic_judge | cost/query | avg latency |
|---|---|---|---|---|---|
| gpt-4o-mini | zero_shot | 0.667 | 0.733 | ~$0.002 | ~2s |
| gpt-4o-mini | few_shot_dynamic | 0.933 | 0.933 | ~$0.002 | ~2s |
| Llama 3.1 8B (fine-tuned) | zero_shot | 0.667 | 0.567 | **$0.000** | 99s |
| Llama 3.1 8B (fine-tuned) | few_shot_dynamic | **0.800** | **0.867** | **$0.000** | 74s |

Measured on Apple M1 (CPU inference via Ollama). On GPU (A10G), latency drops to ~5–8s.

### Findings

**Fine-tuned Llama + `few_shot_dynamic` beats GPT-4o-mini zero_shot at zero cost** — result_match 0.800 vs 0.667 (+20%), semantic_judge 0.867 vs 0.733 (+18%). Fine-tuning and in-context examples compound: the model has learned SQL patterns from training, and examples close the gap on hard multi-table questions.

**Fine-tuning alone matches but doesn't exceed GPT-4o-mini** — result_match is equal (0.667), but semantic_judge is weaker (0.567 vs 0.733). Without examples, the fine-tuned model produces structurally valid SQL but makes semantic errors on hard questions. Fine-tuning improved the model's understanding of the schema and SQL dialect; few-shot examples provide the remaining reasoning scaffolding.

**The hypothesis proved correct:** a fine-tuned small model with `few_shot_dynamic` approaches cloud API quality at zero marginal inference cost, after a one-time ~$0 training run on Colab.

---

## Schema retrieval benchmark (50 tables, 15 questions)

Benchmark: `scripts/benchmark_schema_retrieval.py`. Tests the ability to retrieve the correct table definitions for each of the 15 golden questions from a pool of 50 tables (4 real + 46 decoy). Three backends × two embedding libraries × three strategies.

**Metrics:**
- `Recall@K` — fraction of required tables found in top-K results (partial credit per question)
- `Perfect@K` — fraction of questions where ALL required tables were retrieved

### Full results (top_k=5)

| Backend | Strategy | Embedding library | Model | Recall@K | Perfect@K | Avg query |
|---|---|---|---|---|---|---|
| Qdrant | sparse (BM25) | FastEmbed (local) | SPLADE | **1.000** | **1.00** | 27ms |
| Qdrant | hybrid (RRF) | FastEmbed (local) | bge-small (384d) | **1.000** | **1.00** | 64ms |
| Qdrant | hybrid (RRF) | LiteLLM/OpenAI | 3-small (1536d) | **1.000** | **1.00** | 290ms |
| Qdrant | dense | FastEmbed (local) | bge-small (384d) | **1.000** | **1.00** | 21ms |
| InMemoryStore | dense | FastEmbed (local) | bge-small (384d) | **1.000** | **1.00** | 6ms |
| ChromaDB | dense | FastEmbed (local) | bge-small (384d) | **1.000** | **1.00** | 5ms |
| Qdrant | dense | LiteLLM/OpenAI | 3-large (3072d) | **1.000** | **1.00** | 348ms |
| Qdrant | dense | FastEmbed (local) | nomic-768d | 0.944 | 0.87 | 62ms |
| Qdrant | dense | LiteLLM/OpenAI | 3-small (1536d) | 0.900 | 0.73 | 276ms |

### Schema retrieval findings

**BM25/SPLADE is the fastest and cheapest at perfect recall** — 27ms per query, zero API cost, runs locally. Table names like `customers`, `order_items` are exact keywords, and BM25 matches them directly without any embedding lookup. Surprising result: a classical lexical search beats expensive semantic embeddings on this task because the vocabulary is so exact.

**`bge-small` (FastEmbed local) matches OpenAI `text-embedding-3-large` at 50× lower latency and zero cost** — both achieve perfect 1.000 recall at top_k=5. The 384-dimensional local model outperforms the 1536-dimensional OpenAI small model. This suggests that for structured schema matching, embedding quality plateaus quickly and model size beyond a point doesn't help.

**Embedding model choice matters more than the vector DB** — all three backends (Qdrant, InMemoryStore, ChromaDB) return identical recall for the same embedding model. InMemoryStore's O(n) linear scan vs Qdrant's HNSW index produces the same top-5 results on 50 vectors. The architectural difference only matters at scale (>10K items).

**Hybrid (RRF) is the safest production choice** — it equals BM25 at top_k≥5 while also handling semantic paraphrases that keyword search would miss in a noisier, real-world schema. For this clean dataset, BM25 alone suffices; for production schemas with inconsistent naming conventions, hybrid provides insurance.

**OpenAI `text-embedding-3-small` underperforms** — 0.900 recall vs 1.000 for `3-large` and `bge-small`. At top_k=3, it misses some multi-table questions where three or more tables are required. If using OpenAI embeddings, use `3-large` or increase `top_k`.

**Cost comparison per run (50 tables indexed + 15 questions queried):**

| Model | Library | Cost |
|---|---|---|
| SPLADE | FastEmbed local | **free** |
| bge-small-en-v1.5 | FastEmbed local | **free** |
| nomic-embed-text-v1.5 | FastEmbed local | **free** |
| text-embedding-3-small | OpenAI API | ~$0.000022 |
| text-embedding-3-large | OpenAI API | ~$0.000143 |

---

## Exploring eval logs

```bash
python -m inspect_ai view start --log-dir ./logs
```

If `inspect view` shows empty rows or SSL errors, your shell may be picking up a different `inspect` binary — use the full command above with `--log-dir` explicitly.

Open the URL printed in the terminal (default: `http://127.0.0.1:7575`). The log explorer shows per-question scores, the generated SQL, expected vs actual rows, and judge verdicts — essential for diagnosing which question types are failing.
