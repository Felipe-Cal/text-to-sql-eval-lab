# RAG Infrastructure

The RAG module covers two distinct retrieval use cases:

| Use case | What gets retrieved | Implementation |
|---|---|---|
| **Schema linking** | Relevant table definitions from a 50-table pool | `src/agent/schema_retriever.py` — Qdrant with dense + sparse + hybrid |
| **Knowledge base** | Relevant chunks from a 6 KB KB document | `src/rag/` — chunker + vector store + retriever |

Both retrieve context at query time and inject it into the LLM prompt — keeping prompts focused and avoiding irrelevant noise.

---

## Schema linking — Qdrant hybrid search (`src/agent/schema_retriever.py`)

### The problem it solves

The database schema has 50 table definitions (4 real + 46 decoy tables). Sending all 50 to the model on every request:
- Wastes ~3,000 tokens per call
- Confuses smaller models — they hallucinate columns from irrelevant tables
- Produces wrong SQL on hard questions that require specific joins

The schema retriever embeds the question and retrieves only the 3–5 most relevant tables. The model only sees those.

### Architecture

```
Question
    │
    ├── Dense embed (OpenAI or FastEmbed) ──────────┐
    │                                                │
    └── Sparse embed (SPLADE via FastEmbed) ─────────┤
                                                     ▼
                                          Qdrant in-memory
                                       (dual-vector collection)
                                                     │
                                    Reciprocal Rank Fusion (RRF)
                                                     │
                                                     ▼
                                        top-K table definitions
```

### Three retrieval strategies

| Strategy | How it works | Best for |
|---|---|---|
| `rag_dense` | Cosine similarity on embeddings | Semantic matches — "order history" → `orders` |
| `rag_sparse` | BM25/SPLADE keyword matching (FastEmbed, local, zero cost) | Exact keyword matches — column names, IDs |
| `rag_hybrid` | RRF fusion of dense + sparse | Best overall — handles both semantic and exact matches |

### Implementation

```python
from src.agent.schema_retriever import retrieve_schema

# Returns (formatted_schema_string, list_of_retrieved_table_defs)
schema_str, tables = retrieve_schema(
    question="Who are the top customers by revenue?",
    top_k=5,
    retrieval_type="hybrid",   # "dense" | "sparse" | "hybrid"
)
```

The retriever is a **singleton**: the Qdrant collection is built once on first call, cached in memory, and reused for the lifetime of the process. Re-embedding 50 tables on every request would waste ~1–2s per call.

- Dense embeddings use the `EMBEDDING_MODEL` env var (default: `openai/text-embedding-3-small`)
- Sparse embeddings use SPLADE (`prithivida/Splade_PP_en_v1`) via FastEmbed — runs locally, no API call

### Benchmark results

From `scripts/benchmark_schema_retrieval.py` — 15 golden questions, 50 tables, top_k=5:

| Strategy | Embedding | Library | Recall@5 | Avg query |
|---|---|---|---|---|
| sparse (BM25) | SPLADE | FastEmbed local | **1.000** | 27ms |
| dense | bge-small-en-v1.5 | FastEmbed local | **1.000** | 21ms |
| hybrid (RRF) | bge-small-en-v1.5 | FastEmbed local | **1.000** | 64ms |
| dense | text-embedding-3-large | OpenAI API | **1.000** | 348ms |
| dense | text-embedding-3-small | OpenAI API | 0.900 | 276ms |

See [findings.md](findings.md) for the full results table and analysis.

### Why Qdrant?

- **Native hybrid search** via `FusionQuery(fusion=Fusion.RRF)` — no custom score merging needed
- **Dual-vector collections** — dense and sparse in a single index, queried together
- **In-memory mode** (`QdrantClient(":memory:")`) — zero infrastructure for development
- **Production path**: replace `:memory:` with a Qdrant server URL, exact same API

---

## Knowledge base retrieval — chunked RAG (`src/rag/`)

The `src/rag/` module handles document-level retrieval for the `search_knowledge_base` tool in the `tool_use` agent and the KB benchmark.

Unlike schema retrieval (one-liner table definitions, already chunked), the knowledge base is a multi-section prose document that needs to be split into embeddable units before indexing.

### Architecture

```
Document file
    │
    ▼
Chunker       — splits text into embeddable units (by char, sentence, or line)
    │
    ▼
VectorStore   — embeds each chunk and stores it; queries by cosine similarity
    │
    ▼
DocumentRetriever — public API: index_file() + retrieve() + retrieve_text()
    │
    ▼
top-K chunks  — injected into LLM prompt as context
```

### Chunking strategies (`src/rag/chunker.py`)

| Chunker | How it splits | When to use |
|---|---|---|
| `FixedSizeChunker(chunk_size, overlap)` | Splits every N characters with M-character overlap | Fast baseline; language-agnostic; good for dense prose |
| `SentenceChunker(max_chunk_size, overlap_sentences)` | Groups complete sentences until max_chunk_size, with sentence overlap | Prose documents (FAQs, policies, transcripts) — preserves sentence meaning |
| `SchemaChunker()` | One chunk per line | Schema/table definitions — used implicitly in `schema_retriever.py` |

**Why overlap matters:** a key phrase like "30-day return window" might be split across two fixed-size chunks if you're unlucky. Overlap ensures each boundary is also covered by the adjacent chunk. `FixedSizeChunker` overlaps by characters; `SentenceChunker` by sentence count.

```python
from src.rag.chunker import SentenceChunker

chunks = SentenceChunker(max_chunk_size=512, overlap_sentences=1).chunk(
    text, metadata={"source": "return_policy.md"}
)
# Returns list[Chunk] — each with .text and .metadata
```

### Vector stores (`src/rag/vector_store.py`)

Both stores implement the same abstract interface: `add(chunks)`, `query(question, top_k)`, `reset()`, `count()`. Callers never need to know which backend is in use.

| Store | Persistence | Query algorithm | When to use |
|---|---|---|---|
| `InMemoryStore` | None — lost on restart | O(n) linear scan | Tests, small corpora, quick prototyping |
| `ChromaDBStore` | Disk (HNSW index, survives restarts) | Approximate nearest neighbor | Production, large corpora, multi-process deployments |

Embeddings are generated by the same `get_embeddings()` helper in both stores, controlled by the `EMBEDDING_MODEL` env var. To swap models, change the env var — no code changes needed.

```python
from src.rag.vector_store import create_store

store = create_store("chroma", collection_name="kb_docs", persist_dir="./chroma_db")
# or: create_store("memory")
```

### DocumentRetriever (`src/rag/retriever.py`)

The public interface. Composes a chunker and a store.

```python
from src.rag.retriever import build_retriever

retriever = build_retriever(
    chunker="sentence",   # "fixed" | "sentence" | "schema"
    store="chroma",       # "memory" | "chroma"
    chunk_size=400,
    overlap=1,
    top_k=5,
)

retriever.index_file("datasets/docs/ecommerce_kb.md")

# Returns formatted string — ready to inject into an LLM prompt
context = retriever.retrieve_text("what is the return policy?", top_k=3)

# Returns RetrievedChunk objects with score + metadata
results = retriever.retrieve("return policy", top_k=3)
for r in results:
    print(r.score, r.chunk.metadata["source"], r.chunk.text[:80])
```

### Knowledge base document

`datasets/docs/ecommerce_kb.md` — a 6 KB realistic knowledge base covering:
- Returns and refunds (30-day window, digital product exclusions, refund timeline)
- Shipping and delivery (free shipping threshold, carriers, delivery estimates)
- Payment methods (Visa, Mastercard, PayPal, etc.)
- Product inventory and availability policies
- Customer account management
- Customer support hours and contact channels

This is the document the `search_knowledge_base` tool retrieves from. It is indexed automatically on first tool call (module-level singleton in `tools.py`).

---

## KB retrieval benchmark (`scripts/benchmark_rag.py`)

Sweeps chunking strategies × vector stores × top-K values on 8 knowledge base eval questions. Measures `recall@K` — whether the required answer phrase appeared in any of the top-K retrieved chunks.

```bash
# Default sweep: sentence + fixed chunkers, memory store, top_k 3 and 5
python scripts/benchmark_rag.py

# Include ChromaDB and schema chunker
python scripts/benchmark_rag.py --chunker sentence fixed schema --store memory chroma --top-k 1 3 5 10

# Single config
python scripts/benchmark_rag.py --chunker sentence --store memory --top-k 5
```

On the clean KB document, `SentenceChunker` at top_k≥5 achieves perfect 1.000 recall. The real differentiation appears at scale with noisy documents.

---

## Schema retrieval benchmark (`scripts/benchmark_schema_retrieval.py`)

Comprehensive comparison of all three backends × two embedding libraries × three strategies on schema linking. See [findings.md](findings.md) for full results.

```bash
# Full sweep — all combinations
python scripts/benchmark_schema_retrieval.py

# Quick: local-only models, single top-k (no API cost)
python scripts/benchmark_schema_retrieval.py \
    --embedding fastembed/BAAI/bge-small-en-v1.5 fastembed/nomic-ai/nomic-embed-text-v1.5 \
    --top-k 5

# Qdrant only, all strategies
python scripts/benchmark_schema_retrieval.py --backend qdrant --top-k 5
```

**Embedding models supported:**

| Key | Library | Dims | Cost |
|---|---|---|---|
| `openai/text-embedding-3-small` | LiteLLM/OpenAI API | 1536 | ~$0.00002/run |
| `openai/text-embedding-3-large` | LiteLLM/OpenAI API | 3072 | ~$0.00013/run |
| `fastembed/BAAI/bge-small-en-v1.5` | FastEmbed local | 384 | free |
| `fastembed/nomic-ai/nomic-embed-text-v1.5` | FastEmbed local | 768 | free |

FastEmbed models are downloaded from HuggingFace on first use (~90–275 MB each) and cached locally. All subsequent runs are offline.

---

## Choosing a vector DB for production

| DB | Hosting | Hybrid search | When to choose |
|---|---|---|---|
| Qdrant | Self-hosted or managed (qdrant.tech) | ✅ native RRF | Best default — strong performance, great Python client, BM25 built-in |
| ChromaDB | Self-hosted | ❌ dense only | Easiest to operate; no external service needed; good for < 100K chunks |
| pgvector | Postgres extension | Partial (requires pg_bm25) | No extra infra if you already use Postgres |
| Pinecone | Fully managed | ✅ hybrid available | Serverless pricing; no ops burden; good for bursty workloads |
| Weaviate | Self-hosted or managed | ✅ BM25 + dense | Strong metadata filtering; good for structured + unstructured data |

For this project, **Qdrant** was chosen because:
1. Native dual-vector collections (dense + sparse in one index)
2. RRF fusion is built-in — no custom score merging code
3. `:memory:` mode eliminates all infrastructure for development
4. FastEmbed provides SPLADE sparse embeddings with no API call
