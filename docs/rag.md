# RAG Infrastructure

The RAG module (`src/rag/`) provides a configurable retrieval pipeline used by the `rag` schema-linking strategy, the `tool_use` agent's `search_knowledge_base` tool, and the retrieval benchmark scripts.

There are two distinct retrieval use cases in this project:

| Use case | Source | Backend |
|---|---|---|
| **Schema linking** | 50 table definitions (4 core + 46 decoys) | Qdrant (dense + sparse + hybrid) |
| **Knowledge base** | `datasets/docs/ecommerce_kb.md` (~6 KB) | InMemoryStore or ChromaDB |

---

## Schema linking — Qdrant hybrid search (`src/agent/schema_retriever.py`)

Rather than passing all 50 table definitions to the model, the schema retriever embeds the question and retrieves only the most relevant tables — keeping prompts short and reducing confusion for smaller models.

### Architecture

```
Question
    │
    ├─── dense embed (OpenAI) ──────────┐
    │                                   │
    └─── sparse embed (SPLADE) ─────────┤
                                        ▼
                               Qdrant in-memory
                               (dual-vector collection)
                                        │
                              Reciprocal Rank Fusion
                                        │
                                        ▼
                               top-K table definitions
```

### Three retrieval strategies

| Strategy | How it works | Best for |
|---|---|---|
| `rag_dense` | Cosine similarity on OpenAI embeddings | Semantic matches ("order history" → orders) |
| `rag_sparse` | BM25/SPLADE lexical matching via FastEmbed | Exact keyword matches (column names, IDs) |
| `rag_hybrid` | Reciprocal Rank Fusion of dense + sparse | Best overall — handles both cases |

### Implementation

```python
from src.agent.schema_retriever import retrieve_schema

# Returns a formatted schema string + list of retrieved table defs
schema_str, tables = retrieve_schema(
    question="Who are the top customers by revenue?",
    top_k=5,
    retrieval_type="hybrid",  # "dense" | "sparse" | "hybrid"
)
```

The retriever is a singleton — the Qdrant collection is built once on first call and cached in memory for the lifetime of the process. Dense embeddings use `EMBEDDING_MODEL` env var (default: `openai/text-embedding-3-small`); sparse embeddings use SPLADE (`prithivida/Splade_PP_en_v1`) via FastEmbed (local, no API cost).

### Why Qdrant?

- **Native hybrid search** via `FusionQuery(fusion=Fusion.RRF)` — no custom fusion code needed
- **Dual-vector collections** support dense + sparse in a single index
- **In-memory mode** (`QdrantClient(":memory:")`) — zero infra for development
- **Production path**: replace `:memory:` with a Qdrant server URL, same API

---

## Knowledge base retrieval — chunked RAG (`src/rag/`)

The `src/rag/` module handles document-level retrieval for the `search_knowledge_base` tool and the RAG benchmark. Unlike schema retrieval (already one-liner table defs), knowledge base documents require actual chunking.

### Architecture

```
Document(s)
    │
    ▼
Chunker          — splits text into embeddable units
    │
    ▼
VectorStore      — embeds and stores chunks; queries by similarity
    │
    ▼
DocumentRetriever — public interface composing chunker + store
    │
    ▼
top-K chunks     — injected into LLM prompt as context
```

### Chunking (`src/rag/chunker.py`)

| Strategy | How it splits | Best for |
|---|---|---|
| `FixedSizeChunker` | Character count with configurable overlap | Fast baseline, language-agnostic |
| `SentenceChunker` | Groups complete sentences up to max character limit | Prose documents (FAQs, policies, transcripts) |
| `SchemaChunker` | One chunk per line (table definition) | Structured schema documents |

```python
from src.rag.chunker import SentenceChunker

chunks = SentenceChunker(max_chunk_size=512, overlap_sentences=1).chunk(
    text, metadata={"source": "return_policy.md"}
)
```

### Vector stores (`src/rag/vector_store.py`)

Both implementations share the same `VectorStore` abstract interface (`add`, `query`, `reset`, `count`).

| Store | Persistence | Query time | Best for |
|---|---|---|---|
| `InMemoryStore` | None (lost on restart) | O(n) linear scan | Tests, small corpora, quick prototyping |
| `ChromaDBStore` | Disk (HNSW index, survives restarts) | O(log n) | Production, large corpora, multi-process |

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
context = retriever.retrieve_text("what is the return policy?", top_k=3)
```

### Knowledge base

`datasets/docs/ecommerce_kb.md` — a 6 KB realistic knowledge base covering returns, shipping, payments, account management, and customer support. Indexed automatically by the `search_knowledge_base` tool and the KB benchmark.

---

## Benchmarks

### KB retrieval benchmark (`scripts/benchmark_rag.py`)

Sweeps chunking strategies × vector stores × top-K. Measures `recall@K` on 8 eval questions about the knowledge base.

```bash
# Default sweep: sentence + fixed chunkers, memory store, top_k 3 and 5
python scripts/benchmark_rag.py

# Full sweep including ChromaDB
python scripts/benchmark_rag.py --chunker sentence fixed schema --store memory chroma --top-k 1 3 5 10
```

### Schema retrieval benchmark (`scripts/benchmark_schema_retrieval.py`)

Comprehensive comparison of vector DB backends, embedding models, and retrieval strategies on the schema linking task (50 tables → retrieve required tables given a natural language question).

```bash
# Full sweep (all backends × models × top-k values)
python scripts/benchmark_schema_retrieval.py

# Quick qdrant-only comparison
python scripts/benchmark_schema_retrieval.py --backend qdrant --top-k 5

# Compare dense vs hybrid with a single embedding model
python scripts/benchmark_schema_retrieval.py \
    --strategy dense hybrid \
    --embedding openai/text-embedding-3-small \
    --top-k 3 5 10
```

**What it measures:**
- `Recall@K` — fraction of required tables retrieved (partial credit per question)
- `Perfect@K` — fraction of questions where ALL required tables were in top-K
- Index time, avg query latency, estimated API cost

**Configurations compared:**

| Backend | Strategy | Embedding |
|---|---|---|
| Qdrant | dense | text-embedding-3-small |
| Qdrant | dense | text-embedding-3-large |
| Qdrant | sparse (BM25) | SPLADE (local, no API) |
| Qdrant | hybrid (RRF) | text-embedding-3-small |
| Qdrant | hybrid (RRF) | text-embedding-3-large |
| InMemoryStore | dense | text-embedding-3-small |
| InMemoryStore | dense | text-embedding-3-large |
| ChromaDB | dense | text-embedding-3-small |
| ChromaDB | dense | text-embedding-3-large |

**Why these combinations?** Sparse/BM25 is only available in Qdrant (via FastEmbed SPLADE). InMemoryStore and ChromaDB support dense-only through the shared `get_embeddings()` helper. Hybrid combines the best of both worlds and is the recommended production strategy.

---

## Choosing a vector DB for production

| DB | Hosting | Hybrid search | Notes |
|---|---|---|---|
| Qdrant | Self-hosted or managed | ✅ native RRF | Best performance for this use case |
| ChromaDB | Self-hosted | ❌ dense only | Easy to operate, no external service |
| pgvector | Postgres extension | Partial (pgvector + pg_bm25) | No extra infra if you already use Postgres |
| Pinecone | Fully managed | ✅ | Serverless pricing, no ops |
| Weaviate | Self-hosted or managed | ✅ BM25 + dense | Strong for structured metadata filtering |

For this project, Qdrant was chosen because:
1. Native Python client with first-class support for dual-vector collections
2. Hybrid RRF is built-in — no custom score merging needed
3. `:memory:` mode eliminates all infra for development
4. FastEmbed integration allows sparse embeddings without an API call
