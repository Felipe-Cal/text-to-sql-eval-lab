# RAG Infrastructure

The RAG module (`src/rag/`) provides a configurable retrieval pipeline used by the `rag` schema-linking strategy, the `tool_use` agent's `search_knowledge_base` tool, and the RAG benchmark script.

---

## Architecture

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

---

## Chunking (`src/rag/chunker.py`)

A chunk is the unit of text that gets embedded. Chunk size is one of the highest-impact decisions in a RAG pipeline: too large loses precision, too small loses context.

| Strategy | How it splits | Best for |
|---|---|---|
| `FixedSizeChunker` | Character count with configurable overlap | Fast baseline, language-agnostic |
| `SentenceChunker` | Groups complete sentences up to a max character limit | Prose documents (FAQs, policies, transcripts) |
| `SchemaChunker` | One chunk per line (table definition) | Structured schema documents — the implicit strategy in `schema_retriever.py` |

**Overlap** prevents a relevant phrase from being split across a chunk boundary and missed during retrieval. `FixedSizeChunker` overlaps by characters; `SentenceChunker` by sentence count.

```python
from src.rag.chunker import FixedSizeChunker, SentenceChunker

chunks = SentenceChunker(max_chunk_size=512, overlap_sentences=1).chunk(
    text, metadata={"source": "return_policy.md"}
)
```

---

## Vector stores (`src/rag/vector_store.py`)

Both implementations share the same `VectorStore` abstract interface (`add`, `query`, `reset`, `count`) — callers are never coupled to a specific backend.

| Store | Persistence | Query time | Best for |
|---|---|---|---|
| `InMemoryStore` | None (lost on restart) | O(n) linear scan | Tests, small corpora, quick prototyping |
| `ChromaDBStore` | Disk (survives restarts) | O(log n) HNSW index | Production, large corpora, multi-process |

In production you would swap `ChromaDBStore` for a managed vector DB:
- **pgvector** — already in your Postgres, no extra infra
- **Pinecone** — fully managed, serverless pricing
- **Weaviate** — open-source, supports hybrid BM25 + dense search
- **Qdrant** — fast, open-source, strong Rust performance

```python
from src.rag.vector_store import create_store

store = create_store("chroma", collection_name="kb_docs", persist_dir="./chroma_db")
```

---

## DocumentRetriever (`src/rag/retriever.py`)

The public interface. Composes a chunker and a store with a simple three-method API.

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

# Returns formatted string ready to inject into a prompt
context = retriever.retrieve_text("what is the return policy?", top_k=3)

# Or returns RetrievedChunk objects with scores and metadata
results = retriever.retrieve("return policy", top_k=3)
for r in results:
    print(r.score, r.chunk.metadata["source"], r.chunk.text[:80])
```

---

## Knowledge base

`datasets/docs/ecommerce_kb.md` is a 6 KB realistic knowledge base covering:
- Returns and refunds policy
- Shipping and delivery options
- Pricing and payment methods
- Product inventory policies
- Customer account management
- Customer support hours and contacts

This document is long enough to require actual chunking (unlike table definitions which are already one-liners). It is indexed automatically when the `search_knowledge_base` tool or the `rag` strategy is first used.

---

## Benchmarking

`scripts/benchmark_rag.py` sweeps chunking strategies × vector stores × top-K values and measures `recall@K` — what fraction of 8 eval questions had their required answer present in the top-K chunks.

```bash
# Default sweep: sentence + fixed chunkers, memory store, top_k 3 and 5
python scripts/benchmark_rag.py

# Custom sweep
python scripts/benchmark_rag.py --chunker sentence fixed schema --store memory chroma --top-k 1 3 5 10

# Single config
python scripts/benchmark_rag.py --chunker sentence --store chroma --top-k 5
```

**Interpreting results:**
- `recall@K = 1.0` on a small clean corpus is expected — the real differentiation appears at scale with noisy documents
- `InMemoryStore` and `ChromaDBStore` return identical results for the same query; the difference is latency at scale (linear scan vs HNSW)
- `SentenceChunker` outperforms `FixedSizeChunker` on prose when chunks are small — it preserves sentence boundaries that contain the answer

---

## What's next

The current implementation uses **dense-only retrieval** (embedding cosine similarity). The planned improvement is **hybrid search**:

1. **BM25** (sparse, keyword-based) scores terms exactly — great for product names, error codes, SQL keywords
2. **Dense embeddings** score semantic similarity — great for paraphrases and synonyms
3. **Reciprocal Rank Fusion (RRF)** combines both scores without needing to tune weights

Hybrid search consistently outperforms dense-only on technical corpora and is the recommended production approach.
