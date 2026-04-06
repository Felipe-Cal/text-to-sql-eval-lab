"""
Vector store abstraction with two implementations.

The abstract base (VectorStore) defines a three-method interface:
  add()      — embed and persist a list of Chunks
  query()    — embed a question and return top-K similar chunks
  reset()    — clear all stored chunks

Two concrete implementations are provided:

  InMemoryStore  — stores embeddings in a plain Python list. No external
                   dependencies, instant startup, lost on process exit.
                   Good for tests and small corpora.

  ChromaDBStore  — persists embeddings to disk via ChromaDB. Survives
                   restarts, scales to millions of chunks, supports
                   metadata filtering. What you'd use in production
                   (or swap for pgvector / Pinecone / Weaviate).

Both call the same get_embeddings() helper so the embedding model is
controlled by the EMBEDDING_MODEL env var — no code changes to switch
from text-embedding-3-small to text-embedding-3-large or an open-source
model.
"""

from __future__ import annotations
import os
import math
import uuid
from abc import ABC, abstractmethod

import litellm
import chromadb
from dotenv import load_dotenv

from src.rag.chunker import Chunk

load_dotenv()


# ---------------------------------------------------------------------------
# Shared embedding helper
# ---------------------------------------------------------------------------

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using the configured model via LiteLLM."""
    model = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
    response = litellm.embedding(model=model, input=texts)
    return [item["embedding"] for item in sorted(response.data, key=lambda x: x["index"])]


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(a * a for a in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class RetrievedChunk:
    """A chunk returned by a query, decorated with its similarity score."""

    def __init__(self, chunk: Chunk, score: float):
        self.chunk = chunk
        self.score = score

    def __repr__(self) -> str:
        preview = self.chunk.text[:60].replace("\n", " ")
        return f"RetrievedChunk(score={self.score:.3f}, text={preview!r}...)"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class VectorStore(ABC):
    """
    Minimal interface every vector store must implement.
    Swap implementations without changing any caller code.
    """

    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None:
        """Embed and persist chunks."""

    @abstractmethod
    def query(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the top_k most similar chunks for the question."""

    @abstractmethod
    def reset(self) -> None:
        """Remove all stored chunks (useful for benchmarking and tests)."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored chunks."""


# ---------------------------------------------------------------------------
# Implementation 1: In-memory (numpy-free, pure Python)
# ---------------------------------------------------------------------------

class InMemoryStore(VectorStore):
    """
    Stores embeddings in a Python list. Recomputed and lost on restart.

    Characteristics:
      - Zero dependencies beyond litellm
      - O(n) query time (linear scan) — fine up to ~10k chunks
      - No persistence — repopulate on every process start
      - Best for: unit tests, small corpora, quick prototyping
    """

    def __init__(self):
        self._chunks: list[Chunk] = []
        self._embeddings: list[list[float]] = []

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        embeddings = get_embeddings([c.text for c in chunks])
        self._chunks.extend(chunks)
        self._embeddings.extend(embeddings)

    def query(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not self._chunks:
            return []
        q_emb = get_embeddings([question])[0]
        scored = [
            RetrievedChunk(chunk, cosine_similarity(q_emb, emb))
            for chunk, emb in zip(self._chunks, self._embeddings)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def reset(self) -> None:
        self._chunks = []
        self._embeddings = []

    def count(self) -> int:
        return len(self._chunks)


# ---------------------------------------------------------------------------
# Implementation 2: ChromaDB (persistent, production-grade)
# ---------------------------------------------------------------------------

class ChromaDBStore(VectorStore):
    """
    Persists embeddings to disk via ChromaDB.

    Characteristics:
      - Survives process restarts — no re-embedding on startup
      - Scales to millions of chunks with ANN indexing (HNSW)
      - Supports metadata filtering (e.g. filter by source document)
      - Embeddings are stored by ChromaDB; we supply them pre-computed
        so the embedding model is still controlled by our env var
      - Best for: production, large corpora, multi-process deployments

    In production you'd swap this for a managed vector DB:
      - pgvector     → already in your Postgres, no extra infra
      - Pinecone     → fully managed, serverless pricing
      - Weaviate     → open-source, supports hybrid BM25 + dense search
      - Qdrant       → fast, open-source, good Rust performance
    """

    def __init__(self, collection_name: str = "rag_chunks", persist_dir: str = "./chroma_db"):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            # We supply our own embeddings, so tell Chroma not to embed
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        embeddings = get_embeddings([c.text for c in chunks])
        self._collection.add(
            ids=[str(uuid.uuid4()) for _ in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

    def query(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self._collection.count() == 0:
            return []
        q_emb = get_embeddings([question])[0]
        results = self._collection.query(
            query_embeddings=[q_emb],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB returns cosine distance (0=identical, 2=opposite)
            # Convert to similarity score in [0, 1]
            score = 1.0 - (dist / 2.0)
            chunk = Chunk(text=doc, metadata=meta)
            retrieved.append(RetrievedChunk(chunk, score))
        return retrieved

    def reset(self) -> None:
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def create_store(backend: str = "memory", **kwargs) -> VectorStore:
    """
    Instantiate a vector store by name.

    Args:
        backend: "memory" or "chroma"
        **kwargs: passed through to the store constructor

    Example:
        store = create_store("chroma", collection_name="my_docs", persist_dir="./db")
    """
    if backend == "memory":
        return InMemoryStore()
    elif backend == "chroma":
        return ChromaDBStore(**kwargs)
    else:
        raise ValueError(f"Unknown backend {backend!r}. Choose 'memory' or 'chroma'.")


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.rag.chunker import SentenceChunker

    sample_doc = """
    Customers can return products within 30 days of purchase for a full refund.
    Items must be unused and in original packaging. Digital products are non-refundable.
    To initiate a return, contact our support team with your order number.
    Refunds are issued to the original payment method within 5-7 business days.
    """

    chunks = SentenceChunker(max_chunk_size=200).chunk(
        sample_doc, metadata={"source": "return_policy"}
    )

    print(f"Created {len(chunks)} chunks\n")

    for backend in ["memory", "chroma"]:
        print(f"--- {backend.upper()} store ---")
        store = create_store(backend, collection_name=f"test_{backend}", persist_dir="./chroma_db_test")
        store.reset()
        store.add(chunks)
        print(f"Stored {store.count()} chunks")

        results = store.query("how long do I have to return an item?", top_k=2)
        for r in results:
            print(r)
        print()
