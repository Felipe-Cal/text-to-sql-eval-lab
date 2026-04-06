"""
DocumentRetriever — the public interface for RAG document retrieval.

Composes a Chunker + VectorStore into a single object with a simple API:
  retriever.index(documents)   — chunk, embed and store documents
  retriever.retrieve(question) — embed question, return top-K chunks

This is the layer that agent tools and the benchmark script interact with.
The chunker and vector store are injected at construction time, so every
combination (FixedSize + InMemory, Sentence + ChromaDB, etc.) is a
one-liner to configure.

Design notes:
  - Documents are represented as {"text": str, "metadata": dict} dicts.
    Metadata (source filename, section, page number) passes through to
    every chunk and appears in retrieved results — critical for citations.
  - The retriever is stateful: call index() once, then query() many times.
  - Calling index() again on the same store ADDS documents (no dedup).
    Call retriever.store.reset() first if you want a clean slate.
  - Retrieval quality is measured via retrieval_recall (see benchmark_rag.py):
    what fraction of "required" chunks were returned in the top-K.
"""

from __future__ import annotations
from pathlib import Path

from src.rag.chunker import Chunk, FixedSizeChunker, SentenceChunker, SchemaChunker
from src.rag.vector_store import VectorStore, RetrievedChunk, InMemoryStore, create_store


class DocumentRetriever:
    """
    Chunks + embeds documents, then retrieves relevant chunks for a query.

    Args:
        chunker:     A Chunker instance (FixedSizeChunker, SentenceChunker, ...)
        store:       A VectorStore instance (InMemoryStore, ChromaDBStore, ...)
        default_top_k: Default number of chunks to return per query.

    Example:
        retriever = DocumentRetriever(
            chunker=SentenceChunker(max_chunk_size=512),
            store=create_store("chroma", collection_name="kb"),
        )
        retriever.index_file("datasets/docs/ecommerce_kb.md")
        results = retriever.retrieve("what is the return policy?", top_k=3)
        for r in results:
            print(r.score, r.chunk.text[:100])
    """

    def __init__(
        self,
        chunker: FixedSizeChunker | SentenceChunker | SchemaChunker | None = None,
        store: VectorStore | None = None,
        default_top_k: int = 5,
    ):
        self.chunker = chunker or SentenceChunker()
        self.store = store or InMemoryStore()
        self.default_top_k = default_top_k

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, documents: list[dict]) -> int:
        """
        Chunk, embed and store a list of documents.

        Each document is a dict with:
          - "text"     (str, required)   — the raw text content
          - "metadata" (dict, optional)  — arbitrary metadata (source, section, ...)

        Returns the total number of chunks added.
        """
        all_chunks: list[Chunk] = []
        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            if text.strip():
                all_chunks.extend(self.chunker.chunk(text, metadata))

        if all_chunks:
            self.store.add(all_chunks)

        return len(all_chunks)

    def index_file(self, path: str | Path) -> int:
        """
        Read a text/markdown file and index its contents as a single document.
        The filename is added to the chunk metadata automatically.
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        return self.index([{
            "text": text,
            "metadata": {"source": path.name, "path": str(path)},
        }])

    def index_texts(self, texts: list[str], source: str = "unknown") -> int:
        """Convenience wrapper: index a list of raw strings."""
        docs = [{"text": t, "metadata": {"source": source}} for t in texts]
        return self.index(docs)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """
        Embed the question and return the top_k most relevant chunks.

        Returns an empty list if the store has no documents.
        """
        k = top_k if top_k is not None else self.default_top_k
        return self.store.query(question, top_k=k)

    def retrieve_text(self, question: str, top_k: int | None = None) -> str:
        """
        Like retrieve(), but returns a single formatted string ready to
        inject directly into a prompt as context.
        """
        results = self.retrieve(question, top_k=top_k)
        if not results:
            return "No relevant context found."
        parts = []
        for i, r in enumerate(results, 1):
            source = r.chunk.metadata.get("source", "unknown")
            parts.append(f"[{i}] (source: {source}, score: {r.score:.2f})\n{r.chunk.text}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def chunk_count(self) -> int:
        return self.store.count()

    def __repr__(self) -> str:
        return (
            f"DocumentRetriever("
            f"chunker={self.chunker.__class__.__name__}, "
            f"store={self.store.__class__.__name__}, "
            f"chunks={self.chunk_count})"
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_retriever(
    chunker: str = "sentence",
    store: str = "memory",
    chunk_size: int = 512,
    overlap: int = 64,
    collection_name: str = "rag_chunks",
    persist_dir: str = "./chroma_db",
    top_k: int = 5,
) -> DocumentRetriever:
    """
    Build a DocumentRetriever from string config — useful for benchmarking
    and CLI scripts where you want to sweep over configurations.

    Args:
        chunker:         "fixed", "sentence", or "schema"
        store:           "memory" or "chroma"
        chunk_size:      max characters per chunk (fixed) or sentences window (sentence)
        overlap:         character overlap (fixed) or sentence overlap count (sentence)
        collection_name: ChromaDB collection name (ignored for memory)
        persist_dir:     ChromaDB persistence directory (ignored for memory)
        top_k:           default number of results to return

    Example:
        r = build_retriever(chunker="sentence", store="chroma", chunk_size=400)
        r.index_file("datasets/docs/ecommerce_kb.md")
        print(r.retrieve_text("how do I return a defective item?"))
    """
    if chunker == "fixed":
        c = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    elif chunker == "sentence":
        c = SentenceChunker(max_chunk_size=chunk_size, overlap_sentences=overlap)
    elif chunker == "schema":
        c = SchemaChunker()
    else:
        raise ValueError(f"Unknown chunker {chunker!r}. Choose 'fixed', 'sentence', or 'schema'.")

    s = create_store(
        store,
        collection_name=collection_name,
        persist_dir=persist_dir,
    )

    return DocumentRetriever(chunker=c, store=s, default_top_k=top_k)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path

    kb_path = Path("datasets/docs/ecommerce_kb.md")

    questions = [
        "what is the return window for purchased products?",
        "how long does a refund take to process?",
        "do I get free shipping on large orders?",
        "what payment methods are accepted?",
        "how do I contact customer support?",
    ]

    for chunker_name in ["sentence", "fixed"]:
        print(f"\n{'='*60}")
        print(f"Chunker: {chunker_name.upper()}")
        print(f"{'='*60}")

        retriever = build_retriever(
            chunker=chunker_name,
            store="memory",
            chunk_size=400,
            overlap=1 if chunker_name == "sentence" else 60,
        )
        n = retriever.index_file(kb_path)
        print(f"Indexed {n} chunks from {kb_path.name}\n")

        for q in questions[:2]:
            print(f"Q: {q}")
            results = retriever.retrieve(q, top_k=2)
            for r in results:
                print(f"  score={r.score:.3f} | {r.chunk.text[:100]}...")
            print()
