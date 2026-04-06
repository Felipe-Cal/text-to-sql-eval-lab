"""
Chunking strategies for RAG document preparation.

A chunk is the unit of text that gets embedded and stored in the vector store.
Choosing the right chunking strategy is one of the most impactful decisions in
a RAG pipeline:

  - Too large  → chunks contain irrelevant content, retrieval precision drops
  - Too small  → chunks lose context, the LLM can't reason over them
  - No overlap → a relevant sentence split across two chunks may be missed

Three strategies are provided:

  FixedSizeChunker   — splits on character count with configurable overlap.
                       Fast, predictable, language-agnostic. Good baseline.

  SentenceChunker    — groups complete sentences up to a max character limit.
                       Preserves semantic boundaries. Better for prose.

  SchemaChunker      — one chunk per table definition (what schema_retriever
                       used to do implicitly). Included for benchmarking;
                       shows that "no chunking" is a valid strategy when
                       the source documents are already short and structured.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import re


@dataclass
class Chunk:
    """A single piece of text ready to be embedded."""
    text: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk({preview!r}... metadata={self.metadata})"


class FixedSizeChunker:
    """
    Splits text into chunks of at most `chunk_size` characters, with
    `overlap` characters of shared context between consecutive chunks.

    Example (chunk_size=20, overlap=5):
        "The quick brown fox jumps over the lazy dog"
        → ["The quick brown fox ", "x jumps over the la", "e lazy dog"]

    Overlap prevents a relevant phrase from being split across a boundary
    and missed entirely during retrieval.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        chunks = []
        start = 0
        step = self.chunk_size - self.overlap

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={
                        **metadata,
                        "chunk_index": len(chunks),
                        "start_char": start,
                        "end_char": end,
                        "strategy": "fixed_size",
                    }
                ))
            start += step

        return chunks


class SentenceChunker:
    """
    Groups complete sentences into chunks up to `max_chunk_size` characters.
    Never splits mid-sentence — each chunk boundary falls on a sentence end.

    This preserves semantic coherence better than FixedSizeChunker for
    prose documents (FAQs, policies, call transcripts). The trade-off is
    variable chunk size, which can make embedding quality less uniform.

    Sentence detection uses a simple regex split on [.!?] followed by
    whitespace — sufficient for clean documents; a library like spaCy or
    NLTK would be better for noisy text.
    """

    def __init__(self, max_chunk_size: int = 512, overlap_sentences: int = 1):
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences

    def _split_sentences(self, text: str) -> list[str]:
        # Split on sentence-ending punctuation followed by whitespace or end of string
        raw = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in raw if s.strip()]

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        sentences = self._split_sentences(text)
        chunks = []
        current_sentences: list[str] = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            # If adding this sentence would exceed the limit, flush current buffer
            if current_sentences and current_size + sentence_size + 1 > self.max_chunk_size:
                chunk_text = " ".join(current_sentences).strip()
                if chunk_text:
                    chunks.append(Chunk(
                        text=chunk_text,
                        metadata={
                            **metadata,
                            "chunk_index": len(chunks),
                            "sentence_count": len(current_sentences),
                            "strategy": "sentence",
                        }
                    ))
                # Keep the last N sentences as overlap for the next chunk
                current_sentences = current_sentences[-self.overlap_sentences:] if self.overlap_sentences else []
                current_size = sum(len(s) for s in current_sentences)

            current_sentences.append(sentence)
            current_size += sentence_size + 1  # +1 for space

        # Flush remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={
                        **metadata,
                        "chunk_index": len(chunks),
                        "sentence_count": len(current_sentences),
                        "strategy": "sentence",
                    }
                ))

        return chunks


class SchemaChunker:
    """
    Treats each table definition as a single chunk — no splitting.

    This is the implicit strategy used by the original schema_retriever.
    It works well when source documents are already short and structured
    (e.g. one-line table definitions). Included here to make the implicit
    assumption explicit and allow apples-to-apples benchmarking.
    """

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """Each non-empty line becomes one chunk."""
        metadata = metadata or {}
        chunks = []
        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if line:
                chunks.append(Chunk(
                    text=line,
                    metadata={
                        **metadata,
                        "chunk_index": i,
                        "strategy": "schema",
                    }
                ))
        return chunks


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = """
    Our return policy allows customers to return any product within 30 days of purchase.
    Items must be in their original condition and packaging. Refunds are processed within
    5-7 business days after we receive the returned item. Shipping costs for returns are
    the responsibility of the customer unless the item was defective or incorrectly shipped.
    For defective items, please contact support@example.com within 48 hours of delivery.
    We will arrange a prepaid return label and ship a replacement at no extra cost.
    """

    print("=== FixedSizeChunker (size=200, overlap=40) ===")
    for c in FixedSizeChunker(chunk_size=200, overlap=40).chunk(sample, {"source": "return_policy"}):
        print(c)

    print("\n=== SentenceChunker (max=200, overlap=1) ===")
    for c in SentenceChunker(max_chunk_size=200, overlap_sentences=1).chunk(sample, {"source": "return_policy"}):
        print(c)
