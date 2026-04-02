"""
Few-shot example selection for text-to-SQL prompt optimization.

Two selection strategies:

  static:  always returns the same hand-picked examples (one per difficulty
           level), regardless of the question being evaluated. Zero extra
           latency — no API calls needed.

  dynamic: embeds the current question and all golden candidates in a single
           batch call, then returns the K nearest neighbours by cosine
           similarity. The question itself is excluded to prevent leakage
           when evaluating on the golden set.

Both return a list of Example objects ready to be formatted into a prompt.
"""

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import litellm

GOLDEN_PATH = Path(__file__).parent.parent.parent / "datasets" / "golden" / "questions.json"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Example:
    id: str
    question: str
    sql: str
    difficulty: str


# ---------------------------------------------------------------------------
# Loader (cached — reads the file once per process)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_all_examples() -> tuple[Example, ...]:
    with open(GOLDEN_PATH) as f:
        data = json.load(f)
    return tuple(
        Example(
            id=q["id"],
            question=q["question"],
            sql=q["sql"],
            difficulty=q["difficulty"],
        )
        for q in data
    )


# ---------------------------------------------------------------------------
# Strategy 1: static selection
# ---------------------------------------------------------------------------

def get_static_examples(n: int = 3) -> list[Example]:
    """
    Return a fixed set of n examples, one per difficulty level (easy →
    medium → hard). Always the same regardless of the question, so there
    is no extra latency or API cost.

    The examples are chosen to cover a range of SQL patterns:
    easy (COUNT), medium (JOIN + aggregation), hard (anti-join / HAVING).
    """
    all_examples = _load_all_examples()

    # One representative per difficulty, in increasing order
    seen: dict[str, Example] = {}
    for ex in all_examples:
        if ex.difficulty not in seen:
            seen[ex.difficulty] = ex

    ordered = [seen.get("easy"), seen.get("medium"), seen.get("hard")]
    return [e for e in ordered if e is not None][:n]


# ---------------------------------------------------------------------------
# Strategy 2: dynamic selection via embedding similarity
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def get_dynamic_examples(
    question: str,
    n: int = 3,
    embedding_model: str | None = None,
) -> list[Example]:
    """
    Return the n golden examples most semantically similar to the question.

    All candidates + the query are embedded in a single batch call to keep
    latency down. The exact question is excluded from candidates to prevent
    leakage when evaluating on the golden set.

    Args:
        question:        The natural language question to find examples for.
        n:               Number of examples to return.
        embedding_model: LiteLLM embedding model string.
                         Defaults to EMBEDDING_MODEL env var, then
                         "openai/text-embedding-3-small".
    """
    model = embedding_model or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    all_examples = _load_all_examples()
    # Exclude the current question to avoid leakage on the golden eval set
    candidates = [ex for ex in all_examples if ex.question != question]

    # Batch-embed all candidates + the query in a single API call
    texts = [ex.question for ex in candidates] + [question]
    response = litellm.embedding(model=model, input=texts)
    embeddings = [item["embedding"] for item in response.data]

    candidate_embeddings = embeddings[:-1]
    query_embedding = embeddings[-1]

    # Rank by cosine similarity, highest first
    scored = sorted(
        zip(candidate_embeddings, candidates),
        key=lambda pair: _cosine_similarity(query_embedding, pair[0]),
        reverse=True,
    )

    return [ex for _, ex in scored[:n]]
