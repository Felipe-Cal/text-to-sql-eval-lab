"""
Model router — classifies question difficulty and picks the right model + strategy.

How it works (two-stage):

  Stage 1 — Rule-based (free, instant):
    Scores the question against keyword patterns that signal easy / medium / hard SQL.
    If confidence is high enough (score gap >= RULE_CONFIDENCE_THRESHOLD), uses that result.

  Stage 2 — Embedding k-NN (one API call):
    Embeds the question, finds the K nearest golden examples (which have difficulty labels),
    and takes a majority vote. Used when rule-based confidence is low.

Routing table (configurable via env vars):
  easy   → DEFAULT_MODEL        + zero_shot
  medium → DEFAULT_MODEL        + few_shot_dynamic
  hard   → HARD_MODEL (or DEFAULT_MODEL if not set) + few_shot_dynamic

The idea: cheap models handle easy questions for free; expensive models are reserved
for questions that actually need them.
"""

import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache

import litellm

from src.agent.few_shot import _load_all_examples

# ---------------------------------------------------------------------------
# Routing table — maps difficulty → (model, strategy_string)
# Override HARD_MODEL in .env to escalate hard questions to a stronger model.
# ---------------------------------------------------------------------------

def _routing_table() -> dict[str, dict]:
    default = os.getenv("DEFAULT_MODEL", "openai/gpt-4o-mini")
    hard = os.getenv("HARD_MODEL", default)  # falls back to DEFAULT_MODEL if not set
    return {
        "easy":   {"model": default, "strategy": "zero_shot"},
        "medium": {"model": default, "strategy": "few_shot_dynamic"},
        "hard":   {"model": hard,    "strategy": "few_shot_dynamic"},
    }


# ---------------------------------------------------------------------------
# Rule-based classifier
# ---------------------------------------------------------------------------

# Patterns that signal each difficulty level.
# Each match adds to that level's score.
_RULES: dict[str, list[str]] = {
    "easy": [
        r"\bhow many\b",
        r"\blist (all|the)\b",
        r"\bwhat is the (most|least|highest|lowest|cheapest|expensive)\b",
        r"\bwhich .{0,30} (are|is) from\b",
        r"\bcount\b",
        r"\bdistinct\b",
        r"\bshow (all|me)\b",
    ],
    "medium": [
        r"\btotal (revenue|spend|sales|cost|value)\b",
        r"\baverage\b",
        r"\beach (customer|product|category|country|order)\b",
        r"\bper (month|year|category|customer)\b",
        r"\bgroup\b",
        r"\bhow many orders did\b",
        r"\bsigned up\b",
        r"\bjoined\b",
        r"\bin 20\d\d\b",
    ],
    "hard": [
        r"\bnever\b",
        r"\bnot (in|placed|made|completed)\b",
        r"\bmore than (one|1|two|2)\b",
        r"\btop \d+.{0,30}(spend|revenue|value)\b",
        r"\bmonth.by.month\b",
        r"\bover time\b",
        r"\bfor each (country|region)\b",
        r"\bhaving\b",
        r"\banti.join\b",
        r"\bcompare.{0,20}(country|region|category)\b",
    ],
}


def _rule_based_classify(question: str) -> tuple[str, float]:
    """
    Score question against keyword rules.
    Returns (difficulty, confidence) where confidence is in [0, 1].
    Confidence reflects how clearly one level dominates the others.
    """
    q = question.lower()
    scores: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}

    for difficulty, patterns in _RULES.items():
        for pattern in patterns:
            if re.search(pattern, q):
                scores[difficulty] += 1

    total = sum(scores.values())
    if total == 0:
        return "medium", 0.0  # no signal — default to medium, low confidence

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    winner, top_score = sorted_scores[0]
    second_score = sorted_scores[1][1]

    # Confidence = how much the winner dominates relative to total signal
    gap = (top_score - second_score) / total
    return winner, round(gap, 3)


# ---------------------------------------------------------------------------
# Embedding k-NN classifier
# ---------------------------------------------------------------------------

RULE_CONFIDENCE_THRESHOLD = 0.30  # below this, fall back to embedding


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


@lru_cache(maxsize=1)
def _golden_embeddings() -> tuple[list[list[float]], list[str]]:
    """
    Embed all golden questions and cache them.
    Returns (embeddings, difficulties) in the same order.
    """
    examples = _load_all_examples()
    model = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
    texts = [ex.question for ex in examples]
    response = litellm.embedding(model=model, input=texts)
    embeddings = [item["embedding"] for item in response.data]
    difficulties = [ex.difficulty for ex in examples]
    return embeddings, difficulties


def _embedding_classify(question: str, k: int = 5) -> tuple[str, float]:
    """
    Classify difficulty by k-NN over the golden dataset.
    Returns (difficulty, confidence) where confidence = winning_votes / k.
    """
    model = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
    response = litellm.embedding(model=model, input=[question])
    q_embedding = response.data[0]["embedding"]

    golden_embeddings, difficulties = _golden_embeddings()

    scored = sorted(
        zip(golden_embeddings, difficulties),
        key=lambda pair: _cosine_similarity(q_embedding, pair[0]),
        reverse=True,
    )

    neighbors = [diff for _, diff in scored[:k]]
    votes: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
    for diff in neighbors:
        votes[diff] += 1

    winner = max(votes, key=lambda d: votes[d])
    confidence = round(votes[winner] / k, 3)
    return winner, confidence


# ---------------------------------------------------------------------------
# Router result
# ---------------------------------------------------------------------------

@dataclass
class RoutingDecision:
    difficulty: str       # easy | medium | hard
    model: str            # LiteLLM model string
    strategy: str         # PromptStrategy value string
    method: str           # rule_based | embedding
    confidence: float     # 0–1


def route(question: str) -> RoutingDecision:
    """
    Classify a question and return the model + strategy to use.

    Stage 1: rule-based (free). If confidence >= threshold, use it.
    Stage 2: embedding k-NN (one API call). Used when rules are ambiguous.
    """
    difficulty, confidence = _rule_based_classify(question)
    method = "rule_based"

    if confidence < RULE_CONFIDENCE_THRESHOLD:
        difficulty, confidence = _embedding_classify(question)
        method = "embedding"

    table = _routing_table()
    route_config = table[difficulty]

    return RoutingDecision(
        difficulty=difficulty,
        model=route_config["model"],
        strategy=route_config["strategy"],
        method=method,
        confidence=confidence,
    )
