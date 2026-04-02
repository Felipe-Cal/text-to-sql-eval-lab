"""
Input guardrails for the text-to-SQL pipeline.

Runs before the LLM call. Two independent checks:

  1. sql_injection    — detects attempts to inject SQL commands via the
                        natural language input (DROP, DELETE, --, etc.)
  2. prompt_injection — detects attempts to override the system prompt
                        ("ignore the above", "new instructions:", etc.)

Each check is independent. The first one that fails is reported.
Both are deterministic (regex/keyword), zero LLM cost, sub-millisecond.

Usage:
    from src.guardrails.input import check_input

    result = check_input("How many customers are there?")
    if not result.passed:
        print(f"Blocked [{result.category}]: {result.reason}")
"""

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    passed: bool
    category: str | None = None   # "sql_injection" | "prompt_injection" | None
    reason: str | None = None     # human-readable explanation, None if passed

    def __bool__(self) -> bool:
        return self.passed


PASSED = GuardrailResult(passed=True)


# ---------------------------------------------------------------------------
# Check 1: SQL injection patterns
#
# Looks for SQL keywords and syntax that have no place in a natural
# language question about the database. Flags both word-boundary matches
# (DROP TABLE) and comment/string-escape sequences (-- / /* */).
# ---------------------------------------------------------------------------

_SQL_INJECTION_PATTERNS: list[tuple[str, str]] = [
    # DDL
    (r"\bDROP\s+TABLE\b",   "contains DROP TABLE"),
    (r"\bDROP\s+DATABASE\b","contains DROP DATABASE"),
    (r"\bTRUNCATE\b",       "contains TRUNCATE"),
    (r"\bALTER\s+TABLE\b",  "contains ALTER TABLE"),
    (r"\bCREATE\s+TABLE\b", "contains CREATE TABLE"),
    # DML
    (r"\bDELETE\s+FROM\b",  "contains DELETE FROM"),
    (r"\bINSERT\s+INTO\b",  "contains INSERT INTO"),
    (r"\bUPDATE\s+\w+\s+SET\b", "contains UPDATE ... SET"),
    # Comment sequences used to terminate injected SQL
    (r"--",                 "contains SQL comment sequence (--)"),
    (r"/\*",               "contains SQL block comment (/*)"),
    # UNION-based injection
    (r"\bUNION\s+(ALL\s+)?SELECT\b", "contains UNION SELECT"),
    # Stacked queries
    (r";\s*(DROP|DELETE|INSERT|UPDATE|TRUNCATE|ALTER|CREATE)", "contains stacked query"),
]

_SQL_INJECTION_COMPILED = [
    (re.compile(pattern, re.IGNORECASE), reason)
    for pattern, reason in _SQL_INJECTION_PATTERNS
]


def _check_sql_injection(text: str) -> GuardrailResult:
    for pattern, reason in _SQL_INJECTION_COMPILED:
        if pattern.search(text):
            return GuardrailResult(
                passed=False,
                category="sql_injection",
                reason=f"Input blocked — {reason}.",
            )
    return PASSED


# ---------------------------------------------------------------------------
# Check 2: Prompt injection patterns
#
# Looks for phrasing used to override or escape the system prompt.
# Common patterns from known prompt injection research.
# ---------------------------------------------------------------------------

_PROMPT_INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"\bignore\s+(the\s+)?(above|previous|all|prior)\b",
     "attempts to override previous instructions"),
    (r"\bdisregard\s+(the\s+)?(above|previous|all|prior|your)\b",
     "attempts to disregard instructions"),
    (r"\bforget\s+(the\s+)?(above|previous|all|prior|your)\b",
     "attempts to erase context"),
    (r"\bnew\s+(task|instructions?|prompt|system\s+prompt)\b",
     "attempts to inject new instructions"),
    (r"\byou\s+are\s+now\b",
     "attempts to redefine the model's role"),
    (r"\bact\s+as\s+(a\s+)?(?!sql|database|analyst)",
     "attempts to change the model's persona"),
    (r"\bpretend\s+(you\s+are|to\s+be)\b",
     "attempts role-play persona override"),
    (r"\bdo\s+not\s+follow\b",
     "attempts to suppress instruction-following"),
    (r"\byour\s+(real\s+)?(instructions?|prompt|system)\s+(are|is|say)\b",
     "attempts to reveal or override system prompt"),
    (r"```.*system",
     "attempts to inject a system message block"),
]

_PROMPT_INJECTION_COMPILED = [
    (re.compile(pattern, re.IGNORECASE), reason)
    for pattern, reason in _PROMPT_INJECTION_PATTERNS
]


def _check_prompt_injection(text: str) -> GuardrailResult:
    for pattern, reason in _PROMPT_INJECTION_COMPILED:
        if pattern.search(text):
            return GuardrailResult(
                passed=False,
                category="prompt_injection",
                reason=f"Input blocked — {reason}.",
            )
    return PASSED


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_input(text: str) -> GuardrailResult:
    """
    Run all input guardrail checks against the provided text.

    Checks are run in order; the first failure is returned immediately.
    Returns GuardrailResult(passed=True) if all checks pass.

    Args:
        text: The raw natural language input from the user.

    Returns:
        GuardrailResult with passed=True, or passed=False with category
        and reason populated.
    """
    text = text.strip()

    if not text:
        return GuardrailResult(
            passed=False,
            category="empty_input",
            reason="Input is empty.",
        )

    for check in (_check_sql_injection, _check_prompt_injection):
        result = check(text)
        if not result.passed:
            return result

    return PASSED
