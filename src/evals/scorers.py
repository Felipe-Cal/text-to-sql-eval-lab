"""
Scorers for text-to-SQL evaluation.

Three scorers, increasing in strictness:

  1. syntax_valid    — does the SQL parse without error? (sqlglot)
  2. execution_ok    — does the SQL execute against the real database?
  3. result_match    — do the result rows match the golden expected rows?

Each scorer follows the Inspect AI scorer interface:
  score(state, target) -> Score
"""

import json
from typing import Any

import datetime
from decimal import Decimal

import sqlglot
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState

from src.utils.db import execute_query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_rows(rows: list[tuple]) -> list[tuple]:
    """Normalize values for loose comparison: round floats/Decimals, strip strings, convert dates."""
    normalized = []
    for row in rows:
        new_row = []
        for val in row:
            if isinstance(val, (float, Decimal)):
                new_row.append(round(float(val), 2))
            elif isinstance(val, str):
                new_row.append(val.strip())
            elif isinstance(val, (datetime.date, datetime.datetime)):
                new_row.append(val.isoformat())
            else:
                new_row.append(val)
        normalized.append(tuple(new_row))
    return normalized


def _rows_match(actual: list[tuple], expected: list[tuple], ordered: bool = False) -> bool:
    """
    Compare result sets.
    - For ORDER BY queries (ordered=True) row order matters.
    - Otherwise compare as sets (order-independent).
    """
    actual_n = _normalize_rows(actual)
    expected_n = _normalize_rows(expected)
    if ordered:
        return actual_n == expected_n
    return sorted(actual_n) == sorted(expected_n)


# ---------------------------------------------------------------------------
# Scorer 1: SQL syntax validity
# ---------------------------------------------------------------------------

@scorer(metrics=[accuracy()])
def syntax_valid() -> Scorer:
    """
    Checks whether the generated SQL is syntactically valid using sqlglot.
    Does NOT require a database connection.
    """
    async def score(state: TaskState, target: Target) -> Score:
        sql = state.output.completion.strip()
        if not sql:
            return Score(value=0, explanation="Empty SQL output.")

        try:
            parsed = sqlglot.parse(sql, dialect="duckdb")
            if not parsed or parsed[0] is None:
                return Score(value=0, explanation="sqlglot returned no parse tree.")
            return Score(value=1, explanation=f"Valid SQL: {sql}")
        except sqlglot.errors.ParseError as e:
            return Score(value=0, explanation=f"Parse error: {e}")

    return score


# ---------------------------------------------------------------------------
# Scorer 2: Execution success
# ---------------------------------------------------------------------------

@scorer(metrics=[accuracy()])
def execution_ok() -> Scorer:
    """
    Attempts to execute the generated SQL against the real DuckDB database.
    Checks that it runs without error (does not validate the result yet).
    """
    async def score(state: TaskState, target: Target) -> Score:
        sql = state.output.completion.strip()
        if not sql:
            return Score(value=0, explanation="Empty SQL output.")

        try:
            execute_query(sql)
            return Score(value=1, explanation="SQL executed successfully.")
        except Exception as e:
            return Score(value=0, explanation=f"Execution error: {e}")

    return score


# ---------------------------------------------------------------------------
# Scorer 3: Result correctness
# ---------------------------------------------------------------------------

@scorer(metrics=[accuracy()])
def result_match() -> Scorer:
    """
    Executes the generated SQL and compares the result rows against the
    golden expected rows stored in target.text (JSON-encoded list of lists).

    Scoring:
      1.0  — exact match (after float rounding and string stripping)
      0.5  — same columns returned but wrong values (partial credit)
      0.0  — execution error or completely wrong result
    """
    async def score(state: TaskState, target: Target) -> Score:
        sql = state.output.completion.strip()
        if not sql:
            return Score(value=0, explanation="Empty SQL output.")

        # Parse expected rows from target
        try:
            expected_rows: list[list[Any]] = json.loads(target.text)
            expected_tuples = [tuple(row) for row in expected_rows]
        except (json.JSONDecodeError, TypeError) as e:
            return Score(value=0, explanation=f"Could not parse golden rows: {e}")

        # Execute generated SQL
        try:
            actual_rows = execute_query(sql)
        except Exception as e:
            return Score(value=0, explanation=f"Execution error: {e}")

        # Compare results
        if _rows_match(actual_rows, expected_tuples, ordered=False):
            return Score(
                value=1,
                explanation=f"Result matches. Got {len(actual_rows)} rows.",
            )

        # Partial credit: same number of rows, same number of columns
        if (
            len(actual_rows) == len(expected_tuples)
            and actual_rows
            and len(actual_rows[0]) == len(expected_tuples[0])
        ):
            return Score(
                value=0.5,
                explanation=(
                    f"Shape matches ({len(actual_rows)} rows × {len(actual_rows[0])} cols) "
                    f"but values differ.\nActual:   {actual_rows[:3]}\nExpected: {expected_tuples[:3]}"
                ),
            )

        return Score(
            value=0,
            explanation=(
                f"Result mismatch. Got {len(actual_rows)} rows, expected {len(expected_tuples)}.\n"
                f"Actual:   {actual_rows[:3]}\nExpected: {expected_tuples[:3]}"
            ),
        )

    return score
