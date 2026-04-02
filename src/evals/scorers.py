"""
Scorers for text-to-SQL evaluation.

Four scorers, increasing in sophistication:

  1. syntax_valid    — does the SQL parse without error? (sqlglot)
  2. execution_ok    — does the SQL execute against the real database?
  3. result_match    — do the result rows match the golden expected rows?
  4. semantic_judge  — does an LLM judge agree the SQL answers the question correctly?

Each scorer follows the Inspect AI scorer interface:
  score(state, target) -> Score
"""

import json
import os
from typing import Any

import datetime
from decimal import Decimal

import litellm
import sqlglot
from inspect_ai.scorer import Score, Scorer, Target, accuracy, mean, scorer
from inspect_ai.solver import TaskState
from langfuse import get_client

from src.utils.db import execute_query


def _push_langfuse_score(state: TaskState, name: str, value: float) -> None:
    """Push a score to Langfuse if a trace_id is available. Silently skips if not configured."""
    trace_id = state.metadata.get("langfuse_trace_id")
    if not trace_id:
        return
    try:
        get_client().create_score(trace_id=trace_id, name=name, value=value)
    except Exception:
        pass


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
                _push_langfuse_score(state, "syntax_valid", 0)
                return Score(value=0, explanation="sqlglot returned no parse tree.")
            _push_langfuse_score(state, "syntax_valid", 1)
            return Score(value=1, explanation=f"Valid SQL: {sql}")
        except sqlglot.errors.ParseError as e:
            _push_langfuse_score(state, "syntax_valid", 0)
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
            _push_langfuse_score(state, "execution_ok", 1)
            return Score(value=1, explanation="SQL executed successfully.")
        except Exception as e:
            _push_langfuse_score(state, "execution_ok", 0)
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
            _push_langfuse_score(state, "result_match", 1)
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
            _push_langfuse_score(state, "result_match", 0.5)
            return Score(
                value=0.5,
                explanation=(
                    f"Shape matches ({len(actual_rows)} rows × {len(actual_rows[0])} cols) "
                    f"but values differ.\nActual:   {actual_rows[:3]}\nExpected: {expected_tuples[:3]}"
                ),
            )

        _push_langfuse_score(state, "result_match", 0)
        return Score(
            value=0,
            explanation=(
                f"Result mismatch. Got {len(actual_rows)} rows, expected {len(expected_tuples)}.\n"
                f"Actual:   {actual_rows[:3]}\nExpected: {expected_tuples[:3]}"
            ),
        )

    return score


# ---------------------------------------------------------------------------
# Scorer 4: LLM-as-judge semantic correctness
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are an expert SQL evaluator. Your job is to assess whether a generated SQL \
query correctly answers a natural language question.

You will be given:
- The natural language question
- The generated SQL and its execution result (or an error message)
- The reference expected result rows

Grounding rules — read carefully:
1. Evaluate ONLY based on the data shown to you. Do not use world knowledge to
   infer what rows "should" exist (e.g. do not assume a month-by-month table
   must have 12 rows if the reference result does not).
2. Treat different representations of the same value as equivalent:
   - Dates: "2024-03-05", datetime.date(2024, 3, 5), and "2024-03-05T00:00:00"
     are all the same date.
   - Floats: 1439.74 and 1439.7400 are the same number.
   - Column aliases: "category" and "p.category" refer to the same data.
3. Row order does not matter unless the question explicitly asks for ordering.
4. Focus on whether the VALUES in the result correctly answer the question,
   not on SQL style, alias names, or formatting differences.

Output ONLY valid JSON in exactly this format:
{
  "verdict": "correct" | "partial" | "incorrect",
  "reasoning": "<one or two sentences explaining your judgment>"
}

Verdicts:
- "correct"   — the query returns data that fully and correctly answers the question
- "partial"   — the query runs but returns incomplete or slightly wrong data
                (wrong aggregation level, missing a filter, off-by-one, etc.)
- "incorrect" — the query fails to execute, returns completely wrong data,
                or doesn't address the question at all\
"""

JUDGE_USER_TEMPLATE = """\
Question: {question}

Generated SQL:
{generated_sql}

Generated SQL result:
{generated_result}

Reference expected result:
{expected_result}

Does the generated SQL correctly answer the question? Reply with JSON only.\
"""


@scorer(metrics=[accuracy()])
def semantic_judge(judge_model: str | None = None) -> Scorer:
    """
    Uses an LLM as a judge to evaluate whether the generated SQL semantically
    answers the question correctly — even when the exact rows don't match.

    This catches cases that fool result_match, such as:
    - Different column aliases returning the same data
    - Equivalent SQL written differently (e.g. IN vs JOIN)
    - Float precision edge cases

    The judge sees: the question, generated SQL, its result, and the expected
    result. It outputs a structured verdict: correct / partial / incorrect.

    Args:
        judge_model: LiteLLM model string for the judge.
                     Defaults to JUDGE_MODEL env var, then "openai/gpt-4o-mini".
    """
    _judge_model = judge_model or os.getenv("JUDGE_MODEL", "openai/gpt-4o-mini")

    async def score(state: TaskState, target: Target) -> Score:
        generated_sql = state.output.completion.strip()
        if not generated_sql:
            return Score(value=0, explanation="Empty SQL output — nothing to judge.")

        question = state.input_text

        # Execute generated SQL to get actual rows for the judge to inspect.
        # Normalize before serializing so the judge sees clean values — not
        # Python datetime/Decimal reprs — which previously caused false partials.
        try:
            actual_rows = execute_query(generated_sql)
            normalized = _normalize_rows(actual_rows)
            generated_result_str = str(normalized[:10])
            if len(normalized) > 10:
                generated_result_str += f"  ... ({len(normalized)} rows total)"
        except Exception as e:
            generated_result_str = f"ERROR: {e}"

        # Parse expected rows for display (already plain JSON values)
        try:
            expected_rows = json.loads(target.text)
            expected_result_str = str(expected_rows[:10])
            if len(expected_rows) > 10:
                expected_result_str += f"  ... ({len(expected_rows)} rows total)"
        except Exception:
            expected_result_str = target.text[:500]

        # Build the judge prompt
        user_content = JUDGE_USER_TEMPLATE.format(
            question=question,
            generated_sql=generated_sql,
            generated_result=generated_result_str,
            expected_result=expected_result_str,
        )

        # Call the judge LLM
        try:
            response = litellm.completion(
                model=_judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            verdict_data = json.loads(raw)
        except Exception as e:
            return Score(value=0, explanation=f"Judge call failed: {e}")

        verdict = verdict_data.get("verdict", "incorrect").lower().strip()
        reasoning = verdict_data.get("reasoning", "No reasoning provided.")

        score_map = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
        score_value = score_map.get(verdict, 0.0)

        _push_langfuse_score(state, "semantic_judge", score_value)

        return Score(
            value=score_value,
            explanation=f"[{verdict.upper()}] {reasoning}",
        )

    return score


# ---------------------------------------------------------------------------
# Scorer 5: Agent attempts
# ---------------------------------------------------------------------------

@scorer(metrics=[mean()])
def avg_attempts() -> Scorer:
    """
    Simply returns the number of attempts the agent took as a numerical score.
    When aggregated via mean(), it shows the average retries across the dataset.
    """
    async def score(state: TaskState, target: Target) -> Score:
        attempts = state.metadata.get("attempts", 1)
        return Score(
            value=attempts,
            explanation=f"Took {attempts} attempts to execute."
        )

    return score
