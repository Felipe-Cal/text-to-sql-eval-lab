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
from typing import Any, Literal

import datetime
from decimal import Decimal

import litellm
import instructor
from pydantic import BaseModel, Field
import sqlglot
from inspect_ai.scorer import Score, Scorer, Target, accuracy, mean, scorer
from inspect_ai.solver import TaskState
from langfuse import get_client

from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import GPTModel # DeepEval's default, we can wrap LiteLLM if needed

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

class VerdictData(BaseModel):
    verdict: Literal["correct", "partial", "incorrect"] = Field(
        description="Whether the data fully answers the question (correct), is slightly wrong/incomplete (partial), or is completely wrong (incorrect)."
    )
    reasoning: str = Field(
        description="One or two sentences explaining the judgment."
    )


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

        # Call the judge LLM with Instructor for structured output
        client = instructor.from_litellm(litellm.completion)
        try:
            verdict_obj = client.chat.completions.create(
                model=_judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
                max_tokens=256,
                response_model=VerdictData,
            )
        except Exception as e:
            return Score(value=0, explanation=f"Judge call failed: {e}")

        verdict = verdict_obj.verdict
        reasoning = verdict_obj.reasoning

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


# ---------------------------------------------------------------------------
# Scorer 6: Cost, Latency, and Tokens
# ---------------------------------------------------------------------------

@scorer(metrics=[mean()])
def avg_cost() -> Scorer:
    """Returns the cost in USD of the API calls for a question."""
    async def score(state: TaskState, target: Target) -> Score:
        val = state.metadata.get("cost", 0.0)
        return Score(value=val, explanation=f"${val:.5f} total cost")

    return score


@scorer(metrics=[mean()])
def avg_latency() -> Scorer:
    """Returns the wall-clock execution time for a question."""
    async def score(state: TaskState, target: Target) -> Score:
        val = state.metadata.get("latency", 0.0)
        return Score(value=val, explanation=f"{val:.2f}s latency")

    return score


@scorer(metrics=[mean()])
def avg_total_tokens() -> Scorer:
    """Returns the total token usage (prompt + completion) for a question."""
    async def score(state: TaskState, target: Target) -> Score:
        val = state.metadata.get("total_tokens", 0)
        return Score(value=val, explanation=f"{val} tokens used")

    return score


@scorer(metrics=[mean()])
def avg_tool_calls() -> Scorer:
    """
    Returns the number of tool calls made by the tool_use agent per question.
    For non-tool-use strategies this is always 0, which makes it a useful
    comparison metric — it quantifies the extra LLM round-trips the agentic
    approach costs relative to a single-shot strategy.
    """
    async def score(state: TaskState, target: Target) -> Score:
        tool_calls = state.metadata.get("tool_calls", [])
        n = len(tool_calls)
        tools_used = list({tc.get("tool") for tc in tool_calls}) if tool_calls else []
        explanation = f"{n} tool calls: {tools_used}" if tools_used else "0 tool calls (non-agentic strategy)"
        return Score(value=n, explanation=explanation)

    return score


@scorer(metrics=[mean()])
def retrieval_recall() -> Scorer:
    """Returns the % of required tables that were successfully retrieved by the RAG module."""
    async def score(state: TaskState, target: Target) -> Score:
        golden_sql = state.metadata.get("reference_sql", "")
        try:
            ast = sqlglot.parse_one(golden_sql, read="duckdb")
            # extract table names
            required_tables = set(t.name.lower() for t in ast.find_all(sqlglot.expressions.Table))
        except Exception:
            return Score(value=0.0, explanation="Failed to parse golden SQL")

        # Exclude temporary CTE names or subqueries that might show up as tables
        # For simplicity, we assume they are standard if they match our known lists, but 
        # doing 'in retrieved_tables' covers it well.
            
        retrieved_tables = state.metadata.get("retrieved_tables", [])
        
        if not retrieved_tables:
             return Score(value=1.0, explanation="Full schema used (no retriever)")

        found = 0
        for req in required_tables:
            # Check if any retrieved table definition starts with the table name (e.g., "orders(")
            if any(f"{req}(" in rt.lower() for rt in retrieved_tables):
                found += 1
                
        ratio = found / len(required_tables) if required_tables else 1.0
        return Score(value=ratio, explanation=f"Found {found}/{len(required_tables)} golden tables in top-K")

    return score


# ---------------------------------------------------------------------------
# Scorer 7: DeepEval Diagnostic Metrics
# ---------------------------------------------------------------------------

@scorer(metrics=[mean()])
def faithfulness_score() -> Scorer:
    """
    Measures how well the generated SQL/Reasoning is grounded in the retrieved schema.
    Uses DeepEval's FaithfulnessMetric.
    """
    async def score(state: TaskState, target: Target) -> Score:
        generated_sql = state.output.completion
        retrieved_context = state.metadata.get("retrieved_tables", [])
        
        if not retrieved_context:
            # If no RAG was used, faithfulness is trivially 1.0 (or we skip)
            return Score(value=1.0, explanation="No RAG context to verify faithfulness against.")

        test_case = LLMTestCase(
            input=state.input_text,
            actual_output=generated_sql,
            retrieval_context=retrieved_context
        )
        
        metric = FaithfulnessMetric(threshold=0.5)
        # DeepEval 2026/late 2s uses synchronous measure mostly in these versions, 
        # or we can use async if available. For now, running sync is safe in this harness.
        metric.measure(test_case)
        
        val = float(metric.score)
        reason = metric.reason
        
        _push_langfuse_score(state, "faithfulness", val)
        return Score(value=val, explanation=reason)

    return score


@scorer(metrics=[mean()])
def answer_relevancy_score() -> Scorer:
    """
    Measures if the SQL truly addresses the prompt's intent.
    Uses DeepEval's AnswerRelevancyMetric.
    """
    async def score(state: TaskState, target: Target) -> Score:
        test_case = LLMTestCase(
            input=state.input_text,
            actual_output=state.output.completion,
            retrieval_context=state.metadata.get("retrieved_tables", [])
        )
        
        metric = AnswerRelevancyMetric(threshold=0.5)
        metric.measure(test_case)
        
        val = float(metric.score)
        reason = metric.reason
        
        _push_langfuse_score(state, "answer_relevancy", val)
        return Score(value=val, explanation=reason)

    return score


@scorer(metrics=[mean()])
def sql_quality_geval() -> Scorer:
    """
    Uses DeepEval's G-Eval to score SQL 'elegance' and 'optimization'.
    Rubric: Correctness, Efficiency (no redundant JOINs), Readability.
    """
    async def score(state: TaskState, target: Target) -> Score:
        quality_metric = GEval(
            name="SQL Quality",
            criteria="Determine if the SQL is efficient, readable, and follows best practices (e.g., using window functions correctly, descriptive aliases, no redundant joins).",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.5,
        )
        
        test_case = LLMTestCase(
            input=state.input_text,
            actual_output=state.output.completion
        )
        
        quality_metric.measure(test_case)
        val = float(quality_metric.score)
        reason = quality_metric.reason
        
        _push_langfuse_score(state, "sql_quality", val)
        return Score(value=val, explanation=reason)

    return score

