"""
Inspect AI Task definition for text-to-SQL evaluation.

Concepts:
  Dataset  — our 15-question golden dataset
  Solver   — calls our text-to-sql agent and puts the SQL in state.output
  Scorers  — syntax_valid, execution_ok, result_match (run in parallel)
  Task     — wires everything together

Run with:
  inspect eval src/evals/tasks.py --model openai/gpt-4o-mini
"""

import json
import os
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.scorer import mean
from inspect_ai.solver import Generate, Solver, TaskState, solver

from src.agent.agent import extract_sql, generate_sql
from src.evals.scorers import execution_ok, result_match, syntax_valid
from src.utils.db import get_schema_string

GOLDEN_PATH = Path(__file__).parent.parent.parent / "datasets" / "golden" / "questions.json"


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_golden_dataset(difficulty: str | None = None) -> Dataset:
    """
    Load the golden Q&A pairs as an Inspect AI Dataset.

    Each Sample:
      input  — the natural language question (what the LLM sees)
      target — JSON-encoded expected rows (what the scorer checks against)
      metadata — difficulty, sql, id
    """
    with open(GOLDEN_PATH) as f:
        questions = json.load(f)

    if difficulty:
        questions = [q for q in questions if q["difficulty"] == difficulty]

    samples = [
        Sample(
            input=q["question"],
            target=json.dumps(q["expected_rows"]),
            metadata={
                "id": q["id"],
                "difficulty": q["difficulty"],
                "reference_sql": q["sql"],
                "notes": q.get("notes", ""),
            },
        )
        for q in questions
    ]

    return MemoryDataset(name="text-to-sql-golden", samples=samples)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

@solver
def text_to_sql_solver(model: str | None = None) -> Solver:
    """
    Solver that calls our agent and writes the generated SQL into state.output.
    Inspect AI will pass state.output.completion to each scorer.
    """
    _model = model or os.getenv("DEFAULT_MODEL", "openai/gpt-4o-mini")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        question = state.input_text
        schema = get_schema_string()

        result = generate_sql(question=question, model=_model, schema=schema)

        # Store the SQL as the completion — scorers will read state.output.completion
        state.output.completion = result.sql

        # Attach metadata for Langfuse / Inspect logs
        state.metadata["generated_sql"] = result.sql
        state.metadata["model"] = result.model
        state.metadata["prompt_tokens"] = result.prompt_tokens
        state.metadata["completion_tokens"] = result.completion_tokens
        state.metadata["langfuse_trace_id"] = result.trace_id

        return state

    return solve


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@task
def text_to_sql(
    model: str | None = None,
    difficulty: str | None = None,
) -> Task:
    """
    Full text-to-SQL evaluation task.

    Args:
        model:      LiteLLM model string (overrides DEFAULT_MODEL env var).
        difficulty: Filter dataset by difficulty: 'easy', 'medium', 'hard', or None for all.

    Scorers run in parallel:
      - syntax_valid  (did it parse?)
      - execution_ok  (did it run?)
      - result_match  (did it return the right rows?)
    """
    return Task(
        dataset=load_golden_dataset(difficulty=difficulty),
        solver=text_to_sql_solver(model=model),
        scorer=[
            syntax_valid(),
            execution_ok(),
            result_match(),
        ],
    )
