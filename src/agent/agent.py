"""
Text-to-SQL agent.

Takes a natural language question + database schema and returns a SQL query.
Uses LiteLLM so you can swap models via the DEFAULT_MODEL env var.

Supports three prompt strategies via the PromptStrategy enum:
  zero_shot       — schema + question only (default)
  few_shot_static — prepends the same 3 hand-picked examples every time
  few_shot_dynamic — prepends the 3 most similar golden examples, selected
                     by embedding cosine similarity (one extra API call)
"""

import os
import re
from dataclasses import dataclass, field
from enum import Enum

import litellm
from dotenv import load_dotenv
from langfuse import get_client, observe

from src.agent.few_shot import Example, get_dynamic_examples, get_static_examples
from src.utils.db import get_schema_string

load_dotenv()


# ---------------------------------------------------------------------------
# Prompt strategy
# ---------------------------------------------------------------------------

class PromptStrategy(str, Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT_STATIC = "few_shot_static"
    FEW_SHOT_DYNAMIC = "few_shot_dynamic"


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert SQL engineer. Your job is to convert a natural language question into a single SQL query.

Rules:
- Output ONLY the SQL query, nothing else — no explanation, no markdown fences, no commentary.
- Use the exact table and column names from the schema provided.
- Use DuckDB SQL dialect (supports YEAR(), MONTH(), ROUND(), standard window functions).
- Do not use CTEs unless necessary — prefer subqueries for clarity.
- Always alias aggregated columns with descriptive names.
- If the question is ambiguous, make the most reasonable assumption.
"""

# Inserted before the schema when few-shot examples are provided
FEW_SHOT_EXAMPLE_TEMPLATE = "Question: {question}\nSQL: {sql}"

USER_PROMPT_TEMPLATE = """{few_shot_block}Schema:
{schema}

Question: {question}

SQL:"""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    question: str
    sql: str
    model: str
    strategy: str
    prompt_tokens: int
    completion_tokens: int
    trace_id: str | None = field(default=None)


def extract_sql(raw: str) -> str:
    """Strip markdown fences and whitespace from model output."""
    # Remove ```sql ... ``` or ``` ... ``` blocks
    raw = re.sub(r"```(?:sql)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"```", "", raw)
    return raw.strip()


def _build_few_shot_block(examples: list[Example]) -> str:
    """
    Format a list of examples into a preamble block to prepend to the prompt.
    Returns an empty string when the list is empty (zero-shot path).
    """
    if not examples:
        return ""
    lines = ["Here are some example question → SQL pairs:\n"]
    for ex in examples:
        lines.append(FEW_SHOT_EXAMPLE_TEMPLATE.format(question=ex.question, sql=ex.sql))
        lines.append("")  # blank line between examples
    lines.append("---\n")  # separator before the actual question
    return "\n".join(lines)


@observe(name="text-to-sql")
def generate_sql(
    question: str,
    model: str | None = None,
    schema: str | None = None,
    strategy: PromptStrategy = PromptStrategy.ZERO_SHOT,
) -> AgentResult:
    """
    Call the LLM and return the generated SQL query.

    Args:
        question: Natural language question to convert.
        model:    LiteLLM model string, e.g. "openai/gpt-4o-mini".
                  Defaults to DEFAULT_MODEL env var.
        schema:   Database schema string. Defaults to the e-commerce schema.
        strategy: Prompt strategy controlling how (if at all) few-shot
                  examples are injected. Defaults to ZERO_SHOT.
    """
    model = model or os.getenv("DEFAULT_MODEL", "openai/gpt-4o-mini")
    schema = schema or get_schema_string()

    # Build few-shot block based on strategy
    if strategy == PromptStrategy.FEW_SHOT_STATIC:
        examples = get_static_examples()
    elif strategy == PromptStrategy.FEW_SHOT_DYNAMIC:
        examples = get_dynamic_examples(question)
    else:
        examples = []

    few_shot_block = _build_few_shot_block(examples)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                few_shot_block=few_shot_block,
                schema=schema,
                question=question,
            ),
        },
    ]

    response = litellm.completion(
        model=model,
        messages=messages,
        temperature=0,       # deterministic output for eval reproducibility
        max_tokens=512,
    )

    raw_sql = response.choices[0].message.content or ""
    sql = extract_sql(raw_sql)

    lf = get_client()
    lf.update_current_span(
        input=question,
        output=sql,
        metadata={
            "model": model,
            "strategy": strategy.value,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
    )
    trace_id = lf.get_current_trace_id()

    return AgentResult(
        question=question,
        sql=sql,
        model=model,
        strategy=strategy.value,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        trace_id=trace_id,
    )


if __name__ == "__main__":
    # Quick smoke test
    result = generate_sql("How many customers are there?")
    print(f"Model:  {result.model}")
    print(f"SQL:    {result.sql}")
    print(f"Tokens: {result.prompt_tokens} in / {result.completion_tokens} out")
