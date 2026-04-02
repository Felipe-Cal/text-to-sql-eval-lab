"""
Text-to-SQL agent.

Takes a natural language question + database schema and returns a SQL query.
Uses LiteLLM so you can swap models via the DEFAULT_MODEL env var.
"""

import os
import re
from dataclasses import dataclass, field

import litellm
from dotenv import load_dotenv
from langfuse import get_client, observe

from src.utils.db import get_schema_string

load_dotenv()

SYSTEM_PROMPT = """You are an expert SQL engineer. Your job is to convert a natural language question into a single SQL query.

Rules:
- Output ONLY the SQL query, nothing else — no explanation, no markdown fences, no commentary.
- Use the exact table and column names from the schema provided.
- Use DuckDB SQL dialect (supports YEAR(), MONTH(), ROUND(), standard window functions).
- Do not use CTEs unless necessary — prefer subqueries for clarity.
- Always alias aggregated columns with descriptive names.
- If the question is ambiguous, make the most reasonable assumption.
"""

USER_PROMPT_TEMPLATE = """Schema:
{schema}

Question: {question}

SQL:"""


@dataclass
class AgentResult:
    question: str
    sql: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    trace_id: str | None = field(default=None)


def extract_sql(raw: str) -> str:
    """Strip markdown fences and whitespace from model output."""
    # Remove ```sql ... ``` or ``` ... ``` blocks
    raw = re.sub(r"```(?:sql)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"```", "", raw)
    return raw.strip()


@observe(name="text-to-sql")
def generate_sql(
    question: str,
    model: str | None = None,
    schema: str | None = None,
) -> AgentResult:
    """
    Call the LLM and return the generated SQL query.

    Args:
        question: Natural language question to convert.
        model:    LiteLLM model string, e.g. "openai/gpt-4o-mini".
                  Defaults to DEFAULT_MODEL env var.
        schema:   Database schema string. Defaults to the e-commerce schema.
    """
    model = model or os.getenv("DEFAULT_MODEL", "openai/gpt-4o-mini")
    schema = schema or get_schema_string()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(schema=schema, question=question),
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
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
    )
    trace_id = lf.get_current_trace_id()

    return AgentResult(
        question=question,
        sql=sql,
        model=model,
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
