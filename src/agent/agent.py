"""
Text-to-SQL agent.

Takes a natural language question + database schema and returns a SQL query.
Uses LiteLLM so you can swap models via the DEFAULT_MODEL env var.

Supports four prompt strategies via the PromptStrategy enum:
  zero_shot          — schema + question only (default)
  few_shot_static    — prepends the same 3 hand-picked examples every time
  few_shot_dynamic   — prepends the 3 most similar golden examples, selected
                       by embedding cosine similarity (one extra API call)
  chain_of_thought   — asks the model to reason step-by-step before writing
                       SQL; output is parsed from a structured Reasoning/SQL
                       format and the reasoning is stored in metadata
"""

import os
import re
from dataclasses import dataclass, field
from enum import Enum

import litellm
from dotenv import load_dotenv
from langfuse import get_client, observe

from src.agent.few_shot import Example, get_dynamic_examples, get_static_examples
from src.utils.db import get_schema_string, execute_query

load_dotenv()


# ---------------------------------------------------------------------------
# Prompt strategy
# ---------------------------------------------------------------------------

class PromptStrategy(str, Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT_STATIC = "few_shot_static"
    FEW_SHOT_DYNAMIC = "few_shot_dynamic"
    CHAIN_OF_THOUGHT = "chain_of_thought"


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

COT_SYSTEM_PROMPT = """You are an expert SQL engineer. Your job is to convert a natural language question into a single SQL query.

Think through the query step by step before writing SQL. Use EXACTLY this format — no deviations:

Reasoning:
<Identify which tables are needed, what joins are required, what filters apply, and what aggregations or ordering are needed>

SQL:
<the single SQL query, no markdown fences, no commentary>

Rules for the SQL:
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

# CoT variant — no trailing "SQL:" prompt; the model generates the full
# Reasoning/SQL block itself per the system prompt instructions
COT_USER_PROMPT_TEMPLATE = """{few_shot_block}Schema:
{schema}

Question: {question}"""


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
    reasoning: str | None = field(default=None)   # populated for chain_of_thought
    trace_id: str | None = field(default=None)
    attempts: int = 1


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_sql(raw: str) -> str:
    """
    Extract SQL from model output, handling both plain and CoT formats.

    For CoT output (contains a 'SQL:' marker), everything after the last
    'SQL:' line is taken as the query. For plain output, just strips
    markdown fences. Backward-compatible with all existing strategies.
    """
    # CoT path: find the last "SQL:" marker and take everything after it
    sql_marker = re.search(r"(?im)^SQL:\s*", raw)
    if sql_marker:
        raw = raw[sql_marker.end():]

    # Strip markdown fences (```sql ... ``` or ``` ... ```)
    raw = re.sub(r"```(?:sql)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"```", "", raw)
    return raw.strip()


def _extract_reasoning(raw: str) -> str | None:
    """
    Pull the Reasoning block out of a CoT response.
    Returns None if the expected format isn't found.
    """
    match = re.search(r"(?im)^Reasoning:\s*\n(.*?)(?=^SQL:)", raw, re.DOTALL)
    return match.group(1).strip() if match else None


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


# ---------------------------------------------------------------------------
# Main agent function
# ---------------------------------------------------------------------------

@observe(name="text-to-sql")
def generate_sql(
    question: str,
    model: str | None = None,
    schema: str | None = None,
    strategy: PromptStrategy = PromptStrategy.ZERO_SHOT,
    max_retries: int = 3,
) -> AgentResult:
    """
    Call the LLM and return the generated SQL query. Includes a self-correction 
    retry loop if the generated query fails to execute.

    Args:
        question:    Natural language question to convert.
        model:       LiteLLM model string, e.g. "openai/gpt-4o-mini".
                     Defaults to DEFAULT_MODEL env var.
        schema:      Database schema string. Defaults to the e-commerce schema.
        strategy:    Prompt strategy to use. Defaults to ZERO_SHOT.
        max_retries: Number of attempts to successfully execute the SQL.
    """
    model = model or os.getenv("DEFAULT_MODEL", "openai/gpt-4o-mini")
    schema = schema or get_schema_string()

    is_cot = strategy == PromptStrategy.CHAIN_OF_THOUGHT

    # Build few-shot block (empty for zero_shot and chain_of_thought)
    if strategy == PromptStrategy.FEW_SHOT_STATIC:
        examples = get_static_examples()
    elif strategy == PromptStrategy.FEW_SHOT_DYNAMIC:
        examples = get_dynamic_examples(question)
    else:
        examples = []

    few_shot_block = _build_few_shot_block(examples)

    # Select system prompt and user template based on strategy
    system_prompt = COT_SYSTEM_PROMPT if is_cot else SYSTEM_PROMPT
    user_template = COT_USER_PROMPT_TEMPLATE if is_cot else USER_PROMPT_TEMPLATE

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_template.format(
                few_shot_block=few_shot_block,
                schema=schema,
                question=question,
            ),
        },
    ]

    total_prompt_tokens = 0
    total_completion_tokens = 0

    for attempt in range(max_retries):
        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=0,         # deterministic output for eval reproducibility
            max_tokens=1024 if is_cot else 512,  # CoT needs room for the reasoning
        )

        total_prompt_tokens += response.usage.prompt_tokens
        total_completion_tokens += response.usage.completion_tokens

        raw = response.choices[0].message.content or ""
        sql = extract_sql(raw)
        reasoning = _extract_reasoning(raw) if is_cot else None
        
        # If the model didn't output SQL, ask again
        if not sql:
            if attempt < max_retries - 1:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "You didn't output any valid SQL query. Please provide a SQL query."})
                continue
            else:
                break
                
        # Try to execute the SQL against our DuckDB sandbox
        try:
            execute_query(sql)
            # If we get here, it executed without error! We have successful SQL.
            break
        except Exception as e:
            # It failed to execute. If we have retries left, feed the error back!
            if attempt < max_retries - 1:
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user", 
                    "content": f"The SQL query resulted in an error when executed in DuckDB:\n{str(e)}\n\nPlease fix the query and try again."
                })
            else:
                # Give up on the last attempt
                break

    lf = get_client()
    lf.update_current_span(
        input=question,
        output=sql,
        metadata={
            "model": model,
            "strategy": strategy.value,
            "reasoning": reasoning,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "attempts": attempt + 1
        },
    )
    trace_id = lf.get_current_trace_id()

    return AgentResult(
        question=question,
        sql=sql,
        model=model,
        strategy=strategy.value,
        reasoning=reasoning,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        trace_id=trace_id,
        attempts=attempt + 1,
    )


if __name__ == "__main__":
    # Quick smoke test
    result = generate_sql("How many customers are there?")
    print(f"Model:    {result.model}")
    print(f"Strategy: {result.strategy}")
    print(f"SQL:      {result.sql}")
    print(f"Tokens:   {result.prompt_tokens} in / {result.completion_tokens} out")
