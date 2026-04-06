"""
Text-to-SQL agent.

Takes a natural language question + database schema and returns a SQL query.
Uses LiteLLM so you can swap models via the DEFAULT_MODEL env var.

Supports prompt strategies via the PromptStrategy enum:
  zero_shot          — schema + question only (default)
  few_shot_static    — prepends the same 3 hand-picked examples every time
  few_shot_dynamic   — prepends the 3 most similar golden examples, selected
                       by embedding cosine similarity (one extra API call)
  chain_of_thought   — asks the model to reason step-by-step before writing
                       SQL; output is parsed from a structured Reasoning/SQL
                       format and the reasoning is stored in metadata
  rag                — retrieves top-K relevant tables from the 50-table DWH
                       instead of sending the full schema
  dspy               — uses the DSPy compiled module for inference
  routed             — classifies question difficulty, then routes to the
                       appropriate model + strategy automatically
  tool_use           — agentic strategy: the LLM is given tools (query_database,
                       search_knowledge_base, get_schema) and decides which to
                       call. Can answer both data questions (via SQL) and policy
                       questions (via KB search) in a single interface.
"""

import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TypedDict, Annotated, Sequence
import operator

import litellm
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
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
    RAG_DENSE = "rag_dense"
    RAG_SPARSE = "rag_sparse"
    RAG_HYBRID = "rag_hybrid"
    DSPY = "dspy"
    ROUTED = "routed"
    TOOL_USE = "tool_use"
    RAG = "rag"  # Keep for backward compatibility, defaults to hybrid or dense


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
    reasoning: str | None = field(default=None)       # populated for chain_of_thought
    answer: str | None = field(default=None)          # populated for tool_use (final LLM synthesis)
    tool_calls: list[dict] = field(default_factory=list)  # populated for tool_use
    trace_id: str | None = field(default=None)
    attempts: int = 1
    cost: float = 0.0
    latency: float = 0.0
    retrieved_tables: list[str] = field(default_factory=list)
    routed_difficulty: str | None = field(default=None)   # populated for routed strategy
    router_method: str | None = field(default=None)       # rule_based | embedding


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
# Tool-use agentic loop
# ---------------------------------------------------------------------------

TOOL_USE_SYSTEM_PROMPT = """You are a helpful data assistant for an e-commerce company.
You have access to three tools:
  - query_database: run SQL queries against the live database (customers, orders, products, order_items)
  - search_knowledge_base: search company policies and procedures (returns, shipping, payments, support)
  - get_schema: inspect database table definitions before writing SQL

Use the tools to gather information, then synthesise a clear, concise answer.
Rules:
  - Always use get_schema or inspect the schema before writing complex SQL if you are unsure of column names.
  - Prefer query_database for any question involving numbers, counts, or data from the database.
  - Prefer search_knowledge_base for policy or procedural questions.
  - You may call multiple tools in sequence if needed.
  - When you have enough information, respond with a final answer — do not call more tools than necessary.
  - If a SQL query fails, read the error and retry with a corrected query.
"""


def _run_tool_use_loop(
    question: str,
    model: str,
    max_iterations: int = 10,
) -> tuple[str, str, list[dict], int, int, float, int]:
    """
    Agentic tool-use loop using LiteLLM function calling.

    The LLM receives the question and a set of tool schemas. It decides which
    tools to call, receives their results, and continues until it produces a
    final text response (finish_reason != "tool_calls").

    Returns:
        (answer, sql, tool_calls_log, prompt_tokens, completion_tokens, cost, iterations)
    """
    from src.agent.tools import TOOL_SCHEMAS, execute_tool, ToolCallRecord

    messages = [
        {"role": "system", "content": TOOL_USE_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    tool_calls_log: list[dict] = []
    last_sql: str = ""
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    iterations = 0

    for iteration in range(max_iterations):
        iterations = iteration + 1

        response = litellm.completion(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",   # LLM decides whether to call a tool or respond
            temperature=0,
        )

        total_prompt_tokens += response.usage.prompt_tokens
        total_completion_tokens += response.usage.completion_tokens
        try:
            cost = litellm.completion_cost(completion_response=response)
            if cost:
                total_cost += cost
        except Exception:
            pass

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # Append the assistant message (may contain tool_calls)
        messages.append(message)

        # If the LLM responded with text — we are done
        if finish_reason != "tool_calls" or not message.tool_calls:
            answer = message.content or ""
            return answer, last_sql, tool_calls_log, total_prompt_tokens, total_completion_tokens, total_cost, iterations

        # Execute each tool call and feed results back
        for tc in message.tool_calls:
            name = tc.function.name
            args = tc.function.arguments  # raw JSON string from the model

            result_str, success, error = execute_tool(name, args)

            # Track SQL generated by query_database for eval compatibility
            if name == "query_database":
                try:
                    import json as _json
                    parsed_args = _json.loads(args) if isinstance(args, str) else args
                    last_sql = parsed_args.get("sql", "")
                except Exception:
                    pass

            tool_calls_log.append({
                "iteration": iteration + 1,
                "tool": name,
                "arguments": args,
                "result": result_str[:500],  # truncate for logging
                "success": success,
                "error": error,
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

    # Max iterations reached — return whatever we have
    return (
        "Max tool iterations reached without a final answer.",
        last_sql,
        tool_calls_log,
        total_prompt_tokens,
        total_completion_tokens,
        total_cost,
        iterations,
    )


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

    # Routing: classify difficulty and pick the best model + strategy automatically.
    # The resolved strategy replaces ROUTED for all subsequent logic.
    routed_difficulty: str | None = None
    router_method: str | None = None
    if strategy == PromptStrategy.ROUTED:
        from src.agent.router import route
        decision = route(question)
        model = decision.model
        strategy = PromptStrategy(decision.strategy)
        routed_difficulty = decision.difficulty
        router_method = decision.method

    # Tool-use has its own loop — handle it separately before building prompts
    if strategy == PromptStrategy.TOOL_USE:
        start_time = time.time()
        answer, sql, tool_calls_log, prompt_tokens, completion_tokens, cost, iterations = \
            _run_tool_use_loop(question, model)
        latency = time.time() - start_time

        lf = get_client()
        lf.update_current_span(
            input=question,
            output=answer,
            metadata={
                "model": model,
                "strategy": strategy.value,
                "tool_calls": tool_calls_log,
                "iterations": iterations,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
                "latency": latency,
            },
        )
        trace_id = lf.get_current_trace_id()

        return AgentResult(
            question=question,
            sql=sql,
            answer=answer,
            tool_calls=tool_calls_log,
            model=model,
            strategy=strategy.value,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            trace_id=trace_id,
            attempts=iterations,
            cost=cost,
            latency=latency,
        )

    is_cot = strategy == PromptStrategy.CHAIN_OF_THOUGHT

    if strategy in (PromptStrategy.RAG, PromptStrategy.RAG_DENSE, PromptStrategy.RAG_SPARSE, PromptStrategy.RAG_HYBRID, PromptStrategy.DSPY):
        from src.agent.schema_retriever import retrieve_schema
        
        # Map strategy to retrieval type
        retrieval_type = "hybrid"
        if strategy == PromptStrategy.RAG_DENSE:
            retrieval_type = "dense"
        elif strategy == PromptStrategy.RAG_SPARSE:
            retrieval_type = "sparse"
        elif strategy == PromptStrategy.RAG:
            retrieval_type = "dense" # Default RAG to dense for now
            
        schema_local, retrieved_tables = retrieve_schema(question, top_k=5, retrieval_type=retrieval_type)
        schema = schema or schema_local
    else:
        schema = schema or get_schema_string()
        retrieved_tables = []

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

    start_time = time.time()
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0

    if strategy == PromptStrategy.DSPY:
        from src.agent.dspy_module import get_dspy_agent
        dspy_agent, dspy_lm = get_dspy_agent(model)
        
        for attempt in range(max_retries):
            pred = dspy_agent(schema=schema, question=question)
            raw = pred.sql
            sql = extract_sql(raw)
            reasoning = getattr(pred, "reasoning", "")
            
            try:
                execute_query(sql)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    question = f"{question}\n\nLast SQL failed: {str(e)}. Fix it."
                else:
                    break
                    
        # Extract tokens and cost natively from DSPy 3's history
        if getattr(dspy_lm, "history", None):
            for interaction in dspy_lm.history[-max_retries:]:
                if "usage" in interaction and interaction["usage"]:
                    usage = interaction["usage"]
                    if hasattr(usage, "prompt_tokens"):
                        total_prompt_tokens += getattr(usage, "prompt_tokens", 0)
                        total_completion_tokens += getattr(usage, "completion_tokens", 0)
                    elif isinstance(usage, dict):
                        total_prompt_tokens += usage.get("prompt_tokens", 0)
                        total_completion_tokens += usage.get("completion_tokens", 0)
                
                if "cost" in interaction and interaction["cost"]:
                    total_cost += float(interaction["cost"])
    else:
        class AgentState(TypedDict):
            messages: list[dict]
            model: str
            is_cot: bool
            max_retries: int
            attempt: int
            sql: str
            raw_output: str
            reasoning: str | None
            error: str | None
            prompt_tokens: int
            completion_tokens: int
            cost: float

        def generate_node(state: AgentState):
            response = litellm.completion(
                model=state["model"],
                messages=state["messages"],
                temperature=0,
                max_tokens=1024 if state["is_cot"] else 512,
            )
            
            p_tokens = response.usage.prompt_tokens or 0
            c_tokens = response.usage.completion_tokens or 0
            call_cost = 0.0
            try:
                c = litellm.completion_cost(completion_response=response)
                if c: call_cost = float(c)
            except Exception:
                pass
                
            raw = response.choices[0].message.content or ""
            extracted_sql = extract_sql(raw)
            rsn = _extract_reasoning(raw) if state["is_cot"] else None
            
            new_messages = state["messages"].copy()
            new_messages.append({"role": "assistant", "content": raw})
            
            return {
                "messages": new_messages,
                "sql": extracted_sql,
                "raw_output": raw,
                "reasoning": rsn,
                "attempt": state["attempt"] + 1,
                "prompt_tokens": state["prompt_tokens"] + p_tokens,
                "completion_tokens": state["completion_tokens"] + c_tokens,
                "cost": state["cost"] + call_cost
            }

        def execute_node(state: AgentState):
            current_sql = state["sql"]
            if not current_sql:
                error_msg = "You didn't output any valid SQL query. Please provide a SQL query."
                new_msgs = state["messages"].copy()
                new_msgs.append({"role": "user", "content": error_msg})
                return {"error": error_msg, "messages": new_msgs}
                
            try:
                execute_query(current_sql)
                return {"error": None}
            except Exception as e:
                error_msg = f"The SQL query resulted in an error when executed in DuckDB:\n{str(e)}\n\nPlease fix the query and try again."
                new_msgs = state["messages"].copy()
                new_msgs.append({"role": "user", "content": error_msg})
                return {"error": error_msg, "messages": new_msgs}

        def route_after_execute(state: AgentState):
            if not state["error"]:
                return END
            if state["attempt"] >= state["max_retries"]:
                return END
            return "generate_node"

        workflow = StateGraph(AgentState)
        workflow.add_node("generate_node", generate_node)
        workflow.add_node("execute_node", execute_node)
        
        workflow.set_entry_point("generate_node")
        workflow.add_edge("generate_node", "execute_node")
        workflow.add_conditional_edges(
            "execute_node",
            route_after_execute,
            {
                "generate_node": "generate_node",
                END: END
            }
        )
        
        app = workflow.compile()
        
        initial_state = {
            "messages": messages,
            "model": model,
            "is_cot": is_cot,
            "max_retries": max_retries,
            "attempt": 0,
            "sql": "",
            "raw_output": "",
            "reasoning": None,
            "error": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost": 0.0
        }
        
        final_state = app.invoke(initial_state)
        
        total_prompt_tokens = final_state["prompt_tokens"]
        total_completion_tokens = final_state["completion_tokens"]
        total_cost = final_state["cost"]
        sql = final_state["sql"]
        reasoning = final_state["reasoning"]
        attempt = final_state["attempt"] - 1

    end_time = time.time()
    latency = end_time - start_time

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
            "cost": total_cost,
            "latency": latency,
            "attempts": attempt + 1,
            "retrieved_tables": retrieved_tables,
            "routed_difficulty": routed_difficulty,
            "router_method": router_method,
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
        cost=total_cost,
        latency=latency,
        retrieved_tables=retrieved_tables,
        routed_difficulty=routed_difficulty,
        router_method=router_method,
    )


if __name__ == "__main__":
    # Quick smoke test
    result = generate_sql("How many customers are there?")
    print(f"Model:    {result.model}")
    print(f"Strategy: {result.strategy}")
    print(f"SQL:      {result.sql}")
    print(f"Tokens:   {result.prompt_tokens} in / {result.completion_tokens} out")
