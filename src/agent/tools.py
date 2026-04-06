"""
Tool definitions for the tool-use agent.

LiteLLM supports the OpenAI function-calling format natively across all
providers (Anthropic, Mistral, Ollama, etc.) — the same tool schema works
everywhere.

Three tools are exposed to the agent:

  query_database(sql)          — executes a SQL query against DuckDB and
                                 returns the result rows as a JSON string.
                                 The LLM writes the SQL; this tool runs it.
                                 Self-correction happens here too: if the
                                 query fails, the error is returned as the
                                 tool result so the LLM can fix and retry.

  search_knowledge_base(query) — runs semantic search over the e-commerce
                                 knowledge base documents (returns, shipping,
                                 payments, support policies). Uses the same
                                 DocumentRetriever built in Phase 1.

  get_schema(table_name?)      — returns the database schema. Optionally
                                 filtered to a specific table. Lets the agent
                                 introspect the DB before deciding what SQL
                                 to write — the agent can "look before it leaps".

Design notes:
  - Tool functions return plain strings. The LLM receives these as "tool"
    role messages and synthesises them into a final answer.
  - The KB retriever is initialised once (module-level singleton) and reused
    across calls — same pattern as the schema embedding cache.
  - All tool calls are logged in ToolCallRecord dataclasses so the caller
    can inspect the full reasoning trace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

from src.utils.db import get_schema_string, execute_query
from src.rag.retriever import build_retriever, DocumentRetriever

# ---------------------------------------------------------------------------
# KB retriever singleton (indexed once, reused across requests)
# ---------------------------------------------------------------------------

_KB_RETRIEVER: DocumentRetriever | None = None
_KB_FILE = Path("datasets/docs/ecommerce_kb.md")


def _get_kb_retriever() -> DocumentRetriever:
    global _KB_RETRIEVER
    if _KB_RETRIEVER is None:
        _KB_RETRIEVER = build_retriever(
            chunker="sentence",
            store="memory",
            chunk_size=400,
            overlap=1,
            top_k=3,
        )
        if _KB_FILE.exists():
            n = _KB_RETRIEVER.index_file(_KB_FILE)
            print(f"[tools] KB indexed: {n} chunks from {_KB_FILE.name}")
        else:
            print(f"[tools] Warning: KB file not found at {_KB_FILE}")
    return _KB_RETRIEVER


# ---------------------------------------------------------------------------
# Tool record (for logging / eval)
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRecord:
    """Captures a single tool invocation for logging and eval inspection."""
    name: str
    arguments: dict
    result: str
    success: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def query_database(sql: str) -> str:
    """
    Execute a SQL query against the DuckDB database.

    Returns query results as a JSON string on success, or an error message
    on failure. Returning the error (rather than raising) lets the LLM see
    what went wrong and self-correct with a revised query.
    """
    try:
        rows = execute_query(sql)
        if not rows:
            return "Query executed successfully. No rows returned."
        # Return up to 50 rows to avoid flooding the context window
        truncated = rows[:50]
        result = {
            "row_count": len(rows),
            "truncated": len(rows) > 50,
            "rows": truncated,
        }
        return json.dumps(result, default=str)
    except Exception as e:
        return f"SQL ERROR: {str(e)}"


def search_knowledge_base(query: str) -> str:
    """
    Search the e-commerce knowledge base for relevant information.

    Returns the top matching document chunks, formatted for injection
    into the LLM context.
    """
    retriever = _get_kb_retriever()
    return retriever.retrieve_text(query, top_k=3)


def get_schema(table_name: str | None = None) -> str:
    """
    Return the database schema.

    If table_name is provided, returns the definition for that table only.
    Otherwise returns the full schema. Lets the agent inspect available
    tables before writing SQL.
    """
    full_schema = get_schema_string()
    if not table_name:
        return full_schema

    # Filter to just the requested table
    lines = full_schema.splitlines()
    in_table = False
    result_lines = []
    for line in lines:
        if table_name.lower() in line.lower():
            in_table = True
        if in_table:
            result_lines.append(line)
            if line.strip() == "" and result_lines:
                break

    return "\n".join(result_lines) if result_lines else f"Table '{table_name}' not found in schema."


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def execute_tool(name: str, arguments: dict | str) -> tuple[str, bool, str | None]:
    """
    Dispatch a tool call by name.

    Returns:
        (result_str, success, error_msg)
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return f"Invalid JSON arguments: {arguments}", False, "JSON parse error"

    if name == "query_database":
        sql = arguments.get("sql", "")
        result = query_database(sql)
        success = not result.startswith("SQL ERROR:")
        error = result if not success else None
        return result, success, error

    elif name == "search_knowledge_base":
        query = arguments.get("query", "")
        result = search_knowledge_base(query)
        return result, True, None

    elif name == "get_schema":
        table_name = arguments.get("table_name", None)
        result = get_schema(table_name)
        return result, True, None

    else:
        msg = f"Unknown tool: {name}"
        return msg, False, msg


# ---------------------------------------------------------------------------
# OpenAI-format tool schemas (used by LiteLLM for all providers)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": (
                "Execute a SQL query against the e-commerce DuckDB database. "
                "Use this when the question requires data from the database "
                "(customer counts, revenue, orders, products, etc.). "
                "Write valid DuckDB SQL. If the query fails, you will receive "
                "the error message and can retry with a corrected query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL query to execute. Must be a SELECT statement.",
                    }
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the e-commerce knowledge base for policy and procedural information. "
                "Use this for questions about return policies, shipping options, payment methods, "
                "customer support procedures, or any non-data question about how the business works."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query describing what information you need.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": (
                "Retrieve the database schema to understand available tables and columns "
                "before writing a SQL query. Useful when unsure which table contains the "
                "needed data. Optionally filter to a specific table by name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Optional. Name of a specific table to inspect.",
                    }
                },
                "required": [],
            },
        },
    },
]
