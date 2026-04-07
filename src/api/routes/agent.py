"""
Agent route — POST /query and POST /query/stream

Accepts a natural language question and returns the generated SQL
along with execution metadata.
"""

import json as _json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agent.agent import PromptStrategy, agenerate_sql, agenerate_sql_stream
from src.utils.db import get_connection, get_schema_string

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    model: str | None = None
    strategy: PromptStrategy = PromptStrategy.ZERO_SHOT


class QueryResponse(BaseModel):
    question: str
    sql: str
    model: str
    strategy: str
    reasoning: str | None
    prompt_tokens: int
    completion_tokens: int
    cost: float
    latency: float
    attempts: int
    trace_id: str | None
    routed_difficulty: str | None
    router_method: str | None
    data: list[dict] = []


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Run the text-to-SQL agent on a natural language question."""
    try:
        result = await agenerate_sql(
            question=request.question,
            model=request.model,
            strategy=request.strategy,
        )

        # Execute the returned SQL to get pandas DataFrame -> list of dicts
        con = get_connection()
        try:
            # fillna("") ensures JSON compliant response avoiding NaN/Infinity errors
            df = con.execute(result.sql).fetchdf().fillna("")
            records = df.to_dict(orient="records")
        except Exception as e:
            records = [{"error": str(e)}]
        finally:
            con.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        question=result.question,
        sql=result.sql,
        model=result.model,
        strategy=result.strategy,
        reasoning=result.reasoning,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cost=result.cost,
        latency=result.latency,
        attempts=result.attempts,
        trace_id=result.trace_id,
        routed_difficulty=result.routed_difficulty,
        router_method=result.router_method,
        data=records,
    )


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Stream agent progress as Server-Sent Events (SSE).

    Each SSE data payload is a JSON object with a "type" field:

      {"type": "start",       "strategy": "...", "model": "..."}
      {"type": "sql_token",   "content": "SELECT "}
      {"type": "tool_call",   "tool": "get_schema", "input": {...}}
      {"type": "tool_result", "tool": "get_schema", "output": "...", "success": true}
      {"type": "retry",       "attempt": 2, "error": "column not found"}
      {"type": "done",        "sql": "...", "cost": 0.001, "latency": 2.3,
                              "attempts": 1, "prompt_tokens": 80, "completion_tokens": 40}
      {"type": "error",       "message": "..."}

    tool_use emits tool_call / tool_result events as each invocation happens.
    dspy and routed strategies are not token-streamable — they emit start then done.
    """
    async def event_generator():
        try:
            async for event in agenerate_sql_stream(
                question=request.question,
                model=request.model,
                strategy=request.strategy,
            ):
                yield f"data: {_json.dumps(event)}\n\n"
        except Exception as exc:
            yield f"data: {_json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


class ExecuteSqlRequest(BaseModel):
    sql: str


@router.post("/sql/execute")
async def execute_sql(request: ExecuteSqlRequest):
    """Execute a SQL query directly and return result rows. Used by the UI after streaming."""
    con = get_connection()
    try:
        df = con.execute(request.sql).fetchdf().fillna("")
        return {"data": df.to_dict(orient="records"), "error": None}
    except Exception as e:
        return {"data": [], "error": str(e)}
    finally:
        con.close()
