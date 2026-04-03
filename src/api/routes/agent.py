"""
Agent route — POST /query

Accepts a natural language question and returns the generated SQL
along with execution metadata.
"""

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.agent.agent import PromptStrategy, generate_sql
from src.utils.db import get_connection

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
        result = await asyncio.to_thread(
            generate_sql,
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
        data=records
    )
