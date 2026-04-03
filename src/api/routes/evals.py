"""
Evals routes

  POST /evals/run      — kick off an eval run in the background, returns job_id
  GET  /evals/{job_id} — poll status and results

Eval runs are long (several minutes for a full dataset + judge scorer), so they
run in a thread pool and results are stored in memory keyed by job_id.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from src.agent.agent import PromptStrategy

router = APIRouter()

# In-memory job store: job_id -> job dict
# Fine for a local lab; swap for Redis in production.
_jobs: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class EvalRunRequest(BaseModel):
    model: str | None = None
    strategy: PromptStrategy = PromptStrategy.ZERO_SHOT
    difficulty: Literal["easy", "medium", "hard"] | None = None
    judge_model: str | None = None


class EvalJobResponse(BaseModel):
    job_id: str
    status: Literal["running", "completed", "failed"]
    started_at: str
    finished_at: str | None = None
    results: dict[str, Any] | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _run_eval(job_id: str, request: EvalRunRequest) -> None:
    """Synchronous eval runner — executed in a thread pool by asyncio.to_thread."""
    from inspect_ai import eval as inspect_eval
    from src.evals.tasks import text_to_sql

    try:
        task = text_to_sql(
            model=request.model,
            difficulty=request.difficulty,
            judge_model=request.judge_model,
            strategy=request.strategy,
        )

        # inspect_ai's eval() requires a model string even when our solver
        # handles model selection internally. Pass a dummy or the real one.
        import os
        inspect_model = request.model or os.getenv("DEFAULT_MODEL", "openai/gpt-4o-mini")

        logs = inspect_eval(task, model=inspect_model)
        log = logs[0]

        if log.status == "error":
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(log.error)
        else:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["results"] = _serialize_log(log)

    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)
    finally:
        _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()


def _serialize_log(log: Any) -> dict[str, Any]:
    """Pull the key metrics out of an EvalLog into a plain dict."""
    scores: dict[str, Any] = {}

    if log.results and log.results.scores:
        for eval_score in log.results.scores:
            metrics = {
                name: metric.value
                for name, metric in eval_score.metrics.items()
            }
            scores[eval_score.name] = metrics

    return {
        "total_samples": log.results.total_samples if log.results else 0,
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/run", response_model=EvalJobResponse, status_code=202)
async def run_eval(request: EvalRunRequest, background_tasks: BackgroundTasks):
    """Start an evaluation run. Returns immediately with a job_id to poll."""
    job_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()

    _jobs[job_id] = {
        "status": "running",
        "started_at": started_at,
        "finished_at": None,
        "results": None,
        "error": None,
    }

    background_tasks.add_task(asyncio.to_thread, _run_eval, job_id, request)

    return EvalJobResponse(job_id=job_id, status="running", started_at=started_at)


@router.get("/{job_id}", response_model=EvalJobResponse)
async def get_eval(job_id: str):
    """Poll the status and results of an evaluation job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    return EvalJobResponse(
        job_id=job_id,
        status=job["status"],
        started_at=job["started_at"],
        finished_at=job.get("finished_at"),
        results=job.get("results"),
        error=job.get("error"),
    )
