"""
FastAPI application entry point.

Run with:
  uvicorn src.api.main:app --reload

Endpoints:
  POST /query              — run the text-to-SQL agent
  POST /query/stream       — stream SQL tokens (SSE)
  POST /evals/run          — start an eval suite in the background, returns job_id
  GET  /evals/{job_id}     — poll eval job status and results
  GET  /health             — liveness probe (FastAPI is up)
  GET  /health/backend     — readiness probe (inference backend is reachable)
"""

import time

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.agent import router as agent_router
from src.api.routes.evals import router as evals_router

app = FastAPI(
    title="Text-to-SQL API",
    description="Text-to-SQL agent and evaluation service",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent_router)
app.include_router(evals_router, prefix="/evals")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/backend")
async def health_backend():
    """
    Readiness probe for the inference backend.

    - vLLM: issues a GET to {VLLM_API_BASE}/health and returns its status.
    - OpenAI / cloud: always reports ok (no cheap connectivity check exists).

    Response schema:
      {"status": "ok"|"degraded"|"unavailable", "backend": "vllm"|"openai",
       "model": "<model>", "latency_ms": <float>}
    """
    from src.inference.backend import get_completion_backend

    backend = get_completion_backend()
    start = time.time()

    if backend.is_vllm:
        # vLLM exposes /health at the root, not under /v1
        base_url = backend.api_base.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        health_url = f"{base_url}/health"

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(health_url)
            latency_ms = round((time.time() - start) * 1000, 1)
            status = "ok" if resp.status_code == 200 else "degraded"
            return {
                "status": status,
                "backend": "vllm",
                "model": backend.model_name,
                "latency_ms": latency_ms,
            }
        except Exception as exc:
            return {
                "status": "unavailable",
                "backend": "vllm",
                "model": backend.model_name,
                "error": str(exc),
            }

    latency_ms = round((time.time() - start) * 1000, 1)
    return {
        "status": "ok",
        "backend": "openai",
        "model": backend.model_name,
        "latency_ms": latency_ms,
    }
