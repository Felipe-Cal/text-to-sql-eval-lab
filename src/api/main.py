"""
FastAPI application entry point.

Run with:
  uvicorn src.api.main:app --reload

Endpoints:
  POST /query          — run the text-to-SQL agent
  POST /evals/run      — start an eval suite in the background, returns job_id
  GET  /evals/{job_id} — poll eval job status and results
"""

from fastapi import FastAPI

from src.api.routes.agent import router as agent_router
from src.api.routes.evals import router as evals_router

app = FastAPI(
    title="Text-to-SQL API",
    description="Text-to-SQL agent and evaluation service",
    version="0.1.0",
)

app.include_router(agent_router)
app.include_router(evals_router, prefix="/evals")


@app.get("/health")
def health():
    return {"status": "ok"}
