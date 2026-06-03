"""
Health Check Routes.

Provides liveness and readiness probes for the API.
The readiness probe verifies both database and Ollama connectivity.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_database, get_llm_client
from src.models.llm_client import OllamaClient
from src.storage.database import Database

router = APIRouter(prefix="/api/v1/health", tags=["Health"])


@router.get("/live")
async def liveness():
    """Liveness probe — always returns 200 if the process is running."""
    return {"status": "alive"}


@router.get("/ready")
async def readiness(
    db: Database = Depends(get_database),
    llm: OllamaClient = Depends(get_llm_client),
):
    """Readiness probe — checks database and Ollama connectivity.

    Returns:
        Dict with overall status and per-component health.
    """
    checks = {}

    # Database check
    try:
        db.list_experiments()
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {exc}"

    # Ollama check
    ollama_ok = await llm.health_check()
    checks["ollama"] = "ok" if ollama_ok else "unreachable"

    all_ok = all(v == "ok" for v in checks.values())
    return {
        "status": "ready" if all_ok else "degraded",
        "checks": checks,
    }
