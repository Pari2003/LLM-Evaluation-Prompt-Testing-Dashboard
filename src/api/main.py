"""
FastAPI Application Factory.

Creates and configures the FastAPI application with:
- Lifespan management (startup/shutdown hooks for dependencies)
- CORS middleware
- All API route registrations
- Swagger/OpenAPI documentation

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import init_dependencies, shutdown_dependencies
from src.api.routes import datasets, experiments, health, results
from src.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize dependencies on startup, clean up on shutdown."""
    init_dependencies()
    yield
    await shutdown_dependencies()


app = FastAPI(
    title="LLM Evaluation & Prompt Testing Dashboard",
    description=(
        "A platform for benchmarking LLM responses across latency, token efficiency, "
        "hallucination detection, and prompt variant comparison using controlled experiments. "
        "Runs locally on Llama 3.2 via Ollama."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── Middleware ────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Route Registration ──────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(datasets.router)
app.include_router(experiments.router)
app.include_router(results.router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "LLM Evaluation & Prompt Testing Dashboard",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health/ready",
    }
