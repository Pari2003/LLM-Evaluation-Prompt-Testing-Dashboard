"""
FastAPI Dependency Injection.

Provides shared singleton instances of database, LLM client, experiment runner,
and result aggregator to API route handlers via FastAPI's dependency system.

Usage:
    from src.api.dependencies import get_database
    @router.get("/")
    async def handler(db: Database = Depends(get_database)):
        ...
"""

from __future__ import annotations

from typing import Optional

from src.execution.result_aggregator import ResultAggregator
from src.execution.runner import ExperimentRunner
from src.models.llm_client import OllamaClient
from src.storage.database import Database

# ─── Singleton Instances ──────────────────────────────────────────────────
# Initialized in the FastAPI lifespan and shared across all requests.

_database: Optional[Database] = None
_llm_client: Optional[OllamaClient] = None
_runner: Optional[ExperimentRunner] = None
_aggregator: Optional[ResultAggregator] = None


def init_dependencies() -> None:
    """Initialize all singleton dependencies. Called during FastAPI startup."""
    global _database, _llm_client, _runner, _aggregator
    _database = Database()
    _llm_client = OllamaClient()
    _runner = ExperimentRunner(_llm_client, _database)
    _aggregator = ResultAggregator()


async def shutdown_dependencies() -> None:
    """Clean up all singleton dependencies. Called during FastAPI shutdown."""
    global _llm_client, _database
    if _llm_client:
        await _llm_client.close()
    if _database:
        _database.close()


def get_database() -> Database:
    """FastAPI dependency: get the shared Database instance."""
    assert _database is not None, "Database not initialized. Call init_dependencies() first."
    return _database


def get_llm_client() -> OllamaClient:
    """FastAPI dependency: get the shared OllamaClient instance."""
    assert _llm_client is not None, "LLM client not initialized. Call init_dependencies() first."
    return _llm_client


def get_runner() -> ExperimentRunner:
    """FastAPI dependency: get the shared ExperimentRunner instance."""
    assert _runner is not None, "Runner not initialized. Call init_dependencies() first."
    return _runner


def get_aggregator() -> ResultAggregator:
    """FastAPI dependency: get the shared ResultAggregator instance."""
    assert _aggregator is not None, "Aggregator not initialized. Call init_dependencies() first."
    return _aggregator
