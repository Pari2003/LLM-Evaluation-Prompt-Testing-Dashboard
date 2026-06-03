"""
Centralized configuration management using Pydantic Settings.

All settings are configurable via environment variables or a .env file.
Sensible defaults are provided for local development with Ollama.

Usage:
    from src.config import settings
    print(settings.text_model)  # "llama3.2"
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # ─── Ollama Configuration ─────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    text_model: str = "llama3.2"
    embed_model: str = "nomic-embed-text"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_timeout: float = 120.0
    llm_max_retries: int = 3

    # ─── Evaluation Weights (composite score formula) ─────────────────────
    weight_latency: float = 0.15
    weight_token_efficiency: float = 0.10
    weight_semantic_quality: float = 0.35
    weight_hallucination: float = 0.25
    weight_consistency: float = 0.15

    # ─── Evaluation Thresholds ────────────────────────────────────────────
    hallucination_sim_threshold: float = 0.65
    consistency_sim_threshold: float = 0.85

    # ─── Experiment Defaults ──────────────────────────────────────────────
    default_repetitions: int = 3
    max_concurrent_runs: int = 4
    run_timeout_seconds: float = 120.0

    # ─── Storage Paths ────────────────────────────────────────────────────
    sqlite_path: str = "./data/eval_db.sqlite"

    # ─── API Server ───────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["*"]

    # ─── Embedding ────────────────────────────────────────────────────────
    embedding_dim: int = 768

    # ─── Pydantic Settings Config ─────────────────────────────────────────
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    # ─── Derived Properties ───────────────────────────────────────────────

    @property
    def sqlite_file(self) -> Path:
        """Resolved SQLite database file path."""
        path = Path(self.sqlite_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def composite_weights(self) -> dict[str, float]:
        """Evaluation dimension weights as a dict for the composite scorer."""
        return {
            "latency": self.weight_latency,
            "token_efficiency": self.weight_token_efficiency,
            "semantic_quality": self.weight_semantic_quality,
            "hallucination": self.weight_hallucination,
            "consistency": self.weight_consistency,
        }


# ─── Singleton Instance ───────────────────────────────────────────────────
# Import this everywhere: `from src.config import settings`
settings = Settings()
