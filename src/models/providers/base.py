from __future__ import annotations

import abc
from typing import Any, Optional


class LLMProvider(abc.ABC):
    """Abstract base class for all LLM providers.

    Defines the standard interface required by the evaluation pipeline.
    """

    @abc.abstractmethod
    async def close(self) -> None:
        """Close any underlying network clients."""
        pass

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate text and capture latency + token metrics.

        Returns:
            Tuple of (response_text, metrics_dict)
            metrics_dict must contain: total_ms, prompt_tokens, completion_tokens, tokens_per_second
        """
        pass

    @abc.abstractmethod
    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> dict[str, Any]:
        """Generate a structured JSON response."""
        pass

    @abc.abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        pass

    @abc.abstractmethod
    async def embed_single(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""
        pass

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is reachable and responding."""
        pass
