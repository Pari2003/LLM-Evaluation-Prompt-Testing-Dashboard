"""
Ollama LLM Client with Latency Instrumentation.

Wraps the Ollama REST API to provide:
- Text generation with timing capture (total_ms, tokens/sec)
- Structured JSON generation with retry logic
- Embedding generation (single and batch)
- Token counting from Ollama response metadata

Adapted from the Agentic RAG project's OllamaClient with added
latency profiling for the evaluation pipeline.

Usage:
    client = OllamaClient()
    text, latency, tokens = await client.generate("Explain RAG.")
    embedding = await client.embed_single("Some text")
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

import httpx
import numpy as np
import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


class OllamaClient:
    """Async client for Ollama API with latency instrumentation and retry logic."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        text_model: Optional[str] = None,
        embed_model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.text_model = text_model or settings.text_model
        self.embed_model = embed_model or settings.embed_model
        self.timeout = timeout or settings.llm_timeout
        self.max_retries = max_retries or settings.llm_max_retries

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout, connect=10.0),
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # ─── Text Generation ──────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate text from Ollama and capture latency + token metrics.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system message.
            temperature: Sampling temperature override.
            max_tokens: Max output tokens override.

        Returns:
            Tuple of (response_text, metrics_dict) where metrics_dict contains:
                - total_ms: Total wall-clock generation time
                - prompt_tokens: Number of tokens in the prompt (eval_count from Ollama)
                - completion_tokens: Number of tokens generated
                - tokens_per_second: Generation throughput
        """
        temp = temperature if temperature is not None else settings.llm_temperature
        tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens

        payload: dict[str, Any] = {
            "model": self.text_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        start_time = time.perf_counter()
        response_data = await self._request_with_retry("/api/generate", payload)
        total_ms = (time.perf_counter() - start_time) * 1000

        response_text = response_data.get("response", "").strip()

        # Extract token counts from Ollama metadata
        prompt_eval_count = response_data.get("prompt_eval_count", 0)
        eval_count = response_data.get("eval_count", 0)

        # Compute tokens/sec from Ollama's reported eval_duration (nanoseconds)
        eval_duration_ns = response_data.get("eval_duration", 0)
        if eval_duration_ns > 0 and eval_count > 0:
            tokens_per_second = eval_count / (eval_duration_ns / 1e9)
        else:
            tokens_per_second = 0.0

        metrics = {
            "total_ms": round(total_ms, 1),
            "prompt_tokens": prompt_eval_count,
            "completion_tokens": eval_count,
            "tokens_per_second": round(tokens_per_second, 2),
        }

        logger.debug(
            "ollama_generate_complete",
            model=self.text_model,
            prompt_len=len(prompt),
            response_len=len(response_text),
            **metrics,
        )

        return response_text, metrics

    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> dict[str, Any]:
        """Generate a structured JSON response from Ollama.

        Uses Ollama's JSON format mode to ensure valid JSON output.
        Falls back to extracting JSON from the response text if format mode fails.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system message.
            temperature: Sampling temperature override.

        Returns:
            Parsed JSON dictionary.
        """
        temp = temperature if temperature is not None else settings.llm_temperature

        payload: dict[str, Any] = {
            "model": self.text_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temp,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        response_data = await self._request_with_retry("/api/generate", payload)
        response_text = response_data.get("response", "").strip()

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Attempt to extract JSON from markdown fences
            logger.warning("json_parse_fallback", raw_response=response_text[:200])
            return self._extract_json_from_text(response_text)

    # ─── Embeddings ───────────────────────────────────────────────────────

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        if not texts:
            return []

        payload = {
            "model": self.embed_model,
            "input": texts,
        }

        response_data = await self._request_with_retry("/api/embed", payload)
        embeddings = response_data.get("embeddings", [])

        logger.debug(
            "ollama_embed_complete",
            model=self.embed_model,
            num_texts=len(texts),
            embedding_dim=len(embeddings[0]) if embeddings else 0,
        )

        return embeddings

    async def embed_single(self, text: str) -> list[float]:
        """Generate an embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        embeddings = await self.embed([text])
        return embeddings[0] if embeddings else []

    # ─── Health Check ─────────────────────────────────────────────────────

    async def health_check(self) -> bool:
        """Check if Ollama is reachable and responding.

        Returns:
            True if Ollama is healthy, False otherwise.
        """
        try:
            response = await self._client.get("/api/tags")
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    # ─── Internal Helpers ─────────────────────────────────────────────────

    async def _request_with_retry(
        self, endpoint: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Make an HTTP POST request with exponential backoff retry.

        Args:
            endpoint: The API endpoint path (e.g., "/api/generate").
            payload: The JSON request body.

        Returns:
            Parsed JSON response body.

        Raises:
            httpx.HTTPStatusError: If all retries are exhausted.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self._client.post(endpoint, json=payload)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    wait_seconds = 2 ** attempt
                    logger.warning(
                        "ollama_retry",
                        endpoint=endpoint,
                        attempt=attempt,
                        max_retries=self.max_retries,
                        wait_seconds=wait_seconds,
                        error=str(exc),
                    )
                    import asyncio
                    await asyncio.sleep(wait_seconds)

        logger.error("ollama_request_failed", endpoint=endpoint, error=str(last_error))
        raise last_error  # type: ignore[misc]

    @staticmethod
    def _extract_json_from_text(text: str) -> dict[str, Any]:
        """Attempt to extract JSON from text that may contain markdown fences."""
        import re
        # Try to find JSON between ```json ... ``` fences
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try to find any JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return {}
