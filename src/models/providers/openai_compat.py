"""
OpenAI-Compatible API Provider.

Connects to any LLM endpoint that implements the OpenAI REST API format
(OpenAI, Groq, Together AI, vLLM, etc.).

Uses httpx (already in requirements) to avoid adding the 'openai' python package.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

import httpx
import structlog

from src.config import settings
from src.models.providers.base import LLMProvider

logger = structlog.get_logger(__name__)


class OpenAICompatProvider(LLMProvider):
    """Async provider for OpenAI-compatible APIs (OpenAI, Groq, Together)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        text_model: Optional[str] = None,
        embed_model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self.api_key = api_key or settings.openai_api_key
        self.text_model = text_model or settings.text_model
        self.embed_model = embed_model or settings.embed_model
        self.timeout = timeout or settings.llm_timeout
        self.max_retries = max_retries or settings.llm_max_retries

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(self.timeout, connect=10.0),
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate text and capture latency + token metrics."""
        temp = temperature if temperature is not None else settings.llm_temperature
        tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.text_model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
        }

        start_time = time.perf_counter()
        response_data = await self._request_with_retry("/v1/chat/completions", payload)
        total_ms = (time.perf_counter() - start_time) * 1000

        choices = response_data.get("choices", [])
        response_text = ""
        if choices:
            response_text = choices[0].get("message", {}).get("content", "").strip()

        usage = response_data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Approximate tokens/sec (wall clock)
        if total_ms > 0 and completion_tokens > 0:
            tokens_per_second = completion_tokens / (total_ms / 1000)
        else:
            tokens_per_second = 0.0

        metrics = {
            "total_ms": round(total_ms, 1),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_per_second": round(tokens_per_second, 2),
        }

        logger.debug(
            "openai_generate_complete",
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
        """Generate a structured JSON response."""
        temp = temperature if temperature is not None else settings.llm_temperature

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.text_model,
            "messages": messages,
            "temperature": temp,
            "response_format": {"type": "json_object"},
        }

        response_data = await self._request_with_retry("/v1/chat/completions", payload)

        choices = response_data.get("choices", [])
        response_text = ""
        if choices:
            response_text = choices[0].get("message", {}).get("content", "").strip()

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            logger.warning("json_parse_fallback", raw_response=response_text[:200])
            return self._extract_json_from_text(response_text)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []

        payload = {
            "model": self.embed_model,
            "input": texts,
        }

        response_data = await self._request_with_retry("/v1/embeddings", payload)

        data = response_data.get("data", [])
        # OpenAI returns embeddings in order if sorting is needed, but typically they map 1:1
        data = sorted(data, key=lambda x: x.get("index", 0))
        embeddings = [item.get("embedding", []) for item in data]

        logger.debug(
            "openai_embed_complete",
            model=self.embed_model,
            num_texts=len(texts),
            embedding_dim=len(embeddings[0]) if embeddings else 0,
        )

        return embeddings

    async def embed_single(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""
        embeddings = await self.embed([text])
        return embeddings[0] if embeddings else []

    async def health_check(self) -> bool:
        """Check if the API is reachable."""
        try:
            # We check the models endpoint as a proxy for health
            response = await self._client.get("/v1/models")
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def _request_with_retry(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Make an HTTP POST request with exponential backoff retry."""
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self._client.post(endpoint, json=payload)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException) as exc:
                last_error = exc

                # Check for rate limits (429) specifically
                is_rate_limit = (
                    isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429
                )

                if attempt < self.max_retries:
                    # Exponential backoff (longer for rate limits)
                    wait_seconds = (2**attempt) * (2 if is_rate_limit else 1)

                    logger.warning(
                        "openai_retry",
                        endpoint=endpoint,
                        attempt=attempt,
                        max_retries=self.max_retries,
                        wait_seconds=wait_seconds,
                        error=str(exc),
                    )
                    import asyncio

                    await asyncio.sleep(wait_seconds)

        logger.error("openai_request_failed", endpoint=endpoint, error=str(last_error))
        raise last_error  # type: ignore[misc]

    @staticmethod
    def _extract_json_from_text(text: str) -> dict[str, Any]:
        """Attempt to extract JSON from text that may contain markdown fences."""
        import re

        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return {}
