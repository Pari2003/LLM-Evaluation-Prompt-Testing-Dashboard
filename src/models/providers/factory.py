from __future__ import annotations

from typing import Optional

from src.config import settings
from src.models.providers.base import LLMProvider
from src.models.providers.ollama import OllamaProvider
from src.models.providers.openai_compat import OpenAICompatProvider


def create_provider(provider_type: Optional[str] = None, **kwargs) -> LLMProvider:
    """Factory to create the appropriate LLMProvider based on configuration.

    Args:
        provider_type: Provider identifier ('ollama' or 'openai'). Defaults to settings.llm_provider.
        **kwargs: Additional overrides for the provider constructor.

    Returns:
        Configured LLMProvider instance.
    """
    ptype = (provider_type or settings.llm_provider).lower()

    if ptype == "ollama":
        return OllamaProvider(**kwargs)
    elif ptype in ("openai", "groq", "together"):
        # They all use the OpenAI-compatible API format
        return OpenAICompatProvider(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider type: {ptype}")
