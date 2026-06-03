"""
Token Analyzer.

Computes token usage metrics and efficiency ratios from Ollama response metadata.
Calculates verbosity by comparing output length to reference answer length.

Usage:
    analyzer = TokenAnalyzer()
    metrics = analyzer.analyze(ollama_metrics, response_text, reference_answer)
"""

from __future__ import annotations

import structlog

from src.models.schemas import TokenMetrics

logger = structlog.get_logger(__name__)


class TokenAnalyzer:
    """Computes token efficiency metrics from generation metadata."""

    def analyze(
        self,
        ollama_metrics: dict[str, float],
        response_text: str,
        reference_answer: str,
    ) -> TokenMetrics:
        """Compute token usage and efficiency metrics.

        Args:
            ollama_metrics: Dict from OllamaClient.generate() containing:
                - prompt_tokens: Input token count
                - completion_tokens: Output token count
            response_text: The generated response text.
            reference_answer: The reference (expected) answer text.

        Returns:
            TokenMetrics with usage counts, ratios, and verbosity score.
        """
        prompt_tokens = int(ollama_metrics.get("prompt_tokens", 0))
        completion_tokens = int(ollama_metrics.get("completion_tokens", 0))
        total_tokens = prompt_tokens + completion_tokens

        # Output/input ratio — how many output tokens per input token
        output_input_ratio = (
            completion_tokens / prompt_tokens if prompt_tokens > 0 else 0.0
        )

        # Verbosity score: 1.0 means response and reference are the same length.
        # > 1.0 means response is more verbose, < 1.0 means more concise.
        if not reference_answer:
            verbosity_score = 0.0
        else:
            ref_len = len(reference_answer.split())
            resp_len = len(response_text.split()) if response_text else 0
            verbosity_score = resp_len / ref_len if ref_len > 0 else 0.0

        metrics = TokenMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            output_input_ratio=round(output_input_ratio, 3),
            verbosity_score=round(verbosity_score, 3),
        )

        logger.debug(
            "tokens_analyzed",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            verbosity=metrics.verbosity_score,
        )

        return metrics
