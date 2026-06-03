"""
Latency Analyzer.

Processes raw timing data from Ollama responses to compute
structured latency metrics: total generation time and throughput.

Usage:
    analyzer = LatencyAnalyzer()
    metrics = analyzer.analyze(ollama_metrics)
"""

from __future__ import annotations

import structlog

from src.models.schemas import LatencyMetrics

logger = structlog.get_logger(__name__)


class LatencyAnalyzer:
    """Extracts and normalizes latency metrics from Ollama response metadata."""

    def analyze(self, ollama_metrics: dict[str, float]) -> LatencyMetrics:
        """Compute latency metrics from raw Ollama timing data.

        Args:
            ollama_metrics: Dict from OllamaClient.generate() containing:
                - total_ms: Wall-clock generation time
                - tokens_per_second: Generation throughput from Ollama

        Returns:
            LatencyMetrics with normalized timing values.
        """
        total_ms = ollama_metrics.get("total_ms", 0.0)
        tokens_per_second = ollama_metrics.get("tokens_per_second", 0.0)

        metrics = LatencyMetrics(
            total_ms=round(total_ms, 1),
            tokens_per_second=round(tokens_per_second, 2),
        )

        logger.debug(
            "latency_analyzed",
            total_ms=metrics.total_ms,
            tokens_per_second=metrics.tokens_per_second,
        )

        return metrics
