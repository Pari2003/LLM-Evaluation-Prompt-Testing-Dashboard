"""
Composite Scorer.

Combines all evaluation dimensions (latency, token efficiency, semantic quality,
hallucination, consistency) into a single weighted composite score for ranking
prompt variants.

The composite score normalizes each dimension to a 0-1 scale and applies
configurable weights from settings.

Usage:
    scorer = CompositeScorer()
    score = scorer.compute(latency, tokens, semantic, hallucination, consistency)
"""

from __future__ import annotations

import structlog

from src.config import settings
from src.models.schemas import (
    ConsistencyReport,
    HallucinationReport,
    LatencyMetrics,
    SemanticScore,
    TokenMetrics,
)

logger = structlog.get_logger(__name__)


class CompositeScorer:
    """Computes a weighted composite quality score from multiple evaluation dimensions.

    Normalization Strategy:
    - Latency: Lower is better → normalized as 1.0 - (total_ms / max_acceptable_ms)
    - Token efficiency: Verbosity closer to 1.0 is better → 1.0 - abs(1.0 - verbosity)
    - Semantic quality: Higher is better → judge_average / 5.0 (already 0-1 after normalization)
    - Hallucination: Lower rate is better → 1.0 - hallucination_rate
    - Consistency: Higher agreement is better → agreement_rate (already 0-1)
    """

    # Maximum acceptable latency (ms) for normalization
    MAX_LATENCY_MS = 30000.0  # 30 seconds

    def compute(
        self,
        latency: LatencyMetrics,
        tokens: TokenMetrics,
        semantic: SemanticScore,
        hallucination: HallucinationReport | None = None,
        consistency: ConsistencyReport | None = None,
    ) -> float:
        """Compute the composite score from all evaluation dimensions.

        Args:
            latency: Timing metrics.
            tokens: Token usage metrics.
            semantic: Semantic quality scores.
            hallucination: Optional hallucination analysis report.
            consistency: Optional cross-run consistency report.

        Returns:
            Weighted composite score in the range [0.0, 1.0].
        """
        weights = settings.composite_weights

        # ─── Normalize each dimension to [0, 1] ──────────────────────────

        # Latency (lower is better)
        latency_norm = max(0.0, 1.0 - (latency.total_ms / self.MAX_LATENCY_MS))

        # Token efficiency (verbosity close to 1.0 is ideal)
        verbosity_deviation = abs(1.0 - tokens.verbosity_score)
        token_norm = max(0.0, 1.0 - min(verbosity_deviation, 1.0))

        # Semantic quality (judge average is 0-5, normalize to 0-1)
        if semantic.judge_average > 0:
            semantic_norm = semantic.judge_average / 5.0
        else:
            # Fall back to embedding similarity if judge wasn't run
            semantic_norm = semantic.embedding_similarity

        # Hallucination (lower rate is better)
        if hallucination:
            hallucination_norm = 1.0 - hallucination.hallucination_rate
        else:
            hallucination_norm = 1.0  # Assume no hallucinations if not checked

        # Consistency (higher agreement is better)
        if consistency:
            consistency_norm = consistency.agreement_rate
        else:
            consistency_norm = 1.0  # Assume consistent if not checked

        # ─── Weighted Average ─────────────────────────────────────────────
        composite = (
            weights["latency"] * latency_norm
            + weights["token_efficiency"] * token_norm
            + weights["semantic_quality"] * semantic_norm
            + weights["hallucination"] * hallucination_norm
            + weights["consistency"] * consistency_norm
        )

        composite = round(max(0.0, min(1.0, composite)), 4)

        logger.debug(
            "composite_score_computed",
            latency_norm=round(latency_norm, 3),
            token_norm=round(token_norm, 3),
            semantic_norm=round(semantic_norm, 3),
            hallucination_norm=round(hallucination_norm, 3),
            consistency_norm=round(consistency_norm, 3),
            composite=composite,
        )

        return composite
