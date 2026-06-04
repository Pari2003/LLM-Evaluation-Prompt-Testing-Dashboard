"""
Consistency Checker.

Measures cross-run consistency for repeated executions of the same prompt+input.
Computes pairwise embedding similarity, semantic drift from the centroid,
and agreement rate across all run pairs.

Usage:
    checker = ConsistencyChecker(llm_client)
    report = await checker.check(responses)
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import structlog

from src.config import settings
from src.models.providers.base import LLMProvider
from src.models.schemas import ConsistencyReport

logger = structlog.get_logger(__name__)


class ConsistencyChecker:
    """Evaluates cross-run consistency of LLM responses.

    Checks if repeated executions for the same prompt produce
    semantically similar results, detecting flaky or unstable prompts.
    """

    def __init__(self, llm_client: LLMProvider):
        self.llm_client = llm_client

    async def check(self, responses: list[str]) -> ConsistencyReport:
        """Compute consistency metrics across multiple responses.

        Args:
            responses: List of response texts from repeated runs
                       of the same (prompt, test_case) pair.

        Returns:
            ConsistencyReport with pairwise similarity and drift metrics.
        """
        num_runs = len(responses)

        if num_runs < 2:
            return ConsistencyReport(
                num_runs=num_runs,
                mean_pairwise_similarity=1.0,
                min_pairwise_similarity=1.0,
                semantic_drift=0.0,
                agreement_rate=1.0,
            )

        # Embed all responses
        embeddings = await self.llm_client.embed(responses)
        if len(embeddings) != num_runs:
            logger.warning(
                "consistency_embedding_mismatch",
                expected=num_runs,
                got=len(embeddings),
            )
            return ConsistencyReport(num_runs=num_runs)

        emb_matrix = np.array(embeddings)

        # ─── Pairwise Cosine Similarity ───────────────────────────────────
        pairwise_sims = []
        for i, j in combinations(range(num_runs), 2):
            sim = self._cosine_similarity(emb_matrix[i], emb_matrix[j])
            pairwise_sims.append(sim)

        mean_sim = float(np.mean(pairwise_sims))
        min_sim = float(np.min(pairwise_sims))

        # ─── Semantic Drift (max deviation from centroid) ─────────────────
        centroid = np.mean(emb_matrix, axis=0)
        distances_to_centroid = [1.0 - self._cosine_similarity(emb, centroid) for emb in emb_matrix]
        semantic_drift = float(np.max(distances_to_centroid))

        # ─── Agreement Rate ───────────────────────────────────────────────
        threshold = settings.consistency_sim_threshold
        agreements = sum(1 for s in pairwise_sims if s >= threshold)
        total_pairs = len(pairwise_sims)
        agreement_rate = agreements / total_pairs if total_pairs > 0 else 1.0

        report = ConsistencyReport(
            num_runs=num_runs,
            mean_pairwise_similarity=round(mean_sim, 4),
            min_pairwise_similarity=round(min_sim, 4),
            semantic_drift=round(semantic_drift, 4),
            agreement_rate=round(agreement_rate, 3),
        )

        logger.debug(
            "consistency_checked",
            num_runs=num_runs,
            mean_similarity=report.mean_pairwise_similarity,
            agreement_rate=report.agreement_rate,
        )

        return report

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two numpy vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
