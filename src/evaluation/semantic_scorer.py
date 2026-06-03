"""
Semantic Scorer.

Two-layer semantic quality assessment:
1. Embedding similarity — cosine similarity between response and reference embeddings.
2. LLM-as-Judge — uses Llama 3.2 to score relevance, correctness, and coherence on a 1-5 scale.

Usage:
    scorer = SemanticScorer(llm_client)
    score = await scorer.score(question, response, reference)
"""

from __future__ import annotations

import numpy as np
import structlog

from src.models.llm_client import OllamaClient
from src.models.schemas import SemanticScore

logger = structlog.get_logger(__name__)


class SemanticScorer:
    """Evaluates response quality using embedding similarity and LLM-as-Judge scoring."""

    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client

    async def score(
        self,
        question: str,
        response_text: str,
        reference_answer: str,
        enable_llm_judge: bool = True,
    ) -> SemanticScore:
        """Compute semantic quality scores for a response.

        Args:
            question: The original question/input.
            response_text: The LLM-generated response to evaluate.
            reference_answer: The ground-truth reference answer.
            enable_llm_judge: Whether to run the expensive LLM-as-Judge evaluation.

        Returns:
            SemanticScore with embedding similarity and judge scores.
        """
        # ─── Layer 1: Embedding Similarity ────────────────────────────────
        embeddings = await self.llm_client.embed([response_text, reference_answer])
        if len(embeddings) == 2:
            embedding_sim = self._cosine_similarity(embeddings[0], embeddings[1])
        else:
            embedding_sim = 0.0

        # ─── Layer 2: LLM-as-Judge ────────────────────────────────────────
        judge_relevance = 0.0
        judge_correctness = 0.0
        judge_coherence = 0.0
        judge_average = 0.0

        if enable_llm_judge:
            judge_scores = await self._run_llm_judge(
                question, response_text, reference_answer
            )
            judge_relevance = judge_scores.get("relevance", 0.0)
            judge_correctness = judge_scores.get("correctness", 0.0)
            judge_coherence = judge_scores.get("coherence", 0.0)
            judge_average = round(
                (judge_relevance + judge_correctness + judge_coherence) / 3, 2
            )

        score = SemanticScore(
            embedding_similarity=round(embedding_sim, 4),
            judge_relevance=judge_relevance,
            judge_correctness=judge_correctness,
            judge_coherence=judge_coherence,
            judge_average=judge_average,
        )

        logger.debug(
            "semantic_score_computed",
            embedding_similarity=score.embedding_similarity,
            judge_average=score.judge_average,
        )

        return score

    async def _run_llm_judge(
        self, question: str, response: str, reference: str
    ) -> dict[str, float]:
        """Use Llama 3.2 as a judge to score response quality.

        Evaluates on three dimensions (1-5 scale):
        - Relevance: Does the response address the question?
        - Correctness: Are the facts in the response accurate?
        - Coherence: Is the response well-structured and clear?

        Returns:
            Dict with 'relevance', 'correctness', 'coherence' scores.
        """
        system_prompt = (
            "You are an expert evaluator assessing the quality of an LLM-generated response.\n"
            "You will be given a question, the LLM's response, and a reference (correct) answer.\n"
            "Score the response on three dimensions using a 1-5 integer scale:\n\n"
            "1. **Relevance** (1-5): Does the response directly address the question asked?\n"
            "   1 = Completely irrelevant, 5 = Perfectly relevant\n"
            "2. **Correctness** (1-5): Are the facts in the response accurate compared to the reference?\n"
            "   1 = Entirely wrong, 5 = Fully correct\n"
            "3. **Coherence** (1-5): Is the response well-structured, clear, and easy to understand?\n"
            "   1 = Incoherent, 5 = Perfectly clear\n\n"
            "Output your evaluation as JSON:\n"
            '{"relevance": <int>, "correctness": <int>, "coherence": <int>, '
            '"explanation": "<brief rationale>"}'
        )

        prompt = (
            f"Question:\n{question}\n\n"
            f"LLM Response:\n{response}\n\n"
            f"Reference Answer:\n{reference}\n\n"
            f"Evaluation:"
        )

        try:
            result = await self.llm_client.generate_json(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
            )

            return {
                "relevance": float(result.get("relevance", 0)),
                "correctness": float(result.get("correctness", 0)),
                "coherence": float(result.get("coherence", 0)),
            }
        except Exception as exc:
            logger.warning("llm_judge_failed", error=str(exc))
            return {"relevance": 0.0, "correctness": 0.0, "coherence": 0.0}

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
