"""
Hallucination Checker.

Adapted from the Agentic RAG project's HallucinationDetector.
Extracts atomic factual claims from responses and verifies each against
reference context using:
- Embedding similarity (cosine between claim and reference)
- Keyword/entity overlap (proper nouns and numbers)

Usage:
    checker = HallucinationChecker(llm_client)
    report = await checker.check(response_text, reference_context)
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import structlog

from src.config import settings
from src.models.llm_client import OllamaClient
from src.models.schemas import Claim, ClaimVerification, HallucinationReport

logger = structlog.get_logger(__name__)


class HallucinationChecker:
    """Detects hallucinations by decomposing responses into claims and verifying against context."""

    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client

    async def check(
        self,
        response_text: str,
        reference_context: str,
    ) -> HallucinationReport:
        """Run full hallucination analysis on a response.

        Args:
            response_text: The LLM-generated response to check.
            reference_context: The ground-truth context to verify claims against.

        Returns:
            HallucinationReport with per-claim verification details and aggregate rates.
        """
        if not response_text.strip() or not reference_context.strip():
            return HallucinationReport(
                total_claims=0,
                verified_claims=0,
                hallucinated_claims=0,
                hallucination_rate=0.0,
                overall_confidence=1.0,
            )

        # 1. Extract atomic claims from the response
        claims = await self._extract_claims(response_text)
        if not claims:
            return HallucinationReport(
                total_claims=0,
                verified_claims=0,
                hallucinated_claims=0,
                hallucination_rate=0.0,
                overall_confidence=1.0,
            )

        # 2. Verify each claim against reference context
        verifications = []
        for claim in claims:
            verification = await self._verify_claim(claim, reference_context)
            verifications.append(verification)

        # 3. Aggregate results
        total = len(verifications)
        hallucinated = sum(1 for v in verifications if v.is_hallucination)
        verified = total - hallucinated
        hallucination_rate = hallucinated / total if total > 0 else 0.0
        overall_confidence = (
            sum(v.overall_confidence for v in verifications) / total if total > 0 else 1.0
        )

        report = HallucinationReport(
            total_claims=total,
            verified_claims=verified,
            hallucinated_claims=hallucinated,
            hallucination_rate=round(hallucination_rate, 3),
            claim_verifications=verifications,
            overall_confidence=round(overall_confidence, 3),
        )

        logger.info(
            "hallucination_check_complete",
            total_claims=total,
            hallucinated=hallucinated,
            hallucination_rate=report.hallucination_rate,
        )

        return report

    async def _extract_claims(self, response_text: str) -> list[Claim]:
        """Extract atomic factual claims from the response using Llama 3.2.

        Args:
            response_text: Text to decompose into claims.

        Returns:
            List of Claim objects.
        """
        system_prompt = (
            "You are a linguistic analysis agent. Break down the following text "
            "into individual, self-contained atomic factual claims.\n"
            "An atomic claim is a single statement containing exactly one fact "
            "that can be verified independently.\n"
            "Do not include opinions, meta-commentary, or questions.\n"
            "Output JSON in this format:\n"
            '{"claims": [{"text": "Claim text", "source_sentence": "Original sentence"}]}'
        )

        prompt = f"Decompose this into atomic factual claims:\n\n{response_text}"

        try:
            result = await self.llm_client.generate_json(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
            )

            claims_data = result.get("claims", [])
            claims = []
            for item in claims_data:
                text = item.get("text", "").strip()
                source = item.get("source_sentence", "").strip()
                if text and source:
                    claims.append(Claim(text=text, source_sentence=source))

            logger.debug("claims_extracted", count=len(claims))
            return claims
        except Exception as exc:
            logger.warning("claim_extraction_failed", error=str(exc))
            return []

    async def _verify_claim(
        self, claim: Claim, reference_context: str
    ) -> ClaimVerification:
        """Verify a single claim against the reference context.

        Uses two verification layers:
        1. Embedding similarity (cosine between claim and reference)
        2. Keyword/entity overlap (proper nouns and numbers)

        Args:
            claim: The atomic claim to verify.
            reference_context: The reference context text.

        Returns:
            ClaimVerification with scores and classification.
        """
        # ─── Layer 1: Embedding Similarity ────────────────────────────────
        embeddings = await self.llm_client.embed([claim.text, reference_context])
        if len(embeddings) == 2:
            emb_similarity = self._cosine_similarity(embeddings[0], embeddings[1])
        else:
            emb_similarity = 0.0

        # ─── Layer 2: Keyword/Entity Overlap ──────────────────────────────
        kw_score, matched_kws, missing_kws = self._keyword_overlap(
            claim.text, reference_context
        )

        # ─── Combined Confidence ──────────────────────────────────────────
        # 60% embedding similarity + 40% keyword overlap
        overall_confidence = (0.6 * emb_similarity) + (0.4 * kw_score)

        # Classification threshold
        threshold = settings.hallucination_sim_threshold
        is_hallucination = overall_confidence < threshold

        explanation = (
            f"Embedding similarity: {emb_similarity:.3f}, "
            f"Keyword overlap: {kw_score:.3f} ({len(matched_kws)} matched, "
            f"{len(missing_kws)} missing). "
            f"{'HALLUCINATED' if is_hallucination else 'VERIFIED'} "
            f"(threshold: {threshold})"
        )

        return ClaimVerification(
            claim=claim,
            embedding_similarity=round(emb_similarity, 4),
            keyword_overlap_score=round(kw_score, 3),
            matched_keywords=matched_kws,
            missing_keywords=missing_kws,
            overall_confidence=round(overall_confidence, 3),
            is_hallucination=is_hallucination,
            explanation=explanation,
        )

    @staticmethod
    def _keyword_overlap(
        claim: str, reference: str
    ) -> tuple[float, list[str], list[str]]:
        """Check overlap of numbers and proper nouns between claim and reference.

        Args:
            claim: The claim text.
            reference: The reference context text.

        Returns:
            Tuple of (score, matched_keywords, missing_keywords).
        """
        # Extract numbers (integers, floats, percentages)
        numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", claim)

        # Extract capitalized proper nouns (length >= 2)
        proper_nouns = re.findall(r"\b[A-Z][a-zA-Z0-9-]+\b", claim)

        # Filter stop words
        stop_proper = {
            "The", "A", "An", "In", "On", "Of", "And", "To",
            "For", "With", "By", "At", "From", "It", "Is", "Was",
            "This", "That", "These", "Those",
        }
        proper_nouns = [w for w in proper_nouns if w not in stop_proper]

        keywords = list(set(numbers + proper_nouns))
        if not keywords:
            return 1.0, [], []

        matched = []
        missing = []
        ref_lower = reference.lower()

        for kw in keywords:
            if kw.lower() in ref_lower:
                matched.append(kw)
            else:
                missing.append(kw)

        score = len(matched) / len(keywords) if keywords else 1.0
        return score, matched, missing

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
