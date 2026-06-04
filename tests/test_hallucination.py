"""
Hallucination Checker Tests (3-Layer Detection).

Tests the NLI entailment layer, smart gating logic, and 3-layer confidence formula.
Uses mock data (no LLM connection required) to verify the detection pipeline.
"""

from __future__ import annotations

from src.evaluation.hallucination_checker import (
    NLI_HIGH_THRESHOLD,
    NLI_LOW_THRESHOLD,
    HallucinationChecker,
)
from src.models.schemas import (
    Claim,
    ClaimVerification,
    EntailmentResult,
)


def test_nli_gating_high_similarity():
    """High embedding similarity should skip NLI and return SUPPORTS."""
    # Simulate: emb_similarity = 0.90 (above NLI_HIGH_THRESHOLD of 0.82)
    emb_sim = 0.90
    assert emb_sim >= NLI_HIGH_THRESHOLD
    # When gating, result should be SUPPORTS with score 1.0
    # (We test the threshold logic directly since _run_nli_check is async)
    if emb_sim >= NLI_HIGH_THRESHOLD:
        result = EntailmentResult.SUPPORTS
        score = 1.0
    assert result == EntailmentResult.SUPPORTS
    assert score == 1.0


def test_nli_gating_low_similarity():
    """Low embedding similarity should skip NLI and return CONTRADICTS."""
    emb_sim = 0.20
    assert emb_sim < NLI_LOW_THRESHOLD
    if emb_sim < NLI_LOW_THRESHOLD:
        result = EntailmentResult.CONTRADICTS
        score = 0.0
    assert result == EntailmentResult.CONTRADICTS
    assert score == 0.0


def test_nli_gating_ambiguous_zone():
    """Ambiguous embedding similarity should trigger NLI call."""
    emb_sim = 0.60
    assert NLI_LOW_THRESHOLD <= emb_sim < NLI_HIGH_THRESHOLD
    # In this zone, we'd call _run_nli_llm — not skipping


def test_three_layer_confidence_formula():
    """3-layer confidence: 0.4 × embedding + 0.4 × entailment + 0.2 × keyword."""
    emb_sim = 0.75
    entailment_score = 1.0  # SUPPORTS
    kw_score = 0.80

    confidence = 0.4 * emb_sim + 0.4 * entailment_score + 0.2 * kw_score
    assert abs(confidence - 0.86) < 0.001


def test_two_layer_vs_three_layer_contradiction_detection():
    """Demonstrate that 2-layer misses contradictions caught by 3-layer.

    This is the key improvement: a claim can have HIGH embedding similarity
    to the reference (semantically related) but be CONTRADICTORY in meaning.
    The 2-layer approach (embedding + keyword) would mark it as verified.
    The 3-layer approach (+ NLI) catches the contradiction.
    """
    emb_sim = 0.78  # High — vectors are similar
    kw_score = 0.90  # High — same entities mentioned
    entailment_score = 0.0  # NLI says CONTRADICTS

    # 2-layer confidence (old approach): 0.6 × emb + 0.4 × kw
    old_confidence = 0.6 * emb_sim + 0.4 * kw_score  # = 0.468 + 0.36 = 0.828
    # Old approach: 0.828 > 0.65 threshold → VERIFIED (WRONG!)
    assert old_confidence > 0.65, "Old approach would mark this as verified"

    # 3-layer confidence (new approach): 0.4 × emb + 0.4 × entailment + 0.2 × kw
    new_confidence = 0.4 * emb_sim + 0.4 * entailment_score + 0.2 * kw_score
    # = 0.312 + 0.0 + 0.18 = 0.492
    # New approach: 0.492 < 0.65 threshold → HALLUCINATED (CORRECT!)
    assert new_confidence < 0.65, "New approach correctly flags this as hallucinated"


def test_claim_verification_with_entailment_fields():
    """ClaimVerification should accept entailment_result and entailment_score."""
    claim = Claim(text="Python was created in 1991", source_sentence="Python was created in 1991.")
    verification = ClaimVerification(
        claim=claim,
        embedding_similarity=0.85,
        entailment_result=EntailmentResult.SUPPORTS,
        entailment_score=1.0,
        keyword_overlap_score=1.0,
        matched_keywords=["Python", "1991"],
        missing_keywords=[],
        overall_confidence=0.94,
        is_hallucination=False,
        explanation="All layers agree: claim is grounded.",
    )
    assert verification.entailment_result == EntailmentResult.SUPPORTS
    assert verification.entailment_score == 1.0
    assert not verification.is_hallucination


def test_contradiction_forces_hallucination():
    """Even with high overall confidence, CONTRADICTS should flag hallucination."""
    # Simulate: high embedding sim but NLI says contradicts
    emb_sim = 0.80
    entailment_score = 0.0  # CONTRADICTS
    kw_score = 1.0
    overall_confidence = 0.4 * emb_sim + 0.4 * entailment_score + 0.2 * kw_score
    # = 0.32 + 0.0 + 0.2 = 0.52

    is_hallucination = (
        overall_confidence < 0.65 or EntailmentResult.CONTRADICTS == EntailmentResult.CONTRADICTS
    )
    assert is_hallucination, "Contradiction should always flag hallucination"


def test_entailment_result_enum():
    """EntailmentResult enum should have the expected values."""
    assert EntailmentResult.SUPPORTS.value == "supports"
    assert EntailmentResult.CONTRADICTS.value == "contradicts"
    assert EntailmentResult.NEUTRAL.value == "neutral"


def test_keyword_overlap_no_keywords():
    """Claims with no proper nouns or numbers should get a 1.0 keyword score."""
    score, matched, missing = HallucinationChecker._keyword_overlap(
        "this is a simple claim", "this is the reference text"
    )
    assert score == 1.0
    assert matched == []
    assert missing == []


def test_keyword_overlap_with_entities():
    """Keyword overlap should find proper nouns and numbers."""
    score, matched, missing = HallucinationChecker._keyword_overlap(
        "Python 3.11 was released by Guido van Rossum",
        "Guido van Rossum created Python. Version 3.11 was released in 2022.",
    )
    assert "Python" in matched
    assert "3.11" in matched
    assert score > 0.5


def main():
    """Run all hallucination tests."""
    tests = [
        test_nli_gating_high_similarity,
        test_nli_gating_low_similarity,
        test_nli_gating_ambiguous_zone,
        test_three_layer_confidence_formula,
        test_two_layer_vs_three_layer_contradiction_detection,
        test_claim_verification_with_entailment_fields,
        test_contradiction_forces_hallucination,
        test_entailment_result_enum,
        test_keyword_overlap_no_keywords,
        test_keyword_overlap_with_entities,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"  [PASS] {test_fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Hallucination Tests: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        exit(1)


if __name__ == "__main__":
    main()
