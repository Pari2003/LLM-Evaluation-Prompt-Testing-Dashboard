"""
Evaluation Module Tests.

Tests for the evaluation analyzers (latency, token, composite scorer)
using mock data that doesn't require an LLM connection.
"""

from __future__ import annotations

from src.evaluation.composite_scorer import CompositeScorer
from src.evaluation.latency_analyzer import LatencyAnalyzer
from src.evaluation.token_analyzer import TokenAnalyzer
from src.models.schemas import (
    ConsistencyReport,
    HallucinationReport,
    LatencyMetrics,
    SemanticScore,
    TokenMetrics,
)


def test_latency_analyzer():
    """LatencyAnalyzer should extract metrics from Ollama response data."""
    analyzer = LatencyAnalyzer()
    metrics = analyzer.analyze({
        "total_ms": 1234.5,
        "tokens_per_second": 42.7,
    })
    assert isinstance(metrics, LatencyMetrics)
    assert metrics.total_ms == 1234.5
    assert metrics.tokens_per_second == 42.7


def test_latency_analyzer_missing_fields():
    """LatencyAnalyzer should handle missing fields gracefully."""
    analyzer = LatencyAnalyzer()
    metrics = analyzer.analyze({})
    assert metrics.total_ms == 0.0
    assert metrics.tokens_per_second == 0.0


def test_token_analyzer():
    """TokenAnalyzer should compute ratios and verbosity correctly."""
    analyzer = TokenAnalyzer()
    metrics = analyzer.analyze(
        ollama_metrics={"prompt_tokens": 100, "completion_tokens": 50},
        response_text="This is a short response with ten words in it.",
        reference_answer="This is the reference answer with nine words here.",
    )
    assert isinstance(metrics, TokenMetrics)
    assert metrics.prompt_tokens == 100
    assert metrics.completion_tokens == 50
    assert metrics.total_tokens == 150
    assert metrics.output_input_ratio == 0.5
    # Verbosity: response has 10 words, reference has 9 words → ~1.111
    assert 1.0 < metrics.verbosity_score < 1.2


def test_token_analyzer_empty_reference():
    """TokenAnalyzer should handle empty reference gracefully."""
    analyzer = TokenAnalyzer()
    metrics = analyzer.analyze(
        ollama_metrics={"prompt_tokens": 50, "completion_tokens": 30},
        response_text="Some response",
        reference_answer="",
    )
    assert metrics.verbosity_score == 0.0  # 0 ref words → 0 / 1 = 0


def test_composite_scorer_all_perfect():
    """CompositeScorer should return ~1.0 for perfect metrics."""
    scorer = CompositeScorer()
    score = scorer.compute(
        latency=LatencyMetrics(total_ms=100.0, tokens_per_second=50.0),
        tokens=TokenMetrics(verbosity_score=1.0),
        semantic=SemanticScore(judge_average=5.0),
        hallucination=HallucinationReport(hallucination_rate=0.0),
        consistency=ConsistencyReport(agreement_rate=1.0),
    )
    assert 0.95 <= score <= 1.0


def test_composite_scorer_all_worst():
    """CompositeScorer should return low score for poor metrics."""
    scorer = CompositeScorer()
    score = scorer.compute(
        latency=LatencyMetrics(total_ms=50000.0),  # Way over max
        tokens=TokenMetrics(verbosity_score=5.0),  # Very verbose
        semantic=SemanticScore(judge_average=1.0),  # Poor quality
        hallucination=HallucinationReport(hallucination_rate=1.0),  # All hallucinated
        consistency=ConsistencyReport(agreement_rate=0.0),  # No agreement
    )
    assert score < 0.3


def test_composite_scorer_no_optionals():
    """CompositeScorer should handle None hallucination and consistency."""
    scorer = CompositeScorer()
    score = scorer.compute(
        latency=LatencyMetrics(total_ms=500.0, tokens_per_second=30.0),
        tokens=TokenMetrics(verbosity_score=1.2),
        semantic=SemanticScore(embedding_similarity=0.85),
    )
    # Should still produce a valid score without hallucination/consistency
    assert 0.0 <= score <= 1.0


def test_composite_scorer_embedding_fallback():
    """CompositeScorer should use embedding_similarity when judge_average is 0."""
    scorer = CompositeScorer()
    score = scorer.compute(
        latency=LatencyMetrics(total_ms=200.0),
        tokens=TokenMetrics(verbosity_score=1.0),
        semantic=SemanticScore(
            embedding_similarity=0.9,
            judge_average=0.0,  # Judge not run
        ),
    )
    # semantic_norm should use embedding_similarity (0.9) instead of judge
    assert score > 0.5


def main():
    """Run all evaluation tests."""
    tests = [
        test_latency_analyzer,
        test_latency_analyzer_missing_fields,
        test_token_analyzer,
        test_token_analyzer_empty_reference,
        test_composite_scorer_all_perfect,
        test_composite_scorer_all_worst,
        test_composite_scorer_no_optionals,
        test_composite_scorer_embedding_fallback,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"  ✓ {test_fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Evaluation Tests: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        exit(1)


if __name__ == "__main__":
    main()
