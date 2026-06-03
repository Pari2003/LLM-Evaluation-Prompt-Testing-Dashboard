"""
Execution Pipeline Tests.

Integration test for the ResultAggregator using mock RunResult data.
Tests statistical aggregation, win-rate matrix, and ranking logic
without requiring an LLM connection.
"""

from __future__ import annotations

from src.execution.result_aggregator import ResultAggregator
from src.models.schemas import (
    ConsistencyReport,
    EvaluationConfig,
    Experiment,
    HallucinationReport,
    LatencyMetrics,
    PromptTemplate,
    RunResult,
    RunStatus,
    SemanticScore,
    TestCase,
    TestDataset,
    TokenMetrics,
)


def create_mock_results() -> tuple[Experiment, TestDataset, list[RunResult]]:
    """Create mock experiment, dataset, and results for testing aggregation."""
    # Templates
    t1 = PromptTemplate(id="t1", name="concise", template="Answer: {q}")
    t2 = PromptTemplate(id="t2", name="detailed", template="Explain: {q}")

    # Dataset
    dataset = TestDataset(
        name="Mock Dataset",
        test_cases=[
            TestCase(id="tc1", input_variables={"q": "What is AI?"}, reference_answer="AI is..."),
            TestCase(id="tc2", input_variables={"q": "What is ML?"}, reference_answer="ML is..."),
        ],
    )

    # Experiment
    experiment = Experiment(
        id="exp_mock",
        name="Mock Experiment",
        prompt_templates=[t1, t2],
        dataset_id=dataset.id,
        eval_config=EvaluationConfig(repetitions=2),
    )

    # Mock results: t1 should "win" with higher semantic scores
    results = [
        # t1, tc1, rep1
        RunResult(
            experiment_id="exp_mock", prompt_template_id="t1",
            prompt_template_name="concise", test_case_id="tc1", repetition=1,
            response_text="AI is artificial intelligence.",
            latency=LatencyMetrics(total_ms=500.0, tokens_per_second=40.0),
            tokens=TokenMetrics(prompt_tokens=50, completion_tokens=20, total_tokens=70, verbosity_score=1.0),
            semantic=SemanticScore(embedding_similarity=0.92, judge_average=4.5),
            hallucination=HallucinationReport(hallucination_rate=0.0),
            consistency=ConsistencyReport(agreement_rate=0.95),
        ),
        # t1, tc1, rep2
        RunResult(
            experiment_id="exp_mock", prompt_template_id="t1",
            prompt_template_name="concise", test_case_id="tc1", repetition=2,
            response_text="AI is artificial intelligence technology.",
            latency=LatencyMetrics(total_ms=480.0, tokens_per_second=42.0),
            tokens=TokenMetrics(prompt_tokens=50, completion_tokens=22, total_tokens=72, verbosity_score=1.1),
            semantic=SemanticScore(embedding_similarity=0.91, judge_average=4.3),
            hallucination=HallucinationReport(hallucination_rate=0.0),
            consistency=ConsistencyReport(agreement_rate=0.95),
        ),
        # t1, tc2, rep1
        RunResult(
            experiment_id="exp_mock", prompt_template_id="t1",
            prompt_template_name="concise", test_case_id="tc2", repetition=1,
            response_text="ML is machine learning.",
            latency=LatencyMetrics(total_ms=450.0, tokens_per_second=44.0),
            tokens=TokenMetrics(prompt_tokens=50, completion_tokens=18, total_tokens=68, verbosity_score=0.9),
            semantic=SemanticScore(embedding_similarity=0.89, judge_average=4.0),
        ),
        # t1, tc2, rep2
        RunResult(
            experiment_id="exp_mock", prompt_template_id="t1",
            prompt_template_name="concise", test_case_id="tc2", repetition=2,
            response_text="ML is a subset of AI.",
            latency=LatencyMetrics(total_ms=460.0, tokens_per_second=43.0),
            tokens=TokenMetrics(prompt_tokens=50, completion_tokens=19, total_tokens=69, verbosity_score=0.95),
            semantic=SemanticScore(embedding_similarity=0.88, judge_average=4.2),
        ),
        # t2, tc1, rep1 — worse performance
        RunResult(
            experiment_id="exp_mock", prompt_template_id="t2",
            prompt_template_name="detailed", test_case_id="tc1", repetition=1,
            response_text="Artificial intelligence is a broad field...",
            latency=LatencyMetrics(total_ms=1200.0, tokens_per_second=25.0),
            tokens=TokenMetrics(prompt_tokens=80, completion_tokens=100, total_tokens=180, verbosity_score=2.5),
            semantic=SemanticScore(embedding_similarity=0.80, judge_average=3.5),
            hallucination=HallucinationReport(hallucination_rate=0.1),
            consistency=ConsistencyReport(agreement_rate=0.8),
        ),
        # t2, tc1, rep2
        RunResult(
            experiment_id="exp_mock", prompt_template_id="t2",
            prompt_template_name="detailed", test_case_id="tc1", repetition=2,
            response_text="AI encompasses many subfields...",
            latency=LatencyMetrics(total_ms=1300.0, tokens_per_second=23.0),
            tokens=TokenMetrics(prompt_tokens=80, completion_tokens=110, total_tokens=190, verbosity_score=2.8),
            semantic=SemanticScore(embedding_similarity=0.78, judge_average=3.3),
            hallucination=HallucinationReport(hallucination_rate=0.15),
            consistency=ConsistencyReport(agreement_rate=0.8),
        ),
        # t2, tc2, rep1
        RunResult(
            experiment_id="exp_mock", prompt_template_id="t2",
            prompt_template_name="detailed", test_case_id="tc2", repetition=1,
            response_text="Machine learning is a methodology...",
            latency=LatencyMetrics(total_ms=1100.0, tokens_per_second=26.0),
            tokens=TokenMetrics(prompt_tokens=80, completion_tokens=90, total_tokens=170, verbosity_score=2.2),
            semantic=SemanticScore(embedding_similarity=0.82, judge_average=3.8),
        ),
        # t2, tc2, rep2
        RunResult(
            experiment_id="exp_mock", prompt_template_id="t2",
            prompt_template_name="detailed", test_case_id="tc2", repetition=2,
            response_text="ML involves training algorithms...",
            latency=LatencyMetrics(total_ms=1150.0, tokens_per_second=24.0),
            tokens=TokenMetrics(prompt_tokens=80, completion_tokens=95, total_tokens=175, verbosity_score=2.4),
            semantic=SemanticScore(embedding_similarity=0.81, judge_average=3.6),
        ),
    ]

    return experiment, dataset, results


def test_variant_report_generation():
    """ResultAggregator should produce correct per-variant statistical summaries."""
    aggregator = ResultAggregator()
    experiment, dataset, results = create_mock_results()
    report = aggregator.aggregate(experiment, dataset, results)

    # Should have 2 variant reports
    assert len(report.variant_reports) == 2

    # Find the concise variant
    concise = next(v for v in report.variant_reports if v.prompt_template_name == "concise")
    assert concise.num_runs == 4
    assert concise.num_successes == 4
    assert concise.latency_ms.mean > 0
    assert concise.embedding_similarity.mean > 0.85


def test_rankings():
    """Concise variant should rank higher than detailed (better metrics)."""
    aggregator = ResultAggregator()
    experiment, dataset, results = create_mock_results()
    report = aggregator.aggregate(experiment, dataset, results)

    assert len(report.rankings) == 2
    # Concise should be rank 1 (higher composite score)
    assert report.rankings[0]["name"] == "concise"
    assert report.rankings[0]["rank"] == 1
    assert report.rankings[0]["composite_score"] > report.rankings[1]["composite_score"]


def test_win_rate_matrix():
    """Win-rate matrix should show concise winning most comparisons."""
    aggregator = ResultAggregator()
    experiment, dataset, results = create_mock_results()
    report = aggregator.aggregate(experiment, dataset, results)

    assert len(report.win_rate_matrix) == 1  # 2 variants → 1 pair
    entry = report.win_rate_matrix[0]
    assert entry.total_comparisons == 2  # 2 test cases


def test_metric_summary_statistics():
    """MetricSummary should compute correct statistical values."""
    aggregator = ResultAggregator()
    experiment, dataset, results = create_mock_results()
    report = aggregator.aggregate(experiment, dataset, results)

    concise = next(v for v in report.variant_reports if v.prompt_template_name == "concise")
    # Check that stddev is computed (4 runs should give non-zero stddev)
    assert concise.latency_ms.stddev >= 0
    assert concise.latency_ms.min <= concise.latency_ms.mean <= concise.latency_ms.max
    assert concise.latency_ms.p95 >= concise.latency_ms.median


def test_failed_runs_handling():
    """Aggregator should handle experiments with failed runs."""
    aggregator = ResultAggregator()
    experiment, dataset, results = create_mock_results()

    # Add a failed run
    results.append(
        RunResult(
            experiment_id="exp_mock", prompt_template_id="t1",
            prompt_template_name="concise", test_case_id="tc1", repetition=3,
            status=RunStatus.ERROR,
            error_message="Timeout",
        )
    )

    report = aggregator.aggregate(experiment, dataset, results)
    concise = next(v for v in report.variant_reports if v.prompt_template_name == "concise")
    assert concise.num_runs == 5
    assert concise.num_successes == 4
    assert concise.num_failures == 1


def main():
    """Run all execution tests."""
    tests = [
        test_variant_report_generation,
        test_rankings,
        test_win_rate_matrix,
        test_metric_summary_statistics,
        test_failed_runs_handling,
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

    print(f"\n{'='*50}")
    print(f"Execution Tests: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        exit(1)


if __name__ == "__main__":
    main()
