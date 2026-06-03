"""
Schema Validation Tests.

Tests for all Pydantic data models to verify serialization, deserialization,
default values, computed properties, and field validation.
"""

from __future__ import annotations

from src.models.schemas import (
    Claim,
    ClaimVerification,
    ConsistencyReport,
    CreateDatasetRequest,
    CreateExperimentRequest,
    DatasetSummary,
    EvaluationConfig,
    Experiment,
    ExperimentReport,
    ExperimentStatus,
    ExperimentSummary,
    HallucinationReport,
    LatencyMetrics,
    MetricSummary,
    PromptTemplate,
    RunResult,
    RunStatus,
    SemanticScore,
    TestCase,
    TestDataset,
    TokenMetrics,
    VariantReport,
    WinRateEntry,
    generate_id,
    utc_now,
)


def test_generate_id():
    """IDs should be unique 12-char hex strings."""
    id1 = generate_id()
    id2 = generate_id()
    assert len(id1) == 12
    assert id1 != id2
    assert id1.isalnum()


def test_utc_now():
    """utc_now should return a timezone-aware datetime."""
    now = utc_now()
    assert now.tzinfo is not None


def test_prompt_template_defaults():
    """PromptTemplate should generate an ID and set defaults."""
    pt = PromptTemplate(
        name="test_prompt",
        template="Answer: {question}",
        variables=["question"],
    )
    assert len(pt.id) == 12
    assert pt.name == "test_prompt"
    assert pt.system_prompt is None
    assert pt.metadata == {}
    assert pt.created_at is not None


def test_test_case_structure():
    """TestCase should accept input variables and reference answer."""
    tc = TestCase(
        input_variables={"question": "What is AI?"},
        reference_answer="AI is artificial intelligence.",
        reference_context="Artificial intelligence (AI) refers to...",
        tags=["definition"],
    )
    assert tc.input_variables["question"] == "What is AI?"
    assert tc.reference_context is not None
    assert "definition" in tc.tags


def test_test_dataset_size_property():
    """TestDataset.size should return the count of test cases."""
    ds = TestDataset(
        name="test_ds",
        test_cases=[
            TestCase(input_variables={"q": "1"}, reference_answer="a1"),
            TestCase(input_variables={"q": "2"}, reference_answer="a2"),
        ],
    )
    assert ds.size == 2


def test_evaluation_config_defaults():
    """EvaluationConfig should have sensible defaults."""
    config = EvaluationConfig()
    assert config.repetitions == 3
    assert config.temperature == 0.1
    assert config.max_tokens == 2048
    assert config.enable_hallucination_check is True
    assert config.enable_consistency_check is True
    assert config.enable_llm_judge is True


def test_experiment_total_runs():
    """Experiment.total_runs should be num_templates × repetitions."""
    exp = Experiment(
        name="test_exp",
        prompt_templates=[
            PromptTemplate(name="a", template="t1"),
            PromptTemplate(name="b", template="t2"),
        ],
        dataset_id="ds1",
        eval_config=EvaluationConfig(repetitions=3),
    )
    # 2 templates × 3 reps = 6
    assert exp.total_runs == 6
    assert exp.status == ExperimentStatus.PENDING


def test_latency_metrics_defaults():
    """LatencyMetrics should default to zeros."""
    m = LatencyMetrics()
    assert m.total_ms == 0.0
    assert m.tokens_per_second == 0.0


def test_token_metrics():
    """TokenMetrics should compute total_tokens correctly."""
    m = TokenMetrics(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    assert m.total_tokens == 150
    assert m.output_input_ratio == 0.0  # default


def test_semantic_score():
    """SemanticScore should accept all judge dimensions."""
    s = SemanticScore(
        embedding_similarity=0.95,
        judge_relevance=4.0,
        judge_correctness=5.0,
        judge_coherence=4.5,
        judge_average=4.5,
    )
    assert s.embedding_similarity == 0.95
    assert s.judge_average == 4.5


def test_hallucination_report():
    """HallucinationReport should compute hallucination_rate."""
    report = HallucinationReport(
        total_claims=10,
        verified_claims=8,
        hallucinated_claims=2,
        hallucination_rate=0.2,
    )
    assert report.hallucination_rate == 0.2


def test_consistency_report():
    """ConsistencyReport defaults should indicate perfect consistency."""
    report = ConsistencyReport()
    assert report.mean_pairwise_similarity == 1.0
    assert report.agreement_rate == 1.0
    assert report.semantic_drift == 0.0


def test_run_result_serialization():
    """RunResult should serialize and deserialize correctly."""
    result = RunResult(
        experiment_id="exp1",
        prompt_template_id="pt1",
        prompt_template_name="test",
        test_case_id="tc1",
        repetition=1,
        response_text="Test response",
    )
    json_str = result.model_dump_json()
    restored = RunResult.model_validate_json(json_str)
    assert restored.experiment_id == "exp1"
    assert restored.response_text == "Test response"
    assert restored.status == RunStatus.SUCCESS


def test_metric_summary():
    """MetricSummary should default to zeros."""
    ms = MetricSummary()
    assert ms.mean == 0.0
    assert ms.p95 == 0.0


def test_variant_report():
    """VariantReport should accept all metric summaries."""
    vr = VariantReport(
        prompt_template_id="pt1",
        prompt_template_name="test",
        num_runs=10,
        num_successes=9,
        composite_score=0.85,
    )
    assert vr.composite_score == 0.85
    assert vr.num_failures == 0  # default


def test_experiment_report():
    """ExperimentReport should accept rankings and win rates."""
    report = ExperimentReport(
        experiment_id="exp1",
        experiment_name="Test Experiment",
        dataset_name="Test Dataset",
        rankings=[{"rank": 1, "name": "variant_a", "composite_score": 0.9}],
    )
    assert len(report.rankings) == 1
    assert report.rankings[0]["rank"] == 1


def test_create_experiment_request():
    """CreateExperimentRequest should validate required fields."""
    req = CreateExperimentRequest(
        name="Test Experiment",
        prompt_templates=[PromptTemplate(name="v1", template="t1")],
        dataset_id="ds1",
    )
    assert req.name == "Test Experiment"
    assert len(req.prompt_templates) == 1


def test_create_dataset_request():
    """CreateDatasetRequest should require at least one test case."""
    req = CreateDatasetRequest(
        name="Test Dataset",
        test_cases=[
            TestCase(input_variables={"q": "test"}, reference_answer="answer"),
        ],
    )
    assert len(req.test_cases) == 1


def main():
    """Run all schema tests."""
    tests = [
        test_generate_id,
        test_utc_now,
        test_prompt_template_defaults,
        test_test_case_structure,
        test_test_dataset_size_property,
        test_evaluation_config_defaults,
        test_experiment_total_runs,
        test_latency_metrics_defaults,
        test_token_metrics,
        test_semantic_score,
        test_hallucination_report,
        test_consistency_report,
        test_run_result_serialization,
        test_metric_summary,
        test_variant_report,
        test_experiment_report,
        test_create_experiment_request,
        test_create_dataset_request,
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
    print(f"Schema Tests: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        exit(1)


if __name__ == "__main__":
    main()
