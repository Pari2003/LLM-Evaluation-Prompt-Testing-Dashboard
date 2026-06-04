"""
Storage Layer Tests.

Tests for SQLite database CRUD operations, cascading deletes,
and FTS5 search functionality.
"""

from __future__ import annotations

from pathlib import Path

from src.models.schemas import (
    Experiment,
    ExperimentReport,
    ExperimentStatus,
    LatencyMetrics,
    PromptTemplate,
    RunResult,
    TestCase,
    TestDataset,
    TokenMetrics,
)
from src.storage.database import Database


def get_test_db() -> Database:
    """Create a fresh test database."""
    test_path = Path("data/test_storage.sqlite")
    if test_path.exists():
        test_path.unlink()
    for suffix in ["-wal", "-shm"]:
        p = test_path.with_name(test_path.name + suffix)
        if p.exists():
            p.unlink()
    return Database(db_path=test_path)


def test_dataset_crud():
    """Test create, read, list, and delete for datasets."""
    db = get_test_db()
    try:
        # Create
        dataset = TestDataset(
            name="Test Dataset",
            description="A test dataset",
            test_cases=[
                TestCase(input_variables={"q": "What is AI?"}, reference_answer="AI is..."),
                TestCase(input_variables={"q": "What is ML?"}, reference_answer="ML is..."),
            ],
        )
        db.save_dataset(dataset)

        # Read
        retrieved = db.get_dataset(dataset.id)
        assert retrieved is not None
        assert retrieved.name == "Test Dataset"
        assert retrieved.size == 2

        # List
        summaries = db.list_datasets()
        assert len(summaries) == 1
        assert summaries[0].num_test_cases == 2

        # Delete
        deleted = db.delete_dataset(dataset.id)
        assert deleted is True
        assert db.get_dataset(dataset.id) is None

        # Delete non-existent
        assert db.delete_dataset("nonexistent") is False

    finally:
        db.close()


def test_experiment_crud():
    """Test create, read, list, status update, and delete for experiments."""
    db = get_test_db()
    try:
        # First create a dataset (experiments reference datasets)
        dataset = TestDataset(
            name="Exp Test DS",
            test_cases=[
                TestCase(input_variables={"q": "test"}, reference_answer="answer"),
            ],
        )
        db.save_dataset(dataset)

        # Create experiment
        experiment = Experiment(
            name="Test Experiment",
            description="Testing experiment CRUD",
            prompt_templates=[
                PromptTemplate(name="v1", template="Answer: {q}"),
                PromptTemplate(name="v2", template="Explain: {q}"),
            ],
            dataset_id=dataset.id,
        )
        db.save_experiment(experiment)

        # Read
        retrieved = db.get_experiment(experiment.id)
        assert retrieved is not None
        assert retrieved.name == "Test Experiment"
        assert len(retrieved.prompt_templates) == 2
        assert retrieved.status == ExperimentStatus.PENDING

        # List
        summaries = db.list_experiments()
        assert len(summaries) == 1
        assert summaries[0].num_variants == 2

        # Status update
        db.update_experiment_status(
            experiment.id, ExperimentStatus.RUNNING, started_at="2025-01-01T00:00:00+00:00"
        )
        updated = db.get_experiment(experiment.id)
        assert updated.status == ExperimentStatus.RUNNING

        # Delete (should cascade)
        deleted = db.delete_experiment(experiment.id)
        assert deleted is True
        assert db.get_experiment(experiment.id) is None

    finally:
        db.close()


def test_run_results_crud():
    """Test saving and retrieving run results."""
    db = get_test_db()
    try:
        # Setup
        dataset = TestDataset(
            name="RR Test DS",
            test_cases=[TestCase(input_variables={"q": "t"}, reference_answer="a")],
        )
        db.save_dataset(dataset)
        experiment = Experiment(
            name="RR Test Exp",
            prompt_templates=[PromptTemplate(name="v1", template="t")],
            dataset_id=dataset.id,
        )
        db.save_experiment(experiment)

        # Save individual result
        result1 = RunResult(
            experiment_id=experiment.id,
            prompt_template_id=experiment.prompt_templates[0].id,
            prompt_template_name="v1",
            test_case_id="tc1",
            repetition=1,
            response_text="Test response",
            latency=LatencyMetrics(total_ms=500.0, tokens_per_second=30.0),
            tokens=TokenMetrics(prompt_tokens=50, completion_tokens=30, total_tokens=80),
        )
        db.save_run_result(result1)

        # Save batch
        result2 = RunResult(
            experiment_id=experiment.id,
            prompt_template_id=experiment.prompt_templates[0].id,
            prompt_template_name="v1",
            test_case_id="tc1",
            repetition=2,
            response_text="Test response 2",
        )
        db.save_run_results_batch([result2])

        # Retrieve all for experiment
        results = db.get_experiment_results(experiment.id)
        assert len(results) == 2

        # Retrieve by variant
        variant_results = db.get_variant_results(experiment.id, experiment.prompt_templates[0].id)
        assert len(variant_results) == 2

        # Run count
        count = db.get_run_count(experiment.id)
        assert count == 2

        # Cascade delete: deleting experiment should delete run results
        db.delete_experiment(experiment.id)
        assert db.get_run_count(experiment.id) == 0

    finally:
        db.close()


def test_experiment_report():
    """Test saving and retrieving experiment reports."""
    db = get_test_db()
    try:
        # Setup
        dataset = TestDataset(
            name="Report DS",
            test_cases=[TestCase(input_variables={"q": "t"}, reference_answer="a")],
        )
        db.save_dataset(dataset)
        experiment = Experiment(
            name="Report Exp",
            prompt_templates=[PromptTemplate(name="v1", template="t")],
            dataset_id=dataset.id,
        )
        db.save_experiment(experiment)

        # Save report
        report = ExperimentReport(
            experiment_id=experiment.id,
            experiment_name=experiment.name,
            dataset_name=dataset.name,
            total_runs=10,
            rankings=[{"rank": 1, "name": "v1", "composite_score": 0.9}],
        )
        db.save_experiment_report(report)

        # Retrieve
        retrieved = db.get_experiment_report(experiment.id)
        assert retrieved is not None
        assert retrieved.total_runs == 10
        assert len(retrieved.rankings) == 1

        # Cascade: delete experiment should delete report
        db.delete_experiment(experiment.id)
        assert db.get_experiment_report(experiment.id) is None

    finally:
        db.close()


def test_search_experiments():
    """Test full-text search on experiments."""
    db = get_test_db()
    try:
        dataset = TestDataset(
            name="Search DS",
            test_cases=[TestCase(input_variables={"q": "t"}, reference_answer="a")],
        )
        db.save_dataset(dataset)

        exp1 = Experiment(
            name="Prompt Style Comparison",
            description="Compare concise vs detailed prompts",
            prompt_templates=[PromptTemplate(name="v1", template="t")],
            dataset_id=dataset.id,
        )
        exp2 = Experiment(
            name="Temperature Ablation",
            description="Test temperature effects on quality",
            prompt_templates=[PromptTemplate(name="v1", template="t")],
            dataset_id=dataset.id,
        )
        db.save_experiment(exp1)
        db.save_experiment(exp2)

        # Search by name
        results = db.search_experiments("Prompt")
        assert len(results) >= 1
        assert any("Prompt" in r.name for r in results)

    finally:
        db.close()


def main():
    """Run all storage tests."""
    tests = [
        test_dataset_crud,
        test_experiment_crud,
        test_run_results_crud,
        test_experiment_report,
        test_search_experiments,
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
    print(f"Storage Tests: {passed} passed, {failed} failed out of {len(tests)}")

    # Cleanup
    test_path = Path("data/test_storage.sqlite")
    for suffix in ["", "-wal", "-shm"]:
        p = test_path.with_name(test_path.name + suffix)
        if p.exists():
            p.unlink()

    if failed > 0:
        exit(1)


if __name__ == "__main__":
    main()
