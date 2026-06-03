"""
Experiment Routes.

CRUD endpoints for managing experiments and triggering execution.
Experiments link prompt variants with test datasets and evaluation configs.
Running an experiment executes all combinations and produces evaluation metrics.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from src.api.dependencies import get_aggregator, get_database, get_runner
from src.execution.result_aggregator import ResultAggregator
from src.execution.runner import ExperimentRunner
from src.models.schemas import (
    CreateExperimentRequest,
    Experiment,
    ExperimentReport,
    ExperimentStatus,
    ExperimentSummary,
)
from src.storage.database import Database

router = APIRouter(prefix="/api/v1/experiments", tags=["Experiments"])


@router.post("", response_model=Experiment, status_code=201)
async def create_experiment(
    request: CreateExperimentRequest,
    db: Database = Depends(get_database),
):
    """Create a new experiment.

    The experiment is created in PENDING status and must be explicitly
    run via the /run endpoint.

    Args:
        request: Experiment name, prompt templates, dataset ID, and eval config.

    Returns:
        The created Experiment with generated ID.

    Raises:
        404: If the referenced dataset does not exist.
    """
    # Validate dataset exists
    dataset = db.get_dataset(request.dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {request.dataset_id} not found",
        )

    experiment = Experiment(
        name=request.name,
        description=request.description,
        prompt_templates=request.prompt_templates,
        dataset_id=request.dataset_id,
        eval_config=request.eval_config,
    )
    db.save_experiment(experiment)
    return experiment


@router.get("", response_model=list[ExperimentSummary])
async def list_experiments(db: Database = Depends(get_database)):
    """List all experiments.

    Returns:
        List of ExperimentSummary objects.
    """
    return db.list_experiments()


@router.get("/{experiment_id}", response_model=Experiment)
async def get_experiment(
    experiment_id: str,
    db: Database = Depends(get_database),
):
    """Get an experiment by ID, including full configuration.

    Args:
        experiment_id: The experiment ID.

    Returns:
        The full Experiment object.

    Raises:
        404: If the experiment is not found.
    """
    experiment = db.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(
            status_code=404, detail=f"Experiment {experiment_id} not found"
        )
    return experiment


@router.delete("/{experiment_id}", status_code=204)
async def delete_experiment(
    experiment_id: str,
    db: Database = Depends(get_database),
):
    """Delete an experiment and all its run results.

    Args:
        experiment_id: The experiment ID.

    Raises:
        404: If the experiment is not found.
    """
    deleted = db.delete_experiment(experiment_id)
    if not deleted:
        raise HTTPException(
            status_code=404, detail=f"Experiment {experiment_id} not found"
        )


@router.post("/{experiment_id}/run", status_code=202)
async def run_experiment(
    experiment_id: str,
    background_tasks: BackgroundTasks,
    db: Database = Depends(get_database),
    runner: ExperimentRunner = Depends(get_runner),
    aggregator: ResultAggregator = Depends(get_aggregator),
):
    """Trigger execution of an experiment.

    Runs the experiment in the background and returns immediately with 202 Accepted.
    Poll the experiment status or results endpoint to check progress.

    Args:
        experiment_id: The experiment ID.

    Returns:
        Dict with experiment ID and status.

    Raises:
        404: If the experiment or its dataset is not found.
        409: If the experiment is already running or completed.
    """
    experiment = db.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(
            status_code=404, detail=f"Experiment {experiment_id} not found"
        )

    if experiment.status in (ExperimentStatus.RUNNING, ExperimentStatus.COMPLETED):
        raise HTTPException(
            status_code=409,
            detail=f"Experiment is already {experiment.status.value}",
        )

    dataset = db.get_dataset(experiment.dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {experiment.dataset_id} not found",
        )

    # Run in background
    async def _run_and_aggregate():
        results = await runner.run(experiment, dataset)
        report = aggregator.aggregate(experiment, dataset, results)
        db.save_experiment_report(report)

    background_tasks.add_task(_run_and_aggregate)

    return {
        "experiment_id": experiment_id,
        "status": "accepted",
        "message": "Experiment execution started. Poll /results for progress.",
    }


@router.get("/{experiment_id}/results", response_model=ExperimentReport)
async def get_experiment_results(
    experiment_id: str,
    db: Database = Depends(get_database),
):
    """Get the full comparison report for a completed experiment.

    Args:
        experiment_id: The experiment ID.

    Returns:
        ExperimentReport with per-variant summaries, win-rate matrix, and rankings.

    Raises:
        404: If the experiment or report is not found.
    """
    experiment = db.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(
            status_code=404, detail=f"Experiment {experiment_id} not found"
        )

    report = db.get_experiment_report(experiment_id)
    if not report:
        if experiment.status == ExperimentStatus.RUNNING:
            raise HTTPException(
                status_code=409,
                detail="Experiment is still running. Results not yet available.",
            )
        raise HTTPException(
            status_code=404,
            detail="No results found. Run the experiment first via POST /run.",
        )
    return report


@router.get("/{experiment_id}/compare")
async def compare_variants(
    experiment_id: str,
    db: Database = Depends(get_database),
):
    """Get a simplified head-to-head comparison of prompt variants.

    Returns rankings, win-rate matrix, and key metrics for quick comparison.

    Args:
        experiment_id: The experiment ID.

    Returns:
        Dict with rankings, win_rates, and per-variant key metrics.

    Raises:
        404: If the experiment or report is not found.
    """
    report = db.get_experiment_report(experiment_id)
    if not report:
        raise HTTPException(
            status_code=404,
            detail="No results found. Run the experiment first.",
        )

    # Build a simplified comparison view
    comparison = {
        "experiment": report.experiment_name,
        "dataset": report.dataset_name,
        "total_runs": report.total_runs,
        "rankings": report.rankings,
        "win_rates": [entry.model_dump() for entry in report.win_rate_matrix],
        "variants": [
            {
                "name": v.prompt_template_name,
                "composite_score": v.composite_score,
                "avg_latency_ms": v.latency_ms.mean,
                "avg_tokens_per_sec": v.tokens_per_second.mean,
                "avg_embedding_similarity": v.embedding_similarity.mean,
                "avg_judge_score": v.judge_average.mean,
                "avg_hallucination_rate": v.hallucination_rate.mean,
                "avg_consistency": v.consistency_agreement.mean,
                "success_rate": (
                    v.num_successes / v.num_runs if v.num_runs > 0 else 0.0
                ),
            }
            for v in report.variant_reports
        ],
    }

    return comparison
