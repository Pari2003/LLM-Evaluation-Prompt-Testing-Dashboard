"""
Result Routes.

Endpoints for querying individual run results and raw experiment data.
These complement the experiment routes by providing granular access
to per-run metrics and per-variant breakdowns.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_database
from src.models.schemas import RunResult
from src.storage.database import Database

router = APIRouter(prefix="/api/v1/results", tags=["Results"])


@router.get("/experiment/{experiment_id}", response_model=list[RunResult])
async def get_run_results(
    experiment_id: str,
    db: Database = Depends(get_database),
):
    """Get all individual run results for an experiment.

    Returns the raw per-run data including rendered prompts, responses,
    and all evaluation metrics.

    Args:
        experiment_id: The experiment ID.

    Returns:
        List of RunResult objects.

    Raises:
        404: If no results are found for the experiment.
    """
    results = db.get_experiment_results(experiment_id)
    if not results:
        experiment = db.get_experiment(experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
        raise HTTPException(
            status_code=404,
            detail="No run results found. Run the experiment first.",
        )
    return results


@router.get(
    "/experiment/{experiment_id}/variant/{prompt_template_id}",
    response_model=list[RunResult],
)
async def get_variant_results(
    experiment_id: str,
    prompt_template_id: str,
    db: Database = Depends(get_database),
):
    """Get run results filtered by a specific prompt variant.

    Args:
        experiment_id: The experiment ID.
        prompt_template_id: The prompt template ID.

    Returns:
        List of RunResult objects for the specified variant.

    Raises:
        404: If no results are found.
    """
    results = db.get_variant_results(experiment_id, prompt_template_id)
    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for variant {prompt_template_id}",
        )
    return results


@router.get("/experiment/{experiment_id}/stats")
async def get_run_stats(
    experiment_id: str,
    db: Database = Depends(get_database),
):
    """Get quick statistics about an experiment's runs.

    Args:
        experiment_id: The experiment ID.

    Returns:
        Dict with total runs, status breakdown, and basic metrics.
    """
    results = db.get_experiment_results(experiment_id)
    experiment = db.get_experiment(experiment_id)

    if not experiment:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    status_counts = {}
    for r in results:
        status_counts[r.status.value] = status_counts.get(r.status.value, 0) + 1

    return {
        "experiment_id": experiment_id,
        "experiment_name": experiment.name,
        "status": experiment.status.value,
        "total_runs": len(results),
        "status_breakdown": status_counts,
    }
