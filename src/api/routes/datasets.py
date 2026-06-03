"""
Dataset Routes.

CRUD endpoints for managing test datasets used in experiments.
Datasets contain test cases with input variables, reference answers,
and optional reference context for hallucination checking.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_database
from src.models.schemas import CreateDatasetRequest, DatasetSummary, TestDataset
from src.storage.database import Database

router = APIRouter(prefix="/api/v1/datasets", tags=["Datasets"])


@router.post("", response_model=TestDataset, status_code=201)
async def create_dataset(
    request: CreateDatasetRequest,
    db: Database = Depends(get_database),
):
    """Create a new test dataset.

    Args:
        request: Dataset name, description, and list of test cases.

    Returns:
        The created TestDataset with generated ID.
    """
    dataset = TestDataset(
        name=request.name,
        description=request.description,
        test_cases=request.test_cases,
    )
    db.save_dataset(dataset)
    return dataset


@router.get("", response_model=list[DatasetSummary])
async def list_datasets(db: Database = Depends(get_database)):
    """List all test datasets.

    Returns:
        List of DatasetSummary objects with name, size, and creation date.
    """
    return db.list_datasets()


@router.get("/{dataset_id}", response_model=TestDataset)
async def get_dataset(
    dataset_id: str,
    db: Database = Depends(get_database),
):
    """Get a dataset by ID, including all test cases.

    Args:
        dataset_id: The dataset ID.

    Returns:
        The full TestDataset.

    Raises:
        404: If the dataset is not found.
    """
    dataset = db.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    return dataset


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(
    dataset_id: str,
    db: Database = Depends(get_database),
):
    """Delete a dataset by ID.

    Args:
        dataset_id: The dataset ID.

    Raises:
        404: If the dataset is not found.
        409: If the dataset is referenced by an experiment.
    """
    try:
        deleted = db.delete_dataset(dataset_id)
    except Exception:
        raise HTTPException(
            status_code=409,
            detail="Cannot delete dataset: it is referenced by one or more experiments",
        )
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
