"""
SQLite Storage Layer for the LLM Evaluation Dashboard.

Manages persistence for experiments, datasets, test cases, and run results.
Uses FTS5 for full-text search on experiment names and descriptions.

Design decisions:
- Direct SQLite (no ORM) for simplicity and single-file deployment.
- Cascading deletes: deleting an experiment removes all its runs and results.
- JSON serialization for nested Pydantic models stored in TEXT columns.
- Thread-safe via check_same_thread=False for FastAPI's async context.

Usage:
    db = Database()
    db.save_dataset(dataset)
    db.save_experiment(experiment)
    results = db.get_experiment_results(experiment_id)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import structlog

from src.config import settings
from src.models.schemas import (
    DatasetSummary,
    Experiment,
    ExperimentReport,
    ExperimentStatus,
    ExperimentSummary,
    RunResult,
    TestDataset,
)

logger = structlog.get_logger(__name__)


class Database:
    """SQLite storage backend for the evaluation platform."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.sqlite_file
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        self._create_tables()
        logger.info("database_initialized", path=str(self.db_path))

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # ─── Schema Creation ──────────────────────────────────────────────────

    def _create_tables(self) -> None:
        """Create all tables and FTS5 virtual tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                data_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                data_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                dataset_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE RESTRICT
            );

            CREATE TABLE IF NOT EXISTS run_results (
                id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                prompt_template_id TEXT NOT NULL,
                prompt_template_name TEXT NOT NULL,
                test_case_id TEXT NOT NULL,
                repetition INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'success',
                data_json TEXT NOT NULL,
                executed_at TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS experiment_reports (
                experiment_id TEXT PRIMARY KEY,
                report_json TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            );

            -- Indexes for common query patterns
            CREATE INDEX IF NOT EXISTS idx_run_results_experiment
                ON run_results(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_run_results_variant
                ON run_results(experiment_id, prompt_template_id);
            CREATE INDEX IF NOT EXISTS idx_experiments_status
                ON experiments(status);
        """)

        # FTS5 for searching experiments by name/description
        try:
            self._conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS experiments_fts USING fts5(
                    name,
                    description,
                    content='experiments',
                    content_rowid='rowid'
                );
            """)
        except sqlite3.OperationalError:
            # FTS5 may not be available in all SQLite builds
            logger.warning("fts5_not_available", msg="Full-text search disabled")

        self._conn.commit()

    # ─── Dataset CRUD ─────────────────────────────────────────────────────

    def save_dataset(self, dataset: TestDataset) -> None:
        """Save a test dataset to the database.

        Args:
            dataset: The TestDataset to persist.
        """
        self._conn.execute(
            """INSERT OR REPLACE INTO datasets (id, name, description, data_json, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                dataset.id,
                dataset.name,
                dataset.description,
                dataset.model_dump_json(),
                dataset.created_at.isoformat(),
            ),
        )
        self._conn.commit()
        logger.info("dataset_saved", id=dataset.id, name=dataset.name, size=dataset.size)

    def get_dataset(self, dataset_id: str) -> Optional[TestDataset]:
        """Retrieve a dataset by ID.

        Args:
            dataset_id: The dataset ID.

        Returns:
            TestDataset if found, None otherwise.
        """
        row = self._conn.execute(
            "SELECT data_json FROM datasets WHERE id = ?", (dataset_id,)
        ).fetchone()
        if row:
            return TestDataset.model_validate_json(row["data_json"])
        return None

    def list_datasets(self) -> list[DatasetSummary]:
        """List all datasets as lightweight summaries.

        Returns:
            List of DatasetSummary objects.
        """
        rows = self._conn.execute(
            "SELECT id, name, description, data_json, created_at FROM datasets ORDER BY created_at DESC"
        ).fetchall()
        summaries = []
        for row in rows:
            dataset = TestDataset.model_validate_json(row["data_json"])
            summaries.append(
                DatasetSummary(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    num_test_cases=dataset.size,
                    created_at=dataset.created_at,
                )
            )
        return summaries

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset by ID.

        Args:
            dataset_id: The dataset ID.

        Returns:
            True if the dataset was deleted, False if not found.

        Raises:
            sqlite3.IntegrityError: If the dataset is referenced by an experiment.
        """
        cursor = self._conn.execute(
            "DELETE FROM datasets WHERE id = ?", (dataset_id,)
        )
        self._conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("dataset_deleted", id=dataset_id)
        return deleted

    # ─── Experiment CRUD ──────────────────────────────────────────────────

    def save_experiment(self, experiment: Experiment) -> None:
        """Save an experiment to the database.

        Args:
            experiment: The Experiment to persist.
        """
        self._conn.execute(
            """INSERT OR REPLACE INTO experiments
               (id, name, description, data_json, status, dataset_id,
                created_at, started_at, completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                experiment.id,
                experiment.name,
                experiment.description,
                experiment.model_dump_json(),
                experiment.status.value,
                experiment.dataset_id,
                experiment.created_at.isoformat(),
                experiment.started_at.isoformat() if experiment.started_at else None,
                experiment.completed_at.isoformat() if experiment.completed_at else None,
            ),
        )

        # Update FTS index
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO experiments_fts (rowid, name, description) VALUES ("
                "(SELECT rowid FROM experiments WHERE id = ?), ?, ?)",
                (experiment.id, experiment.name, experiment.description),
            )
        except sqlite3.OperationalError:
            pass  # FTS5 not available

        self._conn.commit()
        logger.info("experiment_saved", id=experiment.id, name=experiment.name)

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Retrieve an experiment by ID.

        Args:
            experiment_id: The experiment ID.

        Returns:
            Experiment if found, None otherwise.
        """
        row = self._conn.execute(
            "SELECT data_json FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        if row:
            return Experiment.model_validate_json(row["data_json"])
        return None

    def list_experiments(self) -> list[ExperimentSummary]:
        """List all experiments as lightweight summaries.

        Returns:
            List of ExperimentSummary objects.
        """
        rows = self._conn.execute(
            """SELECT id, name, description, status, dataset_id, data_json,
                      created_at, completed_at
               FROM experiments ORDER BY created_at DESC"""
        ).fetchall()
        summaries = []
        for row in rows:
            experiment = Experiment.model_validate_json(row["data_json"])
            summaries.append(
                ExperimentSummary(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    status=ExperimentStatus(row["status"]),
                    num_variants=len(experiment.prompt_templates),
                    dataset_id=row["dataset_id"],
                    created_at=experiment.created_at,
                    completed_at=experiment.completed_at,
                )
            )
        return summaries

    def update_experiment_status(
        self,
        experiment_id: str,
        status: ExperimentStatus,
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
    ) -> None:
        """Update the status (and timestamps) of an experiment.

        Args:
            experiment_id: The experiment ID.
            status: The new status.
            started_at: ISO timestamp when the experiment started.
            completed_at: ISO timestamp when the experiment completed.
        """
        # Update the status column
        self._conn.execute(
            "UPDATE experiments SET status = ?, started_at = COALESCE(?, started_at), "
            "completed_at = COALESCE(?, completed_at) WHERE id = ?",
            (status.value, started_at, completed_at, experiment_id),
        )

        # Also update the JSON blob to keep it consistent
        row = self._conn.execute(
            "SELECT data_json FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        if row:
            experiment = Experiment.model_validate_json(row["data_json"])
            experiment.status = status
            if started_at:
                from datetime import datetime
                experiment.started_at = datetime.fromisoformat(started_at)
            if completed_at:
                from datetime import datetime
                experiment.completed_at = datetime.fromisoformat(completed_at)
            self._conn.execute(
                "UPDATE experiments SET data_json = ? WHERE id = ?",
                (experiment.model_dump_json(), experiment_id),
            )

        self._conn.commit()
        logger.info("experiment_status_updated", id=experiment_id, status=status.value)

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and all its run results (CASCADE).

        Args:
            experiment_id: The experiment ID.

        Returns:
            True if the experiment was deleted, False if not found.
        """
        cursor = self._conn.execute(
            "DELETE FROM experiments WHERE id = ?", (experiment_id,)
        )
        self._conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("experiment_deleted", id=experiment_id)
        return deleted

    # ─── Run Results ──────────────────────────────────────────────────────

    def save_run_result(self, result: RunResult) -> None:
        """Save a single run result.

        Args:
            result: The RunResult to persist.
        """
        self._conn.execute(
            """INSERT OR REPLACE INTO run_results
               (id, experiment_id, prompt_template_id, prompt_template_name,
                test_case_id, repetition, status, data_json, executed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result.id,
                result.experiment_id,
                result.prompt_template_id,
                result.prompt_template_name,
                result.test_case_id,
                result.repetition,
                result.status.value,
                result.model_dump_json(),
                result.executed_at.isoformat(),
            ),
        )
        self._conn.commit()

    def save_run_results_batch(self, results: list[RunResult]) -> None:
        """Save multiple run results in a single transaction.

        Args:
            results: List of RunResult objects to persist.
        """
        self._conn.executemany(
            """INSERT OR REPLACE INTO run_results
               (id, experiment_id, prompt_template_id, prompt_template_name,
                test_case_id, repetition, status, data_json, executed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    r.id, r.experiment_id, r.prompt_template_id, r.prompt_template_name,
                    r.test_case_id, r.repetition, r.status.value,
                    r.model_dump_json(), r.executed_at.isoformat(),
                )
                for r in results
            ],
        )
        self._conn.commit()
        logger.info("run_results_batch_saved", count=len(results))

    def get_experiment_results(self, experiment_id: str) -> list[RunResult]:
        """Retrieve all run results for an experiment.

        Args:
            experiment_id: The experiment ID.

        Returns:
            List of RunResult objects.
        """
        rows = self._conn.execute(
            "SELECT data_json FROM run_results WHERE experiment_id = ? ORDER BY executed_at",
            (experiment_id,),
        ).fetchall()
        return [RunResult.model_validate_json(row["data_json"]) for row in rows]

    def get_variant_results(
        self, experiment_id: str, prompt_template_id: str
    ) -> list[RunResult]:
        """Retrieve run results for a specific prompt variant within an experiment.

        Args:
            experiment_id: The experiment ID.
            prompt_template_id: The prompt template ID.

        Returns:
            List of RunResult objects for the specified variant.
        """
        rows = self._conn.execute(
            """SELECT data_json FROM run_results
               WHERE experiment_id = ? AND prompt_template_id = ?
               ORDER BY executed_at""",
            (experiment_id, prompt_template_id),
        ).fetchall()
        return [RunResult.model_validate_json(row["data_json"]) for row in rows]

    # ─── Experiment Reports ───────────────────────────────────────────────

    def save_experiment_report(self, report: ExperimentReport) -> None:
        """Save an experiment comparison report.

        Args:
            report: The ExperimentReport to persist.
        """
        self._conn.execute(
            """INSERT OR REPLACE INTO experiment_reports
               (experiment_id, report_json, generated_at)
               VALUES (?, ?, ?)""",
            (
                report.experiment_id,
                report.model_dump_json(),
                report.generated_at.isoformat(),
            ),
        )
        self._conn.commit()
        logger.info("experiment_report_saved", experiment_id=report.experiment_id)

    def get_experiment_report(self, experiment_id: str) -> Optional[ExperimentReport]:
        """Retrieve the comparison report for an experiment.

        Args:
            experiment_id: The experiment ID.

        Returns:
            ExperimentReport if found, None otherwise.
        """
        row = self._conn.execute(
            "SELECT report_json FROM experiment_reports WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        if row:
            return ExperimentReport.model_validate_json(row["report_json"])
        return None

    # ─── Search ───────────────────────────────────────────────────────────

    def search_experiments(self, query: str) -> list[ExperimentSummary]:
        """Full-text search for experiments by name or description.

        Args:
            query: Search query string.

        Returns:
            List of matching ExperimentSummary objects.
        """
        try:
            rows = self._conn.execute(
                """SELECT e.id, e.name, e.description, e.status, e.dataset_id,
                          e.data_json, e.created_at, e.completed_at
                   FROM experiments e
                   JOIN experiments_fts fts ON e.rowid = fts.rowid
                   WHERE experiments_fts MATCH ?
                   ORDER BY rank""",
                (query,),
            ).fetchall()
        except sqlite3.OperationalError:
            # Fall back to LIKE search if FTS5 is not available
            like_query = f"%{query}%"
            rows = self._conn.execute(
                """SELECT id, name, description, status, dataset_id, data_json,
                          created_at, completed_at
                   FROM experiments
                   WHERE name LIKE ? OR description LIKE ?
                   ORDER BY created_at DESC""",
                (like_query, like_query),
            ).fetchall()

        summaries = []
        for row in rows:
            experiment = Experiment.model_validate_json(row["data_json"])
            summaries.append(
                ExperimentSummary(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    status=ExperimentStatus(row["status"]),
                    num_variants=len(experiment.prompt_templates),
                    dataset_id=row["dataset_id"],
                    created_at=experiment.created_at,
                    completed_at=experiment.completed_at,
                )
            )
        return summaries

    # ─── Statistics ───────────────────────────────────────────────────────

    def get_run_count(self, experiment_id: str) -> int:
        """Get the number of run results for an experiment.

        Args:
            experiment_id: The experiment ID.

        Returns:
            Count of run results.
        """
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM run_results WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        return row["cnt"] if row else 0
