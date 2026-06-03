"""
Experiment Runner.

Orchestrates the full execution of an experiment:
1. Builds the execution matrix: (prompt_variant × test_case × repetition)
2. Renders each prompt template with test case variables
3. Sends each rendered prompt to Ollama and captures response + timing
4. Pipes each response through all evaluation modules
5. Stores RunResult objects and returns the complete set

The runner supports configurable repetitions for statistical significance
and handles timeouts and errors gracefully per-run.

Usage:
    runner = ExperimentRunner(llm_client, database)
    results = await runner.run(experiment, dataset)
"""

from __future__ import annotations

import asyncio
import time

import structlog

from src.config import settings
from src.evaluation.composite_scorer import CompositeScorer
from src.evaluation.consistency_checker import ConsistencyChecker
from src.evaluation.hallucination_checker import HallucinationChecker
from src.evaluation.latency_analyzer import LatencyAnalyzer
from src.evaluation.semantic_scorer import SemanticScorer
from src.evaluation.token_analyzer import TokenAnalyzer
from src.models.llm_client import OllamaClient
from src.models.schemas import (
    Experiment,
    ExperimentStatus,
    PromptTemplate,
    RunResult,
    RunStatus,
    TestCase,
    TestDataset,
    utc_now,
)
from src.storage.database import Database

logger = structlog.get_logger(__name__)


class ExperimentRunner:
    """Executes controlled experiments by running prompt variants against test datasets."""

    def __init__(self, llm_client: OllamaClient, database: Database):
        self.llm_client = llm_client
        self.database = database

        # Initialize evaluation modules
        self.latency_analyzer = LatencyAnalyzer()
        self.token_analyzer = TokenAnalyzer()
        self.semantic_scorer = SemanticScorer(llm_client)
        self.hallucination_checker = HallucinationChecker(llm_client)
        self.consistency_checker = ConsistencyChecker(llm_client)
        self.composite_scorer = CompositeScorer()

    async def run(
        self, experiment: Experiment, dataset: TestDataset
    ) -> list[RunResult]:
        """Execute an experiment: run all prompt variants against all test cases.

        Args:
            experiment: The Experiment to execute.
            dataset: The TestDataset containing test cases.

        Returns:
            List of RunResult objects for every (variant × test_case × repetition).
        """
        start_time = time.perf_counter()
        total_combos = (
            len(experiment.prompt_templates)
            * len(dataset.test_cases)
            * experiment.eval_config.repetitions
        )
        logger.info(
            "experiment_run_start",
            experiment_id=experiment.id,
            experiment_name=experiment.name,
            num_variants=len(experiment.prompt_templates),
            num_test_cases=len(dataset.test_cases),
            repetitions=experiment.eval_config.repetitions,
            total_runs=total_combos,
        )

        # Update status to RUNNING
        self.database.update_experiment_status(
            experiment.id,
            ExperimentStatus.RUNNING,
            started_at=utc_now().isoformat(),
        )

        all_results: list[RunResult] = []

        try:
            # ─── Execute each variant ─────────────────────────────────────
            for template in experiment.prompt_templates:
                variant_results = await self._run_variant(
                    experiment=experiment,
                    template=template,
                    test_cases=dataset.test_cases,
                    eval_config=experiment.eval_config,
                )
                all_results.extend(variant_results)

            # ─── Consistency check (across repetitions for each variant × test_case) ──
            if experiment.eval_config.enable_consistency_check:
                await self._add_consistency_scores(experiment, all_results)

            # ─── Persist all results ──────────────────────────────────────
            self.database.save_run_results_batch(all_results)

            # Update status to COMPLETED
            self.database.update_experiment_status(
                experiment.id,
                ExperimentStatus.COMPLETED,
                completed_at=utc_now().isoformat(),
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "experiment_run_complete",
                experiment_id=experiment.id,
                total_results=len(all_results),
                elapsed_ms=round(elapsed_ms, 1),
            )

        except Exception as exc:
            logger.error(
                "experiment_run_failed",
                experiment_id=experiment.id,
                error=str(exc),
            )
            self.database.update_experiment_status(
                experiment.id,
                ExperimentStatus.FAILED,
                completed_at=utc_now().isoformat(),
            )
            raise

        return all_results

    async def _run_variant(
        self,
        experiment: Experiment,
        template: PromptTemplate,
        test_cases: list[TestCase],
        eval_config,
    ) -> list[RunResult]:
        """Run a single prompt variant against all test cases with repetitions.

        Args:
            experiment: The parent experiment.
            template: The prompt template variant.
            test_cases: List of test cases to evaluate.
            eval_config: Evaluation configuration.

        Returns:
            List of RunResult objects for this variant.
        """
        results: list[RunResult] = []

        for test_case in test_cases:
            for rep in range(1, eval_config.repetitions + 1):
                result = await self._execute_single_run(
                    experiment_id=experiment.id,
                    template=template,
                    test_case=test_case,
                    repetition=rep,
                    eval_config=eval_config,
                )
                results.append(result)

                logger.debug(
                    "run_completed",
                    variant=template.name,
                    test_case_id=test_case.id,
                    repetition=rep,
                    status=result.status.value,
                )

        return results

    async def _execute_single_run(
        self,
        experiment_id: str,
        template: PromptTemplate,
        test_case: TestCase,
        repetition: int,
        eval_config,
    ) -> RunResult:
        """Execute a single (prompt × test_case × repetition) and evaluate the response.

        Args:
            experiment_id: Parent experiment ID.
            template: The prompt template.
            test_case: The test case with input variables and reference answer.
            repetition: The repetition number.
            eval_config: Evaluation configuration.

        Returns:
            RunResult with all metrics populated.
        """
        # 1. Render the prompt
        rendered_prompt = self._render_prompt(template.template, test_case.input_variables)

        try:
            # 2. Generate response from Ollama
            response_text, ollama_metrics = await asyncio.wait_for(
                self.llm_client.generate(
                    prompt=rendered_prompt,
                    system_prompt=template.system_prompt,
                    temperature=eval_config.temperature,
                    max_tokens=eval_config.max_tokens,
                ),
                timeout=settings.run_timeout_seconds,
            )

            # 3. Run evaluation pipeline
            # Latency
            latency = self.latency_analyzer.analyze(ollama_metrics)

            # Token efficiency
            tokens = self.token_analyzer.analyze(
                ollama_metrics, response_text, test_case.reference_answer
            )

            # Semantic quality
            semantic = await self.semantic_scorer.score(
                question=self._get_primary_input(test_case),
                response_text=response_text,
                reference_answer=test_case.reference_answer,
                enable_llm_judge=eval_config.enable_llm_judge,
            )

            # Hallucination check
            hallucination = None
            if eval_config.enable_hallucination_check and test_case.reference_context:
                hallucination = await self.hallucination_checker.check(
                    response_text=response_text,
                    reference_context=test_case.reference_context,
                )

            return RunResult(
                experiment_id=experiment_id,
                prompt_template_id=template.id,
                prompt_template_name=template.name,
                test_case_id=test_case.id,
                repetition=repetition,
                status=RunStatus.SUCCESS,
                prompt_rendered=rendered_prompt,
                response_text=response_text,
                latency=latency,
                tokens=tokens,
                semantic=semantic,
                hallucination=hallucination,
            )

        except asyncio.TimeoutError:
            logger.warning(
                "run_timeout",
                template=template.name,
                test_case_id=test_case.id,
                timeout=settings.run_timeout_seconds,
            )
            return RunResult(
                experiment_id=experiment_id,
                prompt_template_id=template.id,
                prompt_template_name=template.name,
                test_case_id=test_case.id,
                repetition=repetition,
                status=RunStatus.TIMEOUT,
                prompt_rendered=rendered_prompt,
                error_message=f"Timeout after {settings.run_timeout_seconds}s",
            )

        except Exception as exc:
            logger.error(
                "run_error",
                template=template.name,
                test_case_id=test_case.id,
                error=str(exc),
            )
            return RunResult(
                experiment_id=experiment_id,
                prompt_template_id=template.id,
                prompt_template_name=template.name,
                test_case_id=test_case.id,
                repetition=repetition,
                status=RunStatus.ERROR,
                prompt_rendered=rendered_prompt,
                error_message=str(exc),
            )

    async def _add_consistency_scores(
        self, experiment: Experiment, results: list[RunResult]
    ) -> None:
        """Compute consistency scores for each (variant × test_case) group.

        Groups results by (template_id, test_case_id), then runs the consistency
        checker on each group's response texts.

        Args:
            experiment: The parent experiment.
            results: All run results to augment with consistency data.
        """
        from collections import defaultdict

        # Group by (template_id, test_case_id)
        groups: dict[tuple[str, str], list[RunResult]] = defaultdict(list)
        for r in results:
            if r.status == RunStatus.SUCCESS:
                groups[(r.prompt_template_id, r.test_case_id)].append(r)

        for (tid, tcid), group_results in groups.items():
            if len(group_results) < 2:
                continue

            responses = [r.response_text for r in group_results]
            consistency = await self.consistency_checker.check(responses)

            # Attach the consistency report to each run in the group
            for r in group_results:
                r.consistency = consistency

        logger.info(
            "consistency_scores_added",
            num_groups=len(groups),
        )

    @staticmethod
    def _render_prompt(
        template: str, variables: dict[str, str]
    ) -> str:
        """Render a prompt template by substituting variables.

        Args:
            template: Template string with {variable} placeholders.
            variables: Dict of variable_name → value.

        Returns:
            Rendered prompt string.
        """
        try:
            return template.format(**variables)
        except KeyError as exc:
            logger.warning("prompt_render_missing_variable", variable=str(exc))
            # Partial render: replace what we can, leave rest as-is
            result = template
            for key, value in variables.items():
                result = result.replace(f"{{{key}}}", value)
            return result

    @staticmethod
    def _get_primary_input(test_case: TestCase) -> str:
        """Extract the primary input text from a test case's variables.

        Looks for common input variable names: 'question', 'input', 'query', 'prompt'.
        Falls back to the first variable value.

        Args:
            test_case: The test case.

        Returns:
            The primary input text.
        """
        for key in ("question", "input", "query", "prompt"):
            if key in test_case.input_variables:
                return test_case.input_variables[key]
        # Fall back to first variable
        if test_case.input_variables:
            return next(iter(test_case.input_variables.values()))
        return ""
