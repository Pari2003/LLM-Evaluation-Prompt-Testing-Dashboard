"""
Result Aggregator.

Aggregates individual RunResults into statistical summaries per prompt variant
and produces an ExperimentReport with head-to-head comparison and rankings.

Computes mean, median, stddev, min, max, p95 for all numeric metrics,
builds a win-rate matrix across prompt variants, and ranks variants
by composite score.

Usage:
    aggregator = ResultAggregator()
    report = aggregator.aggregate(experiment, dataset, results)
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from itertools import combinations
from typing import Any

import structlog

from src.evaluation.composite_scorer import CompositeScorer
from src.evaluation.significance import cohens_d, welch_t_test
from src.models.schemas import (
    Experiment,
    ExperimentReport,
    MetricSummary,
    RunResult,
    RunStatus,
    SignificanceTest,
    TestDataset,
    VariantReport,
    WinRateEntry,
)

logger = structlog.get_logger(__name__)


class ResultAggregator:
    """Aggregates run results into variant reports and experiment-level comparisons."""

    def __init__(self):
        self.composite_scorer = CompositeScorer()

    def aggregate(
        self,
        experiment: Experiment,
        dataset: TestDataset,
        results: list[RunResult],
    ) -> ExperimentReport:
        """Produce a full experiment report from individual run results.

        Args:
            experiment: The experiment definition.
            dataset: The test dataset used.
            results: All RunResult objects from the experiment execution.

        Returns:
            ExperimentReport with per-variant summaries, win-rate matrix, and rankings.
        """
        logger.info(
            "aggregation_start",
            experiment_id=experiment.id,
            total_results=len(results),
        )

        # ─── Group results by prompt template ─────────────────────────────
        variant_groups: dict[str, list[RunResult]] = defaultdict(list)
        for r in results:
            variant_groups[r.prompt_template_id].append(r)

        # ─── Build per-variant reports ────────────────────────────────────
        variant_reports: list[VariantReport] = []
        for template in experiment.prompt_templates:
            group = variant_groups.get(template.id, [])
            report = self._build_variant_report(template.id, template.name, group)
            variant_reports.append(report)

        # ─── Build win-rate matrix ────────────────────────────────────────
        win_rate_matrix = self._build_win_rate_matrix(variant_reports, results)

        # ─── Rank variants by composite score ─────────────────────────────
        rankings = self._rank_variants(variant_reports)

        # ─── Statistical significance tests (pairwise) ───────────────
        significance_tests = self._run_significance_tests(variant_reports, results)

        report = ExperimentReport(
            experiment_id=experiment.id,
            experiment_name=experiment.name,
            dataset_name=dataset.name,
            total_runs=len(results),
            total_test_cases=len(dataset.test_cases),
            variant_reports=variant_reports,
            win_rate_matrix=win_rate_matrix,
            significance_tests=significance_tests,
            rankings=rankings,
        )

        logger.info(
            "aggregation_complete",
            experiment_id=experiment.id,
            num_variants=len(variant_reports),
            winner=rankings[0]["name"] if rankings else "N/A",
        )

        return report

    def _build_variant_report(
        self,
        template_id: str,
        template_name: str,
        results: list[RunResult],
    ) -> VariantReport:
        """Build a statistical summary for a single prompt variant.

        Args:
            template_id: The prompt template ID.
            template_name: The prompt template name.
            results: All RunResults for this variant.

        Returns:
            VariantReport with per-metric statistical summaries.
        """
        successful = [r for r in results if r.status == RunStatus.SUCCESS]
        num_failures = len(results) - len(successful)

        if not successful:
            return VariantReport(
                prompt_template_id=template_id,
                prompt_template_name=template_name,
                num_runs=len(results),
                num_successes=0,
                num_failures=num_failures,
            )

        # Extract metric values
        latency_vals = [r.latency.total_ms for r in successful]
        tps_vals = [r.latency.tokens_per_second for r in successful]
        prompt_tok_vals = [float(r.tokens.prompt_tokens) for r in successful]
        comp_tok_vals = [float(r.tokens.completion_tokens) for r in successful]
        verbosity_vals = [r.tokens.verbosity_score for r in successful]
        emb_sim_vals = [r.semantic.embedding_similarity for r in successful]
        judge_vals = [r.semantic.judge_average for r in successful]

        halluc_vals = [
            r.hallucination.hallucination_rate for r in successful if r.hallucination is not None
        ]
        consistency_vals = [
            r.consistency.agreement_rate for r in successful if r.consistency is not None
        ]

        # Compute composite score for each run
        composites = []
        for r in successful:
            score = self.composite_scorer.compute(
                latency=r.latency,
                tokens=r.tokens,
                semantic=r.semantic,
                hallucination=r.hallucination,
                consistency=r.consistency,
            )
            composites.append(score)

        avg_composite = statistics.mean(composites) if composites else 0.0

        return VariantReport(
            prompt_template_id=template_id,
            prompt_template_name=template_name,
            num_runs=len(results),
            num_successes=len(successful),
            num_failures=num_failures,
            latency_ms=self._summarize(latency_vals),
            tokens_per_second=self._summarize(tps_vals),
            prompt_tokens=self._summarize(prompt_tok_vals),
            completion_tokens=self._summarize(comp_tok_vals),
            verbosity=self._summarize(verbosity_vals),
            embedding_similarity=self._summarize(emb_sim_vals),
            judge_average=self._summarize(judge_vals),
            hallucination_rate=self._summarize(halluc_vals) if halluc_vals else MetricSummary(),
            consistency_agreement=self._summarize(consistency_vals)
            if consistency_vals
            else MetricSummary(),
            composite_score=round(avg_composite, 4),
        )

    def _build_win_rate_matrix(
        self,
        variant_reports: list[VariantReport],
        results: list[RunResult],
    ) -> list[WinRateEntry]:
        """Build head-to-head win-rate comparison between all variant pairs.

        For each test case, the variant with the higher embedding_similarity wins.
        If no semantic score is available, uses latency (lower wins).

        Args:
            variant_reports: The per-variant reports.
            results: All run results.

        Returns:
            List of WinRateEntry objects for each pair of variants.
        """
        if len(variant_reports) < 2:
            return []

        # Group successful results by (template_id, test_case_id)
        # For each test_case, compute average embedding_similarity per variant
        variant_test_scores: dict[str, dict[str, float]] = defaultdict(dict)
        variant_test_counts: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for r in results:
            if r.status == RunStatus.SUCCESS:
                variant_test_counts[r.prompt_template_id][r.test_case_id].append(
                    r.semantic.embedding_similarity
                )

        for vid, test_cases in variant_test_counts.items():
            for tcid, scores in test_cases.items():
                variant_test_scores[vid][tcid] = statistics.mean(scores)

        # Pairwise comparison
        entries: list[WinRateEntry] = []
        for va, vb in combinations(variant_reports, 2):
            a_wins = 0
            b_wins = 0
            ties = 0
            total = 0

            # Find shared test cases
            a_scores = variant_test_scores.get(va.prompt_template_id, {})
            b_scores = variant_test_scores.get(vb.prompt_template_id, {})
            shared_cases = set(a_scores.keys()) & set(b_scores.keys())

            for tcid in shared_cases:
                total += 1
                diff = a_scores[tcid] - b_scores[tcid]
                if abs(diff) < 0.01:
                    ties += 1
                elif diff > 0:
                    a_wins += 1
                else:
                    b_wins += 1

            a_rate = a_wins / total if total > 0 else 0.0

            entries.append(
                WinRateEntry(
                    variant_a=va.prompt_template_name,
                    variant_b=vb.prompt_template_name,
                    variant_a_wins=a_wins,
                    variant_b_wins=b_wins,
                    ties=ties,
                    total_comparisons=total,
                    variant_a_win_rate=round(a_rate, 3),
                )
            )

        return entries

    @staticmethod
    def _rank_variants(
        variant_reports: list[VariantReport],
    ) -> list[dict[str, Any]]:
        """Rank prompt variants by composite score (descending).

        Args:
            variant_reports: The per-variant reports.

        Returns:
            Sorted list of dicts with rank, name, id, and composite score.
        """
        sorted_variants = sorted(
            variant_reports,
            key=lambda v: v.composite_score,
            reverse=True,
        )

        return [
            {
                "rank": idx + 1,
                "name": v.prompt_template_name,
                "prompt_template_id": v.prompt_template_id,
                "composite_score": v.composite_score,
                "num_runs": v.num_runs,
                "num_successes": v.num_successes,
            }
            for idx, v in enumerate(sorted_variants)
        ]

    def _run_significance_tests(
        self,
        variant_reports: list[VariantReport],
        results: list[RunResult],
    ) -> list[SignificanceTest]:
        """Run pairwise Welch's t-test between all variant pairs.

        For each pair of variants, computes composite scores per run and
        tests whether the difference in means is statistically significant.

        This catches cases where variant A has a higher mean score than B,
        but the difference is within random noise (especially common with
        small repetition counts like 3-5).

        Args:
            variant_reports: The per-variant reports (sorted by composite score).
            results: All run results.

        Returns:
            List of SignificanceTest objects for each variant pair.
        """
        if len(variant_reports) < 2:
            return []

        # Compute per-run composite scores grouped by variant
        variant_scores: dict[str, list[float]] = defaultdict(list)
        for r in results:
            if r.status == RunStatus.SUCCESS:
                score = self.composite_scorer.compute(
                    latency=r.latency,
                    tokens=r.tokens,
                    semantic=r.semantic,
                    hallucination=r.hallucination,
                    consistency=r.consistency,
                )
                variant_scores[r.prompt_template_id].append(score)

        # Sort by mean composite (descending) for pairwise comparison
        sorted_variants = sorted(
            variant_reports,
            key=lambda v: v.composite_score,
            reverse=True,
        )

        sig_tests: list[SignificanceTest] = []
        for va, vb in combinations(sorted_variants, 2):
            scores_a = variant_scores.get(va.prompt_template_id, [])
            scores_b = variant_scores.get(vb.prompt_template_id, [])

            if not scores_a or not scores_b:
                continue

            # Welch's t-test
            t_result = welch_t_test(scores_a, scores_b)

            # Cohen's d effect size
            d_value, d_magnitude = cohens_d(scores_a, scores_b)

            # Build interpretation
            mean_a = statistics.mean(scores_a)
            mean_b = statistics.mean(scores_b)
            if t_result["significant"]:
                interpretation = (
                    f"'{va.prompt_template_name}' scores significantly higher than "
                    f"'{vb.prompt_template_name}' "
                    f"(p={t_result['p_value']:.4f}, Cohen's d={d_value:.2f}, "
                    f"{d_magnitude} effect). This difference is real, not noise."
                )
            else:
                interpretation = (
                    f"No significant difference between "
                    f"'{va.prompt_template_name}' and '{vb.prompt_template_name}' "
                    f"(p={t_result['p_value']:.4f}). The apparent score difference "
                    f"({mean_a:.4f} vs {mean_b:.4f}) may be due to random variation."
                )

            sig_tests.append(
                SignificanceTest(
                    variant_a=va.prompt_template_name,
                    variant_b=vb.prompt_template_name,
                    mean_a=round(mean_a, 4),
                    mean_b=round(mean_b, 4),
                    t_statistic=t_result["t_statistic"],
                    degrees_of_freedom=t_result["degrees_of_freedom"],
                    p_value=t_result["p_value"],
                    significant=t_result["significant"],
                    effect_size=d_value,
                    effect_magnitude=d_magnitude,
                    interpretation=interpretation,
                )
            )

        logger.info(
            "significance_tests_complete",
            num_tests=len(sig_tests),
            any_significant=any(t.significant for t in sig_tests),
        )

        return sig_tests

    @staticmethod
    def _summarize(values: list[float]) -> MetricSummary:
        """Compute statistical summary for a list of numeric values.

        Args:
            values: List of float values.

        Returns:
            MetricSummary with mean, median, stddev, min, max, p95.
        """
        if not values:
            return MetricSummary()

        sorted_vals = sorted(values)
        n = len(sorted_vals)

        # Percentile calculation
        p95_idx = (95 / 100) * (n - 1)
        lower = int(p95_idx)
        upper = min(lower + 1, n - 1)
        weight = p95_idx - lower
        p95 = sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight

        return MetricSummary(
            mean=round(statistics.mean(values), 4),
            median=round(statistics.median(values), 4),
            stddev=round(statistics.stdev(values), 4) if n > 1 else 0.0,
            min=round(min(values), 4),
            max=round(max(values), 4),
            p95=round(p95, 4),
        )
