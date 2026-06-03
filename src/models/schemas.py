"""
Pydantic Data Models for the LLM Evaluation Dashboard.

Defines all request/response schemas, experiment models, metric types,
and evaluation report structures used across the platform.

Architecture:
    PromptTemplate → TestDataset (contains TestCases) → Experiment
    Experiment execution produces ExperimentRuns → RunResults
    Aggregation produces VariantReport → ExperimentReport
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

# ─── Enumerations ─────────────────────────────────────────────────────────


class ExperimentStatus(str, Enum):
    """Lifecycle status of an experiment."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EntailmentResult(str, Enum):
    """NLI entailment classification for hallucination checking."""
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


class RunStatus(str, Enum):
    """Status of an individual experiment run."""
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"


# ─── Helper ───────────────────────────────────────────────────────────────


def generate_id() -> str:
    """Generate a short unique ID for entities."""
    return uuid.uuid4().hex[:12]


def utc_now() -> datetime:
    """Current UTC timestamp."""
    return datetime.now(timezone.utc)


# ─── Prompt Templates ────────────────────────────────────────────────────


class PromptTemplate(BaseModel):
    """A named prompt template with variable placeholders.

    Example:
        name: "concise_qa"
        template: "Answer the following question concisely:\n\nQuestion: {question}\nAnswer:"
        variables: ["question"]
    """
    id: str = Field(default_factory=generate_id)
    name: str = Field(..., min_length=1, max_length=200)
    template: str = Field(..., min_length=1)
    system_prompt: Optional[str] = None
    variables: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


# ─── Test Datasets ───────────────────────────────────────────────────────


class TestCase(BaseModel):
    """A single test case containing input variables and an expected output.

    The `input_variables` dict must supply all variables referenced in the prompt template.
    The `reference_answer` is the ground-truth used for semantic comparison and hallucination checks.
    The optional `reference_context` provides source material for fact verification.
    """
    id: str = Field(default_factory=generate_id)
    input_variables: dict[str, str] = Field(..., min_length=1)
    reference_answer: str = Field(..., min_length=1)
    reference_context: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TestDataset(BaseModel):
    """A named collection of test cases for benchmarking prompt variants."""
    id: str = Field(default_factory=generate_id)
    name: str = Field(..., min_length=1, max_length=200)
    description: str = ""
    test_cases: list[TestCase] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def size(self) -> int:
        """Number of test cases in this dataset."""
        return len(self.test_cases)


# ─── Experiments ──────────────────────────────────────────────────────────


class EvaluationConfig(BaseModel):
    """Configuration controlling how an experiment is evaluated.

    Attributes:
        repetitions: Number of times each (prompt × test_case) is executed.
        temperature: LLM sampling temperature for this experiment.
        max_tokens: Maximum output tokens per generation.
        enable_hallucination_check: Whether to run claim-level hallucination analysis.
        enable_consistency_check: Whether to compute cross-run consistency metrics.
        enable_llm_judge: Whether to use LLM-as-Judge for semantic quality scoring.
    """
    repetitions: int = Field(default=3, ge=1, le=20)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=64, le=8192)
    enable_hallucination_check: bool = True
    enable_consistency_check: bool = True
    enable_llm_judge: bool = True


class Experiment(BaseModel):
    """A controlled experiment comparing prompt variants against a test dataset.

    An experiment links:
    - A list of prompt templates (variants to compare)
    - A test dataset (inputs + expected outputs)
    - An evaluation config (how to run and what to measure)
    """
    id: str = Field(default_factory=generate_id)
    name: str = Field(..., min_length=1, max_length=200)
    description: str = ""
    prompt_templates: list[PromptTemplate] = Field(..., min_length=1)
    dataset_id: str = Field(...)
    eval_config: EvaluationConfig = Field(default_factory=EvaluationConfig)
    status: ExperimentStatus = ExperimentStatus.PENDING
    created_at: datetime = Field(default_factory=utc_now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def total_runs(self) -> int:
        """Total number of individual runs this experiment will execute."""
        return len(self.prompt_templates) * self.eval_config.repetitions


# ─── Metric Models ────────────────────────────────────────────────────────


class LatencyMetrics(BaseModel):
    """Timing metrics for a single LLM generation call.

    Attributes:
        total_ms: Total wall-clock time from request to full response.
        tokens_per_second: Generation throughput (completion tokens / generation time).
    """
    total_ms: float = 0.0
    tokens_per_second: float = 0.0


class TokenMetrics(BaseModel):
    """Token usage and efficiency metrics for a single response.

    Attributes:
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens in the generated output.
        total_tokens: Sum of prompt + completion tokens.
        output_input_ratio: completion_tokens / prompt_tokens.
        verbosity_score: completion length vs reference answer length (1.0 = matched).
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    output_input_ratio: float = 0.0
    verbosity_score: float = 1.0


class SemanticScore(BaseModel):
    """Semantic quality assessment of a response relative to the reference answer.

    Attributes:
        embedding_similarity: Cosine similarity between response and reference embeddings.
        judge_relevance: LLM-as-Judge score for answer relevance (1-5).
        judge_correctness: LLM-as-Judge score for factual correctness (1-5).
        judge_coherence: LLM-as-Judge score for coherence and clarity (1-5).
        judge_average: Mean of the three judge scores.
    """
    embedding_similarity: float = 0.0
    judge_relevance: float = 0.0
    judge_correctness: float = 0.0
    judge_coherence: float = 0.0
    judge_average: float = 0.0


class Claim(BaseModel):
    """An atomic factual claim extracted from a response for hallucination checking."""
    text: str
    source_sentence: str


class ClaimVerification(BaseModel):
    """Verification result for a single claim against reference context.

    Attributes:
        claim: The atomic claim being verified.
        embedding_similarity: Cosine similarity to the closest reference passage.
        keyword_overlap_score: Fraction of key terms found in reference context.
        matched_keywords: Keywords found in the reference.
        missing_keywords: Keywords NOT found in the reference.
        overall_confidence: Combined confidence (embedding + keyword weighted average).
        is_hallucination: Whether this claim is classified as hallucinated.
        explanation: Human-readable explanation of the verification decision.
    """
    claim: Claim
    embedding_similarity: float = 0.0
    keyword_overlap_score: float = 1.0
    matched_keywords: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    overall_confidence: float = 0.0
    is_hallucination: bool = False
    explanation: str = ""


class HallucinationReport(BaseModel):
    """Aggregated hallucination analysis for a single response.

    Attributes:
        total_claims: Number of atomic claims extracted.
        verified_claims: Number of claims verified as grounded.
        hallucinated_claims: Number of claims flagged as hallucinated.
        hallucination_rate: Fraction of claims that are hallucinated (0.0 - 1.0).
        claim_verifications: Detailed per-claim verification results.
        overall_confidence: Mean confidence across all claims.
    """
    total_claims: int = 0
    verified_claims: int = 0
    hallucinated_claims: int = 0
    hallucination_rate: float = 0.0
    claim_verifications: list[ClaimVerification] = Field(default_factory=list)
    overall_confidence: float = 1.0


class ConsistencyReport(BaseModel):
    """Cross-run consistency analysis for repeated executions of the same prompt+input.

    Attributes:
        num_runs: Number of runs being compared.
        mean_pairwise_similarity: Average embedding similarity across all run pairs.
        min_pairwise_similarity: Lowest similarity between any two runs.
        semantic_drift: Max deviation from the centroid response embedding.
        agreement_rate: Fraction of run-pairs with similarity above threshold.
    """
    num_runs: int = 0
    mean_pairwise_similarity: float = 1.0
    min_pairwise_similarity: float = 1.0
    semantic_drift: float = 0.0
    agreement_rate: float = 1.0


# ─── Run Results ──────────────────────────────────────────────────────────


class RunResult(BaseModel):
    """Complete result from a single experiment run (one prompt × one test_case × one repetition).

    Contains the raw response plus all evaluation metrics.
    """
    id: str = Field(default_factory=generate_id)
    experiment_id: str
    prompt_template_id: str
    prompt_template_name: str
    test_case_id: str
    repetition: int = 1
    status: RunStatus = RunStatus.SUCCESS

    # Raw output
    prompt_rendered: str = ""
    response_text: str = ""
    error_message: Optional[str] = None

    # Metrics
    latency: LatencyMetrics = Field(default_factory=LatencyMetrics)
    tokens: TokenMetrics = Field(default_factory=TokenMetrics)
    semantic: SemanticScore = Field(default_factory=SemanticScore)
    hallucination: Optional[HallucinationReport] = None
    consistency: Optional[ConsistencyReport] = None

    # Timestamp
    executed_at: datetime = Field(default_factory=utc_now)


# ─── Aggregated Reports ──────────────────────────────────────────────────


class MetricSummary(BaseModel):
    """Statistical summary for a single numeric metric across multiple runs.

    Used by VariantReport to summarize each metric dimension.
    """
    mean: float = 0.0
    median: float = 0.0
    stddev: float = 0.0
    min: float = 0.0
    max: float = 0.0
    p95: float = 0.0


class VariantReport(BaseModel):
    """Aggregated performance report for a single prompt variant across all test cases.

    Groups all RunResults for one prompt template and computes statistical summaries.
    """
    prompt_template_id: str
    prompt_template_name: str
    num_runs: int = 0
    num_successes: int = 0
    num_failures: int = 0

    # Per-dimension summaries
    latency_ms: MetricSummary = Field(default_factory=MetricSummary)
    tokens_per_second: MetricSummary = Field(default_factory=MetricSummary)
    prompt_tokens: MetricSummary = Field(default_factory=MetricSummary)
    completion_tokens: MetricSummary = Field(default_factory=MetricSummary)
    verbosity: MetricSummary = Field(default_factory=MetricSummary)
    embedding_similarity: MetricSummary = Field(default_factory=MetricSummary)
    judge_average: MetricSummary = Field(default_factory=MetricSummary)
    hallucination_rate: MetricSummary = Field(default_factory=MetricSummary)
    consistency_agreement: MetricSummary = Field(default_factory=MetricSummary)

    # Composite
    composite_score: float = 0.0


class WinRateEntry(BaseModel):
    """Head-to-head win rate between two prompt variants."""
    variant_a: str
    variant_b: str
    variant_a_wins: int = 0
    variant_b_wins: int = 0
    ties: int = 0
    total_comparisons: int = 0
    variant_a_win_rate: float = 0.0


class ExperimentReport(BaseModel):
    """Complete comparison report across all prompt variants in an experiment.

    Contains per-variant summaries, head-to-head win rates, and final rankings.
    """
    experiment_id: str
    experiment_name: str
    dataset_name: str
    total_runs: int = 0
    total_test_cases: int = 0

    variant_reports: list[VariantReport] = Field(default_factory=list)
    win_rate_matrix: list[WinRateEntry] = Field(default_factory=list)
    rankings: list[dict[str, Any]] = Field(default_factory=list)

    generated_at: datetime = Field(default_factory=utc_now)


# ─── API Request/Response Models ─────────────────────────────────────────


class CreateExperimentRequest(BaseModel):
    """Request body for creating a new experiment."""
    name: str = Field(..., min_length=1, max_length=200)
    description: str = ""
    prompt_templates: list[PromptTemplate] = Field(..., min_length=1)
    dataset_id: str
    eval_config: EvaluationConfig = Field(default_factory=EvaluationConfig)


class CreateDatasetRequest(BaseModel):
    """Request body for creating a new test dataset."""
    name: str = Field(..., min_length=1, max_length=200)
    description: str = ""
    test_cases: list[TestCase] = Field(..., min_length=1)


class ExperimentSummary(BaseModel):
    """Lightweight experiment summary for listing endpoints."""
    id: str
    name: str
    description: str
    status: ExperimentStatus
    num_variants: int
    dataset_id: str
    created_at: datetime
    completed_at: Optional[datetime] = None


class DatasetSummary(BaseModel):
    """Lightweight dataset summary for listing endpoints."""
    id: str
    name: str
    description: str
    num_test_cases: int
    created_at: datetime
