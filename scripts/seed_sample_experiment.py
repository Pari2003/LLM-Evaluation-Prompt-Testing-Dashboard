"""
Seed Sample Experiment.

Creates a demo dataset and experiment in the database for quick testing.
Seeds 3 prompt variants with different styles (concise, detailed, chain-of-thought)
against the factual Q&A dataset.

Usage:
    python -m scripts.seed_sample_experiment
"""

from __future__ import annotations

import json
from pathlib import Path

from src.models.schemas import (
    EvaluationConfig,
    Experiment,
    PromptTemplate,
    TestCase,
    TestDataset,
)
from src.storage.database import Database


def seed():
    """Seed the database with demo data."""
    db = Database()

    # ─── Create Test Dataset ──────────────────────────────────────────────
    data_path = Path("data/sample_datasets/qa_factual.json")
    with open(data_path) as f:
        raw_cases = json.load(f)

    test_cases = [TestCase(**tc) for tc in raw_cases]
    dataset = TestDataset(
        name="LLM Factual QA Benchmark",
        description="15 factual Q&A pairs covering LLM/AI concepts with ground-truth answers and reference contexts.",
        test_cases=test_cases,
    )
    db.save_dataset(dataset)
    print(f"  Created dataset: {dataset.name} ({dataset.size} test cases)")
    print(f"  Dataset ID: {dataset.id}")

    # ─── Create Prompt Templates ──────────────────────────────────────────

    # Variant 1: Concise (direct, minimal instructions)
    concise = PromptTemplate(
        name="concise",
        template="Answer the following question concisely in 1-2 sentences.\n\nQuestion: {question}\nAnswer:",
        system_prompt="You are a helpful AI assistant. Give concise, accurate answers.",
        variables=["question"],
        metadata={"style": "concise", "expected_behavior": "short, direct answers"},
    )

    # Variant 2: Detailed (encourage thorough explanation)
    detailed = PromptTemplate(
        name="detailed",
        template=(
            "Please provide a comprehensive and detailed answer to the following question. "
            "Include relevant technical details, examples, and context.\n\n"
            "Question: {question}\n\nDetailed Answer:"
        ),
        system_prompt=(
            "You are a knowledgeable AI technical expert. Provide thorough, well-structured "
            "answers with technical depth. Include relevant examples and context."
        ),
        variables=["question"],
        metadata={"style": "detailed", "expected_behavior": "thorough, technical answers"},
    )

    # Variant 3: Chain-of-Thought (step-by-step reasoning)
    cot = PromptTemplate(
        name="chain-of-thought",
        template=(
            "Answer the following question step by step. First, break down the key concepts, "
            "then explain each one, and finally synthesize your answer.\n\n"
            "Question: {question}\n\n"
            "Let me think through this step by step:\n"
        ),
        system_prompt=(
            "You are an AI assistant that reasons carefully. Always break down your thinking "
            "into clear steps before giving your final answer."
        ),
        variables=["question"],
        metadata={"style": "chain-of-thought", "expected_behavior": "step-by-step reasoning"},
    )

    # ─── Create Experiment ────────────────────────────────────────────────
    experiment = Experiment(
        name="QA Prompt Style Comparison",
        description=(
            "Compare three prompting styles (concise, detailed, chain-of-thought) "
            "on factual LLM/AI questions. Evaluates which style produces the most "
            "accurate, efficient, and consistent responses."
        ),
        prompt_templates=[concise, detailed, cot],
        dataset_id=dataset.id,
        eval_config=EvaluationConfig(
            repetitions=2,
            temperature=0.1,
            max_tokens=1024,
            enable_hallucination_check=True,
            enable_consistency_check=True,
            enable_llm_judge=True,
        ),
    )
    db.save_experiment(experiment)
    print(f"  Created experiment: {experiment.name}")
    print(f"  Experiment ID: {experiment.id}")
    print(f"  Variants: {', '.join(t.name for t in experiment.prompt_templates)}")
    print(f"  Total runs: {len(experiment.prompt_templates) * dataset.size * experiment.eval_config.repetitions}")

    db.close()
    print("\n  Seed complete! Run the experiment with:")
    print(f"    python -m src.cli run-experiment {experiment.id}")


if __name__ == "__main__":
    seed()
