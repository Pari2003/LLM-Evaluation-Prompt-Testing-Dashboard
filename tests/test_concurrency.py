"""
Concurrency Tests.

Validates that the asyncio.Semaphore-based concurrent execution limits
in-flight tasks correctly and falls back to sequential execution when
max_concurrent_runs is 1.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from src.execution.runner import ExperimentRunner
from src.models.providers.base import LLMProvider
from src.models.schemas import Experiment, PromptTemplate, RunStatus, TestCase
from src.storage.database import Database


class MockDelayProvider(LLMProvider):
    """Mock provider that sleeps to simulate network delay."""

    def __init__(self, delay: float = 0.1):
        self.delay = delay
        self.concurrent_calls = 0
        self.max_observed_concurrent = 0

    async def close(self) -> None:
        pass

    async def generate(self, *args, **kwargs) -> tuple[str, dict[str, Any]]:
        self.concurrent_calls += 1
        self.max_observed_concurrent = max(self.max_observed_concurrent, self.concurrent_calls)

        await asyncio.sleep(self.delay)

        self.concurrent_calls -= 1
        return "mock response", {
            "total_ms": self.delay * 1000,
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "tokens_per_second": 100.0,
        }

    async def generate_json(self, *args, **kwargs) -> dict[str, Any]:
        return {}

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Simulate embedding
        return [[0.1] * 384 for _ in texts]

    async def embed_single(self, text: str) -> list[float]:
        return [0.1] * 384

    async def health_check(self) -> bool:
        return True


async def _run_with_concurrency(max_concurrent: int, total_runs: int) -> float:
    """Helper to run a test experiment and return elapsed time and max concurrency."""
    import src.config

    src.config.settings.max_concurrent_runs = max_concurrent

    from pathlib import Path

    db = Database(Path(":memory:"))
    provider = MockDelayProvider(delay=0.1)
    runner = ExperimentRunner(provider, db)

    # We mock the internals of _execute_single_run to just call generate
    # so we don't have to mock all the evaluators.
    async def mock_execute(*args, **kwargs):
        await provider.generate("test")

        class MockResult:
            status = RunStatus.SUCCESS

        return MockResult()

    runner._execute_single_run = mock_execute

    # 1 prompt * (total_runs) test cases * 1 rep
    template = PromptTemplate(name="t1", template="test")
    test_cases = [
        TestCase(input_variables={"v": "v"}, reference_answer="ref") for _ in range(total_runs)
    ]
    exp = Experiment(
        name="test",
        dataset_id="test",
        prompt_templates=[template],
    )

    # mock config repetition
    class EvalConfig:
        repetitions = 1

    start_time = time.perf_counter()
    await runner._run_variant(exp, template, test_cases, EvalConfig())
    elapsed = time.perf_counter() - start_time

    return elapsed, provider.max_observed_concurrent


def test_sequential_execution():
    """When max_concurrent_runs=1, should run strictly sequentially."""
    elapsed, max_concurrent = asyncio.run(_run_with_concurrency(1, 10))

    # 10 runs * 0.1s = 1.0s minimum
    assert elapsed >= 1.0
    assert max_concurrent == 1


def test_concurrent_execution():
    """When max_concurrent_runs > 1, should run concurrently and respect limits."""
    # 10 runs, max 5 concurrent -> should take 2 "batches" (0.2s minimum)
    elapsed, max_concurrent = asyncio.run(_run_with_concurrency(5, 10))

    assert elapsed < 0.5  # Much faster than the 1.0s sequential
    assert max_concurrent == 5


def main():
    """Run concurrency tests."""
    tests = [
        test_sequential_execution,
        test_concurrent_execution,
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
    print(f"Concurrency Tests: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        exit(1)


if __name__ == "__main__":
    main()
