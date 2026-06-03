"""
CLI Interface for the LLM Evaluation Dashboard.

Provides command-line access to create experiments, run evaluations,
and inspect results without going through the REST API.

Usage:
    python -m src.cli run-experiment --name "QA Prompt Test"
    python -m src.cli list-experiments
    python -m src.cli seed-demo
"""

from __future__ import annotations

import asyncio

import click
from rich.console import Console
from rich.table import Table

from src.execution.result_aggregator import ResultAggregator
from src.execution.runner import ExperimentRunner
from src.models.llm_client import OllamaClient
from src.storage.database import Database

console = Console()


@click.group()
def cli():
    """LLM Evaluation & Prompt Testing Dashboard CLI."""
    pass


@cli.command()
def list_experiments():
    """List all experiments in the database."""
    db = Database()
    experiments = db.list_experiments()
    db.close()

    if not experiments:
        console.print("[yellow]No experiments found.[/yellow]")
        return

    table = Table(title="Experiments")
    table.add_column("ID", style="dim")
    table.add_column("Name", style="bold")
    table.add_column("Status")
    table.add_column("Variants", justify="right")
    table.add_column("Created")

    for exp in experiments:
        status_color = {
            "pending": "yellow",
            "running": "blue",
            "completed": "green",
            "failed": "red",
        }.get(exp.status.value, "white")

        table.add_row(
            exp.id,
            exp.name,
            f"[{status_color}]{exp.status.value}[/{status_color}]",
            str(exp.num_variants),
            exp.created_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)


@cli.command()
def list_datasets():
    """List all test datasets."""
    db = Database()
    datasets = db.list_datasets()
    db.close()

    if not datasets:
        console.print("[yellow]No datasets found.[/yellow]")
        return

    table = Table(title="Test Datasets")
    table.add_column("ID", style="dim")
    table.add_column("Name", style="bold")
    table.add_column("Test Cases", justify="right")
    table.add_column("Created")

    for ds in datasets:
        table.add_row(
            ds.id,
            ds.name,
            str(ds.num_test_cases),
            ds.created_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)


@cli.command()
@click.argument("experiment_id")
def show_results(experiment_id: str):
    """Show results for a completed experiment."""
    db = Database()
    report = db.get_experiment_report(experiment_id)
    db.close()

    if not report:
        console.print(f"[red]No results found for experiment {experiment_id}[/red]")
        return

    # Rankings table
    table = Table(title=f"Rankings — {report.experiment_name}")
    table.add_column("Rank", justify="center")
    table.add_column("Prompt Variant", style="bold")
    table.add_column("Composite Score", justify="right")
    table.add_column("Runs", justify="right")
    table.add_column("Successes", justify="right")

    for r in report.rankings:
        rank_style = "green bold" if r["rank"] == 1 else "white"
        table.add_row(
            f"[{rank_style}]#{r['rank']}[/{rank_style}]",
            r["name"],
            f"{r['composite_score']:.4f}",
            str(r["num_runs"]),
            str(r["num_successes"]),
        )

    console.print(table)

    # Per-variant metrics
    for v in report.variant_reports:
        console.print(f"\n[bold cyan]── {v.prompt_template_name} ──[/bold cyan]")
        metrics_table = Table(show_header=True, box=None)
        metrics_table.add_column("Metric", style="dim")
        metrics_table.add_column("Mean", justify="right")
        metrics_table.add_column("Median", justify="right")
        metrics_table.add_column("P95", justify="right")
        metrics_table.add_column("Stddev", justify="right")

        metrics_table.add_row(
            "Latency (ms)", f"{v.latency_ms.mean:.1f}",
            f"{v.latency_ms.median:.1f}", f"{v.latency_ms.p95:.1f}",
            f"{v.latency_ms.stddev:.1f}",
        )
        metrics_table.add_row(
            "Tokens/sec", f"{v.tokens_per_second.mean:.1f}",
            f"{v.tokens_per_second.median:.1f}", f"{v.tokens_per_second.p95:.1f}",
            f"{v.tokens_per_second.stddev:.1f}",
        )
        metrics_table.add_row(
            "Embedding Sim", f"{v.embedding_similarity.mean:.4f}",
            f"{v.embedding_similarity.median:.4f}", f"{v.embedding_similarity.p95:.4f}",
            f"{v.embedding_similarity.stddev:.4f}",
        )
        metrics_table.add_row(
            "Judge Avg", f"{v.judge_average.mean:.2f}",
            f"{v.judge_average.median:.2f}", f"{v.judge_average.p95:.2f}",
            f"{v.judge_average.stddev:.2f}",
        )

        console.print(metrics_table)

    # Win rate matrix
    if report.win_rate_matrix:
        console.print("\n[bold]Win Rate Matrix[/bold]")
        wr_table = Table()
        wr_table.add_column("Variant A")
        wr_table.add_column("Variant B")
        wr_table.add_column("A Wins", justify="right")
        wr_table.add_column("B Wins", justify="right")
        wr_table.add_column("Ties", justify="right")
        wr_table.add_column("A Win Rate", justify="right")

        for entry in report.win_rate_matrix:
            wr_table.add_row(
                entry.variant_a,
                entry.variant_b,
                str(entry.variant_a_wins),
                str(entry.variant_b_wins),
                str(entry.ties),
                f"{entry.variant_a_win_rate:.1%}",
            )

        console.print(wr_table)


@cli.command()
def seed_demo():
    """Seed the database with a demo dataset and experiment for quick testing."""
    from scripts.seed_sample_experiment import seed

    seed()
    console.print("[green]Demo data seeded successfully![/green]")


@cli.command()
@click.argument("experiment_id")
def run_experiment(experiment_id: str):
    """Run an experiment by ID (blocking, with progress output)."""
    db = Database()
    llm_client = OllamaClient()
    runner = ExperimentRunner(llm_client, db)
    aggregator = ResultAggregator()

    experiment = db.get_experiment(experiment_id)
    if not experiment:
        console.print(f"[red]Experiment {experiment_id} not found[/red]")
        return

    dataset = db.get_dataset(experiment.dataset_id)
    if not dataset:
        console.print(f"[red]Dataset {experiment.dataset_id} not found[/red]")
        return

    console.print(f"[bold]Running experiment:[/bold] {experiment.name}")
    console.print(f"  Variants: {len(experiment.prompt_templates)}")
    console.print(f"  Test cases: {dataset.size}")
    console.print(f"  Repetitions: {experiment.eval_config.repetitions}")
    console.print(f"  Total runs: {len(experiment.prompt_templates) * dataset.size * experiment.eval_config.repetitions}")
    console.print()

    async def _run():
        results = await runner.run(experiment, dataset)
        report = aggregator.aggregate(experiment, dataset, results)
        db.save_experiment_report(report)
        await llm_client.close()
        return report

    asyncio.run(_run())
    db.close()

    console.print("\n[green]Experiment completed![/green]")
    console.print(f"Use [bold]show-results {experiment_id}[/bold] to view the report.")


if __name__ == "__main__":
    cli()
