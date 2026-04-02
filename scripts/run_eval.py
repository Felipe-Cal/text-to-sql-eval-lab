"""
Run the text-to-SQL evaluation and print a results table.

Usage:
  python scripts/run_eval.py
  python scripts/run_eval.py --model anthropic/claude-3-haiku-20240307
  python scripts/run_eval.py --model openai/gpt-4o --difficulty easy
  python scripts/run_eval.py --models openai/gpt-4o-mini anthropic/claude-3-haiku-20240307

This wraps `inspect eval` for convenience with extra pretty-printing.
For full Inspect AI options (log viewer, etc.) run directly:
  inspect eval src/evals/tasks.py --model openai/gpt-4o-mini
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from inspect_ai import eval as inspect_eval
from rich.console import Console
from rich.table import Table

from src.evals.tasks import text_to_sql
from src.utils.db import seed_database

load_dotenv()
console = Console()


def run(models: list[str], difficulty: str | None) -> None:
    # Ensure DB exists
    seed_database()

    all_results = []

    for model in models:
        console.rule(f"[bold blue]Evaluating: {model}")

        logs = inspect_eval(
            text_to_sql(model=model, difficulty=difficulty),
            model=model,
            log_dir="./logs",
        )

        log = logs[0]
        results = log.results

        row = {"model": model}
        if results and results.scores:
            for scorer in results.scores:
                for metric_name, metric_val in scorer.metrics.items():
                    row[f"{scorer.name}/{metric_name}"] = round(metric_val.value, 3)

        all_results.append(row)

    # Print summary table
    console.print()
    console.rule("[bold green]Results Summary")

    if not all_results:
        console.print("No results.")
        return

    columns = list(all_results[0].keys())
    table = Table(show_header=True, header_style="bold magenta")
    for col in columns:
        table.add_column(col, justify="right" if col != "model" else "left")

    for row in all_results:
        table.add_row(*[str(row.get(col, "-")) for col in columns])

    console.print(table)
    console.print()
    console.print(
        "[dim]Full logs saved to ./logs — run [bold]inspect view[/bold] to explore them.[/dim]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run text-to-SQL eval")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model", type=str, help="Single model to evaluate")
    group.add_argument("--models", nargs="+", help="Multiple models to compare")
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default=None,
        help="Filter dataset by difficulty (default: all)",
    )
    args = parser.parse_args()

    models = args.models or [args.model or "openai/gpt-4o-mini"]
    run(models=models, difficulty=args.difficulty)


if __name__ == "__main__":
    main()
