"""
Run the text-to-SQL evaluation and print a results table.

Usage:
  # Default: gpt-4o-mini, zero-shot, all 15 questions
  python scripts/run_eval.py

  # Specific model
  python scripts/run_eval.py --model anthropic/claude-3-haiku-20240307

  # Specific prompt strategy
  python scripts/run_eval.py --strategy few_shot_static
  python scripts/run_eval.py --strategy few_shot_dynamic

  # Compare all three strategies side by side
  python scripts/run_eval.py --strategies zero_shot few_shot_static few_shot_dynamic

  # Compare multiple models
  python scripts/run_eval.py --models openai/gpt-4o-mini anthropic/claude-3-haiku-20240307

  # Filter by difficulty
  python scripts/run_eval.py --model openai/gpt-4o --difficulty hard

This wraps `inspect eval` for convenience with extra pretty-printing.
For full Inspect AI options (log viewer, etc.) run directly:
  inspect eval src/evals/tasks.py --model openai/gpt-4o-mini
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from inspect_ai import eval as inspect_eval
from rich.console import Console
from rich.table import Table

from src.agent.agent import PromptStrategy
from src.evals.tasks import text_to_sql
from src.utils.db import seed_database

load_dotenv()
console = Console()

STRATEGY_CHOICES = [s.value for s in PromptStrategy]


def normalize_model_name(model: str) -> str:
    """
    Normalize common provider aliases to the API prefix expected by Inspect AI.
    """
    if model.startswith("gemini/"):
        return model.replace("gemini/", "google/", 1)
    return model


def run(
    models: list[str],
    strategies: list[str],
    difficulty: str | None,
) -> None:
    # Ensure DB exists
    seed_database()

    all_results = []
    comparing_strategies = len(strategies) > 1

    for model in models:
        inspect_model = normalize_model_name(model)
        if inspect_model != model:
            console.print(
                f"[dim]Normalizing Inspect model alias: {model} -> {inspect_model}[/dim]"
            )

        for strategy_str in strategies:
            strategy = PromptStrategy(strategy_str)

            # Row label: include strategy only when comparing more than one
            label_model = model if inspect_model == model else f"{model} (inspect: {inspect_model})"
            label = f"{label_model} [{strategy_str}]" if comparing_strategies else label_model

            console.rule(f"[bold blue]Evaluating: {label}")

            logs = inspect_eval(
                text_to_sql(model=model, difficulty=difficulty, strategy=strategy),
                model=inspect_model,
                log_dir="./logs",
            )

            log = logs[0]
            results = log.results

            row = {"model": label}
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

    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", type=str, help="Single model to evaluate")
    model_group.add_argument("--models", nargs="+", help="Multiple models to compare")

    strategy_group = parser.add_mutually_exclusive_group()
    strategy_group.add_argument(
        "--strategy",
        choices=STRATEGY_CHOICES,
        default=None,
        help="Prompt strategy to use (default: zero_shot)",
    )
    strategy_group.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGY_CHOICES,
        help="Multiple strategies to compare side by side",
    )

    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default=None,
        help="Filter dataset by difficulty (default: all)",
    )

    args = parser.parse_args()

    models = args.models or [args.model or "openai/gpt-4o-mini"]
    strategies = args.strategies or [args.strategy or "zero_shot"]

    run(models=models, strategies=strategies, difficulty=args.difficulty)


if __name__ == "__main__":
    main()
