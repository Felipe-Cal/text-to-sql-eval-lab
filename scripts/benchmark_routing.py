"""
Benchmark model routing vs. a fixed baseline.

Runs the golden dataset three ways and compares quality + cost:
  1. baseline_cheap  — always gpt-4o-mini + zero_shot
  2. baseline_best   — always gpt-4o-mini + few_shot_dynamic
  3. routed          — router picks model + strategy per question

Prints a side-by-side table so you can see whether routing achieves
near-best quality at lower cost than always using the best strategy.

Usage:
  python scripts/benchmark_routing.py

Set HARD_MODEL in .env to escalate hard questions to a stronger model
(e.g. HARD_MODEL=openai/gpt-4o). Without it, the router still routes
to different strategies, just on the same model.
"""

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

RUNS = [
    {"label": "baseline_cheap",  "strategy": PromptStrategy.ZERO_SHOT,        "description": "always gpt-4o-mini + zero_shot"},
    {"label": "baseline_best",   "strategy": PromptStrategy.FEW_SHOT_DYNAMIC,  "description": "always gpt-4o-mini + few_shot_dynamic"},
    {"label": "routed",          "strategy": PromptStrategy.ROUTED,            "description": "router picks strategy per question"},
]

SCORED_METRICS = ["syntax_valid", "execution_ok", "result_match", "semantic_judge"]
COST_METRICS   = ["avg_cost", "avg_latency", "avg_attempts"]


def main() -> None:
    seed_database()
    results = []

    for run in RUNS:
        console.rule(f"[bold blue]{run['label']} — {run['description']}")
        logs = inspect_eval(
            text_to_sql(strategy=run["strategy"]),
            model="openai/gpt-4o-mini",
            log_dir="./logs",
        )
        log = logs[0]
        row = {"label": run["label"]}

        if log.results and log.results.scores:
            for scorer in log.results.scores:
                for metric_name, metric_val in scorer.metrics.items():
                    row[f"{scorer.name}/{metric_name}"] = round(metric_val.value, 3)

        results.append(row)

    # Print comparison table
    console.print()
    console.rule("[bold green]Routing Benchmark — Quality vs Cost")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("run", style="bold")

    # Quality columns
    quality_cols = [f"{m}/accuracy" for m in SCORED_METRICS]
    for col in quality_cols:
        short = col.split("/")[0]
        table.add_column(short, justify="right")

    # Cost columns
    cost_cols = [f"{m}/mean" for m in COST_METRICS]
    cost_labels = {"avg_cost/mean": "cost($)", "avg_latency/mean": "latency(s)", "avg_attempts/mean": "attempts"}
    for col in cost_cols:
        table.add_column(cost_labels.get(col, col), justify="right")

    for row in results:
        cells = [row["label"]]
        for col in quality_cols + cost_cols:
            val = row.get(col, "-")
            if isinstance(val, float):
                cells.append(f"{val:.3f}")
            else:
                cells.append(str(val))
        table.add_row(*cells)

    console.print(table)
    console.print()
    console.print("[dim]Interpretation:[/dim]")
    console.print("  [dim]• routed quality ≈ baseline_best → router is working[/dim]")
    console.print("  [dim]• routed cost < baseline_best    → routing saves money[/dim]")
    console.print("  [dim]• Set HARD_MODEL=openai/gpt-4o in .env to escalate hard questions[/dim]")


if __name__ == "__main__":
    main()
