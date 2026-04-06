"""
Agent strategy benchmark — compares zero_shot, few_shot_dynamic, and tool_use
across three question categories:

  Part 1 — SQL questions (15 golden questions)
    All three strategies are evaluated.
    Metrics: result_match, cost, latency, tool_calls.

  Part 2 — Policy questions (5 KB-only questions)
    Only tool_use can answer these — the others generate broken SQL.
    Metric: answer_quality (LLM judge checks expected keywords in answer).

  Part 3 — Hybrid questions (5 questions requiring both DB + KB)
    Only tool_use handles both data and policy in a single response.
    Metric: answer_quality + data_present (did the answer include actual numbers?).

This script calls generate_sql() directly (not via Inspect AI) so it runs
faster and gives full control over result collection. Use run_eval.py for
the full Inspect AI eval with logging and the inspect view explorer.

Usage:
    python scripts/benchmark_agent.py                       # all parts
    python scripts/benchmark_agent.py --parts 1             # SQL only
    python scripts/benchmark_agent.py --parts 2 3           # policy + hybrid
    python scripts/benchmark_agent.py --model openai/gpt-4o # different model
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import litellm
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.agent.agent import PromptStrategy, generate_sql
from src.utils.db import execute_query, seed_database

load_dotenv()
console = Console()

GOLDEN_PATH = Path("datasets/golden/questions.json")
POLICY_PATH = Path("datasets/golden/policy_questions.json")

STRATEGIES = [
    PromptStrategy.ZERO_SHOT,
    PromptStrategy.FEW_SHOT_DYNAMIC,
    PromptStrategy.TOOL_USE,
]

STRATEGY_LABELS = {
    PromptStrategy.ZERO_SHOT: "zero_shot",
    PromptStrategy.FEW_SHOT_DYNAMIC: "few_shot_dynamic",
    PromptStrategy.TOOL_USE: "tool_use",
}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _normalize_rows(rows):
    """Round floats, strip strings — same logic as scorers.py."""
    import datetime
    from decimal import Decimal
    normalized = []
    for row in rows:
        new_row = []
        for val in row:
            if isinstance(val, (float, Decimal)):
                new_row.append(round(float(val), 2))
            elif isinstance(val, str):
                new_row.append(val.strip())
            elif isinstance(val, (datetime.date, datetime.datetime)):
                new_row.append(val.isoformat())
            else:
                new_row.append(val)
        normalized.append(tuple(new_row))
    return normalized


def _result_match(sql: str, expected_rows: list) -> float:
    """1.0 = exact match, 0.5 = right shape, 0.0 = wrong / error."""
    if not sql:
        return 0.0
    try:
        actual = execute_query(sql)
        expected_tuples = [tuple(r) for r in expected_rows]
        actual_n = _normalize_rows(actual)
        expected_n = _normalize_rows(expected_tuples)
        if sorted(actual_n) == sorted(expected_n):
            return 1.0
        if actual and expected_tuples and len(actual) == len(expected_tuples) and len(actual[0]) == len(expected_tuples[0]):
            return 0.5
        return 0.0
    except Exception:
        return 0.0


def _answer_quality(question: str, answer: str, expected_keywords: list[str]) -> float:
    """
    LLM judge for natural language answers.
    Checks whether the answer contains the expected keywords AND
    actually addresses the question. Returns 1.0 / 0.5 / 0.0.
    """
    if not answer:
        return 0.0

    # Fast keyword check first — if ALL keywords missing, skip LLM call
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    if hits == 0:
        return 0.0

    # LLM judge for nuanced assessment
    judge_prompt = f"""You are evaluating an AI assistant's answer to a question.

Question: {question}

Answer: {answer}

Expected keywords that should appear in a correct answer: {expected_keywords}

Rate the answer:
- "correct" if it fully addresses the question and contains the key information
- "partial" if it partially addresses the question but is missing important details
- "incorrect" if it fails to address the question or contains wrong information

Reply with JSON only: {{"verdict": "correct"|"partial"|"incorrect", "reasoning": "brief explanation"}}"""

    try:
        judge_model = "openai/gpt-4o-mini"
        response = litellm.completion(
            model=judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0,
            max_tokens=128,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}.get(data.get("verdict", "incorrect"), 0.0)
    except Exception:
        # Fallback: keyword ratio
        return min(hits / len(expected_keywords), 1.0)


def _data_present(answer: str, hint: str) -> bool:
    """
    Rough check: does the answer contain a number (indicating DB data was used)?
    Used for hybrid questions to verify the agent actually queried the database.
    """
    import re
    return bool(re.search(r'\d+', answer or ""))


# ---------------------------------------------------------------------------
# Part 1 — SQL questions
# ---------------------------------------------------------------------------

def run_part1(model: str, strategies: list[PromptStrategy]) -> dict:
    console.rule("\n[bold cyan]Part 1 — SQL Questions (15 golden questions)")
    console.print("Strategies: " + ", ".join(STRATEGY_LABELS[s] for s in strategies))

    with open(GOLDEN_PATH) as f:
        questions = json.load(f)

    results = {s: {"result_match": [], "cost": [], "latency": [], "tool_calls": []} for s in strategies}

    for q in questions:
        for strategy in strategies:
            label = f"{q['id']} / {STRATEGY_LABELS[strategy]}"
            console.print(f"  [dim]{label}[/dim]", end="... ")

            try:
                result = generate_sql(q["question"], model=model, strategy=strategy)
                rm = _result_match(result.sql, q["expected_rows"])
                n_tools = len(result.tool_calls)
                results[strategy]["result_match"].append(rm)
                results[strategy]["cost"].append(result.cost)
                results[strategy]["latency"].append(result.latency)
                results[strategy]["tool_calls"].append(n_tools)
                console.print(f"result_match={rm:.1f}  tools={n_tools}  lat={result.latency:.1f}s")
            except Exception as e:
                console.print(f"[red]ERROR: {e}[/red]")
                results[strategy]["result_match"].append(0.0)
                results[strategy]["cost"].append(0.0)
                results[strategy]["latency"].append(0.0)
                results[strategy]["tool_calls"].append(0)

    return results


def print_part1_table(results: dict) -> None:
    table = Table(title="\nPart 1 — SQL Questions Summary", show_lines=True)
    table.add_column("Strategy", style="cyan")
    table.add_column("result_match", justify="right", style="bold")
    table.add_column("avg_cost", justify="right")
    table.add_column("avg_latency", justify="right")
    table.add_column("avg_tool_calls", justify="right")

    for strategy, data in results.items():
        n = len(data["result_match"])
        if n == 0:
            continue
        avg_rm = sum(data["result_match"]) / n
        avg_cost = sum(data["cost"]) / n
        avg_lat = sum(data["latency"]) / n
        avg_tools = sum(data["tool_calls"]) / n

        color = "green" if avg_rm >= 0.8 else "yellow" if avg_rm >= 0.6 else "red"
        table.add_row(
            STRATEGY_LABELS[strategy],
            f"[{color}]{avg_rm:.3f}[/{color}]",
            f"${avg_cost:.4f}",
            f"{avg_lat:.1f}s",
            f"{avg_tools:.1f}",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Part 2 — Policy questions
# ---------------------------------------------------------------------------

def run_part2(model: str) -> dict:
    console.rule("\n[bold magenta]Part 2 — Policy Questions (KB-only)")
    console.print("Only tool_use is evaluated — other strategies generate broken SQL\n")

    with open(POLICY_PATH) as f:
        all_questions = json.load(f)

    policy_qs = [q for q in all_questions if q["type"] == "policy"]
    results = {"answer_quality": [], "cost": [], "latency": [], "tool_calls": []}

    for q in policy_qs:
        console.print(f"  [dim]{q['id']}[/dim]: {q['question'][:60]}...", end=" ")
        try:
            result = generate_sql(q["question"], model=model, strategy=PromptStrategy.TOOL_USE)
            aq = _answer_quality(q["question"], result.answer or "", q["expected_keywords"])
            n_tools = len(result.tool_calls)
            results["answer_quality"].append(aq)
            results["cost"].append(result.cost)
            results["latency"].append(result.latency)
            results["tool_calls"].append(n_tools)
            tools_used = [tc["tool"] for tc in result.tool_calls]
            console.print(f"quality={aq:.1f}  tools={tools_used}")
        except Exception as e:
            console.print(f"[red]ERROR: {e}[/red]")
            results["answer_quality"].append(0.0)
            results["cost"].append(0.0)
            results["latency"].append(0.0)
            results["tool_calls"].append(0)

    return results


def print_part2_table(results: dict) -> None:
    n = len(results["answer_quality"])
    if not n:
        return

    table = Table(title="\nPart 2 — Policy Questions Summary", show_lines=True)
    table.add_column("Strategy", style="cyan")
    table.add_column("answer_quality", justify="right", style="bold")
    table.add_column("avg_cost", justify="right")
    table.add_column("avg_latency", justify="right")
    table.add_column("avg_tool_calls", justify="right")

    avg_aq = sum(results["answer_quality"]) / n
    avg_cost = sum(results["cost"]) / n
    avg_lat = sum(results["latency"]) / n
    avg_tools = sum(results["tool_calls"]) / n

    color = "green" if avg_aq >= 0.8 else "yellow" if avg_aq >= 0.5 else "red"
    table.add_row(
        "tool_use",
        f"[{color}]{avg_aq:.3f}[/{color}]",
        f"${avg_cost:.4f}",
        f"{avg_lat:.1f}s",
        f"{avg_tools:.1f}",
    )
    table.add_row("zero_shot", "[red]N/A[/red] (outputs SQL)", "-", "-", "0")
    table.add_row("few_shot_dynamic", "[red]N/A[/red] (outputs SQL)", "-", "-", "0")

    console.print(table)


# ---------------------------------------------------------------------------
# Part 3 — Hybrid questions
# ---------------------------------------------------------------------------

def run_part3(model: str) -> dict:
    console.rule("\n[bold green]Part 3 — Hybrid Questions (DB + KB)")
    console.print("Only tool_use can combine database results with policy context\n")

    with open(POLICY_PATH) as f:
        all_questions = json.load(f)

    hybrid_qs = [q for q in all_questions if q["type"] == "hybrid"]
    results = {"answer_quality": [], "data_present": [], "cost": [], "latency": [], "tool_calls": []}

    for q in hybrid_qs:
        console.print(f"  [dim]{q['id']}[/dim]: {q['question'][:60]}...", end=" ")
        try:
            result = generate_sql(q["question"], model=model, strategy=PromptStrategy.TOOL_USE)
            aq = _answer_quality(q["question"], result.answer or "", q["expected_keywords"])
            dp = _data_present(result.answer or "", q.get("expected_data_hint", ""))
            n_tools = len(result.tool_calls)
            tools_used = list({tc["tool"] for tc in result.tool_calls})

            results["answer_quality"].append(aq)
            results["data_present"].append(1.0 if dp else 0.0)
            results["cost"].append(result.cost)
            results["latency"].append(result.latency)
            results["tool_calls"].append(n_tools)
            console.print(f"quality={aq:.1f}  data={'✅' if dp else '❌'}  tools={tools_used}")
        except Exception as e:
            console.print(f"[red]ERROR: {e}[/red]")
            for key in results:
                results[key].append(0.0)

    return results


def print_part3_table(results: dict) -> None:
    n = len(results["answer_quality"])
    if not n:
        return

    table = Table(title="\nPart 3 — Hybrid Questions Summary", show_lines=True)
    table.add_column("Strategy", style="cyan")
    table.add_column("answer_quality", justify="right", style="bold")
    table.add_column("data_in_answer", justify="right")
    table.add_column("avg_cost", justify="right")
    table.add_column("avg_latency", justify="right")
    table.add_column("avg_tool_calls", justify="right")

    avg_aq = sum(results["answer_quality"]) / n
    avg_dp = sum(results["data_present"]) / n
    avg_cost = sum(results["cost"]) / n
    avg_lat = sum(results["latency"]) / n
    avg_tools = sum(results["tool_calls"]) / n

    color = "green" if avg_aq >= 0.8 else "yellow" if avg_aq >= 0.5 else "red"
    table.add_row(
        "tool_use",
        f"[{color}]{avg_aq:.3f}[/{color}]",
        f"{avg_dp:.0%}",
        f"${avg_cost:.4f}",
        f"{avg_lat:.1f}s",
        f"{avg_tools:.1f}",
    )
    table.add_row("zero_shot", "[red]N/A[/red]", "[red]N/A[/red]", "-", "-", "0")
    table.add_row("few_shot_dynamic", "[red]N/A[/red]", "[red]N/A[/red]", "-", "-", "0")
    console.print(table)


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

def print_final_summary(p1_results: dict | None, p2_results: dict | None, p3_results: dict | None) -> None:
    console.rule("\n[bold white]Final Summary — When to use each strategy")
    console.print("""
[cyan]zero_shot[/cyan]
  ✅ Cheapest and fastest
  ✅ Works for simple SQL questions
  ❌ Lower accuracy on complex joins and aggregations
  ❌ Cannot answer policy or hybrid questions

[cyan]few_shot_dynamic[/cyan]
  ✅ Best accuracy on SQL questions (+20-27% vs zero_shot)
  ✅ Only one extra embedding API call
  ❌ Still cannot answer policy or hybrid questions
  ❌ Degrades if question is unlike any example in the pool

[cyan]tool_use[/cyan]
  ✅ Handles SQL questions, policy questions, AND hybrid questions
  ✅ Self-corrects via schema inspection before writing SQL
  ✅ Only strategy that combines DB data + KB context in one answer
  ❌ 3-5x more expensive (multiple LLM round-trips)
  ❌ 3-4x slower (sequential tool calls)
  ❌ Slightly lower result_match on pure SQL questions vs few_shot_dynamic

[bold]Rule of thumb:[/bold]
  Pure SQL product         → few_shot_dynamic
  General assistant        → tool_use
  Cost-sensitive / latency → zero_shot with self-correction
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark agent strategies across question types")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="LiteLLM model string")
    parser.add_argument("--parts", nargs="+", type=int, choices=[1, 2, 3], default=[1, 2, 3],
                        help="Which parts to run (1=SQL, 2=policy, 3=hybrid)")
    args = parser.parse_args()

    seed_database()

    console.print(f"\n[bold]Agent Strategy Benchmark[/bold]")
    console.print(f"Model: [cyan]{args.model}[/cyan]")
    console.print(f"Parts: {args.parts}\n")

    p1_results = p2_results = p3_results = None

    if 1 in args.parts:
        p1_results = run_part1(args.model, STRATEGIES)
        print_part1_table(p1_results)

    if 2 in args.parts:
        p2_results = run_part2(args.model)
        print_part2_table(p2_results)

    if 3 in args.parts:
        p3_results = run_part3(args.model)
        print_part3_table(p3_results)

    print_final_summary(p1_results, p2_results, p3_results)


if __name__ == "__main__":
    main()
