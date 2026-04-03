"""
CI evaluation gate — runs evals and fails if scores fall below thresholds.

Designed to be called from GitHub Actions (or any CI system). Exits with
code 1 if any threshold is breached so the pipeline blocks the merge.

Usage:
  python scripts/ci_eval.py
  python scripts/ci_eval.py --strategy few_shot_dynamic
  python scripts/ci_eval.py --strategy zero_shot --syntax-valid 1.0 --result-match 0.70

Thresholds (defaults reflect the zero_shot baseline from experiments):
  --syntax-valid   minimum syntax_valid accuracy  (default: 0.95)
  --execution-ok   minimum execution_ok accuracy  (default: 0.85)
  --result-match   minimum result_match accuracy  (default: 0.70)
  --semantic-judge minimum semantic_judge accuracy (default: 0.80)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from inspect_ai import eval as inspect_eval

from src.agent.agent import PromptStrategy
from src.evals.tasks import text_to_sql
from src.utils.db import seed_database

load_dotenv()

# Conservative thresholds based on zero_shot baseline results.
# They're intentionally set slightly below baseline so CI catches real
# regressions without being brittle to natural LLM variance.
DEFAULT_THRESHOLDS = {
    "syntax_valid": 0.95,
    "execution_ok": 0.85,
    "result_match": 0.70,
    "semantic_judge": 0.80,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="CI eval gate for text-to-SQL")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--strategy", default="zero_shot", choices=[s.value for s in PromptStrategy])
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument("--syntax-valid", type=float, default=DEFAULT_THRESHOLDS["syntax_valid"])
    parser.add_argument("--execution-ok", type=float, default=DEFAULT_THRESHOLDS["execution_ok"])
    parser.add_argument("--result-match", type=float, default=DEFAULT_THRESHOLDS["result_match"])
    parser.add_argument("--semantic-judge", type=float, default=DEFAULT_THRESHOLDS["semantic_judge"])
    args = parser.parse_args()

    thresholds = {
        "syntax_valid": args.syntax_valid,
        "execution_ok": args.execution_ok,
        "result_match": args.result_match,
        "semantic_judge": args.semantic_judge,
    }

    print(f"\n=== CI Eval Gate ===")
    print(f"Model:    {args.model}")
    print(f"Strategy: {args.strategy}")
    print(f"Thresholds: {thresholds}\n")

    seed_database()

    logs = inspect_eval(
        text_to_sql(
            model=args.model,
            difficulty=args.difficulty,
            strategy=PromptStrategy(args.strategy),
        ),
        model=args.model,
        log_dir="./logs",
    )

    log = logs[0]
    if log.status == "error":
        print(f"\n❌ Eval run failed: {log.error}")
        sys.exit(1)

    # Extract accuracy scores from results
    scores: dict[str, float] = {}
    if log.results and log.results.scores:
        for scorer in log.results.scores:
            if "accuracy" in scorer.metrics:
                scores[scorer.name] = round(scorer.metrics["accuracy"].value, 4)

    print("=== Results ===")
    failures = []
    for metric, threshold in thresholds.items():
        actual = scores.get(metric)
        if actual is None:
            print(f"  {metric}: NOT FOUND")
            failures.append(f"{metric}: scorer not found in results")
            continue

        status = "✅" if actual >= threshold else "❌"
        print(f"  {status} {metric}: {actual:.3f} (threshold: {threshold:.3f})")
        if actual < threshold:
            failures.append(f"{metric}: {actual:.3f} < {threshold:.3f}")

    print()
    if failures:
        print("CI FAILED — the following thresholds were breached:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("CI PASSED — all thresholds met.")
        sys.exit(0)


if __name__ == "__main__":
    main()
