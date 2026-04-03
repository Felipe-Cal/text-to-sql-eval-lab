"""
Prepare fine-tuning data for Llama 3.1 8B.

Converts the golden dataset + synthetic tuning set into the JSONL chat format
expected by Unsloth / HuggingFace TRL's SFTTrainer.

Each example becomes a three-turn conversation:
  system    — SQL expert instruction
  user      — schema + question
  assistant — the correct SQL query

Sources (combined and deduplicated):
  datasets/golden/questions.json         (15 hand-crafted Q&A pairs)
  datasets/synthetic/tuning.json         (40 synthetic Q&A pairs)

Output:
  datasets/finetune/train.jsonl          (90% of combined, shuffled)
  datasets/finetune/eval.jsonl           (10% held out for training-time eval)

Usage:
  python scripts/prepare_finetune_data.py
  python scripts/prepare_finetune_data.py --no-golden   # skip golden set
  python scripts/prepare_finetune_data.py --eval-split 0.15
"""

import argparse
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GOLDEN_PATH   = Path("datasets/golden/questions.json")
TUNING_PATH   = Path("datasets/synthetic/tuning.json")
OUT_DIR       = Path("datasets/finetune")

# ---------------------------------------------------------------------------
# System prompt — same rules as the main agent so the model learns the
# exact output format the eval harness expects
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert SQL engineer. Convert the natural language question into a \
single DuckDB SQL query.

Rules:
- Output ONLY the SQL query — no explanation, no markdown fences, no commentary.
- Use the exact table and column names from the schema provided.
- Use DuckDB SQL dialect (supports YEAR(), MONTH(), ROUND(), standard window functions).
- Do not use CTEs unless necessary — prefer subqueries for clarity.
- Always alias aggregated columns with descriptive names.\
"""

# Minimal schema — the 4 core tables only. The model learns to ignore noise
# during fine-tuning, so we don't need the 50-table DWH here.
SCHEMA = """\
customers(id, name, email, country, signup_date DATE)
products(id, name, category, price DECIMAL)
orders(id, customer_id, order_date DATE, status VARCHAR)  -- status: completed | pending | cancelled
order_items(id, order_id, product_id, quantity INTEGER, unit_price DECIMAL)

Relationships:
  orders.customer_id → customers.id
  order_items.order_id → orders.id
  order_items.product_id → products.id\
"""


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def to_chat_example(question: str, sql: str) -> dict:
    """Convert a Q&A pair into the HuggingFace messages chat format."""
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Schema:\n{SCHEMA}\n\nQuestion: {question}"},
            {"role": "assistant", "content": sql.strip()},
        ]
    }


def load_golden() -> list[dict]:
    with open(GOLDEN_PATH) as f:
        data = json.load(f)
    return [{"question": q["question"], "sql": q["sql"]} for q in data]


def load_tuning() -> list[dict]:
    with open(TUNING_PATH) as f:
        data = json.load(f)
    return [{"question": q["question"], "sql": q["sql"]} for q in data]


def write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  Wrote {len(examples)} examples → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fine-tuning JSONL data")
    parser.add_argument("--no-golden",  action="store_true", help="Exclude golden dataset")
    parser.add_argument("--eval-split", type=float, default=0.10, help="Fraction for eval set (default: 0.10)")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load and combine sources
    pairs: list[dict] = []
    if not args.no_golden:
        golden = load_golden()
        pairs.extend(golden)
        print(f"Loaded {len(golden)} golden examples")

    tuning = load_tuning()
    pairs.extend(tuning)
    print(f"Loaded {len(tuning)} synthetic tuning examples")

    # Deduplicate by question text
    seen: set[str] = set()
    unique_pairs = []
    for p in pairs:
        if p["question"] not in seen:
            seen.add(p["question"])
            unique_pairs.append(p)

    print(f"Total unique examples: {len(unique_pairs)}")

    # Shuffle and split
    random.shuffle(unique_pairs)
    split_idx = max(1, int(len(unique_pairs) * (1 - args.eval_split)))
    train_pairs = unique_pairs[:split_idx]
    eval_pairs  = unique_pairs[split_idx:]

    # Convert to chat format
    train_examples = [to_chat_example(p["question"], p["sql"]) for p in train_pairs]
    eval_examples  = [to_chat_example(p["question"], p["sql"]) for p in eval_pairs]

    # Write JSONL
    print(f"\nSplit: {len(train_examples)} train / {len(eval_examples)} eval")
    write_jsonl(OUT_DIR / "train.jsonl", train_examples)
    write_jsonl(OUT_DIR / "eval.jsonl",  eval_examples)

    # Preview first example
    print("\n--- First training example ---")
    ex = train_examples[0]
    for msg in ex["messages"]:
        role = msg["role"].upper()
        content = msg["content"][:120].replace("\n", " ")
        print(f"  [{role}] {content}...")

    print(f"\nReady for fine-tuning. Upload datasets/finetune/ to Colab.")


if __name__ == "__main__":
    main()
