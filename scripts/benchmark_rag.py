"""
RAG retrieval benchmark — compares chunking strategies and vector store backends.

Measures retrieval_recall@K: for each evaluation question, what fraction of the
"required" answer snippets appear in the top-K retrieved chunks?

This is the retrieval-side equivalent of result_match in the SQL eval: a metric
that is independent of LLM quality and isolates the retrieval pipeline itself.

Usage:
    python scripts/benchmark_rag.py                        # default sweep
    python scripts/benchmark_rag.py --top-k 3 5 10        # custom top-K values
    python scripts/benchmark_rag.py --store chroma         # ChromaDB backend only
    python scripts/benchmark_rag.py --chunker sentence     # single chunker
"""

import argparse
import sys
import time
from pathlib import Path
from itertools import product as cartesian_product

from rich.console import Console
from rich.table import Table

# Ensure project root is on sys.path when run as:
#   python scripts/benchmark_rag.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.retriever import build_retriever

console = Console()

# ---------------------------------------------------------------------------
# Evaluation dataset
# Each entry has a question and a list of required substrings that must appear
# in at least one of the top-K retrieved chunks to count as a "hit".
# ---------------------------------------------------------------------------
EVAL_QUESTIONS = [
    {
        "question": "What is the return window for purchased products?",
        "required": ["30 days"],
    },
    {
        "question": "How long does a refund take to process?",
        "required": ["5 to 7 business days", "5-7 business days"],
    },
    {
        "question": "Do I get free shipping on large orders?",
        "required": ["$50", "free standard shipping"],
    },
    {
        "question": "What payment methods are accepted?",
        "required": ["PayPal", "Visa", "Mastercard"],
    },
    {
        "question": "Can I return a digital product?",
        "required": ["non-refundable", "Digital products"],
    },
    {
        "question": "How do I contact customer support?",
        "required": ["support@example.com", "1-800-555-0100"],
    },
    {
        "question": "What happens if my payment fails?",
        "required": ["24 hours", "automatically cancelled"],
    },
    {
        "question": "How quickly does the support team respond?",
        "required": ["2 hours", "business hours"],
    },
]

DOCS_DIR = Path("datasets/docs")
KB_FILE = DOCS_DIR / "ecommerce_kb.md"


def recall_at_k(retrieved_texts: list[str], required: list[str]) -> float:
    """
    Returns 1.0 if ANY of the required substrings appears in ANY retrieved chunk.
    Returns 0.0 otherwise.

    Using OR logic across required phrases (they're alternatives, e.g. "5 to 7
    business days" vs "5-7 business days") and OR logic across chunks.
    """
    combined = " ".join(retrieved_texts).lower()
    return 1.0 if any(r.lower() in combined for r in required) else 0.0


def run_benchmark(
    chunkers: list[str],
    stores: list[str],
    top_ks: list[int],
    chunk_size: int = 400,
    overlap: int = 1,
) -> list[dict]:
    results = []

    configs = list(cartesian_product(chunkers, stores, top_ks))
    console.print(f"\n[bold]Running {len(configs)} configurations × {len(EVAL_QUESTIONS)} questions[/bold]\n")

    for chunker_name, store_name, top_k in configs:
        label = f"{chunker_name} / {store_name} / top_k={top_k}"
        console.print(f"  Benchmarking [cyan]{label}[/cyan]...", end=" ")

        # Build and index
        t0 = time.time()
        retriever = build_retriever(
            chunker=chunker_name,
            store=store_name,
            chunk_size=chunk_size,
            overlap=overlap,
            collection_name=f"bench_{chunker_name}_{store_name}",
            persist_dir="./chroma_db_bench",
        )
        n_chunks = retriever.index_file(KB_FILE)
        index_time = time.time() - t0

        # Evaluate
        t1 = time.time()
        recalls = []
        for item in EVAL_QUESTIONS:
            retrieved = retriever.retrieve(item["question"], top_k=top_k)
            texts = [r.chunk.text for r in retrieved]
            recalls.append(recall_at_k(texts, item["required"]))
        query_time = time.time() - t1

        avg_recall = sum(recalls) / len(recalls)
        console.print(f"recall={avg_recall:.3f}  chunks={n_chunks}  idx={index_time:.1f}s  qry={query_time:.1f}s")

        results.append({
            "chunker": chunker_name,
            "store": store_name,
            "top_k": top_k,
            "n_chunks": n_chunks,
            "recall": avg_recall,
            "index_time": index_time,
            "query_time": query_time,
            "per_question": recalls,
        })

    return results


def print_results_table(results: list[dict]) -> None:
    table = Table(title="\nRAG Benchmark Results", show_lines=True)
    table.add_column("Chunker", style="cyan")
    table.add_column("Store", style="magenta")
    table.add_column("top_k", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Recall@K", justify="right", style="bold")
    table.add_column("Idx time", justify="right")
    table.add_column("Qry time", justify="right")

    # Sort by recall descending
    for r in sorted(results, key=lambda x: x["recall"], reverse=True):
        recall_str = f"{r['recall']:.3f}"
        color = "green" if r["recall"] >= 0.75 else "yellow" if r["recall"] >= 0.5 else "red"
        table.add_row(
            r["chunker"],
            r["store"],
            str(r["top_k"]),
            str(r["n_chunks"]),
            f"[{color}]{recall_str}[/{color}]",
            f"{r['index_time']:.1f}s",
            f"{r['query_time']:.1f}s",
        )

    console.print(table)

    # Per-question breakdown for the best config
    best = max(results, key=lambda x: x["recall"])
    console.print(f"\n[bold]Per-question breakdown for best config "
                  f"({best['chunker']} / {best['store']} / top_k={best['top_k']}):[/bold]")
    for item, recall in zip(EVAL_QUESTIONS, best["per_question"]):
        icon = "✅" if recall == 1.0 else "❌"
        console.print(f"  {icon}  {item['question']}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark RAG chunking strategies and vector stores")
    parser.add_argument("--chunker", nargs="+", default=["sentence", "fixed"],
                        choices=["sentence", "fixed", "schema"],
                        help="Chunking strategies to compare")
    parser.add_argument("--store", nargs="+", default=["memory"],
                        choices=["memory", "chroma"],
                        help="Vector store backends to compare")
    parser.add_argument("--top-k", nargs="+", type=int, default=[3, 5],
                        dest="top_k",
                        help="Top-K values to sweep over")
    parser.add_argument("--chunk-size", type=int, default=400,
                        help="Max chunk size in characters (fixed) or sentences window (sentence)")
    parser.add_argument("--overlap", type=int, default=1,
                        help="Overlap (characters for fixed, sentences for sentence chunker)")
    args = parser.parse_args()

    console.print(f"[bold]RAG Benchmark[/bold]")
    console.print(f"Document: {KB_FILE} ({KB_FILE.stat().st_size // 1024} KB)")
    console.print(f"Questions: {len(EVAL_QUESTIONS)}")
    console.print(f"Chunkers: {args.chunker}")
    console.print(f"Stores: {args.store}")
    console.print(f"Top-K: {args.top_k}")

    results = run_benchmark(
        chunkers=args.chunker,
        stores=args.store,
        top_ks=args.top_k,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    print_results_table(results)


if __name__ == "__main__":
    main()
