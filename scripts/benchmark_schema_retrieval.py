"""
Schema retrieval benchmark — compares vector DB backends, embedding models, and retrieval strategies.

Task: given a natural language question, retrieve the relevant table definitions
from a pool of 50 (4 core + 46 decoy) tables. Measures recall@K — what fraction
of required tables appear in the top-K results.

Backends compared:
  - Qdrant (in-memory)      — dense, sparse (BM25/SPLADE), or hybrid (RRF)
  - InMemoryStore           — dense cosine similarity (numpy-free pure Python)
  - ChromaDB                — dense, persistent HNSW index

Embedding models compared:
  - openai/text-embedding-3-small  (1536 dims, fast, ~$0.00002/1K tokens)
  - openai/text-embedding-3-large  (3072 dims, higher quality, ~$0.00013/1K tokens)

Note: sparse/BM25 uses SPLADE (runs locally via FastEmbed, zero API cost).
      Hybrid fuses dense + sparse scores via Reciprocal Rank Fusion (RRF).

Usage:
    python scripts/benchmark_schema_retrieval.py                  # full sweep
    python scripts/benchmark_schema_retrieval.py --top-k 5        # single top-k
    python scripts/benchmark_schema_retrieval.py --backend qdrant  # qdrant only
    python scripts/benchmark_schema_retrieval.py --strategy hybrid --top-k 3 5 10
"""

import argparse
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from itertools import product as cartesian_product
from pathlib import Path
from typing import Literal

import litellm
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding

console = Console()

# ---------------------------------------------------------------------------
# Table definitions — same 50 tables as schema_retriever.py
# ---------------------------------------------------------------------------

CORE_TABLES = [
    "customers(id, name, email, country, signup_date DATE)",
    "products(id, name, category, price DECIMAL)",
    "orders(id, customer_id, order_date DATE, status VARCHAR) -- status values: 'completed', 'pending', 'cancelled'",
    "order_items(id, order_id, product_id, quantity INTEGER, unit_price DECIMAL)",
]

DECOY_TABLES = [
    "hr_employees(id, name, department, salary DECIMAL, hire_date DATE)",
    "hr_payroll(id, employee_id, month DATE, net_pay DECIMAL, tax DECIMAL)",
    "hr_performance(id, employee_id, review_date DATE, score INTEGER, comments VARCHAR)",
    "logistics_shipments(id, order_id, carrier VARCHAR, track_num VARCHAR, status VARCHAR)",
    "logistics_warehouses(id, location VARCHAR, capacity INTEGER, manager_id INTEGER)",
    "logistics_inventory(warehouse_id, product_id, qty INTEGER, last_recount DATE)",
    "marketing_campaigns(id, name, budget DECIMAL, start_date DATE, end_date DATE)",
    "marketing_clicks(id, campaign_id, user_id, click_time TIMESTAMP, source VARCHAR)",
    "marketing_conversions(id, click_id, amount DECIMAL)",
    "support_tickets(id, customer_id, issue_type VARCHAR, status VARCHAR, created_at TIMESTAMP)",
    "support_agents(id, name, tier INTEGER, shift VARCHAR)",
    "support_messages(id, ticket_id, sender_id, message TEXT, sent_at TIMESTAMP)",
    "finance_ledgers(id, account_id, credit DECIMAL, debit DECIMAL, transaction_date DATE)",
    "finance_accounts(id, name, type VARCHAR)",
    "finance_budgets(department, year INTEGER, allocated DECIMAL, spent DECIMAL)",
    "sales_leads(id, company_name, contact_name, status VARCHAR, potential_value DECIMAL)",
    "sales_meetings(id, lead_id, rep_id, meeting_time TIMESTAMP, transcript TEXT)",
    "it_assets(id, type VARCHAR, assigned_to INTEGER, purchase_date DATE, warranty_exp DATE)",
    "it_incidents(id, asset_id, issue VARCHAR, resolved BOOLEAN, reported_at TIMESTAMP)",
    "legal_contracts(id, party_name, start_date DATE, end_date DATE, value DECIMAL)",
    "legal_compliance(id, regulation VARCHAR, last_audit DATE, pass BOOLEAN)",
    "facilities_buildings(id, address VARCHAR, sqft INTEGER, lease_cost DECIMAL)",
    "facilities_maintenance(id, building_id, task VARCHAR, cost DECIMAL, date DATE)",
    "vendor_suppliers(id, name, rating INTEGER, payment_terms VARCHAR)",
    "vendor_payments(id, supplier_id, invoice_num VARCHAR, amount DECIMAL, paid_date DATE)",
    "social_posts(id, platform VARCHAR, content TEXT, likes INTEGER, shares INTEGER)",
    "social_influencers(id, handle VARCHAR, followers INTEGER, contract_id INTEGER)",
    "ecommerce_cart(id, session_id, user_id, created_at TIMESTAMP)",
    "ecommerce_cart_items(id, cart_id, product_id, qty INTEGER)",
    "ecommerce_sessions(id, ip_address VARCHAR, browser VARCHAR, dur_seconds INTEGER)",
    "app_users(id, username, password_hash VARCHAR, last_login TIMESTAMP)",
    "app_feature_flags(id, feature_name, is_active BOOLEAN)",
    "app_logs(id, level VARCHAR, message TEXT, timestamp TIMESTAMP)",
    "security_audits(id, auditor VARCHAR, findigs TEXT, report_date DATE)",
    "security_breaches(id, severity VARCHAR, affected_records INTEGER, discovered_at TIMESTAMP)",
    "events_attendance(id, event_id, user_id, check_in TIMESTAMP)",
    "events_locations(id, name, capacity INTEGER)",
    "training_courses(id, title VARCHAR, instructor VARCHAR, duration_hours INTEGER)",
    "training_enrollments(id, course_id, employee_id, completion_date DATE)",
    "seo_keywords(id, keyword VARCHAR, search_volume INTEGER, difficulty INTEGER)",
    "seo_rankings(id, url VARCHAR, keyword_id INTEGER, rank INTEGER, check_date DATE)",
    "subscriptions_plans(id, name, price DECIMAL, billing_cycle VARCHAR)",
    "subscriptions_active(id, customer_id, plan_id, auto_renew BOOLEAN)",
    "partners_affiliates(id, name, commission_rate DECIMAL, total_referred INTEGER)",
    "partners_payouts(id, affiliate_id, amount DECIMAL, date DATE)",
]

ALL_TABLES = CORE_TABLES + DECOY_TABLES

# Table name → canonical short name for recall checking
# Each table definition starts with "tablename("
def _table_name(table_def: str) -> str:
    return table_def.split("(")[0].strip()

TABLE_NAMES = [_table_name(t) for t in ALL_TABLES]

# ---------------------------------------------------------------------------
# Eval questions with required tables (derived from golden SQL queries)
# ---------------------------------------------------------------------------

EVAL_QUESTIONS = [
    {"id": "q01", "question": "How many customers are there in total?",
     "required": ["customers"]},
    {"id": "q02", "question": "List all product categories.",
     "required": ["products"]},
    {"id": "q03", "question": "What is the most expensive product?",
     "required": ["products"]},
    {"id": "q04", "question": "How many orders have the status 'cancelled'?",
     "required": ["orders"]},
    {"id": "q05", "question": "Which customers are from the USA?",
     "required": ["customers"]},
    {"id": "q06", "question": "What is the total revenue from completed orders?",
     "required": ["order_items", "orders"]},
    {"id": "q07", "question": "How many orders did each customer place? Show customer name and order count.",
     "required": ["customers", "orders"]},
    {"id": "q08", "question": "What is the average order value for completed orders?",
     "required": ["orders", "order_items"]},
    {"id": "q09", "question": "Which product category generates the most revenue from completed orders?",
     "required": ["order_items", "products", "orders"]},
    {"id": "q10", "question": "List customers who signed up in 2024, showing name and signup date.",
     "required": ["customers"]},
    {"id": "q11", "question": "Who are the top 3 customers by total spend on completed orders?",
     "required": ["customers", "orders", "order_items"]},
    {"id": "q12", "question": "Which customers have never placed a completed order?",
     "required": ["customers", "orders"]},
    {"id": "q13", "question": "What is the month-by-month revenue for 2024 from completed orders?",
     "required": ["orders", "order_items"]},
    {"id": "q14", "question": "For each country, show the number of customers and total completed orders.",
     "required": ["customers", "orders"]},
    {"id": "q15", "question": "Which customers placed more than one order and what is their average order value?",
     "required": ["customers", "orders", "order_items"]},
]

# ---------------------------------------------------------------------------
# Embedding model metadata
# ---------------------------------------------------------------------------

EMBEDDING_MODELS = {
    "openai/text-embedding-3-small": {
        "dims": 1536,
        "cost_per_1k_tokens": 0.00002,
        "label": "3-small",
    },
    "openai/text-embedding-3-large": {
        "dims": 3072,
        "cost_per_1k_tokens": 0.00013,
        "label": "3-large",
    },
}

SPARSE_MODEL = "prithivida/Splade_PP_en_v1"

# Approximate token count for one table definition (avg ~15 tokens)
AVG_TOKENS_PER_TABLE = 15
AVG_QUESTION_TOKENS = 20

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    backend: str
    strategy: str
    embedding_model: str            # "sparse" if SPLADE only
    top_k: int
    recall: float                   # mean across questions (fraction of required tables hit)
    perfect_recall: float           # fraction of questions where ALL required tables retrieved
    per_question: list[float] = field(default_factory=list)
    index_time: float = 0.0
    avg_query_time: float = 0.0
    estimated_cost_usd: float = 0.0

    @property
    def label(self) -> str:
        emb = self.embedding_model.split("/")[-1] if "/" in self.embedding_model else self.embedding_model
        return f"{self.backend}/{self.strategy}/{emb}"

# ---------------------------------------------------------------------------
# Embedding cache — avoid re-embedding the same 50 tables multiple times
# ---------------------------------------------------------------------------

_dense_embedding_cache: dict[str, list[list[float]]] = {}
_sparse_embeddings_cache: list | None = None
_sparse_embedder: SparseTextEmbedding | None = None


def get_dense_embeddings(texts: list[str], model: str) -> list[list[float]]:
    """Embed texts with the given model, caching by (model, text)."""
    cache_key = f"{model}::tables" if texts == ALL_TABLES else None
    if cache_key and cache_key in _dense_embedding_cache:
        return _dense_embedding_cache[cache_key]
    embeddings = [item["embedding"] for item in litellm.embedding(model=model, input=texts).data]
    if cache_key:
        _dense_embedding_cache[cache_key] = embeddings
    return embeddings


def get_sparse_embeddings(texts: list[str]) -> list:
    """Embed texts with SPLADE (local model), caching the table embeddings."""
    global _sparse_embedder, _sparse_embeddings_cache
    if _sparse_embedder is None:
        console.print("  [dim]Loading SPLADE sparse embedder (first time only)...[/dim]")
        _sparse_embedder = SparseTextEmbedding(model_name=SPARSE_MODEL)
    if texts == ALL_TABLES and _sparse_embeddings_cache is not None:
        return _sparse_embeddings_cache
    result = list(_sparse_embedder.embed(texts))
    if texts == ALL_TABLES:
        _sparse_embeddings_cache = result
    return result

# ---------------------------------------------------------------------------
# Recall computation
# ---------------------------------------------------------------------------

def compute_recall(retrieved_table_defs: list[str], required_tables: list[str]) -> float:
    """
    Fraction of required table names that appear in any of the retrieved table definitions.
    1.0 = all required tables retrieved; 0.0 = none retrieved.
    """
    if not required_tables:
        return 1.0
    retrieved_names = set(_table_name(t) for t in retrieved_table_defs)
    hits = sum(1 for t in required_tables if t in retrieved_names)
    return hits / len(required_tables)

# ---------------------------------------------------------------------------
# Backend: Qdrant
# ---------------------------------------------------------------------------

def run_qdrant(
    strategy: Literal["dense", "sparse", "hybrid"],
    embedding_model: str | None,
    top_k: int,
) -> BenchmarkResult:
    """Benchmark Qdrant with the given retrieval strategy and embedding model."""
    label_model = embedding_model or "splade-only"
    console.print(f"  [cyan]qdrant/{strategy}/{label_model.split('/')[-1]} top_k={top_k}[/cyan]", end=" ")

    client = QdrantClient(":memory:")
    collection = f"bench_{strategy}_{(label_model).replace('/', '_').replace('-', '_')}"

    dense_size = EMBEDDING_MODELS[embedding_model]["dims"] if embedding_model else None

    vectors_config = {}
    sparse_vectors_config = {}
    if strategy in ("dense", "hybrid") and dense_size:
        vectors_config["dense"] = models.VectorParams(
            size=dense_size, distance=models.Distance.COSINE
        )
    if strategy in ("sparse", "hybrid"):
        sparse_vectors_config["sparse"] = models.SparseVectorParams(
            index=models.SparseIndexParams(on_disk=False)
        )

    client.create_collection(
        collection_name=collection,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )

    # --- Indexing ---
    t0 = time.time()

    dense_embs = get_dense_embeddings(ALL_TABLES, embedding_model) if embedding_model else None
    sparse_embs = get_sparse_embeddings(ALL_TABLES) if strategy in ("sparse", "hybrid") else None

    points = []
    for i, table_def in enumerate(ALL_TABLES):
        vector: dict = {}
        if dense_embs:
            vector["dense"] = dense_embs[i]
        if sparse_embs:
            vector["sparse"] = models.SparseVector(
                indices=sparse_embs[i].indices.tolist(),
                values=sparse_embs[i].values.tolist(),
            )
        points.append(models.PointStruct(
            id=i,
            vector=vector,
            payload={"table_def": table_def},
        ))

    client.upsert(collection_name=collection, points=points)
    index_time = time.time() - t0

    # --- Retrieval ---
    t_query_start = time.time()
    per_question = []

    for item in EVAL_QUESTIONS:
        question = item["question"]

        if strategy == "dense":
            q_dense = get_dense_embeddings([question], embedding_model)[0]
            results = client.query_points(
                collection_name=collection, query=q_dense, using="dense", limit=top_k
            ).points

        elif strategy == "sparse":
            q_sparse = get_sparse_embeddings([question])[0]
            results = client.query_points(
                collection_name=collection,
                query=models.SparseVector(
                    indices=q_sparse.indices.tolist(),
                    values=q_sparse.values.tolist(),
                ),
                using="sparse",
                limit=top_k,
            ).points

        else:  # hybrid
            q_dense = get_dense_embeddings([question], embedding_model)[0]
            q_sparse = get_sparse_embeddings([question])[0]
            results = client.query_points(
                collection_name=collection,
                prefetch=[
                    models.Prefetch(query=q_dense, using="dense", limit=top_k),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=q_sparse.indices.tolist(),
                            values=q_sparse.values.tolist(),
                        ),
                        using="sparse",
                        limit=top_k,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
            ).points

        retrieved = [hit.payload["table_def"] for hit in results]
        per_question.append(compute_recall(retrieved, item["required"]))

    query_time = time.time() - t_query_start

    # Cost estimate: index (50 tables) + 15 questions, dense only
    tokens = (len(ALL_TABLES) * AVG_TOKENS_PER_TABLE + len(EVAL_QUESTIONS) * AVG_QUESTION_TOKENS)
    cost_per_1k = EMBEDDING_MODELS[embedding_model]["cost_per_1k_tokens"] if embedding_model else 0.0
    estimated_cost = (tokens / 1000) * cost_per_1k

    recall = sum(per_question) / len(per_question)
    perfect = sum(1 for r in per_question if r == 1.0) / len(per_question)

    console.print(f"recall={recall:.3f}  perfect={perfect:.2f}  idx={index_time:.1f}s  qry={query_time:.2f}s")
    return BenchmarkResult(
        backend="qdrant",
        strategy=strategy,
        embedding_model=label_model,
        top_k=top_k,
        recall=recall,
        perfect_recall=perfect,
        per_question=per_question,
        index_time=index_time,
        avg_query_time=query_time / len(EVAL_QUESTIONS),
        estimated_cost_usd=estimated_cost,
    )

# ---------------------------------------------------------------------------
# Backend: InMemoryStore / ChromaDB (via src/rag/vector_store.py)
# ---------------------------------------------------------------------------

def run_legacy_backend(
    backend_name: Literal["memory", "chroma"],
    embedding_model: str,
    top_k: int,
    chroma_tmpdir: str | None = None,
) -> BenchmarkResult:
    """Benchmark InMemoryStore or ChromaDB using the existing vector_store.py infrastructure."""
    label = f"{backend_name}/dense/{embedding_model.split('/')[-1]}"
    console.print(f"  [magenta]{label} top_k={top_k}[/magenta]", end=" ")

    # Override EMBEDDING_MODEL env var so get_embeddings() uses the right model
    original_model = os.environ.get("EMBEDDING_MODEL", "")
    os.environ["EMBEDDING_MODEL"] = embedding_model

    try:
        import src.rag.vector_store as vs_module
        from src.rag.chunker import Chunk

        store = vs_module.create_store(
            backend_name,
            collection_name=f"bench_{backend_name}_{embedding_model.replace('/', '_').replace('-', '_')}",
            persist_dir=chroma_tmpdir or "./chroma_db_schema_bench",
        )
        store.reset()

        # Index: one Chunk per table definition (schema chunking)
        chunks = [Chunk(text=t, metadata={"source": "schema", "table": _table_name(t)}) for t in ALL_TABLES]

        t0 = time.time()
        store.add(chunks)
        index_time = time.time() - t0

        # Retrieve
        t_query_start = time.time()
        per_question = []

        for item in EVAL_QUESTIONS:
            results = store.query(item["question"], top_k=top_k)
            retrieved = [r.chunk.text for r in results]
            per_question.append(compute_recall(retrieved, item["required"]))

        query_time = time.time() - t_query_start

    finally:
        os.environ["EMBEDDING_MODEL"] = original_model

    tokens = (len(ALL_TABLES) * AVG_TOKENS_PER_TABLE + len(EVAL_QUESTIONS) * AVG_QUESTION_TOKENS)
    cost_per_1k = EMBEDDING_MODELS[embedding_model]["cost_per_1k_tokens"]
    estimated_cost = (tokens / 1000) * cost_per_1k

    recall = sum(per_question) / len(per_question)
    perfect = sum(1 for r in per_question if r == 1.0) / len(per_question)

    console.print(f"recall={recall:.3f}  perfect={perfect:.2f}  idx={index_time:.1f}s  qry={query_time:.2f}s")
    return BenchmarkResult(
        backend=backend_name,
        strategy="dense",
        embedding_model=embedding_model,
        top_k=top_k,
        recall=recall,
        perfect_recall=perfect,
        per_question=per_question,
        index_time=index_time,
        avg_query_time=query_time / len(EVAL_QUESTIONS),
        estimated_cost_usd=estimated_cost,
    )

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def print_summary_table(results: list[BenchmarkResult]) -> None:
    table = Table(title="\nSchema Retrieval Benchmark Results", show_lines=True)
    table.add_column("Backend", style="cyan")
    table.add_column("Strategy", style="yellow")
    table.add_column("Embedding", style="blue")
    table.add_column("top_k", justify="right")
    table.add_column("Recall@K", justify="right", style="bold")
    table.add_column("Perfect@K", justify="right")
    table.add_column("Idx time", justify="right")
    table.add_column("Avg qry", justify="right")
    table.add_column("Est. cost", justify="right")

    for r in sorted(results, key=lambda x: (-x.recall, -x.perfect_recall)):
        recall_str = f"{r.recall:.3f}"
        color = "green" if r.recall >= 0.90 else "yellow" if r.recall >= 0.70 else "red"
        emb_label = r.embedding_model.split("/")[-1] if "/" in r.embedding_model else r.embedding_model
        table.add_row(
            r.backend,
            r.strategy,
            emb_label,
            str(r.top_k),
            f"[{color}]{recall_str}[/{color}]",
            f"{r.perfect_recall:.2f}",
            f"{r.index_time:.1f}s",
            f"{r.avg_query_time*1000:.0f}ms",
            f"${r.estimated_cost_usd:.5f}",
        )

    console.print(table)


def print_per_question_breakdown(results: list[BenchmarkResult]) -> None:
    best = max(results, key=lambda x: (x.recall, x.perfect_recall))
    console.print(
        f"\n[bold]Per-question breakdown — best config: "
        f"{best.backend}/{best.strategy}/{best.embedding_model.split('/')[-1]} top_k={best.top_k}[/bold]"
    )
    for item, recall in zip(EVAL_QUESTIONS, best.per_question):
        icon = "✅" if recall == 1.0 else "⚠️ " if recall > 0 else "❌"
        tables_str = ", ".join(item["required"])
        console.print(f"  {icon}  [{item['id']}] {item['question'][:60]:<60} requires: {tables_str}")


def print_strategy_comparison(results: list[BenchmarkResult]) -> None:
    """Show a grouped view: for each strategy, best recall at each top_k."""
    console.print("\n[bold]Strategy comparison (best recall per strategy × top_k):[/bold]")

    top_ks = sorted(set(r.top_k for r in results))
    strategies = ["dense", "sparse", "hybrid"]

    comp_table = Table(show_lines=False)
    comp_table.add_column("Strategy")
    for k in top_ks:
        comp_table.add_column(f"top_k={k}", justify="right")

    for strategy in strategies:
        row = [strategy]
        for k in top_ks:
            matching = [r for r in results if r.strategy == strategy and r.top_k == k]
            if matching:
                best = max(matching, key=lambda x: x.recall)
                color = "green" if best.recall >= 0.90 else "yellow" if best.recall >= 0.70 else "red"
                row.append(f"[{color}]{best.recall:.3f}[/{color}] ({best.backend})")
            else:
                row.append("—")
        comp_table.add_row(*row)

    console.print(comp_table)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark schema retrieval across backends, models, and strategies"
    )
    parser.add_argument(
        "--backend", nargs="+",
        default=["qdrant", "memory", "chroma"],
        choices=["qdrant", "memory", "chroma"],
        help="Vector DB backends to include",
    )
    parser.add_argument(
        "--strategy", nargs="+",
        default=["dense", "sparse", "hybrid"],
        choices=["dense", "sparse", "hybrid"],
        help="Retrieval strategies (sparse and hybrid are Qdrant-only)",
    )
    parser.add_argument(
        "--embedding", nargs="+",
        default=["openai/text-embedding-3-small", "openai/text-embedding-3-large"],
        choices=list(EMBEDDING_MODELS.keys()),
        dest="embeddings",
        help="Embedding models to compare",
    )
    parser.add_argument(
        "--top-k", nargs="+", type=int, default=[3, 5, 10],
        dest="top_k",
    )
    args = parser.parse_args()

    console.print("\n[bold]Schema Retrieval Benchmark[/bold]")
    console.print(f"Tables: {len(ALL_TABLES)} ({len(CORE_TABLES)} core + {len(DECOY_TABLES)} decoy)")
    console.print(f"Questions: {len(EVAL_QUESTIONS)}")
    console.print(f"Backends: {args.backend}")
    console.print(f"Strategies: {args.strategy}")
    console.print(f"Embeddings: {args.embeddings}")
    console.print(f"Top-K: {args.top_k}\n")

    results: list[BenchmarkResult] = []
    chroma_tmpdir = tempfile.mkdtemp(prefix="chroma_schema_bench_")

    try:
        # --- Qdrant configurations ---
        if "qdrant" in args.backend:
            console.print("[bold]Qdrant[/bold]")

            for strategy, top_k in cartesian_product(args.strategy, args.top_k):
                if strategy == "sparse":
                    # Sparse is embedding-model-agnostic (SPLADE)
                    results.append(run_qdrant(strategy="sparse", embedding_model=None, top_k=top_k))
                else:
                    for emb_model in args.embeddings:
                        results.append(run_qdrant(strategy=strategy, embedding_model=emb_model, top_k=top_k))

        # --- InMemoryStore configurations ---
        if "memory" in args.backend and "dense" in args.strategy:
            console.print("\n[bold]InMemoryStore (numpy cosine)[/bold]")
            for emb_model, top_k in cartesian_product(args.embeddings, args.top_k):
                results.append(run_legacy_backend("memory", emb_model, top_k))

        # --- ChromaDB configurations ---
        if "chroma" in args.backend and "dense" in args.strategy:
            console.print("\n[bold]ChromaDB (HNSW)[/bold]")
            for emb_model, top_k in cartesian_product(args.embeddings, args.top_k):
                results.append(run_legacy_backend("chroma", emb_model, top_k, chroma_tmpdir=chroma_tmpdir))

    finally:
        # Clean up ChromaDB temp directory
        import shutil
        shutil.rmtree(chroma_tmpdir, ignore_errors=True)

    print_summary_table(results)
    print_strategy_comparison(results)
    print_per_question_breakdown(results)

    # Key takeaways
    if results:
        best = max(results, key=lambda x: (x.recall, x.perfect_recall))
        fastest = min(results, key=lambda x: x.avg_query_time)
        cheapest = min((r for r in results if r.strategy != "sparse"), key=lambda x: x.estimated_cost_usd)
        console.print("\n[bold]Key takeaways:[/bold]")
        console.print(f"  Best recall:      {best.label} @ top_k={best.top_k} → {best.recall:.3f}")
        console.print(f"  Fastest queries:  {fastest.label} @ top_k={fastest.top_k} → {fastest.avg_query_time*1000:.0f}ms/query")
        console.print(f"  Most cost-efficient: {cheapest.label} → ${cheapest.estimated_cost_usd:.5f}/run")


if __name__ == "__main__":
    main()
