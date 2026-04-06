"""
Schema retrieval benchmark — compares vector DB backends, embedding libraries, and retrieval strategies.

Task: given a natural language question, retrieve the relevant table definitions
from a pool of 50 (4 core + 46 decoy) tables. Measures recall@K — what fraction
of required tables appear in the top-K results.

Backends compared:
  - Qdrant (in-memory)      — dense, sparse (BM25/SPLADE), or hybrid (RRF)
  - InMemoryStore           — dense cosine similarity (pure Python)
  - ChromaDB                — dense, persistent HNSW index

Embedding libraries compared:
  - OpenAI via LiteLLM      — text-embedding-3-small (1536d), text-embedding-3-large (3072d)
  - FastEmbed (local)       — BAAI/bge-small-en-v1.5 (384d), nomic-embed-text-v1.5 (768d)

Note on FastEmbed: models are downloaded on first use (~90MB each). After that they
run fully locally — no API key, no network, zero cost per query.

Sparse/BM25 uses SPLADE via FastEmbed (also local) — separate from the dense local models.
Hybrid fuses dense + sparse via Reciprocal Rank Fusion (RRF).

Usage:
    python scripts/benchmark_schema_retrieval.py               # full sweep
    python scripts/benchmark_schema_retrieval.py --top-k 5     # single top-k
    python scripts/benchmark_schema_retrieval.py --backend qdrant --strategy hybrid
    python scripts/benchmark_schema_retrieval.py \\
        --embedding openai/text-embedding-3-small fastembed/BAAI/bge-small-en-v1.5
"""

import argparse
import os
import shutil
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
from fastembed import SparseTextEmbedding, TextEmbedding

console = Console()

# ---------------------------------------------------------------------------
# Table definitions — 50 tables (4 core + 46 decoy)
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
    "analytics_events(id, user_id, event_type VARCHAR, properties JSONB, created_at TIMESTAMP)",
]

ALL_TABLES = CORE_TABLES + DECOY_TABLES


def _table_name(table_def: str) -> str:
    return table_def.split("(")[0].strip()


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
# Embedding model registry
# Two libraries: "litellm" (API-based) and "fastembed" (local inference)
# ---------------------------------------------------------------------------

EMBEDDING_MODELS: dict[str, dict] = {
    # --- OpenAI via LiteLLM (API, billed per token) ---
    "openai/text-embedding-3-small": {
        "dims": 1536,
        "library": "litellm",
        "cost_per_1k_tokens": 0.00002,
        "label": "openai/3-small",
    },
    "openai/text-embedding-3-large": {
        "dims": 3072,
        "library": "litellm",
        "cost_per_1k_tokens": 0.00013,
        "label": "openai/3-large",
    },
    # --- FastEmbed local models (no API, runs on CPU, downloaded on first use) ---
    "fastembed/BAAI/bge-small-en-v1.5": {
        "dims": 384,
        "library": "fastembed",
        "model_name": "BAAI/bge-small-en-v1.5",
        "cost_per_1k_tokens": 0.0,
        "label": "fastembed/bge-small",
    },
    "fastembed/nomic-ai/nomic-embed-text-v1.5": {
        "dims": 768,
        "library": "fastembed",
        "model_name": "nomic-ai/nomic-embed-text-v1.5",
        "cost_per_1k_tokens": 0.0,
        "label": "fastembed/nomic-768d",
    },
}

SPARSE_MODEL_NAME = "prithivida/Splade_PP_en_v1"

AVG_TOKENS_PER_TABLE = 15
AVG_QUESTION_TOKENS = 20

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    backend: str
    strategy: str
    embedding_model: str
    top_k: int
    recall: float
    perfect_recall: float
    per_question: list[float] = field(default_factory=list)
    index_time: float = 0.0
    avg_query_time: float = 0.0
    estimated_cost_usd: float = 0.0

    @property
    def label(self) -> str:
        meta = EMBEDDING_MODELS.get(self.embedding_model, {})
        short = meta.get("label", self.embedding_model.split("/")[-1])
        return f"{self.backend}/{self.strategy}/{short}"

# ---------------------------------------------------------------------------
# Embedding cache — avoid re-embedding the same 50 tables per library
# ---------------------------------------------------------------------------

_litellm_cache: dict[str, list[list[float]]] = {}
_fastembed_dense_embedders: dict[str, TextEmbedding] = {}
_fastembed_dense_cache: dict[str, list[list[float]]] = {}
_sparse_embedder: SparseTextEmbedding | None = None
_sparse_cache: list | None = None


def _embed_litellm(texts: list[str], model: str) -> list[list[float]]:
    cache_key = f"{model}::{'tables' if texts is ALL_TABLES else id(texts)}"
    if texts is ALL_TABLES and cache_key in _litellm_cache:
        return _litellm_cache[cache_key]
    result = [item["embedding"] for item in litellm.embedding(model=model, input=texts).data]
    if texts is ALL_TABLES:
        _litellm_cache[cache_key] = result
    return result


def _embed_fastembed_dense(texts: list[str], model_key: str) -> list[list[float]]:
    """Run dense embedding via FastEmbed (local CPU inference)."""
    global _fastembed_dense_embedders
    meta = EMBEDDING_MODELS[model_key]
    model_name = meta["model_name"]
    cache_key = f"fastembed::{model_name}"

    if model_name not in _fastembed_dense_embedders:
        console.print(f"  [dim]Downloading/loading FastEmbed model {model_name} (first time only)...[/dim]")
        _fastembed_dense_embedders[model_name] = TextEmbedding(model_name=model_name)

    if texts is ALL_TABLES and cache_key in _fastembed_dense_cache:
        return _fastembed_dense_cache[cache_key]

    embedder = _fastembed_dense_embedders[model_name]
    result = [e.tolist() for e in embedder.embed(texts)]

    if texts is ALL_TABLES:
        _fastembed_dense_cache[cache_key] = result
    return result


def embed_dense(texts: list[str], model_key: str) -> list[list[float]]:
    """Dispatch to the right embedding library based on the model key."""
    meta = EMBEDDING_MODELS[model_key]
    if meta["library"] == "fastembed":
        return _embed_fastembed_dense(texts, model_key)
    else:
        return _embed_litellm(texts, meta.get("model_name", model_key) if "model_name" in meta else model_key)


def embed_sparse(texts: list[str]) -> list:
    """BM25/SPLADE via FastEmbed — same for all dense embedding model choices."""
    global _sparse_embedder, _sparse_cache
    if _sparse_embedder is None:
        console.print(f"  [dim]Loading SPLADE sparse model (first time only)...[/dim]")
        _sparse_embedder = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    if texts is ALL_TABLES and _sparse_cache is not None:
        return _sparse_cache
    result = list(_sparse_embedder.embed(texts))
    if texts is ALL_TABLES:
        _sparse_cache = result
    return result

# ---------------------------------------------------------------------------
# Recall computation
# ---------------------------------------------------------------------------

def compute_recall(retrieved_table_defs: list[str], required_tables: list[str]) -> float:
    """Fraction of required table names found in any retrieved definition."""
    if not required_tables:
        return 1.0
    retrieved_names = {_table_name(t) for t in retrieved_table_defs}
    hits = sum(1 for t in required_tables if t in retrieved_names)
    return hits / len(required_tables)

# ---------------------------------------------------------------------------
# Backend: Qdrant
# ---------------------------------------------------------------------------

def run_qdrant(
    strategy: Literal["dense", "sparse", "hybrid"],
    embedding_model_key: str | None,
    top_k: int,
) -> BenchmarkResult:
    """Benchmark Qdrant with the given retrieval strategy and embedding model."""
    meta = EMBEDDING_MODELS.get(embedding_model_key, {}) if embedding_model_key else {}
    label = meta.get("label", "splade-only") if embedding_model_key else "splade-only"
    console.print(f"  [cyan]qdrant/{strategy}/{label} top_k={top_k}[/cyan]", end=" ")

    client = QdrantClient(":memory:")
    safe_label = label.replace("/", "_").replace("-", "_").replace(".", "_")
    collection = f"bench_qdrant_{strategy}_{safe_label}_k{top_k}"

    dense_size = meta.get("dims") if embedding_model_key else None

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

    # --- Index ---
    t0 = time.time()
    dense_embs = embed_dense(ALL_TABLES, embedding_model_key) if embedding_model_key else None
    sparse_embs = embed_sparse(ALL_TABLES) if strategy in ("sparse", "hybrid") else None

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
        points.append(models.PointStruct(id=i, vector=vector, payload={"table_def": table_def}))

    client.upsert(collection_name=collection, points=points)
    index_time = time.time() - t0

    # --- Retrieve ---
    t_query = time.time()
    per_question = []

    for item in EVAL_QUESTIONS:
        q = item["question"]

        if strategy == "dense":
            q_dense = embed_dense([q], embedding_model_key)[0]
            hits = client.query_points(
                collection_name=collection, query=q_dense, using="dense", limit=top_k
            ).points

        elif strategy == "sparse":
            q_sparse = embed_sparse([q])[0]
            hits = client.query_points(
                collection_name=collection,
                query=models.SparseVector(
                    indices=q_sparse.indices.tolist(),
                    values=q_sparse.values.tolist(),
                ),
                using="sparse",
                limit=top_k,
            ).points

        else:  # hybrid
            q_dense = embed_dense([q], embedding_model_key)[0]
            q_sparse = embed_sparse([q])[0]
            hits = client.query_points(
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

        retrieved = [h.payload["table_def"] for h in hits]
        per_question.append(compute_recall(retrieved, item["required"]))

    query_time = time.time() - t_query

    tokens = len(ALL_TABLES) * AVG_TOKENS_PER_TABLE + len(EVAL_QUESTIONS) * AVG_QUESTION_TOKENS
    cost_per_1k = meta.get("cost_per_1k_tokens", 0.0)
    estimated_cost = (tokens / 1000) * cost_per_1k

    recall = sum(per_question) / len(per_question)
    perfect = sum(1 for r in per_question if r == 1.0) / len(per_question)

    console.print(f"recall={recall:.3f}  perfect={perfect:.2f}  idx={index_time:.1f}s  qry={query_time:.2f}s")
    return BenchmarkResult(
        backend="qdrant",
        strategy=strategy,
        embedding_model=embedding_model_key or "splade-only",
        top_k=top_k,
        recall=recall,
        perfect_recall=perfect,
        per_question=per_question,
        index_time=index_time,
        avg_query_time=query_time / len(EVAL_QUESTIONS),
        estimated_cost_usd=estimated_cost,
    )

# ---------------------------------------------------------------------------
# Backend: InMemoryStore / ChromaDB
# These use src/rag/vector_store.py which has a get_embeddings() helper.
# We patch that helper per-run so the right embedding library is used.
# ---------------------------------------------------------------------------

def run_legacy_backend(
    backend_name: Literal["memory", "chroma"],
    embedding_model_key: str,
    top_k: int,
    chroma_tmpdir: str | None = None,
) -> BenchmarkResult:
    meta = EMBEDDING_MODELS[embedding_model_key]
    label = meta["label"]
    console.print(f"  [magenta]{backend_name}/dense/{label} top_k={top_k}[/magenta]", end=" ")

    import src.rag.vector_store as vs_module
    from src.rag.chunker import Chunk

    # Patch get_embeddings in vector_store so it uses whichever library we want
    original_get_embeddings = vs_module.get_embeddings
    if meta["library"] == "fastembed":
        vs_module.get_embeddings = lambda texts: _embed_fastembed_dense(texts, embedding_model_key)
    else:
        # For litellm models, set env var (get_embeddings reads it at call time)
        original_env = os.environ.get("EMBEDDING_MODEL", "")
        os.environ["EMBEDDING_MODEL"] = embedding_model_key

    try:
        safe_label = label.replace("/", "_").replace("-", "_").replace(".", "_")
        store = vs_module.create_store(
            backend_name,
            collection_name=f"bench_{backend_name}_{safe_label}",
            persist_dir=chroma_tmpdir or "./chroma_db_schema_bench",
        )
        store.reset()

        chunks = [Chunk(text=t, metadata={"source": "schema", "table": _table_name(t)}) for t in ALL_TABLES]

        t0 = time.time()
        store.add(chunks)
        index_time = time.time() - t0

        t_query = time.time()
        per_question = []
        for item in EVAL_QUESTIONS:
            results = store.query(item["question"], top_k=top_k)
            retrieved = [r.chunk.text for r in results]
            per_question.append(compute_recall(retrieved, item["required"]))
        query_time = time.time() - t_query

    finally:
        vs_module.get_embeddings = original_get_embeddings
        if meta["library"] != "fastembed":
            os.environ["EMBEDDING_MODEL"] = original_env

    tokens = len(ALL_TABLES) * AVG_TOKENS_PER_TABLE + len(EVAL_QUESTIONS) * AVG_QUESTION_TOKENS
    estimated_cost = (tokens / 1000) * meta.get("cost_per_1k_tokens", 0.0)
    recall = sum(per_question) / len(per_question)
    perfect = sum(1 for r in per_question if r == 1.0) / len(per_question)

    console.print(f"recall={recall:.3f}  perfect={perfect:.2f}  idx={index_time:.1f}s  qry={query_time:.2f}s")
    return BenchmarkResult(
        backend=backend_name,
        strategy="dense",
        embedding_model=embedding_model_key,
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
    table.add_column("Library", style="dim")
    table.add_column("Embedding model", style="blue")
    table.add_column("top_k", justify="right")
    table.add_column("Recall@K", justify="right", style="bold")
    table.add_column("Perfect@K", justify="right")
    table.add_column("Idx time", justify="right")
    table.add_column("Avg qry", justify="right")
    table.add_column("Est. cost", justify="right")

    for r in sorted(results, key=lambda x: (-x.recall, -x.perfect_recall, x.avg_query_time)):
        meta = EMBEDDING_MODELS.get(r.embedding_model, {})
        lib = meta.get("library", "—")
        model_label = meta.get("label", r.embedding_model.split("/")[-1])
        recall_str = f"{r.recall:.3f}"
        color = "green" if r.recall >= 0.90 else "yellow" if r.recall >= 0.70 else "red"
        cost_str = f"${r.estimated_cost_usd:.5f}" if r.estimated_cost_usd > 0 else "free"

        table.add_row(
            r.backend,
            r.strategy,
            lib,
            model_label,
            str(r.top_k),
            f"[{color}]{recall_str}[/{color}]",
            f"{r.perfect_recall:.2f}",
            f"{r.index_time:.1f}s",
            f"{r.avg_query_time * 1000:.0f}ms",
            cost_str,
        )

    console.print(table)


def print_library_comparison(results: list[BenchmarkResult]) -> None:
    """Side-by-side comparison: OpenAI API vs FastEmbed local at the same top_k."""
    top_ks = sorted(set(r.top_k for r in results))
    openai_models = [k for k, v in EMBEDDING_MODELS.items() if v["library"] == "litellm"]
    local_models = [k for k, v in EMBEDDING_MODELS.items() if v["library"] == "fastembed"]

    if not openai_models or not local_models:
        return

    console.print("\n[bold]Embedding library comparison (Qdrant dense, best recall per model × top_k):[/bold]")

    comp = Table(show_lines=False)
    comp.add_column("Model (library)", min_width=30)
    comp.add_column("Dims", justify="right")
    for k in top_ks:
        comp.add_column(f"top_k={k}", justify="right")
    comp.add_column("Cost/run", justify="right")

    all_models = openai_models + local_models
    for model_key in all_models:
        meta = EMBEDDING_MODELS[model_key]
        matching = [r for r in results if r.embedding_model == model_key and r.backend == "qdrant" and r.strategy == "dense"]
        if not matching:
            continue
        row = [meta["label"], str(meta["dims"])]
        for k in top_ks:
            best = next((r for r in matching if r.top_k == k), None)
            if best:
                color = "green" if best.recall >= 0.90 else "yellow" if best.recall >= 0.70 else "red"
                row.append(f"[{color}]{best.recall:.3f}[/{color}]")
            else:
                row.append("—")
        cost_per_1k = meta.get("cost_per_1k_tokens", 0.0)
        tokens = len(ALL_TABLES) * AVG_TOKENS_PER_TABLE + len(EVAL_QUESTIONS) * AVG_QUESTION_TOKENS
        cost = (tokens / 1000) * cost_per_1k
        row.append(f"${cost:.5f}" if cost > 0 else "[green]free[/green]")
        comp.add_row(*row)

    console.print(comp)


def print_strategy_comparison(results: list[BenchmarkResult]) -> None:
    top_ks = sorted(set(r.top_k for r in results))
    console.print("\n[bold]Strategy comparison (best recall across all embeddings × top_k):[/bold]")

    comp = Table(show_lines=False)
    comp.add_column("Strategy")
    for k in top_ks:
        comp.add_column(f"top_k={k}", justify="right")

    for strategy in ["dense", "sparse", "hybrid"]:
        row = [strategy]
        for k in top_ks:
            matching = [r for r in results if r.strategy == strategy and r.top_k == k]
            if matching:
                best = max(matching, key=lambda x: x.recall)
                color = "green" if best.recall >= 0.90 else "yellow" if best.recall >= 0.70 else "red"
                meta = EMBEDDING_MODELS.get(best.embedding_model, {})
                short = meta.get("label", best.embedding_model.split("/")[-1])
                row.append(f"[{color}]{best.recall:.3f}[/{color}] ({short})")
            else:
                row.append("—")
        comp.add_row(*row)

    console.print(comp)


def print_per_question_breakdown(results: list[BenchmarkResult]) -> None:
    best = max(results, key=lambda x: (x.recall, x.perfect_recall, -x.avg_query_time))
    meta = EMBEDDING_MODELS.get(best.embedding_model, {})
    label = meta.get("label", best.embedding_model)
    console.print(
        f"\n[bold]Per-question breakdown — best config: "
        f"{best.backend}/{best.strategy}/{label} top_k={best.top_k}[/bold]"
    )
    for item, recall in zip(EVAL_QUESTIONS, best.per_question):
        icon = "✅" if recall == 1.0 else "⚠️ " if recall > 0 else "❌"
        tables_str = ", ".join(item["required"])
        console.print(f"  {icon}  [{item['id']}] {item['question'][:62]:<62} requires: {tables_str}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark schema retrieval: backends × embedding libraries × strategies"
    )
    parser.add_argument(
        "--backend", nargs="+",
        default=["qdrant", "memory", "chroma"],
        choices=["qdrant", "memory", "chroma"],
    )
    parser.add_argument(
        "--strategy", nargs="+",
        default=["dense", "sparse", "hybrid"],
        choices=["dense", "sparse", "hybrid"],
        help="sparse and hybrid are Qdrant-only",
    )
    parser.add_argument(
        "--embedding", nargs="+",
        default=list(EMBEDDING_MODELS.keys()),
        choices=list(EMBEDDING_MODELS.keys()),
        dest="embeddings",
        help="Embedding models to compare (mix OpenAI and FastEmbed local)",
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
        # --- Qdrant ---
        if "qdrant" in args.backend:
            console.print("[bold]── Qdrant (in-memory) ──[/bold]")
            for strategy, top_k in cartesian_product(args.strategy, args.top_k):
                if strategy == "sparse":
                    results.append(run_qdrant("sparse", None, top_k))
                else:
                    for emb in args.embeddings:
                        results.append(run_qdrant(strategy, emb, top_k))

        # --- InMemoryStore ---
        if "memory" in args.backend and "dense" in args.strategy:
            console.print("\n[bold]── InMemoryStore (pure Python cosine) ──[/bold]")
            for emb, top_k in cartesian_product(args.embeddings, args.top_k):
                results.append(run_legacy_backend("memory", emb, top_k))

        # --- ChromaDB ---
        if "chroma" in args.backend and "dense" in args.strategy:
            console.print("\n[bold]── ChromaDB (HNSW) ──[/bold]")
            for emb, top_k in cartesian_product(args.embeddings, args.top_k):
                results.append(run_legacy_backend("chroma", emb, top_k, chroma_tmpdir))

    finally:
        shutil.rmtree(chroma_tmpdir, ignore_errors=True)

    print_summary_table(results)
    print_library_comparison(results)
    print_strategy_comparison(results)
    print_per_question_breakdown(results)

    if results:
        best = max(results, key=lambda x: (x.recall, x.perfect_recall, -x.avg_query_time))
        fastest = min(results, key=lambda x: x.avg_query_time)
        free_results = [r for r in results if r.estimated_cost_usd == 0.0 and r.strategy != "sparse"]
        best_free = max(free_results, key=lambda x: x.recall) if free_results else None

        console.print("\n[bold]Key takeaways:[/bold]")
        console.print(f"  Best overall:    {best.label} top_k={best.top_k} → recall={best.recall:.3f}")
        console.print(f"  Fastest:         {fastest.label} top_k={fastest.top_k} → {fastest.avg_query_time*1000:.0f}ms/query")
        if best_free:
            console.print(f"  Best free/local: {best_free.label} top_k={best_free.top_k} → recall={best_free.recall:.3f}")


if __name__ == "__main__":
    main()
