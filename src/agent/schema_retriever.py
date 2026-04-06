import os
import json
from typing import Literal
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
import litellm
from dotenv import load_dotenv

load_dotenv()

# The 4 golden tables our DuckDB instance actually has
CORE_TABLES = [
    "customers(id, name, email, country, signup_date DATE)",
    "products(id, name, category, price DECIMAL)",
    "orders(id, customer_id, order_date DATE, status VARCHAR) -- status values: 'completed', 'pending', 'cancelled'",
    "order_items(id, order_id, product_id, quantity INTEGER, unit_price DECIMAL)"
]

# 46 decoy tables to simulate an unmanageable enterprise data warehouse context
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
    "partners_payouts(id, affiliate_id, amount DECIMAL, date DATE)"
]

ALL_TABLES = CORE_TABLES + DECOY_TABLES

# Configuration
COLLECTION_NAME = "schema_retriever"
DENSE_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
SPARSE_MODEL = "prithivida/Splade_PP_en_v1" # High quality sparse model

# Global state
_QDRANT_CLIENT = None
_SPARSE_EMBEDDER = None

def get_retriever():
    global _QDRANT_CLIENT, _SPARSE_EMBEDDER
    if _QDRANT_CLIENT is None:
        _QDRANT_CLIENT = QdrantClient(":memory:")
        _SPARSE_EMBEDDER = SparseTextEmbedding(model_name=SPARSE_MODEL)
        
        # Dense vector size (OpenAI small is 1536)
        dense_size = 1536
        
        # Check if collection exists (though :memory: is always empty on start)
        if not _QDRANT_CLIENT.collection_exists(COLLECTION_NAME):
            _QDRANT_CLIENT.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "dense": models.VectorParams(
                        size=dense_size,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False
                        )
                    )
                }
            )
        
        # Prepare and upload data
        print("Initializing Qdrant Schema Retriever...")
        
        # 1. Generate Dense Embeddings
        dense_embeddings = [item["embedding"] for item in litellm.embedding(model=DENSE_MODEL, input=ALL_TABLES).data]
        
        # 2. Generate Sparse Embeddings
        sparse_embeddings = list(_SPARSE_EMBEDDER.embed(ALL_TABLES))
        
        points = []
        for i, (table_def, dense, sparse) in enumerate(zip(ALL_TABLES, dense_embeddings, sparse_embeddings)):
            points.append(models.PointStruct(
                id=i,
                vector={
                    "dense": dense,
                    "sparse": models.SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist()
                    )
                },
                payload={"table_def": table_def}
            ))
            
        _QDRANT_CLIENT.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print(f"Indexed {len(ALL_TABLES)} tables in Qdrant.")
        
    return _QDRANT_CLIENT

def retrieve_schema(
    question: str, 
    top_k: int = 5, 
    retrieval_type: Literal["dense", "sparse", "hybrid"] = "hybrid"
) -> tuple[str, list[str]]:
    """
    RAG Schema Linking using Qdrant with support for multiple retrieval types.
    """
    client = get_retriever()
    
    if retrieval_type == "dense":
        # 1. Embed question (dense)
        question_dense = litellm.embedding(model=DENSE_MODEL, input=[question]).data[0]["embedding"]
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=question_dense,
            using="dense",
            limit=top_k
        ).points
        
    elif retrieval_type == "sparse":
        # 2. Embed question (sparse)
        question_sparse = list(_SPARSE_EMBEDDER.embed([question]))[0]
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.SparseVector(
                indices=question_sparse.indices.tolist(),
                values=question_sparse.values.tolist()
            ),
            using="sparse",
            limit=top_k
        ).points
        
    else: # hybrid
        # 3. Hybrid search using Reciprocal Rank Fusion (RRF)
        question_dense = litellm.embedding(model=DENSE_MODEL, input=[question]).data[0]["embedding"]
        question_sparse = list(_SPARSE_EMBEDDER.embed([question]))[0]
        
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=question_dense,
                    using="dense",
                    limit=top_k
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=question_sparse.indices.tolist(),
                        values=question_sparse.values.tolist()
                    ),
                    using="sparse",
                    limit=top_k
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k
        ).points

    top_tables = [hit.payload["table_def"] for hit in results]
    
    # Format the schema string
    schema_lines = [
        f"Database: Enterprise Data Warehouse (DuckDB dialect) - Strategy: {retrieval_type}",
        "",
        "Extracted Semantic Tables:"
    ]
    for table_def in top_tables:
        schema_lines.append(f"  {table_def}")
        
    return "\n".join(schema_lines), top_tables

if __name__ == "__main__":
    q = "Who are the top 3 customers by revenue? (orders keyword match test)"
    for r_type in ["dense", "sparse", "hybrid"]:
        schema, tables = retrieve_schema(q, top_k=5, retrieval_type=r_type)
        print(f"\n--- Strategy: {r_type} ---")
        print(schema)
