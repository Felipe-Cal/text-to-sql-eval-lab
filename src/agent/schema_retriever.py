import os
import json
import math
import litellm
from pathlib import Path
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

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(a * a for a in v2))
    return dot_product / (norm_v1 * norm_v2)

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Fetch embeddings for a batch of texts using LiteLLM."""
    model = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
    response = litellm.embedding(model=model, input=texts)
    # Return embeddings correctly sorted by their implicit index
    return [item["embedding"] for item in sorted(response.data, key=lambda x: x["index"])]

# Cache the table embeddings so we don't re-embed 50 schemas on every eval run
_TABLE_EMBEDDINGS_CACHE = None

def get_table_embeddings() -> list[list[float]]:
    global _TABLE_EMBEDDINGS_CACHE
    if _TABLE_EMBEDDINGS_CACHE is None:
        _TABLE_EMBEDDINGS_CACHE = get_embeddings(ALL_TABLES)
    return _TABLE_EMBEDDINGS_CACHE

def retrieve_schema(question: str, top_k: int = 5) -> tuple[str, list[str]]:
    """
    RAG Schema Linking: Finds the top K most semantically relevant tables for a question.
    Returns:
       schema_string: The formatted schema to inject into the LLM.
       retrieved_tables: List of the raw table definition strings retrieved.
    """
    # 1. Embed the question
    question_embedding = get_embeddings([question])[0]
    
    # 2. Get table embeddings
    table_embeddings = get_table_embeddings()
    
    # 3. Calculate cosine similarity
    scored_tables = []
    for table_def, emb in zip(ALL_TABLES, table_embeddings):
        score = cosine_similarity(question_embedding, emb)
        scored_tables.append((score, table_def))
        
    # 4. Sort descending by score
    scored_tables.sort(key=lambda x: x[0], reverse=True)
    
    # 5. Extract top K
    top_tables = [table_def for score, table_def in scored_tables[:top_k]]
    
    # 6. Format exactly like get_schema_string()
    schema_lines = [
        "Database: Enterprise Data Warehouse (DuckDB dialect)",
        "",
        "Extracted Semantic Tables:"
    ]
    for table_def in top_tables:
        schema_lines.append(f"  {table_def}")
        
    return "\n".join(schema_lines), top_tables

if __name__ == "__main__":
    q = "Who are the top 3 customers by revenue?"
    schema, tables = retrieve_schema(q, top_k=5)
    print(f"Question: {q}\n")
    print(schema)
