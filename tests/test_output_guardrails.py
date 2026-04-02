"""
Tests for output guardrails.

Covers:
  - Valid SELECT queries that must pass (no false positives)
  - DDL statements that must be blocked
  - DML statements that must be blocked
  - Queries referencing unknown tables that must be blocked
  - Edge cases (empty SQL, subqueries, CTEs, JOINs, semicolons)
"""

import pytest
from src.guardrails.output import check_output, ALLOWED_TABLES


# ---------------------------------------------------------------------------
# Valid SELECT queries — must ALL pass (false positive check)
# ---------------------------------------------------------------------------

VALID_QUERIES = [
    # Simple
    "SELECT COUNT(*) AS total FROM customers",
    "SELECT DISTINCT category FROM products ORDER BY category",
    "SELECT name, price FROM products ORDER BY price DESC LIMIT 1",
    "SELECT name FROM customers WHERE country = 'USA'",
    # Joins
    "SELECT c.name, COUNT(o.id) FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.name",
    "SELECT c.name, SUM(oi.quantity * oi.unit_price) FROM customers c JOIN orders o ON o.customer_id = c.id JOIN order_items oi ON oi.order_id = o.id GROUP BY c.name",
    # Subqueries
    "SELECT name, price FROM products WHERE price = (SELECT MAX(price) FROM products)",
    "SELECT * FROM customers WHERE id NOT IN (SELECT customer_id FROM orders WHERE status = 'completed')",
    # Aggregations + date functions
    "SELECT MONTH(order_date) AS month, ROUND(SUM(oi.quantity * oi.unit_price), 2) FROM orders o JOIN order_items oi ON o.id = oi.order_id WHERE YEAR(o.order_date) = 2024 GROUP BY month",
    # Trailing semicolon (common model output)
    "SELECT COUNT(*) FROM customers;",
    # Aliases
    "SELECT p.name AS product_name, p.price FROM products p WHERE p.category = 'Electronics'",
    # WITH clause / CTE
    "WITH revenue AS (SELECT order_id, SUM(quantity * unit_price) AS total FROM order_items GROUP BY order_id) SELECT * FROM revenue",
]

@pytest.mark.parametrize("sql", VALID_QUERIES)
def test_valid_query_passes(sql):
    result = check_output(sql)
    assert result.passed, (
        f"False positive — valid query was blocked.\n"
        f"  SQL:      {sql!r}\n"
        f"  Category: {result.category}\n"
        f"  Reason:   {result.reason}"
    )


# ---------------------------------------------------------------------------
# DDL — must ALL be blocked as disallowed_statement
# ---------------------------------------------------------------------------

DDL_QUERIES = [
    "DROP TABLE customers",
    "DROP TABLE IF EXISTS customers",
    "CREATE TABLE evil (id INT, data TEXT)",
    "ALTER TABLE customers ADD COLUMN phone TEXT",
    "TRUNCATE TABLE orders",
    "TRUNCATE orders",
]

@pytest.mark.parametrize("sql", DDL_QUERIES)
def test_ddl_blocked(sql):
    result = check_output(sql)
    assert not result.passed, f"DDL was not blocked: {sql!r}"
    assert result.category == "disallowed_statement", (
        f"Wrong category for DDL {sql!r}: {result.category}"
    )


# ---------------------------------------------------------------------------
# DML — must ALL be blocked as disallowed_statement
# ---------------------------------------------------------------------------

DML_QUERIES = [
    "DELETE FROM customers WHERE 1=1",
    "DELETE FROM orders",
    "INSERT INTO customers (name, email) VALUES ('hacker', 'h@h.com')",
    "UPDATE customers SET email = 'hacked@x.com' WHERE id = 1",
    "UPDATE products SET price = 0",
]

@pytest.mark.parametrize("sql", DML_QUERIES)
def test_dml_blocked(sql):
    result = check_output(sql)
    assert not result.passed, f"DML was not blocked: {sql!r}"
    assert result.category == "disallowed_statement", (
        f"Wrong category for DML {sql!r}: {result.category}"
    )


# ---------------------------------------------------------------------------
# Schema violations — must ALL be blocked as schema_violation
# ---------------------------------------------------------------------------

SCHEMA_VIOLATION_QUERIES = [
    # System / metadata tables
    "SELECT * FROM sqlite_master",
    "SELECT * FROM information_schema.tables",
    # Tables that don't exist in the schema
    "SELECT * FROM users",
    "SELECT * FROM admin",
    "SELECT * FROM passwords",
    "SELECT * FROM secrets",
    # Injected table via crafted prompt
    "SELECT c.name FROM customers c JOIN stolen_data sd ON c.id = sd.user_id",
]

@pytest.mark.parametrize("sql", SCHEMA_VIOLATION_QUERIES)
def test_schema_violation_blocked(sql):
    result = check_output(sql)
    assert not result.passed, f"Schema violation was not blocked: {sql!r}"
    assert result.category == "schema_violation", (
        f"Wrong category for {sql!r}: {result.category}"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_sql_blocked():
    result = check_output("")
    assert not result.passed
    assert result.category == "empty_output"

def test_whitespace_only_blocked():
    result = check_output("   \n  ")
    assert not result.passed
    assert result.category == "empty_output"

def test_blocked_result_has_reason():
    result = check_output("DROP TABLE customers")
    assert result.reason is not None and len(result.reason) > 0

def test_result_is_falsy_when_blocked():
    assert not check_output("DELETE FROM orders")

def test_result_is_truthy_when_passed():
    assert check_output("SELECT COUNT(*) FROM customers")

def test_custom_allowed_tables():
    """check_output respects a caller-supplied allowed_tables set."""
    custom = frozenset({"users", "posts"})
    # Valid against custom schema
    assert check_output("SELECT * FROM users", allowed_tables=custom).passed
    # Invalid against custom schema (even though 'customers' is in default)
    result = check_output("SELECT * FROM customers", allowed_tables=custom)
    assert not result.passed
    assert result.category == "schema_violation"

def test_select_only_checked_before_schema_scope():
    """DROP on an unknown table should report disallowed_statement, not schema_violation."""
    result = check_output("DROP TABLE nonexistent_table")
    assert not result.passed
    assert result.category == "disallowed_statement"

def test_subquery_unknown_table_blocked():
    """Schema scope check must walk into subqueries."""
    sql = "SELECT * FROM customers WHERE id IN (SELECT user_id FROM stolen_data)"
    result = check_output(sql)
    assert not result.passed
    assert result.category == "schema_violation"

def test_cte_unknown_table_blocked():
    """Schema scope check must walk into CTEs."""
    sql = "WITH x AS (SELECT * FROM secrets) SELECT * FROM x"
    result = check_output(sql)
    assert not result.passed
    assert result.category == "schema_violation"
