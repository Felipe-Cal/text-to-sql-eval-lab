"""
Database utilities — seeds and manages the DuckDB e-commerce database.

Schema overview:
  customers   — who buys
  products    — what's sold
  orders      — a purchase transaction
  order_items — individual lines within an order
"""

import duckdb
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "datasets" / "ecommerce.duckdb"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS customers (
    id          INTEGER PRIMARY KEY,
    name        VARCHAR NOT NULL,
    email       VARCHAR NOT NULL,
    country     VARCHAR NOT NULL,
    signup_date DATE    NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    id       INTEGER PRIMARY KEY,
    name     VARCHAR NOT NULL,
    category VARCHAR NOT NULL,
    price    DECIMAL(10, 2) NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date  DATE    NOT NULL,
    status      VARCHAR NOT NULL   -- 'completed', 'pending', 'cancelled'
);

CREATE TABLE IF NOT EXISTS order_items (
    id         INTEGER PRIMARY KEY,
    order_id   INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL
);
"""

SEED_SQL = """
INSERT INTO customers VALUES
    (1,  'Alice Martin',    'alice@email.com',   'USA',     '2023-01-15'),
    (2,  'Bob Schmidt',     'bob@email.com',      'Germany', '2023-03-22'),
    (3,  'Clara Sousa',     'clara@email.com',    'Brazil',  '2023-06-10'),
    (4,  'David Chen',      'david@email.com',    'Canada',  '2023-07-04'),
    (5,  'Eva Rossi',       'eva@email.com',      'Italy',   '2023-09-18'),
    (6,  'Frank Müller',    'frank@email.com',    'Germany', '2024-01-02'),
    (7,  'Grace Kim',       'grace@email.com',    'USA',     '2024-02-14'),
    (8,  'Hugo Dubois',     'hugo@email.com',     'France',  '2024-03-30'),
    (9,  'Iris Patel',      'iris@email.com',     'India',   '2024-05-05'),
    (10, 'James Wilson',    'james@email.com',    'USA',     '2024-08-20');

INSERT INTO products VALUES
    (1,  'Wireless Mouse',       'Electronics',  29.99),
    (2,  'Mechanical Keyboard',  'Electronics',  89.99),
    (3,  'USB-C Hub',            'Electronics',  49.99),
    (4,  'Desk Lamp',            'Furniture',    34.99),
    (5,  'Ergonomic Chair',      'Furniture',   299.99),
    (6,  'Standing Desk',        'Furniture',   499.99),
    (7,  'Notebook (A5)',        'Stationery',    9.99),
    (8,  'Ballpoint Pens 10pk',  'Stationery',    4.99),
    (9,  'Webcam HD',            'Electronics',  79.99),
    (10, 'Monitor 27"',          'Electronics', 349.99);

INSERT INTO orders VALUES
    (1,  1, '2024-01-10', 'completed'),
    (2,  2, '2024-01-15', 'completed'),
    (3,  1, '2024-02-20', 'completed'),
    (4,  3, '2024-02-25', 'cancelled'),
    (5,  4, '2024-03-05', 'completed'),
    (6,  5, '2024-03-12', 'pending'),
    (7,  2, '2024-04-01', 'completed'),
    (8,  6, '2024-04-18', 'completed'),
    (9,  7, '2024-05-22', 'completed'),
    (10, 8, '2024-06-03', 'cancelled'),
    (11, 9, '2024-06-15', 'completed'),
    (12, 1, '2024-07-07', 'completed'),
    (13, 10,'2024-08-25', 'pending'),
    (14, 3, '2024-09-10', 'completed'),
    (15, 7, '2024-10-01', 'completed');

INSERT INTO order_items VALUES
    (1,  1,  1,  1, 29.99),
    (2,  1,  2,  1, 89.99),
    (3,  2,  5,  1,299.99),
    (4,  3,  3,  2, 49.99),
    (5,  3,  7,  3,  9.99),
    (6,  4,  6,  1,499.99),
    (7,  5,  4,  1, 34.99),
    (8,  5,  9,  1, 79.99),
    (9,  6,  10, 1,349.99),
    (10, 7,  1,  2, 29.99),
    (11, 7,  8,  5,  4.99),
    (12, 8,  2,  1, 89.99),
    (13, 9,  3,  1, 49.99),
    (14, 10, 5,  1,299.99),
    (15, 11, 9,  1, 79.99),
    (16, 12, 10, 1,349.99),
    (17, 13, 6,  1,499.99),
    (18, 14, 4,  2, 34.99),
    (19, 15, 1,  1, 29.99),
    (20, 15, 7,  2,  9.99);
"""


def get_connection(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path))


def seed_database(db_path: Path = DB_PATH, force: bool = False) -> None:
    """Create and seed the e-commerce database. Safe to call multiple times."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = get_connection(db_path)
    try:
        if force:
            con.execute("DROP TABLE IF EXISTS order_items")
            con.execute("DROP TABLE IF EXISTS orders")
            con.execute("DROP TABLE IF EXISTS products")
            con.execute("DROP TABLE IF EXISTS customers")

        con.execute(SCHEMA_SQL)

        # Only seed if tables are empty
        count = con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        if count == 0:
            con.execute(SEED_SQL)
            print("Database seeded successfully.")
        else:
            print(f"Database already has data ({count} customers). Use force=True to reseed.")
    finally:
        con.close()


def get_schema_string(db_path: Path = DB_PATH) -> str:
    """Return an unmanageably large schema description to mimic Enterprise DWH."""
    from src.agent.schema_retriever import ALL_TABLES
    
    tables_str = "\n  ".join(ALL_TABLES)
    return f"""
Database: Enterprise Data Warehouse (DuckDB dialect)

Tables:
  {tables_str}

Relationships:
  orders.customer_id → customers.id
  order_items.order_id → orders.id
  order_items.product_id → products.id
""".strip()


def execute_query(sql: str, db_path: Path = DB_PATH) -> list[tuple]:
    """Run a SQL query and return rows. Raises on error."""
    con = get_connection(db_path)
    try:
        result = con.execute(sql).fetchall()
        return result
    finally:
        con.close()


if __name__ == "__main__":
    seed_database(force=True)
    print("\nSchema:\n", get_schema_string())
    print("\nSample query — top 3 customers by spend:")
    rows = execute_query("""
        SELECT c.name, SUM(oi.quantity * oi.unit_price) AS total_spend
        FROM customers c
        JOIN orders o ON o.customer_id = c.id
        JOIN order_items oi ON oi.order_id = o.id
        WHERE o.status = 'completed'
        GROUP BY c.name
        ORDER BY total_spend DESC
        LIMIT 3
    """)
    for row in rows:
        print(row)
