# Guardrails

Three independent layers protect the pipeline from malicious input, LLM misbehaviour, and accidental data mutation. Each layer uses a different mechanism — no single bypass defeats all three.

```
User question
     │
     ▼
① Input guardrails   (regex — blocks injections before LLM sees them)
     │
     ▼
   LLM generates SQL
     │
     ▼
② Output guardrails  (sqlglot AST — blocks non-SELECT and schema violations)
     │
     ▼
③ Execution guardrail (DuckDB read_only=True — engine-level mutation block)
     │
     ▼
  Results returned
```

Implementation lives in `src/guardrails/`. All checks are synchronous and deterministic — no extra LLM calls.

---

## Layer 1 — Input guardrails (`src/guardrails/input.py`)

Run **before the LLM sees the question** (`check_input(text) → GuardrailResult`). Two independent checks; the first failure wins.

### SQL injection detection

Catches attempts to embed raw SQL in the natural language question. Patterns detected:

| Pattern | Example |
|---|---|
| DDL keywords | `DROP TABLE`, `CREATE TABLE`, `ALTER TABLE`, `TRUNCATE` |
| DML keywords | `INSERT INTO`, `UPDATE ... SET`, `DELETE FROM` |
| Comment sequences | `--`, `/* */`, `#` |
| UNION injection | `UNION SELECT`, `UNION ALL SELECT` |
| Stacked queries | `; DROP`, `; DELETE`, multiple statements separated by `;` |
| Schema introspection | `information_schema`, `pg_catalog`, `sqlite_master` |

All patterns are compiled at module load time and matched case-insensitively.

### Prompt injection detection

Catches phrases designed to override the system prompt or jailbreak the model:

| Pattern | Example |
|---|---|
| Instruction override | `ignore the above`, `ignore previous instructions` |
| Role hijacking | `act as`, `you are now`, `pretend you are` |
| New instructions | `new instructions:`, `forget everything` |
| Direct commands | `instead, do`, `disregard` |

### Usage

```python
from src.guardrails.input import check_input

result = check_input("Which customers are from Germany?")
# GuardrailResult(passed=True, reason=None)

result = check_input("DROP TABLE customers; SELECT 1")
# GuardrailResult(passed=False, reason="SQL injection pattern detected: 'DROP TABLE'")
```

---

## Layer 2 — Output guardrails (`src/guardrails/output.py`)

Run **after the LLM generates SQL, before execution** (`check_output(sql) → GuardrailResult`). Uses the **sqlglot AST** — not regex — making it much harder to evade through obfuscation.

### SELECT-only enforcement

Parses the SQL into an Abstract Syntax Tree and checks that the root statement is a `SELECT`. Blocks:

- `DROP TABLE customers`
- `DELETE FROM orders`
- `INSERT INTO ...`
- `UPDATE products SET ...`
- `CREATE TABLE ...`
- Any DDL or DML that passes through input guardrails

CTEs (`WITH ... AS (SELECT ...)`) are allowed because sqlglot correctly identifies their root as `SELECT`.

### Schema scope enforcement

After confirming the root is `SELECT`, extracts every table reference from the AST and verifies it is in the allowed list:

```python
ALLOWED_TABLES = {"customers", "products", "orders", "order_items"}
```

- Rejects queries referencing system tables (`information_schema.tables`)
- Rejects queries against decoy tables (e.g. `hr_employees`, `logistics_shipments`)
- CTE aliases are excluded from the check (they're not real tables)

### Usage

```python
from src.guardrails.output import check_output

result = check_output("SELECT name FROM customers WHERE country = 'USA'")
# GuardrailResult(passed=True, reason=None)

result = check_output("DROP TABLE customers")
# GuardrailResult(passed=False, reason="Non-SELECT statement detected: Drop")

result = check_output("SELECT * FROM hr_employees")
# GuardrailResult(passed=False, reason="Query references disallowed table: hr_employees")
```

---

## Layer 3 — Execution guardrail (`src/utils/db.py`)

The DuckDB connection used by `execute_query` is opened with **`read_only=True`**:

```python
def get_connection(db_path, read_only=False):
    return duckdb.connect(str(db_path), read_only=read_only)

def execute_query(sql, db_path=DB_PATH):
    con = get_connection(db_path, read_only=True)
    ...
```

This is the last line of defence. Even if a destructive statement bypasses both input and output guardrails, DuckDB itself will refuse to execute it — the connection has no write permissions at the engine level. The error is caught, logged, and returned to the LLM as a correction signal in the self-correction loop.

---

## Defense-in-depth reasoning

| Layer | Mechanism | Catches |
|---|---|---|
| Input | Regex | Obvious injection in natural language |
| Output | sqlglot AST | LLM-generated destructive SQL, schema scope violations |
| Execution | DuckDB `read_only` | Anything that slips through layers 1 and 2 |

Each layer has different failure modes. Regex (layer 1) can be evaded by obfuscation. AST checks (layer 2) are harder to evade but the model generates the SQL, so unusual syntax might parse differently. The engine-level read-only (layer 3) is unconditional — it cannot be bypassed at the application layer.

---

## Tests

```bash
# Happy paths — legitimate questions must not be blocked
pytest tests/test_input_guardrails.py -v

# Output checks — DDL/DML must be blocked, valid SELECTs must pass
pytest tests/test_output_guardrails.py -v

# Adversarial — evasion attempts: obfuscated keywords, leetspeak, Unicode lookalikes
pytest tests/test_guardrails_adversarial.py -v
```

Some adversarial tests are marked `@pytest.mark.xfail` — they document **known gaps** in the regex input layer. For example, `DRОP TABLE` (with a Cyrillic О) passes the input check. These are not silent failures: they are catalogued weaknesses. The output AST layer and the read-only connection catch them regardless.

**The key finding from adversarial testing:** input guardrails (regex) are the weakest layer. Output guardrails (AST) are significantly harder to evade. Defense in depth exists precisely for this — a bypass at layer 1 is expected to be caught at layer 2.
