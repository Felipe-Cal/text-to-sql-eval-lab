# Guardrails

Implementation lives under [`src/guardrails/`](../src/guardrails/): deterministic checks with **no extra LLM calls**. Call `check_input` before the LLM and `check_output` on generated SQL before execution.

```bash
pytest
```

## Input guardrails (`src/guardrails/input.py`)

Run **before** the model sees the question (`check_input`). Two independent checks; the first failure wins:

| Check | What it blocks |
|--------|----------------|
| **SQL injection** | Raw SQL fragments in the user string: DDL/DML keywords, comment tricks (`--`, `/* */`), `UNION` injection, stacked `;` queries, etc. |
| **Prompt injection** | Phrases that try to override instructions ("ignore the above", "new instructions", jailbreak-style patterns). |

Tests in [`tests/test_input_guardrails.py`](../tests/test_input_guardrails.py) cover legitimate golden-style questions (must pass), injection strings (must fail), and edge cases (empty input, mixed case).

## Output guardrails (`src/guardrails/output.py`)

Run **after** the model returns SQL, **before** execution (`check_output`):

| Check | What it does |
|--------|----------------|
| **SELECT-only** | Uses **sqlglot AST**: root statement must be read-only; blocks DDL (e.g. `DROP`, `CREATE`) and DML (`INSERT`, `UPDATE`, `DELETE`). |
| **Schema scope** | Every referenced table must be in the allowlist (`customers`, `products`, `orders`, `order_items` by default). Stops queries against arbitrary tables (e.g. catalog tables). |

Tests in [`tests/test_output_guardrails.py`](../tests/test_output_guardrails.py) cover valid `SELECT`s (including joins, CTEs, trailing `;`), and ensure DDL/DML / unknown tables are rejected.

## Adversarial tests

[`tests/test_guardrails_adversarial.py`](../tests/test_guardrails_adversarial.py) goes beyond happy paths: **evasion attempts** (obfuscated keywords, leetspeak, Unicode lookalikes, prompt-injection phrasing). Some cases are marked **`xfail`** — they document **known gaps** of regex/keyword input filtering (not silent failures).

The key finding: **output** guardrails (AST-based) are harder to evade than **input** (regex-based). Defense in depth means input bypasses may still be caught at output.

```bash
pytest tests/test_input_guardrails.py -v
pytest tests/test_output_guardrails.py -v
pytest tests/test_guardrails_adversarial.py -v
```
