"""
Output guardrails for the text-to-SQL pipeline.

Runs after the LLM generates SQL, before it is executed. Two checks:

  1. select_only    — verifies the root statement is a SELECT; blocks all
                      DDL (DROP, CREATE, ALTER, TRUNCATE) and DML (INSERT,
                      UPDATE, DELETE). Uses the sqlglot AST — no DB needed.

  2. schema_scope   — extracts every table reference from the AST and
                      verifies each one exists in the allowed schema.
                      Prevents queries against tables outside the schema
                      (e.g. sqlite_master, information_schema, or tables
                      injected via a crafted prompt).

Both checks are deterministic, zero extra API calls, and use the same
sqlglot dependency already present for syntax_valid scoring.

Usage:
    from src.guardrails.output import check_output, ALLOWED_TABLES

    result = check_output(generated_sql)
    if not result.passed:
        print(f"Blocked [{result.category}]: {result.reason}")

    # Custom schema:
    result = check_output(sql, allowed_tables=frozenset({"users", "posts"}))
"""

import sqlglot
import sqlglot.expressions as exp

from src.guardrails.input import GuardrailResult, PASSED

# The four tables in the e-commerce schema
ALLOWED_TABLES: frozenset[str] = frozenset({
    "customers",
    "products",
    "orders",
    "order_items",
})

# Statement types that are never permitted
_BLOCKED_TYPES = (
    exp.Drop,
    exp.Create,
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Alter,
    exp.TruncateTable,
    exp.Command,      # catches raw commands sqlglot can't fully parse
)


# ---------------------------------------------------------------------------
# Check 1: SELECT-only enforcement
# ---------------------------------------------------------------------------

def _check_select_only(sql: str) -> GuardrailResult:
    try:
        statements = sqlglot.parse(sql, dialect="duckdb")
    except sqlglot.errors.ParseError as e:
        return GuardrailResult(
            passed=False,
            category="parse_error",
            reason=f"SQL could not be parsed: {e}",
        )

    if not statements or statements[0] is None:
        return GuardrailResult(
            passed=False,
            category="parse_error",
            reason="SQL produced no parse tree.",
        )

    for stmt in statements:
        if stmt is None:
            continue
        if isinstance(stmt, _BLOCKED_TYPES):
            stmt_type = type(stmt).__name__
            return GuardrailResult(
                passed=False,
                category="disallowed_statement",
                reason=f"Only SELECT statements are allowed; got {stmt_type}.",
            )
        if not isinstance(stmt, exp.Select):
            stmt_type = type(stmt).__name__
            return GuardrailResult(
                passed=False,
                category="disallowed_statement",
                reason=f"Only SELECT statements are allowed; got {stmt_type}.",
            )

    return PASSED


# ---------------------------------------------------------------------------
# Check 2: Schema scope enforcement
# ---------------------------------------------------------------------------

def _check_schema_scope(
    sql: str,
    allowed_tables: frozenset[str],
) -> GuardrailResult:
    try:
        statements = sqlglot.parse(sql, dialect="duckdb")
    except sqlglot.errors.ParseError:
        return PASSED  # parse errors are handled by _check_select_only

    for stmt in (statements or []):
        if stmt is None:
            continue
        # CTE aliases are virtual tables defined within the query itself —
        # exclude them from the schema check or every WITH clause would fail.
        cte_aliases = {cte.alias.lower() for cte in stmt.find_all(exp.CTE)}

        for table in stmt.find_all(exp.Table):
            name = table.name.lower()
            if name and name not in allowed_tables and name not in cte_aliases:
                return GuardrailResult(
                    passed=False,
                    category="schema_violation",
                    reason=(
                        f"Query references table '{name}' which is not in "
                        f"the allowed schema {sorted(allowed_tables)}."
                    ),
                )

    return PASSED


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_output(
    sql: str,
    allowed_tables: frozenset[str] = ALLOWED_TABLES,
) -> GuardrailResult:
    """
    Run all output guardrail checks against the generated SQL.

    Checks run in order; the first failure is returned immediately.
    Returns GuardrailResult(passed=True) if all checks pass.

    Args:
        sql:            The SQL string produced by the LLM.
        allowed_tables: Set of table names permitted in the query.
                        Defaults to the e-commerce schema tables.

    Returns:
        GuardrailResult with passed=True, or passed=False with category
        and reason populated.
    """
    sql = sql.strip()

    if not sql:
        return GuardrailResult(
            passed=False,
            category="empty_output",
            reason="No SQL was generated.",
        )

    for check in (
        _check_select_only,
        lambda s: _check_schema_scope(s, allowed_tables),
    ):
        result = check(sql)
        if not result.passed:
            return result

    return PASSED
