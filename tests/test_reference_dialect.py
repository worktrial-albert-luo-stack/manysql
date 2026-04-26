"""End-to-end test of the reference dialect: SQL text -> parse -> lower -> execute -> verify."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from lark import Lark

from manysql.dialects._reference.lowering import lower
from manysql.executor import execute
from manysql.oracle import OracleHarness, Verdict
from manysql.spec import SemanticConfig
from manysql.storage import CATALOG, schema_of, seed_datasets

GRAMMAR_PATH = Path(__file__).parent.parent / "manysql" / "dialects" / "_reference" / "grammar.lark"


@pytest.fixture(scope="session")
def parser() -> Lark:
    return Lark(GRAMMAR_PATH.read_text(), start="start", parser="earley")


def _catalog_schemas() -> dict[str, tuple]:
    return {name: schema_of(name) for name in CATALOG}


def _verify(parser: Lark, sql: str) -> None:
    tree = parser.parse(sql)
    semantics = SemanticConfig.reference()
    plan = lower(tree, semantics, _catalog_schemas())
    catalog = seed_datasets()
    actual = execute(plan, semantics, catalog)
    report = OracleHarness().verify(plan, semantics, catalog, actual)
    assert report.verdict == Verdict.PASS, (
        f"sql={sql!r} verdict={report.verdict} primary={report.primary} "
        f"reason={report.actual_vs_primary_reason} "
        f"disagreements={report.inter_oracle_disagreements} "
        f"errors={[r.error for r in report.oracle_results if r.error]}"
    )


def test_simple_select(parser: Lark) -> None:
    _verify(parser, "SELECT name, salary FROM employees WHERE active")


def test_filter_with_arithmetic(parser: Lark) -> None:
    _verify(parser, "SELECT id, salary * 0.1 AS bonus FROM employees WHERE salary > 100000")


def test_group_by_count(parser: Lark) -> None:
    _verify(parser, "SELECT dept_id, COUNT(*) AS n FROM employees GROUP BY dept_id")


def test_group_by_with_having(parser: Lark) -> None:
    _verify(
        parser,
        "SELECT dept_id, COUNT(*) AS n FROM employees GROUP BY dept_id HAVING COUNT(*) > 1",
    )


def test_inner_join(parser: Lark) -> None:
    _verify(
        parser,
        "SELECT e.name, d.name AS dept "
        "FROM employees AS e INNER JOIN departments AS d ON e.dept_id = d.id",
    )


def test_left_join(parser: Lark) -> None:
    _verify(
        parser,
        "SELECT e.name, d.name AS dept "
        "FROM employees e LEFT JOIN departments d ON e.dept_id = d.id",
    )


def test_order_by_limit(parser: Lark) -> None:
    _verify(parser, "SELECT name, salary FROM employees ORDER BY salary DESC NULLS LAST LIMIT 3")


def test_distinct(parser: Lark) -> None:
    _verify(parser, "SELECT DISTINCT dept_id FROM employees")


def test_in_list(parser: Lark) -> None:
    _verify(parser, "SELECT name FROM employees WHERE dept_id IN (10, 20)")


def test_between(parser: Lark) -> None:
    _verify(parser, "SELECT name FROM employees WHERE salary BETWEEN 80000 AND 120000")


def test_is_null(parser: Lark) -> None:
    _verify(parser, "SELECT name FROM employees WHERE dept_id IS NULL")


def test_case_expr(parser: Lark) -> None:
    _verify(
        parser,
        "SELECT name, CASE WHEN salary > 100000 THEN 'high' ELSE 'low' END AS bucket FROM employees",
    )


def test_union_all(parser: Lark) -> None:
    _verify(
        parser,
        "SELECT id FROM employees WHERE active "
        "UNION ALL "
        "SELECT id FROM departments WHERE budget > 500000",
    )


def test_simple_cte(parser: Lark) -> None:
    _verify(
        parser,
        "WITH active_emps AS (SELECT id, name FROM employees WHERE active) "
        "SELECT name FROM active_emps",
    )


def test_uncorrelated_in_subquery(parser: Lark) -> None:
    _verify(
        parser,
        "SELECT name FROM employees "
        "WHERE dept_id IN (SELECT id FROM departments WHERE budget > 500000)",
    )
