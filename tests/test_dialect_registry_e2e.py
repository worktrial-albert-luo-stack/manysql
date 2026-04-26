"""End-to-end through the dialect registry: load reference dialect, parse, lower, execute, verify."""

from __future__ import annotations

from lark import Lark

from manysql.dialects import DialectRegistry
from manysql.executor import execute
from manysql.oracle import OracleHarness, Verdict
from manysql.storage import CATALOG, schema_of, seed_datasets


def _catalog_schemas() -> dict[str, tuple]:
    return {name: schema_of(name) for name in CATALOG}


def test_load_reference_dialect_and_execute() -> None:
    registry = DialectRegistry()
    listed = registry.list(include_reference=True)
    assert "_reference" in listed

    engine = registry.load("_reference")

    parser = Lark(engine.grammar_text, start="start", parser="earley")
    tree = parser.parse(
        "SELECT dept_id, COUNT(*) AS n FROM employees GROUP BY dept_id ORDER BY n DESC"
    )
    plan = engine.lowering.lower(tree, engine.semantics, _catalog_schemas())

    catalog = seed_datasets()
    actual = execute(plan, engine.semantics, catalog)
    report = OracleHarness().verify(plan, engine.semantics, catalog, actual)
    assert report.verdict == Verdict.PASS, report.to_summary()
