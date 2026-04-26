"""Run every golden query through the multi-oracle harness.

Each query parses with the reference dialect, lowers to IR, executes through
the Polars executor, and is verified by every applicable oracle.
"""

from __future__ import annotations

import pytest
from lark import Lark

from manysql.dialects import DialectRegistry
from manysql.executor import execute
from manysql.golden import GOLDEN_QUERIES, GoldenQuery
from manysql.oracle import OracleHarness, Verdict
from manysql.storage import CATALOG, schema_of, seed_datasets


@pytest.fixture(scope="module")
def engine():
    return DialectRegistry().load("_reference")


@pytest.fixture(scope="module")
def parser(engine) -> Lark:
    return Lark(engine.grammar_text, start="start", parser="earley")


@pytest.fixture(scope="module")
def catalog():
    return seed_datasets()


@pytest.fixture(scope="module")
def schemas():
    return {name: schema_of(name) for name in CATALOG}


@pytest.fixture(scope="module")
def harness():
    return OracleHarness()


CROSS_DIALECT_QUERIES = [q for q in GOLDEN_QUERIES if q.cross_dialect]
NON_CROSS_DIALECT_QUERIES = [q for q in GOLDEN_QUERIES if not q.cross_dialect]


@pytest.mark.parametrize(
    "q", CROSS_DIALECT_QUERIES, ids=[q.id for q in CROSS_DIALECT_QUERIES]
)
def test_golden_query_against_oracles(
    q: GoldenQuery, engine, parser: Lark, catalog, schemas, harness: OracleHarness
) -> None:
    tree = parser.parse(q.sql)
    plan = engine.lowering.lower(tree, engine.semantics, schemas)
    actual = execute(plan, engine.semantics, catalog)
    report = harness.verify(plan, engine.semantics, catalog, actual)

    assert report.verdict == Verdict.PASS, (
        f"[{q.id}] verdict={report.verdict} primary={report.primary} "
        f"reason={report.actual_vs_primary_reason} "
        f"disagreements={report.inter_oracle_disagreements} "
        f"errors={[(r.oracle, r.error) for r in report.oracle_results if r.error]}"
    )


@pytest.mark.parametrize(
    "q", NON_CROSS_DIALECT_QUERIES, ids=[q.id for q in NON_CROSS_DIALECT_QUERIES]
)
def test_golden_query_executes(
    q: GoldenQuery, engine, parser: Lark, catalog, schemas
) -> None:
    """Smoke-test queries that aren't safe to cross-render: parse + execute only."""
    tree = parser.parse(q.sql)
    plan = engine.lowering.lower(tree, engine.semantics, schemas)
    df = execute(plan, engine.semantics, catalog)
    assert df is not None
