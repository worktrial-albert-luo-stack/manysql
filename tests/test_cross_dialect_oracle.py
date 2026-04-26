"""Differential tests across multiple generated dialects.

Generates the mild / moderate / aggressive example dialects into a temp
directory, loads each through the registry, then runs a battery of
reference SQL queries through every dialect and asserts they all agree.

This is a higher-bar version of the multi-oracle harness: rather than
comparing one execution against external SQL engines, we compare many
dialects of *our* engine against each other.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from lark import Lark

from manysql.codegen import write_dialect_package
from manysql.dialects import DialectRegistry
from manysql.oracle import (
    CrossDialectMember,
    CrossDialectOracle,
    CrossDialectVerdict,
)
from manysql.spec.examples import EXAMPLE_SPECS
from manysql.storage import CATALOG, schema_of, seed_datasets


# ---- Battery of reference SQL strings the oracle will run --------------


_REFERENCE_BATTERY: list[str] = [
    "SELECT id FROM employees",
    "SELECT id, name FROM employees WHERE active",
    "SELECT id FROM employees WHERE dept_id IN (10, 20)",
    "SELECT id FROM employees WHERE dept_id IS NULL",
    "SELECT dept_id, COUNT(*) FROM employees GROUP BY dept_id",
    "SELECT id FROM employees ORDER BY id",
    "SELECT id FROM employees ORDER BY salary DESC NULLS LAST",
    "SELECT id FROM employees LIMIT 3",
    "SELECT DISTINCT dept_id FROM employees",
    "SELECT id FROM employees WHERE name LIKE 'a%'",
]


# ---- Fixtures: emit the example dialects once per session --------------


@pytest.fixture(scope="module")
def emitted_dialect_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("cross_dialects")
    for spec_name in ("mild_postgres_ish", "moderate_keyword_swap", "aggressive_alien"):
        write_dialect_package(EXAMPLE_SPECS[spec_name], root)
    return root


@pytest.fixture(scope="module")
def members(emitted_dialect_root: Path) -> list[CrossDialectMember]:
    registry = DialectRegistry(root=emitted_dialect_root)
    out: list[CrossDialectMember] = []
    for spec_name in ("mild_postgres_ish", "moderate_keyword_swap", "aggressive_alien"):
        engine = registry.load(spec_name)
        parser = Lark(engine.grammar_text, start="start", parser="earley")
        out.append(
            CrossDialectMember(
                name=spec_name,
                surface=EXAMPLE_SPECS[spec_name].surface,
                parser=parser,
                lowering=engine.lowering,
                semantics=engine.semantics,
                overrides=engine.overrides,
            )
        )
    return out


@pytest.fixture(scope="module")
def schemas() -> dict:
    return {name: schema_of(name) for name in CATALOG}


@pytest.fixture(scope="module")
def cross_oracle(members, schemas) -> CrossDialectOracle:
    return CrossDialectOracle(members=members, schemas=schemas)


# ---- Tests --------------------------------------------------------------


@pytest.mark.parametrize("ref_sql", _REFERENCE_BATTERY, ids=lambda s: s[:50])
def test_dialects_agree_on_reference_battery(
    ref_sql: str, cross_oracle: CrossDialectOracle
) -> None:
    """Every dialect should produce the same rows for the same logical query."""

    catalog = seed_datasets()
    report = cross_oracle.verify(ref_sql, catalog)
    assert report.verdict == CrossDialectVerdict.PASS, (
        f"sql={ref_sql!r}\n"
        f"verdict={report.verdict}\n"
        f"disagreements={report.disagreements}\n"
        f"errors={[(e.name, e.error) for e in report.executions if e.error]}"
    )


def test_oracle_reports_no_dialects_when_empty(schemas) -> None:
    oracle = CrossDialectOracle(members=[], schemas=schemas)
    report = oracle.verify("SELECT 1", seed_datasets())
    assert report.verdict == CrossDialectVerdict.NO_DIALECTS


def test_oracle_flags_disagreement_when_one_dialect_errors(
    members: list[CrossDialectMember], schemas
) -> None:
    """Sanity: when only one dialect can run a query, the oracle escalates
    to NEEDS_REVIEW (single signal isn't strong enough for a PASS)."""

    # Take just the first member so usable-count = 1.
    oracle = CrossDialectOracle(members=members[:1], schemas=schemas)
    report = oracle.verify("SELECT id FROM employees", seed_datasets())
    assert report.verdict == CrossDialectVerdict.NEEDS_REVIEW


def test_oracle_includes_rewritten_sql_in_executions(
    cross_oracle: CrossDialectOracle,
) -> None:
    """Each execution should expose the per-dialect surface SQL for
    debuggability."""

    catalog = seed_datasets()
    report = cross_oracle.verify("SELECT id FROM employees", catalog)
    assert len(report.executions) >= 2
    rewritten = {e.name: e.rewritten_sql for e in report.executions}
    assert all(rewritten.values()), rewritten
