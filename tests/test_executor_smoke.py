"""Smoke tests for the IR executor on tiny golden plans."""

from __future__ import annotations

import polars as pl

from manysql.executor import execute
from manysql.ir import (
    AggCall,
    AggKind,
    Aggregate,
    BinaryOp,
    ColumnRef,
    Filter,
    Limit,
    Literal,
    Op,
    OrderKey,
    Project,
    Scan,
    Sort,
    SortDirection,
)
from manysql.ir.types import FLOAT, INT, TEXT
from manysql.spec import SemanticConfig
from manysql.storage import schema_of, seed_datasets


def _catalog() -> dict[str, pl.DataFrame]:
    return seed_datasets()


def test_scan_returns_table() -> None:
    plan = Scan(table_name="employees", columns=schema_of("employees"))
    df = execute(plan, SemanticConfig.reference(), _catalog())
    assert df.height == 8
    assert "name" in df.columns


def test_filter_active_employees() -> None:
    emp = Scan(table_name="employees", columns=schema_of("employees"))
    plan = Filter(input=emp, predicate=ColumnRef("active"))
    df = execute(plan, SemanticConfig.reference(), _catalog())
    assert df["active"].to_list() == [True] * df.height
    assert df.height == 7


def test_project_with_arithmetic() -> None:
    emp = Scan(table_name="employees", columns=schema_of("employees"))
    plan = Project(
        input=emp,
        projections=(
            ("name", ColumnRef("name")),
            (
                "bonus",
                BinaryOp(Op.MUL, ColumnRef("salary"), Literal(0.1, FLOAT)),
            ),
        ),
        output_types=(TEXT, FLOAT),
    )
    df = execute(plan, SemanticConfig.reference(), _catalog())
    assert df.columns == ["name", "bonus"]
    assert df.height == 8
    # alice salary 120000 -> bonus 12000
    alice = df.filter(pl.col("name") == "alice")
    assert alice["bonus"].item() == 12000.0


def test_aggregate_count_per_dept() -> None:
    emp = Scan(table_name="employees", columns=schema_of("employees"))
    plan = Aggregate(
        input=emp,
        group_by=(("dept_id", ColumnRef("dept_id")),),
        aggregates=(("n", AggCall(AggKind.COUNT_STAR)),),
        output_types=(INT, INT),
    )
    df = execute(plan, SemanticConfig.reference(), _catalog())
    # 4 distinct dept_id groups including null
    assert df.height == 4


def test_sort_with_default_null_order() -> None:
    emp = Scan(table_name="employees", columns=schema_of("employees"))
    plan = Sort(
        input=emp,
        keys=(OrderKey(expr=ColumnRef("salary"), direction=SortDirection.ASC),),
    )
    df = execute(plan, SemanticConfig.reference(), _catalog())
    # Default ASC null order is LAST in reference SemanticConfig.
    salaries = df["salary"].to_list()
    nulls_at = [i for i, v in enumerate(salaries) if v is None]
    assert nulls_at and nulls_at[0] == df.height - 1


def test_limit() -> None:
    emp = Scan(table_name="employees", columns=schema_of("employees"))
    plan = Limit(input=emp, limit=3, offset=1)
    df = execute(plan, SemanticConfig.reference(), _catalog())
    assert df.height == 3
