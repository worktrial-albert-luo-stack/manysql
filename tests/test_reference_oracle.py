"""The Polars executor and the reference interpreter must agree.

These tests are the v1 backbone of correctness: any disagreement is a bug
in one of them, and the harness uses the same comparison machinery.
"""

from __future__ import annotations

import polars as pl
import pytest

from manysql.executor import execute
from manysql.ir import (
    AggCall,
    AggKind,
    Aggregate,
    BinaryOp,
    ColumnRef,
    Filter,
    Join,
    JoinKind,
    Limit,
    Literal,
    Op,
    OrderKey,
    Project,
    Scan,
    Sort,
    SortDirection,
)
from manysql.ir.types import FLOAT, INT, TEXT, BOOL
from manysql.oracle import ReferenceInterpreter, frames_equal
from manysql.spec import SemanticConfig
from manysql.storage import schema_of, seed_datasets


def _agree(plan, semantics: SemanticConfig | None = None, *, ordered: bool = False):
    semantics = semantics or SemanticConfig.reference()
    catalog = seed_datasets()
    pl_result = execute(plan, semantics, catalog)
    ref_result = ReferenceInterpreter().evaluate(plan, semantics, catalog)
    assert ref_result.error is None, f"reference oracle errored: {ref_result.error}"
    assert ref_result.rows is not None
    eq, reason = frames_equal(pl_result, ref_result.rows, ordered=ordered)
    assert eq, f"executor vs reference disagree: {reason}\nexecutor:\n{pl_result}\nreference:\n{ref_result.rows}"


def test_simple_scan_agrees() -> None:
    plan = Scan(table_name="employees", columns=schema_of("employees"))
    _agree(plan)


def test_filter_active() -> None:
    emp = Scan(table_name="employees", columns=schema_of("employees"))
    plan = Filter(input=emp, predicate=ColumnRef("active"))
    _agree(plan)


def test_project_arithmetic() -> None:
    emp = Scan(table_name="employees", columns=schema_of("employees"))
    plan = Project(
        input=emp,
        projections=(
            ("name", ColumnRef("name")),
            ("bonus", BinaryOp(Op.MUL, ColumnRef("salary"), Literal(0.1, FLOAT))),
        ),
        output_types=(TEXT, FLOAT),
    )
    _agree(plan)


def test_aggregate_count_per_dept() -> None:
    emp = Scan(table_name="employees", columns=schema_of("employees"))
    plan = Aggregate(
        input=emp,
        group_by=(("dept_id", ColumnRef("dept_id")),),
        aggregates=(("n", AggCall(AggKind.COUNT_STAR)),),
        output_types=(INT, INT),
    )
    _agree(plan)


def test_aggregate_sum_avg_with_nulls() -> None:
    emp = Scan(table_name="employees", columns=schema_of("employees"))
    plan = Aggregate(
        input=emp,
        group_by=(("dept_id", ColumnRef("dept_id")),),
        aggregates=(
            ("total", AggCall(AggKind.SUM, arg=ColumnRef("salary"))),
            ("avg_sal", AggCall(AggKind.AVG, arg=ColumnRef("salary"))),
            ("n", AggCall(AggKind.COUNT_STAR)),
        ),
        output_types=(INT, FLOAT, FLOAT, INT),
    )
    _agree(plan)


def test_sort_default_null_order() -> None:
    emp = Scan(table_name="employees", columns=schema_of("employees"))
    plan = Sort(
        input=emp,
        keys=(OrderKey(expr=ColumnRef("salary"), direction=SortDirection.ASC),),
    )
    _agree(plan, ordered=True)


def test_sort_with_ties_uses_secondary_key() -> None:
    sales = Scan(table_name="sales", columns=schema_of("sales"))
    plan = Sort(
        input=sales,
        keys=(
            OrderKey(expr=ColumnRef("sold_on"), direction=SortDirection.ASC),
            OrderKey(expr=ColumnRef("id"), direction=SortDirection.ASC),
        ),
    )
    _agree(plan, ordered=True)


def test_limit_offset() -> None:
    emp = Scan(table_name="employees", columns=schema_of("employees"))
    sorted_emp = Sort(
        input=emp,
        keys=(OrderKey(expr=ColumnRef("id"), direction=SortDirection.ASC),),
    )
    plan = Limit(input=sorted_emp, limit=3, offset=1)
    _agree(plan, ordered=True)


def test_inner_join_employees_to_departments() -> None:
    emp = Scan(table_name="employees", columns=schema_of("employees"), alias="e")
    dept = Scan(table_name="departments", columns=schema_of("departments"), alias="d")
    plan = Join(
        left=emp,
        right=dept,
        kind=JoinKind.INNER,
        on=BinaryOp(Op.EQ, ColumnRef("dept_id", "e"), ColumnRef("id", "d")),
    )
    _agree(plan)


def test_left_join_keeps_unassigned_employee() -> None:
    emp = Scan(table_name="employees", columns=schema_of("employees"), alias="e")
    dept = Scan(table_name="departments", columns=schema_of("departments"), alias="d")
    plan = Join(
        left=emp,
        right=dept,
        kind=JoinKind.LEFT,
        on=BinaryOp(Op.EQ, ColumnRef("dept_id", "e"), ColumnRef("id", "d")),
    )
    _agree(plan)


def test_division_by_zero_null_mode() -> None:
    sales = Scan(table_name="sales", columns=schema_of("sales"))
    plan = Project(
        input=sales,
        projections=(
            ("id", ColumnRef("id")),
            ("amt_per_zero", BinaryOp(Op.DIV, ColumnRef("amount"), Literal(0.0, FLOAT))),
        ),
        output_types=(INT, FLOAT),
    )
    _agree(plan)
