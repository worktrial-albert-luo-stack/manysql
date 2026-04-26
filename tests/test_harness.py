"""Multi-oracle harness: all three oracles agree on standard plans."""

from __future__ import annotations

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
from manysql.ir.types import FLOAT, INT, TEXT
from manysql.oracle import OracleHarness, Verdict
from manysql.spec import SemanticConfig
from manysql.storage import schema_of, seed_datasets


def _verify(plan, semantics: SemanticConfig | None = None) -> None:
    semantics = semantics or SemanticConfig.reference()
    catalog = seed_datasets()
    report = OracleHarness().verify(plan, semantics, catalog)
    assert report.verdict == Verdict.PASS, (
        f"verdict={report.verdict} primary={report.primary} "
        f"reason={report.actual_vs_primary_reason} "
        f"disagreements={report.inter_oracle_disagreements} "
        f"errors={[r.error for r in report.oracle_results if r.error]}"
    )


def test_harness_agrees_simple_scan() -> None:
    plan = Scan(table_name="employees", columns=schema_of("employees"))
    _verify(plan)


def test_harness_agrees_filter() -> None:
    emp = Scan(table_name="employees", columns=schema_of("employees"))
    plan = Filter(input=emp, predicate=ColumnRef("active"))
    _verify(plan)


def test_harness_agrees_aggregate() -> None:
    emp = Scan(table_name="employees", columns=schema_of("employees"))
    plan = Aggregate(
        input=emp,
        group_by=(("dept_id", ColumnRef("dept_id")),),
        aggregates=(
            ("n", AggCall(AggKind.COUNT_STAR)),
            ("total", AggCall(AggKind.SUM, arg=ColumnRef("salary"))),
        ),
        output_types=(INT, INT, FLOAT),
    )
    _verify(plan)


def test_harness_agrees_inner_join() -> None:
    emp = Scan(table_name="employees", columns=schema_of("employees"), alias="e")
    dept = Scan(table_name="departments", columns=schema_of("departments"), alias="d")
    plan = Join(
        left=emp,
        right=dept,
        kind=JoinKind.INNER,
        on=BinaryOp(Op.EQ, ColumnRef("dept_id", "e"), ColumnRef("id", "d")),
    )
    _verify(plan)


def test_harness_agrees_sort_limit() -> None:
    emp = Scan(table_name="employees", columns=schema_of("employees"))
    sorted_plan = Sort(
        input=emp,
        keys=(
            OrderKey(expr=ColumnRef("salary"), direction=SortDirection.DESC),
            OrderKey(expr=ColumnRef("id"), direction=SortDirection.ASC),
        ),
    )
    plan = Limit(input=sorted_plan, limit=3)
    _verify(plan)


def test_harness_handles_div_by_zero_null() -> None:
    sales = Scan(table_name="sales", columns=schema_of("sales"))
    plan = Project(
        input=sales,
        projections=(
            ("id", ColumnRef("id")),
            ("z", BinaryOp(Op.DIV, ColumnRef("amount"), Literal(0.0, FLOAT))),
        ),
        output_types=(INT, FLOAT),
    )
    _verify(plan)
