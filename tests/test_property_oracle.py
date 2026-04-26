"""Direct tests for the property-based oracle.

These tests construct deliberately-broken result frames and verify the
PropertyOracle catches each violation. Combined with the
Hypothesis-driven test (test_property_hypothesis.py), they keep the
property layer sharp without depending on the executor being correct.
"""

from __future__ import annotations

import polars as pl

from manysql.executor import execute
from manysql.ir.expr import (
    AggCall,
    AggKind,
    BinaryOp,
    ColumnRef,
    Literal,
    Op,
    OrderKey,
    SortDirection,
)
from manysql.ir.plan import (
    Aggregate,
    Distinct,
    Filter,
    Join,
    JoinKind,
    Limit,
    Project,
    Scan,
    SetOp,
    SetOpKind,
    Sort,
)
from manysql.ir.types import BOOL, FLOAT, INT
from manysql.oracle import PropertyOracle
from manysql.spec.semantics import SemanticConfig
from manysql.storage import schema_of, seed_datasets


def _oracle() -> PropertyOracle:
    return PropertyOracle()


def _semantics() -> SemanticConfig:
    return SemanticConfig.reference()


def _catalog() -> dict[str, pl.DataFrame]:
    return seed_datasets()


# ---- Schema invariants ---------------------------------------------------


def test_schema_width_mismatch_is_flagged() -> None:
    plan = Scan(table_name="employees", columns=schema_of("employees"))
    actual = pl.DataFrame({"only": [1]})
    result = _oracle().check_properties(plan, actual, _semantics(), _catalog())
    assert result.property_passed is False
    assert any("width mismatch" in n for n in result.notes)


def test_schema_column_name_mismatch_is_flagged() -> None:
    plan = Project(
        input=Scan(table_name="employees", columns=schema_of("employees")),
        projections=(("renamed", ColumnRef("id")),),
        output_types=(INT,),
    )
    actual = pl.DataFrame({"wrong": [1, 2, 3]})
    result = _oracle().check_properties(plan, actual, _semantics(), _catalog())
    assert result.property_passed is False
    assert any("column[0] name mismatch" in n for n in result.notes)


def test_schema_passes_with_qualifier_prefix() -> None:
    """The executor uses `{alias}__{col}` for qualified scans; the oracle
    must accept that form alongside the bare column name."""

    plan = Scan(
        table_name="employees", columns=schema_of("employees"), alias="e"
    )
    actual = execute(plan, _semantics(), _catalog())
    result = _oracle().check_properties(plan, actual, _semantics(), _catalog())
    assert result.property_passed is True, result.notes


# ---- Distinct ------------------------------------------------------------


def test_distinct_with_duplicates_is_flagged() -> None:
    plan = Distinct(
        input=Scan(table_name="employees", columns=schema_of("employees"))
    )
    catalog = _catalog()
    actual = pl.concat([catalog["employees"], catalog["employees"].head(2)])
    result = _oracle().check_properties(plan, actual, _semantics(), catalog)
    assert result.property_passed is False
    assert any("Distinct" in n and "duplicates" in n for n in result.notes)


# ---- Sort ----------------------------------------------------------------


def test_sort_unsorted_output_is_flagged() -> None:
    plan = Sort(
        input=Scan(table_name="employees", columns=schema_of("employees")),
        keys=(OrderKey(expr=ColumnRef("id"), direction=SortDirection.ASC),),
    )
    catalog = _catalog()
    actual = catalog["employees"].sort("id", descending=True)
    result = _oracle().check_properties(plan, actual, _semantics(), catalog)
    assert result.property_passed is False
    assert any("Sort" in n and "not sorted" in n for n in result.notes)


def test_sort_correctly_sorted_passes() -> None:
    plan = Sort(
        input=Scan(table_name="employees", columns=schema_of("employees")),
        keys=(OrderKey(expr=ColumnRef("id"), direction=SortDirection.ASC),),
    )
    catalog = _catalog()
    actual = catalog["employees"].sort("id")
    result = _oracle().check_properties(plan, actual, _semantics(), catalog)
    assert result.property_passed is True, result.notes


# ---- Limit ---------------------------------------------------------------


def test_limit_overflow_is_flagged() -> None:
    plan = Limit(
        input=Scan(table_name="employees", columns=schema_of("employees")),
        limit=2,
    )
    catalog = _catalog()
    actual = catalog["employees"]  # full table is far more than 2 rows
    result = _oracle().check_properties(plan, actual, _semantics(), catalog)
    assert result.property_passed is False
    assert any("Limit" in n for n in result.notes)


def test_limit_within_bound_passes() -> None:
    plan = Limit(
        input=Scan(table_name="employees", columns=schema_of("employees")),
        limit=2,
    )
    catalog = _catalog()
    actual = catalog["employees"].head(2)
    result = _oracle().check_properties(plan, actual, _semantics(), catalog)
    assert result.property_passed is True, result.notes


# ---- Aggregate -----------------------------------------------------------


def test_aggregate_no_group_by_must_be_one_row() -> None:
    plan = Aggregate(
        input=Scan(table_name="employees", columns=schema_of("employees")),
        group_by=(),
        aggregates=(("n", AggCall(AggKind.COUNT_STAR)),),
        output_types=(INT,),
    )
    actual = pl.DataFrame({"n": [1, 2, 3]})
    result = _oracle().check_properties(plan, actual, _semantics(), _catalog())
    assert result.property_passed is False
    assert any("Aggregate without GROUP BY" in n for n in result.notes)


def test_aggregate_with_group_by_must_have_unique_keys() -> None:
    plan = Aggregate(
        input=Scan(table_name="employees", columns=schema_of("employees")),
        group_by=(("dept_id", ColumnRef("dept_id")),),
        aggregates=(("n", AggCall(AggKind.COUNT_STAR)),),
        output_types=(INT, INT),
    )
    actual = pl.DataFrame({"dept_id": [10, 10, 20], "n": [1, 2, 3]})
    result = _oracle().check_properties(plan, actual, _semantics(), _catalog())
    assert result.property_passed is False
    assert any("duplicate group-key tuples" in n for n in result.notes)


# ---- SetOp ---------------------------------------------------------------


def test_setop_distinct_with_duplicates_is_flagged() -> None:
    inp = Scan(table_name="employees", columns=schema_of("employees"))
    plan = SetOp(left=inp, right=inp, kind=SetOpKind.UNION, all=False)
    catalog = _catalog()
    actual = pl.concat([catalog["employees"], catalog["employees"]])
    result = _oracle().check_properties(plan, actual, _semantics(), catalog)
    assert result.property_passed is False
    assert any("SetOp" in n for n in result.notes)


def test_setop_all_allows_duplicates() -> None:
    inp = Scan(table_name="employees", columns=schema_of("employees"))
    plan = SetOp(left=inp, right=inp, kind=SetOpKind.UNION, all=True)
    catalog = _catalog()
    actual = pl.concat([catalog["employees"], catalog["employees"]])
    result = _oracle().check_properties(plan, actual, _semantics(), catalog)
    assert result.property_passed is True, result.notes


# ---- Join ----------------------------------------------------------------


def test_join_inner_cardinality_upper_bound_is_flagged() -> None:
    left = Scan(table_name="employees", columns=schema_of("employees"))
    right = Scan(table_name="departments", columns=schema_of("departments"))
    plan = Join(
        left=left,
        right=right,
        kind=JoinKind.INNER,
        on=BinaryOp(Op.EQ, ColumnRef("dept_id"), ColumnRef("id")),
    )
    catalog = _catalog()
    rows_left = catalog["employees"].height
    rows_right = catalog["departments"].height
    cap = rows_left * rows_right
    fake = pl.DataFrame(
        {f"col{i}": [None] * (cap + 1) for i in range(plan.schema().__len__())}
    )
    result = _oracle().check_properties(plan, fake, _semantics(), catalog)
    assert result.property_passed is False
    assert any("INNER" in n or "width" in n.lower() for n in result.notes)


def test_join_left_cardinality_lower_bound_is_flagged() -> None:
    left = Scan(table_name="employees", columns=schema_of("employees"))
    right = Scan(table_name="departments", columns=schema_of("departments"))
    plan = Join(
        left=left,
        right=right,
        kind=JoinKind.LEFT,
        on=BinaryOp(Op.EQ, ColumnRef("dept_id"), ColumnRef("id")),
    )
    catalog = _catalog()
    width = len(plan.schema())
    fake = pl.DataFrame({f"col{i}": [None] for i in range(width)})
    result = _oracle().check_properties(plan, fake, _semantics(), catalog)
    assert result.property_passed is False
    assert any("LEFT" in n for n in result.notes)


# ---- Real executor: every property must hold ----------------------------


def test_property_oracle_passes_for_real_executor_outputs() -> None:
    """A small set of executor results should satisfy every property."""

    catalog = _catalog()
    semantics = _semantics()
    oracle = _oracle()

    plans: list = [
        Scan(table_name="employees", columns=schema_of("employees")),
        Distinct(input=Scan(table_name="departments", columns=schema_of("departments"))),
        Sort(
            input=Scan(table_name="employees", columns=schema_of("employees")),
            keys=(OrderKey(expr=ColumnRef("id"), direction=SortDirection.ASC),),
        ),
        Limit(
            input=Scan(table_name="employees", columns=schema_of("employees")),
            limit=3,
        ),
        Filter(
            input=Scan(table_name="employees", columns=schema_of("employees")),
            predicate=ColumnRef("active"),
        ),
        Aggregate(
            input=Scan(table_name="employees", columns=schema_of("employees")),
            group_by=(("dept_id", ColumnRef("dept_id")),),
            aggregates=(
                ("n", AggCall(AggKind.COUNT_STAR)),
                ("total", AggCall(AggKind.SUM, arg=ColumnRef("salary"))),
            ),
            output_types=(INT, INT, FLOAT),
        ),
    ]

    for plan in plans:
        actual = execute(plan, semantics, catalog)
        result = oracle.check_properties(plan, actual, semantics, catalog)
        assert result.property_passed is True, (
            f"plan {type(plan).__name__} failed properties: {result.notes}"
        )


# ---- Helper: width-only sanity for Project ----


def test_project_passes_for_correct_executor_output() -> None:
    plan = Project(
        input=Scan(table_name="employees", columns=schema_of("employees")),
        projections=(
            ("id", ColumnRef("id")),
            ("salary_plus_one", BinaryOp(Op.ADD, ColumnRef("salary"), Literal(1, FLOAT))),
            ("is_active", ColumnRef("active")),
        ),
        output_types=(INT, FLOAT, BOOL),
    )
    actual = execute(plan, _semantics(), _catalog())
    result = _oracle().check_properties(plan, actual, _semantics(), _catalog())
    assert result.property_passed is True, result.notes
