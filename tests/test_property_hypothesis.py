"""Hypothesis-driven property fuzzer.

For a fixed set of "structurally-interesting" plans (one per IR node),
Hypothesis generates many random catalogs that satisfy the same schema
as the reference catalog. We run the executor on each random catalog
and assert the PropertyOracle invariants hold.

This complements the golden-query battery in two ways:
1. Coverage breadth: the executor sees many distinct inputs per plan,
   not just the hand-curated reference dataset.
2. Cheap signal: properties are O(1) per plan and don't depend on any
   external SQL engine, so we can run hundreds of trials per CI minute.

The strategies are intentionally small (≤ 6 rows per table) to keep
each example cheap. They cover:
- nulls in every nullable column
- ties on sort keys
- empty groups (some dept_id values absent from employees)
- duplicate rows (so DISTINCT actually has work to do)
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from lark import Lark

from manysql.dialects import DialectRegistry
from manysql.executor import execute
from manysql.ir.expr import (
    AggCall,
    AggKind,
    BinaryOp,
    ColumnRef,
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
    Plan,
    Project,
    Scan,
    SetOp,
    SetOpKind,
    Sort,
)
from manysql.ir.types import BOOL, FLOAT, INT, TEXT
from manysql.oracle import PropertyOracle
from manysql.spec.semantics import SemanticConfig
from manysql.storage import schema_of


# ---- Catalog strategies --------------------------------------------------


_PROFILE = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


def _names() -> st.SearchStrategy[Optional[str]]:
    """Short, occasionally-null string values."""
    return st.one_of(
        st.none(),
        st.sampled_from(["alice", "bob", "carol", "dave", "eve", "frank"]),
    )


def _opt_int(min_value: int, max_value: int) -> st.SearchStrategy[Optional[int]]:
    return st.one_of(st.none(), st.integers(min_value=min_value, max_value=max_value))


def _opt_float() -> st.SearchStrategy[Optional[float]]:
    return st.one_of(
        st.none(),
        st.floats(
            min_value=0.0,
            max_value=1_000_000.0,
            allow_nan=False,
            allow_infinity=False,
            width=64,
        ),
    )


def _date_strategy() -> st.SearchStrategy[date]:
    base = date(2018, 1, 1)
    return st.integers(min_value=0, max_value=2500).map(lambda d: base + timedelta(days=d))


@st.composite
def _employees_strategy(draw) -> pl.DataFrame:  # noqa: ANN001
    n = draw(st.integers(min_value=1, max_value=6))
    ids = list(range(1, n + 1))
    names = [draw(_names()) for _ in range(n)]
    dept_ids = [draw(_opt_int(10, 40)) for _ in range(n)]
    manager_ids = [draw(_opt_int(1, 8)) for _ in range(n)]
    salary = [draw(_opt_float()) for _ in range(n)]
    hired_on = [draw(_date_strategy()) for _ in range(n)]
    active = [draw(st.booleans()) for _ in range(n)]
    return pl.DataFrame(
        {
            "id": ids,
            "name": names,
            "dept_id": dept_ids,
            "manager_id": manager_ids,
            "salary": salary,
            "hired_on": hired_on,
            "active": active,
        },
        schema={
            "id": pl.Int64,
            "name": pl.Utf8,
            "dept_id": pl.Int64,
            "manager_id": pl.Int64,
            "salary": pl.Float64,
            "hired_on": pl.Date,
            "active": pl.Boolean,
        },
    )


@st.composite
def _departments_strategy(draw) -> pl.DataFrame:  # noqa: ANN001
    n = draw(st.integers(min_value=1, max_value=4))
    return pl.DataFrame(
        {
            "id": [10 * (i + 1) for i in range(n)],
            "name": [draw(st.sampled_from(["Eng", "Sales", "RnD", "Mkt"])) for _ in range(n)],
            "region_id": [draw(_opt_int(1, 3)) for _ in range(n)],
            "budget": [draw(_opt_float()) for _ in range(n)],
        },
        schema={
            "id": pl.Int64,
            "name": pl.Utf8,
            "region_id": pl.Int64,
            "budget": pl.Float64,
        },
    )


@st.composite
def _regions_strategy(draw) -> pl.DataFrame:  # noqa: ANN001
    n = draw(st.integers(min_value=1, max_value=3))
    return pl.DataFrame(
        {
            "id": list(range(1, n + 1)),
            "name": [draw(st.sampled_from(["NA", "EU", "APAC"])) for _ in range(n)],
        },
        schema={"id": pl.Int64, "name": pl.Utf8},
    )


@st.composite
def _sales_strategy(draw) -> pl.DataFrame:  # noqa: ANN001
    n = draw(st.integers(min_value=0, max_value=6))
    return pl.DataFrame(
        {
            "id": list(range(101, 101 + n)),
            "employee_id": [draw(_opt_int(1, 6)) for _ in range(n)],
            "amount": [draw(_opt_float()) for _ in range(n)],
            "sold_on": [draw(_date_strategy()) for _ in range(n)],
            "region_id": [draw(_opt_int(1, 3)) for _ in range(n)],
        },
        schema={
            "id": pl.Int64,
            "employee_id": pl.Int64,
            "amount": pl.Float64,
            "sold_on": pl.Date,
            "region_id": pl.Int64,
        },
    )


@st.composite
def _categories_strategy(draw) -> pl.DataFrame:  # noqa: ANN001
    n = draw(st.integers(min_value=1, max_value=5))
    parents: list[Optional[int]] = [None]  # root
    for i in range(2, n + 1):
        parents.append(draw(st.sampled_from([None] + list(range(1, i)))))
    return pl.DataFrame(
        {
            "id": list(range(1, n + 1)),
            "name": [f"node_{i}" for i in range(1, n + 1)],
            "parent_id": parents,
        },
        schema={"id": pl.Int64, "name": pl.Utf8, "parent_id": pl.Int64},
    )


@st.composite
def _catalog_strategy(draw) -> dict[str, pl.DataFrame]:  # noqa: ANN001
    return {
        "employees": draw(_employees_strategy()),
        "departments": draw(_departments_strategy()),
        "regions": draw(_regions_strategy()),
        "sales": draw(_sales_strategy()),
        "categories": draw(_categories_strategy()),
    }


# ---- Plans we'll fuzz ----------------------------------------------------


def _emp_scan(alias: Optional[str] = None) -> Scan:
    return Scan(table_name="employees", columns=schema_of("employees"), alias=alias)


def _dept_scan(alias: Optional[str] = None) -> Scan:
    return Scan(table_name="departments", columns=schema_of("departments"), alias=alias)


def _structural_plans() -> list[Plan]:
    """One plan per IR node we want to exercise structurally."""
    emp = _emp_scan()
    emp_e = _emp_scan(alias="e")
    dept_d = _dept_scan(alias="d")
    return [
        emp,
        Project(
            input=emp,
            projections=(("id", ColumnRef("id")), ("name", ColumnRef("name"))),
            output_types=(INT, TEXT),
        ),
        Filter(input=emp, predicate=ColumnRef("active")),
        Distinct(input=Project(
            input=emp,
            projections=(("dept_id", ColumnRef("dept_id")),),
            output_types=(INT,),
        )),
        Sort(
            input=emp,
            keys=(
                OrderKey(expr=ColumnRef("salary"), direction=SortDirection.DESC),
                OrderKey(expr=ColumnRef("id"), direction=SortDirection.ASC),
            ),
        ),
        Limit(input=emp, limit=2),
        Aggregate(
            input=emp,
            group_by=(),
            aggregates=(("n", AggCall(AggKind.COUNT_STAR)),),
            output_types=(INT,),
        ),
        Aggregate(
            input=emp,
            group_by=(("dept_id", ColumnRef("dept_id")),),
            aggregates=(
                ("n", AggCall(AggKind.COUNT_STAR)),
                ("total", AggCall(AggKind.SUM, arg=ColumnRef("salary"))),
            ),
            output_types=(INT, INT, FLOAT),
        ),
        Join(
            left=emp_e,
            right=dept_d,
            kind=JoinKind.INNER,
            on=BinaryOp(Op.EQ, ColumnRef("dept_id", "e"), ColumnRef("id", "d")),
        ),
        Join(
            left=emp_e,
            right=dept_d,
            kind=JoinKind.LEFT,
            on=BinaryOp(Op.EQ, ColumnRef("dept_id", "e"), ColumnRef("id", "d")),
        ),
        SetOp(
            left=Project(
                input=emp,
                projections=(("id", ColumnRef("id")),),
                output_types=(INT,),
            ),
            right=Project(
                input=_dept_scan(),
                projections=(("id", ColumnRef("id")),),
                output_types=(INT,),
            ),
            kind=SetOpKind.UNION,
            all=False,
        ),
    ]


_PLANS = _structural_plans()


@pytest.mark.parametrize(
    "plan_idx",
    range(len(_PLANS)),
    ids=[type(p).__name__ for p in _PLANS],
)
def test_hypothesis_property_invariants_hold(plan_idx: int) -> None:
    """For each structural plan, Hypothesis fuzzes catalogs and asserts
    the executor satisfies every property invariant."""

    plan = _PLANS[plan_idx]
    oracle = PropertyOracle()
    semantics = SemanticConfig.reference()

    @_PROFILE
    @given(catalog=_catalog_strategy())
    def runner(catalog: dict[str, pl.DataFrame]) -> None:
        actual = execute(plan, semantics, catalog)
        result = oracle.check_properties(plan, actual, semantics, catalog)
        assert result.property_passed is True, (
            f"[{type(plan).__name__}] property violation: {result.notes}; "
            f"catalog heights={ {k: v.height for k, v in catalog.items()} }"
        )

    runner()


# ---- A separate sanity test that uses the parser/grammar path -----------


@pytest.fixture(scope="module")
def reference_engine():
    return DialectRegistry().load("_reference")


@pytest.fixture(scope="module")
def reference_parser(reference_engine) -> Lark:
    return Lark(reference_engine.grammar_text, start="start", parser="earley")


_FUZZ_SQL = [
    "SELECT id FROM employees ORDER BY id",
    "SELECT DISTINCT dept_id FROM employees",
    "SELECT id FROM employees LIMIT 3",
    "SELECT dept_id, COUNT(*) FROM employees GROUP BY dept_id",
    "SELECT * FROM employees WHERE active",
]


@pytest.mark.parametrize("sql", _FUZZ_SQL)
def test_hypothesis_property_invariants_for_parsed_sql(
    sql: str, reference_engine, reference_parser: Lark
) -> None:
    """Same as above but starting from SQL strings to also exercise the
    parser/lowering path under random data."""

    schemas = {name: schema_of(name) for name in ("employees", "departments", "regions", "sales", "categories")}
    tree = reference_parser.parse(sql)
    plan = reference_engine.lowering.lower(tree, reference_engine.semantics, schemas)
    oracle = PropertyOracle()
    semantics = reference_engine.semantics

    @_PROFILE
    @given(catalog=_catalog_strategy())
    def runner(catalog: dict[str, pl.DataFrame]) -> None:
        actual = execute(plan, semantics, catalog)
        result = oracle.check_properties(plan, actual, semantics, catalog)
        assert result.property_passed is True, (
            f"[{sql}] property violation: {result.notes}"
        )

    runner()
