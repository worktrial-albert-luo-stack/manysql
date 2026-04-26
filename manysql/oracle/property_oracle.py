"""Property-based oracle: structural invariants for any IR plan output.

Unlike the row-producing oracles (DuckDB, SQLite, ReferenceInterpreter),
this oracle does not produce expected rows. Instead it asserts a set of
*structural properties* on the actual result frame: properties any correct
executor must satisfy regardless of dialect or input data.

Why this matters:
    Row-producing oracles need a parallel implementation of every
    semantic knob. They are expensive to build and maintain, and they
    can't run on plans whose semantics no engine supports. Properties
    are cheap, run on every plan, and catch entire classes of bugs:

    - "Distinct didn't deduplicate"
    - "Sort returned unsorted rows"
    - "Limit didn't actually limit"
    - "Aggregate without GROUP BY produced 0 or >1 rows"
    - "Aggregate with GROUP BY had duplicate group-key tuples"
    - "Project produced the wrong number of columns"
    - "Join SEMI/ANTI returned right-side columns"

The oracle is intentionally pure-Python: it inspects only the actual
DataFrame, the plan, and the catalog cardinalities. It does not
re-execute subplans, so it cannot catch wrong row *contents* — only
structural violations. That trade-off is the point: cheap, always-
applicable, complementary to row oracles.

Hypothesis integration: see tests/test_property_hypothesis.py for the
fuzzer that generates random catalogs and asserts these properties hold
for every golden plan.
"""

from __future__ import annotations

from typing import Any, Optional

import polars as pl

from manysql.ir.expr import OrderKey, SortDirection
from manysql.ir.plan import (
    Aggregate,
    Apply,
    ApplyKind,
    Distinct,
    Filter,
    Join,
    JoinKind,
    Limit,
    Plan,
    Project,
    RecursiveCTE,
    Scan,
    SetOp,
    Sort,
    Window,
    WithCTE,
)
from manysql.oracle.base import Oracle, OracleCapability, OracleResult
from manysql.spec.semantics import NullOrder, SemanticConfig


class PropertyOracle(Oracle):
    """Structural-invariants oracle. Always applicable. Never produces rows.

    `evaluate(plan, semantics, catalog)` is called by the harness with the
    same signature as row oracles, but it returns `OracleResult` with
    `rows=None` and `property_passed=True/False`. Note strings explain
    each violation.
    """

    @property
    def capability(self) -> OracleCapability:
        return OracleCapability(
            name="property",
            supported_nodes=frozenset(
                {
                    "Scan",
                    "Project",
                    "Filter",
                    "Join",
                    "Aggregate",
                    "Window",
                    "Sort",
                    "Limit",
                    "Distinct",
                    "SetOp",
                    "WithCTE",
                    "RecursiveCTE",
                    "Apply",
                }
            ),
            supported_features=frozenset({"structural_invariants"}),
            confidence=0.4,
        )

    def evaluate(
        self,
        plan: Plan,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
    ) -> OracleResult:
        # The harness only invokes evaluate with `actual` separately;
        # property oracles need actual to inspect. We expose
        # `check_properties` as the real entrypoint.
        return OracleResult(
            oracle=self.capability.name,
            rows=None,
            property_passed=None,
            notes=[
                "PropertyOracle.evaluate called without actual; use "
                "PropertyOracle.check_properties(plan, actual, ...)",
            ],
        )

    def check_properties(
        self,
        plan: Plan,
        actual: pl.DataFrame,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
    ) -> OracleResult:
        """Run all applicable structural checks on `actual`."""
        violations: list[str] = []
        violations.extend(_check_schema(plan, actual))
        violations.extend(_check_top_level_invariants(plan, actual, semantics, catalog))
        return OracleResult(
            oracle=self.capability.name,
            rows=None,
            property_passed=not violations,
            notes=violations,
        )


# ---------- Schema invariants ---------------------------------------------


def _check_schema(plan: Plan, actual: pl.DataFrame) -> list[str]:
    """Width and column-name checks against `plan.schema()`.

    Names are compared case-insensitively to tolerate dialects whose
    surface produces upper/lower-case output. We tolerate the executor's
    qualifier-prefix convention (`{alias}__{name}`) by accepting either
    the bare column name or the prefixed form for qualified columns.
    """
    schema = plan.schema()
    out: list[str] = []
    if actual.width != len(schema):
        out.append(
            f"width mismatch: actual={actual.width} cols, plan.schema()={len(schema)} cols"
        )
        return out
    actual_lower = [c.lower() for c in actual.columns]
    for i, col in enumerate(schema):
        bare = col.name.lower()
        prefixed = (
            f"{col.qualifier}__{col.name}".lower() if col.qualifier else None
        )
        if actual_lower[i] != bare and actual_lower[i] != prefixed:
            out.append(
                f"column[{i}] name mismatch: actual={actual.columns[i]!r}, "
                f"expected={col.name!r}"
                + (f" (qualifier {col.qualifier!r})" if col.qualifier else "")
            )
    return out


# ---------- Top-level node invariants -------------------------------------


def _check_top_level_invariants(
    plan: Plan,
    actual: pl.DataFrame,
    semantics: SemanticConfig,
    catalog: dict[str, pl.DataFrame],
) -> list[str]:
    if isinstance(plan, Distinct):
        return _check_distinct(actual)
    if isinstance(plan, Sort):
        return _check_sort(plan, actual, semantics)
    if isinstance(plan, Limit):
        return _check_limit(plan, actual)
    if isinstance(plan, Aggregate):
        return _check_aggregate(plan, actual)
    if isinstance(plan, SetOp):
        return _check_set_op(plan, actual)
    if isinstance(plan, Scan):
        return _check_scan(plan, actual, catalog)
    if isinstance(plan, Project):
        return _check_project(plan, actual)
    if isinstance(plan, Window):
        return _check_window(plan, actual)
    if isinstance(plan, Join):
        return _check_join(plan, actual, semantics, catalog)
    if isinstance(plan, (Filter, WithCTE, RecursiveCTE, Apply)):
        return []
    return []


def _check_distinct(actual: pl.DataFrame) -> list[str]:
    if actual.is_empty():
        return []
    deduped = actual.unique(maintain_order=False)
    if deduped.height != actual.height:
        return [
            f"Distinct: result has duplicates ({actual.height - deduped.height} extra rows)"
        ]
    return []


def _check_sort(plan: Sort, actual: pl.DataFrame, semantics: SemanticConfig) -> list[str]:
    if actual.height < 2:
        return []
    keys = plan.keys
    sorted_df = _polars_sort_for_keys(actual, keys, semantics)
    if not sorted_df.equals(actual):
        return ["Sort: rows are not sorted by the specified keys"]
    return []


def _polars_sort_for_keys(
    df: pl.DataFrame,
    keys: tuple[OrderKey, ...],
    semantics: SemanticConfig,
) -> pl.DataFrame:
    """Re-sort `df` by `keys` honoring the dialect's null ordering, then
    return that. Tests against this re-sort to detect missorted output."""
    by: list[str] = []
    descending: list[bool] = []
    nulls_last: list[bool] = []
    for k in keys:
        col_name = _order_key_column_name(k, df)
        if col_name is None:
            return df  # cannot resolve a key to a column → bail (no violation reported)
        by.append(col_name)
        descending.append(k.direction == SortDirection.DESC)
        nulls_last.append(_resolved_nulls_last(k, semantics))
    return df.sort(by=by, descending=descending, nulls_last=nulls_last)


def _resolved_nulls_last(key: OrderKey, semantics: SemanticConfig) -> bool:
    from manysql.ir.expr import NullsOrder

    if key.nulls is NullsOrder.FIRST:
        return False
    if key.nulls is NullsOrder.LAST:
        return True
    if key.direction == SortDirection.ASC:
        return semantics.null_order_default_asc == NullOrder.LAST
    return semantics.null_order_default_desc == NullOrder.LAST


def _order_key_column_name(key: OrderKey, df: pl.DataFrame) -> Optional[str]:
    """Best-effort resolution of an OrderKey to a column name.

    For ColumnRef order keys this is exact. For arbitrary expressions we
    return None and the sort check is skipped — callers treat that as
    'cannot verify, so don't fail'.
    """
    from manysql.ir.expr import ColumnRef

    expr = key.expr
    if isinstance(expr, ColumnRef):
        candidates = [expr.name]
        if expr.qualifier:
            candidates.append(f"{expr.qualifier}.{expr.name}")
        for c in candidates:
            if c in df.columns:
                return c
        lowered = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c.lower() in lowered:
                return lowered[c.lower()]
    return None


def _check_limit(plan: Limit, actual: pl.DataFrame) -> list[str]:
    if plan.limit is None:
        return []
    if actual.height > plan.limit:
        return [f"Limit: height={actual.height} > limit={plan.limit}"]
    return []


def _check_aggregate(plan: Aggregate, actual: pl.DataFrame) -> list[str]:
    if not plan.group_by:
        if actual.height != 1:
            return [
                f"Aggregate without GROUP BY: expected exactly 1 row, got {actual.height}"
            ]
        return []
    # With GROUP BY: group-key tuples must be unique.
    key_cols = [name for name, _ in plan.group_by]
    missing = [c for c in key_cols if c not in actual.columns]
    if missing:
        return [
            f"Aggregate: group-by columns missing from result: {missing} "
            f"(actual columns: {actual.columns})"
        ]
    if actual.height == 0:
        return []
    deduped = actual.select(key_cols).unique(maintain_order=False)
    if deduped.height != actual.height:
        return [
            f"Aggregate: duplicate group-key tuples in result "
            f"({actual.height - deduped.height} extra rows)"
        ]
    return []


def _check_set_op(plan: SetOp, actual: pl.DataFrame) -> list[str]:
    if plan.all:
        return []
    if actual.is_empty():
        return []
    deduped = actual.unique(maintain_order=False)
    if deduped.height != actual.height:
        return [
            f"SetOp ({plan.kind.value}, distinct): result has duplicates "
            f"({actual.height - deduped.height} extra rows)"
        ]
    return []


def _check_scan(
    plan: Scan,
    actual: pl.DataFrame,
    catalog: dict[str, pl.DataFrame],
) -> list[str]:
    if plan.table_name not in catalog:
        return [f"Scan: table {plan.table_name!r} not in catalog"]
    expected_height = catalog[plan.table_name].height
    if actual.height != expected_height:
        return [
            f"Scan({plan.table_name}): height={actual.height} != "
            f"catalog height={expected_height}"
        ]
    return []


def _check_project(plan: Project, actual: pl.DataFrame) -> list[str]:
    expected_width = len(plan.projections)
    if actual.width != expected_width:
        return [
            f"Project: width={actual.width} != "
            f"len(projections)={expected_width}"
        ]
    expected_names = [name for name, _ in plan.projections]
    if [c.lower() for c in actual.columns] != [n.lower() for n in expected_names]:
        return [
            f"Project: column names {actual.columns} != "
            f"projection names {expected_names}"
        ]
    return []


def _check_window(plan: Window, actual: pl.DataFrame) -> list[str]:
    expected_width = len(plan.input.schema()) + len(plan.windows)
    if actual.width != expected_width:
        return [
            f"Window: width={actual.width} != "
            f"input.width + len(windows)={expected_width}"
        ]
    return []


def _check_join(
    plan: Join,
    actual: pl.DataFrame,
    semantics: SemanticConfig,  # noqa: ARG001
    catalog: dict[str, pl.DataFrame],
) -> list[str]:
    if plan.kind in (JoinKind.SEMI, JoinKind.ANTI):
        expected_width = len(plan.left.schema())
    else:
        expected_width = len(plan.left.schema()) + len(plan.right.schema())
    if actual.width != expected_width:
        return [
            f"Join({plan.kind.value}): width={actual.width} != "
            f"expected={expected_width}"
        ]
    # Cardinality bounds when both sides are direct Scans (cheap to compute):
    left_height = _cheap_height(plan.left, catalog)
    right_height = _cheap_height(plan.right, catalog)
    if left_height is None or right_height is None:
        return []
    if plan.kind == JoinKind.CROSS:
        expected = left_height * right_height
        if actual.height != expected:
            return [
                f"Join(CROSS): height={actual.height} != "
                f"left*right={expected}"
            ]
    elif plan.kind == JoinKind.INNER:
        upper = left_height * right_height
        if actual.height > upper:
            return [
                f"Join(INNER): height={actual.height} > left*right={upper}"
            ]
    elif plan.kind == JoinKind.LEFT:
        if actual.height < left_height:
            return [
                f"Join(LEFT): height={actual.height} < left.height={left_height}"
            ]
    elif plan.kind == JoinKind.RIGHT:
        if actual.height < right_height:
            return [
                f"Join(RIGHT): height={actual.height} < right.height={right_height}"
            ]
    elif plan.kind in (JoinKind.SEMI, JoinKind.ANTI):
        if actual.height > left_height:
            return [
                f"Join({plan.kind.value}): height={actual.height} > "
                f"left.height={left_height}"
            ]
    return []


def _cheap_height(plan: Plan, catalog: dict[str, pl.DataFrame]) -> Optional[int]:
    """Cardinality only when we can read it directly. Returns None if we
    can't compute without re-running the executor."""
    if isinstance(plan, Scan):
        df = catalog.get(plan.table_name)
        return df.height if df is not None else None
    return None


__all__ = ["PropertyOracle"]
