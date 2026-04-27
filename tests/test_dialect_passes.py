"""Tests for the per-dialect Plan-rewrite passes lane.

A dialect can ship a `passes.py` module that exposes
`PRE_EXECUTION_PASSES: list[Callable[[Plan, SemanticConfig], Plan]]`.
The executor applies them between lowering and dispatch.

We exercise the lane via the synthetic `_test_with_ties` dialect: its
`lowering.py` exposes a helper that produces a non-canonical IR marker,
and its `passes.py` desugars the marker into canonical Window+Filter+
Project that the canonical executor knows how to run.
"""

from __future__ import annotations

import polars as pl
import pytest

from manysql.dialects import DialectRegistry
from manysql.executor import apply_pre_passes, execute
from manysql.ir.expr import (
    ColumnRef,
    OrderKey,
    SortDirection,
)
from manysql.ir.plan import (
    ColumnSchema,
    Filter,
    Project,
    Scan,
    Sort,
)
from manysql.ir.types import INT
from manysql.spec.semantics import SemanticConfig


@pytest.fixture(scope="module")
def with_ties_engine():
    return DialectRegistry().load("_test_with_ties")


def _build_marker_plan(n: int) -> Filter:
    """Build the IR a real lowering would emit for `LIMIT N WITH TIES`."""
    from manysql.dialects._test_with_ties.lowering import make_with_ties_marker

    scan = Scan(
        table_name="t",
        columns=(
            ColumnSchema(name="x", type=INT),
            ColumnSchema(name="y", type=INT),
        ),
    )
    sort = Sort(
        input=scan,
        keys=(OrderKey(expr=ColumnRef(name="x"), direction=SortDirection.ASC),),
    )
    return make_with_ties_marker(sort, n)


def _catalog() -> dict[str, pl.DataFrame]:
    # Three rows tied at x=1 (the lowest), then x=2 and x=3. LIMIT 2 WITH TIES
    # asks for the top two, but RANK() collapses all three 1s to rank 1, so
    # the rewrite must include all three.
    return {
        "t": pl.DataFrame({"x": [1, 1, 1, 2, 3], "y": [10, 11, 12, 20, 30]})
    }


def test_engine_loads_passes_module(with_ties_engine) -> None:
    assert with_ties_engine.passes is not None
    seq = with_ties_engine.passes.PRE_EXECUTION_PASSES
    assert len(seq) == 1
    assert callable(seq[0])


def test_apply_pre_passes_rewrites_marker_into_canonical_ir(
    with_ties_engine,
) -> None:
    plan = _build_marker_plan(n=2)
    rewritten = apply_pre_passes(
        plan, SemanticConfig.reference(), with_ties_engine.passes
    )
    # Rewritten root is Project; its input is Filter (rank <= n); whose
    # input is Window (rank()); whose input is the original Sort.
    from manysql.ir.plan import Filter as F
    from manysql.ir.plan import Project as P
    from manysql.ir.plan import Sort as S
    from manysql.ir.plan import Window as W

    assert isinstance(rewritten, P)
    assert isinstance(rewritten.input, F)
    assert isinstance(rewritten.input.input, W)
    assert isinstance(rewritten.input.input.input, S)
    # Output schema preserved (no leaked rank column).
    assert [n for n, _ in rewritten.projections] == ["x", "y"]


def test_with_ties_executes_to_top_n_plus_ties(with_ties_engine) -> None:
    """LIMIT 2 WITH TIES on (1,1,1,2,3) keeps all three 1s.

    RANK() over ORDER BY x assigns rank 1 to every x=1, then rank 4 to x=2,
    then rank 5 to x=3. Filtering rank <= 2 keeps every row tied with the
    2nd row (which itself has rank 1), so the three 1s come through.
    """
    plan = _build_marker_plan(n=2)
    out = execute(
        plan,
        SemanticConfig.reference(),
        _catalog(),
        passes=with_ties_engine.passes,
    )
    assert sorted(out["x"].to_list()) == [1, 1, 1]
    assert sorted(out["y"].to_list()) == [10, 11, 12]


def test_passes_unwired_leaves_marker_in_plan(with_ties_engine) -> None:
    """Without `passes=`, the canonical executor sees the marker and fails.

    This is the negative control for the lane: the dialect-specific marker
    is meaningless to the canonical executor; only the pass desugars it.
    """
    plan = _build_marker_plan(n=2)
    with pytest.raises(NotImplementedError):
        execute(plan, SemanticConfig.reference(), _catalog())


def test_apply_pre_passes_no_module_is_identity() -> None:
    """`passes_module=None` returns the input plan unchanged."""
    plan = _build_marker_plan(n=2)
    assert apply_pre_passes(plan, SemanticConfig.reference(), None) is plan


def test_apply_pre_passes_empty_list_is_identity() -> None:
    class EmptyPasses:
        PRE_EXECUTION_PASSES: list = []

    plan = _build_marker_plan(n=2)
    assert (
        apply_pre_passes(plan, SemanticConfig.reference(), EmptyPasses())
        is plan
    )


def test_pass_returning_none_raises() -> None:
    class BadPasses:
        @staticmethod
        def _bad(plan, semantics):  # noqa: ARG001
            return None

        PRE_EXECUTION_PASSES = [_bad]

    plan = _build_marker_plan(n=2)
    with pytest.raises(RuntimeError, match="returned None"):
        apply_pre_passes(plan, SemanticConfig.reference(), BadPasses())


def test_oracle_harness_forwards_passes_when_materializing_actual(
    with_ties_engine,
) -> None:
    """When `actual=None`, OracleHarness.verify must forward `passes` so the
    canonical executor can desugar dialect-specific markers before running.

    Without forwarding, the harness's own `polars_execute` call would raise
    `NotImplementedError` on the marker FuncCall, masking the dialect's
    intent.
    """
    from manysql.oracle import OracleHarness

    plan = _build_marker_plan(n=2)
    harness = OracleHarness(oracles=[], property_oracles=[])
    report = harness.verify(
        plan,
        SemanticConfig.reference(),
        _catalog(),
        passes=with_ties_engine.passes,
    )
    # Without oracles the verdict is NO_ORACLE, but `actual` must have
    # been materialized through the pass.
    assert report.actual is not None
    assert sorted(report.actual["x"].to_list()) == [1, 1, 1]
