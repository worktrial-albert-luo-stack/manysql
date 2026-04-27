"""Lowering for the synthetic _test_with_ties dialect.

The real `lower` is a stub: this dialect doesn't have a SQL surface that
gets parsed in tests. Instead, tests construct IR plans directly via
`make_with_ties_marker`, which simulates what a real lowering would
emit for a `LIMIT N WITH TIES` clause: a `Filter` whose predicate is a
sentinel `FuncCall("__manysql_with_ties", [Literal(N)])` wrapping the
sorted plan.

The dialect's `passes.py` recognizes this marker and rewrites it into
canonical IR (Window(RANK) + Filter(rank <= N) + Project).
"""

from __future__ import annotations

from typing import Any

from manysql.ir.expr import FuncCall, Literal
from manysql.ir.plan import Filter, Plan, Sort
from manysql.ir.types import INT


WITH_TIES_MARKER_NAME = "__manysql_with_ties"


def make_with_ties_marker(sort: Sort, n: int) -> Filter:
    """Wrap a Sort plan with the sentinel marker that `passes.py` will desugar."""
    return Filter(
        input=sort,
        predicate=FuncCall(
            name=WITH_TIES_MARKER_NAME,
            args=(Literal(value=n, type=INT),),
        ),
    )


def lower(tree: Any, config: Any, catalog: Any) -> Plan:  # pragma: no cover - stub
    raise NotImplementedError(
        "_test_with_ties has no SQL surface; construct plans via "
        "make_with_ties_marker(sort, n) instead."
    )
