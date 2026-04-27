"""Plan-rewrite pass for _test_with_ties.

Pattern recognized:

    Filter(
      input=Sort(input=<X>, keys=<K>),
      predicate=FuncCall("__manysql_with_ties", [Literal(N)])
    )

Rewritten to:

    Project(
      input=Filter(
        input=Window(
          input=Sort(input=<X>, keys=<K>),
          windows=[("__manysql_rk", WindowCall(RANK, order_by=<K>))]
        ),
        predicate=BinaryOp(LTE, ColumnRef("__manysql_rk"), Literal(N))
      ),
      projections=<original schema columns mapped through unchanged>
    )

The pass walks the entire plan tree, rewriting any subtree that matches
the marker pattern. Other plan shapes pass through unchanged.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

from manysql.ir.expr import (
    BinaryOp,
    ColumnRef,
    FuncCall,
    Literal,
    Op,
    WindowCall,
    WindowKind,
)
from manysql.ir.plan import (
    Aggregate,
    Apply,
    CTEBinding,
    Distinct,
    Filter,
    Join,
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
from manysql.ir.types import INT, IRType, TypeKind
from manysql.spec.semantics import SemanticConfig


_MARKER_NAME = "__manysql_with_ties"
_RANK_COL = "__manysql_rk"


def _is_marker(predicate) -> bool:
    return (
        isinstance(predicate, FuncCall)
        and predicate.name == _MARKER_NAME
        and len(predicate.args) == 1
        and isinstance(predicate.args[0], Literal)
    )


def _materialize_with_ties(sort: Sort, n: int) -> Plan:
    original_schema = sort.schema()
    original_types = tuple(c.type for c in original_schema)
    original_names = tuple(c.name for c in original_schema)

    rank_call = WindowCall(
        kind=WindowKind.RANK,
        args=(),
        partition_by=(),
        order_by=sort.keys,
    )
    rank_type = IRType(kind=TypeKind.INT, nullable=True)
    with_rank = Window(
        input=sort,
        windows=((_RANK_COL, rank_call),),
        output_types=(rank_type,),
    )
    bounded = Filter(
        input=with_rank,
        predicate=BinaryOp(
            op=Op.LTE,
            left=ColumnRef(name=_RANK_COL),
            right=Literal(value=n, type=INT),
        ),
    )
    return Project(
        input=bounded,
        projections=tuple(
            (name, ColumnRef(name=name)) for name in original_names
        ),
        output_types=original_types,
    )


def _map_children(plan: Plan, fn: Callable[[Plan], Plan]) -> Plan:
    """Apply ``fn`` to each immediate child Plan.

    The recursion is explicit per Plan subtype so we don't depend on
    any reflection over field types: every child is a typed slot and
    the IR is closed (see manysql/ir/plan.py).
    """
    if isinstance(plan, Scan):
        return plan
    if isinstance(
        plan,
        (Aggregate, Distinct, Filter, Limit, Project, Sort, Window),
    ):
        return dataclasses.replace(plan, input=fn(plan.input))
    if isinstance(plan, Join):
        return dataclasses.replace(plan, left=fn(plan.left), right=fn(plan.right))
    if isinstance(plan, SetOp):
        return dataclasses.replace(plan, left=fn(plan.left), right=fn(plan.right))
    if isinstance(plan, WithCTE):
        return dataclasses.replace(
            plan,
            bindings=tuple(
                CTEBinding(name=b.name, plan=fn(b.plan)) for b in plan.bindings
            ),
            body=fn(plan.body),
        )
    if isinstance(plan, RecursiveCTE):
        return dataclasses.replace(
            plan,
            seed=fn(plan.seed),
            recursive=fn(plan.recursive),
            body=fn(plan.body),
        )
    if isinstance(plan, Apply):
        return dataclasses.replace(plan, outer=fn(plan.outer), inner=fn(plan.inner))
    return plan


def _rewrite(plan: Plan, semantics: SemanticConfig) -> Plan:
    plan = _map_children(plan, lambda p: _rewrite(p, semantics))
    if (
        isinstance(plan, Filter)
        and _is_marker(plan.predicate)
        and isinstance(plan.input, Sort)
    ):
        n = plan.predicate.args[0].value
        return _materialize_with_ties(plan.input, int(n))
    return plan


PRE_EXECUTION_PASSES: list[Callable[[Plan, SemanticConfig], Plan]] = [_rewrite]
