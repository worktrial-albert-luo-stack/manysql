"""Logical-plan IR for manysql.

A `Plan` is a tree of relational-algebra operators. Every node carries its
output schema (list of typed columns). Plans are immutable (frozen dataclasses).

v1 (Tier A) operator set:
    Scan, Project, Filter, Join, Aggregate, Window, Sort, Limit,
    Distinct, Union, Intersect, Except, WithCTE, RecursiveCTE, Apply.

See manysql/ir/SCOPE.md for what is intentionally excluded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from manysql.ir.expr import Expr, OrderKey
from manysql.ir.types import IRType


@dataclass(frozen=True)
class ColumnSchema:
    """One output column of a Plan."""

    name: str
    type: IRType
    qualifier: Optional[str] = None


@dataclass(frozen=True)
class Plan:
    """Base Plan. All plan nodes inherit from this."""

    def children(self) -> list["Plan"]:
        return []

    def schema(self) -> tuple[ColumnSchema, ...]:
        raise NotImplementedError(f"{type(self).__name__} must implement schema()")


@dataclass(frozen=True)
class Scan(Plan):
    """Scan a base table from the catalog."""

    table_name: str
    columns: tuple[ColumnSchema, ...]
    alias: Optional[str] = None

    def schema(self) -> tuple[ColumnSchema, ...]:
        if self.alias is None:
            return self.columns
        return tuple(
            ColumnSchema(c.name, c.type, qualifier=self.alias) for c in self.columns
        )


@dataclass(frozen=True)
class Project(Plan):
    """Project a list of expressions, each with an output name."""

    input: Plan
    projections: tuple[tuple[str, Expr], ...]
    output_types: tuple[IRType, ...]  # explicit output types (lowering decides them)

    def children(self) -> list[Plan]:
        return [self.input]

    def schema(self) -> tuple[ColumnSchema, ...]:
        return tuple(
            ColumnSchema(name, t)
            for (name, _), t in zip(self.projections, self.output_types, strict=True)
        )


@dataclass(frozen=True)
class Filter(Plan):
    input: Plan
    predicate: Expr

    def children(self) -> list[Plan]:
        return [self.input]

    def schema(self) -> tuple[ColumnSchema, ...]:
        return self.input.schema()


class JoinKind(str, Enum):
    INNER = "INNER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FULL = "FULL"
    CROSS = "CROSS"
    SEMI = "SEMI"
    ANTI = "ANTI"


@dataclass(frozen=True)
class Join(Plan):
    left: Plan
    right: Plan
    kind: JoinKind
    on: Optional[Expr] = None  # CROSS has no `on`
    using: tuple[str, ...] = field(default_factory=tuple)

    def children(self) -> list[Plan]:
        return [self.left, self.right]

    def schema(self) -> tuple[ColumnSchema, ...]:
        if self.kind in (JoinKind.SEMI, JoinKind.ANTI):
            return self.left.schema()
        return self.left.schema() + self.right.schema()


@dataclass(frozen=True)
class Aggregate(Plan):
    """GROUP BY ... + aggregate functions.

    Each `aggregate` is a (output_name, AggCall) pair. Group keys are
    expressions (typically ColumnRef). HAVING is represented as a Filter on
    top of the Aggregate node, not embedded.
    """

    input: Plan
    group_by: tuple[tuple[str, Expr], ...]  # named group keys
    aggregates: tuple[tuple[str, Expr], ...]  # name -> AggCall
    output_types: tuple[IRType, ...]  # types of (group_by + aggregates) in order

    def children(self) -> list[Plan]:
        return [self.input]

    def schema(self) -> tuple[ColumnSchema, ...]:
        names = [n for n, _ in self.group_by] + [n for n, _ in self.aggregates]
        return tuple(
            ColumnSchema(n, t) for n, t in zip(names, self.output_types, strict=True)
        )


@dataclass(frozen=True)
class Window(Plan):
    """Add window-function columns to the input.

    The QUALIFY clause is represented as a Filter on top of a Window.
    """

    input: Plan
    windows: tuple[tuple[str, Expr], ...]  # name -> WindowCall
    output_types: tuple[IRType, ...]  # types of the new window columns, in order

    def children(self) -> list[Plan]:
        return [self.input]

    def schema(self) -> tuple[ColumnSchema, ...]:
        added = tuple(
            ColumnSchema(n, t)
            for (n, _), t in zip(self.windows, self.output_types, strict=True)
        )
        return self.input.schema() + added


@dataclass(frozen=True)
class Sort(Plan):
    input: Plan
    keys: tuple[OrderKey, ...]

    def children(self) -> list[Plan]:
        return [self.input]

    def schema(self) -> tuple[ColumnSchema, ...]:
        return self.input.schema()


@dataclass(frozen=True)
class Limit(Plan):
    input: Plan
    limit: Optional[int] = None
    offset: int = 0

    def children(self) -> list[Plan]:
        return [self.input]

    def schema(self) -> tuple[ColumnSchema, ...]:
        return self.input.schema()


@dataclass(frozen=True)
class Distinct(Plan):
    input: Plan

    def children(self) -> list[Plan]:
        return [self.input]

    def schema(self) -> tuple[ColumnSchema, ...]:
        return self.input.schema()


class SetOpKind(str, Enum):
    UNION = "UNION"
    INTERSECT = "INTERSECT"
    EXCEPT = "EXCEPT"


@dataclass(frozen=True)
class SetOp(Plan):
    """Generic set operation. `all=True` means UNION ALL / INTERSECT ALL / EXCEPT ALL."""

    left: Plan
    right: Plan
    kind: SetOpKind
    all: bool = False

    def children(self) -> list[Plan]:
        return [self.left, self.right]

    def schema(self) -> tuple[ColumnSchema, ...]:
        return self.left.schema()


@dataclass(frozen=True)
class CTEBinding:
    name: str
    plan: Plan


@dataclass(frozen=True)
class WithCTE(Plan):
    """Bind one or more non-recursive CTEs visible in `body`."""

    bindings: tuple[CTEBinding, ...]
    body: Plan

    def children(self) -> list[Plan]:
        return [b.plan for b in self.bindings] + [self.body]

    def schema(self) -> tuple[ColumnSchema, ...]:
        return self.body.schema()


@dataclass(frozen=True)
class RecursiveCTE(Plan):
    """A single recursive CTE binding visible in `body`.

    Semantics: start from `seed`, repeatedly union `recursive` (which references
    `name`) until no new rows are produced.
    """

    name: str
    seed: Plan
    recursive: Plan
    body: Plan
    union_all: bool = True  # most dialects use UNION ALL semantics here

    def children(self) -> list[Plan]:
        return [self.seed, self.recursive, self.body]

    def schema(self) -> tuple[ColumnSchema, ...]:
        return self.body.schema()


class ApplyKind(str, Enum):
    SCALAR = "SCALAR"  # the dependent plan must produce <= 1 row, 1 col
    EXISTS = "EXISTS"
    NOT_EXISTS = "NOT_EXISTS"
    IN = "IN"
    NOT_IN = "NOT_IN"
    CROSS = "CROSS APPLY"  # multiplicative, like LATERAL
    OUTER = "OUTER APPLY"  # additive, like LATERAL with NULL fill


@dataclass(frozen=True)
class Apply(Plan):
    """Dependent join used to lower correlated subqueries.

    `outer` provides correlated values; for each outer row we evaluate `inner`
    (which may reference outer columns), then combine according to `kind`.

    The scalar/in/exists kinds add a single boolean/scalar column with name
    `output_name` to the outer schema.
    """

    outer: Plan
    inner: Plan
    kind: ApplyKind
    output_name: Optional[str] = None  # required for SCALAR/IN/EXISTS variants
    output_type: Optional[IRType] = None  # required for SCALAR

    def children(self) -> list[Plan]:
        return [self.outer, self.inner]

    def schema(self) -> tuple[ColumnSchema, ...]:
        outer = self.outer.schema()
        if self.kind in (ApplyKind.CROSS, ApplyKind.OUTER):
            return outer + self.inner.schema()
        if self.kind == ApplyKind.SCALAR:
            assert self.output_name is not None and self.output_type is not None
            return outer + (ColumnSchema(self.output_name, self.output_type),)
        from manysql.ir.types import BOOL

        assert self.output_name is not None
        return outer + (ColumnSchema(self.output_name, BOOL),)
