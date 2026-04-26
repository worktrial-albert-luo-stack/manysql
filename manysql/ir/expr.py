"""Expression nodes for the IR.

Expressions are pure (no side effects, no schema mutation). Every expression
carries an inferred type once a Plan has been type-checked.

Design notes:
- Aggregate expressions are first-class (AggCall) but only legal inside an
  Aggregate plan node's `aggregates` list.
- Window expressions (WindowCall) are only legal inside a Window plan node.
- Subquery expressions (ScalarSubquery, ExistsSubquery, InSubquery) reference
  a Plan; the planner is responsible for marking them correlated, which
  triggers Apply lowering in the executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

from manysql.ir.types import IRType, PyValue

if TYPE_CHECKING:
    from manysql.ir.plan import Plan


class Op(str, Enum):
    EQ = "="
    NEQ = "<>"
    LT = "<"
    LTE = "<="
    GT = ">"
    GTE = ">="
    NULL_SAFE_EQ = "IS NOT DISTINCT FROM"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    NEG = "NEG"
    CONCAT = "||"
    LIKE = "LIKE"
    ILIKE = "ILIKE"
    IN = "IN"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"
    BETWEEN = "BETWEEN"


class AggKind(str, Enum):
    COUNT = "COUNT"
    COUNT_STAR = "COUNT_STAR"
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"


class WindowKind(str, Enum):
    ROW_NUMBER = "ROW_NUMBER"
    RANK = "RANK"
    DENSE_RANK = "DENSE_RANK"
    LAG = "LAG"
    LEAD = "LEAD"
    FIRST_VALUE = "FIRST_VALUE"
    LAST_VALUE = "LAST_VALUE"
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    COUNT = "COUNT"


class FrameMode(str, Enum):
    ROWS = "ROWS"
    RANGE = "RANGE"
    GROUPS = "GROUPS"


class FrameBoundKind(str, Enum):
    UNBOUNDED_PRECEDING = "UNBOUNDED PRECEDING"
    PRECEDING = "PRECEDING"
    CURRENT_ROW = "CURRENT ROW"
    FOLLOWING = "FOLLOWING"
    UNBOUNDED_FOLLOWING = "UNBOUNDED FOLLOWING"


@dataclass(frozen=True)
class FrameBound:
    kind: FrameBoundKind
    offset: Optional[int] = None


@dataclass(frozen=True)
class WindowFrame:
    mode: FrameMode
    start: FrameBound
    end: FrameBound


@dataclass(frozen=True)
class Expr:
    """Base class. Subclasses must be frozen dataclasses."""

    def children(self) -> list["Expr"]:
        return []


@dataclass(frozen=True)
class Literal(Expr):
    value: PyValue
    type: IRType


@dataclass(frozen=True)
class ColumnRef(Expr):
    """Reference to a column by qualified name.

    `qualifier` is the table/CTE/subquery alias; `name` is the column name.
    Identifier case-folding (per SemanticConfig.identifier_case_fold) is
    applied during AST -> IR lowering, so by the time we see an IR ColumnRef
    the names are already canonical.
    """

    name: str
    qualifier: Optional[str] = None


@dataclass(frozen=True)
class BinaryOp(Expr):
    op: Op
    left: Expr
    right: Expr

    def children(self) -> list[Expr]:
        return [self.left, self.right]


@dataclass(frozen=True)
class UnaryOp(Expr):
    op: Op
    operand: Expr

    def children(self) -> list[Expr]:
        return [self.operand]


@dataclass(frozen=True)
class FuncCall(Expr):
    """Canonical scalar function call.

    `name` is the canonical IR-level function name (e.g. CONCAT, UPPER,
    DATE_ADD), not the dialect-surface name. Dialect-specific names are
    mapped to canonical ones during lowering using SemanticConfig.function_aliases.
    """

    name: str
    args: tuple[Expr, ...] = field(default_factory=tuple)

    def children(self) -> list[Expr]:
        return list(self.args)


@dataclass(frozen=True)
class Case(Expr):
    """SQL CASE WHEN ... THEN ... ELSE ... END.

    Searched form. (Simple CASE is lowered to searched.)
    """

    branches: tuple[tuple[Expr, Expr], ...]
    default: Optional[Expr] = None

    def children(self) -> list[Expr]:
        out: list[Expr] = []
        for cond, val in self.branches:
            out.append(cond)
            out.append(val)
        if self.default is not None:
            out.append(self.default)
        return out


@dataclass(frozen=True)
class Cast(Expr):
    operand: Expr
    target: IRType

    def children(self) -> list[Expr]:
        return [self.operand]


@dataclass(frozen=True)
class InList(Expr):
    operand: Expr
    items: tuple[Expr, ...]

    def children(self) -> list[Expr]:
        return [self.operand, *self.items]


@dataclass(frozen=True)
class IsNull(Expr):
    operand: Expr
    negated: bool = False

    def children(self) -> list[Expr]:
        return [self.operand]


@dataclass(frozen=True)
class Between(Expr):
    operand: Expr
    low: Expr
    high: Expr
    negated: bool = False

    def children(self) -> list[Expr]:
        return [self.operand, self.low, self.high]


@dataclass(frozen=True)
class AggCall(Expr):
    """Aggregate call. Only legal inside Aggregate.aggregates."""

    kind: AggKind
    arg: Optional[Expr] = None
    distinct: bool = False
    filter_pred: Optional[Expr] = None

    def children(self) -> list[Expr]:
        out: list[Expr] = []
        if self.arg is not None:
            out.append(self.arg)
        if self.filter_pred is not None:
            out.append(self.filter_pred)
        return out


@dataclass(frozen=True)
class WindowCall(Expr):
    """Window function call. Only legal inside Window.windows."""

    kind: WindowKind
    args: tuple[Expr, ...] = field(default_factory=tuple)
    partition_by: tuple[Expr, ...] = field(default_factory=tuple)
    order_by: tuple["OrderKey", ...] = field(default_factory=tuple)
    frame: Optional[WindowFrame] = None
    ignore_nulls: bool = False

    def children(self) -> list[Expr]:
        out: list[Expr] = list(self.args)
        out.extend(self.partition_by)
        for ok in self.order_by:
            out.append(ok.expr)
        return out


@dataclass(frozen=True)
class ScalarSubquery(Expr):
    plan: "Plan"
    correlated_columns: tuple[ColumnRef, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ExistsSubquery(Expr):
    plan: "Plan"
    negated: bool = False
    correlated_columns: tuple[ColumnRef, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class InSubquery(Expr):
    operand: Expr
    plan: "Plan"
    negated: bool = False
    correlated_columns: tuple[ColumnRef, ...] = field(default_factory=tuple)

    def children(self) -> list[Expr]:
        return [self.operand]


class SortDirection(str, Enum):
    ASC = "ASC"
    DESC = "DESC"


class NullsOrder(str, Enum):
    FIRST = "NULLS FIRST"
    LAST = "NULLS LAST"
    DEFAULT = "DEFAULT"  # honor SemanticConfig.null_order_default


@dataclass(frozen=True)
class OrderKey:
    expr: Expr
    direction: SortDirection = SortDirection.ASC
    nulls: NullsOrder = NullsOrder.DEFAULT
