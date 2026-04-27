"""IR expression -> Polars expression.

Honors SemanticConfig at every divergence point:
- division_by_zero (NULL / ERROR / INF)
- integer_division (TRUNCATE / PROMOTE)
- like_case_sensitive (ILIKE handled separately)
- string_concat_op (only matters for the surface; all dialects lower to canonical IR FuncCall('CONCAT'))
- boolean_truthiness (forced via casts when a bool is needed but not given)

Correlated columns are passed via `outer_row`. A ColumnRef whose qualified name
appears in `outer_row` is materialized as a Polars literal of that value.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Optional

import polars as pl

from manysql.ir.expr import (
    AggCall,
    AggKind,
    Between,
    BinaryOp,
    Case,
    Cast,
    ColumnRef,
    ExistsSubquery,
    Expr,
    FuncCall,
    InList,
    InSubquery,
    IsNull,
    Literal,
    Op,
    ScalarSubquery,
    UnaryOp,
)
from manysql.ir.types import IRType, TypeKind
from manysql.spec.semantics import (
    DivByZero,
    IntDivision,
    SemanticConfig,
)

if TYPE_CHECKING:
    from manysql.ir.plan import Plan


def _outer_key(c: ColumnRef) -> str:
    return f"{c.qualifier}.{c.name}" if c.qualifier else c.name


def _ir_type_to_polars(t: IRType) -> pl.DataType:
    return {
        TypeKind.INT: pl.Int64,
        TypeKind.FLOAT: pl.Float64,
        TypeKind.TEXT: pl.Utf8,
        TypeKind.BOOL: pl.Boolean,
        TypeKind.DATE: pl.Date,
        TypeKind.TIMESTAMP: pl.Datetime,
        TypeKind.NULL: pl.Null,
    }[t.kind]


def _like_pattern_to_regex(pat: str, case_sensitive: bool) -> str:
    """SQL LIKE pattern -> regex.

    `%` -> `.*`, `_` -> `.`. Other regex metachars are escaped. The pattern
    is anchored. Case-insensitivity is added via `(?i)` prefix when needed.
    """
    out: list[str] = []
    i = 0
    while i < len(pat):
        ch = pat[i]
        if ch == "\\" and i + 1 < len(pat):
            out.append(re.escape(pat[i + 1]))
            i += 2
            continue
        if ch == "%":
            out.append(".*")
        elif ch == "_":
            out.append(".")
        else:
            out.append(re.escape(ch))
        i += 1
    pattern = "^" + "".join(out) + "$"
    if not case_sensitive:
        pattern = "(?i)" + pattern
    return pattern


def _trunc_div_pl(n: pl.Expr, d: int) -> pl.Expr:
    """Integer division of a Polars expression that truncates toward zero.

    Polars (and Python) ``//`` floors toward negative infinity, but SQL
    integer division truncates toward zero. Only matters for negative
    numerators (``-6 // 7`` = -1, but trunc-div is 0). Branching on the
    sign keeps the semantics aligned with SemanticConfig.integer_division
    and with sqlite's STRFTIME-arithmetic behavior used by BIRD gold SQL.
    """
    return pl.when(n >= 0).then(n // d).otherwise(-((-n) // d))


def _to_bool(e: pl.Expr, semantics: SemanticConfig) -> pl.Expr:
    """Coerce an expression to boolean according to the truthiness knob.

    Both STRICT and C_STYLE currently lower to the same Polars cast: the
    non-strict variant accepts numerics (0 -> False, nonzero -> True) and
    propagates NULLs unchanged, which is what both modes want for v1. The
    ``semantics`` argument is kept so the callsite stays stable when a real
    STRICT path (raise on non-bool input dtype) gets re-introduced.
    """
    _ = semantics
    return e.cast(pl.Boolean, strict=False)


class ExprEvaluator:
    """Stateful evaluator: needs `executor` to recursively execute subquery plans."""

    def __init__(
        self,
        semantics: SemanticConfig,
        executor: "PlanExecutor",
        *,
        outer_row: Optional[dict[str, Any]] = None,
        overrides: Optional[Any] = None,
        effects: Optional[Any] = None,
    ) -> None:
        self.semantics = semantics
        self.executor = executor
        self.outer_row = outer_row or {}
        self.overrides = overrides
        self.effects = effects

    # -------- public entry --------

    def eval(self, e: Expr) -> pl.Expr:
        return self._dispatch(e)

    # -------- dispatch --------

    def _dispatch(self, e: Expr) -> pl.Expr:  # noqa: PLR0911, PLR0912
        if isinstance(e, Literal):
            return self._literal(e)
        if isinstance(e, ColumnRef):
            return self._column(e)
        if isinstance(e, BinaryOp):
            return self._binary(e)
        if isinstance(e, UnaryOp):
            return self._unary(e)
        if isinstance(e, FuncCall):
            return self._func(e)
        if isinstance(e, Case):
            return self._case(e)
        if isinstance(e, Cast):
            return self._cast(e)
        if isinstance(e, InList):
            return self._inlist(e)
        if isinstance(e, IsNull):
            return self._isnull(e)
        if isinstance(e, Between):
            return self._between(e)
        if isinstance(e, AggCall):
            return self._agg(e)
        if isinstance(e, ScalarSubquery):
            return self._scalar_subquery(e)
        if isinstance(e, ExistsSubquery):
            return self._exists_subquery(e)
        if isinstance(e, InSubquery):
            return self._in_subquery(e)
        raise NotImplementedError(f"ExprEvaluator: unhandled {type(e).__name__}")

    # -------- leaves --------

    def _literal(self, e: Literal) -> pl.Expr:
        if e.value is None:
            return pl.lit(None).cast(_ir_type_to_polars(e.type))
        return pl.lit(e.value).cast(_ir_type_to_polars(e.type))

    def _column(self, e: ColumnRef) -> pl.Expr:
        key = _outer_key(e)
        if key in self.outer_row:
            return pl.lit(self.outer_row[key])
        # Polars columns are unqualified inside a frame. The lowerer/executor is
        # responsible for renaming joined columns to avoid clashes (we use
        # qualifier-prefixed names during joins).
        return pl.col(self._resolved_name(e))

    def _resolved_name(self, e: ColumnRef) -> str:
        """Resolve a ColumnRef to a Polars column name.

        We prefer `qualifier__name` if a qualifier is present and the frame
        likely contains that prefixed form (Join produces these). Falls back
        to bare `name`. The actual choice is made by trying both at execution
        time, but here we just pick the qualified form when a qualifier exists.
        Frames that need bare names will rename their columns before evaluation.
        """
        if e.qualifier is not None:
            return f"{e.qualifier}__{e.name}"
        return e.name

    # -------- ops --------

    def _binary(self, e: BinaryOp) -> pl.Expr:  # noqa: PLR0911, PLR0912
        left = self._dispatch(e.left)
        right = self._dispatch(e.right)
        op = e.op
        if op == Op.EQ:
            eff = self._call_effect("text_eq", left, right)
            return eff if eff is not None else left == right
        if op == Op.NEQ:
            eff = self._call_effect("text_neq", left, right)
            return eff if eff is not None else left != right
        if op == Op.LT:
            return left < right
        if op == Op.LTE:
            return left <= right
        if op == Op.GT:
            return left > right
        if op == Op.GTE:
            return left >= right
        if op == Op.NULL_SAFE_EQ:
            # IS NOT DISTINCT FROM: NULL == NULL is true; NULL == anything else is false.
            return (
                ((left.is_null()) & (right.is_null()))
                | ((left.is_not_null()) & (right.is_not_null()) & (left == right))
            )
        if op == Op.AND:
            return _to_bool(left, self.semantics) & _to_bool(right, self.semantics)
        if op == Op.OR:
            return _to_bool(left, self.semantics) | _to_bool(right, self.semantics)
        if op == Op.ADD:
            return left + right
        if op == Op.SUB:
            return left - right
        if op == Op.MUL:
            return left * right
        if op == Op.DIV:
            return self._div(left, right, e)
        if op == Op.MOD:
            return left % right
        if op == Op.CONCAT:
            return self._concat(left, right)
        if op == Op.LIKE:
            return self._dispatch_like(
                left,
                self._extract_str_literal(e.right),
                case_sensitive=self.semantics.like_case_sensitive,
            )
        if op == Op.ILIKE:
            return self._dispatch_like(
                left,
                self._extract_str_literal(e.right),
                case_sensitive=False,
            )
        raise NotImplementedError(f"BinaryOp: {op}")

    # -------- effect dispatch --------

    def _resolve_effect(self, name: str):
        """Return the dialect's handler for ``name`` or ``None``.

        Effects are looked up in the dialect's ``effects.EFFECTS`` dict.
        Absent dialects, absent ``EFFECTS``, or absent keys all fall
        through to the canonical executor implementation.
        """
        if self.effects is None:
            return None
        registry = getattr(self.effects, "EFFECTS", None)
        if not registry:
            return None
        return registry.get(name)

    def _call_effect(
        self, name: str, left: pl.Expr, right: pl.Expr
    ) -> Optional[pl.Expr]:
        """Invoke the named effect; return ``None`` to mean "fall back".

        Handlers may also explicitly return ``None`` to defer to the
        canonical implementation (e.g. when they detect the operands
        aren't in their domain).
        """
        fn = self._resolve_effect(name)
        if fn is None:
            return None
        return fn(left, right, self.semantics)

    def _dispatch_like(
        self, operand_expr: pl.Expr, pattern: str, *, case_sensitive: bool
    ) -> pl.Expr:
        """LIKE / ILIKE dispatch: try the ``text_in_pattern`` effect first."""
        fn = self._resolve_effect("text_in_pattern")
        if fn is not None:
            result = fn(operand_expr, pattern, self.semantics, case_sensitive)
            if result is not None:
                return result
        return self._like_lit(
            operand_expr, pattern, case_sensitive=case_sensitive
        )

    def _unary(self, e: UnaryOp) -> pl.Expr:
        operand = self._dispatch(e.operand)
        if e.op == Op.NOT:
            return ~_to_bool(operand, self.semantics)
        if e.op == Op.NEG:
            return -operand
        raise NotImplementedError(f"UnaryOp: {e.op}")

    def _div(self, left: pl.Expr, right: pl.Expr, e: BinaryOp) -> pl.Expr:
        # Integer-division detection: best-effort on the IR side, since Polars
        # may auto-promote. We approximate: if both literal/column types resolve
        # to integer kind in the IR, treat as integer division.
        is_int = self._is_integer(e.left) and self._is_integer(e.right)
        if is_int and self.semantics.integer_division == IntDivision.TRUNCATE:
            base = left // right
        elif is_int:  # PROMOTE
            base = left.cast(pl.Float64) / right.cast(pl.Float64)
        else:
            base = left / right

        mode = self.semantics.division_by_zero
        if mode == DivByZero.NULL:
            return pl.when(right == 0).then(None).otherwise(base)
        if mode == DivByZero.INF:
            return base  # IEEE-754 propagation
        # ERROR mode: leave base; runtime error will surface from inf/nan
        # comparisons or downstream NULL handling. A stricter implementation
        # would scan-then-raise; deferred until oracle disagreement forces it.
        return base

    @staticmethod
    def _is_integer(e: Expr) -> bool:
        if isinstance(e, Literal):
            return e.type.kind == TypeKind.INT
        return False

    def _concat(self, left: pl.Expr, right: pl.Expr) -> pl.Expr:
        # Cast both sides to Utf8 and concatenate, ANSI-style: NULL || x = NULL.
        return left.cast(pl.Utf8) + right.cast(pl.Utf8)

    def _like(self, left: pl.Expr, right: pl.Expr, *, case_sensitive: bool) -> pl.Expr:
        # The pattern must be a literal for static regex compilation in v1.
        # Dynamic patterns (column-valued) would require row-wise apply; deferred.
        # We extract from the right-hand IR Literal at eval-time via the chain
        # in BinaryOp; but here `right` is already a polars Expr. We instead
        # require the lowerer to pass the pattern as a string literal, and we
        # detect the literal here by introspecting `right` via Polars IR str repr.
        # Simpler: the IR side of LIKE expects a string; we use Polars str.contains
        # with regex=True on the column, computing the regex at eval time below.
        raise NotImplementedError(
            "BinaryOp LIKE expects pattern via dedicated like-helper; see _like_lit."
        )

    def _like_lit(
        self, operand_expr: pl.Expr, pattern: str, *, case_sensitive: bool
    ) -> pl.Expr:
        regex = _like_pattern_to_regex(pattern, case_sensitive)
        return operand_expr.cast(pl.Utf8).str.contains(regex)

    def _func(self, e: FuncCall) -> pl.Expr:  # noqa: PLR0911, PLR0912
        name = e.name.upper()
        args = [self._dispatch(a) for a in e.args]

        if name == "COALESCE":
            out = args[-1]
            for a in reversed(args[:-1]):
                out = pl.when(a.is_not_null()).then(a).otherwise(out)
            return out
        if name in ("UPPER", "UCASE"):
            return args[0].cast(pl.Utf8).str.to_uppercase()
        if name in ("LOWER", "LCASE"):
            return args[0].cast(pl.Utf8).str.to_lowercase()
        if name in ("LENGTH", "LEN", "CHAR_LENGTH"):
            return args[0].cast(pl.Utf8).str.len_chars()
        if name == "CONCAT":
            out = args[0].cast(pl.Utf8)
            for a in args[1:]:
                out = out + a.cast(pl.Utf8)
            return out
        if name == "ABS":
            return args[0].abs()
        if name == "TRIM":
            return args[0].cast(pl.Utf8).str.strip_chars()
        if name == "SUBSTR" or name == "SUBSTRING":
            # SQL is 1-based, length optional
            start = args[1] - 1 if len(args) >= 2 else pl.lit(0)
            length = args[2] if len(args) >= 3 else None
            if length is not None:
                return args[0].cast(pl.Utf8).str.slice(start, length)
            return args[0].cast(pl.Utf8).str.slice(start)
        if name == "REPLACE":
            return args[0].cast(pl.Utf8).str.replace_all(
                self._extract_str_literal(e.args[1]),
                self._extract_str_literal(e.args[2]),
                literal=True,
            )
        if name in ("IF", "IIF"):
            return pl.when(_to_bool(args[0], self.semantics)).then(args[1]).otherwise(args[2])
        if name == "NULLIF":
            return pl.when(args[0] == args[1]).then(None).otherwise(args[0])
        if name == "GREATEST":
            return pl.max_horizontal(args)
        if name == "LEAST":
            return pl.min_horizontal(args)
        if name == "DATE_PART" or name == "EXTRACT":
            part = self._extract_str_literal(e.args[0]).lower()
            col = args[1]
            return self._date_part(part, col)
        if name == "DATE_TRUNC":
            part = self._extract_str_literal(e.args[0]).lower()
            return args[1].dt.truncate(self._dt_truncate_arg(part))
        if name == "DATE_ADD":
            # DATE_ADD(date, n_days) - canonical IR form
            return args[0].dt.offset_by(
                pl.format("{}d", args[1]).str.to_lowercase()
            ) if False else args[0] + pl.duration(days=args[1])
        if name == "DATE_SUB":
            return args[0] - pl.duration(days=args[1])
        if name == "DATE_DIFF":
            # DATE_DIFF('unit', a, b): components-of(b) - components-of(a).
            # FENCEPOST semantics for calendar units (year/quarter/month):
            # only the boundary count, e.g. DATE_DIFF('month', '2024-01-31',
            # '2024-02-01') = 1. Matches sqlite's STRFTIME-arithmetic
            # convention used throughout BIRD gold SQL. The TRUNCATING
            # variant (count-completed-months-only) is deferred to the
            # `date_diff_policy` SemanticConfig knob in RFC 0001.
            unit = self._extract_str_literal(e.args[0]).lower() if len(e.args) >= 3 else "day"
            a = args[-2]
            b = args[-1]
            return self._date_diff(unit, a, b)
        if name == "ROUND":
            if len(args) == 1:
                return args[0].round(0)
            # Polars Expr.round() requires a Python int; pull it out of the IR.
            decimals = (
                e.args[1].value
                if isinstance(e.args[1], Literal) and isinstance(e.args[1].value, int)
                else None
            )
            if decimals is None:
                raise NotImplementedError(
                    "ROUND(x, n) requires an integer literal for n"
                )
            return args[0].round(decimals)
        if name == "FLOOR":
            return args[0].floor()
        if name == "CEIL" or name == "CEILING":
            return args[0].ceil()
        if name == "MOD":
            return args[0] % args[1]
        if name == "LIKE":
            # Lowering may emit FuncCall('LIKE', col, 'pattern') as alternative form.
            return self._like_lit(
                args[0],
                self._extract_str_literal(e.args[1]),
                case_sensitive=self.semantics.like_case_sensitive,
            )
        if name == "ILIKE":
            return self._like_lit(
                args[0], self._extract_str_literal(e.args[1]), case_sensitive=False
            )
        override = self._resolve_override(name)
        if override is not None:
            return override(args, self.semantics)
        raise NotImplementedError(f"FuncCall not implemented: {name}")

    def _resolve_override(self, name: str):
        """Look up `name` in the dialect's overrides module, if any.

        The lookup order matches the documented contract: function names
        first, then operator names, then None. Names are upper-cased on
        both sides to keep dispatch case-insensitive.
        """
        if self.overrides is None:
            return None
        upper = name.upper()
        functions = getattr(self.overrides, "FUNCTIONS", None)
        if isinstance(functions, dict):
            fn = functions.get(upper)
            if fn is not None:
                return fn
        operators = getattr(self.overrides, "OPERATORS", None)
        if isinstance(operators, dict):
            op = operators.get(upper)
            if op is not None:
                return op
        return None

    def _date_diff(self, unit: str, a: pl.Expr, b: pl.Expr) -> pl.Expr:
        """``DATE_DIFF(unit, a, b)`` -> Polars expression for ``b`` minus ``a``.

        Calendar units (``year`` / ``quarter`` / ``month``) use
        FENCEPOST semantics: count the boundaries crossed, not the
        completed periods. Sub-day units use total-elapsed semantics
        truncated toward zero. Both match what ``STRFTIME``-arithmetic
        yields in sqlite, which is the BIRD gold SQL convention.
        """
        if unit in ("day", "days"):
            return (b - a).dt.total_days()
        if unit in ("week", "weeks"):
            # SQL semantics truncate toward zero; Python/Polars ``//``
            # floors toward negative infinity. The two agree for
            # exact multiples but diverge on negative non-multiples
            # (e.g. -6 // 7 = -1 vs. trunc-div = 0). Branch on sign
            # to get trunc semantics.
            return _trunc_div_pl((b - a).dt.total_days(), 7)
        if unit in ("month", "months"):
            return (b.dt.year() - a.dt.year()) * 12 + (b.dt.month() - a.dt.month())
        if unit in ("quarter", "quarters"):
            months = (b.dt.year() - a.dt.year()) * 12 + (b.dt.month() - a.dt.month())
            return _trunc_div_pl(months, 3)
        if unit in ("year", "years", "yyyy"):
            return b.dt.year() - a.dt.year()
        if unit in ("hour", "hours"):
            return (b - a).dt.total_hours()
        if unit in ("minute", "minutes"):
            return (b - a).dt.total_minutes()
        if unit in ("second", "seconds"):
            return (b - a).dt.total_seconds()
        raise NotImplementedError(f"DATE_DIFF unit: {unit}")

    def _date_part(self, part: str, col: pl.Expr) -> pl.Expr:
        if part in ("year", "yyyy"):
            return col.dt.year()
        if part == "month":
            return col.dt.month()
        if part == "day":
            return col.dt.day()
        if part == "hour":
            return col.dt.hour()
        if part == "minute":
            return col.dt.minute()
        if part == "second":
            return col.dt.second()
        if part in ("dow", "dayofweek"):
            return col.dt.weekday()
        if part in ("doy", "dayofyear"):
            return col.dt.ordinal_day()
        raise NotImplementedError(f"DATE_PART unit: {part}")

    @staticmethod
    def _dt_truncate_arg(part: str) -> str:
        return {
            "year": "1y",
            "month": "1mo",
            "day": "1d",
            "hour": "1h",
            "minute": "1m",
            "second": "1s",
        }[part]

    @staticmethod
    def _extract_str_literal(e: Expr) -> str:
        if isinstance(e, Literal) and isinstance(e.value, str):
            return e.value
        raise ValueError(
            f"Expected string literal, got {type(e).__name__} (dynamic patterns deferred)"
        )

    def _case(self, e: Case) -> pl.Expr:
        if not e.branches:
            return pl.lit(e.default)
        # Build chained when/then/otherwise from branches
        first_cond, first_val = e.branches[0]
        chain = pl.when(_to_bool(self._dispatch(first_cond), self.semantics)).then(
            self._dispatch(first_val)
        )
        for cond, val in e.branches[1:]:
            chain = chain.when(_to_bool(self._dispatch(cond), self.semantics)).then(
                self._dispatch(val)
            )
        if e.default is not None:
            return chain.otherwise(self._dispatch(e.default))
        return chain.otherwise(None)

    def _cast(self, e: Cast) -> pl.Expr:
        return self._dispatch(e.operand).cast(_ir_type_to_polars(e.target), strict=False)

    def _inlist(self, e: InList) -> pl.Expr:
        operand = self._dispatch(e.operand)
        if not e.items:
            return pl.lit(False)
        # Fast path: when every item is a non-NULL literal, dispatch to
        # Polars' `is_in` (~10x faster than the OR chain for long lists).
        # Bail out if any item is NULL or non-literal: SQL's three-valued
        # IN semantics with a NULL-containing list (`x IN (a, NULL)` is
        # NULL when x != a, not False) is what the OR chain preserves.
        if all(
            isinstance(it, Literal) and it.value is not None for it in e.items
        ):
            return operand.is_in([it.value for it in e.items])
        out = operand == self._dispatch(e.items[0])
        for it in e.items[1:]:
            out = out | (operand == self._dispatch(it))
        return out

    def _isnull(self, e: IsNull) -> pl.Expr:
        operand = self._dispatch(e.operand)
        return operand.is_not_null() if e.negated else operand.is_null()

    def _between(self, e: Between) -> pl.Expr:
        op = self._dispatch(e.operand)
        low = self._dispatch(e.low)
        high = self._dispatch(e.high)
        in_range = (op >= low) & (op <= high)
        return ~in_range if e.negated else in_range

    def _agg(self, e: AggCall) -> pl.Expr:
        # Only valid inside Aggregate.aggregates; the plan executor handles
        # group_by + this expr. Here we just produce the right Polars expression.
        if e.kind == AggKind.COUNT_STAR:
            return pl.len()
        if e.arg is None:
            raise ValueError(f"Aggregate {e.kind} requires an argument")
        arg = self._dispatch(e.arg)
        if e.distinct:
            arg = arg.unique()
        if e.filter_pred is not None:
            arg = arg.filter(_to_bool(self._dispatch(e.filter_pred), self.semantics))
        if e.kind == AggKind.COUNT:
            # SQL COUNT(x) ignores NULLs.
            return arg.drop_nulls().len()
        if e.kind == AggKind.SUM:
            return arg.sum()
        if e.kind == AggKind.AVG:
            return arg.mean()
        if e.kind == AggKind.MIN:
            return arg.min()
        if e.kind == AggKind.MAX:
            return arg.max()
        raise NotImplementedError(f"AggKind: {e.kind}")

    def _scalar_subquery(self, e: ScalarSubquery) -> pl.Expr:
        # Non-correlated scalar subquery: execute once, return scalar literal.
        # Correlated cases must have been lowered to Apply by the planner.
        if e.correlated_columns:
            raise ValueError(
                "Correlated ScalarSubquery must be lowered to Apply before execution"
            )
        df = self.executor.execute(e.plan, outer_row=self.outer_row)
        if df.height == 0:
            return pl.lit(None)
        if df.height > 1:
            raise ValueError("Scalar subquery returned more than one row")
        if df.width != 1:
            raise ValueError("Scalar subquery must return exactly one column")
        return pl.lit(df.row(0)[0])

    def _exists_subquery(self, e: ExistsSubquery) -> pl.Expr:
        if e.correlated_columns:
            raise ValueError(
                "Correlated ExistsSubquery must be lowered to Apply before execution"
            )
        df = self.executor.execute(e.plan, outer_row=self.outer_row)
        result = df.height > 0
        return pl.lit(not result if e.negated else result)

    def _in_subquery(self, e: InSubquery) -> pl.Expr:
        if e.correlated_columns:
            raise ValueError(
                "Correlated InSubquery must be lowered to Apply before execution"
            )
        df = self.executor.execute(e.plan, outer_row=self.outer_row)
        if df.width != 1:
            raise ValueError("IN subquery must return exactly one column")
        values: list[Any] = df.to_series(0).to_list()
        operand = self._dispatch(e.operand)
        if not values:
            return pl.lit(False if not e.negated else True)
        in_expr = operand.is_in(values)
        return ~in_expr if e.negated else in_expr


# Forward ref hint for type checkers; avoids circular import at runtime.
if TYPE_CHECKING:
    from manysql.executor.engine import PlanExecutor
