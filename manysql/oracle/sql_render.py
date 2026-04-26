"""IR -> standard-ish SQL renderer for the SQL-engine oracles (DuckDB, SQLite).

This is *not* a dialect-renderer: dialects use Lark grammars to parse their
surface, not text generation. This renderer exists solely to feed plans to
host SQL engines so they can act as oracles.

Knob handling:
- The renderer takes a SemanticConfig and rewrites SQL to encode the knob's
  semantics (e.g. wraps division to return NULL on zero when the config says
  so, emits explicit NULLS FIRST/LAST, etc.).
- Knobs the host engine cannot express are flagged in the per-engine
  capability metadata so the harness skips that oracle when those knobs are
  non-default.

Per-engine differences are handled via flags on the renderer (e.g. SQLite has
no ILIKE, no FULL OUTER JOIN built-in, no "EXCEPT ALL"/"INTERSECT ALL", etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

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
    NullsOrder,
    Op,
    OrderKey,
    ScalarSubquery,
    SortDirection,
    UnaryOp,
    WindowCall,
    WindowKind,
)
from manysql.ir.plan import (
    Aggregate,
    Apply,
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
    SetOpKind,
    Sort,
    Window,
    WithCTE,
)
from manysql.ir.types import IRType, TypeKind
from manysql.spec.semantics import (
    DivByZero,
    IntDivision,
    NullOrder,
    SemanticConfig,
    SetOpDefault,
)


@dataclass
class SqlDialectFlags:
    """Per-engine surface differences. Not the same as SemanticConfig knobs."""

    name: str = "duckdb"
    supports_ilike: bool = True
    supports_full_join: bool = True
    supports_except_all: bool = True
    supports_intersect_all: bool = True
    supports_lateral: bool = True
    supports_recursive_cte: bool = True
    quote_identifier: str = '"'

    @classmethod
    def duckdb(cls) -> "SqlDialectFlags":
        return cls()

    @classmethod
    def sqlite(cls) -> "SqlDialectFlags":
        return cls(
            name="sqlite",
            supports_ilike=False,
            supports_full_join=False,  # SQLite 3.39+ supports it; we go conservative
            supports_except_all=False,
            supports_intersect_all=False,
            supports_lateral=False,
            quote_identifier='"',
        )


@dataclass
class RenderContext:
    flags: SqlDialectFlags
    semantics: SemanticConfig
    _alias_counter: int = 0
    notes: list[str] = field(default_factory=list)

    def fresh_alias(self, prefix: str = "t") -> str:
        self._alias_counter += 1
        return f"{prefix}{self._alias_counter}"


class UnsupportedByEngine(Exception):
    """Raised when a feature/knob combination cannot be rendered for the engine."""


def render_plan(
    plan: Plan, semantics: SemanticConfig, flags: SqlDialectFlags
) -> tuple[str, list[str]]:
    """Render an IR plan to a SQL string. Returns (sql, notes)."""
    ctx = RenderContext(flags=flags, semantics=semantics)
    sql = _render(plan, ctx, top_level=True)
    return sql, list(ctx.notes)


def _render(plan: Plan, ctx: RenderContext, *, top_level: bool = False) -> str:  # noqa: PLR0911
    if isinstance(plan, Scan):
        return _render_scan(plan, ctx, top_level=top_level)
    if isinstance(plan, Project):
        return _render_project(plan, ctx, top_level=top_level)
    if isinstance(plan, Filter):
        return _render_filter(plan, ctx, top_level=top_level)
    if isinstance(plan, Join):
        return _render_join(plan, ctx, top_level=top_level)
    if isinstance(plan, Aggregate):
        return _render_aggregate(plan, ctx, top_level=top_level)
    if isinstance(plan, Window):
        return _render_window(plan, ctx, top_level=top_level)
    if isinstance(plan, Sort):
        return _render_sort(plan, ctx, top_level=top_level)
    if isinstance(plan, Limit):
        return _render_limit(plan, ctx, top_level=top_level)
    if isinstance(plan, Distinct):
        return _render_distinct(plan, ctx, top_level=top_level)
    if isinstance(plan, SetOp):
        return _render_setop(plan, ctx, top_level=top_level)
    if isinstance(plan, WithCTE):
        return _render_with(plan, ctx, top_level=top_level)
    if isinstance(plan, RecursiveCTE):
        return _render_recursive(plan, ctx, top_level=top_level)
    if isinstance(plan, Apply):
        raise UnsupportedByEngine("Apply (correlated subquery) is handled by reference interpreter only")
    raise NotImplementedError(type(plan).__name__)


def _q(name: str, ctx: RenderContext) -> str:
    q = ctx.flags.quote_identifier
    return f"{q}{name}{q}"


def _render_scan(p: Scan, ctx: RenderContext, *, top_level: bool) -> str:
    # We always alias-prefix columns when an alias is present, so the rendered
    # SQL projects out renamed columns to match the executor's `alias__col`
    # convention.
    table = _q(p.table_name, ctx)
    if p.alias is None:
        return f"SELECT * FROM {table}"
    cols = ", ".join(
        f"{_q(c.name, ctx)} AS {_q(f'{p.alias}__{c.name}', ctx)}" for c in p.columns
    )
    return f"SELECT {cols} FROM {table}"


def _render_project(p: Project, ctx: RenderContext, *, top_level: bool) -> str:
    inner = _render(p.input, ctx)
    proj = ", ".join(f"{_render_expr(e, ctx)} AS {_q(name, ctx)}" for name, e in p.projections)
    sub = ctx.fresh_alias("sub")
    return f"SELECT {proj} FROM ({inner}) AS {sub}"


def _render_filter(p: Filter, ctx: RenderContext, *, top_level: bool) -> str:
    inner = _render(p.input, ctx)
    pred = _render_expr(p.predicate, ctx)
    sub = ctx.fresh_alias("sub")
    return f"SELECT * FROM ({inner}) AS {sub} WHERE {pred}"


def _render_join(p: Join, ctx: RenderContext, *, top_level: bool) -> str:
    if p.kind == JoinKind.FULL and not ctx.flags.supports_full_join:
        raise UnsupportedByEngine(f"{ctx.flags.name} does not support FULL JOIN")
    left_sql = _render(p.left, ctx)
    right_sql = _render(p.right, ctx)
    la = ctx.fresh_alias("l")
    ra = ctx.fresh_alias("r")
    join_kw = {
        JoinKind.INNER: "INNER JOIN",
        JoinKind.LEFT: "LEFT JOIN",
        JoinKind.RIGHT: "RIGHT JOIN",
        JoinKind.FULL: "FULL OUTER JOIN",
        JoinKind.CROSS: "CROSS JOIN",
        JoinKind.SEMI: None,  # Emulated via WHERE EXISTS
        JoinKind.ANTI: None,  # Emulated via WHERE NOT EXISTS
    }[p.kind]

    if join_kw is None:
        return _render_semi_anti(p, left_sql, right_sql, la, ra, ctx)

    if p.kind == JoinKind.CROSS:
        return f"SELECT * FROM ({left_sql}) AS {la} CROSS JOIN ({right_sql}) AS {ra}"

    if p.using:
        using = ", ".join(_q(c, ctx) for c in p.using)
        return (
            f"SELECT * FROM ({left_sql}) AS {la} "
            f"{join_kw} ({right_sql}) AS {ra} USING ({using})"
        )

    on = _render_expr(p.on, ctx) if p.on is not None else "TRUE"
    return f"SELECT * FROM ({left_sql}) AS {la} {join_kw} ({right_sql}) AS {ra} ON {on}"


def _render_semi_anti(
    p: Join,
    left_sql: str,
    right_sql: str,
    la: str,
    ra: str,
    ctx: RenderContext,
) -> str:
    on = _render_expr(p.on, ctx) if p.on is not None else "TRUE"
    not_word = "NOT " if p.kind == JoinKind.ANTI else ""
    return (
        f"SELECT * FROM ({left_sql}) AS {la} "
        f"WHERE {not_word}EXISTS (SELECT 1 FROM ({right_sql}) AS {ra} WHERE {on})"
    )


def _render_aggregate(p: Aggregate, ctx: RenderContext, *, top_level: bool) -> str:
    inner = _render(p.input, ctx)
    sub = ctx.fresh_alias("sub")
    select_parts: list[str] = []
    for name, e in p.group_by:
        select_parts.append(f"{_render_expr(e, ctx)} AS {_q(name, ctx)}")
    for name, e in p.aggregates:
        select_parts.append(f"{_render_aggregate_expr(e, ctx)} AS {_q(name, ctx)}")

    select_clause = ", ".join(select_parts)
    if p.group_by:
        group_clause = ", ".join(
            _render_expr(e, ctx) for _, e in p.group_by
        )
        return f"SELECT {select_clause} FROM ({inner}) AS {sub} GROUP BY {group_clause}"
    return f"SELECT {select_clause} FROM ({inner}) AS {sub}"


def _render_aggregate_expr(e: Expr, ctx: RenderContext) -> str:
    if not isinstance(e, AggCall):
        # Allow non-AggCall aggregate slots (rare; e.g. literal column).
        return _render_expr(e, ctx)
    if e.kind == AggKind.COUNT_STAR:
        rendered = "COUNT(*)"
    else:
        assert e.arg is not None
        arg = _render_expr(e.arg, ctx)
        if e.distinct:
            arg = f"DISTINCT {arg}"
        rendered = {
            AggKind.COUNT: f"COUNT({arg})",
            AggKind.SUM: f"SUM({arg})",
            AggKind.AVG: f"AVG({arg})",
            AggKind.MIN: f"MIN({arg})",
            AggKind.MAX: f"MAX({arg})",
        }[e.kind]
        # Honor SemanticConfig.sum_of_empty_returns_null=False
        if (
            e.kind == AggKind.SUM
            and not ctx.semantics.sum_of_empty_returns_null
        ):
            rendered = f"COALESCE({rendered}, 0)"
    if e.filter_pred is not None:
        rendered += f" FILTER (WHERE {_render_expr(e.filter_pred, ctx)})"
    return rendered


def _render_window(p: Window, ctx: RenderContext, *, top_level: bool) -> str:
    inner = _render(p.input, ctx)
    sub = ctx.fresh_alias("sub")
    star_then_windows = ["*"]
    for name, e in p.windows:
        assert isinstance(e, WindowCall)
        star_then_windows.append(f"{_render_window_call(e, ctx)} AS {_q(name, ctx)}")
    return f"SELECT {', '.join(star_then_windows)} FROM ({inner}) AS {sub}"


def _render_window_call(w: WindowCall, ctx: RenderContext) -> str:
    args = [_render_expr(a, ctx) for a in w.args]
    fn_map = {
        WindowKind.ROW_NUMBER: ("ROW_NUMBER", []),
        WindowKind.RANK: ("RANK", []),
        WindowKind.DENSE_RANK: ("DENSE_RANK", []),
        WindowKind.LAG: ("LAG", args),
        WindowKind.LEAD: ("LEAD", args),
        WindowKind.FIRST_VALUE: ("FIRST_VALUE", args),
        WindowKind.LAST_VALUE: ("LAST_VALUE", args),
        WindowKind.SUM: ("SUM", args),
        WindowKind.AVG: ("AVG", args),
        WindowKind.MIN: ("MIN", args),
        WindowKind.MAX: ("MAX", args),
        WindowKind.COUNT: ("COUNT", args or ["*"]),
    }
    fn, fn_args = fn_map[w.kind]
    fn_call = f"{fn}({', '.join(fn_args)})"

    over_parts: list[str] = []
    if w.partition_by:
        over_parts.append(
            "PARTITION BY " + ", ".join(_render_expr(e, ctx) for e in w.partition_by)
        )
    if w.order_by:
        over_parts.append("ORDER BY " + _render_order_keys(list(w.order_by), ctx))
    over = " ".join(over_parts)
    return f"{fn_call} OVER ({over})"


def _render_sort(p: Sort, ctx: RenderContext, *, top_level: bool) -> str:
    inner = _render(p.input, ctx)
    sub = ctx.fresh_alias("sub")
    keys = _render_order_keys(list(p.keys), ctx)
    return f"SELECT * FROM ({inner}) AS {sub} ORDER BY {keys}"


def _render_order_keys(keys: list[OrderKey], ctx: RenderContext) -> str:
    parts: list[str] = []
    for k in keys:
        expr_sql = _render_expr(k.expr, ctx)
        dir_sql = "DESC" if k.direction == SortDirection.DESC else "ASC"
        nulls = _resolve_nulls(k, ctx)
        parts.append(f"{expr_sql} {dir_sql} NULLS {nulls}")
    return ", ".join(parts)


def _resolve_nulls(k: OrderKey, ctx: RenderContext) -> str:
    if k.nulls == NullsOrder.FIRST:
        return "FIRST"
    if k.nulls == NullsOrder.LAST:
        return "LAST"
    default = (
        ctx.semantics.null_order_default_desc
        if k.direction == SortDirection.DESC
        else ctx.semantics.null_order_default_asc
    )
    return "FIRST" if default == NullOrder.FIRST else "LAST"


def _render_limit(p: Limit, ctx: RenderContext, *, top_level: bool) -> str:
    inner = _render(p.input, ctx)
    sub = ctx.fresh_alias("sub")
    parts = [f"SELECT * FROM ({inner}) AS {sub}"]
    if p.limit is not None:
        parts.append(f"LIMIT {p.limit}")
    if p.offset:
        parts.append(f"OFFSET {p.offset}")
    return " ".join(parts)


def _render_distinct(p: Distinct, ctx: RenderContext, *, top_level: bool) -> str:
    inner = _render(p.input, ctx)
    sub = ctx.fresh_alias("sub")
    return f"SELECT DISTINCT * FROM ({inner}) AS {sub}"


def _render_setop(p: SetOp, ctx: RenderContext, *, top_level: bool) -> str:
    left = _render(p.left, ctx)
    right = _render(p.right, ctx)
    all_mode = p.all or (ctx.semantics.set_op_default == SetOpDefault.ALL and not p.all)
    op_word = {
        SetOpKind.UNION: "UNION",
        SetOpKind.INTERSECT: "INTERSECT",
        SetOpKind.EXCEPT: "EXCEPT",
    }[p.kind]
    if all_mode:
        if p.kind == SetOpKind.EXCEPT and not ctx.flags.supports_except_all:
            raise UnsupportedByEngine(f"{ctx.flags.name} does not support EXCEPT ALL")
        if p.kind == SetOpKind.INTERSECT and not ctx.flags.supports_intersect_all:
            raise UnsupportedByEngine(f"{ctx.flags.name} does not support INTERSECT ALL")
        op_word += " ALL"
    # SQLite forbids parens around set-op operands; wrap each side in a
    # `SELECT * FROM (...) AS subN` so the surface is universally accepted.
    la = ctx.fresh_alias("sub")
    ra = ctx.fresh_alias("sub")
    return f"SELECT * FROM ({left}) AS {la} {op_word} SELECT * FROM ({right}) AS {ra}"


def _render_with(p: WithCTE, ctx: RenderContext, *, top_level: bool) -> str:
    parts: list[str] = []
    for b in p.bindings:
        parts.append(f"{_q(b.name, ctx)} AS ({_render(b.plan, ctx)})")
    body = _render(p.body, ctx)
    return f"WITH {', '.join(parts)} {body}"


def _render_recursive(p: RecursiveCTE, ctx: RenderContext, *, top_level: bool) -> str:
    if not ctx.flags.supports_recursive_cte:
        raise UnsupportedByEngine(f"{ctx.flags.name} does not support recursive CTE")
    seed = _render(p.seed, ctx)
    rec = _render(p.recursive, ctx)
    body = _render(p.body, ctx)
    union = "UNION ALL" if p.union_all else "UNION"
    return (
        f"WITH RECURSIVE {_q(p.name, ctx)} AS ({seed} {union} {rec}) {body}"
    )


# -------- expressions --------


def _render_expr(e: Expr, ctx: RenderContext) -> str:  # noqa: PLR0911, PLR0912
    if isinstance(e, Literal):
        return _render_literal(e)
    if isinstance(e, ColumnRef):
        if e.qualifier is not None:
            return _q(f"{e.qualifier}__{e.name}", ctx)
        return _q(e.name, ctx)
    if isinstance(e, BinaryOp):
        return _render_binary(e, ctx)
    if isinstance(e, UnaryOp):
        operand = _render_expr(e.operand, ctx)
        if e.op == Op.NOT:
            return f"(NOT {operand})"
        if e.op == Op.NEG:
            return f"(-{operand})"
        raise NotImplementedError(e.op)
    if isinstance(e, FuncCall):
        return _render_func(e, ctx)
    if isinstance(e, Case):
        return _render_case(e, ctx)
    if isinstance(e, Cast):
        operand = _render_expr(e.operand, ctx)
        return f"CAST({operand} AS {_render_type(e.target)})"
    if isinstance(e, InList):
        if not e.items:
            return "FALSE"
        operand = _render_expr(e.operand, ctx)
        items = ", ".join(_render_expr(i, ctx) for i in e.items)
        return f"({operand} IN ({items}))"
    if isinstance(e, IsNull):
        operand = _render_expr(e.operand, ctx)
        return f"({operand} IS {'NOT ' if e.negated else ''}NULL)"
    if isinstance(e, Between):
        operand = _render_expr(e.operand, ctx)
        low = _render_expr(e.low, ctx)
        high = _render_expr(e.high, ctx)
        word = "NOT BETWEEN" if e.negated else "BETWEEN"
        return f"({operand} {word} {low} AND {high})"
    if isinstance(e, AggCall):
        # Should only appear inside Aggregate; render generically just in case.
        return _render_aggregate_expr(e, ctx)
    if isinstance(e, ScalarSubquery):
        sub_sql = _render(e.plan, ctx)
        return f"({sub_sql})"
    if isinstance(e, ExistsSubquery):
        sub_sql = _render(e.plan, ctx)
        return f"({'NOT ' if e.negated else ''}EXISTS ({sub_sql}))"
    if isinstance(e, InSubquery):
        operand = _render_expr(e.operand, ctx)
        sub_sql = _render(e.plan, ctx)
        word = "NOT IN" if e.negated else "IN"
        return f"({operand} {word} ({sub_sql}))"
    raise NotImplementedError(type(e).__name__)


def _render_literal(e: Literal) -> str:
    v = e.value
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        escaped = v.replace("'", "''")
        return f"'{escaped}'"
    # Date / datetime: ISO string with explicit cast
    return f"'{v.isoformat()}'"


def _render_binary(e: BinaryOp, ctx: RenderContext) -> str:  # noqa: PLR0911
    left = _render_expr(e.left, ctx)
    right = _render_expr(e.right, ctx)
    op_map = {
        Op.EQ: "=",
        Op.NEQ: "<>",
        Op.LT: "<",
        Op.LTE: "<=",
        Op.GT: ">",
        Op.GTE: ">=",
        Op.AND: "AND",
        Op.OR: "OR",
        Op.ADD: "+",
        Op.SUB: "-",
        Op.MUL: "*",
        Op.MOD: "%",
        Op.CONCAT: "||",
    }
    if e.op in op_map:
        return f"({left} {op_map[e.op]} {right})"
    if e.op == Op.NULL_SAFE_EQ:
        return f"({left} IS NOT DISTINCT FROM {right})"
    if e.op == Op.DIV:
        return _render_div(left, right, e, ctx)
    if e.op == Op.LIKE:
        return f"({left} LIKE {right})" if ctx.semantics.like_case_sensitive else _ilike_emulated(
            left, right, ctx
        )
    if e.op == Op.ILIKE:
        if ctx.flags.supports_ilike:
            return f"({left} ILIKE {right})"
        return _ilike_emulated(left, right, ctx)
    raise NotImplementedError(e.op)


def _ilike_emulated(left: str, right: str, ctx: RenderContext) -> str:
    return f"(LOWER({left}) LIKE LOWER({right}))"


def _render_div(left: str, right: str, e: BinaryOp, ctx: RenderContext) -> str:
    is_int = (
        isinstance(e.left, Literal)
        and e.left.type.kind == TypeKind.INT
        and isinstance(e.right, Literal)
        and e.right.type.kind == TypeKind.INT
    )
    if is_int and ctx.semantics.integer_division == IntDivision.PROMOTE:
        base = f"(CAST({left} AS DOUBLE) / CAST({right} AS DOUBLE))"
    elif is_int and ctx.semantics.integer_division == IntDivision.TRUNCATE:
        base = f"({left} / {right})"  # SQL int / int = int truncation in DuckDB/SQLite
    else:
        base = f"({left} / {right})"
    if ctx.semantics.division_by_zero == DivByZero.NULL:
        return f"(CASE WHEN ({right}) = 0 THEN NULL ELSE {base} END)"
    if ctx.semantics.division_by_zero == DivByZero.ERROR:
        return base  # let engine error
    # INF: let IEEE-754 propagate (DuckDB does this for floats; ints will error)
    return base


def _render_func(e: FuncCall, ctx: RenderContext) -> str:
    name = e.name.upper()
    args = [_render_expr(a, ctx) for a in e.args]
    # Most ANSI-ish functions render as-is in DuckDB. SQLite quirks:
    if ctx.flags.name == "sqlite":
        if name in ("UPPER", "UCASE"):
            return f"UPPER({args[0]})"
        if name in ("LOWER", "LCASE"):
            return f"LOWER({args[0]})"
        if name in ("LENGTH", "LEN", "CHAR_LENGTH"):
            return f"LENGTH({args[0]})"
    if name == "LIKE":
        # LIKE-as-function form
        return f"({args[0]} LIKE {args[1]})"
    if name == "ILIKE":
        if ctx.flags.supports_ilike:
            return f"({args[0]} ILIKE {args[1]})"
        return _ilike_emulated(args[0], args[1], ctx)
    if name == "DATE_PART" or name == "EXTRACT":
        # EXTRACT(part FROM date) — both DuckDB and SQLite accept various forms.
        # Emit DuckDB's date_part(part, date) since it's broadly recognized.
        if ctx.flags.name == "sqlite":
            # SQLite: strftime('%Y', date) as int.
            unit = e.args[0].value.lower() if isinstance(e.args[0], Literal) else None
            fmt = {"year": "%Y", "month": "%m", "day": "%d"}.get(unit or "")
            if fmt is None:
                raise UnsupportedByEngine(f"sqlite EXTRACT unit: {unit}")
            return f"CAST(strftime('{fmt}', {args[1]}) AS INTEGER)"
        return f"DATE_PART({args[0]}, {args[1]})"
    if name == "DATE_DIFF":
        if ctx.flags.name == "sqlite":
            unit = e.args[0].value.lower() if isinstance(e.args[0], Literal) else None
            if unit not in ("day", "days"):
                raise UnsupportedByEngine(f"sqlite DATE_DIFF unit: {unit}")
            return f"CAST((julianday({args[2]}) - julianday({args[1]})) AS INTEGER)"
        return f"DATE_DIFF({args[0]}, {args[1]}, {args[2]})"
    if name == "DATE_ADD":
        if ctx.flags.name == "sqlite":
            return f"DATE({args[0]}, '+' || {args[1]} || ' days')"
        return f"({args[0]} + INTERVAL ({args[1]}) DAY)"
    if name == "DATE_SUB":
        if ctx.flags.name == "sqlite":
            return f"DATE({args[0]}, '-' || {args[1]} || ' days')"
        return f"({args[0]} - INTERVAL ({args[1]}) DAY)"
    if name in ("IF", "IIF"):
        return f"CASE WHEN {args[0]} THEN {args[1]} ELSE {args[2]} END"
    return f"{name}({', '.join(args)})"


def _render_case(e: Case, ctx: RenderContext) -> str:
    parts = ["CASE"]
    for cond, val in e.branches:
        parts.append(f"WHEN {_render_expr(cond, ctx)} THEN {_render_expr(val, ctx)}")
    if e.default is not None:
        parts.append(f"ELSE {_render_expr(e.default, ctx)}")
    parts.append("END")
    return " ".join(parts)


def _render_type(t: IRType) -> str:
    return {
        TypeKind.INT: "BIGINT",
        TypeKind.FLOAT: "DOUBLE",
        TypeKind.TEXT: "VARCHAR",
        TypeKind.BOOL: "BOOLEAN",
        TypeKind.DATE: "DATE",
        TypeKind.TIMESTAMP: "TIMESTAMP",
        TypeKind.NULL: "NULL",
    }[t.kind]


__all__ = [
    "render_plan",
    "SqlDialectFlags",
    "UnsupportedByEngine",
]
