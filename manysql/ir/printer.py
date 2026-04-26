"""Pretty-printer for IR plans and expressions.

Used for debugging, test fixtures, and codegen-failure logs. The output is
deliberately not parseable; for parseable rendering use the IR -> SQL
renderers in manysql/oracle/.
"""

from __future__ import annotations

from manysql.ir.expr import (
    AggCall,
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
    OrderKey,
    ScalarSubquery,
    UnaryOp,
    WindowCall,
)
from manysql.ir.plan import (
    Aggregate,
    Apply,
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


def _pad(indent: int) -> str:
    return "  " * indent


def render_expr(e: Expr) -> str:
    if isinstance(e, Literal):
        return f"lit({e.value!r}:{e.type})"
    if isinstance(e, ColumnRef):
        return f"{e.qualifier}.{e.name}" if e.qualifier else e.name
    if isinstance(e, BinaryOp):
        return f"({render_expr(e.left)} {e.op.value} {render_expr(e.right)})"
    if isinstance(e, UnaryOp):
        return f"({e.op.value} {render_expr(e.operand)})"
    if isinstance(e, FuncCall):
        return f"{e.name}({', '.join(render_expr(a) for a in e.args)})"
    if isinstance(e, Case):
        parts = [
            f"WHEN {render_expr(c)} THEN {render_expr(v)}" for c, v in e.branches
        ]
        if e.default is not None:
            parts.append(f"ELSE {render_expr(e.default)}")
        return "CASE " + " ".join(parts) + " END"
    if isinstance(e, Cast):
        return f"CAST({render_expr(e.operand)} AS {e.target})"
    if isinstance(e, InList):
        items = ", ".join(render_expr(i) for i in e.items)
        return f"{render_expr(e.operand)} IN ({items})"
    if isinstance(e, IsNull):
        op = "IS NOT NULL" if e.negated else "IS NULL"
        return f"({render_expr(e.operand)} {op})"
    if isinstance(e, Between):
        op = "NOT BETWEEN" if e.negated else "BETWEEN"
        return f"({render_expr(e.operand)} {op} {render_expr(e.low)} AND {render_expr(e.high)})"
    if isinstance(e, AggCall):
        arg = render_expr(e.arg) if e.arg is not None else "*"
        d = "DISTINCT " if e.distinct else ""
        out = f"{e.kind.value}({d}{arg})"
        if e.filter_pred is not None:
            out += f" FILTER({render_expr(e.filter_pred)})"
        return out
    if isinstance(e, WindowCall):
        args = ", ".join(render_expr(a) for a in e.args)
        parts = [f"{e.kind.value}({args})"]
        if e.partition_by:
            parts.append(
                "PARTITION BY " + ", ".join(render_expr(p) for p in e.partition_by)
            )
        if e.order_by:
            parts.append("ORDER BY " + ", ".join(_render_order(k) for k in e.order_by))
        if e.frame is not None:
            parts.append(
                f"FRAME {e.frame.mode.value} {_render_bound(e.frame.start)} TO "
                f"{_render_bound(e.frame.end)}"
            )
        if e.ignore_nulls:
            parts.append("IGNORE NULLS")
        return " ".join(parts)
    if isinstance(e, ScalarSubquery):
        return f"SCALAR_SUB({render_plan(e.plan).strip()})"
    if isinstance(e, ExistsSubquery):
        head = "NOT EXISTS" if e.negated else "EXISTS"
        return f"{head}({render_plan(e.plan).strip()})"
    if isinstance(e, InSubquery):
        head = "NOT IN" if e.negated else "IN"
        return f"({render_expr(e.operand)} {head} {render_plan(e.plan).strip()})"
    return f"<{type(e).__name__}>"


def _render_order(k: OrderKey) -> str:
    return f"{render_expr(k.expr)} {k.direction.value} {k.nulls.value}"


def _render_bound(b) -> str:
    if b.offset is not None:
        return f"{b.offset} {b.kind.value.split(' ')[-1]}"
    return b.kind.value


def render_plan(p: Plan, indent: int = 0) -> str:  # noqa: PLR0911, PLR0912
    pad = _pad(indent)
    if isinstance(p, Scan):
        cols = ", ".join(f"{c.name}:{c.type}" for c in p.columns)
        alias = f" AS {p.alias}" if p.alias else ""
        return f"{pad}Scan {p.table_name}{alias} ({cols})\n"
    if isinstance(p, Project):
        out = f"{pad}Project [{', '.join(f'{n}={render_expr(e)}' for n, e in p.projections)}]\n"
        return out + render_plan(p.input, indent + 1)
    if isinstance(p, Filter):
        return f"{pad}Filter {render_expr(p.predicate)}\n" + render_plan(
            p.input, indent + 1
        )
    if isinstance(p, Join):
        cond = (
            f"ON {render_expr(p.on)}"
            if p.on is not None
            else (f"USING ({', '.join(p.using)})" if p.using else "")
        )
        out = f"{pad}{p.kind.value} JOIN {cond}\n"
        return out + render_plan(p.left, indent + 1) + render_plan(p.right, indent + 1)
    if isinstance(p, Aggregate):
        gb = ", ".join(f"{n}={render_expr(e)}" for n, e in p.group_by)
        ag = ", ".join(f"{n}={render_expr(e)}" for n, e in p.aggregates)
        return (
            f"{pad}Aggregate group=[{gb}] agg=[{ag}]\n" + render_plan(p.input, indent + 1)
        )
    if isinstance(p, Window):
        ws = ", ".join(f"{n}={render_expr(e)}" for n, e in p.windows)
        return f"{pad}Window [{ws}]\n" + render_plan(p.input, indent + 1)
    if isinstance(p, Sort):
        keys = ", ".join(_render_order(k) for k in p.keys)
        return f"{pad}Sort [{keys}]\n" + render_plan(p.input, indent + 1)
    if isinstance(p, Limit):
        return f"{pad}Limit {p.limit} OFFSET {p.offset}\n" + render_plan(
            p.input, indent + 1
        )
    if isinstance(p, Distinct):
        return f"{pad}Distinct\n" + render_plan(p.input, indent + 1)
    if isinstance(p, SetOp):
        suffix = " ALL" if p.all else ""
        out = f"{pad}{p.kind.value}{suffix}\n"
        return out + render_plan(p.left, indent + 1) + render_plan(p.right, indent + 1)
    if isinstance(p, WithCTE):
        out = f"{pad}WithCTE\n"
        for b in p.bindings:
            out += f"{_pad(indent + 1)}{b.name} =\n"
            out += render_plan(b.plan, indent + 2)
        out += f"{_pad(indent + 1)}body =\n"
        out += render_plan(p.body, indent + 2)
        return out
    if isinstance(p, RecursiveCTE):
        out = f"{pad}RecursiveCTE {p.name} (union_all={p.union_all})\n"
        out += f"{_pad(indent + 1)}seed =\n" + render_plan(p.seed, indent + 2)
        out += f"{_pad(indent + 1)}recursive =\n" + render_plan(p.recursive, indent + 2)
        out += f"{_pad(indent + 1)}body =\n" + render_plan(p.body, indent + 2)
        return out
    if isinstance(p, Apply):
        head = f"{pad}Apply {p.kind.value}"
        if p.output_name:
            head += f" -> {p.output_name}"
        out = head + "\n"
        return out + render_plan(p.outer, indent + 1) + render_plan(p.inner, indent + 1)
    return f"{pad}<{type(p).__name__}>\n"
