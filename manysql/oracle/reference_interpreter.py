"""Python reference IR interpreter.

Slow, ruthlessly readable, hand-coded semantics. Different code path from the
Polars executor in manysql/executor/. The whole point of this oracle: when
both implementations agree on a result, that result is highly likely correct,
because bugs in two independently-written implementations are unlikely to
collide.

This interpreter operates over plain Python lists of dicts (one dict per row)
to maximize independence from the Polars executor's vectorized abstractions.
It is the *only* oracle that can verify dialects with semantics no SQL engine
expresses, because its semantics are ours, not borrowed from a host engine.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

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
    SetOpKind,
    Sort,
    Window,
    WithCTE,
)
from manysql.ir.types import IRType, TypeKind
from manysql.oracle.base import Oracle, OracleCapability, OracleResult
from manysql.spec.semantics import (
    BoolTruthiness,
    DivByZero,
    IntDivision,
    NullOrder,
    SemanticConfig,
    SetOpDefault,
)

Row = dict[str, Any]
Rows = list[Row]


def _truthy(v: Any, semantics: SemanticConfig) -> Optional[bool]:
    """Three-valued logic: returns True/False or None for SQL NULL."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if semantics.boolean_truthiness == BoolTruthiness.C_STYLE:
        if isinstance(v, (int, float)):
            return v != 0
        if isinstance(v, str):
            return len(v) > 0
        return bool(v)
    # STRICT: only bools count; non-bool here is a coercion error in real SQL,
    # but we treat as Python truthiness for resilience and let cross-oracle
    # comparison catch dialect-specific strictness errors elsewhere.
    return bool(v)


def _three_valued_and(a: Optional[bool], b: Optional[bool]) -> Optional[bool]:
    if a is False or b is False:
        return False
    if a is None or b is None:
        return None
    return True


def _three_valued_or(a: Optional[bool], b: Optional[bool]) -> Optional[bool]:
    if a is True or b is True:
        return True
    if a is None or b is None:
        return None
    return False


def _three_valued_not(a: Optional[bool]) -> Optional[bool]:
    if a is None:
        return None
    return not a


def _eq(a: Any, b: Any) -> Optional[bool]:
    if a is None or b is None:
        return None
    return a == b


def _cmp(a: Any, b: Any, op: Op) -> Optional[bool]:
    if a is None or b is None:
        return None
    if op == Op.LT:
        return a < b
    if op == Op.LTE:
        return a <= b
    if op == Op.GT:
        return a > b
    if op == Op.GTE:
        return a >= b
    raise NotImplementedError(op)


def _arith(a: Any, b: Any, op: Op, semantics: SemanticConfig) -> Any:
    if a is None or b is None:
        return None
    if op == Op.ADD:
        return a + b
    if op == Op.SUB:
        return a - b
    if op == Op.MUL:
        return a * b
    if op == Op.MOD:
        return a % b
    if op == Op.DIV:
        if b == 0:
            mode = semantics.division_by_zero
            if mode == DivByZero.NULL:
                return None
            if mode == DivByZero.ERROR:
                raise ZeroDivisionError("division by zero")
            if mode == DivByZero.INF:
                # Python int/0 raises; only float supports inf.
                if isinstance(a, float) or isinstance(b, float):
                    return float("inf") if a > 0 else float("-inf") if a < 0 else float("nan")
                raise ZeroDivisionError("division by zero (integer, INF mode applies to float)")
        if isinstance(a, int) and isinstance(b, int):
            if semantics.integer_division == IntDivision.TRUNCATE:
                # SQL-style truncate-toward-zero (not Python's floor-toward-negative-inf)
                q = a // b
                # Adjust if the result has the wrong sign for SQL truncation
                if (a % b != 0) and ((a < 0) != (b < 0)):
                    q += 1
                return q
            return a / b
        return a / b
    raise NotImplementedError(op)


def _like(value: Any, pattern: Any, *, case_sensitive: bool) -> Optional[bool]:
    if value is None or pattern is None:
        return None
    import re

    out: list[str] = []
    i = 0
    s = str(pattern)
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s):
            out.append(re.escape(s[i + 1]))
            i += 2
            continue
        if ch == "%":
            out.append(".*")
        elif ch == "_":
            out.append(".")
        else:
            out.append(re.escape(ch))
        i += 1
    flags = 0 if case_sensitive else re.IGNORECASE
    return bool(re.fullmatch("".join(out), str(value), flags))


def _polars_to_rows(df: pl.DataFrame) -> Rows:
    return df.to_dicts()


def _rows_to_polars(
    rows: Rows, schema: Optional[dict[str, pl.DataType]] = None
) -> pl.DataFrame:
    if not rows:
        if schema:
            return pl.DataFrame(schema=schema)
        return pl.DataFrame()
    if schema:
        return pl.DataFrame(rows, schema=schema)
    return pl.DataFrame(rows)


def _resolve_col(row: Row, ref: ColumnRef) -> Any:
    # Try qualified, then unqualified, to match the executor's renaming convention.
    if ref.qualifier is not None:
        qkey = f"{ref.qualifier}__{ref.name}"
        if qkey in row:
            return row[qkey]
    if ref.name in row:
        return row[ref.name]
    if ref.qualifier is not None:
        qdot = f"{ref.qualifier}.{ref.name}"
        if qdot in row:
            return row[qdot]
    raise KeyError(f"column not found: {ref.qualifier}.{ref.name}")


class ReferenceInterpreter(Oracle):
    """Hand-coded IR interpreter over plain Python rows."""

    @property
    def capability(self) -> OracleCapability:
        return OracleCapability(
            name="reference_interpreter",
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
            supported_features=frozenset(
                {
                    "all_tier_a",
                    "correlated_subquery",
                    "recursive_cte",
                    "custom_null_rules",
                    "custom_div_by_zero",
                    "custom_int_division",
                    "custom_like_case",
                    "custom_set_op_default",
                    "custom_window_default_frame",
                    "non_duckdb_semantics",
                }
            ),
            confidence=0.85,
        )

    def evaluate(
        self,
        plan: Plan,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
    ) -> OracleResult:
        try:
            rows = self._exec_plan(plan, semantics, catalog, {}, {})
            if rows:
                df = _rows_to_polars(rows)
            else:
                # Preserve schema for empty results so harness comparisons
                # (which check column names) can still match other oracles.
                schema_cols = [c.name for c in plan.schema()]
                df = pl.DataFrame({c: [] for c in schema_cols})
            return OracleResult(oracle=self.capability.name, rows=df)
        except Exception as exc:  # noqa: BLE001
            return OracleResult(oracle=self.capability.name, error=f"{type(exc).__name__}: {exc}")

    # -------- plan dispatch --------

    def _exec_plan(  # noqa: PLR0911
        self,
        p: Plan,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
        outer_row: Row,
        cte_bindings: dict[str, Rows],
    ) -> Rows:
        if isinstance(p, Scan):
            return self._scan(p, catalog, cte_bindings)
        if isinstance(p, Project):
            return self._project(p, semantics, catalog, outer_row, cte_bindings)
        if isinstance(p, Filter):
            return self._filter(p, semantics, catalog, outer_row, cte_bindings)
        if isinstance(p, Join):
            return self._join(p, semantics, catalog, outer_row, cte_bindings)
        if isinstance(p, Aggregate):
            return self._aggregate(p, semantics, catalog, outer_row, cte_bindings)
        if isinstance(p, Window):
            return self._window(p, semantics, catalog, outer_row, cte_bindings)
        if isinstance(p, Sort):
            return self._sort(p, semantics, catalog, outer_row, cte_bindings)
        if isinstance(p, Limit):
            return self._limit(p, semantics, catalog, outer_row, cte_bindings)
        if isinstance(p, Distinct):
            inp = self._exec_plan(p.input, semantics, catalog, outer_row, cte_bindings)
            return self._unique(inp)
        if isinstance(p, SetOp):
            return self._setop(p, semantics, catalog, outer_row, cte_bindings)
        if isinstance(p, WithCTE):
            return self._with_cte(p, semantics, catalog, outer_row, cte_bindings)
        if isinstance(p, RecursiveCTE):
            return self._recursive_cte(p, semantics, catalog, outer_row, cte_bindings)
        if isinstance(p, Apply):
            return self._apply(p, semantics, catalog, outer_row, cte_bindings)
        raise NotImplementedError(type(p).__name__)

    # -------- handlers --------

    def _scan(
        self,
        p: Scan,
        catalog: dict[str, pl.DataFrame],
        cte_bindings: dict[str, Rows],
    ) -> Rows:
        if p.table_name in cte_bindings:
            rows = [dict(r) for r in cte_bindings[p.table_name]]
        elif p.table_name in catalog:
            rows = _polars_to_rows(catalog[p.table_name])
        else:
            raise KeyError(f"Unknown table: {p.table_name}")
        if p.alias is not None:
            renamed = []
            for r in rows:
                renamed.append({f"{p.alias}__{k}": v for k, v in r.items()})
            return renamed
        return rows

    def _project(
        self,
        p: Project,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
        outer_row: Row,
        cte_bindings: dict[str, Rows],
    ) -> Rows:
        inp = self._exec_plan(p.input, semantics, catalog, outer_row, cte_bindings)
        out: Rows = []
        for r in inp:
            new_row: Row = {}
            for name, e in p.projections:
                new_row[name] = self._eval_expr(e, r, semantics, catalog, outer_row, cte_bindings)
            out.append(new_row)
        return out

    def _filter(
        self,
        p: Filter,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
        outer_row: Row,
        cte_bindings: dict[str, Rows],
    ) -> Rows:
        inp = self._exec_plan(p.input, semantics, catalog, outer_row, cte_bindings)
        out: Rows = []
        for r in inp:
            v = self._eval_expr(p.predicate, r, semantics, catalog, outer_row, cte_bindings)
            if _truthy(v, semantics) is True:
                out.append(r)
        return out

    def _join(
        self,
        p: Join,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
        outer_row: Row,
        cte_bindings: dict[str, Rows],
    ) -> Rows:
        left = self._exec_plan(p.left, semantics, catalog, outer_row, cte_bindings)
        right = self._exec_plan(p.right, semantics, catalog, outer_row, cte_bindings)
        out: Rows = []
        if p.kind == JoinKind.CROSS:
            for lr in left:
                for rr in right:
                    out.append({**lr, **rr})
            return out

        # ON-style join (USING is treated by name equality for the listed cols)
        on_cols = list(p.using)

        def predicate(lr: Row, rr: Row) -> bool:
            merged = {**lr, **rr}
            if on_cols:
                return all(_eq(lr.get(c), rr.get(c)) is True for c in on_cols)
            assert p.on is not None
            v = self._eval_expr(p.on, merged, semantics, catalog, outer_row, cte_bindings)
            return _truthy(v, semantics) is True

        if p.kind in (JoinKind.SEMI, JoinKind.ANTI):
            keep_match = p.kind == JoinKind.SEMI
            for lr in left:
                matched = any(predicate(lr, rr) for rr in right)
                if matched == keep_match:
                    out.append(lr)
            return out

        matched_right_idxs: set[int] = set()
        for lr in left:
            matched = False
            for j, rr in enumerate(right):
                if predicate(lr, rr):
                    out.append({**lr, **rr})
                    matched_right_idxs.add(j)
                    matched = True
            if not matched and p.kind in (JoinKind.LEFT, JoinKind.FULL):
                null_right = (
                    {k: None for k in right[0]} if right else {}
                )
                out.append({**lr, **null_right})
        if p.kind in (JoinKind.RIGHT, JoinKind.FULL):
            for j, rr in enumerate(right):
                if j not in matched_right_idxs:
                    null_left = {k: None for k in left[0]} if left else {}
                    out.append({**null_left, **rr})
        return out

    def _aggregate(
        self,
        p: Aggregate,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
        outer_row: Row,
        cte_bindings: dict[str, Rows],
    ) -> Rows:
        inp = self._exec_plan(p.input, semantics, catalog, outer_row, cte_bindings)

        if p.group_by:
            # Compute group keys per row, group rows by key tuple, agg each.
            groups: dict[tuple, Rows] = {}
            order: list[tuple] = []
            for r in inp:
                key = tuple(
                    self._eval_expr(e, r, semantics, catalog, outer_row, cte_bindings)
                    for _, e in p.group_by
                )
                if key not in groups:
                    groups[key] = []
                    order.append(key)
                groups[key].append(r)
            out: Rows = []
            for key in order:
                bucket = groups[key]
                row: Row = {n: v for (n, _), v in zip(p.group_by, key, strict=True)}
                for name, agg_expr in p.aggregates:
                    row[name] = self._eval_agg(
                        agg_expr, bucket, semantics, catalog, outer_row, cte_bindings
                    )
                out.append(row)
            return out

        # No group-by: single aggregate row.
        row: Row = {}
        for name, agg_expr in p.aggregates:
            row[name] = self._eval_agg(
                agg_expr, inp, semantics, catalog, outer_row, cte_bindings
            )
        return [row]

    def _eval_agg(
        self,
        e: Expr,
        bucket: Rows,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
        outer_row: Row,
        cte_bindings: dict[str, Rows],
    ) -> Any:
        if not isinstance(e, AggCall):
            # Not all aggregate-position expressions need to be AggCall (e.g. literals
            # in some lowerings). Evaluate against an arbitrary row context.
            ref_row = bucket[0] if bucket else {}
            return self._eval_expr(e, ref_row, semantics, catalog, outer_row, cte_bindings)
        # Filter rows
        rows = bucket
        if e.filter_pred is not None:
            rows = [
                r
                for r in rows
                if _truthy(
                    self._eval_expr(
                        e.filter_pred, r, semantics, catalog, outer_row, cte_bindings
                    ),
                    semantics,
                )
                is True
            ]

        if e.kind == AggKind.COUNT_STAR:
            return len(rows)

        assert e.arg is not None
        values: list[Any] = []
        for r in rows:
            values.append(
                self._eval_expr(e.arg, r, semantics, catalog, outer_row, cte_bindings)
            )

        if e.kind == AggKind.COUNT:
            non_null = [v for v in values if v is not None]
            if e.distinct:
                non_null = list({_make_hashable(v): v for v in non_null}.values())
            return len(non_null)
        if e.kind == AggKind.SUM:
            non_null = [v for v in values if v is not None]
            if not non_null:
                return None if semantics.sum_of_empty_returns_null else 0
            if e.distinct:
                non_null = list({_make_hashable(v): v for v in non_null}.values())
            return sum(non_null)
        if e.kind == AggKind.AVG:
            non_null = [v for v in values if v is not None]
            if not non_null:
                return None
            if e.distinct:
                non_null = list({_make_hashable(v): v for v in non_null}.values())
            return sum(non_null) / len(non_null)
        if e.kind == AggKind.MIN:
            non_null = [v for v in values if v is not None]
            return min(non_null) if non_null else None
        if e.kind == AggKind.MAX:
            non_null = [v for v in values if v is not None]
            return max(non_null) if non_null else None
        raise NotImplementedError(e.kind)

    def _window(
        self,
        p: Window,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
        outer_row: Row,
        cte_bindings: dict[str, Rows],
    ) -> Rows:
        inp = [dict(r) for r in self._exec_plan(p.input, semantics, catalog, outer_row, cte_bindings)]
        for name, e in p.windows:
            assert isinstance(e, WindowCall)
            self._apply_window(e, name, inp, semantics, catalog, outer_row, cte_bindings)
        return inp

    def _apply_window(
        self,
        w: WindowCall,
        out_name: str,
        rows: Rows,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
        outer_row: Row,
        cte_bindings: dict[str, Rows],
    ) -> None:
        # Group by partition keys
        def part_key(r: Row) -> tuple:
            return tuple(
                self._eval_expr(p, r, semantics, catalog, outer_row, cte_bindings)
                for p in w.partition_by
            )

        partitions: dict[tuple, list[int]] = {}
        for i, r in enumerate(rows):
            partitions.setdefault(part_key(r), []).append(i)

        for indices in partitions.values():
            sub_rows = [rows[i] for i in indices]
            # Sort by ORDER BY, applying default null order from semantics.
            if w.order_by:
                sub_rows = self._sort_rows(sub_rows, w.order_by, semantics, catalog, outer_row, cte_bindings)
                # Map back: indices reordered to match sub_rows.
                # We need original positions of sub_rows.
                # Simpler: assign output to rows based on identity of sub_rows.
                self._compute_window_values(
                    w, out_name, sub_rows, semantics, catalog, outer_row, cte_bindings
                )
            else:
                self._compute_window_values(
                    w, out_name, sub_rows, semantics, catalog, outer_row, cte_bindings
                )

    def _compute_window_values(
        self,
        w: WindowCall,
        out_name: str,
        rows: Rows,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
        outer_row: Row,
        cte_bindings: dict[str, Rows],
    ) -> None:
        n = len(rows)
        if w.kind == WindowKind.ROW_NUMBER:
            for i, r in enumerate(rows):
                r[out_name] = i + 1
            return
        if w.kind == WindowKind.RANK:
            self._rank_like(rows, w, out_name, dense=False, semantics=semantics, catalog=catalog, outer_row=outer_row, cte_bindings=cte_bindings)
            return
        if w.kind == WindowKind.DENSE_RANK:
            self._rank_like(rows, w, out_name, dense=True, semantics=semantics, catalog=catalog, outer_row=outer_row, cte_bindings=cte_bindings)
            return
        if w.kind in (WindowKind.LAG, WindowKind.LEAD):
            arg_expr = w.args[0]
            offset = 1
            if len(w.args) > 1 and isinstance(w.args[1], Literal):
                offset = int(w.args[1].value or 1)
            for i, r in enumerate(rows):
                target_idx = i - offset if w.kind == WindowKind.LAG else i + offset
                if 0 <= target_idx < n:
                    r[out_name] = self._eval_expr(
                        arg_expr, rows[target_idx], semantics, catalog, outer_row, cte_bindings
                    )
                else:
                    r[out_name] = None
            return
        if w.kind == WindowKind.FIRST_VALUE:
            arg_expr = w.args[0]
            v = (
                self._eval_expr(arg_expr, rows[0], semantics, catalog, outer_row, cte_bindings)
                if rows
                else None
            )
            for r in rows:
                r[out_name] = v
            return
        if w.kind == WindowKind.LAST_VALUE:
            arg_expr = w.args[0]
            for i, r in enumerate(rows):
                # SQL's LAST_VALUE with default frame is "up to current row"; we
                # honor the default-frame knob when no frame given.
                if not w.order_by or w.frame is not None:
                    v = (
                        self._eval_expr(arg_expr, rows[-1], semantics, catalog, outer_row, cte_bindings)
                        if rows
                        else None
                    )
                else:
                    v = self._eval_expr(arg_expr, rows[i], semantics, catalog, outer_row, cte_bindings)
                r[out_name] = v
            return

        # Aggregate windows
        if w.kind in (WindowKind.SUM, WindowKind.AVG, WindowKind.MIN, WindowKind.MAX, WindowKind.COUNT):
            arg_expr = w.args[0] if w.args else Literal(1, IRType(TypeKind.INT))
            running = bool(w.order_by) and w.frame is None  # default frame when ORDER BY present
            full_values: list[Any] = [
                self._eval_expr(arg_expr, r, semantics, catalog, outer_row, cte_bindings)
                for r in rows
            ]
            if running:
                for i, r in enumerate(rows):
                    r[out_name] = self._reduce(full_values[: i + 1], w.kind, semantics)
            else:
                v = self._reduce(full_values, w.kind, semantics)
                for r in rows:
                    r[out_name] = v
            return
        raise NotImplementedError(w.kind)

    def _rank_like(
        self,
        rows: Rows,
        w: WindowCall,
        out_name: str,
        *,
        dense: bool,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
        outer_row: Row,
        cte_bindings: dict[str, Rows],
    ) -> None:
        if not w.order_by:
            for r in rows:
                r[out_name] = 1
            return
        rank = 0
        last_key: Any = object()
        seen = 0
        for r in rows:
            seen += 1
            key = tuple(
                self._eval_expr(k.expr, r, semantics, catalog, outer_row, cte_bindings)
                for k in w.order_by
            )
            if key != last_key:
                rank = seen if not dense else (rank + 1)
                last_key = key
            r[out_name] = rank

    @staticmethod
    def _reduce(values: list[Any], kind: WindowKind, semantics: SemanticConfig) -> Any:
        non_null = [v for v in values if v is not None]
        if kind == WindowKind.COUNT:
            return len(non_null)
        if not non_null:
            if kind == WindowKind.SUM:
                return None if semantics.sum_of_empty_returns_null else 0
            return None
        if kind == WindowKind.SUM:
            return sum(non_null)
        if kind == WindowKind.AVG:
            return sum(non_null) / len(non_null)
        if kind == WindowKind.MIN:
            return min(non_null)
        if kind == WindowKind.MAX:
            return max(non_null)
        raise NotImplementedError(kind)

    def _sort(
        self,
        p: Sort,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
        outer_row: Row,
        cte_bindings: dict[str, Rows],
    ) -> Rows:
        inp = self._exec_plan(p.input, semantics, catalog, outer_row, cte_bindings)
        return self._sort_rows(inp, list(p.keys), semantics, catalog, outer_row, cte_bindings)

    def _sort_rows(
        self,
        rows: Rows,
        keys: list[OrderKey],
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
        outer_row: Row,
        cte_bindings: dict[str, Rows],
    ) -> Rows:
        # Pre-evaluate each key per row to a sort tuple.
        def nulls_last_for(k: OrderKey) -> bool:
            if k.nulls == NullsOrder.FIRST:
                return False
            if k.nulls == NullsOrder.LAST:
                return True
            default = (
                semantics.null_order_default_desc
                if k.direction == SortDirection.DESC
                else semantics.null_order_default_asc
            )
            return default == NullOrder.LAST

        directions = [k.direction == SortDirection.DESC for k in keys]
        nulls_last = [nulls_last_for(k) for k in keys]

        # Build comparable tuples; for NULLs we use a sentinel that places them
        # appropriately (first or last) per direction.
        def make_key(r: Row) -> tuple:
            out: list[tuple] = []
            for k, desc, nl in zip(keys, directions, nulls_last, strict=True):
                v = self._eval_expr(k.expr, r, semantics, catalog, outer_row, cte_bindings)
                # Sort tuple: (is_null_priority, value_for_compare).
                # Lower priority sorts first.
                if v is None:
                    out.append((1 if nl else -1, None))
                else:
                    cmp_value: Any = v
                    if desc:
                        cmp_value = _Reversed(v)
                    out.append((0, cmp_value))
            return tuple(out)

        return sorted(rows, key=make_key)

    def _limit(
        self,
        p: Limit,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
        outer_row: Row,
        cte_bindings: dict[str, Rows],
    ) -> Rows:
        rows = self._exec_plan(p.input, semantics, catalog, outer_row, cte_bindings)
        rows = rows[p.offset :] if p.offset else rows
        if p.limit is not None:
            rows = rows[: p.limit]
        return rows

    def _unique(self, rows: Rows) -> Rows:
        seen: set[tuple] = set()
        out: Rows = []
        for r in rows:
            key = tuple((k, _make_hashable(v)) for k, v in sorted(r.items()))
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    def _setop(
        self,
        p: SetOp,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
        outer_row: Row,
        cte_bindings: dict[str, Rows],
    ) -> Rows:
        left = self._exec_plan(p.left, semantics, catalog, outer_row, cte_bindings)
        right = self._exec_plan(p.right, semantics, catalog, outer_row, cte_bindings)
        all_mode = p.all or (semantics.set_op_default == SetOpDefault.ALL and not p.all)

        # SQL set ops are positional, not name-matched. Re-key right rows by
        # left column names so the comparison is correct even when sides have
        # different column labels.
        left_cols = list(left[0].keys()) if left else (list(right[0].keys()) if right else [])
        right = self._align_columns_positional(right, left_cols)

        def row_tuple(r: Row) -> tuple:
            return tuple(_make_hashable(r.get(c)) for c in left_cols)

        if p.kind == SetOpKind.UNION:
            combined = left + right
            return combined if all_mode else self._unique(combined)
        if p.kind == SetOpKind.INTERSECT:
            right_set = {row_tuple(r) for r in right}
            kept = [r for r in left if row_tuple(r) in right_set]
            return kept if all_mode else self._unique(kept)
        if p.kind == SetOpKind.EXCEPT:
            if all_mode:
                # EXCEPT ALL: multiset subtraction.
                right_counts: dict[tuple, int] = {}
                for r in right:
                    k = row_tuple(r)
                    right_counts[k] = right_counts.get(k, 0) + 1
                kept: list[Row] = []
                for r in left:
                    k = row_tuple(r)
                    if right_counts.get(k, 0) > 0:
                        right_counts[k] -= 1
                        continue
                    kept.append(r)
                return kept
            right_set = {row_tuple(r) for r in right}
            kept = [r for r in left if row_tuple(r) not in right_set]
            return self._unique(kept)
        raise NotImplementedError(p.kind)

    @staticmethod
    def _align_columns_positional(rows: "Rows", target_cols: list[str]) -> "Rows":
        if not rows:
            return rows
        src_cols = list(rows[0].keys())
        if src_cols == target_cols:
            return rows
        out: list[Row] = []
        for r in rows:
            vals = list(r.values())
            out.append({c: vals[i] for i, c in enumerate(target_cols) if i < len(vals)})
        return out

    def _with_cte(
        self,
        p: WithCTE,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
        outer_row: Row,
        cte_bindings: dict[str, Rows],
    ) -> Rows:
        local = dict(cte_bindings)
        for b in p.bindings:
            local[b.name] = self._exec_plan(b.plan, semantics, catalog, outer_row, local)
        return self._exec_plan(p.body, semantics, catalog, outer_row, local)

    def _recursive_cte(
        self,
        p: RecursiveCTE,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
        outer_row: Row,
        cte_bindings: dict[str, Rows],
    ) -> Rows:
        local = dict(cte_bindings)
        seed = self._exec_plan(p.seed, semantics, catalog, outer_row, local)
        accumulated = list(seed)
        new_batch = list(seed)
        max_iters = 10_000
        iters = 0
        while new_batch and iters < max_iters:
            local[p.name] = new_batch
            step = self._exec_plan(p.recursive, semantics, catalog, outer_row, local)
            seen_keys = {
                tuple(sorted((k, _make_hashable(v)) for k, v in r.items())) for r in accumulated
            }
            new_batch = [
                r
                for r in step
                if tuple(sorted((k, _make_hashable(v)) for k, v in r.items())) not in seen_keys
            ]
            if not new_batch:
                break
            if p.union_all:
                accumulated.extend(new_batch)
            else:
                accumulated = self._unique(accumulated + new_batch)
            iters += 1
        if iters == max_iters:
            raise RuntimeError(f"RecursiveCTE {p.name!r}: exceeded {max_iters} iterations")
        local[p.name] = accumulated
        return self._exec_plan(p.body, semantics, catalog, outer_row, local)

    def _apply(
        self,
        p: Apply,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
        outer_row: Row,
        cte_bindings: dict[str, Rows],
    ) -> Rows:
        outer = self._exec_plan(p.outer, semantics, catalog, outer_row, cte_bindings)
        out: Rows = []
        for r in outer:
            merged_outer = {**outer_row, **r}
            inner = self._exec_plan(p.inner, semantics, catalog, merged_outer, cte_bindings)

            if p.kind == ApplyKind.SCALAR:
                assert p.output_name is not None
                if not inner:
                    value = None
                else:
                    first_key = next(iter(inner[0]))
                    value = inner[0][first_key]
                out.append({**r, p.output_name: value})
            elif p.kind in (ApplyKind.EXISTS, ApplyKind.NOT_EXISTS):
                assert p.output_name is not None
                exists = bool(inner)
                v = exists if p.kind == ApplyKind.EXISTS else not exists
                out.append({**r, p.output_name: v})
            elif p.kind == ApplyKind.CROSS:
                for inner_row in inner:
                    out.append({**r, **inner_row})
            elif p.kind == ApplyKind.OUTER:
                if not inner:
                    null_filled = (
                        {k: None for k in inner[0]} if inner else {}
                    )
                    out.append({**r, **null_filled})
                else:
                    for inner_row in inner:
                        out.append({**r, **inner_row})
            else:
                raise NotImplementedError(p.kind)
        return out

    # -------- expression eval --------

    def _eval_expr(  # noqa: PLR0911, PLR0912
        self,
        e: Expr,
        row: Row,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
        outer_row: Row,
        cte_bindings: dict[str, Rows],
    ) -> Any:
        if isinstance(e, Literal):
            return e.value
        if isinstance(e, ColumnRef):
            # Inner scope wins over outer scope. Only fall back to outer_row if
            # the column isn't present locally — this is the standard SQL
            # name-resolution rule for correlated subqueries.
            try:
                return _resolve_col(row, e)
            except KeyError:
                if outer_row:
                    return _resolve_col(outer_row, e)
                raise
        if isinstance(e, BinaryOp):
            return self._eval_binary(e, row, semantics, catalog, outer_row, cte_bindings)
        if isinstance(e, UnaryOp):
            v = self._eval_expr(e.operand, row, semantics, catalog, outer_row, cte_bindings)
            if e.op == Op.NOT:
                return _three_valued_not(_truthy(v, semantics))
            if e.op == Op.NEG:
                return None if v is None else -v
            raise NotImplementedError(e.op)
        if isinstance(e, FuncCall):
            return self._eval_func(e, row, semantics, catalog, outer_row, cte_bindings)
        if isinstance(e, Case):
            for cond, val in e.branches:
                cv = self._eval_expr(cond, row, semantics, catalog, outer_row, cte_bindings)
                if _truthy(cv, semantics) is True:
                    return self._eval_expr(val, row, semantics, catalog, outer_row, cte_bindings)
            if e.default is not None:
                return self._eval_expr(e.default, row, semantics, catalog, outer_row, cte_bindings)
            return None
        if isinstance(e, Cast):
            v = self._eval_expr(e.operand, row, semantics, catalog, outer_row, cte_bindings)
            return _coerce(v, e.target)
        if isinstance(e, InList):
            v = self._eval_expr(e.operand, row, semantics, catalog, outer_row, cte_bindings)
            for it in e.items:
                iv = self._eval_expr(it, row, semantics, catalog, outer_row, cte_bindings)
                if _eq(v, iv) is True:
                    return True
            return False if v is not None else None
        if isinstance(e, IsNull):
            v = self._eval_expr(e.operand, row, semantics, catalog, outer_row, cte_bindings)
            is_null = v is None
            return (not is_null) if e.negated else is_null
        if isinstance(e, Between):
            v = self._eval_expr(e.operand, row, semantics, catalog, outer_row, cte_bindings)
            lo = self._eval_expr(e.low, row, semantics, catalog, outer_row, cte_bindings)
            hi = self._eval_expr(e.high, row, semantics, catalog, outer_row, cte_bindings)
            in_range = _three_valued_and(_cmp(v, lo, Op.GTE), _cmp(v, hi, Op.LTE))
            return _three_valued_not(in_range) if e.negated else in_range
        if isinstance(e, ScalarSubquery):
            inner = self._exec_plan(e.plan, semantics, catalog, outer_row or row, cte_bindings)
            if not inner:
                return None
            if len(inner) > 1:
                raise ValueError("Scalar subquery returned more than one row")
            first_key = next(iter(inner[0]))
            return inner[0][first_key]
        if isinstance(e, ExistsSubquery):
            inner = self._exec_plan(e.plan, semantics, catalog, outer_row or row, cte_bindings)
            exists = bool(inner)
            return (not exists) if e.negated else exists
        if isinstance(e, InSubquery):
            v = self._eval_expr(e.operand, row, semantics, catalog, outer_row, cte_bindings)
            inner = self._exec_plan(e.plan, semantics, catalog, outer_row or row, cte_bindings)
            for ir_row in inner:
                first_key = next(iter(ir_row))
                if _eq(v, ir_row[first_key]) is True:
                    return False if e.negated else True
            return True if e.negated else False
        raise NotImplementedError(type(e).__name__)

    def _eval_binary(
        self,
        e: BinaryOp,
        row: Row,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
        outer_row: Row,
        cte_bindings: dict[str, Rows],
    ) -> Any:
        a = self._eval_expr(e.left, row, semantics, catalog, outer_row, cte_bindings)
        b = self._eval_expr(e.right, row, semantics, catalog, outer_row, cte_bindings)
        op = e.op
        if op == Op.EQ:
            return _eq(a, b)
        if op == Op.NEQ:
            r = _eq(a, b)
            return _three_valued_not(r)
        if op in (Op.LT, Op.LTE, Op.GT, Op.GTE):
            return _cmp(a, b, op)
        if op == Op.NULL_SAFE_EQ:
            if a is None and b is None:
                return True
            if a is None or b is None:
                return False
            return a == b
        if op == Op.AND:
            return _three_valued_and(_truthy(a, semantics), _truthy(b, semantics))
        if op == Op.OR:
            return _three_valued_or(_truthy(a, semantics), _truthy(b, semantics))
        if op in (Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD):
            return _arith(a, b, op, semantics)
        if op == Op.CONCAT:
            if a is None or b is None:
                return None
            return str(a) + str(b)
        if op == Op.LIKE:
            return _like(a, b, case_sensitive=semantics.like_case_sensitive)
        if op == Op.ILIKE:
            return _like(a, b, case_sensitive=False)
        raise NotImplementedError(op)

    def _eval_func(
        self,
        e: FuncCall,
        row: Row,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
        outer_row: Row,
        cte_bindings: dict[str, Rows],
    ) -> Any:
        name = e.name.upper()
        args = [
            self._eval_expr(a, row, semantics, catalog, outer_row, cte_bindings) for a in e.args
        ]
        if name == "COALESCE":
            for v in args:
                if v is not None:
                    return v
            return None
        if name in ("UPPER", "UCASE"):
            return None if args[0] is None else str(args[0]).upper()
        if name in ("LOWER", "LCASE"):
            return None if args[0] is None else str(args[0]).lower()
        if name in ("LENGTH", "LEN", "CHAR_LENGTH"):
            return None if args[0] is None else len(str(args[0]))
        if name == "CONCAT":
            if any(v is None for v in args):
                return None
            return "".join(str(v) for v in args)
        if name == "ABS":
            return None if args[0] is None else abs(args[0])
        if name == "TRIM":
            return None if args[0] is None else str(args[0]).strip()
        if name in ("SUBSTR", "SUBSTRING"):
            s = args[0]
            if s is None:
                return None
            start = int(args[1]) - 1 if len(args) >= 2 else 0
            length = int(args[2]) if len(args) >= 3 else None
            s = str(s)
            return s[start : start + length] if length is not None else s[start:]
        if name == "REPLACE":
            if any(v is None for v in args):
                return None
            return str(args[0]).replace(str(args[1]), str(args[2]))
        if name in ("IF", "IIF"):
            cond = _truthy(args[0], semantics)
            return args[1] if cond is True else args[2]
        if name == "NULLIF":
            return None if _eq(args[0], args[1]) is True else args[0]
        if name == "GREATEST":
            xs = [v for v in args if v is not None]
            return max(xs) if xs else None
        if name == "LEAST":
            xs = [v for v in args if v is not None]
            return min(xs) if xs else None
        if name == "ROUND":
            if args[0] is None:
                return None
            n = int(args[1]) if len(args) > 1 else 0
            return round(args[0], n)
        if name == "FLOOR":
            import math
            return None if args[0] is None else math.floor(args[0])
        if name in ("CEIL", "CEILING"):
            import math
            return None if args[0] is None else math.ceil(args[0])
        if name == "MOD":
            return None if (args[0] is None or args[1] is None) else args[0] % args[1]
        if name == "DATE_ADD":
            from datetime import timedelta
            return None if (args[0] is None or args[1] is None) else args[0] + timedelta(days=int(args[1]))
        if name == "DATE_SUB":
            from datetime import timedelta
            return None if (args[0] is None or args[1] is None) else args[0] - timedelta(days=int(args[1]))
        if name == "DATE_DIFF":
            unit = args[0] if isinstance(args[0], str) else "day"
            a = args[-2]
            b = args[-1]
            if a is None or b is None:
                return None
            if unit.lower() in ("day", "days"):
                return (b - a).days
            raise NotImplementedError(f"DATE_DIFF unit: {unit}")
        if name in ("DATE_PART", "EXTRACT"):
            part = str(args[0]).lower()
            d = args[1]
            if d is None:
                return None
            return _date_part(part, d)
        raise NotImplementedError(f"FuncCall not implemented (reference): {name}")


# ---- helpers shared with tests ----


class _Reversed:
    """Wrapper to reverse comparisons for DESC sorting in heterogeneous tuples."""

    __slots__ = ("v",)

    def __init__(self, v: Any) -> None:
        self.v = v

    def __lt__(self, other: "_Reversed") -> bool:
        return other.v < self.v

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Reversed) and other.v == self.v

    def __hash__(self) -> int:
        return hash(self.v)


def _make_hashable(v: Any) -> Any:
    """Map non-hashable values (e.g. lists) to hashable forms for dedup."""
    if isinstance(v, list):
        return tuple(_make_hashable(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _make_hashable(val)) for k, val in v.items()))
    return v


def _coerce(v: Any, t: IRType) -> Any:
    if v is None:
        return None
    if t.kind == TypeKind.INT:
        return int(v)
    if t.kind == TypeKind.FLOAT:
        return float(v)
    if t.kind == TypeKind.TEXT:
        return str(v)
    if t.kind == TypeKind.BOOL:
        return bool(v)
    return v


def _date_part(part: str, d: Any) -> Any:
    if part in ("year", "yyyy"):
        return d.year
    if part == "month":
        return d.month
    if part == "day":
        return d.day
    if part == "hour":
        return d.hour
    if part == "minute":
        return d.minute
    if part == "second":
        return d.second
    raise NotImplementedError(f"DATE_PART unit: {part}")


__all__ = ["ReferenceInterpreter"]
