"""Plan executor: takes an IR Plan + SemanticConfig + table catalog -> Polars DataFrame.

One handler per IR node. SemanticConfig is read at every divergence point so the
*same plan* on *different configs* really does produce different results.

For Apply (correlated subqueries) we materialize the outer side and execute the
inner plan once per outer row. That's slow but unambiguous; performance tuning
is deferred.
"""

from __future__ import annotations

from typing import Any, Optional

import polars as pl

from manysql.executor.expr_eval import (
    ExprEvaluator,
    _ir_type_to_polars,
    _outer_key,
    _to_bool,
)
from manysql.ir.expr import (
    AggCall,
    AggKind,
    ColumnRef,
    Expr,
    Literal,
    NullsOrder,
    OrderKey,
    SortDirection,
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
from manysql.spec.semantics import (
    NullOrder,
    SemanticConfig,
    SetOpDefault,
)


class PlanExecutor:
    """Execute an IR plan over an in-memory table catalog.

    A dialect plugs into the executor through three optional modules:

    - ``overrides``: object exposing ``FUNCTIONS`` / ``OPERATORS`` dicts
      consulted by the expression evaluator before it raises
      ``NotImplementedError`` for an unrecognized function/operator.
    - ``passes``: object exposing ``PRE_EXECUTION_PASSES`` — a list of
      ``Plan -> Plan`` rewrites applied between lowering and dispatch,
      in list order. Lets a dialect desugar non-canonical IR markers
      (emitted by its own ``lowering.py``) into canonical IR shapes.
    - ``effects``: object exposing ``EFFECTS`` — a dict of named handlers
      swapped into specific executor decision points (see
      ``manysql/codegen/effects_emit.py`` for the v1 registry).
    """

    def __init__(
        self,
        catalog: dict[str, pl.DataFrame],
        semantics: Optional[SemanticConfig] = None,
        overrides: Optional[Any] = None,
        passes: Optional[Any] = None,
        effects: Optional[Any] = None,
    ) -> None:
        self.catalog = catalog
        self.semantics = semantics or SemanticConfig.reference()
        self.overrides = overrides
        self.passes = passes
        self.effects = effects
        self._cte_bindings: dict[str, pl.DataFrame] = {}

    # -------- public entry --------

    def execute(
        self,
        plan: Plan,
        *,
        outer_row: Optional[dict[str, Any]] = None,
    ) -> pl.DataFrame:
        plan = apply_pre_passes(plan, self.semantics, self.passes)
        return self._dispatch(plan, outer_row or {})

    def evaluator(self, outer_row: Optional[dict[str, Any]] = None) -> ExprEvaluator:
        return ExprEvaluator(
            self.semantics,
            self,
            outer_row=outer_row,
            overrides=self.overrides,
            effects=self.effects,
        )

    # -------- dispatch --------

    def _dispatch(self, p: Plan, outer_row: dict[str, Any]) -> pl.DataFrame:  # noqa: PLR0911
        if isinstance(p, Scan):
            return self._scan(p)
        if isinstance(p, Project):
            return self._project(p, outer_row)
        if isinstance(p, Filter):
            return self._filter(p, outer_row)
        if isinstance(p, Join):
            return self._join(p, outer_row)
        if isinstance(p, Aggregate):
            return self._aggregate(p, outer_row)
        if isinstance(p, Window):
            return self._window(p, outer_row)
        if isinstance(p, Sort):
            return self._sort(p, outer_row)
        if isinstance(p, Limit):
            return self._limit(p, outer_row)
        if isinstance(p, Distinct):
            return self._distinct(p, outer_row)
        if isinstance(p, SetOp):
            return self._setop(p, outer_row)
        if isinstance(p, WithCTE):
            return self._with_cte(p, outer_row)
        if isinstance(p, RecursiveCTE):
            return self._recursive_cte(p, outer_row)
        if isinstance(p, Apply):
            return self._apply(p, outer_row)
        raise NotImplementedError(f"PlanExecutor: unhandled {type(p).__name__}")

    # -------- handlers --------

    def _scan(self, p: Scan) -> pl.DataFrame:
        if p.table_name in self._cte_bindings:
            df = self._cte_bindings[p.table_name]
        elif p.table_name in self.catalog:
            df = self.catalog[p.table_name]
        else:
            raise KeyError(f"Unknown table: {p.table_name}")
        if p.alias is not None:
            # Apply qualifier-prefixed naming for downstream join column resolution.
            df = df.rename({c: f"{p.alias}__{c}" for c in df.columns})
        return df

    def _project(self, p: Project, outer_row: dict[str, Any]) -> pl.DataFrame:
        inp = self._dispatch(p.input, outer_row)
        ev = self.evaluator(outer_row)
        exprs: list[pl.Expr] = []
        for name, e in p.projections:
            exprs.append(ev.eval(e).alias(name))
        return inp.select(exprs)

    def _filter(self, p: Filter, outer_row: dict[str, Any]) -> pl.DataFrame:
        inp = self._dispatch(p.input, outer_row)
        ev = self.evaluator(outer_row)
        pred = _to_bool(ev.eval(p.predicate), self.semantics)
        return inp.filter(pred.fill_null(False))

    def _join(self, p: Join, outer_row: dict[str, Any]) -> pl.DataFrame:
        left = self._dispatch(p.left, outer_row)
        right = self._dispatch(p.right, outer_row)

        if p.kind == JoinKind.CROSS:
            return left.join(right, how="cross")

        if p.using:
            how = {
                JoinKind.INNER: "inner",
                JoinKind.LEFT: "left",
                JoinKind.RIGHT: "right",
                JoinKind.FULL: "full",
                JoinKind.SEMI: "semi",
                JoinKind.ANTI: "anti",
            }[p.kind]
            return left.join(right, on=list(p.using), how=how)

        if p.on is None:
            raise ValueError(f"{p.kind} JOIN requires ON or USING")

        # ON-style join: tag both sides with row indices, cross-join,
        # evaluate the predicate, then fill unmatched rows by index lookup.
        # Using indices avoids Polars' NULL-aware-equality semantics (NULL=NULL
        # is *not* a match in equi-join), which would otherwise misclassify
        # left rows containing NULLs as unmatched after re-joining.
        if p.kind in (JoinKind.SEMI, JoinKind.ANTI):
            return self._semi_anti_join(left, right, p, outer_row)

        left_idx = left.with_row_index("__l_idx")
        right_idx = right.with_row_index("__r_idx")
        cross = left_idx.join(right_idx, how="cross")
        ev = self.evaluator(outer_row)
        pred = _to_bool(ev.eval(p.on), self.semantics).fill_null(False)
        matched = cross.filter(pred)

        if p.kind == JoinKind.INNER:
            return matched.drop(["__l_idx", "__r_idx"])

        matched_left = matched["__l_idx"].to_list()
        matched_right = matched["__r_idx"].to_list()
        out = matched.drop(["__l_idx", "__r_idx"])
        target_cols = list(out.columns)

        if p.kind in (JoinKind.LEFT, JoinKind.FULL):
            unmatched_left = left_idx.filter(
                ~pl.col("__l_idx").is_in(matched_left)
            ).drop("__l_idx")
            null_right = [
                pl.lit(None).alias(c).cast(right.schema[c]) for c in right.columns
            ]
            unmatched_left = unmatched_left.with_columns(null_right).select(target_cols)
            out = pl.concat([out, unmatched_left], how="vertical_relaxed")
        if p.kind in (JoinKind.RIGHT, JoinKind.FULL):
            unmatched_right = right_idx.filter(
                ~pl.col("__r_idx").is_in(matched_right)
            ).drop("__r_idx")
            null_left = [
                pl.lit(None).alias(c).cast(left.schema[c]) for c in left.columns
            ]
            unmatched_right = unmatched_right.with_columns(null_left).select(target_cols)
            out = pl.concat([out, unmatched_right], how="vertical_relaxed")
        return out

    def _semi_anti_join(
        self,
        left: pl.DataFrame,
        right: pl.DataFrame,
        p: Join,
        outer_row: dict[str, Any],
    ) -> pl.DataFrame:
        # Tag left rows with index, cross with right, evaluate predicate, then
        # filter left by membership.
        left_idx = left.with_row_index("__manysql_left_idx")
        cross = left_idx.join(right, how="cross")
        ev = self.evaluator(outer_row)
        pred = _to_bool(ev.eval(p.on), self.semantics).fill_null(False)
        matched_idxs = cross.filter(pred)["__manysql_left_idx"].unique().to_list()
        if p.kind == JoinKind.SEMI:
            kept = left_idx.filter(pl.col("__manysql_left_idx").is_in(matched_idxs))
        else:  # ANTI
            kept = left_idx.filter(~pl.col("__manysql_left_idx").is_in(matched_idxs))
        return kept.drop("__manysql_left_idx")

    def _aggregate(self, p: Aggregate, outer_row: dict[str, Any]) -> pl.DataFrame:
        inp = self._dispatch(p.input, outer_row)
        ev = self.evaluator(outer_row)

        if p.group_by:
            group_exprs = [ev.eval(e).alias(name) for name, e in p.group_by]
            # Polars requires column names; we materialize group exprs first.
            inp_with_groups = inp.with_columns(group_exprs)
            group_names = [name for name, _ in p.group_by]
            agg_exprs = [ev.eval(e).alias(name) for name, e in p.aggregates]
            return inp_with_groups.group_by(group_names, maintain_order=True).agg(agg_exprs)

        # No group by: single-row reduction.
        agg_exprs = [ev.eval(e).alias(name) for name, e in p.aggregates]
        if inp.height == 0:
            # Empty input: emit one row with NULL/0 per the SQL semantics for each agg.
            cols = []
            for (name, e), out_t in zip(p.aggregates, p.output_types, strict=True):
                if isinstance(e, AggCall) and e.kind == AggKind.COUNT_STAR:
                    cols.append(pl.lit(0).alias(name).cast(_ir_type_to_polars(out_t)))
                elif (
                    isinstance(e, AggCall)
                    and e.kind == AggKind.SUM
                    and not self.semantics.sum_of_empty_returns_null
                ):
                    cols.append(pl.lit(0).alias(name).cast(_ir_type_to_polars(out_t)))
                else:
                    cols.append(pl.lit(None).alias(name).cast(_ir_type_to_polars(out_t)))
            return pl.DataFrame().with_columns(cols)
        return inp.select(agg_exprs)

    def _window(self, p: Window, outer_row: dict[str, Any]) -> pl.DataFrame:
        inp = self._dispatch(p.input, outer_row)
        ev = self.evaluator(outer_row)
        new_cols: list[pl.Expr] = []
        for (name, e), _t in zip(p.windows, p.output_types, strict=True):
            assert isinstance(e, WindowCall)
            new_cols.append(self._window_call(e, ev).alias(name))
        return inp.with_columns(new_cols)

    def _window_call(self, w: WindowCall, ev: ExprEvaluator) -> pl.Expr:
        # Build partition / sort key list for `over()` and pre-sort if needed.
        partition = [ev.eval(p) for p in w.partition_by]
        sort_keys = [ev.eval(k.expr) for k in w.order_by]
        sort_dirs = [k.direction == SortDirection.DESC for k in w.order_by]
        # Polars `over()` accepts only a single `descending` bool. For mixed
        # directions, negate the desc keys so a uniform ascending sort produces
        # the right order. (Negation works for numeric / temporal keys.)
        sort_keys, descending_uniform = self._normalize_sort_dirs(sort_keys, sort_dirs)

        kind = w.kind

        if kind == WindowKind.ROW_NUMBER:
            return pl.int_range(pl.len()).over(
                partition_by=partition,
                order_by=sort_keys,
                descending=descending_uniform,
            ) + 1
        if kind in (WindowKind.RANK, WindowKind.DENSE_RANK):
            method = "min" if kind == WindowKind.RANK else "dense"
            rank_key = self._rank_key(w)
            ranked = rank_key.rank(method=method)
            return ranked.over(partition) if partition else ranked

        if kind in (WindowKind.LAG, WindowKind.LEAD):
            arg = ev.eval(w.args[0])
            offset = self._lit_int(w.args[1]) if len(w.args) > 1 else 1
            shift_n = -offset if kind == WindowKind.LEAD else offset
            base = arg.shift(shift_n)
            return base.over(
                partition_by=partition, order_by=sort_keys, descending=descending_uniform
            )
        if kind == WindowKind.FIRST_VALUE:
            arg = ev.eval(w.args[0])
            return arg.first().over(
                partition_by=partition, order_by=sort_keys, descending=descending_uniform
            )
        if kind == WindowKind.LAST_VALUE:
            arg = ev.eval(w.args[0])
            return arg.last().over(
                partition_by=partition, order_by=sort_keys, descending=descending_uniform
            )

        # Aggregate windows over partition (no frame): compute over partition only.
        # For ORDER-BY-aware running aggregates we use cum_* over partition.
        if kind in (WindowKind.SUM, WindowKind.AVG, WindowKind.MIN, WindowKind.MAX, WindowKind.COUNT):
            arg = ev.eval(w.args[0]) if w.args else pl.lit(1)
            running = bool(w.order_by)
            if running:
                return self._running_agg(arg, kind, partition, sort_keys, sort_dirs)
            agg = self._partition_agg(arg, kind)
            return agg.over(partition_by=partition) if partition else agg
        raise NotImplementedError(f"WindowKind: {kind}")

    def _rank_key(self, w: WindowCall) -> pl.Expr:
        """Build a single Polars expr that, when ranked ASC with method=min/dense,
        produces SQL RANK / DENSE_RANK semantics honoring per-key direction and
        the dialect's NULL-ordering preference.

        For each ORDER BY key:
          - encode direction by negating numeric/temporal keys when DESC
          - encode null placement by replacing NULLs with a sentinel that
            sorts to the requested end (NULLS FIRST -> -infinity sentinel;
            NULLS LAST -> +infinity sentinel) so all output values are non-null
        """
        ev = self.evaluator({})
        parts: list[pl.Expr] = []
        for k in w.order_by:
            base = ev.eval(k.expr)
            desc = k.direction == SortDirection.DESC
            nulls_first = self._resolve_nulls_first(k, desc)
            casted = base.cast(pl.Float64)
            if desc:
                casted = -casted
            sentinel = float("-inf") if nulls_first else float("inf")
            parts.append(casted.fill_null(sentinel))
        if len(parts) == 1:
            return parts[0]
        return pl.struct(parts)

    def _resolve_nulls_first(self, k: OrderKey, desc: bool) -> bool:
        if k.nulls == NullsOrder.FIRST:
            return True
        if k.nulls == NullsOrder.LAST:
            return False
        default = (
            self.semantics.null_order_default_desc
            if desc
            else self.semantics.null_order_default_asc
        )
        return default == NullOrder.FIRST

    @staticmethod
    def _sort_key_concat(
        sort_keys: list[pl.Expr], sort_dirs: list[bool]
    ) -> pl.Expr:
        # When ranking on multiple keys, Polars rank() on a single expr is the
        # easy path; we use the first key. Multi-key ranking is approximated by
        # concatenating to a struct.
        if len(sort_keys) == 1:
            k = sort_keys[0]
            return -k.cast(pl.Float64) if sort_dirs[0] else k
        return pl.struct(
            [
                (-k.cast(pl.Float64) if d else k)
                for k, d in zip(sort_keys, sort_dirs, strict=True)
            ]
        )

    @staticmethod
    def _partition_agg(arg: pl.Expr, kind: WindowKind) -> pl.Expr:
        if kind == WindowKind.SUM:
            return arg.sum()
        if kind == WindowKind.AVG:
            return arg.mean()
        if kind == WindowKind.MIN:
            return arg.min()
        if kind == WindowKind.MAX:
            return arg.max()
        if kind == WindowKind.COUNT:
            return arg.drop_nulls().len()
        raise NotImplementedError(kind)

    @classmethod
    def _running_agg(
        cls,
        arg: pl.Expr,
        kind: WindowKind,
        partition: list[pl.Expr],
        sort_keys: list[pl.Expr],
        sort_dirs: list[bool],
    ) -> pl.Expr:
        # SQL aggregates treat NULL as "skipped" (sum of nulls = 0, count of
        # nulls = 0). Polars' cum_* operations leak NULLs at the row position,
        # so we substitute 0 for NULLs before accumulating SUM and use
        # is_not_null + cum_sum for COUNT / AVG denominator.
        if kind == WindowKind.SUM:
            base = arg.fill_null(0).cum_sum()
        elif kind == WindowKind.MIN:
            base = arg.cum_min()
        elif kind == WindowKind.MAX:
            base = arg.cum_max()
        elif kind == WindowKind.COUNT:
            base = arg.is_not_null().cast(pl.Int64).cum_sum()
        elif kind == WindowKind.AVG:
            running_sum = arg.fill_null(0).cum_sum()
            running_count = arg.is_not_null().cast(pl.Int64).cum_sum()
            base = running_sum / running_count
        else:
            raise NotImplementedError(kind)
        sort_keys, descending = cls._normalize_sort_dirs(sort_keys, sort_dirs)
        return base.over(
            partition_by=partition, order_by=sort_keys, descending=descending
        )

    @staticmethod
    def _normalize_sort_dirs(
        sort_keys: list[pl.Expr], sort_dirs: list[bool]
    ) -> tuple[list[pl.Expr], bool]:
        """Polars `over()` only takes a scalar `descending` bool. For mixed
        directions, we encode direction by negating numeric keys so that a
        uniform ascending sort produces the desired order.
        """
        if not sort_dirs:
            return sort_keys, False
        if all(sort_dirs):
            return sort_keys, True
        if not any(sort_dirs):
            return sort_keys, False
        adjusted = [
            (-k.cast(pl.Float64) if d else k)
            for k, d in zip(sort_keys, sort_dirs, strict=True)
        ]
        return adjusted, False

    @staticmethod
    def _lit_int(e: Expr) -> int:
        if isinstance(e, Literal) and isinstance(e.value, int):
            return int(e.value)
        raise ValueError("LAG/LEAD offset must be an integer literal")

    def _sort(self, p: Sort, outer_row: dict[str, Any]) -> pl.DataFrame:
        inp = self._dispatch(p.input, outer_row)
        ev = self.evaluator(outer_row)
        # Materialize sort keys as named columns to avoid Polars expr-on-expr
        # complications, then drop the helper columns.
        sort_aliases: list[str] = []
        with_cols: list[pl.Expr] = []
        for i, k in enumerate(p.keys):
            alias = f"__manysql_sort_{i}"
            with_cols.append(ev.eval(k.expr).alias(alias))
            sort_aliases.append(alias)
        descending = [k.direction == SortDirection.DESC for k in p.keys]
        nulls_last = [self._nulls_last(k) for k in p.keys]
        return (
            inp.with_columns(with_cols)
            .sort(sort_aliases, descending=descending, nulls_last=nulls_last)
            .drop(sort_aliases)
        )

    def _nulls_last(self, k: OrderKey) -> bool:
        if k.nulls == NullsOrder.FIRST:
            return False
        if k.nulls == NullsOrder.LAST:
            return True
        # DEFAULT: honor SemanticConfig per direction.
        default = (
            self.semantics.null_order_default_desc
            if k.direction == SortDirection.DESC
            else self.semantics.null_order_default_asc
        )
        return default == NullOrder.LAST

    def _limit(self, p: Limit, outer_row: dict[str, Any]) -> pl.DataFrame:
        inp = self._dispatch(p.input, outer_row)
        if p.offset:
            inp = inp.slice(p.offset)
        if p.limit is not None:
            inp = inp.head(p.limit)
        return inp

    def _distinct(self, p: Distinct, outer_row: dict[str, Any]) -> pl.DataFrame:
        return self._dispatch(p.input, outer_row).unique(maintain_order=True)

    def _setop(self, p: SetOp, outer_row: dict[str, Any]) -> pl.DataFrame:
        left = self._dispatch(p.left, outer_row)
        right = self._dispatch(p.right, outer_row)
        # SQL set ops match by *position*, not by name. Align right's column
        # names to left's so all subsequent operations (concat, join) use a
        # consistent schema.
        if list(right.columns) != list(left.columns):
            if right.width != left.width:
                raise ValueError(
                    f"set-op operands differ in width: {right.width} vs {left.width}"
                )
            right = right.rename(dict(zip(right.columns, left.columns, strict=True)))

        all_mode = p.all
        if not all_mode and self.semantics.set_op_default == SetOpDefault.ALL:
            all_mode = True

        if p.kind == SetOpKind.UNION:
            combined = pl.concat([left, right], how="vertical_relaxed")
            return combined if all_mode else combined.unique(maintain_order=True)
        if p.kind == SetOpKind.INTERSECT:
            cols = list(left.columns)
            inter = left.join(right, on=cols, how="semi")
            return inter if all_mode else inter.unique(maintain_order=True)
        if p.kind == SetOpKind.EXCEPT:
            cols = list(left.columns)
            diff = left.join(right, on=cols, how="anti")
            return diff if all_mode else diff.unique(maintain_order=True)
        raise NotImplementedError(p.kind)

    def _with_cte(self, p: WithCTE, outer_row: dict[str, Any]) -> pl.DataFrame:
        saved = dict(self._cte_bindings)
        try:
            for binding in p.bindings:
                self._cte_bindings[binding.name] = self._dispatch(binding.plan, outer_row)
            return self._dispatch(p.body, outer_row)
        finally:
            self._cte_bindings = saved

    def _recursive_cte(self, p: RecursiveCTE, outer_row: dict[str, Any]) -> pl.DataFrame:
        saved = dict(self._cte_bindings)
        try:
            seed = self._dispatch(p.seed, outer_row)
            self._cte_bindings[p.name] = seed
            accumulated = seed
            new_rows = seed
            max_iters = 10_000
            iters = 0
            while new_rows.height > 0 and iters < max_iters:
                # Bind name to *new rows from the previous iteration* so the
                # recursive step refers to the most recent batch (textbook semantic
                # for SQL recursive CTEs that allow only one self-reference).
                self._cte_bindings[p.name] = new_rows
                step = self._dispatch(p.recursive, outer_row)
                if step.height == 0:
                    break
                # Subtract anything already accumulated to detect fixpoint
                step_minus = step.join(accumulated, on=list(accumulated.columns), how="anti")
                if step_minus.height == 0:
                    break
                if p.union_all:
                    accumulated = pl.concat([accumulated, step_minus], how="vertical_relaxed")
                else:
                    accumulated = pl.concat(
                        [accumulated, step_minus], how="vertical_relaxed"
                    ).unique(maintain_order=True)
                new_rows = step_minus
                iters += 1
            if iters == max_iters:
                raise RuntimeError(
                    f"RecursiveCTE {p.name!r}: exceeded {max_iters} iterations"
                )
            self._cte_bindings[p.name] = accumulated
            return self._dispatch(p.body, outer_row)
        finally:
            self._cte_bindings = saved

    def _apply(self, p: Apply, outer_row: dict[str, Any]) -> pl.DataFrame:
        outer = self._dispatch(p.outer, outer_row)
        if outer.height == 0:
            # Empty outer: produce empty frame with merged schema.
            return outer
        rows: list[dict[str, Any]] = []
        for row in outer.iter_rows(named=True):
            merged_outer = {**outer_row, **row}
            inner = self._dispatch(p.inner, merged_outer)

            if p.kind == ApplyKind.SCALAR:
                assert p.output_name is not None
                value = inner.row(0)[0] if inner.height > 0 else None
                rows.append({**row, p.output_name: value})
            elif p.kind in (ApplyKind.EXISTS, ApplyKind.NOT_EXISTS):
                assert p.output_name is not None
                exists = inner.height > 0
                value = exists if p.kind == ApplyKind.EXISTS else not exists
                rows.append({**row, p.output_name: value})
            elif p.kind == ApplyKind.IN:
                assert p.output_name is not None
                values = inner.to_series(0).to_list()
                # The 'IN' Apply expects an "operand expression" we don't track here;
                # IN is represented in expression form via InSubquery+correlation. The
                # ApplyKind.IN form is reserved for future use; not emitted by v1 lowerer.
                rows.append({**row, p.output_name: values})
            elif p.kind == ApplyKind.NOT_IN:
                assert p.output_name is not None
                values = inner.to_series(0).to_list()
                rows.append({**row, p.output_name: values})
            elif p.kind == ApplyKind.CROSS:
                if inner.height == 0:
                    continue
                for inner_row in inner.iter_rows(named=True):
                    rows.append({**row, **inner_row})
            elif p.kind == ApplyKind.OUTER:
                if inner.height == 0:
                    null_filled = {k: None for k in inner.columns}
                    rows.append({**row, **null_filled})
                else:
                    for inner_row in inner.iter_rows(named=True):
                        rows.append({**row, **inner_row})
            else:
                raise NotImplementedError(p.kind)

        if not rows:
            # Build empty frame with the apply output schema
            schema = self._apply_schema(p, outer)
            return pl.DataFrame(schema=schema)
        return pl.DataFrame(rows)

    @staticmethod
    def _apply_schema(p: Apply, outer: pl.DataFrame) -> dict[str, pl.DataType]:
        out: dict[str, pl.DataType] = dict(outer.schema)
        if p.kind == ApplyKind.SCALAR:
            assert p.output_name is not None and p.output_type is not None
            out[p.output_name] = _ir_type_to_polars(p.output_type)
        elif p.kind in (ApplyKind.EXISTS, ApplyKind.NOT_EXISTS):
            assert p.output_name is not None
            out[p.output_name] = pl.Boolean
        return out


def execute(
    plan: Plan,
    semantics: SemanticConfig,
    catalog: dict[str, pl.DataFrame],
    overrides: Optional[Any] = None,
    *,
    passes: Optional[Any] = None,
    effects: Optional[Any] = None,
) -> pl.DataFrame:
    """Convenience wrapper: build PlanExecutor and run.

    `overrides`, `passes`, and `effects` are forwarded to `PlanExecutor`
    so dialect-specific function/operator bodies, plan rewrites, and
    runtime decision-point handlers are all reachable from a single
    entry point.
    """
    return PlanExecutor(
        catalog,
        semantics,
        overrides=overrides,
        passes=passes,
        effects=effects,
    ).execute(plan)


def apply_pre_passes(
    plan: Plan,
    semantics: SemanticConfig,
    passes_module: Optional[Any],
) -> Plan:
    """Apply a dialect's pre-execution passes to ``plan``.

    A dialect's ``passes.py`` module exposes a ``PRE_EXECUTION_PASSES``
    list of ``Callable[[Plan, SemanticConfig], Plan]``. They run in list
    order between lowering and dispatch. Each pass must return a Plan;
    the empty list (or a missing module) is a no-op.

    A pass is *not* allowed to return ``None`` — that is treated as a
    bug and surfaces as ``RuntimeError`` so dialect authors fail fast
    rather than silently dropping the plan.
    """
    if passes_module is None:
        return plan
    seq = getattr(passes_module, "PRE_EXECUTION_PASSES", None)
    if not seq:
        return plan
    for pass_fn in seq:
        new_plan = pass_fn(plan, semantics)
        if new_plan is None:
            raise RuntimeError(
                f"pre-execution pass {pass_fn!r} returned None; "
                "passes must return a Plan"
            )
        plan = new_plan
    return plan
