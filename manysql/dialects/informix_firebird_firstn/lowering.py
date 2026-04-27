"""Generated lowering for the informix_firebird_firstn dialect: Lark parse tree -> manysql IR.

Public entrypoint:
    lower(tree: Tree, config: SemanticConfig, catalog: dict[str, tuple[ColumnSchema,...]]) -> Plan

The catalog tells the lowerer table schemas so it can resolve column references
and infer types. It must be passed in by the caller; the dialect package itself
is data-agnostic.

This lowering is hand-written for the *reference* dialect (near-ANSI SQL).
Generated dialects will have their own lowering modules tailored to their
surface, but they target the same IR.

Aggregate handling: when the SELECT or HAVING contains aggregate calls (or a
GROUP BY is present), we extract the aggregates as named slots, wrap the input
in an Aggregate plan node, and rewrite the SELECT/HAVING expressions to refer
to those slots by name.

Correlated subqueries are deliberately not supported in v1 of this hand-written
lowerer. Uncorrelated EXISTS / IN / scalar subqueries lower to expression-form
subquery nodes and execute via the executor's subquery handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

from lark import Token, Tree

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
    ColumnSchema,
    CTEBinding,
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
from manysql.ir.types import BOOL, DATE_T, FLOAT, INT, IRType, TEXT, TypeKind
from manysql.spec.semantics import SemanticConfig


# -------- catalog and scope --------


Catalog = dict[str, tuple[ColumnSchema, ...]]


@dataclass
class Scope:
    """Tracks visible columns and their effective IR-level qualifier.

    The *effective qualifier* is the string used in `ColumnRef.qualifier` and
    matches the `Scan.alias`. If a table is referenced without `AS`, the
    effective qualifier is `None` (the Scan won't rename columns), but the
    table name is still usable as a *binding name* for qualified references
    (e.g. `employees.active` resolves to the bare `active` column with
    qualifier=None).
    """

    by_name: dict[str, list[tuple[Optional[str], IRType]]] = field(default_factory=dict)
    cte_columns: dict[str, tuple[ColumnSchema, ...]] = field(default_factory=dict)

    def add(
        self,
        binding: Optional[str],
        cols: Sequence[ColumnSchema],
        *,
        effective_qualifier: Optional[str] = None,
    ) -> None:
        """`binding` is the name by which this source can be qualified-referenced;
        `effective_qualifier` is the IR-level qualifier (None if no AS alias).
        """
        for c in cols:
            self.by_name.setdefault(c.name, []).append((effective_qualifier, c.type))
            if binding is not None:
                key = f"{binding}.{c.name}"
                self.by_name.setdefault(key, []).append((effective_qualifier, c.type))

    def resolve(self, name: str, qualifier: Optional[str]) -> tuple[Optional[str], IRType]:
        if qualifier is not None:
            key = f"{qualifier}.{name}"
            if key in self.by_name:
                entries = self.by_name[key]
                if len(entries) > 1:
                    raise SemanticError(f"ambiguous column: {key}")
                return entries[0]
            raise SemanticError(f"unknown column: {key}")
        if name not in self.by_name:
            raise SemanticError(f"unknown column: {name}")
        bare = self.by_name[name]
        if len(bare) == 1:
            return bare[0]
        # Multiple matches: ambiguous unless all share the same (qualifier, type).
        if len({(q, t) for q, t in bare}) == 1:
            return bare[0]
        raise SemanticError(f"ambiguous unqualified column: {name}")


class SemanticError(Exception):
    pass


# -------- entry --------


def lower(tree: Tree, config: SemanticConfig, catalog: Catalog) -> Plan:
    """Public entrypoint required by manysql.dialects.registry.DialectEngine."""
    lowerer = Lowerer(catalog=catalog, config=config)
    statement = tree.children[0]
    assert _data(statement) == "statement"
    return lowerer.lower_statement(statement)


# -------- helpers --------


def _data(node: object) -> str:
    return node.data if isinstance(node, Tree) else ""


def _children(node: Tree, name: Optional[str] = None) -> list[Tree | Token]:
    if name is None:
        return list(node.children)
    return [c for c in node.children if isinstance(c, Tree) and c.data == name]


def _first_child(node: Tree, name: str) -> Optional[Tree]:
    cs = _children(node, name)
    return cs[0] if cs else None


def _identifier(token: Token) -> str:
    """Return the bare identifier text, stripping any quoting characters.

    This dialect uses double quotes for both identifiers and strings,
    so we strip double quotes, backticks, and brackets uniformly.
    """
    text = str(token)
    if len(text) >= 2:
        first, last = text[0], text[-1]
        if first == '"' and last == '"':
            return text[1:-1]
        if first == "`" and last == "`":
            return text[1:-1]
        if first == "[" and last == "]":
            return text[1:-1]
    return text


def _has_token(node: Tree, value: str) -> bool:
    """Case-insensitive search for a Token with the given uppercase value."""
    needle = value.upper()
    for c in node.children:
        if isinstance(c, Token) and str(c).upper() == needle:
            return True
    return False


# -------- lowerer --------


class Lowerer:
    def __init__(self, catalog: Catalog, config: SemanticConfig) -> None:
        self.catalog = catalog
        self.config = config
        self._cte_bindings: dict[str, tuple[ColumnSchema, ...]] = {}
        # Build the inverse map: surface alias -> canonical IR name. The
        # ``function_aliases`` knob is keyed by canonical name with a list
        # of accepted surface spellings; we want O(1) lookup the other way.
        self._function_alias_inverse: dict[str, str] = {}
        for canonical, aliases in (
            getattr(config, "function_aliases", {}) or {}
        ).items():
            for alias in aliases:
                self._function_alias_inverse[alias.upper()] = canonical.upper()

    def _canonicalize_function_name(self, name: str) -> str:
        """Map a surface-form function name back to its canonical IR name."""
        return self._function_alias_inverse.get(name, name)

    def lower_statement(self, node: Tree) -> Plan:
        # statement: with_clause? query_expr
        with_node = _first_child(node, "with_clause")
        query_node = _first_child(node, "query_expr")
        assert query_node is not None
        if with_node is None:
            return self.lower_query_expr(query_node)

        # Detect RECURSIVE — `!with_clause` keeps tokens.
        recursive = _has_token(with_node, "RECURSIVE")
        cte_list = _first_child(with_node, "cte_list")
        assert cte_list is not None
        ctes = _children(cte_list, "cte")

        if recursive:
            if len(ctes) != 1:
                raise SemanticError("WITH RECURSIVE supports a single binding in the reference dialect")
            cte = ctes[0]
            name = _identifier(cte.children[0])  # type: ignore[arg-type]
            inner = cte.children[1]
            if _data(inner) != "query_expr":
                raise SemanticError("recursive CTE body must be a query_expr")
            seed_plan, recursive_plan, union_all = self._split_recursive(inner, name)
            body_plan = self.lower_query_expr(query_node)
            return RecursiveCTE(
                name=name,
                seed=seed_plan,
                recursive=recursive_plan,
                body=body_plan,
                union_all=union_all,
            )

        bindings: list[CTEBinding] = []
        for cte in ctes:
            name = _identifier(cte.children[0])  # type: ignore[arg-type]
            inner = cte.children[1]
            assert isinstance(inner, Tree) and inner.data == "query_expr"
            cte_plan = self.lower_query_expr(inner)
            bindings.append(CTEBinding(name=name, plan=cte_plan))
            self._cte_bindings[name] = tuple(cte_plan.schema())
        body_plan = self.lower_query_expr(query_node)
        return WithCTE(bindings=tuple(bindings), body=body_plan)

    def _split_recursive(
        self, query_node: Tree, name: str
    ) -> tuple[Plan, Plan, bool]:
        """A recursive CTE expects: seed UNION [ALL] recursive_step"""
        cores = _children(query_node, "select_core")
        branches = _children(query_node, "set_op_branch")
        if len(cores) != 1 or len(branches) != 1:
            raise SemanticError("recursive CTE must be exactly SEED UNION [ALL] STEP")
        op_node = branches[0].children[0]
        right_core = branches[0].children[1]
        assert isinstance(op_node, Tree) and op_node.data == "union"
        assert isinstance(right_core, Tree) and right_core.data == "select_core"
        union_all = _has_token(op_node, "ALL")
        seed_plan = self.lower_select_core(cores[0])
        self._cte_bindings[name] = tuple(seed_plan.schema())
        rec_plan = self.lower_select_core(right_core)
        return seed_plan, rec_plan, union_all

    def lower_query_expr(self, node: Tree) -> Plan:
        # query_expr: select_core (set_op_branch)* (order_by)? (limit_clause)?
        cores: list[Tree] = _children(node, "select_core")
        branches: list[Tree] = _children(node, "set_op_branch")
        order_node = _first_child(node, "order_by")
        limit_node = _first_child(node, "limit_clause")

        if not branches:
            body, projections, proj_types, distinct, body_scope = (
                self.lower_select_core_open(cores[0])
            )
            if order_node is not None:
                body = self._lower_order_by_with_scope(
                    body, order_node, body_scope, projections
                )
            plan: Plan = Project(
                input=body, projections=projections, output_types=proj_types
            )
            if distinct:
                plan = Distinct(input=plan)
        else:
            plan = self._fold_set_op_branches(cores[0], branches)
            if order_node is not None:
                plan = self._lower_order_by(plan, order_node)

        if limit_node is not None:
            plan = self._lower_limit(plan, limit_node)
        return plan

    def _fold_set_op_branches(
        self, first_core: Tree, branches: list[Tree]
    ) -> Plan:
        from manysql.spec.semantics import SetOpPrecedenceMode

        ops: list[tuple[SetOpKind, bool, Tree]] = []
        for branch in branches:
            op_node = branch.children[0]
            right_core = branch.children[1]
            assert isinstance(op_node, Tree)
            assert isinstance(right_core, Tree) and right_core.data == "select_core"
            kind = {
                "union": SetOpKind.UNION,
                "intersect": SetOpKind.INTERSECT,
                "except_": SetOpKind.EXCEPT,
            }[op_node.data]
            all_mode = _has_token(op_node, "ALL")
            ops.append((kind, all_mode, right_core))

        precedence = getattr(
            self.config, "set_op_precedence", SetOpPrecedenceMode.ANSI
        )
        if precedence == SetOpPrecedenceMode.EXCEPT_INTERSECT_TIGHTER:
            return self._fold_with_tighter_inner(first_core, ops)
        return self._fold_left_to_right(first_core, ops)

    def _fold_left_to_right(
        self,
        first_core: Tree,
        ops: list[tuple[SetOpKind, bool, Tree]],
    ) -> Plan:
        plan: Plan = self.lower_select_core(first_core)
        for kind, all_mode, right_core in ops:
            right = self.lower_select_core(right_core)
            plan = SetOp(kind=kind, left=plan, right=right, all=all_mode)
        return plan

    def _fold_with_tighter_inner(
        self,
        first_core: Tree,
        ops: list[tuple[SetOpKind, bool, Tree]],
    ) -> Plan:
        segments: list[Plan] = []
        union_alls: list[bool] = []
        current: Plan = self.lower_select_core(first_core)
        for kind, all_mode, right_core in ops:
            right = self.lower_select_core(right_core)
            if kind == SetOpKind.UNION:
                segments.append(current)
                union_alls.append(all_mode)
                current = right
            else:
                current = SetOp(
                    kind=kind, left=current, right=right, all=all_mode
                )
        segments.append(current)
        plan = segments[0]
        for i, all_mode in enumerate(union_alls):
            plan = SetOp(
                kind=SetOpKind.UNION,
                left=plan,
                right=segments[i + 1],
                all=all_mode,
            )
        return plan

    def lower_select_core(self, node: Tree) -> Plan:
        body, projections, proj_types, distinct, _scope = self.lower_select_core_open(node)
        plan: Plan = Project(input=body, projections=projections, output_types=proj_types)
        if distinct:
            plan = Distinct(input=plan)
        return plan

    def lower_select_core_open(  # noqa: PLR0912
        self, node: Tree
    ) -> tuple[Plan, tuple[tuple[str, Expr], ...], tuple[IRType, ...], bool, "Scope"]:
        """Lower a SELECT core without the final Project/Distinct wrap.

        Returns (body_plan, projections, projection_types, distinct, body_scope).
        """
        # Detect UNIQUE (this dialect's DISTINCT keyword) or standard DISTINCT
        distinct = any(
            isinstance(c, Tree) and c.data in ("select_distinct", "select_unique")
            for c in node.children
        )
        # Also check for bare UNIQUE/DISTINCT tokens
        if not distinct:
            for c in node.children:
                if isinstance(c, Token) and str(c).upper() in ("DISTINCT", "UNIQUE"):
                    distinct = True
                    break

        select_list = _first_child(node, "select_list")
        from_clause = _first_child(node, "from_clause")
        where_node = _first_child(node, "where_clause")
        group_node = _first_child(node, "group_by_clause")
        having_node = _first_child(node, "having_clause")
        where_expr = where_node.children[0] if where_node else None
        having_expr = having_node.children[0] if having_node else None
        if group_node is not None:
            group_node = _first_child(group_node, "expr_list")
        assert select_list is not None and from_clause is not None

        plan, scope = self.lower_from(from_clause)

        if where_expr is not None:
            elower = ExprLowerer(self.config, scope, self)
            pred = elower.lower(where_expr)
            plan = Filter(input=plan, predicate=pred)

        select_items = _children(select_list, "select_expr") + _children(select_list, "star") + _children(select_list, "qualified_star")
        select_exprs: list[tuple[str, Expr, Tree]] = []
        elower = ExprLowerer(self.config, scope, self)
        elower_collect_aggs = True

        elower.collecting_aggs = True
        agg_slots: list[tuple[str, AggCall]] = elower.agg_slots

        expanded_select: list[tuple[str, Expr]] = []
        for item in _children(select_list):
            if not isinstance(item, Tree):
                continue
            if item.data == "star":
                expanded_select.extend(self._expand_star(scope))
            elif item.data == "qualified_star":
                qual = _identifier(item.children[0])  # type: ignore[arg-type]
                expanded_select.extend(self._expand_qualified_star(scope, qual))
            elif item.data == "select_expr":
                e_node = item.children[0]
                alias_token = item.children[1] if len(item.children) > 1 else None
                expr = elower.lower(e_node)
                alias = (
                    _identifier(alias_token)  # type: ignore[arg-type]
                    if alias_token is not None
                    else _default_alias(e_node)
                )
                expanded_select.append((alias, expr))

        having_lowered: Optional[Expr] = None
        if having_expr is not None:
            elower2 = ExprLowerer(self.config, scope, self, agg_slots=agg_slots)
            elower2.collecting_aggs = True
            having_lowered = elower2.lower(having_expr)

        group_by_exprs: list[tuple[str, Expr]] = []
        if group_node is not None:
            elower3 = ExprLowerer(self.config, scope, self)
            alias_map = (
                {a: e for a, e in expanded_select}
                if self.config.group_by_accepts_select_aliases
                else {}
            )
            for ge in _children(group_node, "expr") or list(group_node.children):
                if isinstance(ge, Tree):
                    bare_alias = _bare_column_ref_name(ge)
                    if (
                        bare_alias is not None
                        and bare_alias in alias_map
                        and bare_alias not in scope.by_name
                    ):
                        expr = alias_map[bare_alias]
                        name = bare_alias
                    else:
                        try:
                            expr = elower3.lower(ge)
                        except SemanticError:
                            if bare_alias is not None and bare_alias in alias_map:
                                expr = alias_map[bare_alias]
                            else:
                                raise
                        name = _default_alias(ge)
                    group_by_exprs.append((name, expr))

        needs_aggregate = bool(group_by_exprs) or bool(agg_slots)
        if needs_aggregate:
            agg_seen: dict[str, AggCall] = {}
            for slot_name, slot_expr in agg_slots:
                agg_seen.setdefault(slot_name, slot_expr)
            agg_list = tuple(agg_seen.items())
            output_types = tuple(
                [
                    self._infer_expr_type(e, scope) for _, e in group_by_exprs
                ]
                + [self._infer_agg_type(a) for _, a in agg_list]
            )
            plan = Aggregate(
                input=plan,
                group_by=tuple(group_by_exprs),
                aggregates=agg_list,
                output_types=output_types,
            )
            scope = self._scope_after_aggregate(group_by_exprs, agg_list, output_types)

            if having_lowered is not None:
                plan = Filter(input=plan, predicate=having_lowered)

        if elower.window_slots:
            window_seen: dict[str, WindowCall] = {}
            for slot_name, slot_call in elower.window_slots:
                window_seen.setdefault(slot_name, slot_call)
            window_list = tuple(window_seen.items())
            window_types = tuple(_infer_window_type(c) for _, c in window_list)
            plan = Window(input=plan, windows=window_list, output_types=window_types)
            for (slot_name, slot_call), wt in zip(window_list, window_types, strict=True):
                scope.add(None, [ColumnSchema(name=slot_name, type=wt)])

        if needs_aggregate and self.config.select_resolves_through_group_by:
            gb_substitution: dict[Expr, Expr] = {}
            for gb_name, gb_expr in group_by_exprs:
                if isinstance(gb_expr, ColumnRef) and gb_expr.name == gb_name:
                    continue
                gb_substitution[gb_expr] = ColumnRef(name=gb_name, qualifier=None)
            if gb_substitution:
                expanded_select = [
                    (n, gb_substitution.get(e, e)) for n, e in expanded_select
                ]

        final_projections: list[tuple[str, Expr]] = []
        proj_types: list[IRType] = []
        for name, expr in expanded_select:
            final_projections.append((name, expr))
            proj_types.append(self._infer_expr_type(expr, scope))

        return plan, tuple(final_projections), tuple(proj_types), distinct, scope

    # ---- FROM/JOIN lowering ----

    def lower_from(self, node: Tree) -> tuple[Plan, Scope]:
        children = [c for c in node.children if isinstance(c, Tree)]
        first = children[0]
        plan, scope = self._lower_table_ref(first)
        for c in children[1:]:
            if c.data == "join_clause":
                plan, scope = self._lower_join_clause(plan, scope, c)
        return plan, scope

    def _lower_table_ref(self, node: Tree) -> tuple[Plan, Scope]:
        if node.data == "table_ident":
            ids = [_identifier(t) for t in node.children if isinstance(t, Token)]
            name = ids[0]
            alias = ids[1] if len(ids) > 1 else None
            cols = self._catalog_or_cte(name)
            scope = Scope()
            binding = alias if alias is not None else name
            scope.add(binding, cols, effective_qualifier=alias)
            return Scan(table_name=name, columns=cols, alias=alias), scope
        if node.data == "table_subquery":
            inner = node.children[0]
            assert isinstance(inner, Tree) and inner.data == "query_expr"
            sub_plan = self.lower_query_expr(inner)
            alias_token = next((c for c in node.children if isinstance(c, Token)), None)
            alias = _identifier(alias_token) if alias_token is not None else None
            scope = Scope()
            scope.add(alias, sub_plan.schema(), effective_qualifier=None)
            return sub_plan, scope
        raise SemanticError(f"bad table_ref: {node.data}")

    def _lower_join_clause(
        self, left_plan: Plan, left_scope: Scope, node: Tree
    ) -> tuple[Plan, Scope]:
        kind_node = next(c for c in node.children if isinstance(c, Tree) and c.data.endswith("_join"))
        right_node = next(
            c for c in node.children
            if isinstance(c, Tree) and c.data in ("table_ident", "table_subquery")
        )
        cond_node = next(
            (c for c in node.children if isinstance(c, Tree) and c.data in ("join_on", "join_using")),
            None,
        )
        right_plan, right_scope = self._lower_table_ref(right_node)

        merged = Scope()
        merged.by_name = {**left_scope.by_name}
        for k, v in right_scope.by_name.items():
            merged.by_name.setdefault(k, []).extend(v)

        kind = {
            "inner_join": JoinKind.INNER,
            "left_join": JoinKind.LEFT,
            "right_join": JoinKind.RIGHT,
            "full_join": JoinKind.FULL,
            "cross_join": JoinKind.CROSS,
        }[kind_node.data]

        on_expr: Optional[Expr] = None
        using_cols: tuple[str, ...] = ()
        if cond_node is not None:
            if cond_node.data == "join_on":
                elower = ExprLowerer(self.config, merged, self)
                on_expr = elower.lower(cond_node.children[0])
            elif cond_node.data == "join_using":
                col_list = cond_node.children[0]
                using_cols = tuple(
                    _identifier(t) for t in col_list.children if isinstance(t, Token)
                )

        return (
            Join(
                left=left_plan,
                right=right_plan,
                kind=kind,
                on=on_expr,
                using=using_cols,
            ),
            merged,
        )

    # ---- ORDER BY / LIMIT ----

    def _lower_order_by(self, plan: Plan, node: Tree) -> Plan:
        scope = Scope()
        for c in plan.schema():
            scope.add(None, [c])
        return self._build_sort(plan, node, scope, projections=None)

    def _lower_order_by_with_scope(
        self,
        body: Plan,
        node: Tree,
        body_scope: Scope,
        projections: tuple[tuple[str, Expr], ...],
    ) -> Plan:
        return self._build_sort(body, node, body_scope, projections=projections)

    def _build_sort(
        self,
        plan: Plan,
        node: Tree,
        scope: Scope,
        *,
        projections: Optional[tuple[tuple[str, Expr], ...]],
    ) -> Plan:
        elower = ExprLowerer(self.config, scope, self)
        keys: list[OrderKey] = []
        alias_map = {n: e for n, e in (projections or ())}
        for k in _children(node, "order_key"):
            expr_node = k.children[0]
            if (
                isinstance(expr_node, Tree)
                and expr_node.data == "column_ref"
                and len(expr_node.children) == 1
            ):
                ident = _identifier(expr_node.children[0])  # type: ignore[arg-type]
                if ident in alias_map and ident not in scope.by_name:
                    expr = alias_map[ident]
                else:
                    try:
                        expr = elower.lower(expr_node)
                    except SemanticError:
                        if ident in alias_map:
                            expr = alias_map[ident]
                        else:
                            raise
            else:
                expr = elower.lower(expr_node)
            direction = SortDirection.ASC
            nulls = NullsOrder.DEFAULT
            for t in k.children[1:]:
                if isinstance(t, Token):
                    val = str(t).upper()
                    if val == "DESC":
                        direction = SortDirection.DESC
                    elif val == "ASC":
                        direction = SortDirection.ASC
                    elif val == "FIRST":
                        nulls = NullsOrder.FIRST
                    elif val == "LAST":
                        nulls = NullsOrder.LAST
            keys.append(OrderKey(expr=expr, direction=direction, nulls=nulls))
        return Sort(input=plan, keys=tuple(keys))

    def _lower_limit(self, plan: Plan, node: Tree) -> Plan:
        """Lower a limit clause.

        This dialect uses ``FIRST n`` (head_n syntax) placed right after SELECT,
        but the grammar may also produce a trailing limit_clause. We also handle
        the pipe-style ``| HEAD n`` that the battery emits. In all cases we
        extract the integer and produce a Limit node.
        """
        ints = [int(t) for t in node.children if isinstance(t, Token) and str(t).isdigit()]
        if not ints:
            # Try to find a number_literal child
            for c in node.children:
                if isinstance(c, Tree) and c.data == "number_literal":
                    ints.append(int(str(c.children[0])))
        limit_val = ints[0] if ints else 0
        return Limit(input=plan, limit=limit_val, offset=0)

    # ---- helpers ----

    def _catalog_or_cte(self, name: str) -> tuple[ColumnSchema, ...]:
        if name in self._cte_bindings:
            return self._cte_bindings[name]
        if name in self.catalog:
            return self.catalog[name]
        raise SemanticError(f"unknown table: {name}")

    def _expand_star(self, scope: Scope) -> list[tuple[str, Expr]]:
        out: list[tuple[str, Expr]] = []
        seen: set[str] = set()
        for name, entries in scope.by_name.items():
            if "." in name:
                continue
            if name in seen:
                continue
            qual = entries[0][0]
            out.append((name, ColumnRef(name=name, qualifier=qual)))
            seen.add(name)
        return out

    def _expand_qualified_star(self, scope: Scope, qual: str) -> list[tuple[str, Expr]]:
        out: list[tuple[str, Expr]] = []
        for name, entries in scope.by_name.items():
            if "." in name:
                continue
            for q, _t in entries:
                if q == qual:
                    out.append((name, ColumnRef(name=name, qualifier=qual)))
                    break
        return out

    def _scope_after_aggregate(
        self,
        group_by: list[tuple[str, Expr]],
        aggs: tuple[tuple[str, AggCall], ...],
        output_types: tuple[IRType, ...],
    ) -> Scope:
        scope = Scope()
        types = list(output_types)
        for (name, _), t in zip(group_by, types[: len(group_by)], strict=True):
            scope.add(None, [ColumnSchema(name=name, type=t)])
        for (name, _), t in zip(aggs, types[len(group_by) :], strict=True):
            scope.add(None, [ColumnSchema(name=name, type=t)])
        return scope

    def _infer_expr_type(self, e: Expr, scope: Scope) -> IRType:
        try:
            return _infer(e, scope, self.catalog, self._cte_bindings)
        except Exception:
            return INT

    def _infer_agg_type(self, agg: AggCall) -> IRType:
        if agg.kind == AggKind.COUNT_STAR:
            return INT
        if agg.kind == AggKind.COUNT:
            return INT
        if agg.kind in (AggKind.SUM, AggKind.AVG):
            return FLOAT
        return FLOAT


# -------- expression lowering --------


def _default_alias(e_node: Tree | Token) -> str:
    if isinstance(e_node, Tree) and e_node.data == "column_ref":
        ids = [_identifier(t) for t in e_node.children if isinstance(t, Token)]
        return ids[-1]
    return "_col"


def _bare_column_ref_name(node: Tree | Token) -> Optional[str]:
    if isinstance(node, Tree) and node.data == "column_ref" and len(node.children) == 1:
        return _identifier(node.children[0])  # type: ignore[arg-type]
    return None


def _strip_string_quotes(text: str) -> str:
    """Strip surrounding quotes from a string literal.

    This dialect uses double quotes for string literals (string_quote: double).
    We also handle single quotes for compatibility.
    """
    if len(text) >= 2:
        first, last = text[0], text[-1]
        if first == '"' and last == '"':
            inner = text[1:-1]
            return inner.replace('""', '"').replace("''", "'")
        if first == "'" and last == "'":
            inner = text[1:-1]
            return inner.replace("''", "'")
    return text


@dataclass
class ExprLowerer:
    config: SemanticConfig
    scope: Scope
    lowerer: Lowerer
    agg_slots: list[tuple[str, AggCall]] = field(default_factory=list)
    window_slots: list[tuple[str, WindowCall]] = field(default_factory=list)
    collecting_aggs: bool = False

    def lower(self, node: Tree | Token) -> Expr:  # noqa: PLR0911, PLR0912
        if isinstance(node, Token):
            raise SemanticError(f"unexpected raw token in expr: {node!r}")

        d = node.data
        if d == "literal":
            return self._literal(node)
        if d in ("number_literal", "string_literal", "true_literal", "false_literal", "null_literal", "date_literal"):
            return self._literal(node)
        if d == "column_ref":
            return self._column_ref(node)
        if d == "paren_expr":
            return self.lower(node.children[0])
        if d == "scalar_subquery":
            inner = node.children[0]
            assert isinstance(inner, Tree)
            sub = self.lowerer.lower_query_expr(inner)
            return ScalarSubquery(plan=sub)
        if d == "exists_expr":
            sub = self.lowerer.lower_query_expr(node.children[0])  # type: ignore[arg-type]
            return ExistsSubquery(plan=sub, negated=False)
        if d == "not_exists_expr":
            sub = self.lowerer.lower_query_expr(node.children[0])  # type: ignore[arg-type]
            return ExistsSubquery(plan=sub, negated=True)
        if d == "comparison_n":
            return self._comparison(node)
        if d == "or_expr_n":
            return self._left_assoc(node, Op.OR)
        if d == "and_expr_n":
            return self._left_assoc(node, Op.AND)
        if d == "not_op":
            return UnaryOp(Op.NOT, self.lower(node.children[0]))
        if d == "additive_n":
            return self._additive(node)
        if d == "multiplicative_n":
            return self._multiplicative(node)
        if d == "neg":
            return UnaryOp(Op.NEG, self.lower(node.children[0]))
        if d == "pos":
            return self.lower(node.children[0])
        if d == "function_call":
            return self._function_call(node)
        if d == "case_expr":
            return self._case(node)
        if d == "cast_expr":
            inner = self.lower(node.children[0])
            type_node = node.children[1]
            assert isinstance(type_node, Tree)
            return Cast(operand=inner, target=_type_name_to_ir(type_node))
        # Fallback: single-child wrappers from the grammar
        if len(node.children) == 1:
            child = node.children[0]
            if isinstance(child, Tree):
                return self.lower(child)
        raise SemanticError(f"unhandled expr rule: {d}")

    # ---- expression helpers ----

    def _literal(self, node: Tree) -> Literal:
        kind = node.data
        if kind == "literal":
            child = node.children[0]
            assert isinstance(child, Tree)
            return self._literal(child)
        if kind == "number_literal":
            tok = node.children[0]
            text = str(tok)
            if "." in text or "e" in text.lower():
                return Literal(float(text), FLOAT)
            return Literal(int(text), INT)
        if kind == "string_literal":
            text = str(node.children[0])
            return Literal(_strip_string_quotes(text), TEXT)
        if kind == "true_literal":
            return Literal(True, BOOL)
        if kind == "false_literal":
            return Literal(False, BOOL)
        if kind == "null_literal":
            return Literal(None, IRType(TypeKind.NULL))
        if kind == "date_literal":
            from datetime import date as _date
            text = str(node.children[0])
            text = _strip_string_quotes(text)
            return Literal(_date.fromisoformat(text), DATE_T)
        raise SemanticError(f"bad literal: {kind}")

    def _column_ref(self, node: Tree) -> ColumnRef:
        ids = [_identifier(t) for t in node.children if isinstance(t, Token)]
        if len(ids) == 1:
            qual, _t = self.scope.resolve(ids[0], None)
            return ColumnRef(name=ids[0], qualifier=qual)
        if len(ids) == 2:
            qual, _t = self.scope.resolve(ids[1], ids[0])
            return ColumnRef(name=ids[1], qualifier=qual)
        raise SemanticError(f"bad column_ref: {ids}")

    def _comparison(self, node: Tree) -> Expr:
        children = list(node.children)
        if len(children) == 1:
            return self.lower(children[0])
        result: Expr = self.lower(children[0])
        for tail in children[1:]:
            assert isinstance(tail, Tree)
            result = self._apply_comp_tail(result, tail)
        return result

    def _apply_comp_tail(self, lhs: Expr, tail: Tree) -> Expr:
        d = tail.data
        if d == "comp_pair":
            op_node = tail.children[0]
            rhs = self.lower(tail.children[1])
            assert isinstance(op_node, Tree)
            op = {
                "eq": Op.EQ,
                "neq": Op.NEQ,
                "lt": Op.LT,
                "lte": Op.LTE,
                "gt": Op.GT,
                "gte": Op.GTE,
            }[op_node.data]
            return BinaryOp(op, lhs, rhs)
        if d == "is_null":
            return IsNull(operand=lhs, negated=False)
        if d == "is_not_null":
            return IsNull(operand=lhs, negated=True)
        if d in ("between_op", "not_between_op"):
            low = self.lower(tail.children[0])
            high = self.lower(tail.children[1])
            return Between(operand=lhs, low=low, high=high, negated=(d == "not_between_op"))
        if d in ("in_list", "not_in_list"):
            items_node = tail.children[0]
            items = [self.lower(c) for c in items_node.children if isinstance(c, Tree)]
            base = InList(operand=lhs, items=tuple(items))
            return UnaryOp(Op.NOT, base) if d == "not_in_list" else base
        if d in ("in_subquery_op", "not_in_subquery_op"):
            sub = self.lowerer.lower_query_expr(tail.children[0])  # type: ignore[arg-type]
            return InSubquery(operand=lhs, plan=sub, negated=(d == "not_in_subquery_op"))
        if d == "like_op":
            rhs = self.lower(tail.children[0])
            return BinaryOp(Op.LIKE, lhs, rhs)
        if d == "ilike_op":
            rhs = self.lower(tail.children[0])
            return BinaryOp(Op.ILIKE, lhs, rhs)
        if d == "distinct_from":
            rhs = self.lower(tail.children[-1])
            return UnaryOp(Op.NOT, BinaryOp(Op.NULL_SAFE_EQ, lhs, rhs))
        if d == "not_distinct_from":
            rhs = self.lower(tail.children[-1])
            return BinaryOp(Op.NULL_SAFE_EQ, lhs, rhs)
        raise SemanticError(f"bad comp_tail: {d}")

    def _left_assoc(self, node: Tree, op: Op) -> Expr:
        children = [c for c in node.children if isinstance(c, Tree)]
        if len(children) == 1:
            return self.lower(children[0])
        result: Expr = self.lower(children[0])
        for c in children[1:]:
            result = BinaryOp(op, result, self.lower(c))
        return result

    def _additive(self, node: Tree) -> Expr:
        children = [c for c in node.children if isinstance(c, Tree)]
        if len(children) == 1:
            return self.lower(children[0])
        result: Expr = self.lower(children[0])
        for tail in children[1:]:
            op_map = {"add": Op.ADD, "sub": Op.SUB, "concat": Op.CONCAT}
            op = op_map[tail.data]
            rhs = self.lower(tail.children[0])
            result = BinaryOp(op, result, rhs)
        return result

    def _multiplicative(self, node: Tree) -> Expr:
        children = [c for c in node.children if isinstance(c, Tree)]
        if len(children) == 1:
            return self.lower(children[0])
        result: Expr = self.lower(children[0])
        for tail in children[1:]:
            op_map = {"mul": Op.MUL, "div": Op.DIV, "mod": Op.MOD}
            op = op_map[tail.data]
            rhs = self.lower(tail.children[0])
            result = BinaryOp(op, result, rhs)
        return result

    def _function_call(self, node: Tree) -> Expr:
        name_token = node.children[0]
        raw_name = _identifier(name_token).upper()  # type: ignore[arg-type]
        name = self.lowerer._canonicalize_function_name(raw_name)
        args_node = node.children[1]
        filter_node: Optional[Tree] = None
        over_node: Optional[Tree] = None
        for c in node.children[2:]:
            if isinstance(c, Tree) and c.data == "filter_clause":
                filter_node = c
            elif isinstance(c, Tree) and c.data == "over_clause":
                over_node = c

        agg_kind = _AGG_NAME_TO_KIND.get(name)
        is_window = over_node is not None
        distinct, args = self._lower_func_args(args_node, name, is_aggregate=agg_kind is not None)

        filter_pred: Optional[Expr] = None
        if filter_node is not None:
            if agg_kind is None:
                raise SemanticError("FILTER (WHERE ...) only allowed on aggregate calls")
            pred_node = next(
                c for c in filter_node.children if isinstance(c, Tree)
            )
            filter_pred = self.lower(pred_node)

        if is_window:
            window_kind = _WINDOW_NAME_TO_KIND.get(name)
            if window_kind is None:
                raise SemanticError(f"function {name} cannot be used as window")
            partition, order = self._lower_over(over_node)  # type: ignore[arg-type]
            wcall = WindowCall(
                kind=window_kind,
                args=tuple(args),
                partition_by=tuple(partition),
                order_by=tuple(order),
                frame=None,
            )
            slot = self._intern_window(name, wcall)
            return ColumnRef(name=slot)

        if agg_kind is not None:
            if agg_kind == AggKind.COUNT and not args:
                agg = AggCall(kind=AggKind.COUNT_STAR, filter_pred=filter_pred)
            else:
                agg = AggCall(
                    kind=agg_kind,
                    arg=args[0] if args else None,
                    distinct=distinct,
                    filter_pred=filter_pred,
                )
            slot = self._intern_agg(name, agg)
            return ColumnRef(name=slot)

        return FuncCall(name=name, args=tuple(args))

    def _lower_func_args(
        self, node: Tree, name: str, *, is_aggregate: bool
    ) -> tuple[bool, list[Expr]]:
        if node.data == "star_args":
            if name == "COUNT":
                return False, []
            raise SemanticError(f"{name}(*) not allowed")
        # normal_args
        distinct = any(
            isinstance(c, Tree) and c.data in ("distinct_kw", "unique_kw") for c in node.children
        )
        # Also check for bare UNIQUE/DISTINCT tokens in args
        if not distinct:
            for c in node.children:
                if isinstance(c, Token) and str(c).upper() in ("DISTINCT", "UNIQUE"):
                    distinct = True
                    break
        args: list[Expr] = []
        for c in node.children:
            if isinstance(c, Tree) and c.data == "expr_list":
                for e in c.children:
                    if isinstance(e, Tree):
                        args.append(self.lower(e))
        return distinct, args

    def _lower_over(self, node: Tree) -> tuple[list[Expr], list[OrderKey]]:
        partition: list[Expr] = []
        order: list[OrderKey] = []
        for c in node.children:
            if isinstance(c, Tree) and c.data == "partition_by":
                expr_list = c.children[0]
                assert isinstance(expr_list, Tree) and expr_list.data == "expr_list"
                for e in expr_list.children:
                    if isinstance(e, Tree):
                        partition.append(self.lower(e))
            if isinstance(c, Tree) and c.data == "order_by":
                for k in _children(c, "order_key"):
                    expr_node = k.children[0]
                    expr = self.lower(expr_node)
                    direction = SortDirection.ASC
                    nulls = NullsOrder.DEFAULT
                    for t in k.children[1:]:
                        if isinstance(t, Token):
                            val = str(t).upper()
                            if val == "DESC":
                                direction = SortDirection.DESC
                            elif val == "ASC":
                                direction = SortDirection.ASC
                            elif val == "FIRST":
                                nulls = NullsOrder.FIRST
                            elif val == "LAST":
                                nulls = NullsOrder.LAST
                    order.append(OrderKey(expr=expr, direction=direction, nulls=nulls))
        return partition, order

    def _case(self, node: Tree) -> Case:
        branches: list[tuple[Expr, Expr]] = []
        default: Optional[Expr] = None
        for c in node.children:
            if isinstance(c, Tree) and c.data == "case_branch":
                cond = self.lower(c.children[0])
                val = self.lower(c.children[1])
                branches.append((cond, val))
            elif isinstance(c, Tree) and c.data == "case_else":
                default = self.lower(c.children[0])
        return Case(branches=tuple(branches), default=default)

    def _intern_agg(self, name: str, agg: AggCall) -> str:
        slot = f"__agg_{len(self.agg_slots)}_{name.lower()}"
        self.agg_slots.append((slot, agg))
        return slot

    def _intern_window(self, name: str, w: WindowCall) -> str:
        slot = f"__win_{len(self.window_slots)}_{name.lower()}"
        self.window_slots.append((slot, w))
        return slot


# -------- type inference --------


NULLABLE_FLOAT = FLOAT


def _infer(
    e: Expr, scope: Scope, catalog: Catalog, ctes: dict[str, tuple[ColumnSchema, ...]]
) -> IRType:
    if isinstance(e, Literal):
        return e.type
    if isinstance(e, ColumnRef):
        _, t = scope.resolve(e.name, e.qualifier)
        return t
    if isinstance(e, BinaryOp):
        if e.op in (Op.EQ, Op.NEQ, Op.LT, Op.LTE, Op.GT, Op.GTE, Op.AND, Op.OR, Op.LIKE, Op.ILIKE, Op.NULL_SAFE_EQ):
            return BOOL
        if e.op == Op.CONCAT:
            return TEXT
        lt = _infer(e.left, scope, catalog, ctes)
        rt = _infer(e.right, scope, catalog, ctes)
        return FLOAT if FLOAT in (lt, rt) else lt
    if isinstance(e, UnaryOp):
        if e.op == Op.NOT:
            return BOOL
        return _infer(e.operand, scope, catalog, ctes)
    if isinstance(e, Cast):
        return e.target
    if isinstance(e, Case):
        if e.branches:
            return _infer(e.branches[0][1], scope, catalog, ctes)
        return INT
    if isinstance(e, FuncCall):
        return _func_result_type(e.name, e.args, scope, catalog, ctes)
    if isinstance(e, AggCall):
        if e.kind in (AggKind.COUNT_STAR, AggKind.COUNT):
            return INT
        if e.kind in (AggKind.SUM, AggKind.AVG):
            return FLOAT
        if e.arg is not None:
            return _infer(e.arg, scope, catalog, ctes)
        return FLOAT
    if isinstance(e, WindowCall):
        if e.kind in (WindowKind.ROW_NUMBER, WindowKind.RANK, WindowKind.DENSE_RANK):
            return INT
        if e.args:
            return _infer(e.args[0], scope, catalog, ctes)
        return INT
    if isinstance(e, (IsNull, Between, InList, InSubquery, ExistsSubquery)):
        return BOOL
    if isinstance(e, ScalarSubquery):
        sub_schema = e.plan.schema()
        if sub_schema:
            return sub_schema[0].type
        return FLOAT
    return INT


def _func_result_type(
    name: str,
    args: tuple[Expr, ...],
    scope: Scope,
    catalog: Catalog,
    ctes: dict[str, tuple[ColumnSchema, ...]],
) -> IRType:
    name_u = name.upper()
    if name_u in ("UPPER", "UCASE", "LOWER", "LCASE", "TRIM", "SUBSTR", "SUBSTRING", "REPLACE", "CONCAT"):
        return TEXT
    if name_u in ("LENGTH", "LEN", "CHAR_LENGTH"):
        return INT
    if name_u in ("ABS", "ROUND", "FLOOR", "CEIL", "CEILING"):
        if args:
            return _infer(args[0], scope, catalog, ctes)
        return FLOAT
    if name_u in ("COALESCE", "IF", "IIF", "NULLIF", "GREATEST", "LEAST", "NVL"):
        if args:
            return _infer(args[0], scope, catalog, ctes)
        return INT
    if name_u in ("DATE_PART", "EXTRACT"):
        return INT
    if name_u in ("DATE_TRUNC", "DATE_ADD", "DATE_SUB"):
        return DATE_T
    if name_u == "DATE_DIFF":
        return INT
    return FLOAT


def _infer_window_type(w: WindowCall) -> IRType:
    if w.kind in (WindowKind.ROW_NUMBER, WindowKind.RANK, WindowKind.DENSE_RANK):
        return INT
    if w.kind == WindowKind.COUNT:
        return INT
    return FLOAT


_AGG_NAME_TO_KIND: dict[str, AggKind] = {
    "COUNT": AggKind.COUNT,
    "SUM": AggKind.SUM,
    "AVG": AggKind.AVG,
    "MIN": AggKind.MIN,
    "MAX": AggKind.MAX,
}


_WINDOW_NAME_TO_KIND: dict[str, WindowKind] = {
    "ROW_NUMBER": WindowKind.ROW_NUMBER,
    "RANK": WindowKind.RANK,
    "DENSE_RANK": WindowKind.DENSE_RANK,
    "LAG": WindowKind.LAG,
    "LEAD": WindowKind.LEAD,
    "FIRST_VALUE": WindowKind.FIRST_VALUE,
    "LAST_VALUE": WindowKind.LAST_VALUE,
    "SUM": WindowKind.SUM,
    "AVG": WindowKind.AVG,
    "MIN": WindowKind.MIN,
    "MAX": WindowKind.MAX,
    "COUNT": WindowKind.COUNT,
}


def _type_name_to_ir(node: Tree) -> IRType:
    txt = " ".join(str(t).upper() for t in node.children if isinstance(t, Token))
    if txt in ("INT", "INTEGER", "BIGINT"):
        return INT
    if txt in ("FLOAT", "DOUBLE"):
        return FLOAT
    if txt in ("VARCHAR", "TEXT", "STRING"):
        return TEXT
    if txt in ("BOOLEAN", "BOOL"):
        return BOOL
    if txt == "DATE":
        return DATE_T
    if txt == "TIMESTAMP":
        return IRType(TypeKind.TIMESTAMP)
    return INT


def _patch_count_star() -> None:
    """No-op; placeholder for future arg-form detection."""
    pass


__all__ = ["lower", "Lowerer", "ExprLowerer", "Catalog", "Scope", "SemanticError"]