"""Dialect-specific lowering: Lark parse tree -> manysql IR.

Public entrypoint:
    lower(tree: Tree, config: SemanticConfig, catalog: dict[str, tuple[ColumnSchema,...]]) -> Plan

This lowering handles the ``postgres_sqlite_server_inside`` dialect whose
surface diverges from the reference ANSI dialect in several ways:

* JOIN keywords use ``LINK`` instead of ``JOIN`` (LINK, LEFT LINK, …)
* ORDER BY keyword is ``SORT`` instead of ``ORDER BY``
* Identifier quoting uses brackets ``[ident]``
* ``order_by_position`` is ``inside_from_brace`` (ORDER BY / SORT may appear
  inside the FROM clause scope)
* ``join_syntax`` is ``pipelined``
* Function aliases: IFNULL→COALESCE, LEN→LENGTH, TO_CHAR→EXTRACT,
  SUBSTRING→SUBSTR

The grammar produced by the dialect generator will emit tree node names that
reflect these surface tokens.  This lowering maps every variant back to the
same IR the reference dialect produces.
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
    by_name: dict[str, list[tuple[Optional[str], IRType]]] = field(default_factory=dict)
    cte_columns: dict[str, tuple[ColumnSchema, ...]] = field(default_factory=dict)

    def add(
        self,
        binding: Optional[str],
        cols: Sequence[ColumnSchema],
        *,
        effective_qualifier: Optional[str] = None,
    ) -> None:
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
        if len({(q, t) for q, t in bare}) == 1:
            return bare[0]
        raise SemanticError(f"ambiguous unqualified column: {name}")


class SemanticError(Exception):
    pass


# -------- entry --------


def lower(tree: Tree, config: SemanticConfig, catalog: Catalog) -> Plan:
    """Public entrypoint required by manysql.dialects.registry.DialectEngine."""
    lowerer = Lowerer(catalog=catalog, config=config)
    # The top-level tree may be the statement directly, or wrap it.
    node = tree
    # Walk down to the statement node if present.
    if _data(node) == "start" or _data(node) == "query":
        node = node.children[0]
    if _data(node) == "statement":
        return lowerer.lower_statement(node)
    # If the tree *is* the statement content already, try to handle it.
    if _data(node) in ("query_expr", "select_core"):
        return lowerer.lower_query_expr(node) if _data(node) == "query_expr" else lowerer.lower_select_core(node)
    # Fallback: assume it's a statement wrapper.
    return lowerer.lower_statement(node)


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


def _first_child_any(node: Tree, *names: str) -> Optional[Tree]:
    """Return the first child Tree whose .data matches any of *names*."""
    for c in node.children:
        if isinstance(c, Tree) and c.data in names:
            return c
    return None


def _identifier(token: Token) -> str:
    """Strip surrounding quotes (double-quote or bracket) from an identifier."""
    s = str(token)
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    if s.startswith('[') and s.endswith(']'):
        return s[1:-1]
    return s


def _has_token(node: Tree, value: str) -> bool:
    needle = value.upper()
    for c in node.children:
        if isinstance(c, Token) and str(c).upper() == needle:
            return True
    return False


def _has_token_any(node: Tree, *values: str) -> bool:
    for v in values:
        if _has_token(node, v):
            return True
    return False


def _collect_tokens(node: Tree) -> list[str]:
    """Return all Token string values (uppercased) in a node's direct children."""
    return [str(c).upper() for c in node.children if isinstance(c, Token)]


# -------- lowerer --------


class Lowerer:
    def __init__(self, catalog: Catalog, config: SemanticConfig) -> None:
        self.catalog = catalog
        self.config = config
        self._cte_bindings: dict[str, tuple[ColumnSchema, ...]] = {}

    def lower_statement(self, node: Tree) -> Plan:
        with_node = _first_child(node, "with_clause")
        query_node = _first_child(node, "query_expr")
        if query_node is None:
            # Maybe the statement IS the query_expr
            for c in node.children:
                if isinstance(c, Tree) and c.data in ("query_expr", "select_core"):
                    query_node = c
                    break
        if query_node is None:
            # Last resort: treat the node itself as a query_expr
            query_node = node
        if with_node is None:
            return self.lower_query_expr(query_node)

        recursive = _has_token(with_node, "RECURSIVE")
        cte_list = _first_child(with_node, "cte_list")
        if cte_list is None:
            # CTEs might be direct children of with_clause
            ctes = _children(with_node, "cte")
        else:
            ctes = _children(cte_list, "cte")

        if recursive:
            if len(ctes) != 1:
                raise SemanticError("WITH RECURSIVE supports a single binding in this dialect")
            cte = ctes[0]
            name = _identifier(cte.children[0])
            inner = self._find_cte_body(cte)
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
            name = _identifier(cte.children[0])
            inner = self._find_cte_body(cte)
            cte_plan = self.lower_query_expr(inner)
            bindings.append(CTEBinding(name=name, plan=cte_plan))
            self._cte_bindings[name] = tuple(cte_plan.schema())
        body_plan = self.lower_query_expr(query_node)
        return WithCTE(bindings=tuple(bindings), body=body_plan)

    def _find_cte_body(self, cte_node: Tree) -> Tree:
        """Find the query_expr (or select_core) inside a CTE definition."""
        for c in cte_node.children:
            if isinstance(c, Tree) and c.data in ("query_expr", "select_core"):
                return c
        # Might be the second child after the name token
        for c in cte_node.children[1:]:
            if isinstance(c, Tree):
                return c
        raise SemanticError("cannot find CTE body")

    def _split_recursive(
        self, query_node: Tree, name: str
    ) -> tuple[Plan, Plan, bool]:
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
        # If we got a select_core directly, wrap it.
        if _data(node) == "select_core":
            return self.lower_select_core(node)

        cores: list[Tree] = _children(node, "select_core")
        branches: list[Tree] = _children(node, "set_op_branch")

        # Also look for order_by / sort_by / limit in the query_expr
        order_node = _first_child_any(node, "order_by", "sort_by", "sort_clause", "order_clause")
        limit_node = _first_child_any(node, "limit_clause")

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
            plan = self.lower_select_core(cores[0])
            for branch in branches:
                op_node = branch.children[0]
                right_core = branch.children[1]
                assert isinstance(op_node, Tree)
                assert isinstance(right_core, Tree) and right_core.data == "select_core"
                right = self.lower_select_core(right_core)
                kind = self._set_op_kind(op_node)
                all_mode = _has_token(op_node, "ALL")
                plan = SetOp(kind=kind, left=plan, right=right, all=all_mode)
            if order_node is not None:
                plan = self._lower_order_by(plan, order_node)

        if limit_node is not None:
            plan = self._lower_limit(plan, limit_node)
        return plan

    def _set_op_kind(self, op_node: Tree) -> SetOpKind:
        d = op_node.data.lower()
        if "union" in d:
            return SetOpKind.UNION
        if "intersect" in d:
            return SetOpKind.INTERSECT
        if "except" in d:
            return SetOpKind.EXCEPT
        # Fallback: check tokens
        tokens = _collect_tokens(op_node)
        for t in tokens:
            if "UNION" in t:
                return SetOpKind.UNION
            if "INTERSECT" in t:
                return SetOpKind.INTERSECT
            if "EXCEPT" in t:
                return SetOpKind.EXCEPT
        return SetOpKind.UNION

    def lower_select_core(self, node: Tree) -> Plan:
        body, projections, proj_types, distinct, _scope = self.lower_select_core_open(node)
        plan: Plan = Project(input=body, projections=projections, output_types=proj_types)
        if distinct:
            plan = Distinct(input=plan)
        return plan

    def lower_select_core_open(
        self, node: Tree
    ) -> tuple[Plan, tuple[tuple[str, Expr], ...], tuple[IRType, ...], bool, "Scope"]:
        distinct = any(
            isinstance(c, Tree) and c.data in ("select_distinct", "distinct_kw")
            for c in node.children
        ) or _has_token(node, "DISTINCT")

        select_list = _first_child(node, "select_list")
        from_clause = _first_child_any(node, "from_clause")
        where_node = _first_child_any(node, "where_clause")
        group_node = _first_child_any(node, "group_by_clause")
        having_node = _first_child_any(node, "having_clause")

        # The dialect may place ORDER BY (SORT) inside the FROM clause scope.
        # We'll look for it here and also in the parent query_expr.
        inner_order_node = _first_child_any(node, "order_by", "sort_by", "sort_clause", "order_clause")

        where_expr = None
        if where_node is not None:
            # The WHERE clause child is the predicate expression
            for c in where_node.children:
                if isinstance(c, Tree):
                    where_expr = c
                    break

        having_expr = None
        if having_node is not None:
            for c in having_node.children:
                if isinstance(c, Tree):
                    having_expr = c
                    break

        group_expr_list = None
        if group_node is not None:
            group_expr_list = _first_child(group_node, "expr_list")
            if group_expr_list is None:
                group_expr_list = group_node

        assert select_list is not None
        if from_clause is None:
            raise SemanticError("FROM clause required")

        plan, scope = self.lower_from(from_clause)

        if where_expr is not None:
            elower = ExprLowerer(self.config, scope, self)
            pred = elower.lower(where_expr)
            plan = Filter(input=plan, predicate=pred)

        elower = ExprLowerer(self.config, scope, self)
        elower.collecting_aggs = True
        agg_slots: list[tuple[str, AggCall]] = elower.agg_slots

        expanded_select: list[tuple[str, Expr]] = []
        for item in _children(select_list):
            if not isinstance(item, Tree):
                continue
            if item.data == "star":
                expanded_select.extend(self._expand_star(scope))
            elif item.data == "qualified_star":
                qual = _identifier(item.children[0])
                expanded_select.extend(self._expand_qualified_star(scope, qual))
            elif item.data == "select_expr":
                e_node = item.children[0]
                alias_token = item.children[1] if len(item.children) > 1 else None
                expr = elower.lower(e_node)
                alias = (
                    _identifier(alias_token)
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
        if group_expr_list is not None:
            elower3 = ExprLowerer(self.config, scope, self)
            alias_map = (
                {a: e for a, e in expanded_select}
                if self.config.group_by_accepts_select_aliases
                else {}
            )
            expr_children = _children(group_expr_list, "expr") or [
                c for c in group_expr_list.children if isinstance(c, Tree)
            ]
            for ge in expr_children:
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
                [self._infer_expr_type(e, scope) for _, e in group_by_exprs]
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
            if c.data in ("join_clause", "link_clause"):
                plan, scope = self._lower_join_clause(plan, scope, c)
            elif c.data in ("table_ident", "table_subquery"):
                # Implicit cross join (comma-separated tables)
                right_plan, right_scope = self._lower_table_ref(c)
                merged = Scope()
                merged.by_name = {**scope.by_name}
                for k, v in right_scope.by_name.items():
                    merged.by_name.setdefault(k, []).extend(v)
                plan = Join(left=plan, right=right_plan, kind=JoinKind.CROSS, on=None, using=())
                scope = merged
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
        # Find the join kind node — may be named *_join or *_link
        kind_node = None
        for c in node.children:
            if isinstance(c, Tree) and (c.data.endswith("_join") or c.data.endswith("_link")):
                kind_node = c
                break
        if kind_node is None:
            # Try to infer from tokens
            kind_node = node  # fallback

        right_node = None
        for c in node.children:
            if isinstance(c, Tree) and c.data in ("table_ident", "table_subquery"):
                right_node = c
                break
        if right_node is None:
            raise SemanticError("join clause missing table reference")

        cond_node = None
        for c in node.children:
            if isinstance(c, Tree) and c.data in ("join_on", "link_on", "join_using", "link_using"):
                cond_node = c
                break

        right_plan, right_scope = self._lower_table_ref(right_node)

        merged = Scope()
        merged.by_name = {**left_scope.by_name}
        for k, v in right_scope.by_name.items():
            merged.by_name.setdefault(k, []).extend(v)

        kind = self._resolve_join_kind(kind_node)

        on_expr: Optional[Expr] = None
        using_cols: tuple[str, ...] = ()
        if cond_node is not None:
            if cond_node.data in ("join_on", "link_on"):
                elower = ExprLowerer(self.config, merged, self)
                on_expr = elower.lower(cond_node.children[0])
            elif cond_node.data in ("join_using", "link_using"):
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

    def _resolve_join_kind(self, node: Tree) -> JoinKind:
        d = node.data.lower()
        kind_map = {
            "inner_join": JoinKind.INNER,
            "inner_link": JoinKind.INNER,
            "left_join": JoinKind.LEFT,
            "left_link": JoinKind.LEFT,
            "right_join": JoinKind.RIGHT,
            "right_link": JoinKind.RIGHT,
            "full_join": JoinKind.FULL,
            "full_link": JoinKind.FULL,
            "cross_join": JoinKind.CROSS,
            "cross_link": JoinKind.CROSS,
        }
        if d in kind_map:
            return kind_map[d]
        # Fallback: check tokens in the node
        tokens = " ".join(str(c).upper() for c in node.children if isinstance(c, Token))
        if "LEFT" in tokens:
            return JoinKind.LEFT
        if "RIGHT" in tokens:
            return JoinKind.RIGHT
        if "FULL" in tokens:
            return JoinKind.FULL
        if "CROSS" in tokens:
            return JoinKind.CROSS
        return JoinKind.INNER

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
        for k in _children(node, "order_key") or _children(node, "sort_key"):
            expr_node = k.children[0]
            if (
                isinstance(expr_node, Tree)
                and expr_node.data == "column_ref"
                and len(expr_node.children) == 1
            ):
                ident = _identifier(expr_node.children[0])
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
                elif isinstance(t, Tree):
                    # Could be a nulls_first / nulls_last node
                    td = t.data.lower()
                    if "first" in td:
                        nulls = NullsOrder.FIRST
                    elif "last" in td:
                        nulls = NullsOrder.LAST
                    # Could be asc/desc node
                    if "desc" in td:
                        direction = SortDirection.DESC
                    elif "asc" in td:
                        direction = SortDirection.ASC
            keys.append(OrderKey(expr=expr, direction=direction, nulls=nulls))
        return Sort(input=plan, keys=tuple(keys))

    def _lower_limit(self, plan: Plan, node: Tree) -> Plan:
        ints = [int(t) for t in node.children if isinstance(t, Token) and str(t).isdigit()]
        limit = ints[0] if ints else 0
        offset = ints[1] if len(ints) > 1 else 0
        return Limit(input=plan, limit=limit, offset=offset)

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
        return _identifier(node.children[0])
    return None


# Function name normalization: dialect aliases -> canonical IR names
_FUNC_ALIASES: dict[str, str] = {
    "IFNULL": "COALESCE",
    "LEN": "LENGTH",
    "TO_CHAR": "EXTRACT",
    "SUBSTRING": "SUBSTR",
}


@dataclass
class ExprLowerer:
    config: SemanticConfig
    scope: Scope
    lowerer: Lowerer
    agg_slots: list[tuple[str, AggCall]] = field(default_factory=list)
    window_slots: list[tuple[str, WindowCall]] = field(default_factory=list)
    collecting_aggs: bool = False

    def lower(self, node: Tree | Token) -> Expr:
        if isinstance(node, Token):
            raise SemanticError(f"unexpected raw token in expr: {node!r}")

        d = node.data
        if d == "literal":
            return self._literal(node)
        if d in ("number_literal", "string_literal", "true_literal", "false_literal",
                  "null_literal", "date_literal"):
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
            sub = self.lowerer.lower_query_expr(node.children[0])
            return ExistsSubquery(plan=sub, negated=False)
        if d == "not_exists_expr":
            sub = self.lowerer.lower_query_expr(node.children[0])
            return ExistsSubquery(plan=sub, negated=True)
        if d == "comparison_n":
            return self._comparison(node)
        if d in ("or_expr_n", "or_expr"):
            return self._left_assoc(node, Op.OR)
        if d in ("and_expr_n", "and_expr"):
            return self._left_assoc(node, Op.AND)
        if d in ("not_op", "not_expr"):
            return UnaryOp(Op.NOT, self.lower(node.children[0]))
        if d in ("additive_n", "additive"):
            return self._additive(node)
        if d in ("multiplicative_n", "multiplicative"):
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
        # Fallback: single-child wrappers
        if len(node.children) == 1:
            child = node.children[0]
            if isinstance(child, Tree):
                return self.lower(child)
        raise SemanticError(f"unhandled expr rule: {d}")

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
            # Strip surrounding single quotes
            if text.startswith("'") and text.endswith("'"):
                return Literal(text[1:-1].replace("''", "'"), TEXT)
            return Literal(text, TEXT)
        if kind == "true_literal":
            return Literal(True, BOOL)
        if kind == "false_literal":
            return Literal(False, BOOL)
        if kind == "null_literal":
            return Literal(None, IRType(TypeKind.NULL))
        if kind == "date_literal":
            from datetime import date as _date
            text = str(node.children[0])
            text = text[1:-1]
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
            sub = self.lowerer.lower_query_expr(tail.children[0])
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
            op = op_map.get(tail.data, Op.ADD)
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
            op = op_map.get(tail.data, Op.MUL)
            rhs = self.lower(tail.children[0])
            result = BinaryOp(op, result, rhs)
        return result

    def _function_call(self, node: Tree) -> Expr:
        name_token = node.children[0]
        raw_name = _identifier(name_token).upper()
        # Normalize function aliases to canonical names
        name = _FUNC_ALIASES.get(raw_name, raw_name)

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
            partition, order = self._lower_over(over_node)
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
        distinct = any(
            isinstance(c, Tree) and c.data == "distinct_kw" for c in node.children
        ) or _has_token(node, "DISTINCT")
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
            if isinstance(c, Tree) and c.data in ("order_by", "sort_by", "sort_clause", "order_clause"):
                for k in _children(c, "order_key") or _children(c, "sort_key"):
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
                        elif isinstance(t, Tree):
                            td = t.data.lower()
                            if "first" in td:
                                nulls = NullsOrder.FIRST
                            elif "last" in td:
                                nulls = NullsOrder.LAST
                            if "desc" in td:
                                direction = SortDirection.DESC
                            elif "asc" in td:
                                direction = SortDirection.ASC
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
    if name_u in ("COALESCE", "IF", "IIF", "NULLIF", "IFNULL", "GREATEST", "LEAST"):
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


__all__ = ["lower", "Lowerer", "ExprLowerer", "Catalog", "Scope", "SemanticError"]