"""Parse battery: source SQL strings every generated grammar must accept.

The battery is built deterministically from a fixed list of canonical reference
SQL queries. For a given `DialectSpec`, each canonical query is rewritten into
the spec's surface so that:

  - keyword renames are applied (e.g. SELECT -> PICK)
  - multi-word keyword renames (LEFT JOIN -> LEFT_JOIN) are applied
  - operator renames (= -> EQ, || -> CONCAT) are applied
  - NULL literal renames (NULL -> NIL) are applied
  - LIMIT syntax rewrites are applied
  - statement terminator is appended if requested

Each `BatteryItem` carries a small label so failures are easy to attribute.

Why a hand-curated list rather than the full golden corpus? Speed and
focus. The battery's purpose is to validate *grammar* shape, not semantics —
twenty well-chosen queries cover every grammar production we generate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from lark import Lark
from lark.exceptions import LarkError

from manysql.spec.dialect import (
    DialectSpec,
    IdentifierQuote,
    LimitSyntax,
    NullLiteral,
    SetOpPrecedence,
    StringQuote,
    SurfaceSpec,
    WildcardChar,
)


@dataclass(frozen=True)
class BatteryItem:
    label: str
    source: str


@dataclass(frozen=True)
class BatteryFailure:
    label: str
    source: str
    error: str


@dataclass(frozen=True)
class ValidationReport:
    items: list[BatteryItem]
    failures: list[BatteryFailure]

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        if self.ok:
            return f"parse battery: {len(self.items)} / {len(self.items)} OK"
        return (
            f"parse battery: {len(self.items) - len(self.failures)} / "
            f"{len(self.items)} OK; failures: "
            + ", ".join(f.label for f in self.failures)
        )


# Hand-curated canonical SQL strings written in the *reference* surface.
# Each tuple is (label, sql). Keep these short, valid, and feature-diverse.
_REFERENCE_SQL: list[tuple[str, str]] = [
    ("scan_all", "SELECT * FROM employees"),
    ("scan_subset", "SELECT id, name FROM employees"),
    ("filter_eq", "SELECT id FROM employees WHERE dept_id = 10"),
    ("filter_neq", "SELECT id FROM employees WHERE dept_id <> 10"),
    ("filter_in", "SELECT id FROM employees WHERE dept_id IN (10, 20)"),
    ("filter_between", "SELECT id FROM employees WHERE salary BETWEEN 80000 AND 120000"),
    ("filter_like", "SELECT id FROM employees WHERE name LIKE 'A%'"),
    ("filter_is_null", "SELECT id FROM employees WHERE dept_id IS NULL"),
    ("filter_is_not_null", "SELECT id FROM employees WHERE dept_id IS NOT NULL"),
    ("project_arith", "SELECT id, salary + 1000 AS bumped FROM employees"),
    ("project_concat", "SELECT name || '!' AS shouted FROM employees"),
    ("project_case", "SELECT id, CASE WHEN salary > 100000 THEN 'high' ELSE 'low' END AS tier FROM employees"),
    (
        "join_inner",
        "SELECT e.id, d.name FROM employees e INNER JOIN departments d ON e.dept_id = d.id",
    ),
    (
        "join_left",
        "SELECT e.id, d.name FROM employees e LEFT JOIN departments d ON e.dept_id = d.id",
    ),
    (
        "agg_group_by",
        "SELECT dept_id, COUNT(*) AS n FROM employees GROUP BY dept_id",
    ),
    (
        "agg_having",
        "SELECT dept_id, COUNT(*) AS n FROM employees GROUP BY dept_id HAVING COUNT(*) > 1",
    ),
    ("order_asc", "SELECT id FROM employees ORDER BY salary ASC"),
    ("order_desc_nulls_last", "SELECT id FROM employees ORDER BY dept_id DESC NULLS LAST"),
    ("limit_only", "SELECT id FROM employees LIMIT 5"),
    ("limit_offset", "SELECT id FROM employees LIMIT 5 OFFSET 10"),
    ("distinct", "SELECT DISTINCT dept_id FROM employees"),
    (
        "union_all",
        "SELECT id FROM employees UNION ALL SELECT id FROM departments",
    ),
    (
        "subq_in",
        "SELECT id FROM employees WHERE dept_id IN (SELECT id FROM departments)",
    ),
    (
        "cte_simple",
        "WITH high AS (SELECT id FROM employees WHERE salary > 100000) "
        "SELECT * FROM high",
    ),
    (
        "window_row_number",
        "SELECT id, ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary) AS rn FROM employees",
    ),
    ("cast_int", "SELECT CAST(salary AS INT) FROM employees"),
]


# Axis-targeted items: one canonical reference SQL per non-default surface
# axis. Each item carries a predicate that decides whether the spec exercises
# the axis. Default specs (reference, mild_postgres_ish, etc.) yield empty
# axis-targeted lists, so existing behavior is unchanged.
#
# Every item must (after `apply_surface`) parse against both the reference
# grammar AND the dialect grammar; otherwise the IR battery would compare
# against a parse failure. The items below are explicitly written in
# *reference* surface, with `apply_surface` doing the rewriting.
_AXIS_TARGETED_SQL: list[tuple[str, str, Callable[[SurfaceSpec], bool]]] = [
    (
        "axis_quoted_ident_select",
        'SELECT "id" FROM employees',
        lambda s: s.identifier_quote != IdentifierQuote.DOUBLE,
    ),
    (
        "axis_null_safe_eq",
        "SELECT id FROM employees WHERE dept_id IS NOT DISTINCT FROM 10",
        lambda s: s.null_safe_eq_op is not None
        and s.null_safe_eq_op.upper() != "IS NOT DISTINCT FROM",
    ),
    (
        "axis_function_alias_coalesce",
        "SELECT COALESCE(dept_id, 0) AS dept FROM employees",
        lambda s: _has_function_alias(s, "COALESCE"),
    ),
    (
        "axis_function_alias_length",
        "SELECT LENGTH(name) AS n FROM employees",
        lambda s: _has_function_alias(s, "LENGTH"),
    ),
    (
        "axis_wildcard_select",
        "SELECT * FROM employees",
        lambda s: s.wildcard_char != WildcardChar.STAR,
    ),
    (
        "axis_mod",
        "SELECT id FROM employees WHERE salary % 100 = 0",
        lambda s: s.mod_op != "%",
    ),
    (
        "axis_concat",
        "SELECT name || '!' AS shouted FROM employees",
        # Skip when concat overloads add/sub: apply_surface deliberately
        # leaves || in place there, so the item degenerates into an
        # already-covered case.
        lambda s: s.concat_op != "||"
        and s.concat_op != s.add_op
        and s.concat_op != s.sub_op,
    ),
    (
        "axis_setop_precedence",
        # Three-way set op so the precedence climber and the left-fold produce
        # different IR; tables must all live in the test catalog.
        "SELECT id FROM employees "
        "UNION ALL SELECT id FROM departments "
        "INTERSECT SELECT id FROM regions",
        lambda s: s.set_op_precedence != SetOpPrecedence.ANSI,
    ),
    (
        "axis_comparison_neq",
        "SELECT id FROM employees WHERE dept_id <> 10",
        lambda s: tuple(s.neq_op) != ("<>", "!="),
    ),
    (
        "axis_string_literal",
        "SELECT id FROM employees WHERE name = 'Alice'",
        lambda s: s.string_quote != StringQuote.SINGLE,
    ),
]


def _has_function_alias(surface: SurfaceSpec, canonical: str) -> bool:
    """True when the spec advertises a non-canonical primary alias for ``canonical``."""
    aliases = surface.function_aliases.get(canonical) or []
    return bool(aliases) and aliases[0].upper() != canonical.upper()


def axis_battery_items(spec: DialectSpec) -> list[tuple[str, str]]:
    """Subset of ``_AXIS_TARGETED_SQL`` whose predicate fires for ``spec``.

    Exposed so the IR battery can stay in sync with the parse battery
    without duplicating the predicate table.
    """
    return [
        (label, sql)
        for label, sql, predicate in _AXIS_TARGETED_SQL
        if predicate(spec.surface)
    ]


def build_parse_battery(spec: DialectSpec) -> list[BatteryItem]:
    """Return one BatteryItem per canonical SQL, rewritten into the spec's surface.

    Adds axis-targeted items (operator/wildcard/quoted-ident/etc.) when the
    spec exercises the relevant axis. For default specs the result is the
    historical fixed corpus.
    """
    items: list[BatteryItem] = []
    for label, sql in _REFERENCE_SQL:
        items.append(
            BatteryItem(label=label, source=apply_surface(sql, spec.surface))
        )
    for label, sql in axis_battery_items(spec):
        items.append(
            BatteryItem(label=label, source=apply_surface(sql, spec.surface))
        )
    return items


# --------------------------------------------------------------------------
# Rejection battery
# --------------------------------------------------------------------------
#
# Symmetric to the parse battery: where the parse battery says "this must
# parse", the rejection battery says "this must NOT parse against the
# dialect grammar". Each item is a *reference-form* SQL fragment paired
# with a reason so a divergence is easy to attribute. Items are emitted
# only when the spec actually diverges on the relevant axis, so a
# reference / mild dialect produces an empty rejection battery and the
# build is a no-op.
#
# This catches one specific failure mode that the parse battery cannot:
# dialects whose grammar happens to *also* accept the original ANSI
# spelling of a knob, making the divergence a nominal-only feature
# (e.g. spec advertises ``SELECT -> PICK`` but the grammar never removed
# the ``SELECT`` branch).


@dataclass(frozen=True)
class RejectionBatteryItem:
    """A reference-form SQL fragment the dialect's grammar must refuse to parse."""

    label: str
    source: str
    reason: str


@dataclass(frozen=True)
class RejectionFailure:
    """An item that *parsed* despite the dialect claiming to diverge on its axis."""

    label: str
    source: str
    reason: str


@dataclass(frozen=True)
class RejectionReport:
    items: list[RejectionBatteryItem]
    failures: list[RejectionFailure]

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        if self.ok:
            return f"rejection battery: {len(self.items)} / {len(self.items)} OK"
        return (
            f"rejection battery: {len(self.items) - len(self.failures)} / "
            f"{len(self.items)} OK; over-permissive: "
            + ", ".join(f.label for f in self.failures)
        )


def build_rejection_battery(spec: DialectSpec) -> list[RejectionBatteryItem]:
    """Build the must-fail-to-parse battery for ``spec``.

    Items are tied to non-default surface knobs; an unconfigured spec
    yields an empty list so reference / mild dialects pay nothing for
    this check.
    """
    surface = spec.surface
    items: list[RejectionBatteryItem] = []

    # Single-keyword renames: the reference form should no longer parse.
    keyword_axes: list[tuple[str, str, str]] = [
        ("select_keyword", "SELECT", "SELECT id FROM employees"),
        ("from_keyword", "FROM", "SELECT id FROM employees"),
        ("where_keyword", "WHERE", "SELECT id FROM employees WHERE dept_id = 10"),
        (
            "group_by_keyword",
            "GROUP BY",
            "SELECT dept_id, COUNT(*) FROM employees GROUP BY dept_id",
        ),
        ("limit_keyword", "LIMIT", "SELECT id FROM employees LIMIT 5"),
        ("distinct_keyword", "DISTINCT", "SELECT DISTINCT dept_id FROM employees"),
        ("union_keyword", "UNION", "SELECT id FROM employees UNION SELECT id FROM departments"),
    ]
    for attr, default, sql in keyword_axes:
        current = getattr(surface, attr)
        if current.upper() != default:
            items.append(
                RejectionBatteryItem(
                    label=f"reject_{attr}",
                    source=sql,
                    reason=f"{attr} is {current!r}, must reject {default!r}",
                )
            )

    if surface.requires_semicolon:
        items.append(
            RejectionBatteryItem(
                label="reject_missing_semicolon",
                source="SELECT id FROM employees",
                reason="requires_semicolon=True; statement without terminator must reject",
            )
        )

    if surface.wildcard_char != WildcardChar.STAR:
        items.append(
            RejectionBatteryItem(
                label="reject_star_wildcard",
                source="SELECT * FROM employees",
                reason=f"wildcard_char={surface.wildcard_char.value}; bare * must reject",
            )
        )

    if surface.eq_op != "=":
        items.append(
            RejectionBatteryItem(
                label="reject_ansi_eq",
                source="SELECT id FROM employees WHERE dept_id = 10",
                reason=f"eq_op={surface.eq_op!r}; ANSI = must reject",
            )
        )

    # neq divergence: only meaningful when the spec drops one of the two ANSI
    # spellings entirely.
    neq_set = {op for op in surface.neq_op}
    if "<>" not in neq_set:
        items.append(
            RejectionBatteryItem(
                label="reject_ansi_neq_angle",
                source="SELECT id FROM employees WHERE dept_id <> 10",
                reason=f"neq_op={surface.neq_op}; <> dropped, must reject",
            )
        )
    if "!=" not in neq_set:
        items.append(
            RejectionBatteryItem(
                label="reject_ansi_neq_bang",
                source="SELECT id FROM employees WHERE dept_id != 10",
                reason=f"neq_op={surface.neq_op}; != dropped, must reject",
            )
        )

    if surface.mod_op != "%":
        items.append(
            RejectionBatteryItem(
                label="reject_ansi_mod",
                source="SELECT id FROM employees WHERE salary % 100 = 0",
                reason=f"mod_op={surface.mod_op!r}; ANSI % must reject",
            )
        )

    # Concat: skip when the new spelling overloads add/sub, since the grammar
    # deliberately retains ``||`` in that case for disambiguation. Only assert
    # rejection when concat moves to a brand-new operator.
    if (
        surface.concat_op != "||"
        and surface.concat_op != surface.add_op
        and surface.concat_op != surface.sub_op
    ):
        items.append(
            RejectionBatteryItem(
                label="reject_ansi_concat",
                source="SELECT name || '!' FROM employees",
                reason=f"concat_op={surface.concat_op!r}; ANSI || must reject",
            )
        )

    # The ``null_literal`` knob governs the *standalone* NULL value spelling
    # (e.g. ``SELECT NIL`` instead of ``SELECT NULL``); ``null_keyword`` is
    # what powers ``IS NULL`` syntax. Only emit a must-reject item when the
    # keyword form actually changes — many specs swap only the literal.
    if surface.null_keyword.upper() != "NULL":
        items.append(
            RejectionBatteryItem(
                label="reject_ansi_null_keyword",
                source="SELECT id FROM employees WHERE dept_id IS NULL",
                reason=f"null_keyword={surface.null_keyword!r}; ANSI 'IS NULL' must reject",
            )
        )

    if surface.string_quote != StringQuote.SINGLE:
        items.append(
            RejectionBatteryItem(
                label="reject_ansi_string",
                source="SELECT id FROM employees WHERE name = 'Alice'",
                reason=f"string_quote={surface.string_quote.value}; ANSI ' must reject",
            )
        )

    if (
        surface.null_safe_eq_op is not None
        and surface.null_safe_eq_op.upper() != "IS NOT DISTINCT FROM"
    ):
        items.append(
            RejectionBatteryItem(
                label="reject_ansi_null_safe_eq",
                source="SELECT id FROM employees WHERE dept_id IS NOT DISTINCT FROM 10",
                reason=(
                    f"null_safe_eq_op={surface.null_safe_eq_op!r}; "
                    "ANSI 'IS NOT DISTINCT FROM' must reject"
                ),
            )
        )

    return items


def validate_rejection_battery(
    grammar_text: str,
    items: list[RejectionBatteryItem],
) -> RejectionReport:
    """Confirm the dialect grammar rejects every reference-form must-fail item.

    A grammar that *parses* an item we expected to reject is recorded as
    a failure: it indicates a fake divergence (the grammar still admits
    the original ANSI spelling alongside the new one).
    """
    if not items:
        return RejectionReport(items=items, failures=[])
    try:
        parser = Lark(grammar_text, start="start", parser="earley")
    except LarkError as e:
        # If the grammar itself doesn't compile, the parse battery already
        # captured that; don't double-report.
        return RejectionReport(
            items=items,
            failures=[
                RejectionFailure(
                    label=item.label,
                    source=item.source,
                    reason=f"grammar build failed: {e}",
                )
                for item in items
            ],
        )
    failures: list[RejectionFailure] = []
    for item in items:
        try:
            parser.parse(item.source)
        except Exception:
            continue
        # Parse succeeded -> the dialect did NOT actually diverge on this axis.
        failures.append(
            RejectionFailure(
                label=item.label,
                source=item.source,
                reason=item.reason,
            )
        )
    return RejectionReport(items=items, failures=failures)


def validate_grammar(
    grammar_text: str,
    items: list[BatteryItem],
) -> ValidationReport:
    """Try to parse every battery item with the given Lark grammar.

    Returns a structured report; never raises (grammar-construction errors
    are captured per-item).
    """
    try:
        parser = Lark(grammar_text, start="start", parser="earley")
    except LarkError as e:
        return ValidationReport(
            items=items,
            failures=[
                BatteryFailure(
                    label=item.label,
                    source=item.source,
                    error=f"grammar build failed: {e}",
                )
                for item in items
            ],
        )
    failures: list[BatteryFailure] = []
    for item in items:
        try:
            parser.parse(item.source)
        except Exception as e:
            failures.append(
                BatteryFailure(
                    label=item.label,
                    source=item.source,
                    error=f"{type(e).__name__}: {e}",
                )
            )
    return ValidationReport(items=items, failures=failures)


# --------------------------------------------------------------------------
# Surface rewriter
# --------------------------------------------------------------------------

# Single-word reference keywords mapped to the SurfaceSpec attribute that
# overrides them. Multi-word combos are handled separately so that we don't
# turn `LEFT JOIN` into `<left_keyword> <join_keyword>`.
_SINGLE_KEYWORD_ATTRS: list[tuple[str, str]] = [
    ("SELECT", "select_keyword"),
    ("FROM", "from_keyword"),
    ("WHERE", "where_keyword"),
    ("HAVING", "having_keyword"),
    ("DISTINCT", "distinct_keyword"),
    ("UNION", "union_keyword"),
    ("INTERSECT", "intersect_keyword"),
    ("EXCEPT", "except_keyword"),
    ("WITH", "with_keyword"),
    ("AS", "as_keyword"),
    ("CASE", "case_keyword"),
    ("WHEN", "when_keyword"),
    ("THEN", "then_keyword"),
    ("ELSE", "else_keyword"),
    ("END", "end_keyword"),
    ("CAST", "cast_keyword"),
    ("BETWEEN", "between_keyword"),
    ("LIKE", "like_keyword"),
    ("ILIKE", "ilike_keyword"),
    ("EXISTS", "exists_keyword"),
    ("LIMIT", "limit_keyword"),
]


_MULTI_WORD_REWRITES: list[tuple[str, str]] = [
    # (reference_text, surface_attr_name)
    ("GROUP BY", "group_by_keyword"),
    ("ORDER BY", "order_by_keyword"),
    ("LEFT JOIN", "join_left_keyword"),
    ("RIGHT JOIN", "join_right_keyword"),
    ("FULL JOIN", "join_full_keyword"),
    ("CROSS JOIN", "join_cross_keyword"),
    ("INNER JOIN", "join_inner_keyword"),
    ("NULLS FIRST", "nulls_first_keyword"),
    ("NULLS LAST", "nulls_last_keyword"),
]


# Sentinel used to protect the rewritten `null_keyword` inside `IS [NOT] NULL`
# from being clobbered when we later rewrite a standalone `NULL` literal.
_NULL_KW_PLACEHOLDER = "\x00MS_NULL_KW\x00"


def apply_surface(sql: str, surface: SurfaceSpec) -> str:
    """Rewrite a reference-surface SQL string into the spec's surface form.

    The rewriter operates on a small, fixed canonical corpus so its job is
    narrow: it does not need to handle SQL comments because none of the
    canonical queries contain them. String-literal and identifier-literal
    rewrites are guarded so they only fire on the spec's quote characters.
    """
    out = sql
    out = _rewrite_multi_word(out, surface)
    out = _rewrite_is_null(out, surface)
    out = _rewrite_single_keywords(out, surface)
    out = _rewrite_null_literal(out, surface)
    out = out.replace(_NULL_KW_PLACEHOLDER, surface.null_keyword)
    out = _rewrite_limit(out, surface)
    out = _rewrite_function_aliases(out, surface)
    out = _rewrite_null_safe_eq(out, surface)
    out = _rewrite_operators(out, surface)
    out = _rewrite_wildcard(out, surface)
    out = _rewrite_string_quote(out, surface)
    out = _rewrite_identifier_quote(out, surface)
    if surface.requires_semicolon and surface.statement_terminator:
        out = f"{out.rstrip()}{surface.statement_terminator}"
    return out


def _rewrite_multi_word(text: str, surface: SurfaceSpec) -> str:
    out = text
    for ref, attr in _MULTI_WORD_REWRITES:
        replacement = getattr(surface, attr)
        if replacement.upper() == ref:
            continue
        pattern = r"\b" + r"\s+".join(re.escape(w) for w in ref.split()) + r"\b"
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return out


def _rewrite_is_null(text: str, surface: SurfaceSpec) -> str:
    """Rewrite IS NULL / IS NOT NULL using the spec's NULL keyword.

    Emits a sentinel for `null_keyword` so the literal-NULL rewrite that
    runs later can't clobber it. The placeholder is replaced back in
    `apply_surface` after `_rewrite_null_literal`.
    """
    is_kw = surface.is_keyword
    not_kw = surface.not_keyword
    out = re.sub(
        r"\bIS\s+NOT\s+NULL\b",
        f"{is_kw} {not_kw} {_NULL_KW_PLACEHOLDER}",
        text,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"\bIS\s+NULL\b",
        f"{is_kw} {_NULL_KW_PLACEHOLDER}",
        out,
        flags=re.IGNORECASE,
    )
    return out


def _rewrite_single_keywords(text: str, surface: SurfaceSpec) -> str:
    out = text
    for token, attr in _SINGLE_KEYWORD_ATTRS:
        replacement = getattr(surface, attr)
        if replacement.upper() == token:
            continue
        out = re.sub(
            rf"\b{re.escape(token)}\b",
            replacement,
            out,
            flags=re.IGNORECASE,
        )
    # IS / NOT / IN are used inside multi-word constructs, so rewrite them last.
    for token, attr in [
        ("IS", "is_keyword"),
        ("NOT", "not_keyword"),
        ("IN", "in_keyword"),
    ]:
        replacement = getattr(surface, attr)
        if replacement.upper() == token:
            continue
        out = re.sub(
            rf"\b{re.escape(token)}\b",
            replacement,
            out,
            flags=re.IGNORECASE,
        )
    return out


def _rewrite_null_literal(text: str, surface: SurfaceSpec) -> str:
    """Rewrite remaining NULL occurrences as null literals (post IS NULL)."""
    if surface.null_literal == NullLiteral.NULL:
        return text
    return re.sub(r"\bNULL\b", surface.null_literal.value, text)


def _rewrite_function_aliases(text: str, surface: SurfaceSpec) -> str:
    """Rewrite reference function names to the spec's *primary* alias.

    For each canonical function (e.g. ``COALESCE``, ``LENGTH``) the spec may
    list one or more accepted spellings; the first entry is the primary
    surface form the dialect actually emits. We rewrite the reference SQL
    to use the primary so the parse battery exercises the dialect's
    advertised name. Only function-call sites (``NAME(``) are rewritten so
    we don't accidentally clobber matching identifiers.
    """
    if not surface.function_aliases:
        return text
    out = text
    for canonical, names in surface.function_aliases.items():
        if not names:
            continue
        primary = names[0]
        if primary.upper() == canonical.upper():
            continue
        pattern = re.compile(
            rf"\b{re.escape(canonical)}\s*\(",
            flags=re.IGNORECASE,
        )
        out = pattern.sub(f"{primary}(", out)
    return out


def _rewrite_null_safe_eq(text: str, surface: SurfaceSpec) -> str:
    """Rewrite ``IS NOT DISTINCT FROM`` to the spec's null-safe-eq spelling."""
    nse = surface.null_safe_eq_op
    if not nse or nse.upper() == "IS NOT DISTINCT FROM":
        return text
    return re.sub(
        r"\bIS\s+NOT\s+DISTINCT\s+FROM\b",
        nse,
        text,
        flags=re.IGNORECASE,
    )


# Reference-grammar operator literals, keyed by SurfaceSpec attribute name.
# Order matters: longer-prefix tokens (``<=``) first so ``<`` doesn't eat the
# leading character of ``<=`` during regex substitution.
_REF_OPERATORS: list[tuple[str, str]] = [
    # (reference token, surface_attr_name)
    ("<=", "lte_op"),
    (">=", "gte_op"),
    ("<", "lt_op"),
    (">", "gt_op"),
    ("=", "eq_op"),
    # add_tail / mul_tail
    ("||", "concat_op"),
    ("+", "add_op"),
    ("-", "sub_op"),
    ("*", "mul_op"),
    ("/", "div_op"),
    ("%", "mod_op"),
]


def _rewrite_operators(text: str, surface: SurfaceSpec) -> str:
    """Rewrite reference operator characters to the spec's operators.

    The reference SQL corpus uses ``=``/``<>``/``<=``/``+``/``-``/``*``/``/``/
    ``%``/``||``. For each operator we substitute the spec's value when it
    differs. ``neq_op`` is a list; we rewrite ``<>`` to the spec's first
    entry. String literals are protected from rewrites (the corpus has
    ``'A%'`` which would otherwise be clobbered by the mod_op rewrite).

    When ``concat_op`` overloads ``add_op`` / ``sub_op`` (e.g. SQL Server's
    ``+``-as-concat) the grammar emitter keeps ``||`` alive as a backup
    spelling so the test SQL doesn't become ambiguous; we mirror that
    decision here by NOT rewriting ``||`` in that case.
    """
    out = text
    placeholders: dict[str, str] = {}
    out = _replace_strings_with_placeholders(out, placeholders)

    if list(surface.neq_op) != ["<>", "!="]:
        primary_neq = surface.neq_op[0] if surface.neq_op else "<>"
        out = out.replace("<>", primary_neq)

    concat_overloads_arith = surface.concat_op in (
        surface.add_op,
        surface.sub_op,
    )
    for ref_token, attr in _REF_OPERATORS:
        replacement = getattr(surface, attr)
        if replacement == ref_token:
            continue
        if attr == "concat_op" and concat_overloads_arith:
            # Grammar still accepts ``||``; keep the canonical form rather
            # than introduce a syntactically ambiguous overload.
            continue
        # Multi-character tokens are unique enough to substitute via
        # `str.replace`. Single-character tokens (`+`/`-`/`*`/`/`/`%`/`=`)
        # could appear inside identifiers or numbers, but the canonical
        # corpus uses them only as operators flanked by spaces or operands.
        if len(ref_token) > 1:
            out = out.replace(ref_token, replacement)
        else:
            out = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(ref_token)}(?![A-Za-z0-9_])",
                replacement,
                out,
            )
    out = _restore_placeholders(out, placeholders)
    return out


def _rewrite_wildcard(text: str, surface: SurfaceSpec) -> str:
    """Rewrite ``SELECT *`` and ``COUNT(*)`` for non-star wildcards.

    Acts on the same three sites the deterministic grammar emitter patches.
    Only fires when the spec asks for a non-star wildcard so default specs
    are unaffected.
    """
    if surface.wildcard_char == WildcardChar.STAR:
        return text
    char = {
        WildcardChar.DOT: ".",
        WildcardChar.AT: "@",
    }[surface.wildcard_char]
    out = text
    # SELECT *  /  COUNT(*) — both are isolated * tokens.
    out = re.sub(r"(?<=SELECT)(\s+)\*", rf"\1{char}", out, flags=re.IGNORECASE)
    out = re.sub(r"(?<=\()\s*\*\s*(?=\))", char, out)
    # qualified wildcard: e.g. employees.*
    out = re.sub(r"(\.)\*", rf"\g<1>{char}", out)
    return out


def _rewrite_string_quote(text: str, surface: SurfaceSpec) -> str:
    """Rewrite single-quoted reference strings to the spec's quote char."""
    if surface.string_quote == StringQuote.SINGLE:
        return text
    new_q = {
        StringQuote.DOUBLE: '"',
        StringQuote.BACKTICK: "`",
    }[surface.string_quote]
    out: list[str] = []
    inside = False
    for ch in text:
        if ch == "'":
            out.append(new_q)
            inside = not inside
        else:
            out.append(ch)
    return "".join(out)


def _rewrite_identifier_quote(text: str, surface: SurfaceSpec) -> str:
    """The canonical corpus has no quoted identifiers, so this is currently
    a no-op for default battery items. Axis-targeted items that DO contain
    quoted identifiers are emitted with `"ident"` placeholders and rewritten
    here when the spec changes the identifier quote character.
    """
    if surface.identifier_quote == IdentifierQuote.DOUBLE:
        return text
    chars = {
        IdentifierQuote.BACKTICK: ("`", "`"),
        IdentifierQuote.BRACKET: ("[", "]"),
    }[surface.identifier_quote]
    open_q, close_q = chars
    # Only rewrite occurrences of the canonical double-quoted identifier
    # placeholder, never bare `"` characters that may appear inside string
    # literals (those have already been handled by _rewrite_string_quote).
    return re.sub(
        r'"([A-Za-z_][A-Za-z0-9_]*)"',
        lambda m: f"{open_q}{m.group(1)}{close_q}",
        text,
    )


_STRING_PLACEHOLDER = "\x00MS_STR_{i}\x00"


def _replace_strings_with_placeholders(
    text: str, placeholders: dict[str, str]
) -> str:
    """Hide string literals so operator/wildcard rewrites don't touch them.

    The canonical corpus uses single-quoted strings (the reference quote
    style). Quotes are doubled per ANSI ``''`` escaping rules.
    """
    def repl(match: re.Match[str]) -> str:
        idx = len(placeholders)
        key = _STRING_PLACEHOLDER.format(i=idx)
        placeholders[key] = match.group(0)
        return key

    return re.sub(r"'([^'\\]|\\.|'')*'", repl, text)


def _restore_placeholders(text: str, placeholders: dict[str, str]) -> str:
    out = text
    for key, original in placeholders.items():
        out = out.replace(key, original)
    return out


def _rewrite_limit(text: str, surface: SurfaceSpec) -> str:
    """Rewrite the LIMIT/OFFSET tail per spec.

    Only rewrites when the spec changes the syntax. Assumes the canonical
    input uses the reference form `LIMIT n [OFFSET m]`.
    """
    if surface.limit_syntax == LimitSyntax.LIMIT_OFFSET:
        return text
    pattern = re.compile(
        rf"\b{re.escape(surface.limit_keyword)}\s+(\d+)(?:\s+OFFSET\s+(\d+))?\s*$",
        flags=re.IGNORECASE,
    )

    def repl(m: re.Match[str]) -> str:
        n = m.group(1)
        offset = m.group(2)
        return _format_limit_clause(surface.limit_syntax, n=n, offset=offset)

    return pattern.sub(repl, text)


def _format_limit_clause(
    syntax: LimitSyntax, *, n: str, offset: Optional[str]
) -> str:
    if syntax == LimitSyntax.OFFSET_FETCH:
        off = offset or "0"
        return f"OFFSET {off} ROWS FETCH NEXT {n} ROWS ONLY"
    if syntax == LimitSyntax.TOP_N:
        return f"TOP {n}"  # caller is responsible for moving this to SELECT
    if syntax == LimitSyntax.SAMPLE_N:
        return f"SAMPLE {n}"
    if syntax == LimitSyntax.HEAD_N:
        return f"| HEAD {n}"
    return ""  # pragma: no cover


__all__ = [
    "BatteryFailure",
    "BatteryItem",
    "ValidationReport",
    "apply_surface",
    "build_parse_battery",
    "validate_grammar",
]
