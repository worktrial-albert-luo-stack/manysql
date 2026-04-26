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
from typing import Optional

from lark import Lark
from lark.exceptions import LarkError

from manysql.spec.dialect import (
    DialectSpec,
    LimitSyntax,
    NullLiteral,
    SurfaceSpec,
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


def build_parse_battery(spec: DialectSpec) -> list[BatteryItem]:
    """Return one BatteryItem per canonical SQL, rewritten into the spec's surface."""
    items: list[BatteryItem] = []
    for label, sql in _REFERENCE_SQL:
        items.append(
            BatteryItem(label=label, source=apply_surface(sql, spec.surface))
        )
    return items


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
    narrow: it does not need to handle string literals or comments because
    none of the canonical queries contain SQL text that would alias with
    keywords. Word-boundary regex is sufficient.
    """
    out = sql
    out = _rewrite_multi_word(out, surface)
    out = _rewrite_is_null(out, surface)
    out = _rewrite_single_keywords(out, surface)
    out = _rewrite_null_literal(out, surface)
    out = out.replace(_NULL_KW_PLACEHOLDER, surface.null_keyword)
    out = _rewrite_limit(out, surface)
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
