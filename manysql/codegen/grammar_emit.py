"""Emit grammar.lark for a generated dialect.

Two modes:
- Deterministic (default): take the reference grammar, swap surface keywords
  and operators per the spec. This is mechanical and produces parseable Lark
  out of the box for any DialectSpec whose changes are within the supported
  knobs (renames, alternative quote chars, alternative comment styles, etc.).
- LLM-augmented: optional. The deterministic baseline is the prompt's
  starting point and the LLM is asked to refine specific clauses (e.g. for
  `JoinSyntax.PIPELINED` where mechanical rewriting isn't enough).

The deterministic path is enough to bootstrap end-to-end synthetic dialects
for the codegen pipeline.
"""

from __future__ import annotations

import re
from importlib import resources
from typing import Optional

from manysql.spec.dialect import (
    CommentStyle,
    DialectSpec,
    IdentifierQuote,
    LimitSyntax,
    NullLiteral,
    StringQuote,
    SurfaceSpec,
)

# These are the (uppercase) reference keyword tokens we know how to rewrite
# safely without disturbing the rest of the grammar. Multi-word keywords are
# expressed as their on-grammar form (e.g. `"GROUP"i "BY"i` -> see _replace).
_KEYWORD_REWRITES: list[tuple[str, str]] = [
    # Each entry: (reference token literal, surface attribute on SurfaceSpec)
    # Single-token keywords first; multi-token handled separately.
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
    ("IS", "is_keyword"),
    ("NOT", "not_keyword"),
    ("NULL", "null_keyword"),
    ("IN", "in_keyword"),
    ("BETWEEN", "between_keyword"),
    ("LIKE", "like_keyword"),
    ("ILIKE", "ilike_keyword"),
    ("EXISTS", "exists_keyword"),
    ("LIMIT", "limit_keyword"),
]


def emit_grammar(spec: DialectSpec) -> str:
    """Return the Lark grammar text for the dialect."""
    base = _read_reference_grammar()
    out = _apply_keyword_rewrites(base, spec.surface)
    out = _apply_multi_word_rewrites(out, spec.surface)
    out = _apply_string_quote(out, spec.surface)
    out = _apply_identifier_quote(out, spec.surface)
    out = _apply_comment_styles(out, spec.surface)
    out = _apply_null_literal(out, spec.surface)
    out = _apply_limit_syntax(out, spec.surface)
    out = _apply_ilike_disabled(out, spec)
    return out


def _read_reference_grammar() -> str:
    return resources.read_text(
        "manysql.dialects._reference", "grammar.lark", encoding="utf-8"
    )


def _apply_keyword_rewrites(text: str, surface: SurfaceSpec) -> str:
    out = text
    for token, attr in _KEYWORD_REWRITES:
        replacement = getattr(surface, attr)
        if replacement.upper() == token:
            continue
        # Only replace tokens inside `"...".i` quoted forms (so we don't
        # accidentally clobber rule names or comments).
        pattern = rf'"{token}"i'
        target_words = replacement.split()
        if len(target_words) == 1:
            out = re.sub(pattern, f'"{target_words[0]}"i', out)
        else:
            spaced = " ".join(f'"{w}"i' for w in target_words)
            out = re.sub(pattern, spaced, out)
    return out


def _apply_multi_word_rewrites(text: str, surface: SurfaceSpec) -> str:
    """Handle multi-word reference keywords (`GROUP BY`, `ORDER BY`, `LEFT JOIN` ...).

    For each pattern we only rewrite when the spec's keyword diverges from the
    reference default; otherwise the original Lark fragment (which may include
    optional sub-keywords like `OUTER`) is preserved verbatim.
    """
    out = text
    rewrites: list[tuple[str, str, str]] = [
        ('"GROUP"i "BY"i', "GROUP BY", surface.group_by_keyword),
        ('"ORDER"i "BY"i', "ORDER BY", surface.order_by_keyword),
        (
            '"LEFT"i ("OUTER"i)? "JOIN"i',
            "LEFT JOIN",
            surface.join_left_keyword,
        ),
        (
            '"RIGHT"i ("OUTER"i)? "JOIN"i',
            "RIGHT JOIN",
            surface.join_right_keyword,
        ),
        (
            '"FULL"i ("OUTER"i)? "JOIN"i',
            "FULL JOIN",
            surface.join_full_keyword,
        ),
        ('"CROSS"i "JOIN"i', "CROSS JOIN", surface.join_cross_keyword),
        ('"INNER"i? "JOIN"i', "JOIN", surface.join_inner_keyword),
    ]
    for pattern, default_text, surface_text in rewrites:
        if surface_text.upper().strip() == default_text:
            continue
        out = out.replace(pattern, _format_keyword(surface_text))
    out = out.replace(
        '"NULLS"i ("FIRST"i | "LAST"i)',
        _nulls_clause(surface),
    )
    return out


def _nulls_clause(surface: SurfaceSpec) -> str:
    first = surface.nulls_first_keyword
    last = surface.nulls_last_keyword
    if first.upper() == "NULLS FIRST" and last.upper() == "NULLS LAST":
        return '"NULLS"i ("FIRST"i | "LAST"i)'
    return f"({_format_keyword(first)} | {_format_keyword(last)})"


def _format_keyword(text: str) -> str:
    parts = text.strip().split()
    return " ".join(f'"{p}"i' for p in parts)


def _apply_string_quote(text: str, surface: SurfaceSpec) -> str:
    if surface.string_quote == StringQuote.SINGLE:
        return text
    char = {
        StringQuote.DOUBLE: '"',
        StringQuote.BACKTICK: "`",
    }[surface.string_quote]
    return text.replace(
        "STRING: /'([^'\\\\]|\\\\.|'')*'/",
        f"STRING: /{char}([^{char}\\\\]|\\\\.|{char}{char})*{char}/",
    )


def _apply_identifier_quote(text: str, surface: SurfaceSpec) -> str:
    if surface.identifier_quote == IdentifierQuote.DOUBLE:
        return text
    quote_chars = {
        IdentifierQuote.BACKTICK: ("`", "`"),
        IdentifierQuote.BRACKET: (r"\[", r"\]"),
    }[surface.identifier_quote]
    open_q, close_q = quote_chars
    insertion = (
        f"\nQUOTED_IDENTIFIER: /{open_q}[^{open_q}{close_q}]+{close_q}/\n"
    )
    if "QUOTED_IDENTIFIER" not in text:
        text += insertion
    return text


def _apply_comment_styles(text: str, surface: SurfaceSpec) -> str:
    """Append %ignore patterns for any non-default comment styles requested."""
    extras: list[str] = []
    requested = set(surface.comment_styles)
    if CommentStyle.LINE_HASH in requested:
        extras.append("%ignore /#[^\\n]*/")
    if CommentStyle.LINE_DOUBLE_SLASH in requested:
        extras.append("%ignore /\\/\\/[^\\n]*/")
    if not extras:
        return text
    return text + "\n" + "\n".join(extras) + "\n"


def _apply_null_literal(text: str, surface: SurfaceSpec) -> str:
    if surface.null_literal == NullLiteral.NULL:
        return text
    replacement = surface.null_literal.value
    return text.replace('"NULL"i        -> null_literal', f'"{replacement}"i        -> null_literal')


def _apply_limit_syntax(text: str, surface: SurfaceSpec) -> str:
    if surface.limit_syntax == LimitSyntax.LIMIT_OFFSET:
        return text
    if surface.limit_syntax == LimitSyntax.OFFSET_FETCH:
        # OFFSET m ROWS FETCH NEXT n ROWS ONLY
        new_rule = (
            'limit_clause: "OFFSET"i INT "ROWS"i "FETCH"i "NEXT"i INT "ROWS"i "ONLY"i'
        )
    elif surface.limit_syntax == LimitSyntax.TOP_N:
        # TOP attaches inside SELECT — emitting a separate rule isn't enough; we
        # punt to a no-op rule and rely on the LLM lane for true rewrites.
        new_rule = 'limit_clause: "TOP"i INT'
    elif surface.limit_syntax == LimitSyntax.SAMPLE_N:
        new_rule = 'limit_clause: "SAMPLE"i INT'
    elif surface.limit_syntax == LimitSyntax.HEAD_N:
        new_rule = 'limit_clause: "|" "HEAD"i INT'
    else:  # pragma: no cover - exhaustive
        return text
    return re.sub(
        r"limit_clause:.*",
        new_rule,
        text,
        count=1,
    )


def _apply_ilike_disabled(text: str, spec: DialectSpec) -> str:
    if spec.semantics.ilike_supported is False:
        # Strip the ILIKE branch from the comparison rule.
        text = re.sub(
            r'\s*\|\s*"ILIKE"i additive\s*->\s*ilike_op',
            "",
            text,
        )
    return text


def emit_grammar_with_llm(
    spec: DialectSpec,
    *,
    llm_client: Optional[object],
    extra_directives: Optional[str] = None,
) -> str:  # pragma: no cover - exercised in higher-level integration tests
    """LLM-refined grammar emission. Stub for the codegen refine loop.

    Returns the deterministic baseline if no LLM client is provided. When a
    client is supplied, the codegen pipeline is expected to:
      1. Render the deterministic baseline.
      2. Send it + the spec + parse-battery failures (if any) to the LLM.
      3. Validate the result by parsing the dialect-correctness battery.
      4. Iterate up to N times.

    This stub exists so the public emitter API surface is stable; the real
    refine loop lives in the higher-level `pipeline` module.
    """
    return emit_grammar(spec)
