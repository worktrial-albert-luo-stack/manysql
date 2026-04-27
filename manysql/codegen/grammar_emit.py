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
    WildcardChar,
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
    out = _apply_operator_rewrites(out, spec.surface)
    out = _apply_wildcard_char(out, spec.surface)
    out = _apply_requires_semicolon(out, spec.surface)
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
    """Make ``IDENTIFIER`` accept the spec's quoted-identifier form.

    The reference grammar's ``IDENTIFIER`` matches barewords plus ANSI
    ``"quoted"`` form. For dialects with a non-default identifier quote we
    *swap* the quoted form for the spec's spelling (backtick or bracket),
    so every existing rule that already references ``IDENTIFIER`` (column
    refs, table names, aliases, CTE bindings, ...) silently picks up the
    new surface form. The lowering already strips surrounding quotes.
    """
    if surface.identifier_quote == IdentifierQuote.DOUBLE:
        return text
    quote_pat = {
        IdentifierQuote.BACKTICK: r"`[^`]+`",
        IdentifierQuote.BRACKET: r"\[[^\[\]]+\]",
    }[surface.identifier_quote]
    new_terminal = f"IDENTIFIER: /[A-Za-z_][A-Za-z0-9_]*|{quote_pat}/"
    return re.sub(
        r"^IDENTIFIER:.*$",
        new_terminal,
        text,
        count=1,
        flags=re.MULTILINE,
    )


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


# --------------------------------------------------------------------------
# Operator rewrites
# --------------------------------------------------------------------------

# Defaults match the reference grammar exactly. Every per-axis rewrite is
# guarded against the default so existing dialects (whose specs leave these
# at the reference values) emit the original Lark fragments verbatim.
_REF_NEQ_OPS: tuple[str, ...] = ("<>", "!=")


def _apply_operator_rewrites(text: str, surface: SurfaceSpec) -> str:
    """Patch comparison/arithmetic/concat operators when the spec diverges.

    The reference grammar packs every comparison operator into a single
    ``comp_op`` rule and every additive/multiplicative operator into a
    single ``add_tail`` / ``mul_tail`` rule. When *any* operator on a given
    rule diverges, we rebuild the whole rule from the spec's values; this
    way obsolete spellings (e.g. ``<>`` when the spec only allows ``!``) are
    actively dropped, which lets the rejection battery prove the grammar
    refuses the old form.

    The ``null_safe_eq_op`` lives on its own ``comp_tail -> not_distinct_from``
    branch and is patched in place.
    """
    text = _patch_comp_op(text, surface)
    text = _patch_add_tail(text, surface)
    text = _patch_mul_tail(text, surface)
    text = _patch_null_safe_eq(text, surface)
    return text


def _patch_comp_op(text: str, surface: SurfaceSpec) -> str:
    eq = surface.eq_op
    neq_list = list(surface.neq_op)
    lt = surface.lt_op
    lte = surface.lte_op
    gt = surface.gt_op
    gte = surface.gte_op
    ref_eq, ref_lt, ref_lte, ref_gt, ref_gte = "=", "<", "<=", ">", ">="
    if (
        eq == ref_eq
        and tuple(neq_list) == _REF_NEQ_OPS
        and lt == ref_lt
        and lte == ref_lte
        and gt == ref_gt
        and gte == ref_gte
    ):
        return text
    branches: list[str] = [f'{_lit(eq)} -> eq']
    for op in neq_list:
        branches.append(f'{_lit(op)} -> neq')
    # Order longer-prefix tokens first so Lark's tokenizer doesn't confuse
    # `<` with `<=`. The reference grammar already has this order.
    branches.append(f'{_lit(lte)} -> lte')
    branches.append(f'{_lit(lt)} -> lt')
    branches.append(f'{_lit(gte)} -> gte')
    branches.append(f'{_lit(gt)} -> gt')
    new_rule = "comp_op: " + " | ".join(branches)
    return re.sub(r"^comp_op:.*$", new_rule, text, count=1, flags=re.MULTILINE)


def _patch_add_tail(text: str, surface: SurfaceSpec) -> str:
    add_op = surface.add_op
    sub_op = surface.sub_op
    concat_op = surface.concat_op
    if add_op == "+" and sub_op == "-" and concat_op == "||":
        return text
    branches = [
        f'{_lit(add_op)} multiplicative -> add',
        f'{_lit(sub_op)} multiplicative -> sub',
    ]
    # If concat overloads + (SQL Server style), the dialect can't
    # syntactically disambiguate concat from add. Keep the ANSI ``||``
    # branch alive so concat remains expressible (and the parse battery's
    # ``project_concat`` query parses identically against both reference
    # and dialect). The dialect card still advertises `+` as the primary
    # concat surface form.
    if concat_op == add_op or concat_op == sub_op:
        branches.append('"||" multiplicative -> concat')
    elif concat_op != "||":
        branches.append(f'{_lit(concat_op)} multiplicative -> concat')
    else:
        branches.append('"||" multiplicative -> concat')
    new_rule = "?add_tail: " + " | ".join(branches)
    return re.sub(
        r"^\?add_tail:.*$", new_rule, text, count=1, flags=re.MULTILINE
    )


def _patch_mul_tail(text: str, surface: SurfaceSpec) -> str:
    mul_op = surface.mul_op
    div_op = surface.div_op
    mod_op = surface.mod_op
    if mul_op == "*" and div_op == "/" and mod_op == "%":
        return text
    new_rule = (
        f'?mul_tail: {_lit(mul_op)} unary -> mul'
        f' | {_lit(div_op)} unary -> div'
        f' | {_lit(mod_op)} unary -> mod'
    )
    return re.sub(
        r"^\?mul_tail:.*$", new_rule, text, count=1, flags=re.MULTILINE
    )


def _patch_null_safe_eq(text: str, surface: SurfaceSpec) -> str:
    """Replace the IS NOT DISTINCT FROM branch with the spec's spelling.

    When ``null_safe_eq_op`` is None we strip the branch entirely so the
    grammar rejects the construct (the dialect doesn't support null-safe
    equality at all). When the spelling differs from the default ANSI form,
    we replace the literal token sequence so the grammar accepts only the
    new form.
    """
    nse = surface.null_safe_eq_op
    default = '"IS"i "NOT"i "DISTINCT"i "FROM"i additive -> not_distinct_from'
    if nse is None:
        return re.sub(
            r'\s*\|\s*"IS"i "NOT"i "DISTINCT"i "FROM"i additive\s*->\s*not_distinct_from',
            "",
            text,
        )
    if nse.upper() == "IS NOT DISTINCT FROM":
        return text
    new = f'{_format_null_safe_op(nse)} additive -> not_distinct_from'
    return text.replace(default, new)


def _format_null_safe_op(op: str) -> str:
    """Render a multi-word or symbolic null-safe-eq operator as Lark literals."""
    parts = op.strip().split()
    if len(parts) > 1 and all(part.isalpha() for part in parts):
        return " ".join(f'"{p.upper()}"i' for p in parts)
    return _lit(op)


def _lit(token: str) -> str:
    """Lark literal: keyword-y tokens become case-insensitive, symbols don't."""
    stripped = token.strip()
    if stripped and stripped[0].isalpha():
        return f'"{stripped}"i'
    return f'"{stripped}"'


# --------------------------------------------------------------------------
# Wildcard character
# --------------------------------------------------------------------------

_WILDCARD_LITERAL: dict[WildcardChar, str] = {
    WildcardChar.STAR: "*",
    WildcardChar.DOT: ".",
    WildcardChar.AT: "@",
}


def _apply_wildcard_char(text: str, surface: SurfaceSpec) -> str:
    """Replace ``*`` in the three select/wildcard-bearing rules.

    The reference grammar uses literal ``"*"`` in three places:
    ``select_item -> star``, ``qualified_star``, and ``func_args -> star_args``.
    Other ``"*"`` occurrences are inside terminals (e.g. inside the comment
    ``%ignore`` regex) and must NOT be touched.
    """
    char = surface.wildcard_char
    if char == WildcardChar.STAR:
        return text
    new_lit = _WILDCARD_LITERAL[char]
    out = text
    out = out.replace(
        '"*"                              -> star',
        f'"{new_lit}"                              -> star',
        1,
    )
    out = out.replace(
        'IDENTIFIER "." "*"               -> qualified_star',
        f'IDENTIFIER "." "{new_lit}"               -> qualified_star',
        1,
    )
    out = out.replace(
        '"*"                              -> star_args',
        f'"{new_lit}"                              -> star_args',
        1,
    )
    return out


# --------------------------------------------------------------------------
# Mandatory statement terminator
# --------------------------------------------------------------------------


def _apply_requires_semicolon(text: str, surface: SurfaceSpec) -> str:
    """Make the statement terminator mandatory when the spec demands it.

    The reference grammar's top-level ``statement`` rule does not require a
    terminator. When ``requires_semicolon=True`` we append the spec's
    ``statement_terminator`` (``;`` by default) as a non-optional literal,
    so the grammar actively rejects un-terminated SQL — which the rejection
    battery uses to prove the dialect's strictness.
    """
    if not surface.requires_semicolon or not surface.statement_terminator:
        return text
    terminator = surface.statement_terminator
    new_rule = f'statement: with_clause? query_expr "{terminator}"'
    return re.sub(
        r"^statement:\s*with_clause\?\s*query_expr.*$",
        new_rule,
        text,
        count=1,
        flags=re.MULTILINE,
    )


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
