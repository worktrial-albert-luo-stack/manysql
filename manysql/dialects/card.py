"""Dialect "card" rendering.

Composes the LLM-facing description of a generated dialect: a diff against
the near-ANSI reference, concrete syntactic patterns, function aliases,
semantic divergences, and (when present) the codegen-emitted worked
examples. The result is a block of SQL-style ``--`` comments suitable for
splicing into a system prompt.

Used by:
- ``eval/executors/synthetic_executor.py`` (LLM benchmark surface).
- ``train/env/engine.py`` (RL environment surface).

Both consumers want exactly the same prompt for the same dialect, so the
renderer lives here in the dialects package next to the registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from manysql.dialects.registry import DialectEngine


def render_dialect_card(engine: DialectEngine) -> str:
    """Compose the 'how to write SQL in this dialect' card.

    The card is a diff against the reference dialect (near-ANSI SQL):
    every surface or semantic field that differs from the reference's
    default is enumerated; anything not mentioned matches the reference.

    The codegen agents see the same DialectSpec when they generate the
    grammar/lowering, so this gives the LLM the same shape information
    the dialect itself was built from -- a fair fight rather than a
    guessing game off a one-line description.
    """
    from manysql.spec.dialect import SurfaceSpec  # noqa: PLC0415

    spec_dict = engine.spec or {}
    name = engine.name
    surface = spec_dict.get("surface", {}) or {}
    semantics = spec_dict.get("semantics", {}) or {}

    ref_surface_defaults = SurfaceSpec().model_dump()

    surface_diffs: dict[str, Any] = {}
    for key, value in surface.items():
        if key == "function_aliases":
            continue
        ref_value = ref_surface_defaults.get(key)
        if value == ref_value:
            continue
        surface_diffs[key] = value

    semantic_diffs = {k: v for k, v in semantics.items() if v is not None}

    lines: list[str] = []
    lines.append(f"-- Target dialect: manysql synthetic '{name}'.")
    lines.append(
        "-- Anything below is a divergence from the reference baseline; "
        "anything not listed here matches near-ANSI SQL:"
    )
    lines.append(
        "--   CAST(x AS T)  |  'string lit'  |  \"ident\"  |  LIMIT n OFFSET m"
    )
    lines.append(
        "--   || concat  |  ANSI joins  |  CASE WHEN ... END  |  -- and /* */ comments"
    )
    lines.append("--   NULL literal  |  * wildcard  |  ANSI keywords (SELECT/FROM/...).")
    lines.append("")

    if spec_dict.get("description"):
        lines.append(f"Description: {spec_dict['description']}")
    if spec_dict.get("divergence_level"):
        lines.append(f"Divergence level: {spec_dict['divergence_level']}")
    if spec_dict.get("inspired_by"):
        lines.append(f"Inspired by: {', '.join(spec_dict['inspired_by'])}")
    if spec_dict.get("notes"):
        lines.append(f"Notes: {spec_dict['notes']}")

    if surface_diffs:
        lines.append("")
        lines.append("Surface divergences (lexical + syntactic):")
        for group_label, fields in _SURFACE_FIELD_GROUPS:
            group_lines = [
                f"  {field} = {_pretty(surface_diffs[field])}"
                for field in fields
                if field in surface_diffs
            ]
            if group_lines:
                lines.append(f"  # {group_label}")
                lines.extend(group_lines)
        rendered_keys = {f for _, fields in _SURFACE_FIELD_GROUPS for f in fields}
        leftover = sorted(set(surface_diffs) - rendered_keys)
        if leftover:
            lines.append("  # other")
            for field in leftover:
                lines.append(f"  {field} = {_pretty(surface_diffs[field])}")

    patterns = _canonical_patterns(surface)
    if patterns:
        lines.append("")
        lines.append("Canonical patterns in this dialect (use these forms verbatim):")
        for label, snippet in patterns:
            lines.append(f"  {label}: {snippet}")

    aliases = surface.get("function_aliases") or {}
    if aliases:
        lines.append("")
        lines.append("Function aliases (canonical name -> accepted spellings; first is primary):")
        for canonical, names in aliases.items():
            lines.append(f"  {canonical} -> {', '.join(names)}")

    if semantic_diffs:
        lines.append("")
        lines.append("Semantic divergences (runtime behavior):")
        for key in sorted(semantic_diffs):
            lines.append(f"  {key} = {_pretty(semantic_diffs[key])}")

    lines.append("")
    lines.append("Supported functions (inherited from the manysql IR executor):")
    lines.append("  Aggregates : COUNT, SUM, AVG, MIN, MAX (plus any aliases listed above).")
    lines.append("  Scalars    : COALESCE, NULLIF, CAST, ABS, ROUND, LENGTH, LOWER, UPPER,")
    lines.append("               TRIM, SUBSTR, REPLACE, CONCAT (when the concat op is a fn).")
    lines.append("  Datetimes  : columns are ISO-8601 TEXT; use SUBSTR(col, 1, 7) for")
    lines.append("               'YYYY-MM', SUBSTR(col, 1, 4) for 'YYYY'. strftime is NOT")
    lines.append("               guaranteed in synthetic dialects -- prefer SUBSTR.")
    lines.append("  Avoid      : engine-specific functions (toStartOfMonth, splitByChar,")
    lines.append("               array_agg, group_concat, julianday, etc.) -- the dialect")
    lines.append("               only knows what's in its grammar + overrides.")

    example_lines = _maybe_examples_lines(engine)
    if example_lines:
        lines.append("")
        lines.extend(example_lines)

    lines.append("")
    lines.append(
        "IMPORTANT: the dialect's grammar is strict. Use ONLY surface forms "
        "covered above (or by the reference baseline)."
    )

    return "\n".join("-- " + ln if ln and not ln.startswith("--") else ln for ln in lines)


# Group surface divergences into related buckets so the LLM can scan them
# quickly without parsing dozens of unstructured ``foo = bar`` lines.
_SURFACE_FIELD_GROUPS: list[tuple[str, list[str]]] = [
    (
        "lexical",
        [
            "string_quote",
            "identifier_quote",
            "comment_styles",
            "null_literal",
            "wildcard_char",
        ],
    ),
    (
        "keywords",
        [
            "select_keyword",
            "from_keyword",
            "where_keyword",
            "group_by_keyword",
            "having_keyword",
            "order_by_keyword",
            "limit_keyword",
            "distinct_keyword",
            "union_keyword",
            "intersect_keyword",
            "except_keyword",
            "with_keyword",
            "as_keyword",
            "join_inner_keyword",
            "join_left_keyword",
            "join_right_keyword",
            "join_full_keyword",
            "join_cross_keyword",
            "case_keyword",
            "when_keyword",
            "then_keyword",
            "else_keyword",
            "end_keyword",
            "cast_keyword",
            "is_keyword",
            "not_keyword",
            "null_keyword",
            "in_keyword",
            "between_keyword",
            "like_keyword",
            "ilike_keyword",
            "exists_keyword",
            "nulls_first_keyword",
            "nulls_last_keyword",
        ],
    ),
    (
        "operators",
        [
            "eq_op",
            "neq_op",
            "lt_op",
            "lte_op",
            "gt_op",
            "gte_op",
            "add_op",
            "sub_op",
            "mul_op",
            "div_op",
            "mod_op",
            "concat_op",
            "null_safe_eq_op",
        ],
    ),
    (
        "structural",
        [
            "join_syntax",
            "order_by_position",
            "limit_syntax",
            "cast_syntax",
            "case_syntax",
        ],
    ),
    (
        "statement",
        ["requires_semicolon", "statement_terminator"],
    ),
]


def _pretty(value: Any) -> str:
    """Compact, prompt-friendly rendering of spec field values."""
    if isinstance(value, list):
        return "[" + ", ".join(_pretty(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_pretty(v)}" for k, v in value.items()) + "}"
    if isinstance(value, str):
        return value
    return repr(value)


def _canonical_patterns(surface: dict[str, Any]) -> list[tuple[str, str]]:
    """Render concrete one-line skeletons for the dialect's structural shapes.

    Reads the surface spec and produces literal token sequences for the
    handful of patterns that LLMs habitually botch when handed only the
    abstract spec field name (e.g. ``limit_syntax = offset_fetch`` is
    useless without the actual OFFSET/FETCH wording).
    """
    out: list[tuple[str, str]] = []

    sel = surface.get("select_keyword", "SELECT")
    frm = surface.get("from_keyword", "FROM")
    whr = surface.get("where_keyword", "WHERE")
    out.append(("SELECT skeleton", f"{sel} cols {frm} t {whr} pred"))

    gby = surface.get("group_by_keyword", "GROUP BY")
    hav = surface.get("having_keyword", "HAVING")
    out.append(("GROUP BY / HAVING", f"{gby} c {hav} agg(c) > k"))

    oby = surface.get("order_by_keyword", "ORDER BY")
    nfk = surface.get("nulls_first_keyword", "NULLS FIRST")
    nlk = surface.get("nulls_last_keyword", "NULLS LAST")
    out.append(("ORDER BY", f"{oby} c DESC {nlk}  (or ASC {nfk})"))

    limit_syntax = surface.get("limit_syntax", "limit_offset")
    if limit_syntax == "limit_offset":
        out.append(("LIMIT/OFFSET", "LIMIT 10 OFFSET 0"))
    elif limit_syntax == "offset_fetch":
        out.append(("LIMIT (offset_fetch)", "OFFSET 0 ROWS FETCH NEXT 10 ROWS ONLY"))
    elif limit_syntax == "top_n":
        out.append(("LIMIT (top_n)", "SELECT TOP 10 cols FROM t"))

    jin = surface.get("join_inner_keyword", "JOIN")
    jlt = surface.get("join_left_keyword", "LEFT JOIN")
    out.append(("INNER JOIN", f"a {jin} b ON a.k = b.k"))
    out.append(("LEFT JOIN", f"a {jlt} b ON a.k = b.k"))

    case = surface.get("case_keyword", "CASE")
    when = surface.get("when_keyword", "WHEN")
    then = surface.get("then_keyword", "THEN")
    elsk = surface.get("else_keyword", "ELSE")
    end = surface.get("end_keyword", "END")
    out.append(("CASE", f"{case} {when} pred {then} a {elsk} b {end}"))

    cast_syntax = surface.get("cast_syntax", "function")
    cast_kw = surface.get("cast_keyword", "CAST")
    if cast_syntax == "double_colon":
        out.append(("CAST (double_colon)", "x::INT"))
    else:
        out.append(("CAST (function)", f"{cast_kw}(x AS INT)"))

    isk = surface.get("is_keyword", "IS")
    nk = surface.get("null_keyword", "NULL")
    nnk = surface.get("not_keyword", "NOT")
    out.append(("IS NULL", f"x {isk} {nk}  /  x {isk} {nnk} {nk}"))

    null_lit = surface.get("null_literal", "NULL")
    if null_lit and null_lit != "NULL":
        out.append(("NULL literal", null_lit))

    in_kw = surface.get("in_keyword", "IN")
    out.append(("IN list", f"x {in_kw} (1, 2, 3)"))
    out.append(("IN subquery", f"x {in_kw} (SELECT k FROM t2)"))

    bw = surface.get("between_keyword", "BETWEEN")
    out.append(("BETWEEN", f"x {bw} a AND b"))

    like_kw = surface.get("like_keyword", "LIKE")
    out.append(("LIKE", f"name {like_kw} 'A%'"))
    if surface.get("ilike_keyword") and surface.get("ilike_keyword") != "ILIKE":
        out.append(("ILIKE", f"name {surface['ilike_keyword']} 'a%'"))

    concat_op = surface.get("concat_op", "||")
    out.append(("Concat", f"a {concat_op} b"))

    nse = surface.get("null_safe_eq_op")
    if nse:
        out.append(("Null-safe equal", f"a {nse} b"))

    wk = surface.get("with_keyword", "WITH")
    ask = surface.get("as_keyword", "AS")
    out.append(("CTE", f"{wk} cte {ask} (SELECT ...) SELECT * FROM cte"))

    union = surface.get("union_keyword", "UNION")
    out.append(("Set op", f"q1 {union} ALL q2"))

    return out


def _maybe_examples_lines(engine: DialectEngine) -> list[str]:
    """Return ``examples.sql`` as a list of lines for the dialect card.

    Codegen writes 26 hand-curated canonical queries into ``examples.sql``
    using the dialect's surface forms. Including a trimmed view of them
    in the prompt gives the LLM a Rosetta stone of working dialect SQL --
    the most direct way to teach it the dialect's idioms. Header chatter
    (the ``manysql-codegen``/``Re-generate`` blurb at the top of the
    file) is stripped so the prompt only contains the labeled queries.
    """
    from manysql.dialects.registry import DialectRegistry  # noqa: PLC0415

    examples_path = DialectRegistry().root / engine.name / "examples.sql"
    if not examples_path.exists():
        return []
    raw = examples_path.read_text().strip()
    if not raw:
        return []
    src_lines = raw.splitlines()
    i = 0
    while i < len(src_lines) and src_lines[i].lstrip().startswith("--"):
        i += 1
    while i < len(src_lines) and not src_lines[i].strip():
        i += 1
    body = src_lines[i:]
    if not body:
        return []
    out: list[str] = [
        "Worked examples in this dialect (hand-curated canonical queries; "
        "use these as a syntax reference):"
    ]
    for ln in body:
        out.append(f"  {ln}" if ln else "")
    return out


__all__ = ["render_dialect_card"]
