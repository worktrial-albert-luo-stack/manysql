"""Synthetic-dialect executor.

Loads a manysql-generated dialect from `manysql.dialects.<name>/` via the
`DialectRegistry`, builds a Lark parser from the dialect's grammar, lowers
parsed trees through the dialect's `lowering.lower` to manysql IR, and
executes the IR against an in-memory Polars catalog seeded with the same
synthetic GitHub-events corpus the SQLite backend uses.

Important caveat: the question suite's reference SQL is written in
SQLite syntax. When you eval against a dialect whose surface diverges
from SQLite (e.g. anything beyond the near-ANSI "mild" tier), pair the
synthetic backend with a separate reference executor (`SqliteExecutor`)
so ground truth is computed via SQLite while the LLM's SQL is judged
through the dialect engine. The runner accepts a `reference_executor`
argument exactly for this.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from eval.dataset.github_events import SCHEMA_PROMPT, seed_rows
from eval.executors.base import ExecResult, SqlExecutor

if TYPE_CHECKING:
    import polars as pl
    from lark import Lark

    from manysql.dialects.registry import DialectEngine
    from manysql.ir.plan import ColumnSchema


# Single-table catalog mirroring the SQLite DDL. Polars dtypes are chosen
# so the IR types we emit (TEXT/INT) match the executor's expectations.
_GITHUB_EVENTS_POLARS_DTYPES: dict[str, str] = {
    "file_time": "Utf8",
    "event_type": "Utf8",
    "actor_login": "Utf8",
    "repo_name": "Utf8",
    "created_at": "Utf8",
    "updated_at": "Utf8",
    "action": "Utf8",
    "comment_id": "Int64",
    "commit_id": "Utf8",
    "body": "Utf8",
    "ref": "Utf8",
    "number": "Int64",
    "title": "Utf8",
    "labels": "Utf8",
    "state": "Utf8",
    "locked": "Int64",
    "assignee": "Utf8",
    "comments": "Int64",
    "author_association": "Utf8",
    "closed_at": "Utf8",
    "merged_at": "Utf8",
    "merged": "Int64",
    "commits": "Int64",
    "additions": "Int64",
    "deletions": "Int64",
    "changed_files": "Int64",
    "push_size": "Int64",
    "release_tag_name": "Utf8",
    "release_name": "Utf8",
    "review_state": "Utf8",
}


class SyntheticExecutor(SqlExecutor):
    """Executes SQL through a manysql-generated dialect engine."""

    name = "synthetic"

    def __init__(
        self,
        *,
        dialect: str = "_reference",
        seed: int = 0xDB,
        n_rows: int = 5_000,
    ) -> None:
        self.dialect = dialect
        self.seed = seed
        self.n_rows = n_rows
        self._engine: DialectEngine | None = None
        self._parser: Lark | None = None
        self._catalog: dict[str, pl.DataFrame] = {}
        self._schemas: dict[str, tuple[ColumnSchema, ...]] = {}
        self._dialect_hints: str = ""

    def setup(self) -> None:
        # Lazy imports keep `pip install eval` snappy and avoid pulling
        # polars/lark for users who only ever touch the SQLite backend.
        import polars as pl  # noqa: PLC0415
        from lark import Lark, LarkError  # noqa: PLC0415

        from manysql.dialects.registry import DialectRegistry  # noqa: PLC0415
        from manysql.ir.plan import ColumnSchema  # noqa: PLC0415
        from manysql.ir.types import INT, TEXT  # noqa: PLC0415

        engine = DialectRegistry().load(self.dialect)
        try:
            parser = Lark(engine.grammar_text, start="start", parser="earley")
        except LarkError as exc:
            raise RuntimeError(
                f"failed to build parser for dialect {self.dialect!r}: {exc}"
            ) from exc

        polars_schema = {
            col: getattr(pl, dtype) for col, dtype in _GITHUB_EVENTS_POLARS_DTYPES.items()
        }
        rows = seed_rows(seed=self.seed, n=self.n_rows)
        df = pl.DataFrame(rows, schema=polars_schema)

        ir_type_map = {pl.Utf8: TEXT, pl.Int64: INT}
        cols = tuple(
            ColumnSchema(name=name, type=ir_type_map[dtype])
            for name, dtype in polars_schema.items()
        )

        self._engine = engine
        self._parser = parser
        self._catalog = {"github_events": df}
        self._schemas = {"github_events": cols}
        self._dialect_hints = _render_dialect_card(engine)

    def execute(self, sql: str) -> ExecResult:
        from manysql.executor import execute as plan_execute  # noqa: PLC0415

        if self._engine is None or self._parser is None:
            raise RuntimeError("SyntheticExecutor.setup() was not called")

        sql = sql.strip().rstrip(";").strip()
        if not sql:
            return ExecResult(success=False, error="empty SQL", backend=self.name)

        start = time.perf_counter()
        try:
            tree = self._parser.parse(sql)
            plan = self._engine.lowering.lower(
                tree, self._engine.semantics, self._schemas
            )
            df = plan_execute(
                plan, self._engine.semantics, self._catalog, self._engine.overrides
            )
        except Exception as exc:
            return ExecResult(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                execution_time_s=time.perf_counter() - start,
                backend=self.name,
            )
        rows: list[dict[str, Any]] = df.to_dicts()
        return ExecResult(
            success=True,
            rows=rows,
            columns=list(df.columns),
            execution_time_s=time.perf_counter() - start,
            backend=self.name,
        )

    def schema_prompt(self) -> str:
        hints = self._dialect_hints.rstrip()
        if hints:
            return f"{hints}\n\n{SCHEMA_PROMPT}"
        return SCHEMA_PROMPT

    def dialect_label(self) -> str:
        return f"manysql:{self.dialect}"

    def teardown(self) -> None:
        self._engine = None
        self._parser = None
        self._catalog = {}
        self._schemas = {}


def _render_dialect_card(engine: DialectEngine) -> str:
    """Compose a comprehensive 'how to write SQL in this dialect' card.

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

    # Reference defaults: a SurfaceSpec() instance is the reference baseline
    # by construction (every field defaults to the reference dialect).
    ref_surface_defaults = SurfaceSpec().model_dump()

    # Surface diffs: render only fields that differ from reference defaults.
    # ``function_aliases`` is excluded here because it gets its own dedicated
    # section below; rendering it twice just bloats the prompt.
    surface_diffs: dict[str, Any] = {}
    for key, value in surface.items():
        if key == "function_aliases":
            continue
        ref_value = ref_surface_defaults.get(key)
        if value == ref_value:
            continue
        surface_diffs[key] = value

    # Semantic diffs: SemanticDivergences uses None to mean "use reference",
    # so any non-None entry is a divergence by definition.
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

    # Identity / metadata
    if spec_dict.get("description"):
        lines.append(f"Description: {spec_dict['description']}")
    if spec_dict.get("divergence_level"):
        lines.append(f"Divergence level: {spec_dict['divergence_level']}")
    if spec_dict.get("inspired_by"):
        lines.append(f"Inspired by: {', '.join(spec_dict['inspired_by'])}")
    if spec_dict.get("notes"):
        lines.append(f"Notes: {spec_dict['notes']}")

    # Surface diffs grouped by category
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
        # any surface field not in the group whitelist (forward-compat)
        rendered_keys = {f for _, fields in _SURFACE_FIELD_GROUPS for f in fields}
        leftover = sorted(set(surface_diffs) - rendered_keys)
        if leftover:
            lines.append("  # other")
            for field in leftover:
                lines.append(f"  {field} = {_pretty(surface_diffs[field])}")

    # Concrete syntactic patterns. Field names like 'limit_syntax = offset_fetch'
    # are too abstract on their own -- the LLM needs to see the literal token
    # sequence ('OFFSET 0 ROWS FETCH NEXT 10 ROWS ONLY') it has to emit.
    patterns = _canonical_patterns(surface)
    if patterns:
        lines.append("")
        lines.append("Canonical patterns in this dialect (use these forms verbatim):")
        for label, snippet in patterns:
            lines.append(f"  {label}: {snippet}")

    # Function aliases get their own section -- they're the most common
    # source of "the LLM wrote the wrong spelling" failures.
    aliases = surface.get("function_aliases") or {}
    if aliases:
        lines.append("")
        lines.append("Function aliases (canonical name -> accepted spellings; first is primary):")
        for canonical, names in aliases.items():
            lines.append(f"  {canonical} -> {', '.join(names)}")

    # Semantic diffs
    if semantic_diffs:
        lines.append("")
        lines.append("Semantic divergences (runtime behavior):")
        for key in sorted(semantic_diffs):
            lines.append(f"  {key} = {_pretty(semantic_diffs[key])}")

    # Always-on guidance about the executor's function library, since
    # generated dialects share the reference executor's scalar/agg set.
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

    # Worked examples in this dialect's surface (if codegen wrote them).
    example_lines = _maybe_examples_lines(engine)
    if example_lines:
        lines.append("")
        lines.extend(example_lines)

    # Hard rule
    lines.append("")
    lines.append(
        "IMPORTANT: the dialect's grammar is strict. Use ONLY surface forms "
        "covered above (or by the reference baseline)."
    )

    # Reduce the whole blurb to SQL-style line comments so it reads like
    # the rest of the schema_prompt() block.
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

    Each entry is ``(label, snippet)``; both are plain strings already
    formatted for inclusion in the dialect card.
    """
    out: list[tuple[str, str]] = []

    # SELECT skeleton
    sel = surface.get("select_keyword", "SELECT")
    frm = surface.get("from_keyword", "FROM")
    whr = surface.get("where_keyword", "WHERE")
    out.append(("SELECT skeleton", f"{sel} cols {frm} t {whr} pred"))

    # GROUP BY / HAVING
    gby = surface.get("group_by_keyword", "GROUP BY")
    hav = surface.get("having_keyword", "HAVING")
    out.append(("GROUP BY / HAVING", f"{gby} c {hav} agg(c) > k"))

    # ORDER BY
    oby = surface.get("order_by_keyword", "ORDER BY")
    nfk = surface.get("nulls_first_keyword", "NULLS FIRST")
    nlk = surface.get("nulls_last_keyword", "NULLS LAST")
    out.append(("ORDER BY", f"{oby} c DESC {nlk}  (or ASC {nfk})"))

    # LIMIT — the highest-value pattern since it varies most
    limit_syntax = surface.get("limit_syntax", "limit_offset")
    if limit_syntax == "limit_offset":
        out.append(("LIMIT/OFFSET", "LIMIT 10 OFFSET 0"))
    elif limit_syntax == "offset_fetch":
        out.append(("LIMIT (offset_fetch)", "OFFSET 0 ROWS FETCH NEXT 10 ROWS ONLY"))
    elif limit_syntax == "top_n":
        out.append(("LIMIT (top_n)", "SELECT TOP 10 cols FROM t"))

    # JOIN
    jin = surface.get("join_inner_keyword", "JOIN")
    jlt = surface.get("join_left_keyword", "LEFT JOIN")
    out.append(("INNER JOIN", f"a {jin} b ON a.k = b.k"))
    out.append(("LEFT JOIN", f"a {jlt} b ON a.k = b.k"))

    # CASE WHEN
    case = surface.get("case_keyword", "CASE")
    when = surface.get("when_keyword", "WHEN")
    then = surface.get("then_keyword", "THEN")
    elsk = surface.get("else_keyword", "ELSE")
    end = surface.get("end_keyword", "END")
    out.append(("CASE", f"{case} {when} pred {then} a {elsk} b {end}"))

    # CAST
    cast_syntax = surface.get("cast_syntax", "function")
    cast_kw = surface.get("cast_keyword", "CAST")
    if cast_syntax == "double_colon":
        out.append(("CAST (double_colon)", "x::INT"))
    else:
        out.append(("CAST (function)", f"{cast_kw}(x AS INT)"))

    # IS NULL / IS NOT NULL
    isk = surface.get("is_keyword", "IS")
    nk = surface.get("null_keyword", "NULL")
    nnk = surface.get("not_keyword", "NOT")
    out.append(("IS NULL", f"x {isk} {nk}  /  x {isk} {nnk} {nk}"))

    # NULL literal
    null_lit = surface.get("null_literal", "NULL")
    if null_lit and null_lit != "NULL":
        out.append(("NULL literal", null_lit))

    # IN list / IN subquery
    in_kw = surface.get("in_keyword", "IN")
    out.append(("IN list", f"x {in_kw} (1, 2, 3)"))
    out.append(("IN subquery", f"x {in_kw} (SELECT k FROM t2)"))

    # BETWEEN
    bw = surface.get("between_keyword", "BETWEEN")
    out.append(("BETWEEN", f"x {bw} a AND b"))

    # LIKE / ILIKE
    like_kw = surface.get("like_keyword", "LIKE")
    out.append(("LIKE", f"name {like_kw} 'A%'"))
    if surface.get("ilike_keyword") and surface.get("ilike_keyword") != "ILIKE":
        out.append(("ILIKE", f"name {surface['ilike_keyword']} 'a%'"))

    # Concat
    concat_op = surface.get("concat_op", "||")
    out.append(("Concat", f"a {concat_op} b"))

    # Null-safe equal
    nse = surface.get("null_safe_eq_op")
    if nse:
        out.append(("Null-safe equal", f"a {nse} b"))

    # WITH / CTE
    wk = surface.get("with_keyword", "WITH")
    ask = surface.get("as_keyword", "AS")
    out.append(("CTE", f"{wk} cte {ask} (SELECT ...) SELECT * FROM cte"))

    # Set ops
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
    # Drop the codegen banner, which is always the leading contiguous
    # run of comment lines (codegen separates it from the first labeled
    # example with a blank line). This is more robust than substring
    # heuristics across banner phrasing changes.
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
        # Indent so the line obviously belongs to the examples block;
        # the per-line ``-- `` prefix is added by the caller's join.
        out.append(f"  {ln}" if ln else "")
    return out


__all__ = ["SyntheticExecutor"]
