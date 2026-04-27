"""Postgres clone: a faithful real-world dialect baseline.

Modeled on production PostgreSQL. Pairs with `snowflake_clone` and
`sqlite_clone` as a third real-world reference point in the eval harness:
where SQLite is the lenient end (NULL-on-error, ASCII-CI LIKE) and
Snowflake is the strict, UPPER-fold, integer-promote end, Postgres lives
in the middle: lower-fold identifiers, error on division-by-zero,
truncating integer division, case-sensitive LIKE *and* native ILIKE.

Real Postgres characteristics this spec encodes:

Surface
- Single-quoted strings, double-quoted identifiers (ANSI).
- Comments: ``--`` line, ``/* */`` block.
- ``LIMIT n OFFSET m`` (Postgres also accepts the SQL-standard
  ``OFFSET m FETCH NEXT n ROWS ONLY`` and the early-Postgres
  comma form, but those are moderate variants — the IR / lowerer
  only needs one canonical surface).
- ``CAST(x AS T)`` (Postgres also has the ``::`` shorthand; that
  belongs in a moderate variant since it changes the parse).
- Function aliases that ship in real Postgres: ``CHAR_LENGTH`` for
  ``LENGTH`` (SQL standard), ``SUBSTRING`` for ``SUBSTR``,
  ``COALESCE`` is canonical (no ``NVL`` / ``IFNULL``).
- Native ``ILIKE`` operator and ``IS [NOT] DISTINCT FROM``.

Semantics
- Unquoted identifiers fold to lower (vs Snowflake upper).
- Quoted identifiers are case-sensitive ("Foo" != foo).
- Default null ordering: NULLS LAST on ASC, NULLS FIRST on DESC.
- Division by zero raises ``division_by_zero`` (Postgres default).
- Integer division truncates toward zero (1/2 -> 0, -7/2 -> -3
  matches Postgres int4/int4).
- LIKE is case-sensitive, ILIKE is supported natively.
- Empty SUM returns NULL (ANSI; matches Postgres).
- GROUP BY accepts SELECT aliases; SELECT resolves through GROUP BY
  (Postgres extension, also accepted by SQLite/MySQL/DuckDB).
- ``IS [NOT] DISTINCT FROM`` is the null-safe equality form.
- String concat is ``||``; UNION/INTERSECT/EXCEPT default to DISTINCT.
- Boolean type is strict: comparisons return BOOL, no implicit C-style
  truthiness from numeric/text values.
"""

from __future__ import annotations

from manysql.spec.dialect import (
    CommentStyle,
    DialectSpec,
    DivergenceLevel,
    SemanticDivergences,
    SurfaceSpec,
)
from manysql.spec.semantics import (
    BoolTruthiness,
    CaseFold,
    DivByZero,
    IntDivision,
    NullOrder,
)

POSTGRES_CLONE = DialectSpec(
    name="postgres_clone",
    description="Faithful Postgres clone: lower identifier fold, NULLS LAST "
    "(ASC) / NULLS FIRST (DESC), error on division-by-zero, truncating "
    "integer division, case-sensitive LIKE, native ILIKE, strict "
    "booleans, CHAR_LENGTH/SUBSTRING aliases.",
    divergence=DivergenceLevel.MILD,
    inspired_by=["postgres"],
    surface=SurfaceSpec(
        comment_styles=[
            CommentStyle.LINE_DASH_DASH,
            CommentStyle.BLOCK_C,
        ],
        function_aliases={
            "COALESCE": ["COALESCE"],
            "LENGTH": ["LENGTH", "CHAR_LENGTH"],
            "SUBSTR": ["SUBSTR", "SUBSTRING"],
            "UPPER": ["UPPER"],
            "LOWER": ["LOWER"],
        },
    ),
    semantics=SemanticDivergences(
        identifier_case_fold=CaseFold.LOWER,
        quoted_identifiers_case_sensitive=True,
        null_order_default_asc=NullOrder.LAST,
        null_order_default_desc=NullOrder.FIRST,
        null_safe_eq_supported=True,
        division_by_zero=DivByZero.ERROR,
        integer_division=IntDivision.TRUNCATE,
        like_case_sensitive=True,
        ilike_supported=True,
        boolean_truthiness=BoolTruthiness.STRICT,
        sum_of_empty_returns_null=True,
        group_by_accepts_select_aliases=True,
        select_resolves_through_group_by=True,
    ),
    notes="Baseline target for the eval harness alongside `snowflake_clone` "
    "and `sqlite_clone`. Differs from `mild_postgres_ish` (which uses the "
    "same semantic knobs but is intentionally light on surface aliases) "
    "by enumerating Postgres's standard CHAR_LENGTH / SUBSTRING aliases "
    "so the dialect card surfaces them to the LLM.",
)
