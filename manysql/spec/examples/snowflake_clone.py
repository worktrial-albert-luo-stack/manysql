"""Snowflake clone: a faithful real-world dialect baseline.

Modeled on production Snowflake. The point isn't to invent anything — the
point is to give the eval harness a `--synthetic-dialect snowflake_clone`
target that mirrors a real, well-documented dialect, so any LLM regression
on this slot is interpretable as "the model got worse at Snowflake" rather
than "the model got worse at our synthetic surface."

Real Snowflake characteristics this spec encodes:

Surface
- Single-quoted strings, double-quoted identifiers (ANSI).
- Comments: ``--`` line, ``/* */`` block, and ``//`` line (Snowflake-specific).
- ``LIMIT n OFFSET m`` (FETCH form is also accepted by Snowflake but the
  IR / lowerer only needs one canonical surface).
- ``CAST(x AS T)`` syntax (Snowflake also supports ``::`` but the spec only
  has one cast slot; ``::`` would live as a moderate variant).
- Function aliases that ship in real Snowflake: ``NVL``/``IFNULL`` for
  ``COALESCE``, ``LEN`` for ``LENGTH``, ``SUBSTRING`` for ``SUBSTR``,
  ``TRY_CAST`` for the failable form of ``CAST``.

Semantics
- Unquoted identifiers fold to UPPER (vs Postgres lower).
- Quoted identifiers are case-sensitive.
- Default null ordering: NULLS FIRST on ASC, NULLS LAST on DESC.
- Division by zero raises (Snowflake's ``ERROR_ON_DIVISION_BY_ZERO`` is on
  by default).
- Integer division promotes to NUMBER (1/2 -> 0.5).
- LIKE is case-sensitive, ILIKE is supported natively.
- ``IS [NOT] DISTINCT FROM`` is supported (alongside ``EQUAL_NULL``).
- GROUP BY accepts SELECT aliases; SELECT resolves through GROUP BY.
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
    CaseFold,
    DivByZero,
    IntDivision,
    NullOrder,
)

SNOWFLAKE_CLONE = DialectSpec(
    name="snowflake_clone",
    description="Faithful Snowflake clone: UPPER identifier fold, ILIKE, "
    "NULLS FIRST/LAST defaults, division-by-zero error, integer division "
    "promotes, // line comments, TRY_CAST/NVL/IFNULL aliases.",
    divergence=DivergenceLevel.MILD,
    inspired_by=["snowflake"],
    surface=SurfaceSpec(
        comment_styles=[
            CommentStyle.LINE_DASH_DASH,
            CommentStyle.BLOCK_C,
            CommentStyle.LINE_DOUBLE_SLASH,
        ],
        function_aliases={
            "COALESCE": ["COALESCE", "NVL", "IFNULL"],
            "LENGTH": ["LENGTH", "LEN"],
            "SUBSTR": ["SUBSTR", "SUBSTRING"],
            "CAST": ["CAST", "TRY_CAST"],
            "UPPER": ["UPPER"],
            "LOWER": ["LOWER"],
        },
    ),
    semantics=SemanticDivergences(
        identifier_case_fold=CaseFold.UPPER,
        quoted_identifiers_case_sensitive=True,
        null_order_default_asc=NullOrder.FIRST,
        null_order_default_desc=NullOrder.LAST,
        null_safe_eq_supported=True,
        division_by_zero=DivByZero.ERROR,
        integer_division=IntDivision.PROMOTE,
        like_case_sensitive=True,
        ilike_supported=True,
        sum_of_empty_returns_null=True,
        group_by_accepts_select_aliases=True,
        select_resolves_through_group_by=True,
    ),
    notes="Baseline target for the eval harness. Differs from "
    "`mild_snowflake_upper` in keeping the Snowflake-standard NULLS FIRST "
    "(ASC) / NULLS LAST (DESC) ordering rather than the latter's "
    "session-style 'NULLS LAST everywhere'.",
)
