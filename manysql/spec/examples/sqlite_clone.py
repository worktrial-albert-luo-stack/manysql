"""SQLite clone: a faithful real-world dialect baseline.

Modeled on production SQLite. Pairs with `snowflake_clone` to give the
eval harness two real-world reference points: SQLite is the small, lenient
end (NULL-on-error, case-insensitive LIKE, C-style truthiness) and
Snowflake is the strict end (UPPER fold, error on div-by-zero, strict
booleans). Anything in between exercises the codegen surface.

Real SQLite characteristics this spec encodes:

Surface
- Single-quoted strings, double-quoted identifiers (with the well-known
  string-literal fallback handled by the executor at runtime, not the
  surface).
- Comments: ``--`` line, ``/* */`` block (no ``//``).
- ``LIMIT n OFFSET m``.
- ``CAST(x AS T)`` only — SQLite has no ``::`` form.
- Function aliases that ship in real SQLite: ``IFNULL`` for the 2-arg form
  of ``COALESCE``, ``SUBSTRING`` accepted alongside ``SUBSTR``.

Semantics
- Identifiers preserve case but compare ASCII-case-insensitively, so the
  spec uses ``identifier_case_fold = preserve`` paired with
  ``quoted_identifiers_case_sensitive = False``.
- Default null ordering: NULLs are smaller than any value, so ASC sorts
  NULLs FIRST and DESC sorts NULLs LAST.
- Division by zero returns NULL (SQLite's "lenient arithmetic").
- Integer division truncates toward zero (5/2 -> 2).
- LIKE is ASCII case-insensitive by default.
- ILIKE is *not* supported.
- Boolean truthiness is C-style: 0 is false, nonzero is true, '' is false.
  SQLite has no native BOOLEAN type, so values flow through as INTEGER.
- ``IS [NOT] DISTINCT FROM`` is supported in modern SQLite.
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
    BoolTruthiness,
    CaseFold,
    DivByZero,
    IntDivision,
    NullOrder,
)

SQLITE_CLONE = DialectSpec(
    name="sqlite_clone",
    description="Faithful SQLite clone: preserve-case identifiers (ASCII "
    "case-insensitive), NULLS FIRST/LAST defaults, NULL-on-divide-by-zero, "
    "truncating integer division, case-insensitive LIKE, no ILIKE, "
    "C-style boolean truthiness, IFNULL/SUBSTRING aliases.",
    divergence=DivergenceLevel.MILD,
    inspired_by=["sqlite"],
    surface=SurfaceSpec(
        comment_styles=[
            CommentStyle.LINE_DASH_DASH,
            CommentStyle.BLOCK_C,
        ],
        function_aliases={
            "COALESCE": ["COALESCE", "IFNULL"],
            "SUBSTR": ["SUBSTR", "SUBSTRING"],
            "LENGTH": ["LENGTH"],
            "UPPER": ["UPPER"],
            "LOWER": ["LOWER"],
        },
    ),
    semantics=SemanticDivergences(
        identifier_case_fold=CaseFold.PRESERVE,
        quoted_identifiers_case_sensitive=False,
        null_order_default_asc=NullOrder.FIRST,
        null_order_default_desc=NullOrder.LAST,
        null_safe_eq_supported=True,
        division_by_zero=DivByZero.NULL,
        integer_division=IntDivision.TRUNCATE,
        like_case_sensitive=False,
        ilike_supported=False,
        boolean_truthiness=BoolTruthiness.C_STYLE,
        sum_of_empty_returns_null=True,
        group_by_accepts_select_aliases=True,
        select_resolves_through_group_by=True,
    ),
    notes="Baseline target for the eval harness. Pairs with "
    "`snowflake_clone` to bracket realistic real-world divergence.",
)
