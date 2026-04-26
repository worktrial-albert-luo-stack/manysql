"""Mild divergence example: a Postgres-ish dialect.

Diverges from the reference only in semantic knobs and a couple of operator
extensions that any real DB ships with. Surface stays effectively ANSI.
"""

from __future__ import annotations

from manysql.spec.dialect import (
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

MILD_POSTGRES_ISH = DialectSpec(
    name="mild_postgres_ish",
    description="Postgres-flavored mild divergence: lowercase fold, "
    "NULLS FIRST default DESC, integer division truncates, ::cast.",
    divergence=DivergenceLevel.MILD,
    inspired_by=["postgres"],
    surface=SurfaceSpec(
        function_aliases={
            "LENGTH": ["LENGTH", "CHAR_LENGTH"],
            "LOWER": ["LOWER", "LCASE"],
        },
    ),
    semantics=SemanticDivergences(
        identifier_case_fold=CaseFold.LOWER,
        null_order_default_asc=NullOrder.LAST,
        null_order_default_desc=NullOrder.FIRST,
        division_by_zero=DivByZero.ERROR,
        integer_division=IntDivision.TRUNCATE,
        like_case_sensitive=True,
        ilike_supported=True,
        sum_of_empty_returns_null=True,
    ),
    notes="Closest to the reference; a smoke test for the codegen pipeline.",
)
