"""T-SQL-ish divergence example: motivates the new effects lane.

T-SQL's most pervasive silent divergence is the default
``Latin1_General_CI_AS`` collation: ``=`` between text columns is
case-insensitive, and so is ``LIKE``. Neither is a closed-world
``SemanticConfig`` enum: real-world dialects pick from a vast space
of collations, so the right place for this divergence is the
open-world `effects.py` lane (per RFC 0002).

Beyond collation, T-SQL has plenty of surface knobs the closed
``SurfaceSpec`` already covers:

  * ``[ident]`` quoted identifiers
  * ``+`` for string concat
  * ``LEN`` for ``LENGTH``
  * ``ISNULL`` aliasing ``COALESCE`` (2-arg form)
  * preserve-case identifiers

This spec ships those knobs and is intentionally small enough that
the deterministic codegen emitter handles it end-to-end (no LLM
required). The collation effect is hand-populated *after* codegen
to demonstrate the new lane plugging into a generated package.
"""

from __future__ import annotations

from manysql.spec.dialect import (
    DialectSpec,
    DivergenceLevel,
    IdentifierQuote,
    SemanticDivergences,
    SurfaceSpec,
)
from manysql.spec.semantics import (
    CaseFold,
    DivByZero,
    StringConcatOp,
)

TSQL_ISH = DialectSpec(
    name="tsql_ish",
    description="T-SQL-flavored: bracket identifiers, + concat, LEN, "
    "case-insensitive default collation (effects.py lane).",
    divergence=DivergenceLevel.MODERATE,
    inspired_by=["sql_server"],
    surface=SurfaceSpec(
        identifier_quote=IdentifierQuote.BRACKET,
        concat_op="+",
        function_aliases={
            "LENGTH": ["LEN"],
            "COALESCE": ["COALESCE", "ISNULL"],
            "UPPER": ["UPPER"],
            "LOWER": ["LOWER"],
        },
    ),
    semantics=SemanticDivergences(
        identifier_case_fold=CaseFold.PRESERVE,
        quoted_identifiers_case_sensitive=False,
        string_concat_op=StringConcatOp.PLUS,
        like_case_sensitive=False,
        ilike_supported=False,
        division_by_zero=DivByZero.ERROR,
        sum_of_empty_returns_null=True,
    ),
    notes="The CI-AS default collation is intentionally NOT a SemanticConfig knob; "
    "it lives in the per-dialect effects.py lane (RFC 0002).",
)
