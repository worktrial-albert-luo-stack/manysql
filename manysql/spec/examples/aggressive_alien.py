"""Aggressive divergence example: invented operators, exotic syntax.

This is intentionally far from real SQL: the surface is recognizable but
several pillars are reshaped. Used to stress the codegen refine loop.
"""

from __future__ import annotations

from manysql.spec.dialect import (
    CastSyntax,
    DialectSpec,
    DivergenceLevel,
    LimitSyntax,
    NullLiteral,
    SemanticDivergences,
    StringQuote,
    SurfaceSpec,
)
from manysql.spec.semantics import (
    BoolTruthiness,
    CaseFold,
    CountDistinctNull,
    DivByZero,
    NullOrder,
    SetOpDefault,
    StringConcatOp,
)

AGGRESSIVE_ALIEN = DialectSpec(
    name="aggressive_alien",
    description="Invented dialect: NIL nulls, double-colon casts, `~=` for "
    "null-safe-eq, `+` for concat, OFFSET/FETCH limits, no ILIKE.",
    divergence=DivergenceLevel.AGGRESSIVE,
    inspired_by=["sql_server", "kdb", "research"],
    surface=SurfaceSpec(
        string_quote=StringQuote.SINGLE,
        null_literal=NullLiteral.NIL,
        select_keyword="SELECT",
        from_keyword="FROM",
        where_keyword="WHERE",
        having_keyword="HAVE",
        order_by_keyword="ORDERED BY",
        cast_syntax=CastSyntax.DOUBLE_COLON,
        limit_syntax=LimitSyntax.OFFSET_FETCH,
        ilike_keyword="ILIKE",  # nominally retained but disabled in semantics
        eq_op="=",
        neq_op=["!=", "<>"],
        concat_op="+",
        null_safe_eq_op="~=",
        function_aliases={
            "COALESCE": ["COALESCE", "FIRSTNONNIL"],
            "LENGTH": ["LEN"],
        },
    ),
    semantics=SemanticDivergences(
        identifier_case_fold=CaseFold.PRESERVE,
        quoted_identifiers_case_sensitive=True,
        null_order_default_asc=NullOrder.FIRST,
        null_order_default_desc=NullOrder.FIRST,
        division_by_zero=DivByZero.INF,
        like_case_sensitive=False,
        ilike_supported=False,
        string_concat_op=StringConcatOp.PLUS,
        set_op_default=SetOpDefault.ALL,
        boolean_truthiness=BoolTruthiness.C_STYLE,
        count_distinct_null=CountDistinctNull.INCLUDED,
        sum_of_empty_returns_null=False,
    ),
    notes="Maximal stress test: many surface and semantic knobs flipped at once.",
)
