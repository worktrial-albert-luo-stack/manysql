"""Moderate divergence example: keyword renames + structural tweaks.

Surface still feels SQL-y but uses different keywords and has stricter rules:
no ILIKE, MySQL-ish backtick identifiers, integer division promotes to float.
"""

from __future__ import annotations

from manysql.spec.dialect import (
    CommentStyle,
    DialectSpec,
    DivergenceLevel,
    IdentifierQuote,
    SemanticDivergences,
    SurfaceSpec,
)
from manysql.spec.semantics import (
    CaseFold,
    DivByZero,
    IntDivision,
    NullOrder,
    SetOpDefault,
)

MODERATE_KEYWORD_SWAP = DialectSpec(
    name="moderate_keyword_swap",
    description="Keyword swaps (PICK/FROM/COND/CLUSTER), backtick identifiers, "
    "DIV-by-zero returns NULL, set-op default = ALL.",
    divergence=DivergenceLevel.MODERATE,
    inspired_by=["mysql", "kdb"],
    surface=SurfaceSpec(
        identifier_quote=IdentifierQuote.BACKTICK,
        comment_styles=[CommentStyle.LINE_DASH_DASH, CommentStyle.LINE_HASH],
        select_keyword="PICK",
        where_keyword="COND",
        group_by_keyword="CLUSTER BY",
        having_keyword="OF",
        order_by_keyword="SORT BY",
        limit_keyword="TAKE",
        union_keyword="MERGE",
        intersect_keyword="BOTH",
        except_keyword="WITHOUT",
        ilike_keyword="MATCHES",
        function_aliases={
            "LENGTH": ["LEN"],
            "UPPER": ["UCASE"],
            "LOWER": ["LCASE"],
            "COALESCE": ["NVL", "IFNULL"],
        },
    ),
    semantics=SemanticDivergences(
        identifier_case_fold=CaseFold.UPPER,
        quoted_identifiers_case_sensitive=False,
        null_order_default_asc=NullOrder.FIRST,
        null_order_default_desc=NullOrder.LAST,
        division_by_zero=DivByZero.NULL,
        integer_division=IntDivision.PROMOTE,
        like_case_sensitive=False,
        ilike_supported=False,
        set_op_default=SetOpDefault.ALL,
        sum_of_empty_returns_null=False,
    ),
    notes="Stresses keyword renaming and the set-op default knob.",
)
