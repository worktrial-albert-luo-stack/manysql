"""Executor effects for the tsql_ish dialect.

T-SQL's default ``Latin1_General_CI_AS`` collation makes ``=`` and
``<>`` between text columns case-insensitive. Neither is captured by
the closed-world ``SemanticConfig`` because real-world dialects pick
from a vast space of collations — exactly what RFC 0002's effects
lane is for.

The handlers below replace the canonical Polars ``=`` / ``<>``
implementation in ``ExprEvaluator._binary`` for this dialect. They
return ``None`` for non-text operands so numeric and date comparisons
fall through to the canonical implementation unchanged.

The ``text_in_pattern`` slot is intentionally not populated: T-SQL's
``LIKE`` *is* case-insensitive under the same default collation, but
that's already expressible via ``SemanticConfig.like_case_sensitive``
(set in this dialect's spec to False), so the canonical path produces
the right answer.

Effect signatures (see manysql/codegen/effects_emit.py):

    text_eq(left, right, semantics) -> Optional[pl.Expr]
    text_neq(left, right, semantics) -> Optional[pl.Expr]
    text_in_pattern(operand, pattern, semantics, case_sensitive) -> Optional[pl.Expr]
"""

from __future__ import annotations

from typing import Callable, Optional

import polars as pl


def _ci_eq(  # noqa: ARG001
    left: pl.Expr, right: pl.Expr, semantics
) -> Optional[pl.Expr]:
    return left.cast(pl.Utf8).str.to_lowercase() == right.cast(
        pl.Utf8
    ).str.to_lowercase()


def _ci_neq(  # noqa: ARG001
    left: pl.Expr, right: pl.Expr, semantics
) -> Optional[pl.Expr]:
    return left.cast(pl.Utf8).str.to_lowercase() != right.cast(
        pl.Utf8
    ).str.to_lowercase()


EFFECTS: dict[str, Callable] = {
    "text_eq": _ci_eq,
    "text_neq": _ci_neq,
}
