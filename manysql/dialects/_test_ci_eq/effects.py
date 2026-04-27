"""Executor effects for the _test_ci_eq dialect.

The dialect treats `=` between text-typed values as collation-insensitive
(roughly T-SQL's default `Latin1_General_CI_AS`). Numeric comparisons
that happen to be wrapped in this handler still produce the correct
result because Polars `cast(Utf8)` round-trips integers losslessly and
`str.to_lowercase` is a no-op for digits — but production dialects with
real schema awareness would gate this on the operand type.

Effect signatures (see manysql/codegen/effects_emit.py):

    text_eq(left, right, semantics) -> Optional[pl.Expr]
    text_neq(left, right, semantics) -> Optional[pl.Expr]
    text_in_pattern(operand, pattern, semantics, case_sensitive) -> Optional[pl.Expr]
"""

from __future__ import annotations

from typing import Callable

import polars as pl


def _ci_eq(left: pl.Expr, right: pl.Expr, semantics) -> pl.Expr:  # noqa: ARG001
    return left.cast(pl.Utf8).str.to_lowercase() == right.cast(
        pl.Utf8
    ).str.to_lowercase()


def _ci_neq(left: pl.Expr, right: pl.Expr, semantics) -> pl.Expr:  # noqa: ARG001
    return left.cast(pl.Utf8).str.to_lowercase() != right.cast(
        pl.Utf8
    ).str.to_lowercase()


EFFECTS: dict[str, Callable] = {
    "text_eq": _ci_eq,
    "text_neq": _ci_neq,
}
