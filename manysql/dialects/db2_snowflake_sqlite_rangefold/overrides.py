"""Generated operator/function overrides for the db2_snowflake_sqlite_rangefold dialect.

This module exposes two public dicts:

    FUNCTIONS: dict[str, Callable]
        Map UPPERCASE function name -> callable that accepts
        `(args: list[pl.Expr], semantics: SemanticConfig) -> pl.Expr`.
        Functions in this dict take precedence over the executor's
        built-in handlers when their name matches.

    OPERATORS: dict[str, Callable]
        Map UPPERCASE operator label (e.g. "TILDE_EQ" for `~=`) -> same
        callable shape. Reserved for dialects whose lowering encodes
        novel operators as canonical FuncCalls (e.g. FuncCall("TILDE_EQ", a, b)).

The deterministic codegen writes empty dicts. LLM-refined emitters may
populate them as needed for the spec's invented features.
"""

from __future__ import annotations

from typing import Any, Callable

import polars as pl


FUNCTIONS: dict[str, Callable[[list[pl.Expr], Any], pl.Expr]] = {}
OPERATORS: dict[str, Callable[[list[pl.Expr], Any], pl.Expr]] = {}

# Surface aliases recorded from the spec (informational):
#   COALESCE: 'COALESCE', 'IFNULL', 'VALUE'
#   LENGTH: 'LENGTH', 'CHAR_LENGTH', 'CHARACTER_LENGTH'
#   SUBSTR: 'SUBSTR', 'SUBSTRING'
#   UPPER: 'UPPER', 'UCASE'
#   LOWER: 'LOWER', 'LCASE'
#   TRIM: 'TRIM', 'STRIP'
#   NOW: 'CURRENT_TIMESTAMP', 'NOW'

