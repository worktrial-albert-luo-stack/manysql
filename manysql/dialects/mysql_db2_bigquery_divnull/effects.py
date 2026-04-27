"""Generated executor effects for the mysql_db2_bigquery_divnull dialect.

This module exposes one public dict:

    EFFECTS: dict[str, Callable]
        Maps a v1 effect name to its handler. Absent names fall back to
        the canonical executor implementation.

v1 registry (see manysql/codegen/effects_emit.py for full signatures):

    text_eq(left, right, semantics) -> pl.Expr
    text_neq(left, right, semantics) -> pl.Expr
    text_in_pattern(value, pattern, semantics) -> pl.Expr
"""

from __future__ import annotations

from typing import Callable


EFFECTS: dict[str, Callable] = {}
