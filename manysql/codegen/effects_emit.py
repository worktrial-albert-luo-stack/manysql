"""Emit effects.py for a generated dialect.

Effects are named handlers that the canonical executor consults at a
fixed set of decision points. They let a dialect swap the
*implementation* of a canonical operation without changing the IR shape
or the surface grammar.

The contract:

    EFFECTS: dict[str, Callable]

Each key is a registered effect name (see the v1 registry below) and
each value is a callable whose signature is fixed for that name. When
an effect name is absent the executor falls back to its canonical
behavior.

This is the open-world counterpart to the closed-world `SemanticConfig`
knobs: knobs cover divergences whose space is small and known up front
(e.g. `null_order_default_asc`); effects cover divergences whose space
is open-ended (e.g. *which* collation to use for text equality).

v1 effect registry
==================

    text_eq(left: pl.Expr, right: pl.Expr, semantics: SemanticConfig) -> pl.Expr
        Implementation of `=` between two text-typed operands.
    text_neq(left: pl.Expr, right: pl.Expr, semantics: SemanticConfig) -> pl.Expr
        Implementation of `<>` / `!=` between two text-typed operands.
    text_in_pattern(value: pl.Expr, pattern: pl.Expr, semantics: SemanticConfig) -> pl.Expr
        Implementation of LIKE / ILIKE pattern matching.

New effect names land via RFC + executor wiring; this module is
deliberately not a free-form hook surface.
"""

from __future__ import annotations

from manysql.spec.dialect import DialectSpec


_EFFECTS_TEMPLATE = '''"""Generated executor effects for the {name} dialect.

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


EFFECTS: dict[str, Callable] = {{}}
'''


def emit_effects(spec: DialectSpec) -> str:
    """Return the effects.py source for the dialect.

    Currently always returns the empty-dict template. Spec-aware
    population is deferred until the codegen pipeline grows knowledge
    of features that *require* an effect (e.g. an explicit
    `text_collation` field in `SemanticDivergences`).
    """
    return _EFFECTS_TEMPLATE.format(name=spec.name)


__all__ = ["emit_effects"]
