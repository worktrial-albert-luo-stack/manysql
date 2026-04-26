"""Emit overrides.py for a generated dialect.

Overrides are Python implementations of dialect-specific functions /
operators that the shared executor doesn't natively support. They live in
a single module per dialect with a strict, documented API:

    FUNCTIONS: dict[str, Callable[[list[pl.Expr], SemanticConfig], pl.Expr]]
    OPERATORS: dict[str, Callable[[list[pl.Expr], SemanticConfig], pl.Expr]]

The executor consults these dicts when it encounters a `FuncCall` (or
operator) whose name it cannot resolve. The deterministic emitter writes
empty dicts: shipping the file unconditionally keeps every dialect package
self-describing and makes the LLM lane a one-line patch instead of an
"add a new file" gesture.

Sandboxing: see `manysql.codegen.overrides_loader` for the restricted
import path. Generated overrides may use polars/pyarrow but cannot use
`os`, `sys`, `subprocess`, network libraries, or `open`.
"""

from __future__ import annotations

from manysql.spec.dialect import DialectSpec


_OVERRIDES_TEMPLATE = '''"""Generated operator/function overrides for the {name} dialect.

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


FUNCTIONS: dict[str, Callable[[list[pl.Expr], Any], pl.Expr]] = {{}}
OPERATORS: dict[str, Callable[[list[pl.Expr], Any], pl.Expr]] = {{}}
{extra}
'''


def emit_overrides(spec: DialectSpec) -> str:
    """Return the overrides.py source for the dialect.

    Currently always returns an empty-dict template. A future LLM lane
    can populate `FUNCTIONS` / `OPERATORS` based on the spec's invented
    features (e.g. `function_aliases` that don't map cleanly to canonical
    builtins, novel operators introduced via surface knobs).
    """
    extra = _emit_aliases_comment(spec)
    return _OVERRIDES_TEMPLATE.format(name=spec.name, extra=extra)


def _emit_aliases_comment(spec: DialectSpec) -> str:
    """Document any function aliases for the LLM lane to consume.

    Aliases are typically resolved at the lowering layer (dialect surface
    name -> canonical FuncCall name), but recording them here gives the
    overrides emitter and downstream LLM passes a single visible spot.
    """
    if not spec.surface.function_aliases:
        return ""
    lines = ["", "# Surface aliases recorded from the spec (informational):"]
    for canonical, surface_names in spec.surface.function_aliases.items():
        joined = ", ".join(repr(n) for n in surface_names)
        lines.append(f"#   {canonical}: {joined}")
    return "\n".join(lines) + "\n"


__all__ = ["emit_overrides"]
