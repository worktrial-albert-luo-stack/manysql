"""Emit semantics.json from a DialectSpec.

This is purely mechanical: copy reference defaults, then apply non-None
overrides from the spec. No LLM involved.
"""

from __future__ import annotations

import json

from manysql.spec.dialect import DialectSpec, SetOpPrecedence
from manysql.spec.semantics import SemanticConfig, SetOpPrecedenceMode


def compose_semantic_config(spec: DialectSpec) -> SemanticConfig:
    """Merge ``spec.semantics`` with the surface-level bridges.

    Two surface knobs are honored at runtime via ``SemanticConfig``:

    - ``surface.function_aliases``: lives on ``SurfaceSpec`` (because it's
      how the dialect *spells* IR functions) but the executor reads it
      through ``SemanticConfig.function_aliases``. Without the bridge,
      aliases like ``NVL`` or ``LEN`` are inert at runtime.
    - ``surface.set_op_precedence``: lives on ``SurfaceSpec`` (because it's
      a parsing-precedence choice) but is honored at runtime because the
      deterministic grammar emitter keeps a flat parse tree — precedence
      climbing happens in ``lower_query_expr``.

    Use this helper anywhere you want the SemanticConfig the *deployed*
    dialect uses, not just the SemanticDivergences subset. The IR battery
    relies on this so reference and dialect lowerings see the same config
    and produce matching plans.
    """
    cfg = spec.semantics.to_semantic_config()
    overrides: dict[str, object] = {}
    if spec.surface.function_aliases:
        overrides["function_aliases"] = dict(spec.surface.function_aliases)
    if spec.surface.set_op_precedence != SetOpPrecedence.ANSI:
        overrides["set_op_precedence"] = SetOpPrecedenceMode(
            spec.surface.set_op_precedence.value
        )
    if overrides:
        cfg = cfg.model_copy(update=overrides)
    return cfg


def emit_semantic_config(spec: DialectSpec) -> str:
    """Return the JSON text for the dialect's semantics.json."""
    cfg = compose_semantic_config(spec)
    payload = cfg.model_dump(mode="json")
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
