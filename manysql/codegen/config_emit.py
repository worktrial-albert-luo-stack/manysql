"""Emit semantics.json from a DialectSpec.

This is purely mechanical: copy reference defaults, then apply non-None
overrides from the spec. No LLM involved.
"""

from __future__ import annotations

import json

from manysql.spec.dialect import DialectSpec


def emit_semantic_config(spec: DialectSpec) -> str:
    """Return the JSON text for the dialect's semantics.json."""
    cfg = spec.semantics.to_semantic_config()
    payload = cfg.model_dump(mode="json")
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
