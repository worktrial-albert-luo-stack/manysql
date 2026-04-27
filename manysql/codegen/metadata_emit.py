"""Emit metadata.json and spec.json for a generated dialect.

These are bookkeeping files (no executor influence). They record what
generated the dialect and what surface/semantic decisions it embodies, so
the registry can list and triage dialects without re-running codegen.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Optional

from manysql.codegen.card_conformance import CardWarning
from manysql.spec.dialect import DialectSpec


def emit_metadata(
    spec: DialectSpec,
    *,
    model: Optional[str],
    provider: str,
    lifecycle: str = "draft",
    prompts: Optional[dict[str, str]] = None,
    retry_log: Optional[list[dict[str, object]]] = None,
    created_at: Optional[str] = None,
    updated_at: Optional[str] = None,
    card_warnings: Optional[list[CardWarning]] = None,
) -> str:
    """Return the JSON text for the dialect's metadata.json."""
    now = _isoformat(dt.datetime.now(tz=dt.timezone.utc))
    payload: dict[str, object] = {
        "lifecycle": lifecycle,
        "generation": {
            "model": model,
            "provider": provider,
            "prompts": prompts or {},
            "retry_log": retry_log or [],
            "created_at": created_at or now,
            "updated_at": updated_at or now,
        },
        "card_warnings": [
            {"label": w.label, "source": w.source, "error": w.error}
            for w in (card_warnings or [])
        ],
    }
    _ = spec  # kept for future provenance hooks
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def emit_spec_json(spec: DialectSpec) -> str:
    """Return the JSON text for the dialect's spec.json.

    Contains both:
    - a curated human-readable summary (``surface_features`` flag list,
      ``semantic_overrides`` non-None dict) for triage / dashboards, and
    - the full ``DialectSpec`` dump under ``surface`` / ``semantics`` so
      downstream consumers (eval prompt builder, oracle harness, anything
      that needs to reason about the original codegen input) can recover
      every field without re-importing the original Python module.
    """
    surface_features = _surface_features_from_spec(spec)
    semantic_overrides = {
        k: v
        for k, v in spec.semantics.model_dump(mode="json").items()
        if v is not None
    }
    payload = {
        "name": spec.name,
        "description": spec.description,
        "divergence_level": spec.divergence.value,
        "inspired_by": list(spec.inspired_by),
        "surface_features": surface_features,
        "semantic_overrides": semantic_overrides,
        "surface": spec.surface.model_dump(mode="json"),
        "semantics": spec.semantics.model_dump(mode="json"),
        "notes": spec.notes or "",
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _isoformat(d: dt.datetime) -> str:
    # JSON likes a trailing 'Z'; datetime won't add it for us when tz-aware.
    return d.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _surface_features_from_spec(spec: DialectSpec) -> list[str]:
    """Heuristic enumeration of the surface features a spec enables.

    We list every feature the reference supports and tag those that this spec
    keeps available. Codegen later honors the same list when emitting the
    grammar.
    """
    feats = [
        "select_distinct",
        "where",
        "group_by",
        "having",
        "order_by_nulls_first_last",
        "inner_left_right_full_cross_join",
        "join_using",
        "ctes",
        "subqueries_uncorrelated",
        "case_when",
        "between",
        "in_list",
        "in_subquery",
        "exists",
        "is_null",
        "is_distinct_from",
        "like",
        "windows_basic",
    ]
    if spec.semantics.ilike_supported is not False:
        feats.append("ilike")
    if spec.surface.limit_syntax.value != "limit_offset":
        feats.append(f"limit_via_{spec.surface.limit_syntax.value}")
    else:
        feats.append("limit_offset")
    if spec.surface.cast_syntax.value != "cast_as":
        feats.append(f"cast_via_{spec.surface.cast_syntax.value}")
    else:
        feats.append("cast_as")
    return sorted(set(feats))
