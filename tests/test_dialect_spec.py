"""Sanity tests for the DialectSpec schema and example specs.

These tests don't exercise codegen — they just confirm the spec round-trips
through Pydantic and that the example library renders to a coherent
SemanticConfig. Once the codegen pipeline lands, these specs become the
seeds of the parametric end-to-end tests.
"""

from __future__ import annotations

import pytest

from manysql.spec import (
    DialectSpec,
    DivergenceLevel,
    SemanticConfig,
    SemanticDivergences,
    SurfaceSpec,
)
from manysql.spec.examples import EXAMPLE_SPECS


def test_default_spec_matches_reference_semantics() -> None:
    """An empty SemanticDivergences should produce the reference SemanticConfig."""
    cfg = SemanticDivergences().to_semantic_config()
    assert cfg == SemanticConfig.reference()


def test_default_dialect_spec_round_trips() -> None:
    spec = DialectSpec(name="empty")
    blob = spec.model_dump_json()
    restored = DialectSpec.model_validate_json(blob)
    assert restored == spec


def test_dialect_spec_immutable() -> None:
    spec = DialectSpec(name="immutable")
    with pytest.raises(Exception):
        spec.name = "other"  # type: ignore[misc]


@pytest.mark.parametrize("name", list(EXAMPLE_SPECS))
def test_example_spec_validates(name: str) -> None:
    spec = EXAMPLE_SPECS[name]
    assert isinstance(spec, DialectSpec)
    cfg = spec.semantics.to_semantic_config()
    assert isinstance(cfg, SemanticConfig)
    assert spec.divergence in (
        DivergenceLevel.NONE,
        DivergenceLevel.MILD,
        DivergenceLevel.MODERATE,
        DivergenceLevel.AGGRESSIVE,
    )


def test_example_specs_diverge_at_increasing_levels() -> None:
    levels_seen = {EXAMPLE_SPECS[k].divergence for k in EXAMPLE_SPECS}
    assert {DivergenceLevel.MILD, DivergenceLevel.MODERATE, DivergenceLevel.AGGRESSIVE} <= levels_seen


def test_surface_spec_supports_function_aliases() -> None:
    spec = SurfaceSpec(function_aliases={"LENGTH": ["LEN", "CHAR_LENGTH"]})
    assert spec.function_aliases["LENGTH"][0] == "LEN"
