"""Tests for the per-dialect executor effects lane.

A dialect can ship an `effects.py` module that exposes
`EFFECTS: dict[str, Callable]`. Names in the dict swap canonical
implementations of executor decision points (see
`manysql/codegen/effects_emit.py` for the v1 registry).

We exercise the lane via the synthetic `_test_ci_eq` dialect: its
`effects.py` installs a `text_eq` handler that lowercases both sides of
`=` so equality is collation-insensitive.
"""

from __future__ import annotations

import polars as pl
import pytest

from manysql.dialects import DialectRegistry
from manysql.executor import execute
from manysql.ir.expr import BinaryOp, ColumnRef, Literal, Op
from manysql.ir.plan import ColumnSchema, Filter, Scan
from manysql.ir.types import TEXT
from manysql.spec.semantics import SemanticConfig


@pytest.fixture(scope="module")
def ci_eq_engine():
    return DialectRegistry().load("_test_ci_eq")


def _ci_catalog() -> dict[str, pl.DataFrame]:
    return {
        "users": pl.DataFrame({"name": ["Alice", "BOB", "carol", "Dave"]}),
    }


def _filter_eq_plan(rhs: str) -> Filter:
    scan = Scan(
        table_name="users",
        columns=(ColumnSchema(name="name", type=TEXT),),
    )
    return Filter(
        input=scan,
        predicate=BinaryOp(
            op=Op.EQ,
            left=ColumnRef(name="name"),
            right=Literal(value=rhs, type=TEXT),
        ),
    )


def test_engine_loads_effects_module(ci_eq_engine) -> None:
    assert ci_eq_engine.effects is not None
    assert "text_eq" in ci_eq_engine.effects.EFFECTS
    assert "text_neq" in ci_eq_engine.effects.EFFECTS


def test_eq_without_effects_is_case_sensitive() -> None:
    """Sanity check: canonical executor uses the default `==`."""
    plan = _filter_eq_plan("alice")
    out = execute(plan, SemanticConfig.reference(), _ci_catalog())
    assert out["name"].to_list() == []


def test_eq_with_ci_effect_matches_alice_regardless_of_case(
    ci_eq_engine,
) -> None:
    plan = _filter_eq_plan("alice")
    out = execute(
        plan,
        SemanticConfig.reference(),
        _ci_catalog(),
        effects=ci_eq_engine.effects,
    )
    assert out["name"].to_list() == ["Alice"]


def test_eq_with_ci_effect_matches_bob_regardless_of_case(ci_eq_engine) -> None:
    plan = _filter_eq_plan("Bob")
    out = execute(
        plan,
        SemanticConfig.reference(),
        _ci_catalog(),
        effects=ci_eq_engine.effects,
    )
    assert out["name"].to_list() == ["BOB"]


def test_neq_with_ci_effect_excludes_case_variants(ci_eq_engine) -> None:
    """`name <> 'alice'` should drop "Alice" under collation-insensitive eq."""
    scan = Scan(
        table_name="users",
        columns=(ColumnSchema(name="name", type=TEXT),),
    )
    plan = Filter(
        input=scan,
        predicate=BinaryOp(
            op=Op.NEQ,
            left=ColumnRef(name="name"),
            right=Literal(value="alice", type=TEXT),
        ),
    )
    out = execute(
        plan,
        SemanticConfig.reference(),
        _ci_catalog(),
        effects=ci_eq_engine.effects,
    )
    assert sorted(out["name"].to_list()) == ["BOB", "Dave", "carol"]


def test_effects_registry_missing_key_falls_through() -> None:
    """An effects module with EFFECTS = {} keeps canonical behavior."""

    class EmptyEffects:
        EFFECTS: dict = {}

    plan = _filter_eq_plan("Alice")
    out = execute(
        plan,
        SemanticConfig.reference(),
        _ci_catalog(),
        effects=EmptyEffects(),
    )
    assert out["name"].to_list() == ["Alice"]
