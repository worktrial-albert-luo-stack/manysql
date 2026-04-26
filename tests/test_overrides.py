"""Tests for the operator/function overrides scaffold."""

from __future__ import annotations

import polars as pl
import pytest

from manysql.codegen.overrides_emit import emit_overrides
from manysql.codegen.overrides_loader import (
    OverrideImportError,
    load_overrides,
)
from manysql.executor import execute
from manysql.executor.engine import PlanExecutor
from manysql.ir.expr import FuncCall, Literal
from manysql.ir.plan import Project
from manysql.ir.types import IRType, TypeKind
from manysql.spec.examples import EXAMPLE_SPECS
from manysql.spec.semantics import SemanticConfig


# ---- Emitter --------------------------------------------------------------


def test_emit_overrides_for_default_spec_is_loadable() -> None:
    spec = EXAMPLE_SPECS["mild_postgres_ish"]
    text = emit_overrides(spec)
    assert "FUNCTIONS" in text
    assert "OPERATORS" in text
    module = load_overrides(text, fullname="_test_overrides_default")
    assert module.FUNCTIONS == {}
    assert module.OPERATORS == {}


def test_emit_overrides_records_function_aliases_as_comments() -> None:
    spec = EXAMPLE_SPECS["aggressive_alien"]
    text = emit_overrides(spec)
    assert "Surface aliases" in text
    assert "FIRSTNONNIL" in text


# ---- Sandboxed loader -----------------------------------------------------


def test_loader_blocks_disallowed_imports() -> None:
    bad = "import os\nFUNCTIONS = {}\nOPERATORS = {}\n"
    with pytest.raises(OverrideImportError):
        load_overrides(bad, fullname="_test_overrides_bad")


def test_loader_allows_polars_import() -> None:
    good = (
        "import polars as pl\n"
        "FUNCTIONS = {'IDENTITY': lambda args, sem: args[0]}\n"
        "OPERATORS = {}\n"
    )
    module = load_overrides(good, fullname="_test_overrides_good")
    assert "IDENTITY" in module.FUNCTIONS


def test_loader_strips_dangerous_builtins() -> None:
    src = (
        "FUNCTIONS = {}\n"
        "OPERATORS = {}\n"
        "missing = [name for name in ('open', 'eval', 'exec') if name in dir(__builtins__)]\n"
    )
    module = load_overrides(src, fullname="_test_overrides_strip")
    assert module.missing == []


# ---- Executor wiring ------------------------------------------------------


def _identity_fn(args, semantics):  # noqa: ARG001
    return args[0]


def _double_fn(args, semantics):  # noqa: ARG001
    return args[0] * 2


def test_executor_consults_overrides_dict_for_unknown_funccall() -> None:
    """When a FuncCall name isn't a built-in, the executor looks it up in
    `overrides.FUNCTIONS` and calls it with `(args, semantics)`."""

    class FakeOverrides:
        FUNCTIONS = {"DOUBLE_IT": _double_fn}
        OPERATORS = {}

    plan = Project(
        input=_one_row_scan(),
        projections=(
            (
                "doubled",
                FuncCall(
                    name="DOUBLE_IT",
                    args=(Literal(value=21, type=IRType(kind=TypeKind.INT)),),
                ),
            ),
        ),
        output_types=(IRType(kind=TypeKind.INT),),
    )
    executor = PlanExecutor(
        catalog=_one_row_catalog(),
        semantics=SemanticConfig.reference(),
        overrides=FakeOverrides(),
    )
    out = executor.execute(plan)
    assert out["doubled"].to_list() == [42]


def test_executor_falls_back_to_overrides_via_operators_dict() -> None:
    class FakeOverrides:
        FUNCTIONS = {}
        OPERATORS = {"DOUBLE_IT": _double_fn}

    plan = Project(
        input=_one_row_scan(),
        projections=(
            (
                "doubled",
                FuncCall(
                    name="double_it",  # case-insensitive lookup
                    args=(Literal(value=11, type=IRType(kind=TypeKind.INT)),),
                ),
            ),
        ),
        output_types=(IRType(kind=TypeKind.INT),),
    )
    out = execute(
        plan,
        SemanticConfig.reference(),
        _one_row_catalog(),
        overrides=FakeOverrides(),
    )
    assert out["doubled"].to_list() == [22]


def test_executor_raises_when_funccall_unresolved() -> None:
    plan = Project(
        input=_one_row_scan(),
        projections=(
            (
                "x",
                FuncCall(
                    name="UNKNOWN_FN",
                    args=(Literal(value=1, type=IRType(kind=TypeKind.INT)),),
                ),
            ),
        ),
        output_types=(IRType(kind=TypeKind.INT),),
    )
    with pytest.raises(NotImplementedError):
        execute(plan, SemanticConfig.reference(), _one_row_catalog())


# ---- helpers --------------------------------------------------------------


def _one_row_catalog() -> dict[str, pl.DataFrame]:
    return {"t": pl.DataFrame({"x": [1]})}


def _one_row_scan():
    from manysql.ir.plan import ColumnSchema, Scan

    return Scan(
        table_name="t",
        columns=(
            ColumnSchema(
                name="x",
                type=IRType(kind=TypeKind.INT, nullable=True),
                qualifier=None,
            ),
        ),
    )
