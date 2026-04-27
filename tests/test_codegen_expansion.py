"""Tests for the LLM-lane expansion (Wave 1).

Covers the new deterministic emitter passes (operators, wildcard,
semicolon, identifier-quote folding), the expanded ``apply_surface``
rewrites, the rejection battery, function-alias propagation into the
emitted ``SemanticConfig``, card-conformance gating, and the LLM
rollback path when an LLM regresses against the rejection battery.

These tests are deliberately narrow: each one nails one new behavior so
a regression points at the specific emitter or rewriter it touched.
"""

from __future__ import annotations

import json

import pytest
from lark import Lark

from manysql.codegen.battery_emit import emit_battery_json
from manysql.codegen.card_conformance import (
    build_card_examples,
    validate_card_conformance,
)
from manysql.codegen.config_emit import compose_semantic_config, emit_semantic_config
from manysql.codegen.grammar_agent import generate_grammar
from manysql.codegen.grammar_emit import emit_grammar
from manysql.codegen.parse_battery import (
    apply_surface,
    build_parse_battery,
    build_rejection_battery,
    validate_grammar,
    validate_rejection_battery,
)
from manysql.llm.client import LLMBackend, LLMClient, LLMConfig, LLMResponse
from manysql.spec.dialect import (
    DialectSpec,
    IdentifierQuote,
    NullLiteral,
    SetOpPrecedence,
    StringQuote,
    SurfaceSpec,
    WildcardChar,
)


# ----------------------------------------------------------------------
# apply_surface — the rewriter must propagate every divergent axis through
# canonical reference SQL so the parse battery actually exercises it.
# ----------------------------------------------------------------------


def test_apply_surface_rewrites_operator_neq() -> None:
    surface = SurfaceSpec(neq_op=["!<>"])
    out = apply_surface("SELECT id FROM employees WHERE dept_id <> 10", surface)
    assert "!<>" in out
    assert "<>" not in out.replace("!<>", "")


def test_apply_surface_rewrites_mod_keyword() -> None:
    surface = SurfaceSpec(mod_op="MOD")
    out = apply_surface("SELECT id FROM employees WHERE salary % 100 = 0", surface)
    assert " MOD " in out or "MOD " in out
    assert "%" not in out


def test_apply_surface_rewrites_concat_to_alt_op() -> None:
    surface = SurfaceSpec(concat_op="..")
    out = apply_surface("SELECT name || '!' AS shouted FROM employees", surface)
    assert ".." in out
    assert "||" not in out


def test_apply_surface_skips_concat_when_overloaded_with_add() -> None:
    """concat_op=='+' and add_op=='+' would make ``||`` ambiguous in the
    grammar; the rewriter deliberately leaves ``||`` so the canonical concat
    query still parses identically against both reference and dialect."""
    surface = SurfaceSpec(concat_op="+")
    out = apply_surface("SELECT name || '!' AS shouted FROM employees", surface)
    assert "||" in out, "rewriter should preserve || when concat overloads +"


def test_apply_surface_rewrites_null_safe_eq() -> None:
    surface = SurfaceSpec(null_safe_eq_op="<=>")
    out = apply_surface(
        "SELECT id FROM employees WHERE dept_id IS NOT DISTINCT FROM 10",
        surface,
    )
    assert "<=>" in out
    assert "IS NOT DISTINCT FROM" not in out


def test_apply_surface_rewrites_wildcard() -> None:
    surface = SurfaceSpec(wildcard_char=WildcardChar.AT)
    out = apply_surface("SELECT * FROM employees", surface)
    assert "@" in out
    assert "*" not in out


def test_apply_surface_rewrites_function_alias_primary() -> None:
    surface = SurfaceSpec(function_aliases={"COALESCE": ["NVL", "IFNULL"]})
    out = apply_surface("SELECT COALESCE(dept_id, 0) FROM employees", surface)
    assert "NVL(" in out
    assert "COALESCE(" not in out


def test_apply_surface_rewrites_string_quote_to_double() -> None:
    surface = SurfaceSpec(string_quote=StringQuote.DOUBLE)
    out = apply_surface("SELECT id FROM employees WHERE name = 'Alice'", surface)
    assert '"Alice"' in out
    assert "'Alice'" not in out


def test_apply_surface_rewrites_identifier_quote_to_backtick() -> None:
    surface = SurfaceSpec(identifier_quote=IdentifierQuote.BACKTICK)
    out = apply_surface('SELECT "id" FROM employees', surface)
    assert "`id`" in out


# ----------------------------------------------------------------------
# Grammar emitter — operator / wildcard / semicolon patches.
# ----------------------------------------------------------------------


def test_grammar_emit_drops_old_neq_branches() -> None:
    spec = DialectSpec(name="t", surface=SurfaceSpec(neq_op=["!<>"]))
    grammar = emit_grammar(spec)
    parser = Lark(grammar, start="start", parser="earley")
    parser.parse("SELECT id FROM employees WHERE dept_id !<> 10")
    with pytest.raises(Exception):
        parser.parse("SELECT id FROM employees WHERE dept_id <> 10")
    with pytest.raises(Exception):
        parser.parse("SELECT id FROM employees WHERE dept_id != 10")


def test_grammar_emit_drops_old_mod_branch() -> None:
    spec = DialectSpec(name="t", surface=SurfaceSpec(mod_op="MOD"))
    grammar = emit_grammar(spec)
    parser = Lark(grammar, start="start", parser="earley")
    parser.parse("SELECT id FROM employees WHERE salary MOD 100 = 0")
    with pytest.raises(Exception):
        parser.parse("SELECT id FROM employees WHERE salary % 100 = 0")


def test_grammar_emit_swaps_wildcard_char() -> None:
    spec = DialectSpec(name="t", surface=SurfaceSpec(wildcard_char=WildcardChar.AT))
    grammar = emit_grammar(spec)
    parser = Lark(grammar, start="start", parser="earley")
    parser.parse("SELECT @ FROM employees")
    parser.parse("SELECT COUNT(@) FROM employees")
    with pytest.raises(Exception):
        parser.parse("SELECT * FROM employees")


def test_grammar_emit_requires_semicolon() -> None:
    spec = DialectSpec(name="t", surface=SurfaceSpec(requires_semicolon=True))
    grammar = emit_grammar(spec)
    parser = Lark(grammar, start="start", parser="earley")
    parser.parse("SELECT id FROM employees;")
    with pytest.raises(Exception):
        parser.parse("SELECT id FROM employees")


def test_grammar_emit_identifier_quote_backtick_swaps_for_ansi() -> None:
    """The IDENTIFIER terminal is a single regex; swapping to backtick means
    the grammar accepts backticks but no longer accepts ANSI double-quoted
    identifiers (the divergence the dialect actually advertises)."""
    spec = DialectSpec(
        name="t", surface=SurfaceSpec(identifier_quote=IdentifierQuote.BACKTICK)
    )
    grammar = emit_grammar(spec)
    parser = Lark(grammar, start="start", parser="earley")
    parser.parse("SELECT `id` FROM employees")
    with pytest.raises(Exception):
        parser.parse('SELECT "id" FROM employees')


# ----------------------------------------------------------------------
# Rejection battery — must-fail items per non-default knob.
# ----------------------------------------------------------------------


def test_rejection_battery_empty_for_default_spec() -> None:
    spec = DialectSpec(name="default")
    items = build_rejection_battery(spec)
    assert items == []


def test_rejection_battery_catches_overpermissive_grammar() -> None:
    """If we hand-craft a grammar that *still* admits ANSI ``%`` even though
    the spec says ``mod_op=MOD``, the rejection battery flags it."""
    spec = DialectSpec(name="t", surface=SurfaceSpec(mod_op="MOD"))
    items = build_rejection_battery(spec)
    assert any(item.label == "reject_ansi_mod" for item in items)
    overpermissive_grammar = emit_grammar(DialectSpec(name="t"))
    report = validate_rejection_battery(overpermissive_grammar, items)
    assert not report.ok
    assert any(f.label == "reject_ansi_mod" for f in report.failures)


def test_rejection_battery_passes_for_correct_grammar() -> None:
    spec = DialectSpec(name="t", surface=SurfaceSpec(mod_op="MOD"))
    items = build_rejection_battery(spec)
    grammar = emit_grammar(spec)
    report = validate_rejection_battery(grammar, items)
    assert report.ok, report.summary()


def test_rejection_battery_covers_keyword_renames() -> None:
    spec = DialectSpec(name="t", surface=SurfaceSpec(select_keyword="PICK"))
    items = build_rejection_battery(spec)
    labels = {item.label for item in items}
    assert "reject_select_keyword" in labels


# ----------------------------------------------------------------------
# function_aliases must reach the emitted SemanticConfig.
# ----------------------------------------------------------------------


def test_function_aliases_propagate_to_semantic_config() -> None:
    spec = DialectSpec(
        name="t",
        surface=SurfaceSpec(function_aliases={"COALESCE": ["NVL", "IFNULL"]}),
    )
    cfg = compose_semantic_config(spec)
    assert cfg.function_aliases == {"COALESCE": ["NVL", "IFNULL"]}


def test_function_aliases_round_trip_through_semantics_json() -> None:
    spec = DialectSpec(
        name="t",
        surface=SurfaceSpec(function_aliases={"LENGTH": ["LEN"]}),
    )
    payload = json.loads(emit_semantic_config(spec))
    assert payload["function_aliases"] == {"LENGTH": ["LEN"]}


def test_set_op_precedence_propagates_to_semantic_config() -> None:
    spec = DialectSpec(
        name="t",
        surface=SurfaceSpec(set_op_precedence=SetOpPrecedence.EXCEPT_INTERSECT_TIGHTER),
    )
    cfg = compose_semantic_config(spec)
    assert cfg.set_op_precedence.value == "except_intersect_tighter"


# ----------------------------------------------------------------------
# Axis-targeted parse battery items only fire when the axis diverges.
# ----------------------------------------------------------------------


def test_axis_battery_items_empty_for_default_spec() -> None:
    spec = DialectSpec(name="default")
    items = build_parse_battery(spec)
    labels = {item.label for item in items}
    assert not any(label.startswith("axis_") for label in labels)


def test_axis_battery_items_fire_on_divergent_spec() -> None:
    spec = DialectSpec(name="t", surface=SurfaceSpec(mod_op="MOD"))
    items = build_parse_battery(spec)
    labels = {item.label for item in items}
    assert "axis_mod" in labels


# ----------------------------------------------------------------------
# Every axis-targeted item must round-trip through the IR battery against
# the reference lowering. A bad table name or a malformed canonical SQL
# would surface here as a reference-side ``SemanticError``, even before
# the dialect-side lowering runs. This is the gate that would have caught
# ``axis_setop_precedence`` referencing a phantom ``projects`` table.
# ----------------------------------------------------------------------


_AXIS_ROUND_TRIP_SPECS: list[tuple[str, DialectSpec]] = [
    (
        "axis_quoted_ident_select",
        DialectSpec(
            name="t",
            surface=SurfaceSpec(identifier_quote=IdentifierQuote.BACKTICK),
        ),
    ),
    (
        "axis_null_safe_eq",
        DialectSpec(name="t", surface=SurfaceSpec(null_safe_eq_op="<=>")),
    ),
    (
        "axis_function_alias_coalesce",
        DialectSpec(
            name="t",
            surface=SurfaceSpec(function_aliases={"COALESCE": ["NVL"]}),
        ),
    ),
    (
        "axis_function_alias_length",
        DialectSpec(
            name="t",
            surface=SurfaceSpec(function_aliases={"LENGTH": ["LEN"]}),
        ),
    ),
    (
        "axis_wildcard_select",
        DialectSpec(name="t", surface=SurfaceSpec(wildcard_char=WildcardChar.AT)),
    ),
    (
        "axis_mod",
        DialectSpec(name="t", surface=SurfaceSpec(mod_op="MOD")),
    ),
    (
        "axis_concat",
        DialectSpec(name="t", surface=SurfaceSpec(concat_op="..")),
    ),
    (
        "axis_setop_precedence",
        DialectSpec(
            name="t",
            surface=SurfaceSpec(
                set_op_precedence=SetOpPrecedence.EXCEPT_INTERSECT_TIGHTER
            ),
        ),
    ),
    (
        "axis_comparison_neq",
        DialectSpec(name="t", surface=SurfaceSpec(neq_op=["!<>"])),
    ),
    (
        "axis_string_literal",
        DialectSpec(
            name="t", surface=SurfaceSpec(string_quote=StringQuote.DOUBLE)
        ),
    ),
]


@pytest.mark.parametrize(
    "axis_label,spec",
    _AXIS_ROUND_TRIP_SPECS,
    ids=[label for label, _ in _AXIS_ROUND_TRIP_SPECS],
)
def test_axis_battery_item_round_trips_through_ir_battery(
    axis_label: str, spec: DialectSpec
) -> None:
    """Every axis-targeted item must fully round-trip: dialect lowering of the
    rewritten SQL must equal the reference lowering of the canonical SQL.

    This catches three classes of bug at once:
      - Canonical SQL that references a non-existent table (the
        ``projects`` regression).
      - ``apply_surface`` forgetting an axis it should rewrite (the
        dialect parser wouldn't accept the rewritten form).
      - The reference lowering missing an axis bridge (``set_op_precedence``,
        ``function_aliases``) that ``compose_semantic_config`` is supposed
        to propagate.
    """
    import importlib.util
    import sys
    import tempfile

    from manysql.codegen.config_emit import compose_semantic_config
    from manysql.codegen.ir_battery import build_ir_battery, validate_lowering
    from manysql.codegen.lowering_emit import emit_lowering

    grammar = emit_grammar(spec)
    lowering_src = emit_lowering(spec)
    cfg = compose_semantic_config(spec)

    with tempfile.NamedTemporaryFile(
        suffix=".py", delete=False, mode="w"
    ) as fh:
        fh.write(lowering_src)
        path = fh.name
    try:
        mod_spec = importlib.util.spec_from_file_location(
            f"_axis_round_trip_{axis_label}", path
        )
        assert mod_spec is not None and mod_spec.loader is not None
        module = importlib.util.module_from_spec(mod_spec)
        sys.modules[mod_spec.name] = module
        mod_spec.loader.exec_module(module)
        items = build_ir_battery(spec)
        assert any(it.label == axis_label for it in items), (
            f"expected axis_battery_items({spec.name}) to include "
            f"{axis_label!r}; got {[it.label for it in items]}"
        )
        report = validate_lowering(
            grammar_text=grammar,
            lowering_module=module,
            semantics=cfg,
            items=items,
        )
    finally:
        sys.modules.pop(mod_spec.name, None) if mod_spec else None
        try:
            import os

            os.unlink(path)
        except OSError:
            pass

    assert report.ok, (
        f"IR battery divergences for spec exercising {axis_label}: "
        f"{report.summary()}; "
        + "; ".join(
            f"{d.label}: {d.error}" for d in report.divergences
        )
    )


# ----------------------------------------------------------------------
# Card conformance gate.
# ----------------------------------------------------------------------


def test_card_conformance_passes_for_reference_grammar() -> None:
    spec = DialectSpec(name="t")
    grammar = emit_grammar(spec)
    examples = build_card_examples(spec)
    report = validate_card_conformance(grammar, examples)
    assert report.ok, report.summary()


def test_card_conformance_catches_drift() -> None:
    """A grammar that doesn't accept the dialect's advertised wildcard char
    must surface a card warning for the wildcard skeleton."""
    spec = DialectSpec(
        name="t", surface=SurfaceSpec(wildcard_char=WildcardChar.AT)
    )
    grammar = emit_grammar(DialectSpec(name="t"))  # use ANSI grammar instead
    examples = build_card_examples(spec)
    report = validate_card_conformance(grammar, examples)
    assert not report.ok
    assert any(w.label == "card_wildcard" for w in report.warnings)


def test_card_warnings_appear_in_metadata_json() -> None:
    """Wired-in path: build_package_bundle writes card warnings into
    metadata.json so external consumers can triage drift without re-running
    the parser."""
    from manysql.codegen.pipeline import build_package_bundle

    spec = DialectSpec(name="t")
    bundle, _g, _l, card_report = build_package_bundle(spec)
    payload = json.loads(bundle.metadata_json)
    assert "card_warnings" in payload
    assert payload["card_warnings"] == []
    assert card_report.ok


# ----------------------------------------------------------------------
# LLM rollback when a forced polish iteration regresses against the
# rejection battery (the new safety net).
# ----------------------------------------------------------------------


class _FakeLLMClient(LLMClient):
    """Test double that returns a fixed grammar reply."""

    def __init__(self, reply: str) -> None:
        self.config = LLMConfig(
            backend=LLMBackend.OPENAI,
            api_key="x",
            base_url="x",
            default_model="x",
        )
        self._reply = reply
        self.calls = 0

    def chat(self, *, system, user, temperature=0.0, max_tokens=None):  # noqa: D401, ARG002
        self.calls += 1
        return LLMResponse(
            text=self._reply,
            model=self.config.default_model,
            backend=self.config.backend,
            prompt_tokens=0,
            completion_tokens=0,
            raw={},
        )


def test_force_llm_rolls_back_when_rejection_battery_fails() -> None:
    """If a forced LLM polish iteration produces a grammar that admits the
    very ANSI form the dialect claims to reject, the agent must revert to
    the deterministic baseline rather than ship the regression."""
    spec = DialectSpec(name="t", surface=SurfaceSpec(mod_op="MOD"))
    deterministic_grammar = emit_grammar(spec)
    deterministic_report = validate_grammar(
        deterministic_grammar, build_parse_battery(spec)
    )
    deterministic_rejection = validate_rejection_battery(
        deterministic_grammar, build_rejection_battery(spec)
    )
    assert deterministic_report.ok
    assert deterministic_rejection.ok

    # The "regressed" grammar is the ANSI/reference grammar — it parses
    # ``%`` as MOD, defeating the dialect's divergence.
    bad_grammar = emit_grammar(DialectSpec(name="t"))
    client = _FakeLLMClient(reply=bad_grammar)

    result = generate_grammar(
        spec,
        llm_client=client,
        max_iterations=1,
        force_llm=True,
    )

    # The agent ran the LLM (force_llm) but rolled back to the deterministic
    # grammar because the LLM's reply failed the rejection battery.
    assert client.calls == 1
    assert result.grammar == deterministic_grammar
    assert result.report.ok
    assert result.rejection_report is not None and result.rejection_report.ok


# ----------------------------------------------------------------------
# Battery JSON now records the rejection report.
# ----------------------------------------------------------------------


def test_battery_json_includes_rejection_section() -> None:
    spec = DialectSpec(name="t", surface=SurfaceSpec(mod_op="MOD"))
    parse_items = build_parse_battery(spec)
    grammar = emit_grammar(spec)
    parse_report = validate_grammar(grammar, parse_items)
    rejection_items = build_rejection_battery(spec)
    rejection_report = validate_rejection_battery(grammar, rejection_items)
    # Provide trivial empty IR to focus the assertion on the rejection block.
    from manysql.codegen.ir_battery import IRBatteryItem, IREquivalenceReport

    ir_items: list[IRBatteryItem] = []
    ir_report = IREquivalenceReport(items=ir_items, divergences=[])
    payload = json.loads(
        emit_battery_json(
            parse_items=parse_items,
            parse_report=parse_report,
            ir_items=ir_items,
            ir_report=ir_report,
            rejection_items=rejection_items,
            rejection_report=rejection_report,
        )
    )
    assert "rejection" in payload
    assert payload["rejection"]["validation"]["ok"] is True
    labels = {item["label"] for item in payload["rejection"]["items"]}
    assert "reject_ansi_mod" in labels
