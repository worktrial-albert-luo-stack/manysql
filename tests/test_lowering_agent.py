"""Tests for the IR equivalence battery and lowering codegen agent."""

from __future__ import annotations

import pytest

from manysql.codegen.grammar_emit import emit_grammar
from manysql.codegen.ir_battery import (
    build_ir_battery,
    default_schemas,
    validate_lowering,
)
from manysql.codegen.lowering_agent import (
    LoweringAgentResult,
    _load_module,
    _strip_code_fences,
    generate_lowering,
)
from manysql.codegen.lowering_emit import emit_lowering
from manysql.llm.client import (
    LLMBackend,
    LLMClient,
    LLMConfig,
    LLMResponse,
    NullLLMClient,
)
from manysql.spec.dialect import DialectSpec, JoinSyntax, SurfaceSpec
from manysql.spec.examples import EXAMPLE_SPECS


# ---- IR battery shape ------------------------------------------------------


def test_ir_battery_pairs_ref_and_dialect_sql() -> None:
    spec = EXAMPLE_SPECS["moderate_keyword_swap"]
    items = build_ir_battery(spec)
    assert len(items) >= 20
    # Reference SQL is unchanged; dialect SQL has the spec's surface applied.
    for item in items:
        assert "SELECT" in item.ref_sql.upper() or "WITH" in item.ref_sql.upper()
        assert item.ref_sql != item.dialect_sql or item.label.startswith("scan")


def test_ir_battery_passes_for_reference_clone() -> None:
    spec = DialectSpec(name="reference_clone")
    grammar = emit_grammar(spec)
    module = _load_module(emit_lowering(spec), "_test_reference_clone")
    items = build_ir_battery(spec)
    semantics = spec.semantics.to_semantic_config()
    report = validate_lowering(
        grammar_text=grammar,
        lowering_module=module,
        semantics=semantics,
        items=items,
        schemas=default_schemas(),
    )
    assert report.ok, report.summary()


def test_ir_battery_passes_for_mild_postgres_ish() -> None:
    spec = EXAMPLE_SPECS["mild_postgres_ish"]
    grammar = emit_grammar(spec)
    module = _load_module(emit_lowering(spec), "_test_mild")
    items = build_ir_battery(spec)
    report = validate_lowering(
        grammar_text=grammar,
        lowering_module=module,
        semantics=spec.semantics.to_semantic_config(),
        items=items,
    )
    assert report.ok, report.summary()


def test_ir_battery_passes_for_moderate_keyword_swap() -> None:
    spec = EXAMPLE_SPECS["moderate_keyword_swap"]
    grammar = emit_grammar(spec)
    module = _load_module(emit_lowering(spec), "_test_moderate")
    items = build_ir_battery(spec)
    report = validate_lowering(
        grammar_text=grammar,
        lowering_module=module,
        semantics=spec.semantics.to_semantic_config(),
        items=items,
    )
    assert report.ok, report.summary()


def test_ir_battery_reports_grammar_failure_uniformly() -> None:
    spec = DialectSpec(name="reference_clone")
    items = build_ir_battery(spec)
    module = _load_module(emit_lowering(spec), "_test_grammar_failure")
    report = validate_lowering(
        grammar_text="start: NEVER_TERMINAL",
        lowering_module=module,
        semantics=spec.semantics.to_semantic_config(),
        items=items,
    )
    assert not report.ok
    assert len(report.divergences) == len(items)


# ---- Lowering agent --------------------------------------------------------


def test_generate_lowering_passes_for_surface_spec() -> None:
    spec = EXAMPLE_SPECS["mild_postgres_ish"]
    result = generate_lowering(spec)
    assert isinstance(result, LoweringAgentResult)
    assert result.ok
    assert result.attempts[0].source == "deterministic"


def test_generate_lowering_skips_llm_when_null_client() -> None:
    spec = EXAMPLE_SPECS["mild_postgres_ish"]
    client = NullLLMClient(canned_reply="should not be called")
    result = generate_lowering(spec, llm_client=client)
    assert result.ok
    assert client.record == []


def test_generate_lowering_records_skipped_for_structural_specs() -> None:
    """Structural specs raise NotImplementedError in the deterministic path;
    without an LLM the agent reports a 'skipped' attempt without crashing."""
    spec = DialectSpec(
        name="experimental",
        surface=SurfaceSpec(join_syntax=JoinSyntax.PIPELINED),
    )
    result = generate_lowering(spec)
    assert not result.ok
    assert result.attempts[0].source == "skipped"
    assert result.lowering_py == ""


def test_generate_lowering_uses_llm_when_deterministic_fails(monkeypatch) -> None:
    """If the deterministic lowering yields the wrong IR, the LLM should be
    called and its reply accepted when it works."""
    spec = EXAMPLE_SPECS["mild_postgres_ish"]
    good_lowering = emit_lowering(spec)

    monkeypatch.setattr(
        "manysql.codegen.lowering_agent.emit_lowering",
        lambda s: "def lower(tree, config, catalog):\n    return None\n",
    )

    class FakeClient(LLMClient):
        def __init__(self):
            self.config = LLMConfig(
                backend=LLMBackend.OPENAI,
                api_key="x",
                base_url="x",
                default_model="x",
            )
            self.calls = 0

        def chat(self, **kwargs):
            self.calls += 1
            return LLMResponse(
                text=f"```python\n{good_lowering}\n```",
                model="fake",
                backend=self.config.backend,
                prompt_tokens=0,
                completion_tokens=0,
            )

        def close(self):
            return None

    client = FakeClient()
    result = generate_lowering(spec, llm_client=client, max_iterations=2)
    assert result.ok
    assert client.calls == 1
    assert result.attempts[-1].source == "llm"


def test_strip_code_fences_passthrough() -> None:
    assert _strip_code_fences("plain") == "plain"
    assert _strip_code_fences("```py\nbody\n```") == "body"


# ---- force_llm: invoke the LLM even when the baseline already passes ------


def _fake_lowering_client(reply_text: str):
    """Build a non-Null LLMClient that returns `reply_text` on every chat()."""

    class FakeClient(LLMClient):
        def __init__(self) -> None:
            self.config = LLMConfig(
                backend=LLMBackend.OPENAI,
                api_key="x",
                base_url="x",
                default_model="x",
            )
            self.calls: list[dict] = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return LLMResponse(
                text=reply_text,
                model="fake",
                backend=self.config.backend,
                prompt_tokens=0,
                completion_tokens=0,
            )

        def close(self) -> None:
            return None

    return FakeClient()


def test_generate_lowering_force_llm_invokes_llm_on_passing_baseline() -> None:
    """force_llm=True must call the LLM even when the deterministic baseline
    already produces the reference IR for every battery query."""
    spec = EXAMPLE_SPECS["mild_postgres_ish"]
    good_lowering = emit_lowering(spec)

    client = _fake_lowering_client(good_lowering)
    result = generate_lowering(spec, llm_client=client, force_llm=True)

    assert result.ok
    assert len(client.calls) == 1, "expected exactly one forced LLM polish call"
    sources = [a.source for a in result.attempts]
    assert sources[0] == "deterministic"
    assert sources[-1] == "llm"
    assert result.attempts[-1].report.ok


def test_generate_lowering_force_llm_rolls_back_on_regression() -> None:
    """A forced LLM iteration that regresses against the IR battery must be
    discarded; the deterministic baseline is what ships."""
    spec = EXAMPLE_SPECS["mild_postgres_ish"]
    deterministic = emit_lowering(spec)

    # A `lower` that returns None will diverge from the reference IR for
    # every battery item.
    bad_reply = "def lower(tree, config, catalog):\n    return None\n"
    client = _fake_lowering_client(bad_reply)

    result = generate_lowering(spec, llm_client=client, force_llm=True)

    assert client.calls, "LLM should have been called on force_llm=True"
    assert result.ok, "regressed LLM output must not become the final lowering"
    assert result.lowering_py == deterministic
    sources = [a.source for a in result.attempts]
    assert sources[0] == "deterministic"
    assert "llm" in sources
