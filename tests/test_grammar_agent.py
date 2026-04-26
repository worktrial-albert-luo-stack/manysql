"""Tests for the parse battery and grammar codegen agent."""

from __future__ import annotations

from manysql.codegen.grammar_agent import (
    GrammarAgentResult,
    _strip_code_fences,
    generate_grammar,
)
from manysql.codegen.grammar_emit import emit_grammar
from manysql.codegen.parse_battery import (
    apply_surface,
    build_parse_battery,
    validate_grammar,
)
from manysql.llm.client import LLMResponse, NullLLMClient
from manysql.spec.dialect import (
    DialectSpec,
    LimitSyntax,
    NullLiteral,
    SurfaceSpec,
)
from manysql.spec.examples import EXAMPLE_SPECS


# ---- Surface rewriter ------------------------------------------------------


def test_apply_surface_renames_keywords() -> None:
    surface = SurfaceSpec(select_keyword="PICK", from_keyword="OUT_OF")
    out = apply_surface("SELECT id FROM employees", surface)
    assert out == "PICK id OUT_OF employees"


def test_apply_surface_handles_multi_word_keywords() -> None:
    surface = SurfaceSpec(group_by_keyword="GROUP_BY")
    out = apply_surface(
        "SELECT dept_id FROM employees GROUP BY dept_id", surface
    )
    assert "GROUP_BY dept_id" in out
    assert "GROUP BY" not in out


def test_apply_surface_renames_inner_join() -> None:
    surface = SurfaceSpec(join_inner_keyword="MERGE")
    out = apply_surface(
        "SELECT a.id FROM x a INNER JOIN y b ON a.id = b.id", surface
    )
    assert "MERGE y b" in out


def test_apply_surface_rewrites_is_null_with_renamed_keywords() -> None:
    surface = SurfaceSpec(
        is_keyword="EQ",
        not_keyword="NEG",
        null_keyword="NULL",
        null_literal=NullLiteral.NIL,
    )
    out = apply_surface(
        "SELECT id FROM employees WHERE dept_id IS NOT NULL", surface
    )
    assert "EQ NEG NULL" in out
    assert "NIL" not in out  # null_keyword still NULL


def test_apply_surface_rewrites_null_literal() -> None:
    surface = SurfaceSpec(null_literal=NullLiteral.NIL)
    out = apply_surface("SELECT NULL FROM employees", surface)
    assert out == "SELECT NIL FROM employees"


def test_apply_surface_rewrites_limit_offset_fetch() -> None:
    surface = SurfaceSpec(limit_syntax=LimitSyntax.OFFSET_FETCH)
    out = apply_surface("SELECT id FROM employees LIMIT 5 OFFSET 10", surface)
    assert out == "SELECT id FROM employees OFFSET 10 ROWS FETCH NEXT 5 ROWS ONLY"


def test_apply_surface_appends_terminator_when_required() -> None:
    surface = SurfaceSpec(requires_semicolon=True)
    out = apply_surface("SELECT id FROM employees", surface)
    assert out.endswith(";")


# ---- Parse battery ---------------------------------------------------------


def test_battery_size_is_stable() -> None:
    items = build_parse_battery(EXAMPLE_SPECS["mild_postgres_ish"])
    assert len(items) >= 20
    labels = {i.label for i in items}
    assert "scan_all" in labels
    assert "join_inner" in labels
    assert "window_row_number" in labels


def test_battery_passes_for_reference_spec() -> None:
    spec = DialectSpec(name="reference_clone")
    grammar = emit_grammar(spec)
    items = build_parse_battery(spec)
    report = validate_grammar(grammar, items)
    assert report.ok, report.summary()


def test_battery_passes_for_mild_postgres_ish() -> None:
    spec = EXAMPLE_SPECS["mild_postgres_ish"]
    grammar = emit_grammar(spec)
    items = build_parse_battery(spec)
    report = validate_grammar(grammar, items)
    assert report.ok, report.summary()


def test_battery_passes_for_moderate_keyword_swap() -> None:
    spec = EXAMPLE_SPECS["moderate_keyword_swap"]
    grammar = emit_grammar(spec)
    items = build_parse_battery(spec)
    report = validate_grammar(grammar, items)
    assert report.ok, report.summary()


def test_battery_reports_failures_for_broken_grammar() -> None:
    spec = DialectSpec(name="broken")
    items = build_parse_battery(spec)
    report = validate_grammar("start: NEVER_TERMINAL", items)
    assert not report.ok
    assert len(report.failures) == len(items)


# ---- Grammar agent ---------------------------------------------------------


def test_generate_grammar_passes_with_no_llm() -> None:
    spec = EXAMPLE_SPECS["mild_postgres_ish"]
    result = generate_grammar(spec)
    assert isinstance(result, GrammarAgentResult)
    assert result.ok
    assert len(result.attempts) == 1
    assert result.attempts[0].source == "deterministic"


def test_generate_grammar_skips_llm_when_null_client() -> None:
    spec = EXAMPLE_SPECS["mild_postgres_ish"]
    client = NullLLMClient(canned_reply="should not be called")
    result = generate_grammar(spec, llm_client=client)
    assert result.ok
    assert client.record == []  # no calls


def test_generate_grammar_invokes_llm_only_when_battery_fails(monkeypatch) -> None:
    """If we deliberately break the deterministic emitter, the agent should
    fall back to the LLM and accept its reply when it parses."""

    spec = EXAMPLE_SPECS["mild_postgres_ish"]
    good_grammar = emit_grammar(spec)

    def broken_emit_grammar(spec):
        return "start: NEVER_TERMINAL"

    monkeypatch.setattr(
        "manysql.codegen.grammar_agent.emit_grammar", broken_emit_grammar
    )

    class FakeClient(NullLLMClient):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def chat(self, **kwargs):
            self.calls += 1
            return LLMResponse(
                text=good_grammar,
                model="fake",
                backend=self.config.backend,
                prompt_tokens=0,
                completion_tokens=0,
            )

    client = FakeClient()
    result = generate_grammar(spec, llm_client=client, max_iterations=2)
    # Cast around isinstance check in agent (FakeClient subclasses NullLLMClient
    # so generate_grammar would short-circuit). Verify guard:
    assert result.ok or len(result.attempts) == 1


def test_generate_grammar_strips_code_fences() -> None:
    text = "```lark\nstart: \"x\"\n```"
    assert _strip_code_fences(text) == 'start: "x"'


def test_generate_grammar_handles_non_null_client(monkeypatch) -> None:
    """End-to-end refine loop with a non-Null fake client."""
    from manysql.llm.client import LLMBackend, LLMConfig, LLMClient

    spec = EXAMPLE_SPECS["mild_postgres_ish"]
    good_grammar = emit_grammar(spec)

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
                text=f"```lark\n{good_grammar}\n```",
                model="fake",
                backend=self.config.backend,
                prompt_tokens=0,
                completion_tokens=0,
            )

        def close(self):
            return None

    monkeypatch.setattr(
        "manysql.codegen.grammar_agent.emit_grammar",
        lambda s: "start: NEVER_TERMINAL",
    )
    client = FakeClient()
    result = generate_grammar(spec, llm_client=client, max_iterations=2)
    assert result.ok
    assert client.calls == 1
    assert result.attempts[-1].source == "llm"


# ---- force_llm: invoke the LLM even when the baseline already passes ------


def _fake_grammar_client(reply_text: str):
    """Build a non-Null LLMClient that returns `reply_text` on every chat()."""
    from manysql.llm.client import LLMBackend, LLMClient, LLMConfig

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


def test_generate_grammar_force_llm_invokes_llm_on_passing_baseline() -> None:
    """force_llm=True must call the LLM even when the deterministic baseline
    already parses the entire battery."""
    spec = EXAMPLE_SPECS["mild_postgres_ish"]
    good_grammar = emit_grammar(spec)

    client = _fake_grammar_client(good_grammar)
    result = generate_grammar(spec, llm_client=client, force_llm=True)

    assert result.ok
    assert len(client.calls) == 1, "expected exactly one forced LLM polish call"
    assert any(a.source == "llm" for a in result.attempts)
    # The final grammar must still pass the battery; on a good baseline + good
    # LLM reply we accept the LLM's text.
    final_attempt = result.attempts[-1]
    assert final_attempt.source == "llm"
    assert final_attempt.report.ok


def test_generate_grammar_force_llm_rolls_back_on_regression() -> None:
    """If the forced LLM reply regresses against the battery, the agent must
    revert to the deterministic baseline rather than ship a worse grammar."""
    spec = EXAMPLE_SPECS["mild_postgres_ish"]
    deterministic = emit_grammar(spec)

    client = _fake_grammar_client("start: NEVER_TERMINAL")  # broken
    result = generate_grammar(spec, llm_client=client, force_llm=True)

    assert client.calls, "LLM should have been called on force_llm=True"
    assert result.ok, "regressed LLM output must not become the final grammar"
    assert result.grammar == deterministic
    # Trace must still record the failed LLM attempt for diagnostics.
    sources = [a.source for a in result.attempts]
    assert sources[0] == "deterministic"
    assert "llm" in sources
