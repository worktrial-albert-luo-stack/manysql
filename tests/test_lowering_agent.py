"""Tests for the IR equivalence battery and lowering codegen agent."""

from __future__ import annotations

import pytest

from manysql.codegen.config_emit import compose_semantic_config
from manysql.codegen.grammar_emit import emit_grammar
from manysql.codegen.ir_battery import (
    build_ir_battery,
    default_schemas,
    validate_lowering,
)
from manysql.codegen.lowering_agent import (
    LoweringAgentResult,
    _load_module,
    _parse_failure_trees,
    _refine_with_llm,
    _strip_code_fences,
    generate_lowering,
)
from manysql.codegen.lowering_emit import emit_lowering, emit_lowering_seed
from manysql.codegen.ir_battery import IRDivergence
from manysql.llm.client import (
    LLMBackend,
    LLMClient,
    LLMConfig,
    LLMResponse,
    NullLLMClient,
)
from manysql.spec.dialect import (
    CaseSyntax,
    DialectSpec,
    JoinSyntax,
    LimitSyntax,
    SurfaceSpec,
)
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
        semantics=compose_semantic_config(spec),
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
        semantics=compose_semantic_config(spec),
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


# ---- IR battery: skip impossible round-trips per surface ------------------


def test_ir_battery_keeps_limit_offset_for_capable_surfaces() -> None:
    """LIMIT_OFFSET and OFFSET_FETCH can express OFFSET, so the
    `limit_offset` battery item must be kept."""
    for syntax in (LimitSyntax.LIMIT_OFFSET, LimitSyntax.OFFSET_FETCH):
        spec = DialectSpec(name=f"capable_{syntax.value}", surface=SurfaceSpec(limit_syntax=syntax))
        labels = {it.label for it in build_ir_battery(spec)}
        assert "limit_offset" in labels, f"{syntax.value} must keep limit_offset"


def test_ir_battery_drops_limit_offset_for_offset_blind_surfaces() -> None:
    """HEAD_N/SAMPLE_N/TOP_N silently drop OFFSET in the surface rewrite
    (`_format_limit_clause`), so any plan with offset != 0 cannot
    round-trip. The IR battery must skip the `limit_offset` item rather
    than saddle the lowering with an impossible reconstruction."""
    for syntax in (LimitSyntax.HEAD_N, LimitSyntax.SAMPLE_N, LimitSyntax.TOP_N):
        spec = DialectSpec(name=f"blind_{syntax.value}", surface=SurfaceSpec(limit_syntax=syntax))
        labels = {it.label for it in build_ir_battery(spec)}
        assert "limit_offset" not in labels, (
            f"{syntax.value} cannot encode OFFSET; build_ir_battery must skip it"
        )
        # Sanity: `limit_only` is unaffected — these surfaces all encode a row count.
        assert "limit_only" in labels


# ---- Lowering seed: deterministic patches survive structural specs --------


def test_emit_lowering_seed_patches_offset_fetch_limit_for_structural_spec() -> None:
    """`emit_lowering` raises for case_syntax=SWITCH, but `emit_lowering_seed`
    must still return the patched _lower_limit body so the LLM lane can use
    it as a starting point instead of rewriting the helper from scratch."""
    spec = DialectSpec(
        name="switch_offset_fetch",
        surface=SurfaceSpec(
            case_syntax=CaseSyntax.SWITCH,
            limit_syntax=LimitSyntax.OFFSET_FETCH,
        ),
    )
    with pytest.raises(NotImplementedError):
        emit_lowering(spec)
    seed = emit_lowering_seed(spec)
    # The OFFSET_FETCH-specific body assigns offset=ints[0], limit=ints[1].
    assert "offset = ints[0]" in seed
    assert "limit = ints[1]" in seed


def test_emit_lowering_seed_returns_unmodified_for_simple_spec() -> None:
    """For a fully surface-only spec the seed is identical to `emit_lowering`."""
    spec = EXAMPLE_SPECS["mild_postgres_ish"]
    assert emit_lowering_seed(spec) == emit_lowering(spec)


def test_generate_lowering_seeds_llm_with_patched_lowering(monkeypatch) -> None:
    """When the deterministic emitter raises, the agent must still seed the
    LLM with `emit_lowering_seed(spec)` so OFFSET_FETCH lowering is
    pre-baked instead of reinvented by the model."""
    spec = DialectSpec(
        name="seeded_switch",
        surface=SurfaceSpec(
            case_syntax=CaseSyntax.SWITCH,
            limit_syntax=LimitSyntax.OFFSET_FETCH,
        ),
    )

    captured: dict = {}

    class CapturingClient(LLMClient):
        def __init__(self) -> None:
            self.config = LLMConfig(
                backend=LLMBackend.OPENAI, api_key="x",
                base_url="x", default_model="x",
            )

        def chat(self, **kwargs):
            captured["user"] = kwargs.get("user", "")
            # Return something obviously broken so the agent doesn't accept
            # it. We only care that the prompt contained the seed.
            return LLMResponse(
                text="def lower(tree, config, catalog):\n    return None\n",
                model="fake", backend=self.config.backend,
                prompt_tokens=0, completion_tokens=0,
            )

        def close(self) -> None:
            return None

    generate_lowering(spec, llm_client=CapturingClient(), max_iterations=1)
    assert "offset = ints[0]" in captured["user"], (
        "the LLM prompt should contain the deterministic OFFSET_FETCH seed"
    )


# ---- Refine prompt includes parse trees for failing items -----------------


def test_refine_prompt_includes_dialect_parse_tree() -> None:
    """The fix-mode prompt must show the actual Lark parse tree of each
    failing item. Without this, the model can't see that anonymous keyword
    terminals are filtered out of the tree and ends up writing token
    state machines that never match anything (the sqlite_bigquery_db2
    failure mode in batch-15)."""
    spec = EXAMPLE_SPECS["mild_postgres_ish"]
    grammar = emit_grammar(spec)
    item = build_ir_battery(spec)[0]

    diverg = IRDivergence(
        label=item.label,
        ref_sql=item.ref_sql,
        dialect_sql=item.dialect_sql,
        ref_plan="ref",
        dialect_plan="dial",
        error="plan mismatch",
    )

    captured: dict = {}

    class CapturingClient(LLMClient):
        def __init__(self) -> None:
            self.config = LLMConfig(
                backend=LLMBackend.OPENAI, api_key="x",
                base_url="x", default_model="x",
            )

        def chat(self, **kwargs):
            captured["user"] = kwargs.get("user", "")
            return LLMResponse(
                text="ok", model="fake", backend=self.config.backend,
                prompt_tokens=0, completion_tokens=0,
            )

        def close(self) -> None:
            return None

    from manysql.codegen.ir_battery import IREquivalenceReport
    report = IREquivalenceReport(items=[item], divergences=[diverg])
    _refine_with_llm(
        spec=spec,
        lowering_text="# unused\n",
        grammar_text=grammar,
        items=[item],
        report=report,
        llm_client=CapturingClient(),
        polish=False,
    )
    user = captured["user"]
    assert "dialect parse tree" in user
    # The first canonical SQL is `SELECT * FROM employees`, whose tree
    # contains a `select_core` rule. That label should make it into the
    # rendered tree.
    assert "select_core" in user


def test_parse_failure_trees_handles_unparseable_items() -> None:
    """If a failing item happens not to parse with the dialect grammar
    (e.g. mid-iteration the LLM left grammar inconsistent), we should
    annotate the prompt with the parse error instead of raising."""
    grammar = "start: \"x\""  # accepts only "x"
    diverg = IRDivergence(
        label="bogus",
        ref_sql="SELECT 1",
        dialect_sql="SELECT 1",
        ref_plan=None,
        dialect_plan=None,
        error="something",
    )
    out = _parse_failure_trees(grammar, [diverg])
    assert "parse failed" in out["bogus"]


def test_parse_failure_trees_handles_grammar_build_failure() -> None:
    """A broken grammar should not crash the prompt builder; every
    divergence should get the same explanatory error annotation."""
    diverg = IRDivergence(
        label="x",
        ref_sql="",
        dialect_sql="",
        ref_plan=None,
        dialect_plan=None,
        error="",
    )
    out = _parse_failure_trees("start: NEVER_TERMINAL", [diverg])
    assert "grammar build failed" in out["x"]
