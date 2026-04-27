"""Tests for the batch (multi-dialect) codegen orchestrator.

Drives the orchestrator with a queued ``NullLLMClient`` subclass so each
chat() call yields the next canned reply. Inner-pipeline calls receive
the same client; the existing agents short-circuit on ``NullLLMClient``
and run the deterministic emitters, so packaging is hermetic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from manysql.codegen.batch import (
    CAMPAIGNS_DIRNAME,
    CampaignBrief,
    CampaignConfig,
    CampaignReporter,
    _compute_theme_schedule,
    _dedup_name,
    design_dialect_batch,
    expand_campaign_brief,
    run_campaign,
)
from manysql.dialects.registry import DialectRegistry
from manysql.llm import LLMResponse, NullLLMClient
from manysql.spec.examples import EXAMPLE_SPECS


class _ScriptedLLM(NullLLMClient):
    """NullLLMClient that returns a queued reply per chat() call.

    After the queue is drained, repeats the last reply so the inner
    pipeline can keep calling without exhausting the test fixture.
    """

    def __init__(self, replies: list[str]) -> None:
        super().__init__(canned_reply="{}")
        self._replies = list(replies)
        self._index = 0

    def chat(self, **kwargs: Any) -> LLMResponse:  # type: ignore[override]
        self.record.append({"kwargs": kwargs})
        if self._index < len(self._replies):
            text = self._replies[self._index]
            self._index += 1
        elif self._replies:
            text = self._replies[-1]
        else:
            text = "{}"
        return LLMResponse(
            text=text,
            model=kwargs.get("model") or self.config.default_model,
            backend=self.config.backend,
            prompt_tokens=0,
            completion_tokens=0,
        )


def _spec_payload(name: str, *, divergence: str = "mild") -> dict[str, Any]:
    """Build a valid DialectSpec JSON payload from the mild example."""
    base = json.loads(EXAMPLE_SPECS["mild_postgres_ish"].model_dump_json())
    base["name"] = name
    base["divergence"] = divergence
    return base


def _design_reply(
    name: str,
    *,
    divergence: str = "mild",
    description: str = "test dialect",
    primary_axes: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "spec": _spec_payload(name, divergence=divergence),
            "ledger_entry": {
                "name": name,
                "description": description,
                "primary_axes": primary_axes or ["axis-a", "axis-b"],
            },
        }
    )


def _brief_reply() -> str:
    return json.dumps(
        {
            "inspirations": ["sql_server", "snowflake"],
            "suggested_axes": ["TOP n limits", "case-fold UPPER"],
            "forbidden_knobs": [],
            "style_notes": "stay close to mssql ergonomics.",
        }
    )


def _empty_brief() -> CampaignBrief:
    return CampaignBrief(
        prior="",
        inspirations=(),
        suggested_axes=(),
        forbidden_knobs=(),
        style_notes="",
    )


# --- _compute_theme_schedule ---


def test_theme_schedule_mixed_round_robins() -> None:
    assert _compute_theme_schedule("mixed", 5) == [
        "mild",
        "moderate",
        "aggressive",
        "mild",
        "moderate",
    ]


def test_theme_schedule_explicit_uniform() -> None:
    assert _compute_theme_schedule("aggressive", 3) == [
        "aggressive",
        "aggressive",
        "aggressive",
    ]


def test_theme_schedule_zero() -> None:
    assert _compute_theme_schedule("mixed", 0) == []


# --- _dedup_name ---


def test_dedup_name_unique_passes_through() -> None:
    assert _dedup_name("snowflake_lite", set()) == "snowflake_lite"


def test_dedup_name_collision_appends_index() -> None:
    taken = {"snowflake_lite"}
    assert _dedup_name("snowflake_lite", taken) == "snowflake_lite_2"
    taken.add("snowflake_lite_2")
    assert _dedup_name("snowflake_lite", taken) == "snowflake_lite_3"


def test_dedup_name_normalizes_chars() -> None:
    assert _dedup_name("Snowflake-Lite!", set()) == "snowflake_lite"


def test_dedup_name_avoids_reserved() -> None:
    assert _dedup_name("registry", set()) != "registry"


# --- expand_campaign_brief ---


def test_expand_brief_parses_minimal_reply() -> None:
    llm = _ScriptedLLM([_brief_reply()])
    cfg = CampaignConfig(
        n=1, prior="mssql vs snowflake", inspired_by=("sql_server",)
    )
    brief = expand_campaign_brief(cfg, llm)
    assert brief.prior == "mssql vs snowflake"
    assert "sql_server" in brief.inspirations
    assert "TOP n limits" in brief.suggested_axes
    assert brief.forbidden_knobs == ()


def test_expand_brief_appends_user_excluded_knobs() -> None:
    reply = json.dumps(
        {
            "inspirations": [],
            "suggested_axes": [],
            "forbidden_knobs": ["limit_syntax"],
            "style_notes": "",
        }
    )
    llm = _ScriptedLLM([reply])
    cfg = CampaignConfig(
        n=1,
        exclude_knobs=("identifier_quote", "limit_syntax"),
    )
    brief = expand_campaign_brief(cfg, llm)
    assert set(brief.forbidden_knobs) == {"limit_syntax", "identifier_quote"}


def test_expand_brief_user_payload_contains_prior_and_excludes() -> None:
    llm = _ScriptedLLM([_brief_reply()])
    cfg = CampaignConfig(
        n=1,
        prior="mssql vs snowflake",
        inspired_by=("sql_server",),
        exclude_knobs=("identifier_quote",),
    )
    expand_campaign_brief(cfg, llm)
    payload = json.loads(llm.record[0]["kwargs"]["user"])
    assert payload["prior"] == "mssql vs snowflake"
    assert payload["exclude_knobs"] == ["identifier_quote"]
    assert payload["inspired_by"] == ["sql_server"]


# --- design_dialect_batch ---


def test_design_batch_grows_ledger() -> None:
    replies = [
        _design_reply("dialect_a", divergence="mild"),
        _design_reply("dialect_b", divergence="moderate"),
        _design_reply("dialect_c", divergence="aggressive"),
    ]
    llm = _ScriptedLLM(replies)
    cfg = CampaignConfig(n=3, theme="mixed")
    drafted, failed = design_dialect_batch(
        _empty_brief(), cfg, llm, existing_names=set()
    )
    assert failed == []
    assert [s.name for s, _ in drafted] == ["dialect_a", "dialect_b", "dialect_c"]
    third_payload = json.loads(llm.record[2]["kwargs"]["user"])
    seen = [e["name"] for e in third_payload["ledger_so_far"]]
    assert seen == ["dialect_a", "dialect_b"]
    assert third_payload["target_divergence"] == "aggressive"


def test_design_batch_retries_on_validation_failure_then_succeeds() -> None:
    bad = json.dumps({"spec": {"name": 123}, "ledger_entry": {}})
    good = _design_reply("dialect_ok")
    llm = _ScriptedLLM([bad, good])
    cfg = CampaignConfig(n=1)
    drafted, failed = design_dialect_batch(
        _empty_brief(), cfg, llm, existing_names=set()
    )
    assert failed == []
    assert [s.name for s, _ in drafted] == ["dialect_ok"]
    assert len(llm.record) == 2
    assert "RETRY" in llm.record[1]["kwargs"]["user"]


def test_design_batch_records_failure_after_two_tries() -> None:
    bad = json.dumps({"spec": {"name": 123}, "ledger_entry": {}})
    llm = _ScriptedLLM([bad, bad])
    cfg = CampaignConfig(n=1)
    drafted, failed = design_dialect_batch(
        _empty_brief(), cfg, llm, existing_names=set()
    )
    assert drafted == []
    assert len(failed) == 1
    assert failed[0]["slot"] == 0
    assert failed[0]["target_divergence"] == "mild"
    assert "reason" in failed[0]


def test_design_batch_dedups_against_existing() -> None:
    existing = {"taken_name"}
    llm = _ScriptedLLM([_design_reply("taken_name")])
    cfg = CampaignConfig(n=1)
    drafted, failed = design_dialect_batch(
        _empty_brief(), cfg, llm, existing_names=existing
    )
    assert failed == []
    spec, entry = drafted[0]
    assert entry.name == "taken_name_2"
    assert spec.name == "taken_name_2"


# --- run_campaign end-to-end ---


def test_run_campaign_end_to_end_writes_manifest_and_packages(
    tmp_path: Path,
) -> None:
    replies = [
        _brief_reply(),
        _design_reply("alpha", divergence="mild"),
        _design_reply("beta", divergence="moderate"),
    ]
    llm = _ScriptedLLM(replies)
    cfg = CampaignConfig(n=2, theme="mixed", max_concurrency=2)
    # Point at a nested non-existent directory to confirm run_campaign
    # creates the dialects root before listing existing entries.
    fresh_root = tmp_path / "nested" / "manysql-batch"
    result = run_campaign(cfg, llm=llm, dialects_root=fresh_root)

    assert len(result.drafted) == 2
    assert len(result.packaged) == 2
    assert result.failed_specs == []
    assert result.failed_packages == []

    manifest_path = fresh_root / CAMPAIGNS_DIRNAME / f"{result.id}.json"
    assert manifest_path.is_file()
    payload = json.loads(manifest_path.read_text())
    assert payload["id"] == result.id
    assert {p["name"] for p in payload["packaged"]} == {"alpha", "beta"}
    assert payload["brief"]["prior"] == ""
    assert payload["config"]["n"] == 2

    registry = DialectRegistry(fresh_root)
    assert set(registry.list()) == {"alpha", "beta"}
    for name in ("alpha", "beta"):
        engine = registry.load(name)
        assert engine.name == name


def test_run_campaign_records_failed_specs_without_aborting(
    tmp_path: Path,
) -> None:
    bad = json.dumps({"spec": {"name": 123}})
    replies = [_brief_reply(), bad, bad, _design_reply("only_good")]
    llm = _ScriptedLLM(replies)
    cfg = CampaignConfig(n=2, theme="mild")
    result = run_campaign(cfg, llm=llm, dialects_root=tmp_path / "fresh")

    assert {s.name for s, _ in result.drafted} == {"only_good"}
    assert len(result.failed_specs) == 1
    assert len(result.packaged) == 1
    assert result.packaged[0].name == "only_good"


# --- reporter ---


class _RecordingReporter(CampaignReporter):
    """Captures (event_name, kwargs) tuples for assertions."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def on_campaign_start(self, **kw: Any) -> None:
        self.events.append(("campaign_start", kw))

    def on_brief_start(self) -> None:
        self.events.append(("brief_start", {}))

    def on_brief_done(self, **kw: Any) -> None:
        self.events.append(("brief_done", kw))

    def on_design_phase_start(self, **kw: Any) -> None:
        self.events.append(("design_phase_start", kw))

    def on_design_slot_attempt(self, **kw: Any) -> None:
        self.events.append(("design_slot_attempt", kw))

    def on_design_slot_done(self, **kw: Any) -> None:
        self.events.append(("design_slot_done", kw))

    def on_design_slot_failed(self, **kw: Any) -> None:
        self.events.append(("design_slot_failed", kw))

    def on_package_phase_start(self, **kw: Any) -> None:
        self.events.append(("package_phase_start", kw))

    def on_package_done(self, **kw: Any) -> None:
        self.events.append(("package_done", kw))

    def on_package_failed(self, **kw: Any) -> None:
        self.events.append(("package_failed", kw))

    def on_manifest_written(self, **kw: Any) -> None:
        self.events.append(("manifest_written", kw))

    def on_campaign_done(self, **kw: Any) -> None:
        self.events.append(("campaign_done", kw))

    def on_interrupted(self, **kw: Any) -> None:
        self.events.append(("interrupted", kw))


def test_run_campaign_emits_reporter_events(tmp_path: Path) -> None:
    """Happy-path campaign emits the expected event sequence in order.

    Asserts: one start/end pair per stage, one attempt+done per drafted
    slot, no failure events when nothing fails.
    """
    bad = json.dumps({"spec": {"name": 123}, "ledger_entry": {}})
    replies = [
        _brief_reply(),
        bad,
        _design_reply("alpha", divergence="mild"),
        _design_reply("beta", divergence="moderate"),
    ]
    llm = _ScriptedLLM(replies)
    reporter = _RecordingReporter()
    cfg = CampaignConfig(n=2, theme="mixed", max_concurrency=2)
    result = run_campaign(
        cfg,
        llm=llm,
        dialects_root=tmp_path / "fresh",
        reporter=reporter,
    )

    names = [e[0] for e in reporter.events]
    assert names[0] == "campaign_start"
    assert names[1] == "brief_start"
    assert names[2] == "brief_done"
    assert names[3] == "design_phase_start"
    assert names[-1] == "campaign_done"
    assert names[-2] == "manifest_written"

    attempts = [e for e in reporter.events if e[0] == "design_slot_attempt"]
    assert len(attempts) == 3
    assert attempts[0][1]["attempt"] == 1
    assert attempts[1][1]["attempt"] == 2
    assert attempts[2][1]["attempt"] == 1

    slot_done = [e for e in reporter.events if e[0] == "design_slot_done"]
    assert [e[1]["entry"].name for e in slot_done] == ["alpha", "beta"]
    assert all("design_slot_failed" != n for n in names)

    pkg_done = [e for e in reporter.events if e[0] == "package_done"]
    assert {e[1]["name"] for e in pkg_done} == {"alpha", "beta"}
    for e in pkg_done:
        assert e[1]["elapsed_s"] >= 0.0
        assert isinstance(e[1]["summary"], dict)

    pkg_phase = next(e for e in reporter.events if e[0] == "package_phase_start")
    assert pkg_phase[1]["n"] == 2
    assert pkg_phase[1]["max_concurrency"] == 2

    manifest_event = next(
        e for e in reporter.events if e[0] == "manifest_written"
    )
    assert manifest_event[1]["path"].name == f"{result.id}.json"


def test_run_campaign_emits_failure_events(tmp_path: Path) -> None:
    """Spec validation failures surface via on_design_slot_failed."""
    bad = json.dumps({"spec": {"name": 123}, "ledger_entry": {}})
    replies = [_brief_reply(), bad, bad]
    llm = _ScriptedLLM(replies)
    reporter = _RecordingReporter()
    cfg = CampaignConfig(n=1, theme="mild")
    run_campaign(
        cfg,
        llm=llm,
        dialects_root=tmp_path / "fresh",
        reporter=reporter,
    )

    failed = [e for e in reporter.events if e[0] == "design_slot_failed"]
    assert len(failed) == 1
    assert failed[0][1]["target_divergence"] == "mild"
    assert failed[0][1]["reason"]
    assert not [e for e in reporter.events if e[0] == "design_slot_done"]


# --- ctrl-c / interrupt handling ---


class _InterruptingLLM(_ScriptedLLM):
    """Raises KeyboardInterrupt on the Nth call to simulate ctrl-c."""

    def __init__(self, replies: list[str], *, raise_on_call: int) -> None:
        super().__init__(replies)
        self._raise_on_call = raise_on_call
        self._calls = 0

    def chat(self, **kwargs: Any) -> LLMResponse:  # type: ignore[override]
        self._calls += 1
        if self._calls == self._raise_on_call:
            raise KeyboardInterrupt
        return super().chat(**kwargs)


def test_run_campaign_interrupt_during_design_writes_partial_manifest(
    tmp_path: Path,
) -> None:
    """Ctrl-C in the design phase still produces a manifest and re-raises.

    Slot 0 succeeds, slot 1's first LLM call raises KeyboardInterrupt;
    we expect: drafted={alpha}, no packages, manifest on disk, KI surfaces
    to the caller for non-zero exit.
    """
    replies = [
        _brief_reply(),
        _design_reply("alpha", divergence="mild"),
    ]
    llm = _InterruptingLLM(replies, raise_on_call=3)
    reporter = _RecordingReporter()
    cfg = CampaignConfig(n=2, theme="mixed")
    root = tmp_path / "fresh"

    import pytest

    with pytest.raises(KeyboardInterrupt):
        run_campaign(cfg, llm=llm, dialects_root=root, reporter=reporter)

    names = [e[0] for e in reporter.events]
    assert "interrupted" in names
    assert "manifest_written" in names

    manifest_event = next(e for e in reporter.events if e[0] == "manifest_written")
    manifest_path = manifest_event[1]["path"]
    assert manifest_path.is_file()
    payload = json.loads(manifest_path.read_text())
    assert {d["spec"]["name"] for d in payload["drafted"]} == {"alpha"}
    assert payload["packaged"] == []
