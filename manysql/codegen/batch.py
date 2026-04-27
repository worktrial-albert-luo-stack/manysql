"""Batch codegen orchestrator: design N diverse dialects in one campaign.

Pipeline:
    1. ``expand_campaign_brief``   - one LLM call expands a free-form prior
       (e.g. "variants between mssql and snowflake") plus structured knobs
       into a structured ``CampaignBrief`` that every worker sees.
    2. ``design_dialect_batch``    - sequential per-slot LLM calls. Each call
       sees the brief and the running ledger (one-line descriptions of the
       drafted specs so far) so the model can deliberately diversify. The
       theme schedule (round-robin across mild/moderate/aggressive when
       ``theme=mixed``) assigns a target divergence level per slot.
    3. ``run_campaign``            - drives 1 + 2, then fans the drafted
       specs into the existing ``write_dialect_package`` pipeline via a
       ``ThreadPoolExecutor`` (the inner pipeline is LLM-bound, so threads
       are sufficient). Every drafted spec becomes a dialect package or a
       recorded failure; the campaign continues past individual failures.
    4. A campaign manifest (``manysql/dialects/_campaigns/<id>.json``)
       records the config, brief, every drafted spec, and per-package
       status. The ``_campaigns`` directory is skipped by the registry's
       ``list()`` because it has no ``metadata.json``.

Spec generation goes through the existing ``LLMClient.chat_json`` helper
(provider-side JSON mode + Pydantic validation + a single retry on
validation failure). Schema-enforced structured outputs (OpenAI's
``json_schema`` mode, Anthropic tool-use forcing) are out of scope for v1
and would slot in via a new ``LLMClient.chat_structured(BaseModel)`` method
without changing this module.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import ValidationError

from manysql.codegen.pipeline import (
    BatteryError,
    PackageWriteResult,
    write_dialect_package,
)
from manysql.dialects.registry import (
    DIALECTS_DIR,
    RESERVED_NAMES,
    DialectRegistry,
)
from manysql.llm.client import LLMClient, LLMError, NullLLMClient
from manysql.spec.dialect import DialectSpec
from manysql.spec.examples import EXAMPLE_SPECS

ThemeLiteral = Literal["mild", "moderate", "aggressive", "mixed"]
THEME_CHOICES: tuple[str, ...] = ("mild", "moderate", "aggressive", "mixed")
_ROTATION: tuple[str, ...] = ("mild", "moderate", "aggressive")
CAMPAIGNS_DIRNAME = "_campaigns"

_BRIEF_TEMPERATURE = 0.3
_DESIGN_TEMPERATURE = 0.7

_NAME_RE = re.compile(r"[^a-z0-9_]+")


@dataclass(frozen=True)
class CampaignConfig:
    """User-facing knobs for one batch run.

    ``inspired_by`` and ``exclude_knobs`` are tuples (not lists) so the
    config can sit inside a frozen dataclass without footguns.
    """

    n: int
    prior: Optional[str] = None
    theme: ThemeLiteral = "mixed"
    inspired_by: tuple[str, ...] = ()
    exclude_knobs: tuple[str, ...] = ()
    seed: Optional[int] = None
    model: Optional[str] = None
    max_concurrency: int = 4
    require_battery_pass: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "prior": self.prior,
            "theme": self.theme,
            "inspired_by": list(self.inspired_by),
            "exclude_knobs": list(self.exclude_knobs),
            "seed": self.seed,
            "model": self.model,
            "max_concurrency": self.max_concurrency,
            "require_battery_pass": self.require_battery_pass,
        }


@dataclass(frozen=True)
class CampaignBrief:
    """One-shot expansion of the prior + structured knobs.

    Workers see ``inspirations`` as starting points, ``suggested_axes`` as a
    palette of divergence ideas, ``forbidden_knobs`` as hard constraints,
    and ``style_notes`` as free-form guidance.
    """

    prior: str
    inspirations: tuple[str, ...]
    suggested_axes: tuple[str, ...]
    forbidden_knobs: tuple[str, ...]
    style_notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prior": self.prior,
            "inspirations": list(self.inspirations),
            "suggested_axes": list(self.suggested_axes),
            "forbidden_knobs": list(self.forbidden_knobs),
            "style_notes": self.style_notes,
        }


@dataclass(frozen=True)
class LedgerEntry:
    """Short description of a drafted spec, shown to subsequent workers.

    Kept much smaller than the full ``DialectSpec`` so the per-slot prompt
    stays cheap as the ledger grows.
    """

    name: str
    description: str
    primary_axes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "primary_axes": list(self.primary_axes),
        }


@dataclass
class CampaignResult:
    """End-to-end record of one batch run.

    ``failed_specs`` holds slots whose model output never validated;
    ``failed_packages`` holds drafted specs that the inner pipeline could
    not turn into a package (e.g. ``BatteryError`` under
    ``require_battery_pass=True``).
    """

    id: str
    config: CampaignConfig
    brief: CampaignBrief
    drafted: list[tuple[DialectSpec, LedgerEntry]]
    packaged: list[PackageWriteResult]
    failed_specs: list[dict[str, Any]]
    failed_packages: list[dict[str, Any]]
    started_at: str
    finished_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "config": self.config.to_dict(),
            "brief": self.brief.to_dict(),
            "drafted": [
                {
                    "spec": json.loads(spec.model_dump_json()),
                    "ledger": entry.to_dict(),
                }
                for spec, entry in self.drafted
            ],
            "packaged": [_package_summary(p) for p in self.packaged],
            "failed_specs": self.failed_specs,
            "failed_packages": self.failed_packages,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


class CampaignReporter:
    """Hook surface for live campaign progress.

    Default implementation is a no-op so library callers (and tests) can
    skip the reporter entirely. The CLI provides a Rich-backed subclass.

    Thread safety: ``on_package_*`` events fire from worker threads
    inside the package fan-out. Implementations that touch shared state
    must guard it themselves; the Rich console used by the CLI is itself
    thread-safe.
    """

    def on_campaign_start(
        self, *, config: "CampaignConfig", model: str
    ) -> None: ...

    def on_brief_start(self) -> None: ...

    def on_brief_done(
        self, *, brief: "CampaignBrief", elapsed_s: float
    ) -> None: ...

    def on_design_phase_start(self, *, schedule: list[str]) -> None: ...

    def on_design_slot_attempt(
        self,
        *,
        slot: int,
        total: int,
        target_divergence: str,
        attempt: int,
    ) -> None: ...

    def on_design_slot_done(
        self,
        *,
        slot: int,
        total: int,
        entry: "LedgerEntry",
        divergence: str,
        elapsed_s: float,
    ) -> None: ...

    def on_design_slot_failed(
        self,
        *,
        slot: int,
        total: int,
        target_divergence: str,
        reason: str,
    ) -> None: ...

    def on_package_phase_start(
        self, *, n: int, max_concurrency: int
    ) -> None: ...

    def on_package_done(
        self,
        *,
        name: str,
        summary: dict[str, Any],
        elapsed_s: float,
    ) -> None: ...

    def on_package_failed(
        self,
        *,
        name: str,
        reason: str,
        elapsed_s: float,
    ) -> None: ...

    def on_manifest_written(self, *, path: Path) -> None: ...

    def on_campaign_done(self, *, result: "CampaignResult") -> None: ...


_NULL_REPORTER = CampaignReporter()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def expand_campaign_brief(
    config: CampaignConfig,
    llm: LLMClient,
    *,
    reporter: Optional[CampaignReporter] = None,
) -> CampaignBrief:
    """Stage 0: turn the user-supplied prior + knobs into a structured brief.

    Precedence is enforced here, not in the prompt: anything the user
    listed in ``--exclude-knobs`` is appended verbatim to ``forbidden_knobs``
    even if the model omitted or renamed it.
    """
    rep = reporter or _NULL_REPORTER
    user_payload = {
        "prior": config.prior or "",
        "theme": config.theme,
        "inspired_by": list(config.inspired_by),
        "exclude_knobs": list(config.exclude_knobs),
    }
    rep.on_brief_start()
    started = time.monotonic()
    reply = llm.chat_json(
        system=_BRIEF_SYSTEM_PROMPT,
        user=json.dumps(user_payload, indent=2),
        model=config.model,
        temperature=_BRIEF_TEMPERATURE,
    )
    inspirations = _strs(reply.get("inspirations"))
    suggested_axes = _strs(reply.get("suggested_axes"))
    forbidden_knobs = list(_strs(reply.get("forbidden_knobs")))
    for k in config.exclude_knobs:
        if k not in forbidden_knobs:
            forbidden_knobs.append(k)
    style_notes = str(reply.get("style_notes") or "").strip()
    brief = CampaignBrief(
        prior=config.prior or "",
        inspirations=tuple(inspirations),
        suggested_axes=tuple(suggested_axes),
        forbidden_knobs=tuple(forbidden_knobs),
        style_notes=style_notes,
    )
    rep.on_brief_done(brief=brief, elapsed_s=time.monotonic() - started)
    return brief


def design_dialect_batch(
    brief: CampaignBrief,
    config: CampaignConfig,
    llm: LLMClient,
    *,
    existing_names: set[str],
    reporter: Optional[CampaignReporter] = None,
) -> tuple[list[tuple[DialectSpec, LedgerEntry]], list[dict[str, Any]]]:
    """Stage 1: sequentially design ``config.n`` specs.

    Each iteration builds a prompt from the brief plus the running ledger,
    asks the model for ``{spec, ledger_entry}``, and validates the spec
    against ``DialectSpec``. On the first validation failure the slot is
    retried once with the validation error in-prompt; a second failure is
    recorded and the campaign continues. Returned specs always have unique
    names (deduped against existing dialects + the running ledger).
    """
    rep = reporter or _NULL_REPORTER
    schedule = _compute_theme_schedule(config.theme, config.n)
    rep.on_design_phase_start(schedule=list(schedule))
    schema_text = json.dumps(DialectSpec.model_json_schema(), indent=2)
    examples_text = _few_shot_examples_text()
    drafted: list[tuple[DialectSpec, LedgerEntry]] = []
    failures: list[dict[str, Any]] = []
    taken_names: set[str] = set(existing_names)
    ledger_so_far: list[LedgerEntry] = []
    total = len(schedule)
    for slot, target_divergence in enumerate(schedule):
        slot_started = time.monotonic()
        outcome = _design_one_slot(
            slot=slot,
            total=total,
            target_divergence=target_divergence,
            brief=brief,
            ledger_so_far=ledger_so_far,
            taken_names=taken_names,
            schema_text=schema_text,
            examples_text=examples_text,
            llm=llm,
            config=config,
            reporter=rep,
        )
        if isinstance(outcome, dict):
            rep.on_design_slot_failed(
                slot=slot,
                total=total,
                target_divergence=target_divergence,
                reason=str(outcome.get("reason", "")),
            )
            failures.append(outcome)
            continue
        spec, entry = outcome
        drafted.append((spec, entry))
        taken_names.add(entry.name)
        ledger_so_far.append(entry)
        rep.on_design_slot_done(
            slot=slot,
            total=total,
            entry=entry,
            divergence=spec.divergence.value,
            elapsed_s=time.monotonic() - slot_started,
        )
    return drafted, failures


def run_campaign(
    config: CampaignConfig,
    *,
    llm: LLMClient,
    dialects_root: Optional[Path] = None,
    reporter: Optional[CampaignReporter] = None,
) -> CampaignResult:
    """Drive brief expansion + design + parallel package fan-out, write manifest."""
    rep = reporter or _NULL_REPORTER
    root = dialects_root or DIALECTS_DIR
    # The registry's list() calls iterdir(), which raises FileNotFoundError
    # on a fresh --dialects-dir. The campaign needs this directory anyway
    # (manifest + per-dialect packages land here), so create it eagerly.
    root.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    registry = DialectRegistry(root)
    existing = set(registry.list())

    rep.on_campaign_start(config=config, model=llm.config.default_model)

    brief = expand_campaign_brief(config, llm, reporter=rep)
    drafted, failed_specs = design_dialect_batch(
        brief, config, llm, existing_names=existing, reporter=rep
    )

    packaged, failed_packages = _fan_out_packages(
        drafted=drafted,
        config=config,
        llm=llm,
        dialects_root=root,
        reporter=rep,
    )

    finished_at = datetime.now(timezone.utc).isoformat()
    cid = _make_campaign_id(started_at, config)
    result = CampaignResult(
        id=cid,
        config=config,
        brief=brief,
        drafted=drafted,
        packaged=packaged,
        failed_specs=failed_specs,
        failed_packages=failed_packages,
        started_at=started_at,
        finished_at=finished_at,
    )
    manifest_path = write_campaign_manifest(result, root)
    rep.on_manifest_written(path=manifest_path)
    rep.on_campaign_done(result=result)
    return result


def write_campaign_manifest(result: CampaignResult, dialects_root: Path) -> Path:
    """Persist a campaign manifest under ``<root>/_campaigns/<id>.json``."""
    campaigns_dir = dialects_root / CAMPAIGNS_DIRNAME
    campaigns_dir.mkdir(parents=True, exist_ok=True)
    path = campaigns_dir / f"{result.id}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return path


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _compute_theme_schedule(theme: str, n: int) -> list[str]:
    """Round-robin across ``mild/moderate/aggressive`` for ``mixed``; uniform otherwise."""
    if n <= 0:
        return []
    if theme == "mixed":
        return [_ROTATION[i % len(_ROTATION)] for i in range(n)]
    return [theme] * n


def _design_one_slot(
    *,
    slot: int,
    total: int,
    target_divergence: str,
    brief: CampaignBrief,
    ledger_so_far: list[LedgerEntry],
    taken_names: set[str],
    schema_text: str,
    examples_text: str,
    llm: LLMClient,
    config: CampaignConfig,
    reporter: CampaignReporter,
) -> tuple[DialectSpec, LedgerEntry] | dict[str, Any]:
    """Design one spec with at most one retry on validation failure."""
    system = _DESIGN_SYSTEM_PROMPT.format(
        schema=schema_text, examples=examples_text
    )
    base_user = _build_design_user_payload(
        slot=slot,
        target_divergence=target_divergence,
        brief=brief,
        ledger=ledger_so_far,
        taken_names=taken_names,
        seed=config.seed,
    )
    last_error = ""
    last_raw = ""
    for attempt in range(2):
        reporter.on_design_slot_attempt(
            slot=slot,
            total=total,
            target_divergence=target_divergence,
            attempt=attempt + 1,
        )
        user = base_user
        if attempt > 0:
            user = (
                f"{base_user}\n\n--- RETRY ---\n"
                "Your previous attempt failed validation with this error:\n"
                f"{last_error}\n\n"
                "Fix the spec to match the schema and reply again. "
                "Do not change the assigned divergence level."
            )
        try:
            reply = llm.chat_json(
                system=system,
                user=user,
                model=config.model,
                temperature=_DESIGN_TEMPERATURE,
            )
        except LLMError as exc:
            last_error = f"LLM error: {exc}"
            last_raw = ""
            continue
        if not isinstance(reply, dict):
            last_error = f"reply was not a JSON object: {type(reply).__name__}"
            last_raw = json.dumps(reply)
            continue
        last_raw = json.dumps(reply)
        spec_obj = reply.get("spec")
        ledger_obj = reply.get("ledger_entry") or {}
        if not isinstance(spec_obj, dict):
            last_error = "missing or non-object 'spec' field in reply"
            continue
        try:
            spec = DialectSpec.model_validate(spec_obj)
        except ValidationError as exc:
            last_error = str(exc)
            continue
        try:
            entry = _build_ledger_entry(spec, ledger_obj, taken_names)
        except (TypeError, ValueError) as exc:
            last_error = f"ledger entry malformed: {exc}"
            continue
        if spec.name != entry.name:
            spec = spec.model_copy(update={"name": entry.name})
        return spec, entry
    return {
        "slot": slot,
        "target_divergence": target_divergence,
        "reason": last_error or "no reply",
        "raw": last_raw,
    }


def _build_ledger_entry(
    spec: DialectSpec, ledger_obj: Any, taken_names: set[str]
) -> LedgerEntry:
    if not isinstance(ledger_obj, dict):
        ledger_obj = {}
    raw_name = ledger_obj.get("name") or spec.name
    if not isinstance(raw_name, str) or not raw_name.strip():
        raw_name = spec.name
    name = _dedup_name(raw_name, taken_names)
    description = str(ledger_obj.get("description") or spec.description or "").strip()
    raw_axes = ledger_obj.get("primary_axes") or []
    if not isinstance(raw_axes, list):
        raw_axes = []
    axes = tuple(str(a).strip() for a in raw_axes if str(a).strip())[:5]
    return LedgerEntry(name=name, description=description, primary_axes=axes)


def _dedup_name(name: str, taken: set[str]) -> str:
    """Make ``name`` filesystem-safe and unique against ``taken``."""
    base = _NAME_RE.sub("_", name.lower()).strip("_") or "dialect"
    if base.startswith("_"):
        base = "d" + base
    if base in RESERVED_NAMES or base == CAMPAIGNS_DIRNAME.lstrip("_"):
        base = f"{base}_dialect"
    if base not in taken:
        return base
    i = 2
    while True:
        candidate = f"{base}_{i}"
        if candidate not in taken:
            return candidate
        i += 1


def _build_design_user_payload(
    *,
    slot: int,
    target_divergence: str,
    brief: CampaignBrief,
    ledger: list[LedgerEntry],
    taken_names: set[str],
    seed: Optional[int],
) -> str:
    payload = {
        "slot": slot,
        "target_divergence": target_divergence,
        "campaign_brief": brief.to_dict(),
        "ledger_so_far": [e.to_dict() for e in ledger],
        "existing_dialect_names": sorted(taken_names),
        "seed": seed,
    }
    return json.dumps(payload, indent=2)


def _fan_out_packages(
    *,
    drafted: list[tuple[DialectSpec, LedgerEntry]],
    config: CampaignConfig,
    llm: LLMClient,
    dialects_root: Path,
    reporter: CampaignReporter,
) -> tuple[list[PackageWriteResult], list[dict[str, Any]]]:
    """Drive ``write_dialect_package`` per drafted spec in parallel."""
    packaged: list[PackageWriteResult] = []
    failed: list[dict[str, Any]] = []
    if not drafted:
        return packaged, failed
    real_llm = not isinstance(llm, NullLLMClient)
    workers = max(1, config.max_concurrency)
    reporter.on_package_phase_start(n=len(drafted), max_concurrency=workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_name: dict[Future[PackageWriteResult], str] = {}
        # Track per-future submit time so we can report wall-clock
        # latency per package even when many run concurrently.
        future_started: dict[Future[PackageWriteResult], float] = {}
        for spec, _entry in drafted:
            fut = pool.submit(
                write_dialect_package,
                spec,
                dialects_root,
                model=config.model,
                provider="llm" if real_llm else "deterministic",
                lifecycle="generated",
                overwrite=False,
                llm_client=llm,
                require_battery_pass=config.require_battery_pass,
                force_llm=real_llm,
            )
            future_to_name[fut] = spec.name
            future_started[fut] = time.monotonic()
        for fut in as_completed(future_to_name):
            name = future_to_name[fut]
            elapsed = time.monotonic() - future_started[fut]
            try:
                pkg = fut.result()
            except BatteryError as exc:
                reason = f"battery: {exc}"
                failed.append({"name": name, "reason": reason})
                reporter.on_package_failed(
                    name=name, reason=reason, elapsed_s=elapsed
                )
            except FileExistsError as exc:
                reason = f"file_exists: {exc}"
                failed.append({"name": name, "reason": reason})
                reporter.on_package_failed(
                    name=name, reason=reason, elapsed_s=elapsed
                )
            except Exception as exc:  # noqa: BLE001 - blanket guard around inner pipeline
                reason = f"{type(exc).__name__}: {exc}"
                failed.append({"name": name, "reason": reason})
                reporter.on_package_failed(
                    name=name, reason=reason, elapsed_s=elapsed
                )
            else:
                packaged.append(pkg)
                reporter.on_package_done(
                    name=name,
                    summary=_package_summary(pkg),
                    elapsed_s=elapsed,
                )
    return packaged, failed


def _package_summary(p: PackageWriteResult) -> dict[str, Any]:
    return {
        "name": p.name,
        "path": str(p.path),
        "files": list(p.written_files),
        "grammar_ok": p.grammar_result.ok if p.grammar_result is not None else None,
        "lowering_ok": p.lowering_result.ok if p.lowering_result is not None else None,
    }


def _make_campaign_id(started_at: str, config: CampaignConfig) -> str:
    payload = json.dumps(
        {"start": started_at, "config": config.to_dict()}, sort_keys=True
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:8]
    # Compact ISO timestamp: 2026-04-26T22:34:55+00:00 -> 20260426T223455
    compact = re.sub(r"[^0-9T]", "", started_at)[:15]
    return f"{compact}-{digest}"


def _strs(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(v).strip() for v in value if isinstance(v, str) and v.strip())


def _few_shot_examples_text() -> str:
    """Render the three bundled example specs as a JSON few-shot block."""
    items = []
    for name in ("mild_postgres_ish", "moderate_keyword_swap", "aggressive_alien"):
        spec = EXAMPLE_SPECS[name]
        items.append(
            {
                "spec": json.loads(spec.model_dump_json()),
                "ledger_entry": {
                    "name": spec.name,
                    "description": spec.description,
                    "primary_axes": _example_axes(name),
                },
            }
        )
    return json.dumps(items, indent=2)


def _example_axes(name: str) -> list[str]:
    if name == "mild_postgres_ish":
        return ["lowercase fold", "NULL ordering knobs", "integer division truncate"]
    if name == "moderate_keyword_swap":
        return ["keyword renames (PICK/COND/...)", "backtick idents", "ALL set-op default"]
    if name == "aggressive_alien":
        return ["NIL nulls", "OFFSET/FETCH limits", "+ for concat, ::cast"]
    return []


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


_BRIEF_SYSTEM_PROMPT = """You are designing a campaign of synthetic SQL dialects for benchmarking.
Given a free-form prior and structured constraints, expand them into a structured
CAMPAIGN BRIEF that downstream workers will follow when designing individual dialects.

Reply with ONLY a single JSON object matching this schema:
{
  "inspirations": [string, ...],     // real-world dialects to draw from (e.g. "postgres", "mysql", "snowflake", "kdb")
  "suggested_axes": [string, ...],   // 4-8 short axis ideas, e.g. "TOP n limits", "case-fold UPPER", "backtick idents"
  "forbidden_knobs": [string, ...],  // DialectSpec field names workers MUST NOT change
  "style_notes": string              // one short paragraph of high-level guidance
}

Rules:
- Treat the user-supplied "exclude_knobs" as hard constraints; they MUST appear in forbidden_knobs verbatim.
- "inspired_by" supplied by the user MUST appear in inspirations; expand reasonably from the prior.
- Suggested_axes are HINTS, not commands. Each entry is a short concept the per-dialect worker can reach for.
- Do not produce any text outside the JSON object. No markdown fences, no commentary.
"""


_DESIGN_SYSTEM_PROMPT = """You design a single SQL dialect spec for a benchmark campaign.

Reply with ONLY a single JSON object of the form:
{{
  "spec": <DialectSpec JSON>,
  "ledger_entry": {{
    "name": string,                 // lowercase snake_case, ASCII, unique
    "description": string,          // one sentence, what makes this dialect distinctive
    "primary_axes": [string, ...]   // 2-3 short strings, the headline divergences
  }}
}}

Hard rules:
- The "spec" object MUST validate against the DialectSpec JSON schema below.
- Use enum string values exactly as defined (case-sensitive).
- spec.name MUST equal ledger_entry.name and MUST NOT collide with any existing dialect listed in the user payload.
- Honor "forbidden_knobs" from the campaign brief: do NOT set those DialectSpec fields away from their defaults.
- Match the assigned "target_divergence" level. Mild = a few semantic knobs only. Moderate = several keyword renames plus a structural quirk. Aggressive = bolder operator/keyword reshapes.
- Pick axes from the brief's suggested_axes plus axes NOT yet covered by ledger_so_far. Avoid near-duplicates of prior entries.

Output discipline:
- No markdown fences, no commentary, no trailing whitespace. Just the JSON object.
- All booleans, enums, and required fields must be present and well-typed.

DialectSpec JSON schema:
{schema}

Reference example specs (canonical, follow this shape):
{examples}
"""


__all__ = [
    "CAMPAIGNS_DIRNAME",
    "CampaignBrief",
    "CampaignConfig",
    "CampaignReporter",
    "CampaignResult",
    "LedgerEntry",
    "THEME_CHOICES",
    "ThemeLiteral",
    "design_dialect_batch",
    "expand_campaign_brief",
    "run_campaign",
    "write_campaign_manifest",
]
