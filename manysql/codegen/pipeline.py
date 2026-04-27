"""High-level codegen orchestration: spec in, dialect package out.

`build_package_bundle(spec)` returns a `PackageBundle` (text-only, no FS I/O).
`write_dialect_package(spec, root)` calls the bundler, then writes the files.

Validation (parse battery, IR-equivalence battery, multi-oracle harness) is
intentionally NOT inside this module — the caller drives the refine loop. This
separation keeps codegen pure and makes failure logs easy to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from manysql.codegen.battery_emit import emit_battery_json, emit_examples_sql
from manysql.codegen.config_emit import emit_semantic_config
from manysql.codegen.grammar_agent import (
    GrammarAgentResult,
    generate_grammar,
)
from manysql.codegen.grammar_emit import emit_grammar
from manysql.codegen.lowering_agent import (
    LoweringAgentResult,
    generate_lowering,
)
from manysql.codegen.effects_emit import emit_effects
from manysql.codegen.lowering_emit import emit_lowering
from manysql.codegen.metadata_emit import emit_metadata, emit_spec_json
from manysql.codegen.overrides_emit import emit_overrides
from manysql.codegen.passes_emit import emit_passes
from manysql.llm.client import LLMClient
from manysql.spec.dialect import DialectSpec


@dataclass(frozen=True)
class PackageBundle:
    """Text-only representation of a dialect package."""

    grammar: str
    lowering_py: str
    semantics_json: str
    metadata_json: str
    spec_json: str
    init_py: str
    overrides_py: str
    passes_py: str
    effects_py: str
    battery_json: str
    examples_sql: str

    def files(self) -> dict[str, str]:
        return {
            "grammar.lark": self.grammar,
            "lowering.py": self.lowering_py,
            "overrides.py": self.overrides_py,
            "passes.py": self.passes_py,
            "effects.py": self.effects_py,
            "semantics.json": self.semantics_json,
            "metadata.json": self.metadata_json,
            "spec.json": self.spec_json,
            "battery.json": self.battery_json,
            "examples.sql": self.examples_sql,
            "__init__.py": self.init_py,
        }


@dataclass(frozen=True)
class PackageWriteResult:
    name: str
    path: Path
    written_files: list[str]
    grammar_result: Optional[GrammarAgentResult] = None
    lowering_result: Optional[LoweringAgentResult] = None


def build_package_bundle(
    spec: DialectSpec,
    *,
    model: Optional[str] = None,
    provider: str = "deterministic",
    lifecycle: str = "draft",
    llm_client: Optional[LLMClient] = None,
    grammar_max_iterations: int = 3,
    lowering_max_iterations: int = 3,
    force_llm: bool = False,
) -> tuple[PackageBundle, GrammarAgentResult, LoweringAgentResult]:
    """Produce all dialect-package files as text. No I/O.

    Returns the bundle plus both agent results so the caller can inspect
    which iteration produced each artifact (deterministic vs LLM-refined)
    and any battery failures along the way.

    `force_llm=True` runs at least one LLM refinement pass for both grammar
    and lowering even when the deterministic baseline already passes. The
    LLM's output is rolled back automatically if it regresses against either
    battery.
    """
    grammar_result = generate_grammar(
        spec,
        llm_client=llm_client,
        max_iterations=grammar_max_iterations,
        force_llm=force_llm,
    )
    lowering_result = generate_lowering(
        spec,
        grammar_text=grammar_result.grammar,
        llm_client=llm_client,
        max_iterations=lowering_max_iterations,
        force_llm=force_llm,
    )
    lowering_text = (
        lowering_result.lowering_py
        if lowering_result.lowering_py
        else _emit_lowering_or_stub(spec)
    )
    parse_items = grammar_result.report.items
    ir_items = lowering_result.report.items
    bundle = PackageBundle(
        grammar=grammar_result.grammar if grammar_result.ok else emit_grammar(spec),
        lowering_py=lowering_text,
        overrides_py=emit_overrides(spec),
        passes_py=emit_passes(spec),
        effects_py=emit_effects(spec),
        semantics_json=emit_semantic_config(spec),
        metadata_json=emit_metadata(
            spec,
            model=model,
            provider=_provenance_provider(
                provider, grammar_result, lowering_result
            ),
            lifecycle=lifecycle,
        ),
        spec_json=emit_spec_json(spec),
        init_py=_INIT_TEMPLATE.format(name=spec.name),
        battery_json=emit_battery_json(
            parse_items=parse_items,
            parse_report=grammar_result.report,
            ir_items=ir_items,
            ir_report=lowering_result.report,
        ),
        examples_sql=emit_examples_sql(spec=spec, parse_items=parse_items),
    )
    return bundle, grammar_result, lowering_result


def _emit_lowering_or_stub(spec: DialectSpec) -> str:
    """Return either the deterministic lowering or a clear failure stub.

    A stub is necessary when the spec needs structural changes and no LLM
    is configured: the package still gets written, but `lowering.py`
    raises immediately if anyone tries to load it. This keeps the on-disk
    package layout consistent and makes diagnostics obvious.
    """
    try:
        return emit_lowering(spec)
    except NotImplementedError as exc:
        return _LOWERING_STUB_TEMPLATE.format(reason=str(exc))


def _provenance_provider(
    base: str,
    grammar_result: GrammarAgentResult,
    lowering_result: LoweringAgentResult,
) -> str:
    """Reflect whether an LLM was actually used to refine either artifact."""
    suffixes: list[str] = []
    if any(a.source == "llm" for a in grammar_result.attempts):
        suffixes.append("llm-grammar")
    if any(a.source == "llm" for a in lowering_result.attempts):
        suffixes.append("llm-lowering")
    if not suffixes:
        return base
    return f"{base}+{'+'.join(suffixes)}"


def write_dialect_package(
    spec: DialectSpec,
    root: Path,
    *,
    model: Optional[str] = None,
    provider: str = "deterministic",
    lifecycle: str = "draft",
    overwrite: bool = False,
    llm_client: Optional[LLMClient] = None,
    grammar_max_iterations: int = 3,
    lowering_max_iterations: int = 3,
    require_battery_pass: bool = False,
    force_llm: bool = False,
) -> PackageWriteResult:
    """Materialize the dialect package to `root/<spec.name>/...`.

    `root` is typically `manysql/dialects/`.

    Args:
        require_battery_pass: when True, refuse to write the package if any
            battery (parse or IR equivalence) still has failures after
            refinement.
        force_llm: when True, run at least one LLM refinement pass on top of
            the deterministic baseline for both grammar and lowering, even
            when the baseline already passes. Requires `llm_client`.
    """
    bundle, grammar_result, lowering_result = build_package_bundle(
        spec,
        model=model,
        provider=provider,
        lifecycle=lifecycle,
        llm_client=llm_client,
        grammar_max_iterations=grammar_max_iterations,
        lowering_max_iterations=lowering_max_iterations,
        force_llm=force_llm,
    )
    if require_battery_pass:
        failures: list[str] = []
        if not grammar_result.ok:
            failures.append(f"grammar: {grammar_result.report.summary()}")
        if not lowering_result.ok:
            failures.append(f"lowering: {lowering_result.report.summary()}")
        if failures:
            raise BatteryError(
                f"battery failed for dialect {spec.name!r}: "
                + "; ".join(failures),
                grammar_result=grammar_result,
                lowering_result=lowering_result,
            )
    target = root / spec.name
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"Dialect directory exists: {target}. Pass overwrite=True to replace."
        )
    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for filename, contents in bundle.files().items():
        p = target / filename
        p.write_text(contents, encoding="utf-8")
        written.append(filename)
    return PackageWriteResult(
        name=spec.name,
        path=target,
        written_files=written,
        grammar_result=grammar_result,
        lowering_result=lowering_result,
    )


class BatteryError(RuntimeError):
    """Raised when `require_battery_pass=True` and any battery has failures."""

    def __init__(
        self,
        message: str,
        *,
        grammar_result: GrammarAgentResult,
        lowering_result: LoweringAgentResult,
    ) -> None:
        super().__init__(message)
        self.grammar_result = grammar_result
        self.lowering_result = lowering_result


_INIT_TEMPLATE = '"""Generated dialect package: {name}."""\n'

_LOWERING_STUB_TEMPLATE = '''"""Lowering stub: codegen could not produce a lowering for this spec.

The deterministic emitter raised:
  {reason}

Loading this dialect will fail until a hand-authored or LLM-refined
lowering replaces this stub.
"""

from typing import Any


def lower(tree: Any, config: Any, catalog: Any):  # pragma: no cover - stub
    raise NotImplementedError(
        "lowering stub for spec; provide a real implementation"
    )
'''
