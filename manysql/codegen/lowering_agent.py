"""Lowering codegen agent: deterministic baseline + IR-equivalence refine loop.

For surface-only dialects the deterministic emitter (`emit_lowering`) reuses
the reference lowering verbatim, so the IR battery passes on the first try.

For structural dialects (e.g. `JoinSyntax.PIPELINED`) the deterministic
emitter raises `NotImplementedError`. In that case the agent:

  1. Renders the deterministic baseline (using the reference lowering as a
     prompt seed even though it doesn't apply directly).
  2. Compiles + validates against the IR battery using the spec's grammar
     (already produced by the grammar agent).
  3. Asks the LLM to rewrite specific lowering helpers given the spec, the
     reference lowering source, and the divergent battery items.
  4. Iterates up to `max_iterations`.

The agent is bounded and stateless across runs; results are returned as a
`LoweringAgentResult` so the caller can inspect attempts.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from importlib.util import module_from_spec, spec_from_loader
from types import ModuleType
from typing import Optional

from lark import Lark
from lark.exceptions import LarkError

from manysql.codegen.config_emit import compose_semantic_config
from manysql.codegen.grammar_emit import emit_grammar
from manysql.codegen.ir_battery import (
    IRBatteryItem,
    IREquivalenceReport,
    build_ir_battery,
    validate_lowering,
)
from manysql.codegen.lowering_emit import emit_lowering, emit_lowering_seed
from manysql.llm.client import LLMClient, LLMError, NullLLMClient
from manysql.spec.dialect import DialectSpec


@dataclass(frozen=True)
class LoweringAttempt:
    iteration: int
    source: str  # "deterministic" | "llm" | "skipped"
    lowering_py: str
    report: IREquivalenceReport


@dataclass(frozen=True)
class LoweringAgentResult:
    lowering_py: str
    report: IREquivalenceReport
    attempts: list[LoweringAttempt] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.report.ok


_LOWERING_SYSTEM_PROMPT = """You are a Python engineer. You modify a Lark
parse-tree-to-IR lowering module so that the dialect's parse trees lower to
the same manysql IR plans the reference dialect produces.

Rules:
- Reply with ONLY the full Python file. No markdown fences, no commentary.
- Keep the public function signature: `def lower(tree, config, catalog) -> Plan`.
- Use `from manysql.ir.plan import ...` and `from manysql.ir.expr import ...`
  for IR types; do NOT redefine them.
- Preserve all imports and helpers required by `lower`. Do not introduce
  external dependencies.
"""


_LOWERING_VERIFY_INSTRUCTION = (
    "Verify that the lowering implements EVERY non-default semantic axis from\n"
    "the DialectSpec, producing the same IR plans as the reference dialect for\n"
    "every battery query. Per axis: function-alias canonicalization (NVL/IFNULL\n"
    "-> COALESCE, LEN -> LENGTH, etc.), set-op precedence (ANSI vs\n"
    "EXCEPT_INTERSECT_TIGHTER, read from SemanticConfig.set_op_precedence),\n"
    "operator overloads, identifier-quote stripping, and any structural surface\n"
    "rewrites the deterministic emitter could not handle (CASE syntax, JOIN\n"
    "syntax, LIMIT shape). The IR-equivalence battery is the ground truth: a\n"
    "passing battery is sufficient and necessary."
)


_LOWERING_POLISH_INSTRUCTION = _LOWERING_VERIFY_INSTRUCTION


def generate_lowering(
    spec: DialectSpec,
    *,
    grammar_text: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    max_iterations: int = 5,
    force_llm: bool = False,
) -> LoweringAgentResult:
    """Produce a lowering module that satisfies the IR battery.

    Args:
        grammar_text: the dialect's grammar (typically from the grammar
            agent). If omitted, the deterministic grammar is rendered.
        llm_client: optional LLM client. With a `NullLLMClient` (or no
            client) the agent only runs the deterministic path.
        max_iterations: max LLM rounds when refining.
        force_llm: when True, run at least one LLM refinement pass even when
            the deterministic baseline already passes the IR battery. The
            LLM's output is accepted only if the battery still passes;
            otherwise we revert to the deterministic baseline.
    """
    grammar = grammar_text or emit_grammar(spec)
    items = build_ir_battery(spec)
    semantics = compose_semantic_config(spec)

    try:
        deterministic_text = emit_lowering(spec)
        deterministic_module = _load_module(deterministic_text, f"_codegen_{spec.name}_det")
    except NotImplementedError:
        deterministic_text = ""
        deterministic_module = None

    no_llm = llm_client is None or isinstance(llm_client, NullLLMClient)

    if deterministic_module is not None:
        report = validate_lowering(
            grammar_text=grammar,
            lowering_module=deterministic_module,
            semantics=semantics,
            items=items,
        )
        attempts = [
            LoweringAttempt(
                iteration=0,
                source="deterministic",
                lowering_py=deterministic_text,
                report=report,
            )
        ]
        if no_llm or (report.ok and not force_llm):
            return LoweringAgentResult(
                lowering_py=deterministic_text,
                report=report,
                attempts=attempts,
            )
        text = deterministic_text
        # Snapshot a known-good (text, report) so a forced polish pass that
        # regresses can be rolled back.
        last_good: Optional[tuple[str, IREquivalenceReport]] = (
            (deterministic_text, report) if report.ok else None
        )
    else:
        empty_report = IREquivalenceReport(
            items=items,
            divergences=[
                # Pre-populate so the LLM prompt has the items the spec
                # cannot lower deterministically.
                _placeholder_divergence(item)
                for item in items
            ],
        )
        attempts = [
            LoweringAttempt(
                iteration=0,
                source="skipped",
                lowering_py="",
                report=empty_report,
            )
        ]
        if no_llm:
            return LoweringAgentResult(
                lowering_py="",
                report=empty_report,
                attempts=attempts,
            )
        # Seed the LLM with a reference lowering that already has every
        # surface-only patch applied (e.g. _lower_limit specialized for
        # this spec's LimitSyntax). The structural pieces that triggered
        # NotImplementedError remain for the model to fix, but it no
        # longer has to reinvent the limit/offset extraction by reading
        # tokens that Lark filters out of the tree.
        text = emit_lowering_seed(spec)
        report = empty_report
        last_good = None

    for iteration in range(1, max_iterations + 1):
        try:
            text = _refine_with_llm(
                spec=spec,
                lowering_text=text,
                grammar_text=grammar,
                items=items,
                report=report,
                llm_client=llm_client,
                polish=report.ok,
            )
        except LLMError:
            break
        try:
            module = _load_module(text, f"_codegen_{spec.name}_iter{iteration}")
        except SyntaxError as exc:
            attempts.append(
                LoweringAttempt(
                    iteration=iteration,
                    source="llm",
                    lowering_py=text,
                    report=IREquivalenceReport(
                        items=items,
                        divergences=[
                            _placeholder_divergence(items[0], error=f"SyntaxError: {exc}")
                        ],
                    ),
                )
            )
            if last_good is not None:
                # Forced polish produced unparseable Python; revert.
                text, report = last_good
                break
            continue
        new_report = validate_lowering(
            grammar_text=grammar,
            lowering_module=module,
            semantics=semantics,
            items=items,
        )
        attempts.append(
            LoweringAttempt(
                iteration=iteration,
                source="llm",
                lowering_py=text,
                report=new_report,
            )
        )
        if new_report.ok:
            report = new_report
            last_good = (text, report)
            break
        if last_good is not None:
            # Regressed against a previously passing baseline; revert.
            text, report = last_good
            break
        report = new_report
    return LoweringAgentResult(
        lowering_py=text,
        report=report,
        attempts=attempts,
    )


def _refine_with_llm(
    *,
    spec: DialectSpec,
    lowering_text: str,
    grammar_text: str,
    items: list[IRBatteryItem],
    report: IREquivalenceReport,
    llm_client: LLMClient,
    polish: bool = False,
) -> str:
    """Send the LLM the spec, current lowering, and divergent battery items.

    `polish=True` means the IR battery is already passing and the caller is
    forcing an LLM iteration; we tell the model to refine without changing
    the IR plans returned for any battery query.

    For non-polish (fix-mode) calls we additionally render the actual Lark
    parse tree for each failing item using ``grammar_text``. Lark drops
    anonymous string terminals (e.g. ``"OFFSET"i``) from the tree by
    default, so without seeing the tree the model often hallucinates
    keyword tokens that aren't there at runtime — exactly the failure mode
    we observed in batch-15's ``sqlite_bigquery_db2`` warn.
    """
    spec_summary = json.dumps(
        {
            "name": spec.name,
            "divergence": spec.divergence.value,
            "surface": spec.surface.model_dump(mode="json"),
        },
        indent=2,
    )
    battery_block = "\n".join(
        f"  - {item.label}\n      ref:     {item.ref_sql}\n      dialect: {item.dialect_sql}"
        for item in items
    )
    if polish:
        task_block = (
            f"{_LOWERING_VERIFY_INSTRUCTION}\n\n"
            f"IR battery (every item must lower to the reference plan):\n"
            f"{battery_block}"
        )
    else:
        trees = _parse_failure_trees(grammar_text, report.divergences)
        failure_block = "\n\n".join(
            f"### {d.label}\n"
            f"reference SQL:\n  {d.ref_sql}\n"
            f"dialect SQL:\n  {d.dialect_sql}\n"
            f"dialect parse tree (Lark; anonymous keyword terminals are NOT shown "
            f"because Lark filters them out by default):\n{trees.get(d.label, '  <unavailable>')}\n"
            f"reference plan:\n{d.ref_plan or '<unavailable>'}\n"
            f"dialect plan:\n{d.dialect_plan or '<unavailable>'}\n"
            f"error:\n  {d.error}"
            for d in report.divergences
        )
        task_block = (
            f"{_LOWERING_VERIFY_INSTRUCTION}\n\n"
            f"IR battery (every item must lower to the reference plan):\n"
            f"{battery_block}\n\n"
            f"Current divergences (must be repaired):\n{failure_block}"
        )
    user = (
        f"DialectSpec:\n```json\n{spec_summary}\n```\n\n"
        f"Current lowering.py:\n```python\n{lowering_text}\n```\n\n"
        f"{task_block}\n\n"
        "Reply with the full corrected lowering.py only."
    )
    response = llm_client.chat(
        system=_LOWERING_SYSTEM_PROMPT,
        user=user,
        temperature=0.0,
    )
    return _strip_code_fences(response.text)


def _parse_failure_trees(
    grammar_text: str, divergences: list
) -> dict[str, str]:
    """Render the dialect parse tree for each failing item.

    Returns a per-label string. Best-effort: if the grammar fails to
    build, or a particular item won't parse, we return a short error
    string instead of raising — the prompt is still useful with the rest
    of the trees, and the agent's existing failure handling will retry.
    """
    try:
        parser = Lark(grammar_text, start="start", parser="earley")
    except LarkError as exc:
        return {d.label: f"  <grammar build failed: {exc}>" for d in divergences}
    out: dict[str, str] = {}
    for d in divergences:
        try:
            tree = parser.parse(d.dialect_sql)
        except Exception as exc:  # noqa: BLE001 - surface in-prompt
            out[d.label] = f"  <parse failed: {type(exc).__name__}: {exc}>"
            continue
        pretty = tree.pretty()
        out[d.label] = "\n".join("  " + line for line in pretty.splitlines())
    return out


def _load_module(source: str, fullname: str) -> ModuleType:
    """Compile a Python source string into a fresh module object.

    The module is registered in `sys.modules` before exec, because Python's
    `dataclass` machinery (used heavily in the IR module imports) looks up
    `cls.__module__` in `sys.modules` to resolve type annotations.
    """

    class _Loader:
        def create_module(self, spec):  # noqa: D401, ARG002
            return None

        def exec_module(self, module):  # noqa: D401
            exec(compile(source, fullname, "exec"), module.__dict__)

    spec = spec_from_loader(fullname, _Loader())
    if spec is None:
        raise RuntimeError(f"could not build module spec for {fullname}")
    module = module_from_spec(spec)
    sys.modules[fullname] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(fullname, None)
        raise
    return module


def _placeholder_divergence(
    item: IRBatteryItem, *, error: str = "no deterministic lowering"
):
    from manysql.codegen.ir_battery import IRDivergence

    return IRDivergence(
        label=item.label,
        ref_sql=item.ref_sql,
        dialect_sql=item.dialect_sql,
        ref_plan=None,
        dialect_plan=None,
        error=error,
    )


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


__all__ = [
    "LoweringAgentResult",
    "LoweringAttempt",
    "generate_lowering",
]
