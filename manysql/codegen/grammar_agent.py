"""Grammar codegen agent: deterministic baseline + optional LLM refine loop.

The deterministic emitter (`emit_grammar`) handles the supported surface
knobs mechanically. Any spec whose changes are within those knobs should
pass the parse battery on the first attempt.

For specs that exercise structural changes the deterministic emitter cannot
express (e.g. `JoinSyntax.PIPELINED`, exotic `LimitSyntax.HEAD_N`), the
agent can fall back to an LLM refine loop:

  1. Render the deterministic baseline.
  2. Run the parse battery; if it passes, done.
  3. Otherwise, send the LLM (a) the spec, (b) the current grammar text,
     (c) every battery failure with its source SQL and parser error.
  4. Replace the grammar with the LLM's reply; rerun the battery.
  5. Repeat up to `max_iterations`.

The agent is deliberately bounded and stateless across runs — it returns a
final `GrammarAgentResult` with the grammar text plus a structured trace
caller can log or inspect.

Without an LLM client (or with `NullLLMClient`), the agent simply runs the
deterministic emitter once and reports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from manysql.codegen.grammar_emit import emit_grammar
from manysql.codegen.parse_battery import (
    BatteryItem,
    RejectionBatteryItem,
    RejectionReport,
    ValidationReport,
    build_parse_battery,
    build_rejection_battery,
    validate_grammar,
    validate_rejection_battery,
)
from manysql.llm.client import LLMClient, LLMError, NullLLMClient
from manysql.spec.dialect import DialectSpec


@dataclass(frozen=True)
class GrammarAttempt:
    iteration: int
    source: str  # "deterministic" | "llm"
    grammar: str
    report: ValidationReport
    rejection_report: Optional[RejectionReport] = None


@dataclass(frozen=True)
class GrammarAgentResult:
    grammar: str
    report: ValidationReport
    attempts: list[GrammarAttempt] = field(default_factory=list)
    rejection_items: list[RejectionBatteryItem] = field(default_factory=list)
    rejection_report: Optional[RejectionReport] = None

    @property
    def ok(self) -> bool:
        rejection_ok = (
            self.rejection_report is None or self.rejection_report.ok
        )
        return self.report.ok and rejection_ok


_GRAMMAR_SYSTEM_PROMPT = """You are a grammar engineer. You modify a Lark
grammar so that it parses every example query for a target SQL dialect.

Rules:
- Reply with ONLY the full grammar text. No markdown fences, no commentary.
- Preserve all rule names that downstream tooling depends on (start, statement,
  query_expr, select_core, select_list, from_clause, table_ref, join_clause,
  join_kind, where_clause, group_by_clause, having_clause, order_by, order_key,
  limit_clause, expr, expr_list, comparison, additive, multiplicative, unary,
  primary, literal, column_ref, function_call, case_expr, cast_expr, type_name,
  with_clause, cte_list, cte, set_op, union, intersect, except_, eq, neq, lt,
  lte, gt, gte, add, sub, concat, mul, div, mod, neg, pos, not_op, is_null,
  is_not_null, between_op, not_between_op, in_list, not_in_list, in_subquery_op,
  not_in_subquery_op, like_op, ilike_op, distinct_from, not_distinct_from,
  number_literal, string_literal, true_literal, false_literal, null_literal,
  date_literal, exists_expr, not_exists_expr, scalar_subquery, paren_expr,
  star_args, normal_args, distinct_kw, filter_clause, over_clause, partition_by,
  order_by, case_branch, case_else, IDENTIFIER, NUMBER, STRING, INT).
- Do not introduce new terminals unless absolutely required.
- Keep all `%ignore` directives. Add new ones only for new comment styles.
"""


_GRAMMAR_VERIFY_INSTRUCTION = (
    "Verify that the grammar implements EVERY non-default surface knob from the\n"
    "DialectSpec. For each axis below, the grammar must (a) accept the dialect's\n"
    "spelling (exercised by the must-parse battery), and (b) REJECT the original\n"
    "ANSI spelling (exercised by the must-reject battery). Remove any obsolete\n"
    "branches from the deterministic baseline that still admit the ANSI form.\n"
    "Specifically check: keyword renames, operator renames (eq/neq/lt/lte/gt/ge,\n"
    "add/sub/mul/div/mod, concat, null-safe-eq), wildcard char, identifier and\n"
    "string quote characters, NULL literal spelling, statement terminator, and\n"
    "set-op precedence (parsed flat; precedence handled in lowering)."
)


_GRAMMAR_POLISH_INSTRUCTION = _GRAMMAR_VERIFY_INSTRUCTION


def generate_grammar(
    spec: DialectSpec,
    *,
    llm_client: Optional[LLMClient] = None,
    max_iterations: int = 5,
    force_llm: bool = False,
) -> GrammarAgentResult:
    """Produce a grammar that parses the parse battery, refining via LLM if needed.

    Args:
        spec: the dialect spec.
        llm_client: optional LLM client. If `None` or a `NullLLMClient`, no
            refinement is attempted; the deterministic baseline is returned
            as-is even on failure.
        max_iterations: max LLM rounds (in addition to the deterministic
            baseline). Ignored when there is no LLM.
        force_llm: when True, run at least one LLM refinement pass even if
            the deterministic baseline already passes the battery. The LLM's
            output is accepted only if it still passes; otherwise we revert to
            the last known-good grammar. This is the path that exercises the
            LLM lane on simple specs the deterministic emitter can already
            handle.
    """
    items = build_parse_battery(spec)
    rejection_items = build_rejection_battery(spec)
    grammar = emit_grammar(spec)
    report = validate_grammar(grammar, items)
    rejection_report = validate_rejection_battery(grammar, rejection_items)
    attempts: list[GrammarAttempt] = [
        GrammarAttempt(
            iteration=0,
            source="deterministic",
            grammar=grammar,
            report=report,
            rejection_report=rejection_report,
        )
    ]
    no_llm = llm_client is None or isinstance(llm_client, NullLLMClient)

    def _both_ok(parse: ValidationReport, reject: RejectionReport) -> bool:
        return parse.ok and reject.ok

    if no_llm or (_both_ok(report, rejection_report) and not force_llm):
        return GrammarAgentResult(
            grammar=grammar,
            report=report,
            attempts=attempts,
            rejection_items=rejection_items,
            rejection_report=rejection_report,
        )

    # Snapshot the last known-good (grammar, report, rejection_report) so we
    # can revert if a forced LLM polish pass regresses against either battery.
    last_good: Optional[tuple[str, ValidationReport, RejectionReport]] = (
        (grammar, report, rejection_report)
        if _both_ok(report, rejection_report)
        else None
    )

    for iteration in range(1, max_iterations + 1):
        try:
            new_grammar = _refine_with_llm(
                spec=spec,
                grammar=grammar,
                items=items,
                report=report,
                rejection_items=rejection_items,
                rejection_report=rejection_report,
                llm_client=llm_client,
                polish=_both_ok(report, rejection_report),
            )
        except LLMError:
            break
        new_report = validate_grammar(new_grammar, items)
        new_rejection = validate_rejection_battery(new_grammar, rejection_items)
        attempts.append(
            GrammarAttempt(
                iteration=iteration,
                source="llm",
                grammar=new_grammar,
                report=new_report,
                rejection_report=new_rejection,
            )
        )
        if _both_ok(new_report, new_rejection):
            grammar = new_grammar
            report = new_report
            rejection_report = new_rejection
            last_good = (grammar, report, rejection_report)
            # One passing LLM iteration is sufficient. For force_llm callers
            # this is enough to prove the LLM contributed; for fix-mode both
            # batteries are satisfied.
            break
        if last_good is not None:
            # We had a passing baseline (or earlier passing iteration) and the
            # LLM regressed. Don't keep a worse grammar around.
            grammar, report, rejection_report = last_good
            break
        # Baseline was already failing; keep the LLM's text as the working
        # grammar so the next iteration sees the most recent attempt.
        grammar = new_grammar
        report = new_report
        rejection_report = new_rejection
    return GrammarAgentResult(
        grammar=grammar,
        report=report,
        attempts=attempts,
        rejection_items=rejection_items,
        rejection_report=rejection_report,
    )


def _refine_with_llm(
    *,
    spec: DialectSpec,
    grammar: str,
    items: list[BatteryItem],
    report: ValidationReport,
    rejection_items: list[RejectionBatteryItem],
    rejection_report: RejectionReport,
    llm_client: LLMClient,
    polish: bool = False,
) -> str:
    """Send the LLM the spec + current grammar + both batteries.

    The LLM must produce a grammar that satisfies BOTH batteries:
      - every parse-battery item must still parse
      - every rejection-battery item must FAIL to parse
    ``polish`` is informational; the verify-axis prompt is the same in both
    branches so the model always sees the full task.
    """
    spec_summary = json.dumps(
        {
            "name": spec.name,
            "divergence": spec.divergence.value,
            "surface": spec.surface.model_dump(mode="json"),
        },
        indent=2,
    )
    must_parse_block = "\n".join(
        f"  - {item.label}: {item.source}" for item in items
    )
    if rejection_items:
        must_reject_block = "\n".join(
            f"  - {item.label}: {item.source}  # {item.reason}"
            for item in rejection_items
        )
    else:
        must_reject_block = "  (none — spec is at default on every rejection axis)"
    parse_failure_block = "\n\n".join(
        f"### parse-fail: {f.label}\nSQL:\n  {f.source}\nError:\n  {f.error}"
        for f in report.failures
    ) or "  (none)"
    reject_failure_block = "\n\n".join(
        f"### over-permissive: {f.label}\nSQL:\n  {f.source}\nReason:\n  {f.reason}"
        for f in rejection_report.failures
    ) or "  (none)"
    task_block = (
        f"{_GRAMMAR_VERIFY_INSTRUCTION}\n\n"
        f"Must-parse battery (every item must parse against the final grammar):\n"
        f"{must_parse_block}\n\n"
        f"Must-reject battery (every item must FAIL to parse against the final grammar):\n"
        f"{must_reject_block}\n\n"
        f"Current parse-battery failures:\n{parse_failure_block}\n\n"
        f"Current rejection-battery failures (grammar still accepts these "
        f"reference-form strings, defeating the divergence):\n{reject_failure_block}"
    )
    user = (
        f"DialectSpec:\n```json\n{spec_summary}\n```\n\n"
        f"Current grammar:\n```lark\n{grammar}\n```\n\n"
        f"{task_block}\n\n"
        "Reply with the full corrected grammar text only."
    )
    response = llm_client.chat(
        system=_GRAMMAR_SYSTEM_PROMPT,
        user=user,
        temperature=0.0,
    )
    return _strip_code_fences(response.text)


def _strip_code_fences(text: str) -> str:
    """If the LLM wrapped the grammar in ``` fences, peel them off."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    # Drop opening fence (and an optional language tag).
    lines = lines[1:]
    # Drop trailing fence if present.
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


__all__ = [
    "GrammarAgentResult",
    "GrammarAttempt",
    "generate_grammar",
]
