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
    ValidationReport,
    build_parse_battery,
    validate_grammar,
)
from manysql.llm.client import LLMClient, LLMError, NullLLMClient
from manysql.spec.dialect import DialectSpec


@dataclass(frozen=True)
class GrammarAttempt:
    iteration: int
    source: str  # "deterministic" | "llm"
    grammar: str
    report: ValidationReport


@dataclass(frozen=True)
class GrammarAgentResult:
    grammar: str
    report: ValidationReport
    attempts: list[GrammarAttempt] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.report.ok


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


_GRAMMAR_POLISH_INSTRUCTION = (
    "The current grammar already parses every battery query. Make small, "
    "targeted refinements (clearer rule names where harmless, tighter "
    "alternations, deduplicated terminals) WITHOUT changing the set of strings "
    "the grammar accepts. Every battery query must still parse."
)


def generate_grammar(
    spec: DialectSpec,
    *,
    llm_client: Optional[LLMClient] = None,
    max_iterations: int = 3,
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
    grammar = emit_grammar(spec)
    report = validate_grammar(grammar, items)
    attempts: list[GrammarAttempt] = [
        GrammarAttempt(
            iteration=0,
            source="deterministic",
            grammar=grammar,
            report=report,
        )
    ]
    no_llm = llm_client is None or isinstance(llm_client, NullLLMClient)
    if no_llm or (report.ok and not force_llm):
        return GrammarAgentResult(grammar=grammar, report=report, attempts=attempts)

    # Snapshot the last known-good (grammar, report) so we can revert if a
    # forced LLM polish pass regresses against the battery.
    last_good: Optional[tuple[str, "ValidationReport"]] = (
        (grammar, report) if report.ok else None
    )

    for iteration in range(1, max_iterations + 1):
        try:
            new_grammar = _refine_with_llm(
                spec=spec,
                grammar=grammar,
                items=items,
                report=report,
                llm_client=llm_client,
                polish=report.ok,
            )
        except LLMError:
            break
        new_report = validate_grammar(new_grammar, items)
        attempts.append(
            GrammarAttempt(
                iteration=iteration,
                source="llm",
                grammar=new_grammar,
                report=new_report,
            )
        )
        if new_report.ok:
            grammar = new_grammar
            report = new_report
            last_good = (grammar, report)
            # One passing LLM iteration is sufficient. For force_llm callers
            # this is enough to prove the LLM contributed; for fix-mode the
            # battery is satisfied.
            break
        if last_good is not None:
            # We had a passing baseline (or earlier passing iteration) and the
            # LLM regressed. Don't keep a worse grammar around.
            grammar, report = last_good
            break
        # Baseline was already failing; keep the LLM's text as the working
        # grammar so the next iteration sees the most recent attempt.
        grammar = new_grammar
        report = new_report
    return GrammarAgentResult(grammar=grammar, report=report, attempts=attempts)


def _refine_with_llm(
    *,
    spec: DialectSpec,
    grammar: str,
    items: list[BatteryItem],
    report: ValidationReport,
    llm_client: LLMClient,
    polish: bool = False,
) -> str:
    """Send the LLM the spec + current grammar, asking for a fix or polish.

    `polish=True` means the battery is currently passing and the caller is
    forcing an LLM iteration; we tell the model to refine without changing
    the language. Otherwise we send each failing battery item.
    """
    spec_summary = json.dumps(
        {
            "name": spec.name,
            "divergence": spec.divergence.value,
            "surface": spec.surface.model_dump(mode="json"),
        },
        indent=2,
    )
    if polish:
        task_block = (
            f"{_GRAMMAR_POLISH_INSTRUCTION}\n\n"
            f"Battery (all currently parse):\n"
            + "\n".join(f"  - {item.label}: {item.source}" for item in items)
        )
    else:
        failure_block = "\n\n".join(
            f"### {f.label}\nSQL:\n  {f.source}\nError:\n  {f.error}"
            for f in report.failures
        )
        task_block = (
            f"Parse-battery failures (must all parse after your fix):\n{failure_block}"
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
