"""Cross-dialect differential oracle.

Premise: a SQL question is *the same question* regardless of which dialect
expresses it. If we generate two dialects from different specs, ask each
the same logical query, and they disagree, then either:

  - the logical query is genuinely sensitive to a dialect-specific
    semantic knob (we want to flag this and curate it), or
  - one of the dialects has a bug in its generated grammar/lowering/
    semantics (we want to flag this and refine it).

Either way, disagreement is a high-signal training-data label and the
strongest test we have for "did codegen produce a self-consistent
dialect family?"

Architecture: callers provide a small set of `CrossDialectMember`
records — one per dialect — pre-loaded with parser, lowering, semantics,
overrides, and the `SurfaceSpec` needed to rewrite reference SQL into
the dialect's surface. The oracle takes a reference SQL string + catalog
and runs the full pipeline (rewrite -> parse -> lower -> execute) on
every dialect, then compares results pairwise.

Members are pre-loaded rather than created on demand so that one
expensive setup (parser construction, dialect package generation) can
amortize across many queries.

Comparison: results are normalized (column-name and ordering tolerant)
before comparison, mirroring the inter-oracle comparison in
`OracleHarness`. Order-sensitive plans (those whose SQL contains
ORDER BY at the top level) compare position-wise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import polars as pl
from lark import Lark

from manysql.codegen.parse_battery import apply_surface
from manysql.executor import execute as polars_execute
from manysql.ir.plan import ColumnSchema
from manysql.oracle.base import frames_equal, normalize_for_comparison
from manysql.spec.dialect import SurfaceSpec
from manysql.spec.semantics import SemanticConfig


@dataclass
class CrossDialectMember:
    """One dialect participating in a differential comparison."""

    name: str
    surface: SurfaceSpec
    parser: Lark
    lowering: Any  # the lowering module
    semantics: SemanticConfig
    overrides: Optional[Any] = None
    passes: Optional[Any] = None
    effects: Optional[Any] = None


class CrossDialectVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NEEDS_REVIEW = "needs_review"
    NO_DIALECTS = "no_dialects"


@dataclass
class DialectExecution:
    """One member's pipeline trace for a single query."""

    name: str
    rewritten_sql: str
    rows: Optional[pl.DataFrame] = None
    error: Optional[str] = None


@dataclass
class CrossDialectReport:
    verdict: CrossDialectVerdict
    reference_sql: str
    executions: list[DialectExecution] = field(default_factory=list)
    disagreements: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_summary(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "reference_sql": self.reference_sql,
            "errored_dialects": {
                e.name: e.error for e in self.executions if e.error
            },
            "disagreements": self.disagreements,
            "notes": self.notes,
        }


class CrossDialectOracle:
    """Runs the same logical query across multiple dialects and compares."""

    def __init__(
        self,
        members: list[CrossDialectMember],
        schemas: dict[str, tuple[ColumnSchema, ...]],
    ) -> None:
        self.members = members
        self.schemas = schemas

    def verify(
        self,
        reference_sql: str,
        catalog: dict[str, pl.DataFrame],
    ) -> CrossDialectReport:
        """Run `reference_sql` (in reference surface) through each dialect.

        Catalog is shared: every dialect reads the same data, so disagreement
        cannot be blamed on inconsistent inputs.
        """
        if not self.members:
            return CrossDialectReport(
                verdict=CrossDialectVerdict.NO_DIALECTS,
                reference_sql=reference_sql,
                notes=["no dialects configured"],
            )

        executions: list[DialectExecution] = []
        for m in self.members:
            executions.append(self._run_one(m, reference_sql, catalog))

        usable = [e for e in executions if e.error is None and e.rows is not None]
        if len(usable) < 2:
            verdict = (
                CrossDialectVerdict.NEEDS_REVIEW
                if usable
                else CrossDialectVerdict.FAIL
            )
            return CrossDialectReport(
                verdict=verdict,
                reference_sql=reference_sql,
                executions=executions,
                notes=["fewer than two dialects produced rows"],
            )

        ordered = _is_order_sensitive_sql(reference_sql)
        disagreements: list[str] = []
        baseline = usable[0]
        for other in usable[1:]:
            eq, reason = _compare_rows(baseline.rows, other.rows, ordered=ordered)
            if not eq:
                disagreements.append(
                    f"{baseline.name} vs {other.name}: {reason}"
                )

        if disagreements:
            return CrossDialectReport(
                verdict=CrossDialectVerdict.FAIL,
                reference_sql=reference_sql,
                executions=executions,
                disagreements=disagreements,
            )
        return CrossDialectReport(
            verdict=CrossDialectVerdict.PASS,
            reference_sql=reference_sql,
            executions=executions,
        )

    def _run_one(
        self,
        member: CrossDialectMember,
        reference_sql: str,
        catalog: dict[str, pl.DataFrame],
    ) -> DialectExecution:
        rewritten = apply_surface(reference_sql, member.surface)
        try:
            tree = member.parser.parse(rewritten)
            plan = member.lowering.lower(tree, member.semantics, self.schemas)
            rows = polars_execute(
                plan,
                member.semantics,
                catalog,
                overrides=member.overrides,
                passes=member.passes,
                effects=member.effects,
            )
        except Exception as exc:  # noqa: BLE001
            return DialectExecution(
                name=member.name,
                rewritten_sql=rewritten,
                error=f"{type(exc).__name__}: {exc}",
            )
        return DialectExecution(
            name=member.name,
            rewritten_sql=rewritten,
            rows=rows,
        )


def _is_order_sensitive_sql(sql: str) -> bool:
    """Heuristic: a query whose top-level structure includes ORDER BY or
    LIMIT must be compared position-wise. We use a cheap textual check so
    we don't need to re-parse the SQL here.
    """
    upper = sql.upper()
    return "ORDER BY" in upper or "LIMIT" in upper


def _compare_rows(
    a: pl.DataFrame, b: pl.DataFrame, *, ordered: bool
) -> tuple[bool, str]:
    if a.width != b.width:
        return False, f"width mismatch: {a.width} vs {b.width}"
    if a.height != b.height:
        return False, f"height mismatch: {a.height} vs {b.height}"
    a_norm = normalize_for_comparison(a, sort_keys=None if ordered else list(a.columns))
    b_norm = normalize_for_comparison(b, column_names=list(a.columns))
    if not ordered:
        b_norm = normalize_for_comparison(b_norm, sort_keys=list(b_norm.columns))
    if ordered:
        b_norm = normalize_for_comparison(b, column_names=list(a.columns))
    return frames_equal(a_norm, b_norm, ordered=ordered)


__all__ = [
    "CrossDialectMember",
    "CrossDialectOracle",
    "CrossDialectReport",
    "CrossDialectVerdict",
    "DialectExecution",
]
