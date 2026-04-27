"""Multi-oracle verification harness.

Given a plan + semantics + catalog + an "actual" result (typically from the
Polars executor or a generated dialect), the harness:

1. Filters oracles to those applicable (`can_evaluate`).
2. Picks a primary by `confidence`.
3. Runs all applicable oracles.
4. Compares actual vs. each oracle, and oracles against each other.
5. Returns a Verdict: PASS / FAIL / NEEDS_REVIEW / NO_ORACLE.

Disagreement between oracles is *itself* informative: it usually means the IR
plan touches a corner case where engines diverge. We surface this as
NEEDS_REVIEW so the codegen/dataset curator can intervene rather than blindly
trusting any single oracle.

When `actual` is not supplied, the harness builds it via the Polars executor
and forwards the dialect's `overrides`, `passes`, and `effects` modules so
the canonical executor reflects whichever extension lanes the dialect uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import polars as pl

from manysql.executor import execute as polars_execute
from manysql.ir.plan import Plan
from manysql.oracle.base import (
    Oracle,
    OracleResult,
    frames_equal,
    is_order_sensitive,
)
from manysql.oracle.duckdb_oracle import DuckDBOracle
from manysql.oracle.property_oracle import PropertyOracle
from manysql.oracle.reference_interpreter import ReferenceInterpreter
from manysql.oracle.sqlite_oracle import SQLiteOracle
from manysql.spec.semantics import SemanticConfig


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NEEDS_REVIEW = "needs_review"
    NO_ORACLE = "no_oracle"


@dataclass
class HarnessReport:
    verdict: Verdict
    primary: Optional[str]
    oracle_results: list[OracleResult] = field(default_factory=list)
    actual_vs_primary_reason: Optional[str] = None
    inter_oracle_disagreements: list[str] = field(default_factory=list)
    property_failures: list[str] = field(default_factory=list)
    actual: Optional[pl.DataFrame] = None

    def to_summary(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "primary": self.primary,
            "oracles_run": [r.oracle for r in self.oracle_results],
            "oracle_errors": {r.oracle: r.error for r in self.oracle_results if r.error},
            "primary_reason": self.actual_vs_primary_reason,
            "disagreements": self.inter_oracle_disagreements,
            "property_failures": self.property_failures,
        }


def default_oracles() -> list[Oracle]:
    """Row-producing oracles in order of theoretical confidence; harness
    re-sorts per plan."""
    return [
        DuckDBOracle(),
        SQLiteOracle(),
        ReferenceInterpreter(),
    ]


def default_property_oracles() -> list[PropertyOracle]:
    """Property oracles run in addition to row oracles. Always applicable."""
    return [PropertyOracle()]


class OracleHarness:
    def __init__(
        self,
        oracles: Optional[list[Oracle]] = None,
        property_oracles: Optional[list[PropertyOracle]] = None,
    ) -> None:
        self.oracles = oracles or default_oracles()
        self.property_oracles = (
            property_oracles
            if property_oracles is not None
            else default_property_oracles()
        )

    def verify(
        self,
        plan: Plan,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
        actual: Optional[pl.DataFrame] = None,
        *,
        overrides: Optional[Any] = None,
        passes: Optional[Any] = None,
        effects: Optional[Any] = None,
    ) -> HarnessReport:
        """If `actual` is not provided, the Polars executor is used as the actual.

        When the harness materializes `actual` itself, it forwards the
        dialect's `overrides`, `passes`, and `effects` modules to the
        executor so dialect-specific function bodies, plan rewrites, and
        runtime decision-point handlers are honored. The oracles
        themselves run on the same plan + semantics + catalog and don't
        receive these modules — they are reference engines, not the
        system under test.
        """
        if actual is None:
            actual = polars_execute(
                plan,
                semantics,
                catalog,
                overrides=overrides,
                passes=passes,
                effects=effects,
            )

        property_failures = self._run_property_oracles(
            plan, actual, semantics, catalog
        )

        applicable = [o for o in self.oracles if o.can_evaluate(plan, semantics)]
        if not applicable:
            verdict = Verdict.FAIL if property_failures else Verdict.NO_ORACLE
            return HarnessReport(
                verdict=verdict,
                primary=None,
                actual=actual,
                property_failures=property_failures,
            )

        applicable.sort(key=lambda o: o.capability.confidence, reverse=True)

        results: list[OracleResult] = []
        for o in applicable:
            results.append(o.evaluate(plan, semantics, catalog))

        usable = [r for r in results if r.error is None and r.rows is not None]
        if not usable:
            verdict = Verdict.FAIL if property_failures else Verdict.NO_ORACLE
            return HarnessReport(
                verdict=verdict,
                primary=None,
                oracle_results=results,
                actual=actual,
                actual_vs_primary_reason="all oracles errored",
                property_failures=property_failures,
            )

        ordered = is_order_sensitive(plan)
        disagreements: list[str] = []
        for i in range(len(usable)):
            for j in range(i + 1, len(usable)):
                eq, reason = _compare(usable[i].rows, usable[j].rows, ordered=ordered)
                if not eq:
                    disagreements.append(
                        f"{usable[i].oracle} vs {usable[j].oracle}: {reason}"
                    )

        primary = usable[0]

        if property_failures:
            return HarnessReport(
                verdict=Verdict.FAIL,
                primary=primary.oracle,
                oracle_results=results,
                inter_oracle_disagreements=disagreements,
                property_failures=property_failures,
                actual=actual,
                actual_vs_primary_reason="property oracle violation",
            )

        if disagreements:
            return HarnessReport(
                verdict=Verdict.NEEDS_REVIEW,
                primary=primary.oracle,
                oracle_results=results,
                inter_oracle_disagreements=disagreements,
                actual=actual,
                property_failures=property_failures,
            )

        eq, reason = _compare(actual, primary.rows, ordered=ordered)
        if eq:
            return HarnessReport(
                verdict=Verdict.PASS,
                primary=primary.oracle,
                oracle_results=results,
                actual=actual,
                property_failures=property_failures,
            )
        return HarnessReport(
            verdict=Verdict.FAIL,
            primary=primary.oracle,
            oracle_results=results,
            actual_vs_primary_reason=reason,
            actual=actual,
            property_failures=property_failures,
        )

    def _run_property_oracles(
        self,
        plan: Plan,
        actual: pl.DataFrame,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
    ) -> list[str]:
        """Run every property oracle and return concatenated violation notes."""
        out: list[str] = []
        for po in self.property_oracles:
            result = po.check_properties(plan, actual, semantics, catalog)
            if result.property_passed is False:
                for note in result.notes:
                    out.append(f"{po.capability.name}: {note}")
        return out


def _compare(a: pl.DataFrame, b: pl.DataFrame, *, ordered: bool) -> tuple[bool, str]:
    """Compare two oracle frames. Tolerates column-name divergence: if both
    frames have the same width, we rename b's columns to match a's before
    comparing."""
    if a.width == b.width and a.columns != b.columns:
        b = b.rename(dict(zip(b.columns, a.columns, strict=True)))
    return frames_equal(a, b, ordered=ordered)


__all__ = [
    "OracleHarness",
    "HarnessReport",
    "Verdict",
    "default_oracles",
    "default_property_oracles",
]
