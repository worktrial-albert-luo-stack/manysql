"""DuckDB oracle: render IR -> DuckDB SQL, execute, normalize result."""

from __future__ import annotations

from typing import Any

import duckdb
import polars as pl

from manysql.ir.plan import Plan
from manysql.oracle.base import Oracle, OracleCapability, OracleResult
from manysql.oracle.sql_render import (
    SqlDialectFlags,
    UnsupportedByEngine,
    render_plan,
)
from manysql.spec.semantics import SemanticConfig


class DuckDBOracle(Oracle):
    """Run plans through DuckDB. Highest-coverage SQL oracle.

    Knobs DuckDB cannot honor (without per-row Python rewriting):
    - boolean_truthiness=C_STYLE (DuckDB is strict)
    - count_distinct_null=INCLUDED (DuckDB excludes NULLs always)
    - quoted_identifiers_case_sensitive=False (DuckDB is always case-sensitive)
    - identifier_case_fold=PRESERVE/UPPER (DuckDB folds to lowercase by default)
    """

    @property
    def capability(self) -> OracleCapability:
        return OracleCapability(
            name="duckdb",
            supported_nodes=frozenset(
                {
                    "Scan",
                    "Project",
                    "Filter",
                    "Join",
                    "Aggregate",
                    "Window",
                    "Sort",
                    "Limit",
                    "Distinct",
                    "SetOp",
                    "WithCTE",
                    "RecursiveCTE",
                }
            ),
            supported_features=frozenset(
                {
                    "ansi_core",
                    "windows",
                    "set_ops",
                    "ctes",
                    "recursive_cte",
                    "scalar_subquery_uncorrelated",
                    "exists_uncorrelated",
                }
            ),
            unsupported_knobs=frozenset(
                {
                    "boolean_truthiness",
                    "count_distinct_null",
                    "quoted_identifiers_case_sensitive",
                    "identifier_case_fold",
                }
            ),
            confidence=0.9,
        )

    def evaluate(
        self,
        plan: Plan,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
    ) -> OracleResult:
        try:
            sql, notes = render_plan(plan, semantics, SqlDialectFlags.duckdb())
        except UnsupportedByEngine as exc:
            return OracleResult(oracle=self.capability.name, error=f"unsupported: {exc}")
        try:
            con = duckdb.connect(":memory:")
            for name, df in catalog.items():
                # DuckDB can register Polars/Arrow frames natively
                con.register(name, df.to_arrow())
            rel = con.execute(sql)
            try:
                result = rel.pl()
            except Exception:
                arrow_tbl = rel.arrow()
                result = pl.from_arrow(arrow_tbl)
                if not isinstance(result, pl.DataFrame):
                    result = pl.DataFrame(result)
            return OracleResult(oracle=self.capability.name, rows=result, notes=notes)
        except Exception as exc:  # noqa: BLE001
            return OracleResult(
                oracle=self.capability.name,
                error=f"{type(exc).__name__}: {exc}\nSQL was: {sql}",
            )


__all__ = ["DuckDBOracle"]
