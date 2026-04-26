"""SQLite oracle: secondary SQL engine for cross-engine bug detection.

SQLite has stricter limits than DuckDB. We mark them in capability so the
harness skips this oracle when those limits are exceeded.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from typing import Any

import polars as pl

from manysql.ir.plan import Plan
from manysql.oracle.base import Oracle, OracleCapability, OracleResult
from manysql.oracle.sql_render import (
    SqlDialectFlags,
    UnsupportedByEngine,
    render_plan,
)
from manysql.spec.semantics import SemanticConfig


class SQLiteOracle(Oracle):
    """Render IR -> SQLite SQL, load tables, execute. Less coverage than DuckDB."""

    @property
    def capability(self) -> OracleCapability:
        return OracleCapability(
            name="sqlite",
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
                    "set_ops_distinct_only",
                    "ctes",
                    "recursive_cte",
                }
            ),
            unsupported_knobs=frozenset(
                {
                    "boolean_truthiness",
                    "count_distinct_null",
                    "quoted_identifiers_case_sensitive",
                    "identifier_case_fold",
                    # SQLite has eccentric type affinity; we conservatively skip
                    # cases where the knob is non-default.
                }
            ),
            confidence=0.7,
        )

    def evaluate(
        self,
        plan: Plan,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
    ) -> OracleResult:
        try:
            sql, notes = render_plan(plan, semantics, SqlDialectFlags.sqlite())
        except UnsupportedByEngine as exc:
            return OracleResult(oracle=self.capability.name, error=f"unsupported: {exc}")
        try:
            con = sqlite3.connect(":memory:")
            con.execute("PRAGMA foreign_keys = ON")
            # SQLite's default LIKE is ASCII-case-insensitive. Force the more
            # widely shared case-sensitive behavior so we agree with DuckDB and
            # the reference interpreter on `LIKE` semantics.
            con.execute("PRAGMA case_sensitive_like = ON")
            self._load_catalog(con, catalog)
            cur = con.execute(sql)
            cols = [d[0] for d in cur.description]
            rows: list[dict[str, Any]] = [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]
            df = self._coerce(rows, cols, plan, catalog)
            return OracleResult(oracle=self.capability.name, rows=df, notes=notes)
        except Exception as exc:  # noqa: BLE001
            return OracleResult(
                oracle=self.capability.name,
                error=f"{type(exc).__name__}: {exc}\nSQL was: {sql}",
            )

    def _load_catalog(self, con: sqlite3.Connection, catalog: dict[str, pl.DataFrame]) -> None:
        """Create one table per catalog entry and insert all rows."""
        for name, df in catalog.items():
            cols_sql = ", ".join(f'"{c}" {self._sqlite_type(df.schema[c])}' for c in df.columns)
            con.execute(f'CREATE TABLE "{name}" ({cols_sql})')
            placeholders = ", ".join(["?"] * len(df.columns))
            insert = f'INSERT INTO "{name}" VALUES ({placeholders})'
            for row in df.iter_rows():
                con.executemany(insert, [tuple(self._py_for_sqlite(v) for v in row)])

    @staticmethod
    def _sqlite_type(t: pl.DataType) -> str:
        if t == pl.Int64 or t == pl.Int32:
            return "INTEGER"
        if t == pl.Float64 or t == pl.Float32:
            return "REAL"
        if t == pl.Utf8:
            return "TEXT"
        if t == pl.Boolean:
            return "INTEGER"  # SQLite has no bool; we coerce on read.
        if t == pl.Date or t == pl.Datetime:
            return "TEXT"  # ISO date string.
        return "TEXT"

    @staticmethod
    def _py_for_sqlite(v: Any) -> Any:
        if isinstance(v, bool):
            return 1 if v else 0
        if isinstance(v, (date, datetime)):
            return v.isoformat()
        return v

    def _coerce(
        self,
        rows: list[dict[str, Any]],
        cols: list[str],
        plan: Plan,
        catalog: dict[str, pl.DataFrame],
    ) -> pl.DataFrame:
        """SQLite returns ints for booleans and strings for dates. Best-effort
        coerce by inspecting the catalog schema for the leaf-most column with
        a matching name. Imperfect but acceptable for a secondary oracle.
        """
        if not rows:
            return pl.DataFrame({c: pl.Series(c, [], dtype=pl.Null) for c in cols})
        df = pl.DataFrame(rows, infer_schema_length=len(rows))
        # Try to recover bool/date columns by matching on bare names found in catalog.
        bool_cols: set[str] = set()
        date_cols: set[str] = set()
        for tbl in catalog.values():
            for c, t in tbl.schema.items():
                if t == pl.Boolean:
                    bool_cols.add(c)
                if t == pl.Date or t == pl.Datetime:
                    date_cols.add(c)
        casts: list[pl.Expr] = []
        for c in df.columns:
            base = c.split("__", 1)[1] if "__" in c else c
            if base in bool_cols and df.schema[c] == pl.Int64:
                casts.append(pl.col(c).cast(pl.Boolean))
            elif base in date_cols and df.schema[c] == pl.Utf8:
                casts.append(pl.col(c).str.to_date(strict=False))
        if casts:
            df = df.with_columns(casts)
        return df


__all__ = ["SQLiteOracle"]
