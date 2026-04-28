"""SQLite execution backend.

In-memory SQLite seeded with a small synthetic GitHub-events corpus shaped
after tinybirdco/llm-benchmark. Default backend for the eval harness because
it requires no external services and runs the same on any laptop.

The schema is intentionally a SQLite-friendly subset of Tinybird's
`github_events.datasource`: ClickHouse-only types (`LowCardinality`,
`Array(...)`, `Enum8/16`, `Nullable(...)`, etc.) collapse to the closest
SQLite affinity, and array columns are stored as comma-separated TEXT
(LLMs are told this in the schema prompt).
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from eval.dataset.github_events import (
    SCHEMA_DDL,
    SCHEMA_PROMPT,
    seed_rows,
)
from eval.executors.base import ExecResult, SqlExecutor

if TYPE_CHECKING:
    from eval.dataset.questions import Question


class SqliteExecutor(SqlExecutor):
    """In-memory SQLite executor seeded with synthetic GitHub events."""

    name = "sqlite"

    def __init__(
        self,
        *,
        db_path: str | Path = ":memory:",
        seed: int = 0xDB,
        n_rows: int = 5_000,
        query_timeout_s: float = 10.0,
    ) -> None:
        self.db_path = str(db_path)
        self.seed = seed
        self.n_rows = n_rows
        self.query_timeout_s = query_timeout_s
        self._conn: sqlite3.Connection | None = None

    def setup(self) -> None:
        # `check_same_thread=False` so the runner can be reused from async
        # callers later; we still serialize all execution.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Read-only-ish: we want the LLM's SELECTs to succeed, INSERTs/etc
        # to fail. SQLite has no per-connection RO mode for :memory:, so we
        # rely on the prompt + post-query validation.
        cur = self._conn.cursor()
        cur.executescript(SCHEMA_DDL)
        rows = seed_rows(seed=self.seed, n=self.n_rows)
        if rows:
            cols = list(rows[0].keys())
            placeholders = ",".join(["?"] * len(cols))
            insert_sql = f"INSERT INTO github_events ({', '.join(cols)}) VALUES ({placeholders})"
            cur.executemany(insert_sql, [tuple(r[c] for c in cols) for r in rows])
        self._conn.commit()

    def execute(self, sql: str, *, question: Question | None = None) -> ExecResult:
        del question  # global schema; per-question pointers are ignored.
        if self._conn is None:
            raise RuntimeError("SqliteExecutor.setup() not called")

        sql = sql.strip().rstrip(";").strip()
        if not sql:
            return ExecResult(
                success=False, error="empty SQL", backend=self.name
            )

        # Defense-in-depth: refuse anything that isn't a single SELECT/WITH.
        # Generated SQL is untrusted; SQLite has no DDL/DML sandbox.
        if not _is_read_only(sql):
            return ExecResult(
                success=False,
                error="only read-only SELECT/WITH queries are allowed",
                backend=self.name,
            )

        start = time.perf_counter()
        try:
            cur = self._conn.cursor()
            cur.execute(sql)
            fetched = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = [_row_to_dict(r, cols) for r in fetched]
        except sqlite3.Error as exc:
            return ExecResult(
                success=False,
                error=str(exc),
                execution_time_s=time.perf_counter() - start,
                backend=self.name,
            )

        return ExecResult(
            success=True,
            rows=rows,
            columns=cols,
            execution_time_s=time.perf_counter() - start,
            backend=self.name,
        )

    def schema_prompt(self) -> str:
        return SCHEMA_PROMPT

    def dialect_label(self) -> str:
        return "sqlite"

    def teardown(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def _row_to_dict(row: sqlite3.Row, cols: list[str]) -> dict[str, Any]:
    return {c: row[c] for c in cols}


_READ_ONLY_PREFIXES = ("select", "with")
_FORBIDDEN = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "attach",
    "detach",
    "vacuum",
    "pragma",
    "replace",
)


def _is_read_only(sql: str) -> bool:
    lowered = sql.lower().lstrip("(").lstrip()
    if not lowered.startswith(_READ_ONLY_PREFIXES):
        return False
    # Reject statements that *contain* a write keyword as a top-level token.
    # This is a heuristic; full SQL parsing is overkill here. We only flag
    # whitespace-bounded matches to avoid false positives like "selected_at".
    for word in _FORBIDDEN:
        needle = f" {word} "
        if needle in f" {lowered} ":
            return False
    return True


__all__ = ["SqliteExecutor"]
