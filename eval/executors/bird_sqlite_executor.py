"""BIRD-SQL execution backend.

Each BIRD question targets a different ``.sqlite`` database, so this
executor's ``execute(sql, question=...)`` opens (or reuses) a
per-``db_path`` read-only stdlib ``sqlite3`` connection rather than
running every query against one global schema.

Threading: the eval runner can fan questions out across a thread pool
(``--concurrency N``), so each thread gets its own connection cache
via ``threading.local()``. SQLite handles cross-thread access with
``check_same_thread=False`` but still serializes statements at the
connection level; per-thread caches sidestep that contention entirely
and also let us treat each connection as single-thread-owned.

Read-only enforcement:
* The connection is opened with the URI flag ``mode=ro``, so SQLite
  itself will refuse writes.
* Defense in depth: the same ``_is_read_only`` heuristic the
  github-events :class:`SqliteExecutor` uses also vetoes the SQL
  before it hits the engine, so an LLM that tries ``ATTACH``,
  ``PRAGMA``, etc. gets a clear error rather than a silent no-op.
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
import time
from typing import TYPE_CHECKING, Any

from eval.executors.base import ExecResult, SqlExecutor
from eval.executors.sqlite_executor import _is_read_only, _row_to_dict

if TYPE_CHECKING:
    from eval.dataset.questions import Question

# Default schema-prompt blurb. The actual per-question schema is
# inlined into ``Question.prompt`` by ``eval.dataset.bird``; the
# system prompt only needs to tell the LLM where to look.
from eval.dataset.bird import _BIRD_EVAL_SCHEMA_PROMPT


class BirdSqliteExecutor(SqlExecutor):
    """Per-question SQLite executor backed by BIRD ``.sqlite`` files.

    Holds a thread-local cache of open connections keyed by absolute
    DB path. ``setup()`` is a no-op (we don't know which DBs we'll
    need until the runner starts dispatching questions);
    ``teardown()`` closes every cached connection across all threads
    so we don't leak file descriptors when the runner exits.
    """

    name = "bird"

    def __init__(
        self,
        *,
        query_timeout_s: float = 30.0,
        schema_prompt: str | None = None,
    ) -> None:
        self.query_timeout_s = query_timeout_s
        self._schema_prompt = schema_prompt or _BIRD_EVAL_SCHEMA_PROMPT
        # Per-thread storage; populated lazily in execute().
        self._tls = threading.local()
        # Cross-thread registry of all opened connections so teardown
        # can close them. Guarded by a lock since add/close runs from
        # multiple threads.
        self._all_conns: list[sqlite3.Connection] = []
        self._conn_lock = threading.Lock()

    def setup(self) -> None:
        # Nothing to do up front. Connections open lazily in
        # _connection_for(db_path).
        return None

    def execute(
        self, sql: str, *, question: Question | None = None
    ) -> ExecResult:
        if question is None or not question.db_path:
            return ExecResult(
                success=False,
                error=(
                    "BirdSqliteExecutor requires a Question with db_path "
                    "set. Did you mix the bird backend with a non-BIRD "
                    "question source?"
                ),
                backend=self.name,
            )

        sql = sql.strip().rstrip(";").strip()
        if not sql:
            return ExecResult(
                success=False, error="empty SQL", backend=self.name
            )
        if not _is_read_only(sql):
            return ExecResult(
                success=False,
                error="only read-only SELECT/WITH queries are allowed",
                backend=self.name,
            )

        try:
            conn = self._connection_for(question.db_path)
        except sqlite3.Error as exc:
            return ExecResult(
                success=False,
                error=f"failed to open BIRD DB {question.db_path}: {exc}",
                backend=self.name,
            )

        start = time.perf_counter()
        try:
            cur = conn.cursor()
            cur.execute(sql)
            fetched = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
            rows: list[dict[str, Any]] = [_row_to_dict(r, cols) for r in fetched]
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
        return self._schema_prompt

    def dialect_label(self) -> str:
        # Match the questions[].reference_sql["sqlite"] key so the
        # runner's dialect-substring lookup picks the BIRD gold SQL.
        return "sqlite"

    def teardown(self) -> None:
        with self._conn_lock:
            for conn in self._all_conns:
                with contextlib.suppress(sqlite3.Error):
                    conn.close()
            self._all_conns.clear()
        # Drop the thread-local cache too; reusing the executor after
        # teardown() should reopen fresh connections.
        self._tls = threading.local()

    # -- internals --

    def _connection_for(self, db_path: str) -> sqlite3.Connection:
        cache: dict[str, sqlite3.Connection] = getattr(
            self._tls, "conns", None
        ) or {}
        conn = cache.get(db_path)
        if conn is not None:
            return conn

        # Open URI-mode read-only so the engine refuses writes even
        # if the SQL slipped past _is_read_only (defense in depth).
        # ``immutable=1`` would also work and skip locking but trips
        # if any process holds the DB open RW; ``mode=ro`` is the
        # safer default.
        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True,
            check_same_thread=False,
            timeout=self.query_timeout_s,
        )
        conn.row_factory = sqlite3.Row
        cache[db_path] = conn
        self._tls.conns = cache
        with self._conn_lock:
            self._all_conns.append(conn)
        return conn


__all__ = ["BirdSqliteExecutor"]
