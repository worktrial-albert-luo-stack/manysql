"""Synthetic-dialect executor.

Loads a manysql-generated dialect from `manysql.dialects.<name>/` via the
`DialectRegistry`, builds a Lark parser from the dialect's grammar, lowers
parsed trees through the dialect's `lowering.lower` to manysql IR, and
executes the IR against an in-memory Polars catalog seeded with the same
synthetic GitHub-events corpus the SQLite backend uses.

Important caveat: the question suite's reference SQL is written in
SQLite syntax. When you eval against a dialect whose surface diverges
from SQLite (e.g. anything beyond the near-ANSI "mild" tier), pair the
synthetic backend with a separate reference executor (`SqliteExecutor`)
so ground truth is computed via SQLite while the LLM's SQL is judged
through the dialect engine. The runner accepts a `reference_executor`
argument exactly for this.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from eval.dataset.github_events import SCHEMA_PROMPT, seed_rows
from eval.executors.base import ExecResult, SqlExecutor
from manysql.dialects.card import render_dialect_card

if TYPE_CHECKING:
    import polars as pl
    from lark import Lark

    from eval.dataset.questions import Question
    from manysql.dialects.registry import DialectEngine
    from manysql.ir.plan import ColumnSchema


# Single-table catalog mirroring the SQLite DDL. Polars dtypes are chosen
# so the IR types we emit (TEXT/INT) match the executor's expectations.
_GITHUB_EVENTS_POLARS_DTYPES: dict[str, str] = {
    "file_time": "Utf8",
    "event_type": "Utf8",
    "actor_login": "Utf8",
    "repo_name": "Utf8",
    "created_at": "Utf8",
    "updated_at": "Utf8",
    "action": "Utf8",
    "comment_id": "Int64",
    "commit_id": "Utf8",
    "body": "Utf8",
    "ref": "Utf8",
    "number": "Int64",
    "title": "Utf8",
    "labels": "Utf8",
    "state": "Utf8",
    "locked": "Int64",
    "assignee": "Utf8",
    "comments": "Int64",
    "author_association": "Utf8",
    "closed_at": "Utf8",
    "merged_at": "Utf8",
    "merged": "Int64",
    "commits": "Int64",
    "additions": "Int64",
    "deletions": "Int64",
    "changed_files": "Int64",
    "push_size": "Int64",
    "release_tag_name": "Utf8",
    "release_name": "Utf8",
    "review_state": "Utf8",
}


class SyntheticExecutor(SqlExecutor):
    """Executes SQL through a manysql-generated dialect engine."""

    name = "synthetic"

    def __init__(
        self,
        *,
        dialect: str = "_reference",
        seed: int = 0xDB,
        n_rows: int = 5_000,
    ) -> None:
        self.dialect = dialect
        self.seed = seed
        self.n_rows = n_rows
        self._engine: DialectEngine | None = None
        self._parser: Lark | None = None
        self._catalog: dict[str, pl.DataFrame] = {}
        self._schemas: dict[str, tuple[ColumnSchema, ...]] = {}
        self._dialect_hints: str = ""
        # Filled in by setup() from engine.spec["surface"]. Drives whether
        # we strip-or-add a trailing terminator on every candidate SQL: some
        # dialects require the terminator (their grammar makes it mandatory),
        # most reject it. Treating both with a blanket rstrip(";") used to
        # produce 100% parse failures on the requires_semicolon=True dialects.
        self._requires_semicolon: bool = False
        self._statement_terminator: str = ";"

    def setup(self) -> None:
        # Lazy imports keep `pip install eval` snappy and avoid pulling
        # polars/lark for users who only ever touch the SQLite backend.
        import polars as pl  # noqa: PLC0415
        from lark import Lark, LarkError  # noqa: PLC0415

        from manysql.dialects.registry import DialectRegistry  # noqa: PLC0415
        from manysql.ir.plan import ColumnSchema  # noqa: PLC0415
        from manysql.ir.types import INT, TEXT  # noqa: PLC0415

        engine = DialectRegistry().load(self.dialect)
        try:
            parser = Lark(engine.grammar_text, start="start", parser="earley")
        except LarkError as exc:
            raise RuntimeError(
                f"failed to build parser for dialect {self.dialect!r}: {exc}"
            ) from exc

        polars_schema = {
            col: getattr(pl, dtype) for col, dtype in _GITHUB_EVENTS_POLARS_DTYPES.items()
        }
        rows = seed_rows(seed=self.seed, n=self.n_rows)
        df = pl.DataFrame(rows, schema=polars_schema)

        ir_type_map = {pl.Utf8: TEXT, pl.Int64: INT}
        cols = tuple(
            ColumnSchema(name=name, type=ir_type_map[dtype])
            for name, dtype in polars_schema.items()
        )

        self._engine = engine
        self._parser = parser
        self._catalog = {"github_events": df}
        self._schemas = {"github_events": cols}
        self._dialect_hints = render_dialect_card(engine)

        surface = (engine.spec or {}).get("surface", {})
        self._requires_semicolon = bool(surface.get("requires_semicolon", False))
        self._statement_terminator = surface.get("statement_terminator") or ";"

    def execute(self, sql: str, *, question: Question | None = None) -> ExecResult:
        del question  # global schema; per-question pointers are ignored.
        from manysql.executor import execute as plan_execute  # noqa: PLC0415

        if self._engine is None or self._parser is None:
            raise RuntimeError("SyntheticExecutor.setup() was not called")

        # Normalize the trailing terminator against the dialect's grammar.
        # We always strip whatever the model emitted (zero or more), then
        # re-append exactly one if the dialect requires it. A bare rstrip(";")
        # used to silently corrupt requires_semicolon=True dialects (e.g.
        # snowacle_qualify, hive_teradata_nullsafe_wild) by removing the
        # mandatory token before the parser saw it -> 100% UnexpectedEOF.
        sql = sql.strip()
        term = self._statement_terminator
        while sql.endswith(term):
            sql = sql[: -len(term)].rstrip()
        if self._requires_semicolon:
            sql = sql + term
        if not sql or sql == term:
            return ExecResult(success=False, error="empty SQL", backend=self.name)

        start = time.perf_counter()
        try:
            tree = self._parser.parse(sql)
            plan = self._engine.lowering.lower(
                tree, self._engine.semantics, self._schemas
            )
            df = plan_execute(
                plan,
                self._engine.semantics,
                self._catalog,
                self._engine.overrides,
                passes=self._engine.passes,
                effects=self._engine.effects,
            )
        except Exception as exc:
            return ExecResult(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                execution_time_s=time.perf_counter() - start,
                backend=self.name,
            )
        rows: list[dict[str, Any]] = df.to_dicts()
        return ExecResult(
            success=True,
            rows=rows,
            columns=list(df.columns),
            execution_time_s=time.perf_counter() - start,
            backend=self.name,
        )

    def schema_prompt(self) -> str:
        hints = self._dialect_hints.rstrip()
        if hints:
            return f"{hints}\n\n{SCHEMA_PROMPT}"
        return SCHEMA_PROMPT

    def dialect_label(self) -> str:
        return f"manysql:{self.dialect}"

    def teardown(self) -> None:
        self._engine = None
        self._parser = None
        self._catalog = {}
        self._schemas = {}


__all__ = ["SyntheticExecutor"]
