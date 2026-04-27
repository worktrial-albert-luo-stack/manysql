"""Query-latency bench: real PostgreSQL vs manysql `postgres_clone`.

This script is the perf counterpart to `eval/__main__.py`. The standard
eval harness scores LLM-generated SQL for *correctness*; this harness
holds the SQL fixed and asks: how does our generated dialect engine's
runtime compare to the real database it clones?

Setup
-----
Both backends are seeded from the same deterministic synthetic
``github_events`` corpus (`eval/dataset/github_events.py`), so any
latency delta is dominated by query execution rather than I/O or
caching effects. Real Postgres runs the SQL through psycopg; the
manysql side parses + lowers via the `postgres_clone` dialect package
and executes against an in-memory Polars catalog.

Usage
-----
::

    # one-off run against a local Postgres
    uv sync --extra bench
    export BENCH_POSTGRES_URL=postgresql://localhost:5432/manysql_bench
    uv run manysql-perf-bench --rows 50000 --repeats 7

    # run a subset of queries
    uv run manysql-perf-bench --queries q01,q05_window --repeats 5

    # write the per-query timing JSON to a custom path
    uv run manysql-perf-bench --output results/perf_postgres_clone.json

The script picks up the connection URL from ``--postgres-url`` /
``BENCH_POSTGRES_URL`` / ``DATABASE_URL`` (in that order). Without one
it prints a copy-pasteable docker-run hint and exits cleanly so dev
boxes without Postgres still get a useful error.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from eval.dataset.github_events import seed_rows

if TYPE_CHECKING:  # pragma: no cover - typing only
    import polars as pl
    from lark import Lark

    from manysql.dialects.registry import DialectEngine
    from manysql.ir.plan import ColumnSchema


# ---------------------------------------------------------------------------
# Curated query suite
# ---------------------------------------------------------------------------
#
# Every query MUST run cleanly on:
#   - real PostgreSQL (>=14), and
#   - the manysql `postgres_clone` dialect engine (parser + executor).
#
# That rules out SQLite-only spellings like ``strftime('%Y', ...)`` and
# Postgres-only-but-not-yet-lowered constructs like the ``EXTRACT(YEAR
# FROM x)`` keyword form. We use:
#
#   - ``SUBSTRING(text, n, m)`` for year/month extraction from the ISO 8601
#     timestamp text (both engines support it; on Postgres SUBSTRING is
#     SQL-standard and reads the literal positional args).
#   - ``DATE_PART('year', CAST(x AS TIMESTAMP))`` for date-part queries
#     (Postgres native; manysql expr_eval handles DATE_PART via
#     ``Polars.dt.year()``).
#   - Plain SELECT / WHERE / GROUP BY / ORDER BY / LIMIT / DISTINCT /
#     UNION / CASE / CAST / arithmetic / window functions / CTEs /
#     IN-subqueries — every one mapped 1:1 in both engines.


@dataclass(frozen=True)
class BenchQuery:
    name: str
    description: str
    sql: str
    category: str = "core"


_BENCH_QUERIES: list[BenchQuery] = [
    BenchQuery(
        "q01_count_stars",
        "Count of WatchEvent rows.",
        "SELECT count(*) AS stars FROM github_events WHERE event_type = 'WatchEvent'",
    ),
    BenchQuery(
        "q02_top_starred_repos",
        "Top 10 repositories by star count.",
        "SELECT repo_name, count(*) AS stars FROM github_events "
        "WHERE event_type = 'WatchEvent' GROUP BY repo_name "
        "ORDER BY stars DESC, repo_name ASC LIMIT 10",
    ),
    BenchQuery(
        "q03_count_distinct_repos",
        "Distinct repositories in the dataset.",
        "SELECT count(DISTINCT repo_name) AS repos FROM github_events",
    ),
    BenchQuery(
        "q04_stars_by_year",
        "Stars per year via SUBSTRING on ISO timestamp text.",
        "SELECT SUBSTRING(created_at, 1, 4) AS year, count(*) AS stars "
        "FROM github_events WHERE event_type = 'WatchEvent' "
        "GROUP BY year ORDER BY year ASC",
    ),
    BenchQuery(
        "q05_top_pushers_having",
        "Top pushers (>=3 pushes) by push count.",
        "SELECT actor_login, count(*) AS pushes FROM github_events "
        "WHERE event_type = 'PushEvent' GROUP BY actor_login "
        "HAVING count(*) >= 3 ORDER BY pushes DESC, actor_login ASC LIMIT 20",
    ),
    BenchQuery(
        "q06_filter_like",
        "Repos whose name contains common ML org slugs.",
        "SELECT count(*) AS hits FROM github_events "
        "WHERE repo_name LIKE '%pytorch%' OR repo_name LIKE '%tensorflow%' "
        "OR repo_name LIKE '%huggingface%'",
    ),
    BenchQuery(
        "q07_case_classification",
        "PushEvent rows bucketed by additions size.",
        "SELECT CASE WHEN additions > 100 THEN 'large' "
        "WHEN additions > 0 THEN 'small' ELSE 'none' END AS bucket, "
        "count(*) AS n FROM github_events WHERE event_type = 'PushEvent' "
        "GROUP BY bucket ORDER BY n DESC, bucket ASC",
    ),
    BenchQuery(
        "q08_in_subquery",
        "Top 10 non-tensorflow repos with PRs from tensorflow-PR authors.",
        "SELECT repo_name, count(*) AS prs FROM github_events "
        "WHERE event_type = 'PullRequestEvent' AND action = 'opened' "
        "AND actor_login IN ("
        "SELECT actor_login FROM github_events "
        "WHERE event_type = 'PullRequestEvent' AND action = 'opened' "
        "AND repo_name = 'tensorflow/tensorflow') "
        "AND repo_name <> 'tensorflow/tensorflow' "
        "GROUP BY repo_name ORDER BY prs DESC, repo_name ASC LIMIT 10",
    ),
    BenchQuery(
        "q09_window_row_number",
        "First (alphabetic) starrer per repository via ROW_NUMBER.",
        "SELECT repo_name, actor_login FROM ("
        "SELECT repo_name, actor_login, "
        "ROW_NUMBER() OVER (PARTITION BY repo_name ORDER BY actor_login) AS rn "
        "FROM github_events WHERE event_type = 'WatchEvent') AS t "
        "WHERE rn = 1 ORDER BY repo_name ASC LIMIT 10",
    ),
    BenchQuery(
        "q10_cte_aggregate",
        "Repos with >=5 stars (CTE + count aggregate).",
        "WITH per_repo AS ("
        "SELECT repo_name, count(*) AS c FROM github_events "
        "WHERE event_type = 'WatchEvent' GROUP BY repo_name) "
        "SELECT count(*) AS active_repos FROM per_repo WHERE c >= 5",
    ),
    BenchQuery(
        "q11_union_distinct",
        "Pushers vs starrers (UNION distinct).",
        "SELECT actor_login FROM github_events WHERE event_type = 'PushEvent' "
        "UNION "
        "SELECT actor_login FROM github_events WHERE event_type = 'WatchEvent'",
    ),
    BenchQuery(
        "q12_sum_arith",
        "Net additions-deletions per repo (sum + arithmetic).",
        "SELECT repo_name, sum(additions) AS adds, sum(deletions) AS dels, "
        "sum(additions) - sum(deletions) AS net "
        "FROM github_events WHERE event_type = 'PushEvent' "
        "GROUP BY repo_name ORDER BY net DESC, repo_name ASC LIMIT 10",
    ),
    BenchQuery(
        "q13_month_buckets",
        "Issue-open volume per YYYY-MM bucket.",
        "SELECT SUBSTRING(created_at, 1, 7) AS month, count(*) AS issues "
        "FROM github_events WHERE event_type = 'IssuesEvent' AND action = 'opened' "
        "GROUP BY month ORDER BY month ASC",
    ),
    BenchQuery(
        "q14_distinct_pairs",
        "Distinct (event_type, action) combinations.",
        "SELECT DISTINCT event_type, action FROM github_events ORDER BY event_type ASC, action ASC",
    ),
    BenchQuery(
        "q15_distinct_repos_per_actor",
        "Top 20 actors by distinct repos touched.",
        "SELECT actor_login, count(DISTINCT repo_name) AS distinct_repos "
        "FROM github_events GROUP BY actor_login "
        "ORDER BY distinct_repos DESC, actor_login ASC LIMIT 20",
    ),
]
# NOTE: a ``DATE_PART('year', CAST(created_at AS TIMESTAMP))`` query was
# considered for the suite but dropped: manysql's Cast(TEXT -> TIMESTAMP)
# currently delegates to Polars's strict=False cast, which returns NULL
# for ISO 8601 strings (Polars expects ``str.to_datetime(format=...)``).
# Postgres handles the same cast natively, so the two engines disagree
# on output rows and the perf comparison stops being apples-to-apples.
# Once that codepath grows a string-parsing fallback, add the query back.


# ---------------------------------------------------------------------------
# Postgres backend
# ---------------------------------------------------------------------------
#
# The DDL mirrors `eval/dataset/github_events.SCHEMA_DDL` but uses
# Postgres-native types (TEXT/BIGINT, no SQLite-style affinity quirks)
# and DROP TABLE IF EXISTS for idempotent reseeding. We DELIBERATELY
# leave timestamps as TEXT so the surface SQL is identical to what the
# manysql synthetic side runs against (a Polars Utf8 column).


_PG_SCHEMA_DDL = """
DROP TABLE IF EXISTS github_events;
CREATE TABLE github_events (
    file_time           TEXT NOT NULL,
    event_type          TEXT NOT NULL,
    actor_login         TEXT NOT NULL DEFAULT '',
    repo_name           TEXT NOT NULL DEFAULT '',
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    action              TEXT NOT NULL,
    comment_id          BIGINT NOT NULL DEFAULT 0,
    commit_id           TEXT NOT NULL DEFAULT '',
    body                TEXT,
    ref                 TEXT NOT NULL DEFAULT '',
    number              BIGINT NOT NULL DEFAULT 0,
    title               TEXT,
    labels              TEXT NOT NULL DEFAULT '',
    state               TEXT NOT NULL DEFAULT '',
    locked              BIGINT NOT NULL DEFAULT 0,
    assignee            TEXT NOT NULL DEFAULT '',
    comments            BIGINT NOT NULL DEFAULT 0,
    author_association  TEXT NOT NULL DEFAULT 'NONE',
    closed_at           TEXT NOT NULL DEFAULT '',
    merged_at           TEXT NOT NULL DEFAULT '',
    merged              BIGINT NOT NULL DEFAULT 0,
    commits             BIGINT NOT NULL DEFAULT 0,
    additions           BIGINT NOT NULL DEFAULT 0,
    deletions           BIGINT NOT NULL DEFAULT 0,
    changed_files       BIGINT NOT NULL DEFAULT 0,
    push_size           BIGINT NOT NULL DEFAULT 0,
    release_tag_name    TEXT NOT NULL DEFAULT '',
    release_name        TEXT NOT NULL DEFAULT '',
    review_state        TEXT NOT NULL DEFAULT ''
);
CREATE INDEX idx_pg_event_type   ON github_events(event_type);
CREATE INDEX idx_pg_repo_name    ON github_events(repo_name);
CREATE INDEX idx_pg_actor_login  ON github_events(actor_login);
CREATE INDEX idx_pg_created_at   ON github_events(created_at);
"""


@dataclass
class PostgresBench:
    """Thin Postgres driver wrapper used only by the perf bench.

    Not registered as a public ``eval/executors`` backend on purpose —
    the eval harness's question suite is SQLite-flavored, and a Postgres
    backend would need its own per-question reference SQL. For *perf*
    we only run a curated set of dialect-neutral queries, so a small
    self-contained class is the right granularity.
    """

    dsn: str
    n_rows: int = 5_000
    seed: int = 0xDB

    _conn: Any = None

    def setup(self) -> None:
        try:
            import psycopg  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - import-time message
            raise SystemExit(
                "manysql-perf-bench needs psycopg. Install with "
                "'uv sync --extra bench' (or 'pip install psycopg[binary]')."
            ) from exc
        self._conn = psycopg.connect(self.dsn, autocommit=False)
        rows = seed_rows(seed=self.seed, n=self.n_rows)
        with self._conn.cursor() as cur:
            cur.execute(_PG_SCHEMA_DDL)
            if rows:
                cols = list(rows[0].keys())
                placeholders = ",".join(["%s"] * len(cols))
                insert_sql = (
                    f"INSERT INTO github_events ({', '.join(cols)}) VALUES ({placeholders})"
                )
                cur.executemany(insert_sql, [tuple(r[c] for c in cols) for r in rows])
            cur.execute("ANALYZE github_events")
        self._conn.commit()

    def execute(self, sql: str) -> tuple[float, list[tuple[Any, ...]]]:
        """Run ``sql`` once and return ``(elapsed_s, rows)``.

        ``rows`` is a list of plain tuples so we can compare it with the
        Polars output from the manysql side without hitting psycopg's
        Row/Column abstractions (which differ subtly between dict-row
        and tuple-row factories).
        """
        if self._conn is None:
            raise RuntimeError("PostgresBench.setup() not called")
        with self._conn.cursor() as cur:
            start = time.perf_counter()
            cur.execute(sql)
            rows = cur.fetchall()
            elapsed = time.perf_counter() - start
        # Roll back any implicit transaction state so the next query starts
        # clean (we're read-only here, but psycopg keeps a tx open).
        self._conn.rollback()
        return elapsed, [tuple(r) for r in rows]

    def teardown(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Manysql synthetic backend
# ---------------------------------------------------------------------------


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


@dataclass
class ManysqlBench:
    """Drives the manysql `postgres_clone` engine on the same seeded data.

    Mirrors the lifecycle of the standard ``SyntheticExecutor`` (parse +
    lower + execute) but skips the ExecResult wrapper so we can return
    raw Polars rows for cross-engine equivalence checking.
    """

    dialect: str = "postgres_clone"
    n_rows: int = 5_000
    seed: int = 0xDB

    _engine: DialectEngine | None = None
    _parser: Lark | None = None
    _catalog: dict[str, pl.DataFrame] = field(default_factory=dict)
    _schemas: dict[str, tuple[ColumnSchema, ...]] = field(default_factory=dict)

    def setup(self) -> None:
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

    def execute(self, sql: str) -> tuple[float, list[tuple[Any, ...]]]:
        from manysql.executor import execute as plan_execute  # noqa: PLC0415

        if self._engine is None or self._parser is None:
            raise RuntimeError("ManysqlBench.setup() not called")
        sql = sql.strip().rstrip(";").strip()
        start = time.perf_counter()
        tree = self._parser.parse(sql)
        plan = self._engine.lowering.lower(tree, self._engine.semantics, self._schemas)
        df = plan_execute(
            plan,
            self._engine.semantics,
            self._catalog,
            self._engine.overrides,
            passes=self._engine.passes,
            effects=self._engine.effects,
        )
        elapsed = time.perf_counter() - start
        return elapsed, [tuple(r) for r in df.iter_rows()]

    def teardown(self) -> None:
        self._engine = None
        self._parser = None
        self._catalog = {}
        self._schemas = {}


# ---------------------------------------------------------------------------
# Result equivalence
# ---------------------------------------------------------------------------


def _normalize_rows(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """Coerce Decimals/floats to a comparable form, drop ordering noise."""
    norm: list[tuple[Any, ...]] = []
    for r in rows:
        norm.append(tuple(_normalize_cell(v) for v in r))
    # Sort to ignore any residual ordering differences for equivalence
    # (queries with explicit ORDER BY are already deterministic; this just
    # makes implicit-order queries comparable too).
    return sorted(norm, key=lambda t: tuple(repr(v) for v in t))


def _normalize_cell(v: Any) -> Any:
    if v is None:
        return None
    # Postgres returns Decimal for numeric; Polars returns Python int/float.
    # Compare as float when the value is numeric-shaped.
    try:
        from decimal import Decimal  # noqa: PLC0415

        if isinstance(v, Decimal):
            return float(v)
    except ImportError:  # pragma: no cover
        pass
    if isinstance(v, float):
        # Round to 6 decimals to absorb the IEEE-754 difference between the
        # two engines (Postgres uses arbitrary-precision NUMERIC for division,
        # Polars uses Float64).
        return round(v, 6)
    if isinstance(v, bool):
        return int(v)
    return v


@dataclass
class QueryTimings:
    name: str
    description: str
    sql: str
    pg_times_s: list[float] = field(default_factory=list)
    manysql_times_s: list[float] = field(default_factory=list)
    pg_rows: int = 0
    manysql_rows: int = 0
    rows_match: bool | None = None
    error: str | None = None

    @property
    def pg_median_s(self) -> float | None:
        return statistics.median(self.pg_times_s) if self.pg_times_s else None

    @property
    def manysql_median_s(self) -> float | None:
        return statistics.median(self.manysql_times_s) if self.manysql_times_s else None

    @property
    def speedup_pg_over_manysql(self) -> float | None:
        """How many times faster Postgres is than manysql (>1 = PG wins)."""
        if not self.pg_median_s or not self.manysql_median_s:
            return None
        if self.pg_median_s == 0:
            return None
        return self.manysql_median_s / self.pg_median_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "sql": self.sql,
            "pg_median_s": self.pg_median_s,
            "manysql_median_s": self.manysql_median_s,
            "pg_min_s": min(self.pg_times_s) if self.pg_times_s else None,
            "manysql_min_s": min(self.manysql_times_s) if self.manysql_times_s else None,
            "pg_times_s": self.pg_times_s,
            "manysql_times_s": self.manysql_times_s,
            "pg_rows": self.pg_rows,
            "manysql_rows": self.manysql_rows,
            "rows_match": self.rows_match,
            "speedup_pg_over_manysql": self.speedup_pg_over_manysql,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Bench loop
# ---------------------------------------------------------------------------


def _select_queries(names: list[str] | None, limit: int | None) -> list[BenchQuery]:
    qs = list(_BENCH_QUERIES)
    if names:
        wanted = set(names)
        # Allow prefix match (so `--queries q01` picks `q01_count_stars`).
        picked = [q for q in qs if q.name in wanted or any(q.name.startswith(w) for w in wanted)]
        if not picked:
            raise SystemExit(f"no bench queries matched {names!r}. Known: {[q.name for q in qs]}")
        qs = picked
    if limit is not None:
        qs = qs[:limit]
    return qs


def _resolve_dsn(args: argparse.Namespace) -> str | None:
    return args.postgres_url or os.getenv("BENCH_POSTGRES_URL") or os.getenv("DATABASE_URL") or None


def _print_dsn_hint(console: Console) -> None:
    console.print(
        "[red]error[/red]: no Postgres connection URL found.\n"
        "Set one via [cyan]--postgres-url[/cyan], "
        "[cyan]BENCH_POSTGRES_URL[/cyan], or [cyan]DATABASE_URL[/cyan].\n\n"
        "Quickstart with Docker:\n"
        "  [cyan]docker run -d --name manysql-pg -p 5432:5432 "
        "-e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=manysql_bench "
        "postgres:16[/cyan]\n"
        "  [cyan]export BENCH_POSTGRES_URL="
        "postgresql://postgres:postgres@localhost:5432/manysql_bench[/cyan]"
    )


def run_bench(
    *,
    dsn: str,
    dialect: str,
    queries: list[BenchQuery],
    n_rows: int,
    seed: int,
    repeats: int,
    warmup: int,
    console: Console,
    quiet: bool = False,
) -> list[QueryTimings]:
    pg = PostgresBench(dsn=dsn, n_rows=n_rows, seed=seed)
    ms = ManysqlBench(dialect=dialect, n_rows=n_rows, seed=seed)

    if not quiet:
        console.print(
            f"[bold]Setting up backends[/bold] "
            f"(rows={n_rows}, seed=0x{seed:X}, dialect={dialect})..."
        )
    pg_setup_start = time.perf_counter()
    pg.setup()
    pg_setup_elapsed = time.perf_counter() - pg_setup_start

    ms_setup_start = time.perf_counter()
    ms.setup()
    ms_setup_elapsed = time.perf_counter() - ms_setup_start

    if not quiet:
        console.print(f"  [green]postgres ready[/green]  (connect+seed={pg_setup_elapsed:.2f}s)")
        console.print(f"  [green]manysql ready[/green]   (load+frame={ms_setup_elapsed:.2f}s)")

    results: list[QueryTimings] = []
    try:
        for q in queries:
            timing = QueryTimings(name=q.name, description=q.description, sql=q.sql)
            try:
                # Warmup: don't record timings, but populate any cold caches
                # (psycopg statement cache, polars JIT, etc.).
                for _ in range(warmup):
                    pg.execute(q.sql)
                    ms.execute(q.sql)
                # Capture *one* row payload from each engine for equivalence
                # checking; subsequent timed runs discard rows to avoid
                # paying the materialization cost on every iteration.
                pg_warmup_elapsed, pg_payload = pg.execute(q.sql)
                ms_warmup_elapsed, ms_payload = ms.execute(q.sql)
                _ = pg_warmup_elapsed, ms_warmup_elapsed  # already accounted for in warmup
                timing.pg_rows = len(pg_payload)
                timing.manysql_rows = len(ms_payload)
                timing.rows_match = _normalize_rows(pg_payload) == _normalize_rows(ms_payload)

                for _ in range(repeats):
                    pg_t, _ = pg.execute(q.sql)
                    timing.pg_times_s.append(pg_t)
                for _ in range(repeats):
                    ms_t, _ = ms.execute(q.sql)
                    timing.manysql_times_s.append(ms_t)
            except Exception as exc:
                timing.error = f"{type(exc).__name__}: {exc}"
            results.append(timing)
            if not quiet:
                _print_progress_row(console, timing)
    finally:
        pg.teardown()
        ms.teardown()
    return results


def _print_progress_row(console: Console, timing: QueryTimings) -> None:
    if timing.error:
        console.print(f"  [red]err[/red]  {timing.name}: {timing.error}")
        return
    pg_ms = (timing.pg_median_s or 0.0) * 1000
    ms_ms = (timing.manysql_median_s or 0.0) * 1000
    badge = "[green]match[/green]" if timing.rows_match else "[yellow]diff[/yellow]"
    console.print(
        f"  [dim]ok [/dim] {timing.name:<28}  "
        f"pg={pg_ms:7.2f}ms  manysql={ms_ms:8.2f}ms  "
        f"rows={timing.pg_rows}/{timing.manysql_rows} {badge}"
    )


def _print_summary(console: Console, results: list[QueryTimings]) -> None:
    table = Table(title="manysql perf bench: postgres vs postgres_clone")
    table.add_column("query", overflow="fold")
    table.add_column("pg p50 (ms)", justify="right")
    table.add_column("manysql p50 (ms)", justify="right")
    table.add_column("manysql/pg", justify="right")
    table.add_column("rows pg/manysql", justify="right")
    table.add_column("equiv")

    for t in results:
        if t.error:
            table.add_row(t.name, "-", "-", "-", "-", f"[red]{t.error[:40]}[/red]")
            continue
        pg_ms = f"{(t.pg_median_s or 0.0) * 1000:.2f}"
        ms_ms = f"{(t.manysql_median_s or 0.0) * 1000:.2f}"
        ratio = f"{t.speedup_pg_over_manysql:.1f}x" if t.speedup_pg_over_manysql else "-"
        rows = f"{t.pg_rows}/{t.manysql_rows}"
        equiv = (
            "[green]yes[/green]"
            if t.rows_match
            else "[yellow]no[/yellow]"
            if t.rows_match is False
            else "-"
        )
        table.add_row(t.name, pg_ms, ms_ms, ratio, rows, equiv)
    console.print()
    console.print(table)

    valid = [t for t in results if t.pg_median_s and t.manysql_median_s]
    if valid:
        ratios = [t.speedup_pg_over_manysql for t in valid if t.speedup_pg_over_manysql]
        if ratios:
            geomean = statistics.geometric_mean(ratios)
            equiv_count = sum(1 for t in valid if t.rows_match)
            console.print(
                f"\n[bold]aggregate[/bold]  geomean(manysql/pg) = "
                f"[cyan]{geomean:.2f}x[/cyan]  "
                f"({equiv_count}/{len(valid)} produce identical rows)"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="manysql-perf-bench",
        description=(
            "Compare query latency between real PostgreSQL and the manysql "
            "`postgres_clone` dialect engine on a shared synthetic dataset. "
            "Useful as a sanity check on the codegen pipeline (cross-engine "
            "row equivalence) and as an order-of-magnitude reference for "
            "the dialect-engine cost relative to a production database."
        ),
    )
    p.add_argument(
        "--postgres-url",
        default=None,
        help="Postgres DSN, e.g. postgresql://user:pass@host:5432/db. "
        "Falls back to BENCH_POSTGRES_URL or DATABASE_URL.",
    )
    p.add_argument(
        "--dialect",
        default="postgres_clone",
        help="manysql dialect name to load (default: postgres_clone). "
        "Generate it first with `manysql-codegen gen postgres_clone`.",
    )
    p.add_argument(
        "--rows",
        type=int,
        default=5_000,
        help="Synthetic event rows to seed (default: 5000). The same "
        "rows are loaded into both engines so query plans see the same "
        "shape.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0xDB,
        help="Deterministic seed for the synthetic event generator (default: 0xDB).",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Timed runs per query per backend (default: 5). Median is "
        "reported; raw samples are written to the JSON output.",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Untimed warmup runs per query per backend before the timed "
        "runs (default: 2). Helps absorb statement-cache and JIT effects.",
    )
    p.add_argument(
        "--queries",
        default=None,
        help="Comma-separated list of bench query names (or prefixes). Default: all.",
    )
    p.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Run at most this many queries (applied after --queries).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Path to write per-query JSON timings. Default: "
        "results/perf_<dialect>.json. Pass empty string to skip writing.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print the bench query suite and exit.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-query progress lines (the final table still prints).",
    )
    return p


def _print_query_list(console: Console) -> None:
    table = Table(title="bench query suite")
    table.add_column("name")
    table.add_column("description", overflow="fold")
    for q in _BENCH_QUERIES:
        table.add_row(q.name, q.description)
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _build_parser().parse_args(argv)
    console = Console()

    if args.list:
        _print_query_list(console)
        return 0

    dsn = _resolve_dsn(args)
    if not dsn:
        _print_dsn_hint(console)
        return 2

    names = [n.strip() for n in args.queries.split(",") if n.strip()] if args.queries else None
    queries = _select_queries(names, args.limit)
    if not queries:
        console.print("[yellow]no queries selected[/yellow]")
        return 1

    if not args.quiet:
        console.print(
            f"[bold]manysql perf bench[/bold]  "
            f"queries={len(queries)}  rows={args.rows}  repeats={args.repeats}  "
            f"warmup={args.warmup}  dialect={args.dialect}"
        )

    results = run_bench(
        dsn=dsn,
        dialect=args.dialect,
        queries=queries,
        n_rows=args.rows,
        seed=args.seed,
        repeats=args.repeats,
        warmup=args.warmup,
        console=console,
        quiet=args.quiet,
    )

    _print_summary(console, results)

    output: Path | None
    if args.output == "":
        output = None
    elif args.output is None:
        output = Path("results") / f"perf_{args.dialect}.json"
    else:
        output = Path(args.output)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dialect": args.dialect,
            "rows": args.rows,
            "seed": args.seed,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "queries": [t.to_dict() for t in results],
        }
        output.write_text(json.dumps(payload, indent=2, default=str))
        if not args.quiet:
            console.print(f"[green]wrote[/green] {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
