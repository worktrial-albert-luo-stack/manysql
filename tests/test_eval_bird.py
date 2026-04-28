"""Surface tests for BIRD-SQL eval support.

These tests don't touch HuggingFace or the ~5GB BIRD train zip.
We construct a tiny synthetic ``.sqlite`` on a tmp path, then exercise:

* The schema introspector (``_introspect_db``)
* The per-question prompt renderer (original column names with quoting)
* The :class:`BirdSqliteExecutor` round-trip on a real :class:`Question`
  with ``db_path`` set
* The factory wire-up (``get_executor("bird")``)

The HF-driven entry point (``select_bird``) is exercised indirectly:
its building blocks (DB resolver, introspector, prompt renderer) are
all covered here, and the HF call itself is a thin
``datasets.load_dataset`` wrapper.
"""

from __future__ import annotations

import inspect
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from eval.dataset.bird import (
    _BIRD_EVAL_SCHEMA_PROMPT,
    _BirdEvalEntry,
    _entry_to_question,
    _introspect_db,
    _looks_like_bird_db_dir,
    _quote_ident,
    _render_user_prompt,
)
from eval.dataset.questions import Question
from eval.executors import get_executor
from eval.executors.bird_sqlite_executor import BirdSqliteExecutor
from eval.executors.sqlite_executor import SqliteExecutor
from eval.executors.synthetic_executor import SyntheticExecutor
from eval.executors.tinybird_executor import TinybirdExecutor


def _build_tiny_bird_db(root: Path, db_id: str = "school") -> Path:
    """Create a 2-table BIRD-shaped SQLite at ``root/<db_id>/<db_id>.sqlite``.

    Mirrors the BIRD layout (``<root>/<db_id>/<db_id>.sqlite``) and
    deliberately uses a column name that needs quoting so we can
    confirm the prompt renderer + executor handle it.
    """
    db_dir = root / db_id
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / f"{db_id}.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            'CREATE TABLE schools ('
            'id INTEGER PRIMARY KEY, '
            'name TEXT, '
            '"Free Meal Count (K-12)" INTEGER, '
            'opened_on DATETIME)'
        )
        cur.executemany(
            'INSERT INTO schools (id, name, "Free Meal Count (K-12)", opened_on) '
            "VALUES (?, ?, ?, ?)",
            [
                (1, "Alpha", 120, "2010-09-01"),
                (2, "Beta", 80, "2012-09-01"),
                (3, "Gamma", 0, "2015-09-01"),
            ],
        )
        cur.execute(
            "CREATE TABLE districts ("
            "district_id INTEGER PRIMARY KEY, "
            "district_name TEXT)"
        )
        cur.executemany(
            "INSERT INTO districts VALUES (?, ?)",
            [(10, "North"), (20, "South")],
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def test_quote_ident() -> None:
    assert _quote_ident("name") == "name"
    assert _quote_ident("snake_case_42") == "snake_case_42"
    # Anything with spaces / parens / mixed case gets quoted.
    assert _quote_ident("Free Meal Count (K-12)") == '"Free Meal Count (K-12)"'
    # Embedded double-quotes get doubled.
    assert _quote_ident('weird"col') == '"weird""col"'


def test_introspect_db_walks_tables(tmp_path: Path) -> None:
    db_path = _build_tiny_bird_db(tmp_path)
    tables = _introspect_db(db_path, sample_rows=2)
    by_name = {t.name: t for t in tables}
    assert set(by_name) == {"schools", "districts"}

    schools = by_name["schools"]
    assert schools.columns == [
        "id",
        "name",
        "Free Meal Count (K-12)",
        "opened_on",
    ]
    # Type strings come from the CREATE TABLE declarations,
    # uppercased.
    assert "INTEGER" in schools.types[0]
    assert "DATETIME" in schools.types[3]
    # Sample-row cap is honored.
    assert len(schools.sample_rows) == 2
    assert schools.n_rows == 3


def test_render_user_prompt_quotes_special_columns(tmp_path: Path) -> None:
    db_path = _build_tiny_bird_db(tmp_path)
    tables = _introspect_db(db_path, sample_rows=2)
    entry = _BirdEvalEntry(
        question_id=42,
        db_id="school",
        db_path=db_path,
        question="how many schools opened before 2013?",
        evidence="opened_on is stored as ISO YYYY-MM-DD",
        sql=(
            "SELECT COUNT(*) FROM schools "
            "WHERE opened_on < '2013-01-01'"
        ),
        difficulty="simple",
        tables=tables,
    )
    rendered = _render_user_prompt(entry, max_value_chars=40)

    assert "Question: how many schools opened before 2013?" in rendered
    assert "Evidence: opened_on is stored as ISO YYYY-MM-DD" in rendered
    assert "Database: school  (SQLite)" in rendered
    assert "Difficulty: simple" in rendered
    # Bare-safe identifiers stay bare.
    assert "schools  (3 rows)" in rendered
    assert "id : INTEGER" in rendered
    # Quoted column appears verbatim with the SQLite type tag.
    assert '"Free Meal Count (K-12)" : INTEGER' in rendered
    # Sample-row header repeats the same quoting.
    assert '| "Free Meal Count (K-12)" |' in rendered


def test_entry_to_question_sets_db_path_and_reference_sql(tmp_path: Path) -> None:
    db_path = _build_tiny_bird_db(tmp_path)
    tables = _introspect_db(db_path, sample_rows=1)
    entry = _BirdEvalEntry(
        question_id=7,
        db_id="school",
        db_path=db_path,
        question="how many schools?",
        evidence="",
        sql="SELECT COUNT(*) FROM schools",
        difficulty="moderate",
        tables=tables,
    )
    q = _entry_to_question(entry, max_value_chars=40)

    assert isinstance(q, Question)
    assert q.name == "bird_school_7"
    assert q.db_path == str(db_path)
    assert q.reference_sql == {"sqlite": "SELECT COUNT(*) FROM schools"}
    assert "BIRD/school/qid=7/moderate" in q.notes
    # Per-question schema is inlined into the prompt.
    assert "schools" in q.prompt
    assert "Sample rows" in q.prompt


def test_looks_like_bird_db_dir_threshold(tmp_path: Path) -> None:
    # No DBs yet -> not recognized.
    assert not _looks_like_bird_db_dir(tmp_path, {"school", "movies"})
    _build_tiny_bird_db(tmp_path, db_id="school")
    # 1/2 of referenced DBs present -> still passes the >= half rule
    # (max(1, len // 2) == 1).
    assert _looks_like_bird_db_dir(tmp_path, {"school", "movies"})
    # All referenced DBs present.
    assert _looks_like_bird_db_dir(tmp_path, {"school"})
    # Wrong root.
    assert not _looks_like_bird_db_dir(tmp_path / "missing", {"school"})


def test_factory_returns_bird_executor() -> None:
    ex = get_executor("bird")
    assert isinstance(ex, BirdSqliteExecutor)
    assert ex.dialect_label() == "sqlite"
    assert _BIRD_EVAL_SCHEMA_PROMPT in ex.schema_prompt()


def test_bird_executor_executes_against_per_question_db(tmp_path: Path) -> None:
    db_path = _build_tiny_bird_db(tmp_path)
    q = Question(
        name="bird_school_1",
        prompt="<schema inlined>",
        reference_sql={"sqlite": "SELECT COUNT(*) FROM schools"},
        db_path=str(db_path),
    )
    ex = BirdSqliteExecutor()
    ex.setup()
    try:
        ok = ex.execute("SELECT COUNT(*) AS n FROM schools", question=q)
        assert ok.success, ok.error
        assert ok.rows == [{"n": 3}]
        assert ok.columns == ["n"]
        assert ok.backend == "bird"

        # Quoted column round-trips.
        ok2 = ex.execute(
            'SELECT "Free Meal Count (K-12)" AS fm FROM schools '
            "WHERE id = 1",
            question=q,
        )
        assert ok2.success, ok2.error
        assert ok2.rows == [{"fm": 120}]
    finally:
        ex.teardown()


def test_bird_executor_rejects_writes(tmp_path: Path) -> None:
    db_path = _build_tiny_bird_db(tmp_path)
    q = Question(
        name="bird_school_1",
        prompt="<schema inlined>",
        reference_sql={"sqlite": "SELECT 1"},
        db_path=str(db_path),
    )
    ex = BirdSqliteExecutor()
    ex.setup()
    try:
        bad = ex.execute(
            "DELETE FROM schools WHERE id = 1", question=q
        )
        assert not bad.success
        assert "read-only" in (bad.error or "")
    finally:
        ex.teardown()


def test_bird_executor_requires_question_with_db_path() -> None:
    ex = BirdSqliteExecutor()
    ex.setup()
    try:
        miss = ex.execute("SELECT 1", question=None)
        assert not miss.success
        assert "db_path" in (miss.error or "")

        no_path = Question(
            name="x",
            prompt="x",
            reference_sql={"sqlite": "SELECT 1"},
            db_path=None,
        )
        miss2 = ex.execute("SELECT 1", question=no_path)
        assert not miss2.success
        assert "db_path" in (miss2.error or "")
    finally:
        ex.teardown()


def test_question_dataclass_db_path_default_is_none() -> None:
    """Backwards compat: existing callers don't have to set db_path."""
    q = Question(
        name="ge_q01",
        prompt="...",
        reference_sql={"sqlite": "SELECT 1"},
    )
    assert q.db_path is None


def test_bird_executor_is_thread_safe_under_concurrent_evals(
    tmp_path: Path,
) -> None:
    """The runner's `--concurrency N` fans `execute()` out across N
    worker threads. Each worker should get its own sqlite3 connection
    via the thread-local cache (so no GIL-bound contention on a single
    connection's statement-level lock), and parallel reads must
    return the same answers a sequential run would.

    Test shape: 8 worker threads x 32 reads each = 256 concurrent
    executes against the same fake BIRD DB. We mix three query
    shapes -- COUNT(*), a quoted-column lookup, and a join -- so
    that any cross-thread state leak (shared cursor, cached row
    factory, etc.) would surface as a wrong row.
    """
    db_path = _build_tiny_bird_db(tmp_path)
    q = Question(
        name="bird_school_1",
        prompt="<schema inlined>",
        reference_sql={"sqlite": "SELECT 1"},
        db_path=str(db_path),
    )
    ex = BirdSqliteExecutor()
    ex.setup()

    queries: list[tuple[str, list[dict]]] = [
        (
            "SELECT COUNT(*) AS n FROM schools",
            [{"n": 3}],
        ),
        (
            'SELECT "Free Meal Count (K-12)" AS fm FROM schools '
            "WHERE id = 2",
            [{"fm": 80}],
        ),
        (
            "SELECT s.name AS sname FROM schools s "
            "ORDER BY s.id LIMIT 1",
            [{"sname": "Alpha"}],
        ),
    ]
    n_workers = 8
    reads_per_worker = 32
    seen_thread_ids: set[int] = set()
    seen_lock = threading.Lock()

    def _run_one(idx: int) -> None:
        sql, expected = queries[idx % len(queries)]
        result = ex.execute(sql, question=q)
        assert result.success, result.error
        assert result.rows == expected
        with seen_lock:
            seen_thread_ids.add(threading.get_ident())

    try:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(_run_one, i)
                for i in range(n_workers * reads_per_worker)
            ]
            for fut in as_completed(futures):
                fut.result()  # re-raise any worker exception

        # Every worker thread that touched the executor opened
        # exactly one connection for this db_path (per-thread cache),
        # so the global registry size must equal the number of
        # distinct workers we observed.
        assert len(seen_thread_ids) >= 2, (
            "Concurrency test degenerated to a single thread; the "
            "ThreadPoolExecutor never spread the work."
        )
        assert len(ex._all_conns) == len(seen_thread_ids)
    finally:
        ex.teardown()
    # teardown() must have closed every cached connection.
    assert ex._all_conns == []


@pytest.mark.parametrize(
    "ex_cls",
    [SqliteExecutor, SyntheticExecutor, TinybirdExecutor],
)
def test_existing_executors_ignore_question_kwarg(ex_cls: type) -> None:
    """Adding the optional kwarg must not change behavior for the
    global-schema executors. We don't need them to actually run SQL
    here -- just that the signature accepts ``question=...`` without
    raising.
    """
    sig = inspect.signature(ex_cls.execute)
    assert "question" in sig.parameters
    assert sig.parameters["question"].default is None
