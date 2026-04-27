"""Tests for the SynSQL-2.5M data source in ``train.env.synsql``.

These tests don't actually fetch SynSQL from HuggingFace. We exercise:

* :func:`iter_json_array_items` -- the custom incremental JSON-array
  parser, which is the most novel and isolation-friendly piece. We
  feed it byte streams with various edge shapes (chunks landing
  mid-string, mid-multibyte-char, nested arrays, escaped quotes,
  trailing whitespace).

* :class:`SynSqlCatalog` -- argument validation, sample-cache
  round-trip, build with a hand-crafted SQLite db_dir, table
  namespacing, schema prompt placeholder.

* :class:`SynSqlTaskGenerator` -- end-to-end task construction with
  ``urllib.request.urlopen`` monkey-patched to a BytesIO of fake
  ``data.json`` content, and the database directory pre-populated
  with synthetic SQLite files. Verifies gold rows, ``drop_empty``,
  and prompt rendering.

* :func:`_render_user_prompt` -- dataset-independent prompt format.

The fixtures intentionally cover only the surface area needed to
prove the streaming + catalog wiring works; the row-cap path
inherited from :mod:`train.env.bird` is covered by the BIRD test
suite.
"""

from __future__ import annotations

import io
import json
import sqlite3
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import pytest

from train.env.synsql import (
    _SYNSQL_SCHEMA_PROMPT,
    SynSqlCatalog,
    SynSqlEntry,
    SynSqlTableInfo,
    SynSqlTaskConfig,
    SynSqlTaskGenerator,
    _format_cell,
    _format_type_label,
    _render_user_prompt,
    iter_json_array_items,
)

# ---------------------------------------------------------------------------
# iter_json_array_items: incremental JSON-array parser
# ---------------------------------------------------------------------------


class _ChunkedStream:
    """Test stream that hands out bytes in fixed-size chunks.

    The parser must work regardless of where chunk boundaries fall
    (mid-string, mid-multibyte UTF-8, between items, etc.). Forcing
    pathological chunk sizes is the simplest way to reach those
    boundaries deterministically.
    """

    def __init__(self, data: bytes, chunk: int = 1) -> None:
        self._data = data
        self._chunk = chunk
        self._pos = 0

    def read(self, n: int) -> bytes:
        # Honour the underlying constraint, but only ever hand out at
        # most self._chunk bytes per call. The parser passes its own
        # ``chunk_size`` to ``read``; we cap further so byte boundaries
        # land in the worst possible places.
        take = min(n, self._chunk, len(self._data) - self._pos)
        if take <= 0:
            return b""
        out = self._data[self._pos : self._pos + take]
        self._pos += take
        return out


def _drain(stream: _ChunkedStream, **kwargs: Any) -> list[Any]:
    return list(iter_json_array_items(stream, **kwargs))


def test_iter_json_array_items_simple() -> None:
    raw = b'[{"a": 1}, {"a": 2}, {"a": 3}]'
    items = _drain(_ChunkedStream(raw), chunk_size=64)
    assert items == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_iter_json_array_items_empty_array() -> None:
    items = _drain(_ChunkedStream(b"[]"), chunk_size=4)
    assert items == []


def test_iter_json_array_items_handles_tiny_chunks() -> None:
    """1-byte chunks force the parser to glue every byte through state."""
    raw = b'[{"a": 1, "b": "x"}, {"a": 2, "b": "y, z"}]'
    items = _drain(_ChunkedStream(raw, chunk=1), chunk_size=1)
    assert items == [{"a": 1, "b": "x"}, {"a": 2, "b": "y, z"}]


def test_iter_json_array_items_handles_nested_objects_and_arrays() -> None:
    raw = b'[{"a": [1, 2, {"b": 3}], "c": {"d": [4]}}, {"e": []}]'
    items = _drain(_ChunkedStream(raw, chunk=3), chunk_size=3)
    assert items == [
        {"a": [1, 2, {"b": 3}], "c": {"d": [4]}},
        {"e": []},
    ]


def test_iter_json_array_items_string_with_escaped_quotes_and_brackets() -> None:
    raw = b'[{"q": "she said \\"hi\\"", "p": "[]{}"}]'
    items = _drain(_ChunkedStream(raw, chunk=2), chunk_size=2)
    assert items == [{"q": 'she said "hi"', "p": "[]{}"}]


def test_iter_json_array_items_unicode_split_across_chunks() -> None:
    """A 4-byte UTF-8 char split across chunks must decode cleanly."""
    raw = json.dumps([{"k": "你好"}], ensure_ascii=False).encode("utf-8")
    items = _drain(_ChunkedStream(raw, chunk=1), chunk_size=1)
    assert items == [{"k": "你好"}]


def test_iter_json_array_items_tolerates_leading_whitespace() -> None:
    raw = b'   \n  [ {"a": 1} ,  {"a": 2}  ]  '
    items = _drain(_ChunkedStream(raw, chunk=2), chunk_size=2)
    assert items == [{"a": 1}, {"a": 2}]


def test_iter_json_array_items_caller_can_break_early() -> None:
    """Bailing out of the generator does not consume past the break point.

    This is what the catalog relies on to keep network use proportional
    to ``n_samples`` rather than the full 9.36GB file.
    """
    raw = b'[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]'
    stream = _ChunkedStream(raw, chunk=1)
    out: list[Any] = []
    for item in iter_json_array_items(stream, chunk_size=1):
        out.append(item)
        if len(out) == 2:
            break
    assert out == [{"a": 1}, {"a": 2}]
    # Stream still has bytes left -- we did NOT drain it.
    assert stream._pos < len(raw)


# ---------------------------------------------------------------------------
# Synthetic SQLite db_dir + fake data.json fixtures
# ---------------------------------------------------------------------------


def _make_sqlite(path: Path, *, statements: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        for stmt in statements:
            conn.executescript(stmt)
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def synsql_db_root(tmp_path: Path) -> Path:
    """Create ``<root>/<db_id>/<db_id>.sqlite`` for two tiny test DBs.

    Layout matches what the official ``databases.zip`` extracts to,
    so :meth:`SynSqlCatalog._db_path_for` resolves both DBs without
    auto-download.
    """
    root = tmp_path / "synsql_dbs"
    _make_sqlite(
        root / "city_db" / "city_db.sqlite",
        statements=[
            """
            CREATE TABLE cities (
                id INTEGER PRIMARY KEY,
                name TEXT,
                population INTEGER,
                country TEXT
            );
            INSERT INTO cities VALUES
                (1, 'Tokyo', 13000000, 'JP'),
                (2, 'Reykjavik', 130000, 'IS'),
                (3, 'Lima', 9700000, 'PE');
            """,
        ],
    )
    _make_sqlite(
        root / "shop_db" / "shop_db.sqlite",
        statements=[
            """
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT,
                price REAL
            );
            INSERT INTO products VALUES
                (1, 'apple', 1.5),
                (2, 'bread', 3.0);
            """,
        ],
    )
    return root


def _fake_data_json_payload() -> bytes:
    """JSON-array payload mimicking SynSQL's ``data.json`` shape.

    Item indices, complexity bands, and ``external_knowledge`` are
    chosen to exercise the catalog's filtering + prompt-rendering
    behaviors:

    * idx 0: simple, valid (matches default complexity filter).
    * idx 1: moderate, has external_knowledge -> exercises optional field.
    * idx 2: complex, dropped by default complexities=("simple","moderate").
    * idx 3: simple, gold SQL returns no rows -> exercises drop_empty.
    * idx 4: simple, references an unknown db_id -> exercises missing-DB warn.
    """
    items = [
        {
            "db_id": "city_db",
            "question": "How many cities are listed?",
            "sql": "SELECT COUNT(*) FROM cities",
            "external_knowledge": "",
            "sql_complexity": "simple",
            "question_style": "formal",
            "cot": "...",
        },
        {
            "db_id": "shop_db",
            "question": "What are the products and prices?",
            "sql": "SELECT name, price FROM products ORDER BY id",
            "external_knowledge": "Currency is USD.",
            "sql_complexity": "moderate",
            "question_style": "conversational",
            "cot": "...",
        },
        {
            "db_id": "city_db",
            "question": "List Japanese cities (complex).",
            "sql": "SELECT name FROM cities WHERE country = 'JP'",
            "external_knowledge": "",
            "sql_complexity": "complex",
            "question_style": "vague",
            "cot": "...",
        },
        {
            "db_id": "city_db",
            "question": "Cities with population over a quintillion?",
            "sql": "SELECT name FROM cities WHERE population > 1000000000000000000",
            "external_knowledge": "",
            "sql_complexity": "simple",
            "question_style": "formal",
            "cot": "...",
        },
        {
            "db_id": "missing_db",
            "question": "Anything from a non-existent DB.",
            "sql": "SELECT 1",
            "external_knowledge": "",
            "sql_complexity": "simple",
            "question_style": "formal",
            "cot": "...",
        },
    ]
    return json.dumps(items).encode("utf-8")


@pytest.fixture
def patched_data_json(monkeypatch: pytest.MonkeyPatch):
    """Replace ``urllib.request.urlopen`` with a BytesIO of fake data.json.

    The catalog's streaming code calls ``urllib.request.urlopen(req)``
    inside a ``with`` block; we hand back an object that supports
    ``__enter__`` / ``__exit__`` / ``read`` so the with-statement
    works unchanged.
    """
    payload = _fake_data_json_payload()

    class _FakeResponse(io.BytesIO):
        headers: dict[str, str] = {}

        def __enter__(self):  # type: ignore[override]
            return self

        def __exit__(self, *a: object) -> None:  # type: ignore[override]
            self.close()

    def _fake_urlopen(req, *args: Any, **kwargs: Any):  # noqa: ARG001
        return _FakeResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    return payload


# ---------------------------------------------------------------------------
# SynSqlCatalog: argument validation
# ---------------------------------------------------------------------------


def test_synsql_catalog_validates_args() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        SynSqlCatalog(n_samples=0)
    with pytest.raises(ValueError, match="split"):
        SynSqlCatalog(split="bogus")
    with pytest.raises(ValueError, match="sql_complexity"):
        SynSqlCatalog(complexities=("simple", "trivially_easy"))
    with pytest.raises(ValueError, match="sample_rows"):
        SynSqlCatalog(sample_rows=-1)


# ---------------------------------------------------------------------------
# SynSqlCatalog: sample cache layer
# ---------------------------------------------------------------------------


def test_synsql_sample_cache_path_components(tmp_path: Path) -> None:
    cat = SynSqlCatalog(
        n_samples=10,
        split="train",
        seed=7,
        complexities=("simple", "highly complex"),
        cache_dir=str(tmp_path),
    )
    p = cat._sample_cache_path()
    name = p.name
    assert "samples_train_seed7_n10" in name
    assert "highly_complex" in name  # space replaced with underscore
    assert "simple" in name
    assert name.endswith(".jsonl")


def test_synsql_sample_cache_round_trip(tmp_path: Path) -> None:
    """Cache hits skip the network: pre-populate the cache file and
    confirm ``_load_or_stream_questions`` returns it without calling
    ``_stream_questions``.
    """
    cat = SynSqlCatalog(
        n_samples=2,
        split="train",
        seed=0,
        complexities=("simple",),
        cache_dir=str(tmp_path),
        start_index=0,
    )
    cache_path = cat._sample_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    items = [
        {"db_id": "x", "question": "q1", "sql": "S1", "sql_complexity": "simple"},
        {"db_id": "y", "question": "q2", "sql": "S2", "sql_complexity": "simple"},
    ]
    with cache_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    def _explode(*_a: Any, **_kw: Any):
        raise AssertionError("network must not be called when cache hits")

    cat._stream_questions = _explode  # type: ignore[assignment]
    out = cat._load_or_stream_questions()
    assert out == items


# ---------------------------------------------------------------------------
# SynSqlCatalog: full build (with synthetic db_dir + fake data.json stream)
# ---------------------------------------------------------------------------


def test_synsql_catalog_build_streaming_end_to_end(
    tmp_path: Path,
    synsql_db_root: Path,
    patched_data_json,
) -> None:
    cat = SynSqlCatalog(
        n_samples=4,
        split="train",
        seed=0,
        complexities=("simple", "moderate"),
        start_index=0,
        db_dir=str(synsql_db_root),
        auto_download=False,
        cache_dir=str(tmp_path / "cache"),
        max_rows_per_table=None,  # tiny DBs, skip the row-cap mirror path
        sample_rows=2,
    )
    snap = cat.build()

    # Two referenced DBs (city_db, shop_db); each has one table; the
    # complex item and the missing_db item don't make it past filtering.
    # The empty-result item still contributes its catalog entry (drop_empty
    # is a task-level concern, not a catalog one).
    assert "city_db__cities" in snap.tables
    assert "shop_db__products" in snap.tables
    assert _SYNSQL_SCHEMA_PROMPT == snap.schema_prompt

    schemas = snap.schemas
    cities_cols = [c.name for c in schemas["city_db__cities"]]
    assert all(c.startswith("c_") for c in cities_cols)

    entries = cat.entries()
    # Three of the five items pass: 0 (city_db simple), 1 (shop_db
    # moderate), 3 (city_db simple, empty-result). #2 dropped by
    # complexity filter, #4 dropped because db_id missing.
    assert len(entries) == 3
    db_ids = sorted({e.db_id for e in entries})
    assert db_ids == ["city_db", "shop_db"]
    # Each entry's tables view is db-scoped (only that DB's tables).
    for e in entries:
        for t in e.tables:
            assert t.catalog_table_name.startswith(f"{e.db_id}__")


def test_synsql_catalog_build_is_memoized(
    tmp_path: Path, synsql_db_root: Path, patched_data_json
) -> None:
    cat = SynSqlCatalog(
        n_samples=2,
        split="train",
        seed=0,
        complexities=("simple",),
        start_index=0,
        db_dir=str(synsql_db_root),
        auto_download=False,
        cache_dir=str(tmp_path / "cache"),
        max_rows_per_table=None,
    )
    snap1 = cat.build()
    snap2 = cat.build()
    assert snap1 is snap2


def test_synsql_catalog_writes_sample_cache_after_streaming(
    tmp_path: Path, synsql_db_root: Path, patched_data_json
) -> None:
    cache_dir = tmp_path / "cache"
    cat = SynSqlCatalog(
        n_samples=2,
        split="train",
        seed=0,
        complexities=("simple",),
        start_index=0,
        db_dir=str(synsql_db_root),
        auto_download=False,
        cache_dir=str(cache_dir),
        max_rows_per_table=None,
    )
    cat.build()
    cache_path = cat._sample_cache_path()
    assert cache_path.is_file()
    lines = cache_path.read_text(encoding="utf-8").strip().splitlines()
    parsed = [json.loads(line) for line in lines]
    # Two simple items survive the filter (idx 0 and idx 3).
    assert len(parsed) == 2
    assert all(p["sql_complexity"] == "simple" for p in parsed)
    # The absolute-stream-index stamp is reproducible for downstream
    # task ids.
    assert {p["_synsql_index"] for p in parsed} == {0, 3}


def test_synsql_catalog_start_index_skips_items(
    tmp_path: Path, synsql_db_root: Path, patched_data_json
) -> None:
    """start_index=2 skips the first two items; only idx 3 (the simple
    empty-result one) and idx 4 (missing_db, dropped) remain at the
    head of the filtered stream.
    """
    cat = SynSqlCatalog(
        n_samples=5,
        split="train",
        seed=0,
        complexities=("simple", "moderate", "complex"),
        start_index=2,
        db_dir=str(synsql_db_root),
        auto_download=False,
        cache_dir=str(tmp_path / "cache"),
        max_rows_per_table=None,
    )
    cat.build()
    # Idx >= 2 -> items #2 (complex city_db), #3 (simple city_db),
    # #4 (simple missing_db). Of those, #4 drops because its DB is
    # absent. So 2 entries.
    indices = sorted(e.item_index for e in cat.entries())
    assert indices == [2, 3]


# ---------------------------------------------------------------------------
# SynSqlTaskGenerator: end-to-end with gold SQL execution
# ---------------------------------------------------------------------------


def test_synsql_task_generator_builds_tasks_with_gold_rows(
    tmp_path: Path, synsql_db_root: Path, patched_data_json
) -> None:
    cat = SynSqlCatalog(
        n_samples=4,
        split="train",
        seed=0,
        complexities=("simple", "moderate"),
        start_index=0,
        db_dir=str(synsql_db_root),
        auto_download=False,
        cache_dir=str(tmp_path / "cache"),
        max_rows_per_table=None,
        sample_rows=2,
    )
    cfg = SynSqlTaskConfig(
        target_dialect="aggressive_alien",
        n_samples=4,
        split="train",
        seed=0,
        complexities=("simple", "moderate"),
        start_index=0,
        db_dir=str(synsql_db_root),
        auto_download=False,
        cache_dir=str(tmp_path / "cache"),
        max_rows_per_table=None,
        catalog=cat,
    )
    gen = SynSqlTaskGenerator(cfg)
    gen.build()
    tasks = gen.all_tasks()

    # Three catalog entries; one is empty-result and gets dropped by
    # default drop_empty=True.
    assert len(tasks) == 2
    for t in tasks:
        assert t.meta.dialect == "aggressive_alien"
        assert t.meta.generator == "synsql"
        assert t.meta.task_id.startswith("synsql_")
        assert t.gold_rows  # non-empty (drop_empty filter)
        assert "Question:" in t.prompt
        assert "Database:" in t.prompt
        assert "Tables (in this database):" in t.prompt
        assert "external_knowledge_present" in t.meta.meta

    # The shop_db moderate task carries the external_knowledge hint.
    shop_task = next(t for t in tasks if "shop_db" in t.meta.task_id)
    assert "External knowledge: Currency is USD." in shop_task.prompt
    # And the city_db simple task does NOT print an empty
    # "External knowledge:" line.
    city_task = next(t for t in tasks if "city_db" in t.meta.task_id)
    assert "External knowledge:" not in city_task.prompt


def test_synsql_task_generator_keeps_empty_when_drop_empty_false(
    tmp_path: Path, synsql_db_root: Path, patched_data_json
) -> None:
    cat = SynSqlCatalog(
        n_samples=4,
        split="train",
        seed=0,
        complexities=("simple", "moderate"),
        start_index=0,
        db_dir=str(synsql_db_root),
        auto_download=False,
        cache_dir=str(tmp_path / "cache"),
        max_rows_per_table=None,
    )
    cfg = SynSqlTaskConfig(
        target_dialect="aggressive_alien",
        complexities=("simple", "moderate"),
        start_index=0,
        db_dir=str(synsql_db_root),
        auto_download=False,
        cache_dir=str(tmp_path / "cache"),
        max_rows_per_table=None,
        catalog=cat,
        drop_empty=False,
    )
    gen = SynSqlTaskGenerator(cfg)
    gen.build()
    tasks = gen.all_tasks()
    # All three valid entries survive when drop_empty is off.
    assert len(tasks) == 3


def test_synsql_task_generator_idempotent_build(
    tmp_path: Path, synsql_db_root: Path, patched_data_json
) -> None:
    cfg = SynSqlTaskConfig(
        target_dialect="aggressive_alien",
        n_samples=2,
        split="train",
        complexities=("simple", "moderate"),
        start_index=0,
        db_dir=str(synsql_db_root),
        auto_download=False,
        cache_dir=str(tmp_path / "cache"),
        max_rows_per_table=None,
    )
    gen = SynSqlTaskGenerator(cfg)
    gen.build()
    n1 = len(gen.all_tasks())
    gen.build()
    n2 = len(gen.all_tasks())
    assert n1 == n2


# ---------------------------------------------------------------------------
# Prompt rendering helpers (no I/O)
# ---------------------------------------------------------------------------


def test_format_type_label_passes_through_non_text() -> None:
    assert _format_type_label("INT", "INTEGER") == "INT"
    assert _format_type_label("FLOAT", "REAL") == "FLOAT"
    assert _format_type_label("BOOL", "BOOLEAN") == "BOOL"


def test_format_type_label_marks_text_dates() -> None:
    assert _format_type_label("TEXT", "DATETIME") == "TEXT, was DATETIME"
    assert _format_type_label("TEXT", "DATE") == "TEXT, was DATE"
    assert _format_type_label("TEXT", "TIMESTAMP") == "TEXT, was TIMESTAMP"
    assert _format_type_label("TEXT", "DATETIME(3)") == "TEXT, was DATETIME"
    assert _format_type_label("TEXT", "VARCHAR(255)") == "TEXT"
    assert _format_type_label("TEXT", "") == "TEXT"


def test_format_cell_truncation_and_normalization() -> None:
    assert _format_cell(None) == "NULL"
    assert _format_cell("a | b") == "a \\| b"
    assert _format_cell("line1\nline2") == "line1 line2"
    assert _format_cell("   trim   me   ") == "trim me"
    long = "x" * 100
    assert _format_cell(long, max_chars=20).endswith("...")
    assert len(_format_cell(long, max_chars=20)) == 20


def test_schema_prompt_mentions_text_date_contract() -> None:
    p = _SYNSQL_SCHEMA_PROMPT
    assert "DATE/TIME COLUMNS" in p
    assert "ISO" in p
    assert "CAST" in p
    assert "EXTRACT" in p


def test_render_user_prompt_includes_external_knowledge_when_present() -> None:
    entry = SynSqlEntry(
        item_index=0,
        db_id="sample_db",
        question="how many sales happened in 2024?",
        external_knowledge="Sales = SUM(amount).",
        sql="SELECT 1",
        sql_complexity="moderate",
        question_style="formal",
        db_path="/tmp/ignored.sqlite",
        tables=[
            SynSqlTableInfo(
                original_name="sales",
                catalog_table_name="sample_db__sales",
                safe_columns=["c_id", "c_amount", "c_sold_on"],
                original_columns=["id", "amount", "sold_on"],
                types=["INT", "FLOAT", "TEXT"],
                sqlite_types=["INTEGER", "REAL", "DATETIME"],
                sample_rows=[
                    {"c_id": 1, "c_amount": 10.0, "c_sold_on": "2024-01-15"}
                ],
                n_rows=1,
            )
        ],
    )
    out = _render_user_prompt(entry)
    assert "Question: how many sales happened in 2024?" in out
    assert "External knowledge: Sales = SUM(amount)." in out
    assert "Database: sample_db" in out
    assert "SQL complexity: moderate" in out
    # Sanitized + original column listing with date-affinity annotation.
    assert "c_sold_on  <-  sold_on  (TEXT, was DATETIME)" in out
    assert "c_amount  <-  amount  (FLOAT)" in out
    # Catalog name surfaces so the model knows what to FROM.
    assert "sample_db__sales" in out


def test_render_user_prompt_omits_external_knowledge_when_blank() -> None:
    entry = SynSqlEntry(
        item_index=1,
        db_id="d",
        question="q",
        external_knowledge="",
        sql="SELECT 1",
        sql_complexity="simple",
        question_style="formal",
        db_path="/tmp/x.sqlite",
        tables=[
            SynSqlTableInfo(
                original_name="t",
                catalog_table_name="d__t",
                safe_columns=["c_a"],
                original_columns=["a"],
                types=["INT"],
                sqlite_types=["INTEGER"],
                sample_rows=[],
                n_rows=0,
            )
        ],
    )
    out = _render_user_prompt(entry)
    assert "External knowledge:" not in out


# ---------------------------------------------------------------------------
# Integration with DialectRuntime: tasks actually execute
# ---------------------------------------------------------------------------


def test_synsql_tasks_executable_in_target_dialect(
    tmp_path: Path, synsql_db_root: Path, patched_data_json
) -> None:
    """End-to-end: build SynSQL tasks against a synthetic db_dir + fake
    data.json, then execute each task's gold SQL against a real
    :class:`DialectRuntime`. Mirrors the contract used by the WikiSQL
    test: the catalog is dialect-independent, so a target dialect must
    be able to run the same gold SQL the generator did against SQLite.

    Note: SynSQL's gold SQL uses the *original* (un-sanitized) table
    names like ``cities``, ``products`` -- those are looked up by
    :class:`DialectRuntime` against the in-memory catalog whose tables
    are namespaced ``<db_id>__<table>``. So the gold SQL won't run
    cleanly through DialectRuntime by default; the manysql RL contract
    is that gold rows come from ``sqlite3`` against the source file,
    and the *model* generates SQL referencing the namespaced tables.
    We therefore only assert the catalog is loadable by the runtime,
    which is what a training step actually needs.
    """
    from train.env.engine import DialectRuntime  # noqa: PLC0415

    cat = SynSqlCatalog(
        n_samples=4,
        split="train",
        seed=0,
        complexities=("simple", "moderate"),
        start_index=0,
        db_dir=str(synsql_db_root),
        auto_download=False,
        cache_dir=str(tmp_path / "cache"),
        max_rows_per_table=None,
    )
    cat.build()
    rt = DialectRuntime(dialect="aggressive_alien", catalog=cat)
    rt.setup()
    try:
        # A trivial query against a namespaced table must round-trip.
        # Keep the SQL minimal -- ORDER BY / LIMIT spellings vary across
        # synthetic dialects, but vanilla projection should always parse.
        run = rt.run("SELECT c_name FROM city_db__cities")
        assert run.exec_result.success, run.exec_result.error
        assert run.exec_result.rows
        names = {r["c_name"] for r in run.exec_result.rows}
        assert names == {"Tokyo", "Reykjavik", "Lima"}
    finally:
        rt.teardown()
