"""Tests for the WikiSQL data source in ``train.env.wikisql``.

We don't actually download from HuggingFace in tests -- ``load_dataset``
is monkey-patched to return a tiny in-memory shim that mimics the real
WikiSQL row schema:

    {
      "question": str,
      "table": {"id": str, "header": list[str], "types": list[str],
                "rows": list[list[Any]]},
      "sql":   {"sel": int, "agg": int,
                "conds": {"column_index": [...],
                          "operator_index": [...],
                          "condition": [...]}},
    }

Coverage:

* :func:`_safe_ident` / :func:`_safe_table_name` / :func:`_dedupe_columns`
  -- pure-Python sanitization helpers (no deps).
* :class:`WikiSqlCatalog` -- snapshot shape, memoization, dedupe of
  table-id collisions, schema-prompt placeholder.
* :class:`WikiSqlTaskGenerator` -- gold rows computed via reference
  dialect, ``drop_empty`` filter, prompt rendering, ``catalog`` reuse.
* :func:`_build_canonical_sql` -- aggregator + condition handling, real
  vs text literal quoting, malformed-index fallback.
"""

from __future__ import annotations

from typing import Any

import pytest

from train.env.wikisql import (
    WikiSqlCatalog,
    WikiSqlEntry,
    WikiSqlTaskConfig,
    WikiSqlTaskGenerator,
    _build_canonical_sql,
    _coerce_float,
    _dedupe_columns,
    _safe_ident,
    _safe_table_name,
)

# ---------------------------------------------------------------------------
# In-memory WikiSQL shim
# ---------------------------------------------------------------------------


def _wikisql_row(
    *,
    qid: str,
    question: str,
    header: list[str],
    types: list[str],
    rows: list[list[Any]],
    sel: int,
    agg: int = 0,
    conds: list[tuple[int, int, Any]] | None = None,
) -> dict[str, Any]:
    """Build one WikiSQL-shaped row for the in-memory shim."""
    cs = conds or []
    return {
        "question": question,
        "table": {
            "id": qid,
            "header": header,
            "types": types,
            "rows": rows,
        },
        "sql": {
            "sel": sel,
            "agg": agg,
            "conds": {
                "column_index": [c[0] for c in cs],
                "operator_index": [c[1] for c in cs],
                "condition": [c[2] for c in cs],
            },
        },
    }


class _FakeWikiSql:
    """Mimics ``datasets.Dataset`` for the rows we care about: ``len``,
    integer indexing, and ``__getitem__``. Nothing else.
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._rows[idx]


@pytest.fixture
def fake_dataset() -> list[dict[str, Any]]:
    """A tiny, deterministic WikiSQL fragment that exercises:

    * Aggregators (COUNT, SUM, MAX) and the no-aggregator path.
    * Real-typed and text-typed columns.
    * Conditions on both real and text columns.
    * Duplicate sanitized headers (forces dedupe).
    * Empty-result query (filtered by default).
    * Tied table ids (forces unique-table-name suffixing).
    """
    return [
        # 0: COUNT of populations > 1M
        _wikisql_row(
            qid="t-cities-1",
            question="how many cities have population over 1000000?",
            header=["City Name", "Population", "Country"],
            types=["text", "real", "text"],
            rows=[
                ["Tokyo", 13000000, "JP"],
                ["Reykjavik", 130000, "IS"],
                ["Lima", 9700000, "PE"],
            ],
            sel=0,
            agg=3,  # COUNT
            conds=[(1, 1, 1000000)],  # population > 1000000
        ),
        # 1: simple SELECT name FROM cities WHERE country = 'JP'
        _wikisql_row(
            qid="t-cities-1",  # tied id! forces dedupe
            question="what cities are in Japan?",
            header=["City Name", "Population", "Country"],
            types=["text", "real", "text"],
            rows=[
                ["Tokyo", 13000000, "JP"],
                ["Reykjavik", 130000, "IS"],
                ["Osaka", 2700000, "JP"],
            ],
            sel=0,
            agg=0,
            conds=[(2, 0, "JP")],
        ),
        # 2: empty-result query (filter matches nothing)
        _wikisql_row(
            qid="t-cities-2",
            question="cities with population over a quintillion?",
            header=["City Name", "Population"],
            types=["text", "real"],
            rows=[["Tokyo", 13000000]],
            sel=0,
            agg=0,
            conds=[(1, 1, 10**18)],
        ),
        # 3: duplicate sanitized headers ("Score (1)" / "Score (2)")
        _wikisql_row(
            qid="t-scoreboard",
            question="max of the second score?",
            header=["Player", "Score (1)", "Score (2)"],
            types=["text", "real", "real"],
            rows=[
                ["A", 10, 20],
                ["B", 30, 40],
            ],
            sel=2,
            agg=1,  # MAX
        ),
    ]


@pytest.fixture
def patched_wikisql(monkeypatch: pytest.MonkeyPatch, fake_dataset):
    """Stub ``datasets.load_dataset`` for the duration of the test.

    Returns nothing -- the side effect is the patched loader. Anything
    that builds a :class:`WikiSqlCatalog` inside this fixture's scope
    sees the in-memory shim instead of hitting the network.
    """
    fake = _FakeWikiSql(fake_dataset)

    def _fake_load_dataset(name: str, **_kwargs: Any):
        # Real ``load_dataset`` accepts split, revision, etc. We accept
        # everything and ignore -- the test shim is keyed only by the
        # rows fixture, not the kwargs.
        return fake

    # Patch on the import site (it's a local import inside build()).
    import datasets  # noqa: PLC0415

    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset)
    return fake


# ---------------------------------------------------------------------------
# Sanitization helpers (pure functions, no deps)
# ---------------------------------------------------------------------------


def test_safe_ident_basic() -> None:
    assert _safe_ident("Population") == "c_population"
    assert _safe_ident("city name") == "c_city_name"
    assert _safe_ident("Year (BCE)") == "c_year_bce"


def test_safe_ident_strips_diacritics_and_unicode() -> None:
    assert _safe_ident("Café") == "c_cafe"
    # All-non-ASCII -> falls back to c_<fallback>
    assert _safe_ident("北京", fallback="col0") == "c_col0"


def test_safe_ident_empty_falls_back() -> None:
    assert _safe_ident("", fallback="x") == "c_x"
    assert _safe_ident("###", fallback="x") == "c_x"


def test_safe_table_name() -> None:
    assert _safe_table_name("1-2-x-y") == "wikisql_1_2_x_y"
    assert _safe_table_name("") == "wikisql_anon"
    assert _safe_table_name("@@@") == "wikisql_anon"


def test_dedupe_columns() -> None:
    assert _dedupe_columns(["a", "b", "a", "a", "b"]) == ["a", "b", "a_1", "a_2", "b_1"]
    assert _dedupe_columns([]) == []


def test_coerce_float() -> None:
    assert _coerce_float(3) == 3.0
    assert _coerce_float(3.5) == 3.5
    assert _coerce_float("4.25") == 4.25
    assert _coerce_float("  -7  ") == -7.0
    assert _coerce_float(None) is None
    assert _coerce_float("not a number") is None
    assert _coerce_float(True) == 1.0  # booleans coerce


# ---------------------------------------------------------------------------
# WikiSqlCatalog
# ---------------------------------------------------------------------------


def test_wikisql_catalog_validates_args() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        WikiSqlCatalog(n_samples=0)
    with pytest.raises(ValueError, match="split"):
        WikiSqlCatalog(split="bogus")
    with pytest.raises(ValueError, match="sample_rows"):
        WikiSqlCatalog(sample_rows=-1)


def test_wikisql_catalog_build_shape(patched_wikisql) -> None:
    cat = WikiSqlCatalog(n_samples=4, seed=0)
    snap = cat.build()
    # All 4 rows materialized despite tied id (suffix collision -> _1).
    assert len(snap.tables) == 4
    assert len(snap.schemas) == 4
    # Tables are uniquely named.
    assert len(set(snap.tables)) == len(snap.tables)
    # Schema prompt is the placeholder, not a giant grammar dump.
    assert "WikiSQL" in snap.schema_prompt
    assert "c_" in snap.schema_prompt  # mentions the sanitization scheme

    # Entries align with tables.
    entries = cat.entries()
    assert len(entries) == 4
    for e in entries:
        assert e.table_name in snap.tables
        # Sanitized header is the c_* form.
        for h in e.safe_header:
            assert h.startswith("c_")
        # Original header preserved unchanged.
        assert "City Name" in e.original_header or "Player" in e.original_header


def test_wikisql_catalog_build_is_memoized(patched_wikisql) -> None:
    """Multiple build()s return the same snapshot instance.

    Critical for multi-dialect: N runtimes share one catalog and we
    don't want N HF downloads + N polars allocations.
    """
    cat = WikiSqlCatalog(n_samples=4, seed=0)
    snap1 = cat.build()
    snap2 = cat.build()
    assert snap1 is snap2


def test_wikisql_catalog_dedupes_table_ids(patched_wikisql) -> None:
    """Tied-id rows get unique ``wikisql_<safe>_N`` table names."""
    cat = WikiSqlCatalog(n_samples=4, seed=0)
    cat.build()
    table_names = sorted(cat._snapshot.tables)
    cities_tables = [t for t in table_names if t.startswith("wikisql_t_cities_1")]
    # Two rows tied on id "t-cities-1" -> two tables, distinguished by suffix.
    assert len(cities_tables) == 2


def test_wikisql_catalog_dedupes_columns(patched_wikisql) -> None:
    """Tables with collision-prone headers get suffix-deduped columns."""
    cat = WikiSqlCatalog(n_samples=4, seed=0)
    cat.build()
    score_entry = next(
        e for e in cat.entries() if e.table_name.startswith("wikisql_t_scoreboard")
    )
    # "Score (1)" / "Score (2)" -> c_score_1 / c_score_2 (or _1 suffix).
    assert len(score_entry.safe_header) == 3
    assert len(set(score_entry.safe_header)) == 3


# ---------------------------------------------------------------------------
# WikiSqlTaskGenerator
# ---------------------------------------------------------------------------


def test_wikisql_task_generator_builds_tasks(patched_wikisql) -> None:
    cfg = WikiSqlTaskConfig(target_dialect="aggressive_alien", n_samples=4, seed=0)
    gen = WikiSqlTaskGenerator(cfg)
    gen.build()
    tasks = gen.all_tasks()
    # 4 rows; one should be filtered by drop_empty (the quintillion query).
    assert 1 <= len(tasks) <= 4
    for t in tasks:
        assert t.meta.dialect == "aggressive_alien"
        assert t.meta.generator == "wikisql"
        assert t.meta.task_id.startswith("wikisql_t-")
        assert "Question:" in t.prompt
        assert "Table:" in t.prompt
        assert "Sample rows:" in t.prompt
        # Gold rows are non-empty (drop_empty filter).
        assert t.gold_rows
        # Gold SQL uses sanitized identifiers (lower-case c_* form).
        assert "c_" in t.gold_sql
        assert "wikisql_" in t.gold_sql


def test_wikisql_task_generator_drop_empty_off_keeps_all(patched_wikisql) -> None:
    cfg = WikiSqlTaskConfig(
        target_dialect="aggressive_alien",
        n_samples=4,
        seed=0,
        drop_empty=False,
    )
    gen = WikiSqlTaskGenerator(cfg)
    gen.build()
    # With drop_empty=False, the empty-result task is preserved (modulo
    # any reference-engine failures).
    tasks_with_empty = gen.all_tasks()
    cfg2 = WikiSqlTaskConfig(target_dialect="aggressive_alien", n_samples=4, seed=0)
    gen2 = WikiSqlTaskGenerator(cfg2)
    gen2.build()
    tasks_default = gen2.all_tasks()
    assert len(tasks_with_empty) >= len(tasks_default)


def test_wikisql_task_generator_reuses_passed_catalog(patched_wikisql) -> None:
    """When config.catalog is supplied, the generator reuses it instead
    of building its own. Required for multi-dialect to avoid N
    materializations.
    """
    cat = WikiSqlCatalog(n_samples=4, seed=0)
    cat.build()
    snap_id = id(cat.build())

    cfg = WikiSqlTaskConfig(
        target_dialect="aggressive_alien", n_samples=4, seed=0, catalog=cat
    )
    gen = WikiSqlTaskGenerator(cfg)
    gen.build()
    # Catalog identity preserved -> no rebuild.
    assert gen.catalog is cat
    assert id(cat.build()) == snap_id


def test_wikisql_task_generator_idempotent_build(patched_wikisql) -> None:
    cfg = WikiSqlTaskConfig(target_dialect="aggressive_alien", n_samples=4, seed=0)
    gen = WikiSqlTaskGenerator(cfg)
    gen.build()
    n1 = len(gen.all_tasks())
    gen.build()  # second call is a no-op
    n2 = len(gen.all_tasks())
    assert n1 == n2


# ---------------------------------------------------------------------------
# _build_canonical_sql
# ---------------------------------------------------------------------------


def _make_entry(
    *,
    sel: int,
    agg: int = 0,
    conds: list[dict[str, Any]] | None = None,
    safe_header: list[str] | None = None,
    types: list[str] | None = None,
) -> WikiSqlEntry:
    return WikiSqlEntry(
        raw_id="x",
        table_name="wikisql_x",
        original_header=["City", "Pop"],
        safe_header=safe_header or ["c_city", "c_pop"],
        types=types or ["text", "real"],
        sample_rows=[],
        n_rows=0,
        question="?",
        sel=sel,
        agg=agg,
        conds=conds or [],
    )


def test_canonical_sql_no_agg_no_cond() -> None:
    sql = _build_canonical_sql(_make_entry(sel=0))
    assert sql == "SELECT c_city FROM wikisql_x"


def test_canonical_sql_agg() -> None:
    # COUNT
    sql = _build_canonical_sql(_make_entry(sel=0, agg=3))
    assert sql == "SELECT COUNT(c_city) FROM wikisql_x"
    # MAX on a real column
    sql = _build_canonical_sql(_make_entry(sel=1, agg=1))
    assert sql == "SELECT MAX(c_pop) FROM wikisql_x"


def test_canonical_sql_text_condition_quoted() -> None:
    sql = _build_canonical_sql(
        _make_entry(
            sel=0,
            conds=[{"column_index": 0, "operator_index": 0, "condition": "Lima"}],
        )
    )
    assert sql == "SELECT c_city FROM wikisql_x WHERE c_city = 'Lima'"


def test_canonical_sql_real_condition_numeric() -> None:
    sql = _build_canonical_sql(
        _make_entry(
            sel=0,
            conds=[{"column_index": 1, "operator_index": 1, "condition": 1000000}],
        )
    )
    assert sql == "SELECT c_city FROM wikisql_x WHERE c_pop > 1000000.0"


def test_canonical_sql_real_condition_unparseable_falls_back_to_string() -> None:
    sql = _build_canonical_sql(
        _make_entry(
            sel=0,
            conds=[{"column_index": 1, "operator_index": 0, "condition": "not-numeric"}],
        )
    )
    assert "'not-numeric'" in sql


def test_canonical_sql_text_condition_escapes_apostrophes() -> None:
    sql = _build_canonical_sql(
        _make_entry(
            sel=0,
            conds=[{"column_index": 0, "operator_index": 0, "condition": "O'Brien"}],
        )
    )
    assert "'O''Brien'" in sql


def test_canonical_sql_drops_out_of_range_indices() -> None:
    # column_index=99 is past the end -> condition silently dropped
    sql = _build_canonical_sql(
        _make_entry(
            sel=0,
            conds=[{"column_index": 99, "operator_index": 0, "condition": "x"}],
        )
    )
    assert "WHERE" not in sql


def test_canonical_sql_malformed_sel_returns_empty_query() -> None:
    sql = _build_canonical_sql(_make_entry(sel=99))
    assert "WHERE 1 = 0" in sql


# ---------------------------------------------------------------------------
# Integration with DialectRuntime: tasks actually execute
# ---------------------------------------------------------------------------


def test_wikisql_tasks_executable_in_target_dialect(patched_wikisql) -> None:
    """End-to-end: build WikiSQL tasks, then run the gold SQL through a
    real dialect runtime. The reference engine produced these gold rows
    in the generator; here we confirm a target dialect can also execute
    them (the whole point of the catalog being shared across dialects).
    """
    from train.env.engine import DialectRuntime  # noqa: PLC0415

    cat = WikiSqlCatalog(n_samples=4, seed=0)
    cat.build()
    cfg = WikiSqlTaskConfig(
        target_dialect="aggressive_alien", n_samples=4, seed=0, catalog=cat
    )
    gen = WikiSqlTaskGenerator(cfg)
    gen.build()
    tasks = gen.all_tasks()
    assert tasks

    rt = DialectRuntime(dialect="aggressive_alien", catalog=cat)
    rt.setup()
    try:
        for t in tasks:
            run = rt.run(t.gold_sql)
            assert run.exec_result.success, (
                f"target dialect failed gold SQL {t.gold_sql!r}: "
                f"{run.exec_result.error}"
            )
            # Same rows as the reference dialect produced (data is
            # dialect-independent).
            assert run.exec_result.rows == t.gold_rows
    finally:
        rt.teardown()
