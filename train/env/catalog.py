"""Catalog providers: in-memory tables + IR schemas + a schema prompt.

A ``CatalogProvider`` bundles three things the env needs to drive a
dialect engine:

1. ``tables``  - a ``{name: polars.DataFrame}`` dict matching what the
   manysql executor expects to find when it sees a ``Scan(name)``.
2. ``schemas`` - a ``{name: tuple[ColumnSchema, ...]}`` dict the
   dialect's ``lowering.lower(tree, semantics, schemas)`` needs.
3. ``schema_prompt`` - a free-form text blob that the env splices into
   the system prompt so the LLM knows what columns/types are available.

Two providers ship out of the box:

* :class:`GoldenCatalog` wraps the canonical 5-table manysql catalog
  (employees / departments / regions / sales / categories). Rich
  enough for joins, recursive CTEs, windows, correlated subqueries -
  the same shape ``manysql.golden.queries`` exercises.

* :class:`GithubEventsCatalog` wraps the eval suite's single-table
  ``github_events`` corpus. Use this when you want to drop the env
  into the existing eval question pool (50 NL questions in
  ``eval.dataset.questions``).

Adding a new catalog (e.g. for a synthetic spider-style benchmark)
means subclassing ``CatalogProvider`` and producing the three artifacts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class CatalogSnapshot:
    """Materialized view of a catalog provider.

    Held by ``DialectRuntime`` once setup is done. Frozen-ish: callers
    should not mutate ``tables`` after construction (the executor reads
    them by reference and may build lazy views over them).

    Types are loose (``Any``) here to keep this module import-cheap;
    the actual contents are ``polars.DataFrame`` and tuples of
    ``manysql.ir.plan.ColumnSchema``. Callers that need the strict
    types annotate at the use site.
    """

    tables: dict[str, Any]
    schemas: dict[str, tuple[Any, ...]]
    schema_prompt: str


class CatalogProvider(ABC):
    """ABC for catalog providers.

    Implementations should be cheap to construct but defer heavy work
    (allocating large DataFrames, reading parquet, etc.) to ``build()``.
    The env calls ``build()`` once per task generator / per process.
    """

    name: str = "abstract"

    @abstractmethod
    def build(self) -> CatalogSnapshot:
        """Materialize tables + schemas. May be slow."""


# ---------------------------------------------------------------------------
# Golden 5-table catalog
# ---------------------------------------------------------------------------


class GoldenCatalog(CatalogProvider):
    """The canonical manysql test catalog.

    Same dataset that ``manysql.golden.queries`` and the oracle harness
    use. Five tables (employees / departments / regions / sales /
    categories) deliberately shaped to exercise nullables, ties, leap
    days, correlated subqueries, recursive CTEs, and empty groups.

    This is the high-quality option for RL: a small but rich schema
    that supports virtually every IR feature, paired with the
    hand-curated golden query corpus for ground truth.
    """

    name = "golden"

    def build(self) -> CatalogSnapshot:
        from manysql.storage import CATALOG, schema_of, seed_datasets  # noqa: PLC0415

        tables = seed_datasets()
        schemas = {name: schema_of(name) for name in CATALOG}
        return CatalogSnapshot(
            tables=tables,
            schemas=schemas,
            schema_prompt=_GOLDEN_SCHEMA_PROMPT,
        )


# Hand-written schema description for the golden catalog. Mirrors what
# ``eval.dataset.github_events.SCHEMA_PROMPT`` does for the eval suite:
# columns + types + a few notes the LLM would otherwise have to guess at.
# Kept close to the data so a refactor of ``manysql.storage.catalog``
# notices when this drifts.
_GOLDEN_SCHEMA_PROMPT = """\
Tables (manysql canonical test catalog; small but feature-complete):

  employees(id INT, name TEXT, dept_id INT?, manager_id INT?, salary FLOAT?,
            hired_on DATE, active BOOL)
    8 rows. ``dept_id`` is nullable (heidi is unassigned). ``manager_id``
    is nullable for org roots. ``salary`` has a NULL (grace) and a leap-day
    hire (2024-02-29).

  departments(id INT, name TEXT, region_id INT?, budget FLOAT)
    4 rows. ``region_id`` is nullable for Marketing.

  regions(id INT, name TEXT)
    3 rows: NA, EU, APAC. APAC has no department (LEFT/FULL JOIN test).

  sales(id INT, employee_id INT, amount FLOAT?, sold_on DATE, region_id INT?)
    10 rows. ``amount`` includes a NULL and a 0.0. ``sold_on`` includes
    a leap day and ties (two rows on 2024-07-04). ``region_id`` is
    nullable for unallocated sales.

  categories(id INT, name TEXT, parent_id INT?)
    7-row tree (root -> electronics/{phones, computers/laptops},
    books/fiction). ``parent_id`` is NULL only at the root. Designed
    for recursive CTEs.

Notes:
- Date columns are real DATE (not TEXT). ``EXTRACT`` and date arithmetic
  work; for portable grouping by month use ``CAST(hired_on AS TEXT)`` and
  SUBSTR (1, 7).
- BOOLEAN ``active`` may be compared with ``= true``/``= false`` or used
  bare (``WHERE active``) - whichever the dialect's grammar accepts.
- All five tables are read-only and held in memory; queries that try to
  modify them will fail at parse time (the dialects ship SELECT-only
  grammars).
"""


# ---------------------------------------------------------------------------
# Eval-suite single-table catalog
# ---------------------------------------------------------------------------


class GithubEventsCatalog(CatalogProvider):
    """The eval suite's synthetic GitHub-events corpus.

    Single table, ~5000 rows by default, deterministic from a seed.
    Same data the SQLite eval backend uses, so reference SQL written
    for the eval suite executes identically here.

    Pair this with :class:`EvalSuiteTaskGenerator` to drop the RL env
    into the existing 50-question NL benchmark.
    """

    name = "github_events"

    def __init__(self, *, seed: int = 0xDB, n_rows: int = 5_000) -> None:
        self.seed = seed
        self.n_rows = n_rows

    def build(self) -> CatalogSnapshot:
        import polars as pl  # noqa: PLC0415

        from eval.dataset.github_events import SCHEMA_PROMPT, seed_rows  # noqa: PLC0415
        from manysql.ir.plan import ColumnSchema  # noqa: PLC0415
        from manysql.ir.types import INT, TEXT  # noqa: PLC0415

        # Polars dtypes mirror the SQLite affinities used by the eval
        # backend, so the IR types we hand to the lowering are the
        # same regardless of which engine ends up executing the query.
        polars_dtypes: dict[str, type] = {
            "file_time": pl.Utf8,
            "event_type": pl.Utf8,
            "actor_login": pl.Utf8,
            "repo_name": pl.Utf8,
            "created_at": pl.Utf8,
            "updated_at": pl.Utf8,
            "action": pl.Utf8,
            "comment_id": pl.Int64,
            "commit_id": pl.Utf8,
            "body": pl.Utf8,
            "ref": pl.Utf8,
            "number": pl.Int64,
            "title": pl.Utf8,
            "labels": pl.Utf8,
            "state": pl.Utf8,
            "locked": pl.Int64,
            "assignee": pl.Utf8,
            "comments": pl.Int64,
            "author_association": pl.Utf8,
            "closed_at": pl.Utf8,
            "merged_at": pl.Utf8,
            "merged": pl.Int64,
            "commits": pl.Int64,
            "additions": pl.Int64,
            "deletions": pl.Int64,
            "changed_files": pl.Int64,
            "push_size": pl.Int64,
            "release_tag_name": pl.Utf8,
            "release_name": pl.Utf8,
            "review_state": pl.Utf8,
        }
        rows = seed_rows(seed=self.seed, n=self.n_rows)
        df = pl.DataFrame(rows, schema=polars_dtypes)

        ir_type_map = {pl.Utf8: TEXT, pl.Int64: INT}
        cols = tuple(
            ColumnSchema(name=col, type=ir_type_map[dtype])
            for col, dtype in polars_dtypes.items()
        )
        return CatalogSnapshot(
            tables={"github_events": df},
            schemas={"github_events": cols},
            schema_prompt=SCHEMA_PROMPT,
        )


__all__ = [
    "CatalogProvider",
    "CatalogSnapshot",
    "GithubEventsCatalog",
    "GoldenCatalog",
]
