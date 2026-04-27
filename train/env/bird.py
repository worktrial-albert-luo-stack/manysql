"""BIRD-SQL prompt / database / answer triples as RL training data.

BIRD-SQL ([Li et al. 2023](https://bird-bench.github.io/)) is a
multi-database text-to-SQL benchmark whose questions sit roughly an
order of magnitude harder than WikiSQL's: real Kaggle-derived
schemas (5-25 tables per DB), evidence-style domain hints the model
has to apply, and SQL with joins, subqueries, CTEs, windows, and
CASE expressions. That harder rung is exactly what we want when GRPO
on a Qwen3-4B-Instruct hits the WikiSQL ceiling early.

This module is the BIRD analogue of :mod:`train.env.wikisql`.

Wiring:

* :class:`BirdCatalog` -- a :class:`CatalogProvider` that loads only
  the databases referenced by the sampled question subset, materializes
  each SQLite table into a Polars DataFrame with **sanitized column
  names**, and namespaces tables as ``<db_id>__<table>`` so the global
  catalog stays unique. Memoized like :class:`WikiSqlCatalog` so the
  same instance can be shared across multiple
  :class:`DialectRuntime` workers.

* :class:`BirdTaskGenerator` -- emits one :class:`SqlTask` per
  question. The user prompt embeds the relevant DB's schema (sanitized
  + original column headers, types, sample rows for each table) and
  the BIRD ``evidence`` field so the model has the same context a
  human BI analyst would. Gold rows are computed by running the
  original BIRD ``SQL`` field through stdlib ``sqlite3`` against the
  source ``.sqlite`` file -- column-name comparison is column-name-
  insensitive (see :func:`eval.validator.compare_results`), so we never
  need to rewrite the gold SQL to use sanitized identifiers.

Data sources:

* Questions: HuggingFace ``birdsql/bird23-train-filtered`` (6,601
  curated train rows; "filtered" drops the ~30% of original BIRD
  examples that had schema/answer drift) or
  ``birdsql/bird_sql_dev_20251106`` (1,534 dev rows; the
  community-reviewed update).
* Database files: NOT on HuggingFace. The train DB pack ships as a
  zip on a Beijing OSS bucket (``train.zip`` ~5GB unpacked, public
  URL); the dev DB pack ships on Google Drive (no direct URL,
  manual download). We auto-download the train zip on first use into
  ``~/.cache/manysql/bird/<split>/`` with progress bars; for dev we
  fail-fast with an actionable error pointing the user at the GDrive
  link unless ``--bird-db-dir`` is set explicitly.

Loading the dataset requires ``datasets>=2.20``. Database loading
needs only stdlib ``sqlite3``. Both are imported lazily so this
module is import-cheap on stripped-down environments.

Why we don't rewrite the gold SQL:

The model writes SQL against tables with **sanitized** column names
(``c_free_meal_count_k_12``); the BIRD gold SQL uses the
**original** backtick-quoted names (``` `Free Meal Count (K-12)` ```).
Two SQL strings, two row sets, but the row-comparison metric is
column-name-insensitive (it canonicalizes by sorted-tuple of values),
so as long as the values agree the candidate matches the gold. This
saves us from the SQL rewriting tarpit -- BIRD gold SQL uses
SQLite-specific functions (``STRFTIME``, ``IIF``, ``JULIANDAY``) that
generic SQL parsers struggle with.
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from train.env._ident import dedupe_columns, safe_ident, safe_table_name
from train.env.catalog import CatalogProvider, CatalogSnapshot
from train.env.tasks import SqlTask, TaskGenerator
from train.env.types import TaskMeta

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HF dataset names. The 20251106 dev release is the latest community-
# reviewed snapshot; bird23-train-filtered is the quality-filtered train.
_HF_DATASET_TRAIN = "birdsql/bird23-train-filtered"
_HF_DATASET_DEV = "birdsql/bird_sql_dev_20251106"

# DB-pack download URLs. Only the train zip has a public direct link.
# Dev redirects to a GDrive page; we don't try to script through it.
_TRAIN_DB_ZIP_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"
_DEV_DB_GDRIVE_HINT = (
    "https://drive.google.com/file/d/13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG/view"
)

_DEFAULT_DIFFICULTIES: tuple[str, ...] = ("simple", "moderate")
_VALID_DIFFICULTIES: frozenset[str] = frozenset(
    ("simple", "moderate", "challenging")
)

# Safety defaults. ~500 MB / DB excludes the long tail (donor=4.5GB,
# synthea, european_football_2 with attachments) but keeps every dev DB
# and most of the train pack. ~200k rows / table mirrors what fits
# comfortably in Polars on a 32 GB box and matches BIRD's typical
# question scale -- almost all questions filter or aggregate to far
# fewer than 200k rows of input.
_DEFAULT_MAX_DB_BYTES: int = 500 * 1024 * 1024  # 500 MB
_DEFAULT_MAX_ROWS_PER_TABLE: int = 200_000


# ---------------------------------------------------------------------------
# Materialized BIRD example
# ---------------------------------------------------------------------------


@dataclass
class BirdEntry:
    """One materialized BIRD example (post-sanitization).

    Held by :class:`BirdCatalog` after ``build()``; consumed by
    :class:`BirdTaskGenerator` to render the per-task prompt and look
    up the matching gold rows.

    ``tables`` is the per-task schema view (db-scoped) used for prompt
    rendering; the global :class:`CatalogSnapshot` carries the actual
    DataFrames keyed by ``catalog_table_name``.
    """

    question_id: int
    db_id: str
    question: str
    evidence: str
    sql: str
    difficulty: str
    db_path: str
    tables: list["BirdTableInfo"]


@dataclass
class BirdTableInfo:
    """Per-table view of a BIRD database, sanitized for prompt + catalog."""

    original_name: str
    catalog_table_name: str  # `<db_id>__<table>` (lowercased, deduped)
    safe_columns: list[str]   # sanitized identifiers (`c_*`)
    original_columns: list[str]
    types: list[str]          # IR type names (INT, FLOAT, TEXT, BOOL, DATE)
    # Original SQLite type declarations (e.g. "DATETIME", "DATE",
    # "VARCHAR(255)"). Most BIRD DBs declare DATE/DATETIME columns
    # but SQLite stores them as TEXT under the hood, so the IR-level
    # ``types`` collapses them to "TEXT". Surfacing the original
    # affinity in the prompt lets the model know "this TEXT column
    # actually holds date strings; CAST before EXTRACT/DATE_DIFF".
    sqlite_types: list[str] = field(default_factory=list)
    sample_rows: list[dict[str, Any]] = field(default_factory=list)  # using safe_columns as keys
    n_rows: int = 0


# ---------------------------------------------------------------------------
# BIRD catalog
# ---------------------------------------------------------------------------


class BirdCatalog(CatalogProvider):
    """Catalog provider backed by N BIRD-SQL examples.

    Loads the HF question subset first, then walks the union of
    referenced ``db_id`` values and pulls each one's tables out of the
    matching ``.sqlite`` file via stdlib ``sqlite3``. Tables are
    namespaced ``<db_id>__<table>`` so the global table set stays
    unique; column names are sanitized to ``c_<lowercase_alnum>``
    (with dedup suffixes on collision) so they parse in every
    dialect's grammar.

    Memoization: ``build()`` caches the snapshot. Reuse the same
    instance across multiple :class:`DialectRuntime` workers so we
    don't materialize ~5GB of SQLite data per dialect in cross-product
    multi-dialect runs.

    Database files:

    * If ``db_dir`` is provided, look there. Each DB lives at
      ``<db_dir>/<db_id>/<db_id>.sqlite``.
    * Else if ``auto_download`` is True (default for the train split),
      download + unzip the official train pack into
      ``~/.cache/manysql/bird/<split>/`` on first use.
    * Else raise with a clear pointer to the manual download URL.
    """

    name = "bird"

    def __init__(
        self,
        *,
        n_samples: int = 1000,
        split: str = "train",
        seed: int = 0,
        difficulties: tuple[str, ...] = _DEFAULT_DIFFICULTIES,
        db_dir: str | None = None,
        auto_download: bool = True,
        sample_rows: int = 3,
        max_table_value_chars: int = 40,
        max_db_bytes: int | None = _DEFAULT_MAX_DB_BYTES,
        max_rows_per_table: int | None = _DEFAULT_MAX_ROWS_PER_TABLE,
    ) -> None:
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if split not in {"train", "dev"}:
            raise ValueError(f"split must be 'train' | 'dev', got {split!r}")
        bad_diff = set(difficulties) - _VALID_DIFFICULTIES
        if bad_diff:
            raise ValueError(
                f"unknown difficulty levels {sorted(bad_diff)}; "
                f"valid: {sorted(_VALID_DIFFICULTIES)}"
            )
        if sample_rows < 0:
            raise ValueError(f"sample_rows must be non-negative, got {sample_rows}")
        # Treat 0 / negative as "disabled" for ergonomic CLI parsing
        # (argparse can't represent "None" cleanly).
        if max_db_bytes is not None and max_db_bytes <= 0:
            max_db_bytes = None
        if max_rows_per_table is not None and max_rows_per_table <= 0:
            max_rows_per_table = None

        self.n_samples = n_samples
        self.split = split
        self.seed = seed
        self.difficulties = tuple(difficulties)
        self.db_dir_override = db_dir
        self.auto_download = auto_download
        self.sample_rows_n = sample_rows
        self.max_table_value_chars = max_table_value_chars
        self.max_db_bytes = max_db_bytes
        self.max_rows_per_table = max_rows_per_table

        self._snapshot: CatalogSnapshot | None = None
        self._entries: list[BirdEntry] = []
        self._db_dir: Path | None = None
        # Per-question db_path overrides, populated when a question's
        # source DB gets mirrored to a sampled copy. Used so gold SQL
        # runs against the same row subset Polars sees.
        self._effective_db_path: dict[str, Path] = {}

    # -- public API --

    def build(self) -> CatalogSnapshot:
        if self._snapshot is not None:
            return self._snapshot

        import polars as pl  # noqa: PLC0415

        from manysql.ir.plan import ColumnSchema  # noqa: PLC0415

        questions = self._load_questions()
        sampled = self._sample_questions(questions)
        self._db_dir = self._resolve_db_dir({q["db_id"] for q in sampled})

        # First pass: load the union of referenced DBs once each, into
        # a {db_id: {table_name: (df, safe_cols, orig_cols, types)}} map.
        # Skips DBs whose .sqlite file is missing instead of crashing.
        # Also skips DBs whose on-disk size exceeds ``max_db_bytes``
        # (the long tail of BIRD has 1-5GB DBs that OOM Polars in-mem
        # loading); records the per-DB effective path (sampled mirror
        # if ``max_rows_per_table`` is set).
        per_db: dict[
            str,
            dict[str, tuple[Any, list[str], list[str], list[str], list[str]]],
        ] = {}
        missing_dbs: set[str] = set()
        oversized_dbs: list[tuple[str, int]] = []
        sampled_dbs: list[tuple[str, int]] = []  # (db_id, max_orig_table_rows)
        per_db_effective_path: dict[str, Path] = {}
        for db_id in sorted({q["db_id"] for q in sampled}):
            db_path = self._db_path_for(db_id)
            if db_path is None:
                missing_dbs.add(db_id)
                continue
            if self.max_db_bytes is not None:
                try:
                    size = db_path.stat().st_size
                except OSError:
                    size = 0
                if size > self.max_db_bytes:
                    oversized_dbs.append((db_id, size))
                    missing_dbs.add(db_id)
                    continue
            effective_path = db_path
            if self.max_rows_per_table is not None:
                effective_path, max_orig = _ensure_sampled_db(
                    db_path,
                    db_id=db_id,
                    split=self.split,
                    max_rows_per_table=self.max_rows_per_table,
                )
                if max_orig > self.max_rows_per_table:
                    sampled_dbs.append((db_id, max_orig))
            try:
                per_db[db_id] = _load_sqlite_to_polars(effective_path)
                per_db_effective_path[db_id] = effective_path
            except Exception as exc:
                print(
                    f"[bird] WARN: failed to load DB {db_id!r} from "
                    f"{effective_path}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                missing_dbs.add(db_id)
        if oversized_dbs:
            preview = ", ".join(
                f"{db_id}={_human_bytes(s)}"
                for db_id, s in sorted(
                    oversized_dbs, key=lambda x: -x[1]
                )[:5]
            )
            print(
                f"[bird] WARN: skipped {len(oversized_dbs)} DB(s) "
                f"exceeding --bird-max-db-bytes "
                f"({_human_bytes(self.max_db_bytes)}): {preview}"
                f"{'...' if len(oversized_dbs) > 5 else ''}; questions "
                f"referencing them will be dropped",
                file=sys.stderr,
            )
        if sampled_dbs:
            preview = ", ".join(
                f"{db_id}({n})" for db_id, n in sorted(
                    sampled_dbs, key=lambda x: -x[1]
                )[:5]
            )
            print(
                f"[bird] INFO: {len(sampled_dbs)} DB(s) had tables "
                f"capped to {self.max_rows_per_table} rows "
                f"(largest source-table sizes: {preview}"
                f"{'...' if len(sampled_dbs) > 5 else ''}); gold SQL is "
                f"re-executed on the sampled mirror so model outputs "
                f"and gold rows stay aligned.",
                file=sys.stderr,
            )
        if missing_dbs:
            still_missing = sorted(missing_dbs - {d for d, _ in oversized_dbs})
            if still_missing:
                print(
                    f"[bird] WARN: {len(still_missing)} DB(s) missing or "
                    f"unreadable; questions referencing them will be "
                    f"dropped: {still_missing[:5]}"
                    f"{'...' if len(still_missing) > 5 else ''}",
                    file=sys.stderr,
                )
        self._effective_db_path = {
            db_id: path for db_id, path in per_db_effective_path.items()
        }

        # Build the global Polars catalog and per-table info.
        tables: dict[str, Any] = {}
        schemas: dict[str, tuple[ColumnSchema, ...]] = {}
        per_db_table_info: dict[str, dict[str, BirdTableInfo]] = {}
        for db_id, table_map in per_db.items():
            per_db_table_info[db_id] = {}
            db_part = safe_table_name(db_id, fallback="db")
            for tbl_name, (
                df,
                safe_cols,
                orig_cols,
                types,
                sqlite_types,
            ) in table_map.items():
                # Build the catalog name from independently-sanitized
                # parts joined with ``__`` so the separator survives
                # (a single ``safe_table_name(f"{db}__{t}")`` would
                # collapse ``__`` to ``_`` along with any other
                # non-alnum runs, hiding the boundary in the prompt).
                tbl_part = safe_table_name(tbl_name, fallback="tbl")
                catalog_name = f"{db_part}__{tbl_part}"
                if catalog_name in tables:
                    suffix = 1
                    while f"{catalog_name}_{suffix}" in tables:
                        suffix += 1
                    catalog_name = f"{catalog_name}_{suffix}"

                tables[catalog_name] = df
                schemas[catalog_name] = tuple(
                    ColumnSchema(name=c, type=_ir_type_for(t))
                    for c, t in zip(safe_cols, types, strict=False)
                )

                sample = (
                    df.head(self.sample_rows_n).to_dicts()
                    if self.sample_rows_n > 0
                    else []
                )
                per_db_table_info[db_id][tbl_name] = BirdTableInfo(
                    original_name=tbl_name,
                    catalog_table_name=catalog_name,
                    safe_columns=safe_cols,
                    original_columns=orig_cols,
                    types=types,
                    sqlite_types=list(sqlite_types),
                    sample_rows=sample,
                    n_rows=df.height,
                )

        # Second pass: per-question entries (one per task), only for
        # questions whose DB loaded successfully. Each entry's
        # ``db_path`` points at the *effective* DB (sampled mirror if
        # ``max_rows_per_table`` triggered, original otherwise) so the
        # gold-SQL run in :meth:`BirdTaskGenerator.build` evaluates
        # against the same row subset Polars holds.
        entries: list[BirdEntry] = []
        for q in sampled:
            if q["db_id"] not in per_db_table_info:
                continue
            tbl_views = list(per_db_table_info[q["db_id"]].values())
            if not tbl_views:
                continue
            db_path = self._effective_db_path.get(q["db_id"])
            if db_path is None:
                continue
            entries.append(
                BirdEntry(
                    question_id=int(q["question_id"]),
                    db_id=q["db_id"],
                    question=q["question"],
                    evidence=q.get("evidence") or "",
                    sql=q["SQL"],
                    difficulty=q.get("difficulty") or "",
                    db_path=str(db_path),
                    tables=tbl_views,
                )
            )
        # Stable ordering: by question_id, so seeds are reproducible
        # downstream.
        entries.sort(key=lambda e: e.question_id)

        self._entries = entries
        self._snapshot = CatalogSnapshot(
            tables=tables,
            schemas=schemas,
            schema_prompt=_BIRD_SCHEMA_PROMPT,
        )
        # Polars DataFrames live until the catalog is GC'd. ~5-100 MB
        # per BIRD DB, ~1 GB worst case for an 80-DB fan-out.
        return self._snapshot

    def entries(self) -> list[BirdEntry]:
        if self._snapshot is None:
            self.build()
        return list(self._entries)

    # -- HF loading --

    def _load_questions(self) -> list[dict[str, Any]]:
        """Pull the BIRD HF dataset and return rows as plain dicts.

        Filters by ``self.difficulties`` here so the random subset
        sampling sees only the eligible pool. Tries the configured HF
        repo first; on failure raises with a clear message.
        """
        from datasets import load_dataset  # noqa: PLC0415

        if self.split == "train":
            ds_name = _HF_DATASET_TRAIN
            split_name = "train"
        else:
            ds_name = _HF_DATASET_DEV
            split_name = "dev_20251106"  # custom split name in the new release

        ds = load_dataset(ds_name, split=split_name)
        rows: list[dict[str, Any]] = []
        for idx, r in enumerate(ds):
            difficulty = r.get("difficulty") or "simple"
            if difficulty not in self.difficulties:
                continue
            row = dict(r)
            # Newer ``birdsql/bird23-train-filtered`` releases ship only
            # ``db_id, question, evidence, SQL`` -- no ``question_id``.
            # Synthesize a stable id from the dataset row index so
            # downstream task_id construction and seeded sampling
            # remain reproducible.
            row.setdefault("question_id", idx)
            row.setdefault("difficulty", difficulty)
            rows.append(row)
        return rows

    def _sample_questions(self, pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Reproducible random subset of ``self.n_samples`` rows."""
        import random  # noqa: PLC0415

        n = min(self.n_samples, len(pool))
        rng = random.Random(self.seed)
        idxs = rng.sample(range(len(pool)), n)
        return [pool[i] for i in idxs]

    # -- DB filesystem layout --

    def _resolve_db_dir(self, referenced_db_ids: set[str]) -> Path:
        """Find or fetch the directory holding ``<db_id>/<db_id>.sqlite``.

        Resolution order:
            1. ``self.db_dir_override`` if set.
            2. ``$BIRD_DB_DIR`` env var if set (split is appended:
               ``$BIRD_DB_DIR/<split>/``).
            3. ``~/.cache/manysql/bird/<split>/`` if it already
               exists from a prior run.
            4. Auto-download (train only, when ``auto_download``).

        Raises with an actionable message if none of the above work.
        """
        candidates: list[Path] = []
        if self.db_dir_override:
            candidates.append(Path(self.db_dir_override).expanduser())
        env_root = os.environ.get("BIRD_DB_DIR")
        if env_root:
            candidates.append(Path(env_root).expanduser() / self.split)
            candidates.append(Path(env_root).expanduser())
        cache_root = Path.home() / ".cache" / "manysql" / "bird" / self.split
        candidates.append(cache_root)

        for cand in candidates:
            if _looks_like_bird_db_dir(cand, referenced_db_ids):
                return cand

        # Last resort: auto-download (train only).
        if self.split == "train" and self.auto_download:
            cache_root.mkdir(parents=True, exist_ok=True)
            _download_and_extract_train_dbs(
                cache_root,
                referenced_db_ids=referenced_db_ids,
                max_db_bytes=self.max_db_bytes,
            )
            if _looks_like_bird_db_dir(cache_root, referenced_db_ids):
                return cache_root
            # The download succeeded but doesn't have the layout we
            # expect. Probably a future BIRD repackaging; surface the
            # current directory listing so the user can debug.
            raise RuntimeError(
                f"BIRD train auto-download finished but {cache_root} "
                f"doesn't match the expected layout "
                f"(<db_id>/<db_id>.sqlite). Top-level entries: "
                f"{sorted(p.name for p in cache_root.iterdir())[:20]}"
            )

        # Nothing worked.
        if self.split == "dev":
            raise RuntimeError(
                "BIRD dev databases not found and auto-download is not "
                "supported for the dev split (Google Drive blocks "
                "scripted access). Download manually from\n"
                f"  {_DEV_DB_GDRIVE_HINT}\n"
                "and unzip into ~/.cache/manysql/bird/dev/ "
                "(or pass --bird-db-dir / set $BIRD_DB_DIR)."
            )
        raise RuntimeError(
            "BIRD train databases not found and auto-download is "
            "disabled. Either pass --bird-db-dir / set $BIRD_DB_DIR, "
            f"or re-run with auto_download=True (will fetch ~5GB from "
            f"{_TRAIN_DB_ZIP_URL})."
        )

    def _db_path_for(self, db_id: str) -> Path | None:
        if self._db_dir is None:
            return None
        # Standard BIRD layout: <root>/<db_id>/<db_id>.sqlite.
        candidate = self._db_dir / db_id / f"{db_id}.sqlite"
        if candidate.is_file():
            return candidate
        # Some older packings name the file <db_id>.db; try that too.
        alt = self._db_dir / db_id / f"{db_id}.db"
        if alt.is_file():
            return alt
        return None


def _looks_like_bird_db_dir(path: Path, referenced_db_ids: set[str]) -> bool:
    """True iff at least half of the referenced DBs live under ``path``.

    We tolerate a few missing DBs (BIRD splits sometimes ship with
    minor file-naming inconsistencies) but a directory that has *none*
    of the expected DBs is almost certainly the wrong root, and we'd
    rather try the next candidate than silently load 0 tasks.
    """
    if not path.is_dir():
        return False
    if not referenced_db_ids:
        return any((path / sub).is_dir() for sub in path.iterdir())  # any DB
    hits = 0
    for db_id in referenced_db_ids:
        if (path / db_id / f"{db_id}.sqlite").is_file():
            hits += 1
        elif (path / db_id / f"{db_id}.db").is_file():
            hits += 1
    return hits >= max(1, len(referenced_db_ids) // 2)


# ---------------------------------------------------------------------------
# SQLite -> Polars
# ---------------------------------------------------------------------------


def _load_sqlite_to_polars(
    db_path: Path,
) -> dict[str, tuple[Any, list[str], list[str], list[str], list[str]]]:
    """Load every user table in ``db_path`` into a Polars DataFrame.

    Returns ``{table_name: (df, safe_cols, orig_cols, ir_type_names,
    sqlite_type_decls)}``. Skips ``sqlite_*`` system tables and views.

    Type-mapping: SQLite affinities are mapped to manysql IR type
    names ("INT" / "FLOAT" / "TEXT" / "DATE" / "BOOL"). DATE is
    detected heuristically from column names ending in ``_date`` /
    starting with ``date_`` etc; BIRD doesn't carry a real DATE
    affinity (SQLite stores dates as TEXT). We default to TEXT for
    anything ambiguous. The original SQLite type declaration string
    is also returned so the per-task prompt can surface "this TEXT
    column was originally declared DATETIME" -- otherwise the model
    has no signal that a Utf8 column actually holds date strings.
    """
    import polars as pl  # noqa: PLC0415

    out: dict[str, tuple[Any, list[str], list[str], list[str], list[str]]] = {}
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
        table_names = [r[0] for r in cur.fetchall()]

        for tbl in table_names:
            cur.execute(f'PRAGMA table_info("{tbl}")')
            cols_info = cur.fetchall()  # (cid, name, type, notnull, dflt, pk)
            if not cols_info:
                continue
            orig_cols = [c[1] for c in cols_info]
            sqlite_types = [c[2] or "" for c in cols_info]
            ir_type_names = [_sqlite_to_ir_type(t) for t in sqlite_types]
            safe_cols = dedupe_columns(
                [safe_ident(c, fallback=f"col{i}") for i, c in enumerate(orig_cols)]
            )

            # Read all rows. BIRD DBs are small enough (<=100k rows
            # per table for the largest); pandas-or-arrow streaming is
            # overkill.
            quoted_cols = ", ".join(f'"{c}"' for c in orig_cols)
            cur.execute(f'SELECT {quoted_cols} FROM "{tbl}"')
            rows = cur.fetchall()

            # Build columnar Python lists, then hand to Polars with
            # explicit dtypes so a column that looks numeric in
            # PRAGMA but contains stray strings doesn't blow up
            # construction. ``strict=False`` semantics: bad cells
            # become null.
            col_data: list[list[Any]] = [[] for _ in orig_cols]
            for row in rows:
                for i, v in enumerate(row):
                    col_data[i].append(v)

            series: list[Any] = []
            for name, ir_t, raw in zip(safe_cols, ir_type_names, col_data, strict=False):
                if ir_t == "INT":
                    series.append(_build_int_series(name, raw))
                elif ir_t == "FLOAT":
                    series.append(_build_float_series(name, raw))
                else:
                    # TEXT / DATE / BOOL -- store as TEXT in Polars,
                    # let the dialect engine cast at query time. BIRD
                    # SQL routinely uses CAST(... AS REAL) anyway.
                    series.append(
                        pl.Series(
                            name,
                            [None if v is None else str(v) for v in raw],
                            dtype=pl.Utf8,
                        )
                    )
            df = pl.DataFrame(series) if series else pl.DataFrame()
            out[tbl] = (df, safe_cols, orig_cols, ir_type_names, sqlite_types)
    finally:
        conn.close()
    return out


def _human_bytes(n: int) -> str:
    """Compact size formatter that picks a unit so small thresholds
    (e.g. 150KB in tests) don't render as ``0MB``."""
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.0f}KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / 1024 / 1024:.0f}MB"
    return f"{n / 1024 / 1024 / 1024:.1f}GB"


def _ensure_sampled_db(
    src_path: Path,
    *,
    db_id: str,
    split: str,
    max_rows_per_table: int,
) -> tuple[Path, int]:
    """Materialize a row-capped mirror of ``src_path`` if it has any
    table over ``max_rows_per_table``. Cached on disk so subsequent
    runs reuse it.

    Returns ``(effective_path, max_source_table_rows)``:

    * ``effective_path`` is the sampled-mirror file when capping was
      applied, otherwise ``src_path`` itself (no copy made; saves
      disk on already-small DBs like every BIRD dev DB).
    * ``max_source_table_rows`` is the largest source-table row count
      observed; the caller uses it to log which DBs got capped.

    Mirror layout::

        ~/.cache/manysql/bird/_sampled_r<N>/<split>/<db_id>/<db_id>.sqlite

    The mirror is built by:

    1. Counting rows in every user table in the source DB.
    2. If the max row count is <= ``max_rows_per_table``, returning
       ``src_path`` unchanged.
    3. Otherwise opening a fresh SQLite file, copying the schema
       (``sqlite_master``) verbatim, then INSERT-SELECT'ing each
       table with ``ORDER BY ROWID LIMIT max_rows_per_table``.
       Indexes are skipped -- BIRD gold SQL runs full scans either
       way at our scales.

    Determinism: ``ORDER BY ROWID`` is reproducible across runs as
    long as the source DB doesn't change. We *don't* randomize
    sampling because gold SQL must be re-executed against the same
    rows the model sees, and a deterministic key keeps that easy.
    Bias toward early rows is acceptable for an RL signal.

    Note on referential integrity: per-table row caps applied
    independently CAN produce empty joins (e.g. if a foreign key
    points at a row that lands past ``LIMIT``). Tasks that join
    over capped tables and end up with empty result sets get
    filtered downstream by ``BirdTaskConfig.drop_empty``.
    """
    sizes = _table_row_counts(src_path)
    if not sizes:
        return src_path, 0
    max_orig = max(sizes.values())
    if max_orig <= max_rows_per_table:
        return src_path, max_orig

    cache_root = (
        Path.home() / ".cache" / "manysql" / "bird" /
        f"_sampled_r{max_rows_per_table}" / split / db_id
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    dest = cache_root / f"{db_id}.sqlite"
    if dest.is_file():
        return dest, max_orig
    tmp = dest.with_suffix(".sqlite.partial")
    if tmp.exists():
        tmp.unlink()
    print(
        f"[bird] capping {db_id} to {max_rows_per_table} rows/table "
        f"(largest source table = {max_orig} rows); building "
        f"sampled mirror at {dest}",
        file=sys.stderr,
    )
    _build_sampled_db(
        src_path=src_path,
        dest_path=tmp,
        max_rows_per_table=max_rows_per_table,
        table_sizes=sizes,
    )
    tmp.replace(dest)
    return dest, max_orig


def _table_row_counts(db_path: Path) -> dict[str, int]:
    """Return ``{table_name: row_count}`` for every user table."""
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        names = [r[0] for r in cur.fetchall()]
        out: dict[str, int] = {}
        for n in names:
            try:
                cur.execute(f'SELECT COUNT(*) FROM "{n}"')
                out[n] = int(cur.fetchone()[0])
            except sqlite3.Error:
                # Corrupt or virtual table -- skip silently; the
                # downstream load_sqlite_to_polars will surface a
                # clearer error if it actually needs the table.
                continue
        return out
    finally:
        conn.close()


def _build_sampled_db(
    *,
    src_path: Path,
    dest_path: Path,
    max_rows_per_table: int,
    table_sizes: dict[str, int],
) -> None:
    """Copy the schema and (capped) row contents of ``src_path`` to
    ``dest_path`` using SQLite's ``ATTACH DATABASE`` so the heavy
    lifting stays inside the C engine (no row materialization in
    Python).
    """
    src = sqlite3.connect(str(src_path))
    try:
        src.row_factory = None
        cur = src.cursor()
        # Pull every user table's CREATE TABLE statement. Skip views
        # (we don't carry them through; gold SQL on BIRD never targets
        # views) and indexes (rebuilt on demand by SQLite if anyone
        # actually issues a query that benefits, which our scans
        # mostly don't).
        cur.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "AND sql IS NOT NULL"
        )
        ddl = cur.fetchall()
        # ATTACH the dest DB to the source connection so we can do
        # cross-DB INSERT-SELECT in one statement.
        cur.execute(f"ATTACH DATABASE '{dest_path}' AS dest")
        try:
            for name, create_sql in ddl:
                # Rewrite the CREATE TABLE to target the attached DB.
                # Most BIRD CREATE TABLEs start with `CREATE TABLE
                # "<name>"` or `CREATE TABLE <name>`; using a regex
                # to splice in `dest.` after CREATE TABLE keeps
                # column / constraint definitions intact.
                rewritten = _rewrite_create_for_dest(create_sql, name)
                cur.execute(rewritten)
                n = table_sizes.get(name, 0)
                if n == 0:
                    continue
                limit = min(n, max_rows_per_table)
                # ORDER BY ROWID is the cheap deterministic key. For
                # tables without explicit ROWID (WITHOUT ROWID), this
                # falls back to insertion order via the PK -- BIRD
                # uses regular ROWID-having tables almost universally.
                try:
                    cur.execute(
                        f'INSERT INTO dest."{name}" '
                        f'SELECT * FROM main."{name}" '
                        f'ORDER BY ROWID LIMIT {limit}'
                    )
                except sqlite3.Error:
                    # WITHOUT ROWID fallback: just take an unordered
                    # subset.
                    cur.execute(
                        f'INSERT INTO dest."{name}" '
                        f'SELECT * FROM main."{name}" LIMIT {limit}'
                    )
            src.commit()
        finally:
            cur.execute("DETACH DATABASE dest")
    finally:
        src.close()


def _rewrite_create_for_dest(create_sql: str, table_name: str) -> str:
    """Rewrite a SQLite ``CREATE TABLE`` statement to target the
    attached ``dest`` database.

    Handles the common BIRD shapes:
        CREATE TABLE "<name>" (...)
        CREATE TABLE <name> (...)
        CREATE TABLE IF NOT EXISTS "<name>" (...)
    """
    # Find the table name in the CREATE statement; everything before
    # it is the prefix (CREATE TABLE [IF NOT EXISTS]); everything
    # after is the body. We splice ``dest.`` between the prefix and
    # the table name.
    pattern = re.compile(
        r"^(\s*CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?)"
        r'("?)' + re.escape(table_name) + r'("?)',
        re.IGNORECASE,
    )
    m = pattern.match(create_sql)
    if not m:
        # Fallback: replace first occurrence of the table name with
        # dest."name". Less robust but covers oddly-formatted DDL.
        return create_sql.replace(
            table_name, f'dest."{table_name}"', 1
        )
    return (
        m.group(1) + f'dest."{table_name}"' + create_sql[m.end():]
    )


def _build_int_series(name: str, raw: list[Any]) -> Any:
    import polars as pl  # noqa: PLC0415

    coerced: list[int | None] = []
    bad = False
    for v in raw:
        if v is None or v == "":
            coerced.append(None)
            continue
        if isinstance(v, bool):
            coerced.append(int(v))
            continue
        try:
            coerced.append(int(v))
        except (TypeError, ValueError):
            try:
                # Some BIRD INT columns sneak in floats like 3.0.
                coerced.append(int(float(v)))
            except (TypeError, ValueError):
                coerced.append(None)
                bad = True
    if bad:
        # Fall back to TEXT silently if the column is hopelessly
        # non-numeric -- safer than a build-time crash on one bad cell.
        return pl.Series(
            name,
            [None if v is None else str(v) for v in raw],
            dtype=pl.Utf8,
        )
    return pl.Series(name, coerced, dtype=pl.Int64)


def _build_float_series(name: str, raw: list[Any]) -> Any:
    import polars as pl  # noqa: PLC0415

    coerced: list[float | None] = []
    for v in raw:
        if v is None or v == "":
            coerced.append(None)
            continue
        if isinstance(v, bool):
            coerced.append(float(v))
            continue
        try:
            coerced.append(float(v))
        except (TypeError, ValueError):
            coerced.append(None)
    return pl.Series(name, coerced, dtype=pl.Float64)


def _sqlite_to_ir_type(sqlite_type: str) -> str:
    """Map a SQLite affinity declaration to a manysql IR type name.

    SQLite uses 5 affinities (TEXT, NUMERIC, INTEGER, REAL, BLOB) but
    BIRD DBs declare a wider variety of pseudo-types ("VARCHAR",
    "DATETIME", ...). We follow SQLite's affinity-determination rules
    plus a few BIRD-specific shortcuts.
    """
    s = (sqlite_type or "").upper()
    if not s:
        return "TEXT"
    if "INT" in s:
        return "INT"
    if "REAL" in s or "FLOA" in s or "DOUB" in s or "NUM" in s or "DEC" in s:
        return "FLOAT"
    if "BOOL" in s:
        return "BOOL"
    if "DATE" in s or "TIME" in s:
        # BIRD/SQLite stores dates/times as TEXT under the hood.
        # Tagging these as DATE in the IR would force the lowering to
        # validate parse-able formats, which is a different rabbit
        # hole. Keep them as TEXT and let the SQL handle conversion.
        return "TEXT"
    if "CHAR" in s or "TEXT" in s or "CLOB" in s or "STRING" in s:
        return "TEXT"
    # Unknown affinity. Default to TEXT as the most permissive.
    return "TEXT"


def _ir_type_for(name: str) -> Any:
    from manysql.ir.types import BOOL, DATE_T, FLOAT, INT, TEXT  # noqa: PLC0415

    return {
        "INT": INT,
        "FLOAT": FLOAT,
        "TEXT": TEXT,
        "BOOL": BOOL,
        "DATE": DATE_T,
    }.get(name, TEXT)


# ---------------------------------------------------------------------------
# Auto-download (train only)
# ---------------------------------------------------------------------------


def _download_and_extract_train_dbs(
    dest: Path,
    *,
    referenced_db_ids: set[str] | None = None,
    max_db_bytes: int | None = None,
) -> None:
    """Fetch ``train.zip`` (~5GB) into ``dest`` and selectively unzip.

    Uses ``urllib.request`` so we don't pull a heavy dep just for one
    download. Streams to a temp file with a periodic-progress print so
    the user knows something's happening.

    Selective extraction (disk-saver, important on GPU nodes that
    can't afford the full ~30GB unpacked tree):

    * If ``referenced_db_ids`` is non-empty, only extract files
      under ``train/train_databases/<db_id>/...`` for those ids.
    * If ``max_db_bytes`` is set, skip any ``.sqlite`` whose
      uncompressed ``ZipInfo.file_size`` exceeds the cap (saves the
      4.5 GB ``donor`` file etc. from ever touching disk).

    Re-entrancy: the function early-returns if every referenced
    db_id already has its ``.sqlite`` file extracted under
    ``dest``, so subsequent runs with the same sample don't
    re-download. Runs with a *different* sample whose new db_ids
    aren't already on disk will trigger a fresh download +
    selective re-extract.

    The zip itself is deleted after a successful extraction since
    most users won't change their referenced-id set on every run.
    Set ``$BIRD_KEEP_ZIP=1`` to keep it (helpful for fast iteration
    on the referenced-id set).
    """
    refs = referenced_db_ids or set()
    # If every referenced DB is already extracted, nothing to do.
    if refs and all(
        (dest / d / f"{d}.sqlite").is_file() for d in refs
    ):
        return
    zip_path = dest / "train.zip"
    if not zip_path.is_file():
        print(
            f"[bird] downloading {_TRAIN_DB_ZIP_URL} into {zip_path} "
            f"(this is a one-time ~5GB download)"
        )
        with urllib.request.urlopen(_TRAIN_DB_ZIP_URL) as resp:  # noqa: S310
            total = int(resp.headers.get("Content-Length") or 0)
            tmp = zip_path.with_suffix(".zip.partial")
            done = 0
            chunk = 1 << 20  # 1 MB
            last_pct = -1
            with tmp.open("wb") as out:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    out.write(buf)
                    done += len(buf)
                    if total > 0:
                        pct = int(done * 100 / total)
                        if pct != last_pct and pct % 5 == 0:
                            sys.stderr.write(
                                f"\r[bird] download {pct}% "
                                f"({done >> 20} / {total >> 20} MB)"
                            )
                            sys.stderr.flush()
                            last_pct = pct
            sys.stderr.write("\n")
            tmp.replace(zip_path)

    # Per-ZipInfo decision: extract or skip?
    def _wanted(info: zipfile.ZipInfo) -> bool:
        # Normalize path separators (zipfile always uses '/'). Two
        # observed BIRD train zip layouts:
        #   (a) flat:   ``train/train_databases/<db_id>/<db_id>.sqlite``
        #   (b) nested: ``train/train_databases.zip`` (an inner zip
        #       containing the same ``train_databases/<db_id>/...``
        #       tree). Current upstream uses (b); we stream the inner
        #       zip to a tempfile after this loop -- not into ``dest``
        #       -- so it doesn't compete with the per-db outputs for
        #       the workspace NFS quota.
        parts = info.filename.split("/")
        if parts and parts[-1] == "train_databases.zip":
            return False
        try:
            idx = parts.index("train_databases")
        except ValueError:
            # File not under train_databases (README, license, etc.).
            # Keep small ones, drop large ones.
            return info.file_size < (1 << 20)
        if idx + 1 >= len(parts):
            return False
        db_id = parts[idx + 1]
        if refs and db_id not in refs:
            return False
        if (
            max_db_bytes is not None
            and info.file_size > max_db_bytes
            and info.filename.endswith(".sqlite")
        ):
            return False
        return True

    skipped_oversized = 0
    skipped_unreferenced = 0
    extracted_count = 0
    extracted_bytes = 0
    print(
        f"[bird] extracting {zip_path} into {dest}"
        + (f" (filtering to {len(refs)} referenced db_id(s))" if refs else "")
        + (
            f" (skipping .sqlite > {_human_bytes(max_db_bytes)})"
            if max_db_bytes is not None
            else ""
        )
    )
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if _wanted(info):
                zf.extract(info, dest)
                extracted_count += 1
                extracted_bytes += info.file_size
                continue
            # Categorize the skip for the summary log.
            parts = info.filename.split("/")
            if (
                max_db_bytes is not None
                and info.file_size > max_db_bytes
                and info.filename.endswith(".sqlite")
            ):
                skipped_oversized += 1
            elif (
                refs
                and "train_databases" in parts
            ):
                skipped_unreferenced += 1
    print(
        f"[bird] extract done: {extracted_count} entries, "
        f"{_human_bytes(extracted_bytes)}; "
        f"skipped oversized={skipped_oversized}, "
        f"skipped unreferenced={skipped_unreferenced}"
    )

    # Layout (b): the outer zip contains a single nested
    # ``train_databases.zip`` rather than the per-db tree directly.
    # Stream that inner zip into a tempfile on /tmp (not into ``dest``,
    # which is on the workspace NFS volume with a tight per-user
    # quota), drop the now-redundant outer zip to reclaim ~8GB of
    # quota, then extract the per-db files we want into
    # ``train/train_databases/<db_id>/...`` so the flatten step below
    # picks them up unchanged.
    inner_zip_paths: list[Path] = []
    inner_zip_outer_names: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = info.filename.split("/")
            if not parts or parts[-1] != "train_databases.zip":
                continue
            tmp_fd, tmp_name = tempfile.mkstemp(
                prefix="bird_inner_", suffix=".zip"
            )
            os.close(tmp_fd)
            tmp_path = Path(tmp_name)
            print(
                f"[bird] streaming nested {info.filename} "
                f"into tempfile {tmp_path} ({_human_bytes(info.file_size)})"
            )
            with zf.open(info) as src, tmp_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            inner_zip_paths.append(tmp_path)
            inner_zip_outer_names.append(info.filename)

    # Drop the outer zip ASAP -- the inner zip(s) are safely on /tmp now,
    # and the per-db extraction below needs the workspace quota space.
    if inner_zip_paths and not os.environ.get("BIRD_KEEP_ZIP"):
        try:
            zip_path.unlink()
            print(
                f"[bird] removed {zip_path} early to free workspace "
                f"quota during inner-zip extraction"
            )
        except OSError:
            pass

    for inner_zip_path, outer_name in zip(inner_zip_paths, inner_zip_outer_names):
        # Mirror the layout the flatten step expects:
        # ``<dest>/<outer_dir>/train_databases/<db_id>/...``.
        outer_parts = outer_name.split("/")
        outer_dir = (
            dest.joinpath(*outer_parts[:-1]) if len(outer_parts) > 1 else dest
        )
        inner_target = outer_dir / "train_databases"
        inner_target.mkdir(parents=True, exist_ok=True)
        inner_extracted = 0
        inner_bytes = 0
        inner_skipped_oversized = 0
        inner_skipped_unreferenced = 0
        print(f"[bird] unpacking nested {outer_name} from {inner_zip_path}")
        with zipfile.ZipFile(inner_zip_path) as inner_zf:
            for info in inner_zf.infolist():
                if info.is_dir():
                    continue
                inner_parts = info.filename.split("/")
                # Locate the db_id: typical layout is
                # ``train_databases/<db_id>/...`` but some packings
                # drop the leading ``train_databases/`` segment.
                try:
                    inner_idx = inner_parts.index("train_databases")
                    db_id_segment = inner_idx + 1
                except ValueError:
                    db_id_segment = 0
                if db_id_segment >= len(inner_parts):
                    continue
                db_id = inner_parts[db_id_segment]
                if refs and db_id not in refs:
                    inner_skipped_unreferenced += 1
                    continue
                if (
                    max_db_bytes is not None
                    and info.file_size > max_db_bytes
                    and info.filename.endswith(".sqlite")
                ):
                    inner_skipped_oversized += 1
                    continue
                # Re-root onto ``<inner_target>/<db_id>/...`` so the
                # flatten step finds the expected directory tree.
                rel_parts = inner_parts[db_id_segment:]
                out_path = inner_target.joinpath(*rel_parts)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with inner_zf.open(info) as src, out_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                inner_extracted += 1
                inner_bytes += info.file_size
        print(
            f"[bird] inner extract done: {inner_extracted} entries, "
            f"{_human_bytes(inner_bytes)}; "
            f"skipped oversized={inner_skipped_oversized}, "
            f"skipped unreferenced={inner_skipped_unreferenced}"
        )
        try:
            inner_zip_path.unlink()
        except OSError:
            pass

    # If we didn't take the early-drop branch (BIRD_KEEP_ZIP=1, or no
    # inner zip was present), clean up the outer zip below as before.

    # The official zip has a top-level ``train/`` directory. Flatten
    # it so the layout we expect (``<db_id>/<db_id>.sqlite`` directly
    # under ``dest``) matches.
    nested = dest / "train" / "train_databases"
    flat = dest
    if nested.is_dir():
        for child in nested.iterdir():
            target = flat / child.name
            if not target.exists():
                shutil.move(str(child), str(target))
        # Clean up the now-empty ``train/`` top-level if possible.
        try:
            (nested.parent).rmdir()
            (nested.parent.parent / "train").rmdir()
        except OSError:
            pass

    if not os.environ.get("BIRD_KEEP_ZIP"):
        try:
            zip_path.unlink()
            print(
                f"[bird] removed {zip_path} to reclaim ~5GB disk "
                f"(set BIRD_KEEP_ZIP=1 to retain it)"
            )
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Schema prompt (system-prompt slot)
# ---------------------------------------------------------------------------

_BIRD_SCHEMA_PROMPT = (
    "Each task targets ONE BIRD-SQL database from the BIRD benchmark "
    "(real Kaggle-derived multi-table schemas, ~5-25 tables per DB). "
    "The relevant database's tables -- with sanitized column names "
    "(``c_<lowercase_alnum>``), original headers, types, and a few "
    "sample rows -- are provided in the user message; reference that "
    "block, not a global schema. Tables are namespaced "
    "``<db_id>__<table>`` so they're unique across the whole training "
    "corpus.\n\n"
    "DATE/TIME COLUMNS: SQLite stores dates and timestamps as TEXT, "
    "and the catalog preserves that storage shape -- so any column "
    "whose original SQLite type was DATE / DATETIME / TIMESTAMP is "
    "exposed as a TEXT column whose values are ISO-formatted strings "
    "(``YYYY-MM-DD`` or ``YYYY-MM-DD HH:MM:SS``). The per-table column "
    "listing tags these with ``(TEXT, was DATETIME)`` etc. so they "
    "are identifiable. To use ``EXTRACT`` / ``DATE_PART`` / "
    "``DATE_TRUNC`` / ``DATE_DIFF`` on such a column, CAST it first "
    "(e.g. ``EXTRACT(YEAR FROM CAST(c_hire_date AS DATE))``); without "
    "the CAST the engine will reject the call because the underlying "
    "type is still text. Direct string ordering (``c_hire_date >= "
    "'2020-01-01'``) works without a CAST because ISO format sorts "
    "lexicographically."
)


# ---------------------------------------------------------------------------
# Task generator
# ---------------------------------------------------------------------------


@dataclass
class BirdTaskConfig:
    """Configuration for :class:`BirdTaskGenerator`.

    Pass a pre-built ``catalog`` to share across multiple dialect
    runtimes (the recommended path for multi-dialect curricula).
    Otherwise the generator constructs its own with the same
    n_samples/split/seed/difficulties.

    ``drop_empty`` filters out tasks whose gold SQL returns zero rows.
    BIRD has a small fraction of such cases (often when a date filter
    excludes everything in the test slice); empty-result tasks are a
    poor training signal because "any SELECT that returns nothing"
    matches.

    Memory / disk safety:

    * ``max_db_bytes`` -- skip any source ``.sqlite`` larger than
      this (also used to filter zip entries during auto-download so
      huge DBs never touch disk). 0 / None disables the filter.
    * ``max_rows_per_table`` -- cap each loaded table at N rows.
      Triggers a per-DB sampled mirror under
      ``~/.cache/manysql/bird/_sampled_r<N>/<split>/<db_id>/``;
      both Polars catalog loading AND gold-SQL execution use the
      mirror so the model and the gold answer agree on the input
      rowset. 0 / None disables the cap.
    """

    target_dialect: str
    n_samples: int = 1000
    split: str = "train"  # 'train' | 'dev'
    seed: int = 0
    difficulties: tuple[str, ...] = _DEFAULT_DIFFICULTIES
    sample_rows: int = 3
    db_dir: str | None = None
    auto_download: bool = True
    catalog: BirdCatalog | None = None
    drop_empty: bool = True
    max_db_bytes: int | None = _DEFAULT_MAX_DB_BYTES
    max_rows_per_table: int | None = _DEFAULT_MAX_ROWS_PER_TABLE
    extra_meta: dict[str, Any] = field(default_factory=dict)


class BirdTaskGenerator(TaskGenerator):
    """Build NL->SQL tasks from BIRD-SQL.

    Each task pairs a natural-language question (plus the BIRD evidence
    field as a hint) with the relevant database's schema (in the user
    prompt) and an executable canonical gold SQL. Gold rows are
    computed once via stdlib ``sqlite3`` against the source ``.sqlite``
    file; the resulting :class:`SqlTask` list is dialect-tagged via
    ``config.target_dialect`` but the rows are dialect-independent --
    higher layers (e.g. :func:`train.grpo_sql.build_runtimes_and_tasks`)
    clone tasks across additional dialects without re-running SQLite.
    """

    name = "bird"

    def __init__(self, config: BirdTaskConfig) -> None:
        self.config = config
        self.catalog = config.catalog or BirdCatalog(
            n_samples=config.n_samples,
            split=config.split,
            seed=config.seed,
            difficulties=config.difficulties,
            db_dir=config.db_dir,
            auto_download=config.auto_download,
            sample_rows=config.sample_rows,
            max_db_bytes=config.max_db_bytes,
            max_rows_per_table=config.max_rows_per_table,
        )
        self._tasks: list[SqlTask] = []
        self._built = False

    def build(self) -> None:
        if self._built:
            return
        self.catalog.build()

        # Gold rows: open one sqlite3 connection per (DB) and reuse it
        # across all questions in that DB. Cheaper than reopening per
        # question.
        conns: dict[str, sqlite3.Connection] = {}
        try:
            for entry in self.catalog.entries():
                conn = conns.get(entry.db_path)
                if conn is None:
                    conn = sqlite3.connect(entry.db_path)
                    conns[entry.db_path] = conn
                rows = _execute_gold(conn, entry.sql)
                if rows is None:
                    # SQLite couldn't run the gold SQL at all (rare in
                    # the filtered train split, more common in dev).
                    # Drop the task; it'd be a 0-signal sample.
                    continue
                if self.config.drop_empty and not rows:
                    continue
                meta = TaskMeta(
                    task_id=f"bird_{entry.db_id}_{entry.question_id}",
                    dialect=self.config.target_dialect,
                    generator=self.name,
                    meta={
                        "db_id": entry.db_id,
                        "split": self.config.split,
                        "difficulty": entry.difficulty,
                        "n_tables": len(entry.tables),
                        "evidence_present": bool(entry.evidence),
                        **self.config.extra_meta,
                    },
                )
                prompt = _render_user_prompt(
                    entry, max_value_chars=self.catalog.max_table_value_chars
                )
                self._tasks.append(
                    SqlTask(
                        meta=meta,
                        prompt=prompt,
                        gold_rows=rows,
                        gold_sql=entry.sql,
                        catalog=self.catalog,
                        notes=(
                            f"BIRD {self.config.split}/{entry.db_id}"
                            f"/qid={entry.question_id}/{entry.difficulty}"
                        ),
                    )
                )
        finally:
            for c in conns.values():
                c.close()
        self._built = True

    def all_tasks(self) -> list[SqlTask]:
        if not self._built:
            self.build()
        return list(self._tasks)


def _execute_gold(conn: sqlite3.Connection, sql: str) -> list[dict[str, Any]] | None:
    """Run a gold SQL string through SQLite. Returns rows or None on error."""
    try:
        cur = conn.cursor()
        cur.execute(sql)
        cols = [c[0] for c in cur.description] if cur.description else []
        rows = cur.fetchall()
    except sqlite3.Error:
        return None
    return [dict(zip(cols, r, strict=False)) for r in rows]


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def _render_user_prompt(entry: BirdEntry, *, max_value_chars: int = 40) -> str:
    """Format the per-task user prompt.

    Layout:
        Question: <q>
        Evidence: <evidence (if present)>
        Database: <db_id>
        Tables (in this database):
          <table1> (catalog name: <db_id>__<table1>)
            Columns (sanitized <- original; type):
              c_xxx  <-  Original Heading  (TEXT)
              ...
            Sample rows (3):
              | c_xxx | ... |
              | ...   | ... |
            (Total rows: <n>)
          <table2> ...

        Write a SELECT in the target dialect that answers the question.
        Reference each table by its catalog name (``<db_id>__<table>``)
        and use the sanitized column names (the c_* identifiers) above.
    """
    lines: list[str] = []
    lines.append(f"Question: {entry.question}")
    if entry.evidence:
        lines.append(f"Evidence: {entry.evidence}")
    lines.append("")
    lines.append(f"Database: {entry.db_id}")
    if entry.difficulty:
        lines.append(f"Difficulty: {entry.difficulty}")
    lines.append("")
    lines.append("Tables (in this database):")
    for tbl in entry.tables:
        lines.append("")
        lines.append(
            f"  {tbl.original_name}  "
            f"(catalog name: ``{tbl.catalog_table_name}``, "
            f"{tbl.n_rows} rows)"
        )
        lines.append("    Columns (sanitized <- original; type):")
        # Pad sqlite_types to the column length in case an older
        # entry was constructed without it.
        sqlite_types = list(tbl.sqlite_types) + [""] * max(
            0, len(tbl.safe_columns) - len(tbl.sqlite_types)
        )
        for safe, orig, t, src in zip(
            tbl.safe_columns,
            tbl.original_columns,
            tbl.types,
            sqlite_types,
            strict=False,
        ):
            lines.append(
                f"      {safe}  <-  {orig}  ({_format_type_label(t, src)})"
            )
        if tbl.sample_rows:
            lines.append(f"    Sample rows ({len(tbl.sample_rows)}):")
            cols = tbl.safe_columns
            header = " | ".join(cols)
            sep = " | ".join("---" for _ in cols)
            lines.append(f"      | {header} |")
            lines.append(f"      | {sep} |")
            for row in tbl.sample_rows:
                cells = [
                    _format_cell(row.get(c), max_chars=max_value_chars)
                    for c in cols
                ]
                lines.append("      | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Write a SELECT in the target dialect that answers the question. "
        "Reference each table by its catalog name (the "
        "``<db_id>__<table>`` identifier shown above) and use the "
        "sanitized column names (the c_* identifiers). Apply the "
        "evidence field's domain hints when relevant."
    )
    return "\n".join(lines)


def _format_type_label(ir_type: str, sqlite_type: str) -> str:
    """Render the column's IR type, annotating TEXT-stored dates.

    BIRD stores DATE/DATETIME columns as TEXT; surfacing the original
    affinity tells the model "this Utf8 column actually holds date
    strings, you'll need to CAST before EXTRACT/DATE_DIFF/etc." Other
    type collapses (VARCHAR -> TEXT, NUMERIC -> FLOAT) aren't worth
    annotating because they don't affect query semantics.
    """
    if ir_type != "TEXT":
        return ir_type
    src = (sqlite_type or "").upper()
    if not src:
        return ir_type
    if "DATE" in src or "TIME" in src:
        # Trim parenthesized size hints (``DATETIME(3)``) for terseness.
        clean = re.sub(r"\s*\([^)]*\)\s*", "", src).strip()
        return f"TEXT, was {clean}"
    return ir_type


def _format_cell(v: Any, *, max_chars: int = 40) -> str:
    if v is None:
        return "NULL"
    s = str(v)
    s = s.replace("|", "\\|").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_chars:
        s = s[: max_chars - 3] + "..."
    return s


__all__ = [
    "BirdCatalog",
    "BirdEntry",
    "BirdTableInfo",
    "BirdTaskConfig",
    "BirdTaskGenerator",
]
