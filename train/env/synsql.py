"""SynSQL-2.5M prompt / database / answer triples as RL training data.

SynSQL-2.5M ([Li et al. 2025](https://arxiv.org/abs/2503.02240)) is the
first million-scale text-to-SQL corpus: 2,544,390 synthetic
``(question, sql, db_id, cot)`` quads spanning 16,583 synthetic
SQLite databases, with SQL complexity bands (``simple`` / ``moderate``
/ ``complex`` / ``highly complex``) and a wide spread of NL question
styles (``formal``, ``conversational``, ``vague``, ``metaphorical``,
...). Strictly larger and more diverse than BIRD; useful when GRPO
on Qwen3-4B-Instruct ceilings out on BIRD's simple+moderate slice.

This module is the SynSQL analogue of :mod:`train.env.bird`. Wiring:

* :class:`SynSqlCatalog` -- :class:`CatalogProvider` that loads only
  the databases referenced by the sampled question subset, materializes
  each SQLite table into a Polars DataFrame with sanitized column
  names, and namespaces tables ``<db_id>__<table>`` so the global
  catalog stays unique. Memoized like :class:`BirdCatalog`.

* :class:`SynSqlTaskGenerator` -- emits one :class:`SqlTask` per
  question. The user prompt embeds the relevant DB's schema (sanitized
  + original column headers, types, sample rows), the question, and
  the SynSQL ``external_knowledge`` field when present. Gold rows are
  computed by running the original ``sql`` field through stdlib
  ``sqlite3`` against the source ``.sqlite`` file -- same column-name-
  insensitive comparison as BIRD, so we never need to rewrite the gold
  SQL to use sanitized identifiers.

Data sources
------------

The dataset lives entirely on HuggingFace at ``seeklhy/SynSQL-2.5M``,
in three files:

* ``data.json`` (9.36 GB) -- one giant JSON array of 2.54M items.
* ``databases.zip`` (54.5 MB compressed; ~16k tiny SQLite files).
* ``tables.json`` (307 MB) -- per-DB schema metadata. Not used here;
  the schema is read from each ``.sqlite`` via stdlib ``sqlite3``
  PRAGMA queries (same path as BIRD).

The 9.36 GB ``data.json`` is the only real loading challenge --
downloading and parsing the whole file just to grab a few thousand
training examples is wasteful and OOM-prone.

Streaming strategy
------------------

We never download the full ``data.json``. Instead:

1. Open an HTTPS streaming connection to the HF resolve URL.

2. Walk the JSON array item-by-item with a small chunk-based
   incremental parser (see :func:`iter_json_array_items`). The parser
   tracks brace depth + string state so it only invokes ``json.loads``
   on complete top-level items.

3. Skip the first ``start_index`` items (for split offsets), then
   collect items that pass the optional ``complexities`` filter,
   stopping after ``n_samples``. Close the connection.

4. Cache the parsed slice as JSONL at
   ``~/.cache/manysql/synsql/samples_<split>_seed<seed>_n<N>_skip<K>.jsonl``
   so subsequent runs with the same parameters skip the network
   entirely.

For ``n_samples=1000`` with no skip this transfers ~5-10 MB instead
of the full 9.36 GB. Larger ``n_samples`` or large ``start_index``
values scale roughly linearly in bytes streamed (same items / sec
through the parser, same CDN throughput).

Database files
--------------

``databases.zip`` is small (54.5 MB) and downloads in one shot. We
extract it to ``~/.cache/manysql/synsql/databases/`` (selectively if
the sampled subset only references a fraction of the 16k DBs). The
zip is removed after extraction unless ``$SYNSQL_KEEP_ZIP=1`` --
unlike BIRD's ~5 GB train zip the savings are minor, but the
behavior matches ``bird.py`` for consistency.

Train / dev split convention
----------------------------

SynSQL-2.5M ships as one shuffled corpus with no built-in
train/dev/test partition. We define our own deterministic, mutually-
exclusive slices via ``start_index``:

* ``split='train'`` -> default ``start_index=0`` (head of the array).
* ``split='dev'``   -> default ``start_index=2_000_000`` (last ~544k
  items, well clear of any reasonable train sample size).
* ``split='test'``  -> default ``start_index=2_400_000`` (last ~144k).

Pass ``start_index=N`` explicitly to override. The defaults are
designed so train and dev never overlap regardless of
``n_samples`` (the absolute item indices differ by 2M+).

Caveat: SynSQL was constructed by the OmniSQL team using their own
synthesis pipeline; there's no published "official" eval split. The
above convention is a manysql convention chosen for reproducibility.
For benchmark comparability, evaluate trained models against BIRD or
Spider, not the SynSQL dev slice -- see ``OmniSQL`` paper §5.

Loading the dataset requires only stdlib (``urllib``, ``zipfile``,
``json``, ``sqlite3``). The HF ``datasets`` library is NOT used --
it can't stream this dataset reliably (the JSON-array shape is too
large for the auto-parquet conversion service). Polars + the
manysql IR are imported lazily inside ``build()`` so this module is
import-cheap on stripped-down environments.
"""

from __future__ import annotations

import codecs
import json
import os
import shutil
import sqlite3
import sys
import urllib.request
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from train.env._ident import dedupe_columns, safe_ident, safe_table_name
from train.env.catalog import CatalogProvider, CatalogSnapshot
from train.env.tasks import SqlTask, TaskGenerator
from train.env.types import TaskMeta

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HF_REPO = "seeklhy/SynSQL-2.5M"
_HF_BASE_URL = f"https://huggingface.co/datasets/{_HF_REPO}/resolve/main"
_DATA_JSON_URL = f"{_HF_BASE_URL}/data.json"
_DATABASES_ZIP_URL = f"{_HF_BASE_URL}/databases.zip"

# SynSQL-2.5M is one shuffled JSON array; we partition train/dev/test
# by absolute index. The defaults are large enough that no reasonable
# sample size from one split overlaps another. Override via
# ``start_index`` if you need finer control.
_DEFAULT_SPLIT_START_INDEX: dict[str, int] = {
    "train": 0,
    "dev": 2_000_000,
    "test": 2_400_000,
}

_DEFAULT_COMPLEXITIES: tuple[str, ...] = ("simple", "moderate")
_VALID_COMPLEXITIES: frozenset[str] = frozenset(
    ("simple", "moderate", "complex", "highly complex")
)

# Same row-cap default as BIRD. SynSQL DBs are typically tiny (<<1MB)
# so this rarely triggers; left in for safety against pathological
# generated DBs.
_DEFAULT_MAX_ROWS_PER_TABLE: int = 200_000


# ---------------------------------------------------------------------------
# Streaming JSON-array parser
# ---------------------------------------------------------------------------


def iter_json_array_items(
    stream: IO[bytes] | IO[str],
    *,
    chunk_size: int = 1 << 16,
) -> Iterator[Any]:
    """Yield each top-level item from a JSON array, parsed incrementally.

    Reads ``stream`` in ``chunk_size`` chunks (UTF-8 incremental decode
    so multi-byte glyphs straddling chunk boundaries don't corrupt),
    tracking nesting depth and string state to identify item
    boundaries. Each completed top-level value is parsed via
    ``json.loads`` and yielded.

    Why we don't use ``ijson`` / ``json-stream`` / ``msgspec``:
        SynSQL adds dataset support to the *training* env, which has
        no other ``ijson`` consumer. Adding a transitive dep just for
        this would be heavyweight; the parser here is ~50 lines and
        handles the only shape we ever see (a flat array of objects).

    Caller contract:
        * ``stream.read(chunk_size)`` returns ``bytes`` or ``str``.
        * The wire payload must start with ``[`` (after optional
          whitespace) and contain JSON values (objects, arrays,
          strings, numbers, booleans, ``null``) separated by ``,``.
        * Nested arrays / objects inside items are handled.
        * Trailing whitespace / commas before ``]`` are tolerated.

    Bail-out:
        Caller can simply ``break`` out of the generator once it has
        enough items; the underlying socket / file is closed by the
        caller via context manager. The generator itself does not
        consume bytes past the point where the caller stops asking.
    """
    decoder = codecs.getincrementaldecoder("utf-8")()
    state = 0  # 0 = before array, 1 = between items, 2 = collecting
    depth = 0
    in_str = False
    esc = False
    buf: list[str] = []
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            # Flush the incremental decoder; any trailing partial char
            # gets replaced with the U+FFFD substitute. We don't have
            # final=True until here so split-multibyte chunks work.
            try:
                tail = decoder.decode(b"", final=True)
            except UnicodeDecodeError:
                tail = ""
            text = tail
            if not text:
                return
        else:
            text = (
                decoder.decode(chunk)
                if isinstance(chunk, bytes)
                else chunk
            )
        for ch in text:
            if state == 0:
                if ch == "[":
                    state = 1
                continue
            if state == 1:
                if ch == "{" or ch == "[":
                    buf.append(ch)
                    depth = 1
                    in_str = False
                    esc = False
                    state = 2
                elif ch == "]":
                    return
                # else: whitespace / comma -- ignore
                continue
            # state == 2: collecting one top-level item
            buf.append(ch)
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{" or ch == "[":
                depth += 1
            elif ch == "}" or ch == "]":
                depth -= 1
                if depth == 0:
                    yield json.loads("".join(buf))
                    buf = []
                    state = 1
        if not chunk:
            return


# ---------------------------------------------------------------------------
# Materialized SynSQL example
# ---------------------------------------------------------------------------


@dataclass
class SynSqlEntry:
    """One materialized SynSQL example (post-sanitization).

    Held by :class:`SynSqlCatalog` after ``build()``; consumed by
    :class:`SynSqlTaskGenerator` to render the per-task prompt and
    look up the matching gold rows.

    ``tables`` is the per-task schema view (db-scoped) used for
    prompt rendering; the global :class:`CatalogSnapshot` carries the
    actual DataFrames keyed by ``catalog_table_name``.
    """

    item_index: int
    db_id: str
    question: str
    external_knowledge: str
    sql: str
    sql_complexity: str
    question_style: str
    db_path: str
    tables: list["SynSqlTableInfo"]


@dataclass
class SynSqlTableInfo:
    """Per-table view of a SynSQL database, sanitized for prompt + catalog."""

    original_name: str
    catalog_table_name: str  # `<db_id>__<table>` (lowercased, deduped)
    safe_columns: list[str]
    original_columns: list[str]
    types: list[str]
    sqlite_types: list[str] = field(default_factory=list)
    sample_rows: list[dict[str, Any]] = field(default_factory=list)
    n_rows: int = 0


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


class SynSqlCatalog(CatalogProvider):
    """Catalog provider backed by N SynSQL-2.5M examples.

    Streams ``data.json`` from HF, takes the first ``n_samples`` items
    that pass the ``complexities`` filter (after skipping
    ``start_index``), then walks the union of referenced ``db_id``
    values and pulls each one's tables out of the matching
    ``.sqlite`` file via stdlib ``sqlite3``.

    Tables are namespaced ``<db_id>__<table>`` so the global table set
    stays unique; column names are sanitized to ``c_<lowercase_alnum>``
    (with dedup suffixes on collision) so they parse in every
    dialect's grammar.

    Memoization: ``build()`` caches the snapshot. Reuse the same
    instance across multiple :class:`DialectRuntime` workers so we
    don't re-stream / re-extract per dialect in cross-product
    multi-dialect runs.
    """

    name = "synsql"

    def __init__(
        self,
        *,
        n_samples: int = 1000,
        split: str = "train",
        seed: int = 0,
        complexities: tuple[str, ...] = _DEFAULT_COMPLEXITIES,
        start_index: int | None = None,
        db_dir: str | None = None,
        auto_download: bool = True,
        sample_rows: int = 3,
        max_table_value_chars: int = 40,
        max_rows_per_table: int | None = _DEFAULT_MAX_ROWS_PER_TABLE,
        cache_dir: str | None = None,
    ) -> None:
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if split not in _DEFAULT_SPLIT_START_INDEX:
            raise ValueError(
                f"split must be one of {sorted(_DEFAULT_SPLIT_START_INDEX)}, "
                f"got {split!r}"
            )
        bad_cx = set(complexities) - _VALID_COMPLEXITIES
        if bad_cx:
            raise ValueError(
                f"unknown sql_complexity levels {sorted(bad_cx)}; "
                f"valid: {sorted(_VALID_COMPLEXITIES)}"
            )
        if sample_rows < 0:
            raise ValueError(f"sample_rows must be non-negative, got {sample_rows}")
        if max_rows_per_table is not None and max_rows_per_table <= 0:
            max_rows_per_table = None

        self.n_samples = n_samples
        self.split = split
        self.seed = seed
        self.complexities = tuple(complexities)
        self.start_index = (
            start_index
            if start_index is not None
            else _DEFAULT_SPLIT_START_INDEX[split]
        )
        self.db_dir_override = db_dir
        self.auto_download = auto_download
        self.sample_rows_n = sample_rows
        self.max_table_value_chars = max_table_value_chars
        self.max_rows_per_table = max_rows_per_table
        self.cache_dir = (
            Path(cache_dir).expanduser()
            if cache_dir is not None
            else Path.home() / ".cache" / "manysql" / "synsql"
        )

        self._snapshot: CatalogSnapshot | None = None
        self._entries: list[SynSqlEntry] = []
        self._db_dir: Path | None = None
        self._effective_db_path: dict[str, Path] = {}

    # -- public API --

    def build(self) -> CatalogSnapshot:
        if self._snapshot is not None:
            return self._snapshot

        from manysql.ir.plan import ColumnSchema  # noqa: PLC0415

        # Lazy-import the heavy SQLite-to-Polars helper from the BIRD
        # module so we don't fork that logic. It's a private symbol
        # but stable across the package; the alternative is a shared
        # ``train.env._sqlite_loader`` extraction which isn't worth
        # the churn for one extra caller.
        from train.env.bird import (  # noqa: PLC0415
            _ensure_sampled_db,
            _ir_type_for,
            _load_sqlite_to_polars,
        )

        sampled = self._load_or_stream_questions()
        self._db_dir = self._resolve_db_dir({q["db_id"] for q in sampled})

        per_db: dict[
            str,
            dict[str, tuple[Any, list[str], list[str], list[str], list[str]]],
        ] = {}
        missing_dbs: set[str] = set()
        sampled_dbs: list[tuple[str, int]] = []
        per_db_effective_path: dict[str, Path] = {}
        for db_id in sorted({q["db_id"] for q in sampled}):
            db_path = self._db_path_for(db_id)
            if db_path is None:
                missing_dbs.add(db_id)
                continue
            effective_path = db_path
            if self.max_rows_per_table is not None:
                effective_path, max_orig = _ensure_sampled_db(
                    db_path,
                    db_id=db_id,
                    split=f"synsql__{self.split}",
                    max_rows_per_table=self.max_rows_per_table,
                )
                if max_orig > self.max_rows_per_table:
                    sampled_dbs.append((db_id, max_orig))
            try:
                per_db[db_id] = _load_sqlite_to_polars(effective_path)
                per_db_effective_path[db_id] = effective_path
            except Exception as exc:
                print(
                    f"[synsql] WARN: failed to load DB {db_id!r} from "
                    f"{effective_path}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                missing_dbs.add(db_id)
        if sampled_dbs:
            preview = ", ".join(
                f"{db_id}({n})" for db_id, n in sorted(
                    sampled_dbs, key=lambda x: -x[1]
                )[:5]
            )
            print(
                f"[synsql] INFO: {len(sampled_dbs)} DB(s) had tables "
                f"capped to {self.max_rows_per_table} rows "
                f"(largest source-table sizes: {preview}"
                f"{'...' if len(sampled_dbs) > 5 else ''}); gold SQL "
                f"is re-executed on the sampled mirror so model "
                f"outputs and gold rows stay aligned.",
                file=sys.stderr,
            )
        if missing_dbs:
            preview = sorted(missing_dbs)[:5]
            print(
                f"[synsql] WARN: {len(missing_dbs)} DB(s) missing or "
                f"unreadable; questions referencing them will be "
                f"dropped: {preview}"
                f"{'...' if len(missing_dbs) > 5 else ''}",
                file=sys.stderr,
            )
        self._effective_db_path = dict(per_db_effective_path)

        tables: dict[str, Any] = {}
        schemas: dict[str, tuple[ColumnSchema, ...]] = {}
        per_db_table_info: dict[str, dict[str, SynSqlTableInfo]] = {}
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
                per_db_table_info[db_id][tbl_name] = SynSqlTableInfo(
                    original_name=tbl_name,
                    catalog_table_name=catalog_name,
                    safe_columns=safe_cols,
                    original_columns=orig_cols,
                    types=types,
                    sqlite_types=list(sqlite_types),
                    sample_rows=sample,
                    n_rows=df.height,
                )

        # Per-question entries (one per task), only for questions
        # whose DB loaded successfully. Each entry's ``db_path`` points
        # at the *effective* DB (sampled mirror if row-capping
        # triggered) so the gold SQL run in
        # :meth:`SynSqlTaskGenerator.build` evaluates against the same
        # row subset Polars sees.
        entries: list[SynSqlEntry] = []
        for i, q in enumerate(sampled):
            if q["db_id"] not in per_db_table_info:
                continue
            tbl_views = list(per_db_table_info[q["db_id"]].values())
            if not tbl_views:
                continue
            db_path = self._effective_db_path.get(q["db_id"])
            if db_path is None:
                continue
            entries.append(
                SynSqlEntry(
                    item_index=int(q.get("_synsql_index", i)),
                    db_id=q["db_id"],
                    question=q.get("question") or "",
                    external_knowledge=q.get("external_knowledge") or "",
                    sql=q.get("sql") or "",
                    sql_complexity=q.get("sql_complexity") or "",
                    question_style=q.get("question_style") or "",
                    db_path=str(db_path),
                    tables=tbl_views,
                )
            )
        # Stable ordering: by stream index. Reproducible across runs.
        entries.sort(key=lambda e: e.item_index)

        self._entries = entries
        self._snapshot = CatalogSnapshot(
            tables=tables,
            schemas=schemas,
            schema_prompt=_SYNSQL_SCHEMA_PROMPT,
        )
        return self._snapshot

    def entries(self) -> list[SynSqlEntry]:
        if self._snapshot is None:
            self.build()
        return list(self._entries)

    # -- question loading: cache or stream --

    def _load_or_stream_questions(self) -> list[dict[str, Any]]:
        """Return a deterministic ``n_samples``-sized question subset.

        Cache layout:
            ~/.cache/manysql/synsql/samples_<split>_seed<seed>_n<N>_skip<K>.jsonl

        Cache hits skip the network entirely; cache misses stream
        ``data.json`` from HF, persist the parsed slice, and return it.
        """
        cache_path = self._sample_cache_path()
        if cache_path.is_file():
            return _load_jsonl(cache_path)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        items = list(self._stream_questions())
        # Persist atomically so concurrent workers don't read partials.
        tmp = cache_path.with_suffix(".jsonl.partial")
        with tmp.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        tmp.replace(cache_path)
        return items

    def _sample_cache_path(self) -> Path:
        # Filter discriminator: hash of (sorted complexities) so two
        # configs with different filters don't collide. We keep the
        # filename human-legible since users often delete caches by
        # eyeballing them; the extra suffix is tiny.
        cx_tag = "+".join(sorted(c.replace(" ", "_") for c in self.complexities))
        name = (
            f"samples_{self.split}_seed{self.seed}_n{self.n_samples}_"
            f"skip{self.start_index}_cx_{cx_tag}.jsonl"
        )
        return self.cache_dir / name

    def _stream_questions(self) -> Iterator[dict[str, Any]]:
        """Stream ``data.json`` from HF and yield ``n_samples`` items.

        Items are yielded in stream order. Filtering by
        ``complexities`` happens here so the caller doesn't have to
        retain dropped items. ``start_index`` skips the first K items
        of the array regardless of filter (so two splits with
        different ``start_index`` don't collide on item indices).

        The HTTPS connection is closed as soon as enough items are
        collected. Bandwidth scales with ``start_index + n_samples``,
        not the full 9.36 GB file.
        """
        print(
            f"[synsql] streaming {self.n_samples} item(s) from "
            f"{_DATA_JSON_URL} (split={self.split}, "
            f"start_index={self.start_index}, "
            f"complexities={self.complexities})",
            file=sys.stderr,
        )
        req = urllib.request.Request(
            _DATA_JSON_URL,
            headers={"User-Agent": "manysql/synsql"},
        )
        token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        collected = 0
        scanned = 0
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            for item in iter_json_array_items(resp):
                scanned += 1
                if scanned <= self.start_index:
                    continue
                cx = item.get("sql_complexity") or "simple"
                if self.complexities and cx not in self.complexities:
                    continue
                # Stamp the absolute item index for reproducibility +
                # debugging; downstream code uses it as a stable id.
                item["_synsql_index"] = scanned - 1
                yield item
                collected += 1
                if collected >= self.n_samples:
                    return
        if collected == 0:
            raise RuntimeError(
                "[synsql] streamed the full dataset without finding any "
                f"items matching complexities={self.complexities}; "
                f"check the filter."
            )
        if collected < self.n_samples:
            print(
                f"[synsql] WARN: only collected {collected} item(s); "
                f"requested {self.n_samples}. Either start_index is "
                f"too far into the array, or the complexity filter "
                f"is rejecting most rows.",
                file=sys.stderr,
            )

    # -- DB filesystem layout --

    def _resolve_db_dir(self, referenced_db_ids: set[str]) -> Path:
        """Find or fetch the directory holding ``<db_id>/<db_id>.sqlite``.

        Resolution order matches the BIRD pattern:
            1. ``self.db_dir_override`` if set.
            2. ``$SYNSQL_DB_DIR`` env var if set.
            3. ``~/.cache/manysql/synsql/databases/`` if it already
               exists from a prior run.
            4. Auto-download (when ``auto_download``).
        """
        candidates: list[Path] = []
        if self.db_dir_override:
            candidates.append(Path(self.db_dir_override).expanduser())
        env_root = os.environ.get("SYNSQL_DB_DIR")
        if env_root:
            candidates.append(Path(env_root).expanduser())
        cache_root = self.cache_dir / "databases"
        candidates.append(cache_root)

        for cand in candidates:
            if _looks_like_synsql_db_dir(cand, referenced_db_ids):
                return cand

        if self.auto_download:
            cache_root.mkdir(parents=True, exist_ok=True)
            _download_and_extract_databases(
                cache_root,
                referenced_db_ids=referenced_db_ids,
            )
            if _looks_like_synsql_db_dir(cache_root, referenced_db_ids):
                return cache_root
            raise RuntimeError(
                f"SynSQL auto-download finished but {cache_root} doesn't "
                f"match the expected layout (<db_id>/<db_id>.sqlite). "
                f"Top-level entries: "
                f"{sorted(p.name for p in cache_root.iterdir())[:20]}"
            )

        raise RuntimeError(
            "SynSQL databases not found and auto-download is disabled. "
            "Either pass --synsql-db-dir / set $SYNSQL_DB_DIR, or "
            f"re-run with auto_download=True (will fetch ~55MB from "
            f"{_DATABASES_ZIP_URL})."
        )

    def _db_path_for(self, db_id: str) -> Path | None:
        if self._db_dir is None:
            return None
        candidate = self._db_dir / db_id / f"{db_id}.sqlite"
        if candidate.is_file():
            return candidate
        alt = self._db_dir / db_id / f"{db_id}.db"
        if alt.is_file():
            return alt
        return None


def _looks_like_synsql_db_dir(path: Path, referenced_db_ids: set[str]) -> bool:
    """True iff at least half of the referenced DBs live under ``path``."""
    if not path.is_dir():
        return False
    if not referenced_db_ids:
        return any((path / sub).is_dir() for sub in path.iterdir())
    hits = 0
    for db_id in referenced_db_ids:
        if (path / db_id / f"{db_id}.sqlite").is_file():
            hits += 1
        elif (path / db_id / f"{db_id}.db").is_file():
            hits += 1
    return hits >= max(1, len(referenced_db_ids) // 2)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Auto-download
# ---------------------------------------------------------------------------


def _download_and_extract_databases(
    dest: Path,
    *,
    referenced_db_ids: Iterable[str] | None = None,
) -> None:
    """Fetch ``databases.zip`` and selectively unpack it under ``dest``.

    The zip is small (~55 MB) so we always pull the whole file, but
    extraction is filtered to ``referenced_db_ids`` when provided
    (the catalog only ever needs ~N DBs, so we don't waste inode
    space on the other ~16k).

    Re-entrancy: if every referenced DB is already extracted, returns
    without touching the network.
    """
    refs = set(referenced_db_ids or ())
    if refs and all(
        (dest / d / f"{d}.sqlite").is_file() for d in refs
    ):
        return

    zip_path = dest / "databases.zip"
    if not zip_path.is_file():
        token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )
        req = urllib.request.Request(
            _DATABASES_ZIP_URL,
            headers={"User-Agent": "manysql/synsql"},
        )
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        print(
            f"[synsql] downloading {_DATABASES_ZIP_URL} into {zip_path} "
            f"(one-time ~55MB download)",
            file=sys.stderr,
        )
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            total = int(resp.headers.get("Content-Length") or 0)
            tmp = zip_path.with_suffix(".zip.partial")
            done = 0
            chunk = 1 << 20
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
                                f"\r[synsql] download {pct}% "
                                f"({done >> 20} / {total >> 20} MB)"
                            )
                            sys.stderr.flush()
                            last_pct = pct
            sys.stderr.write("\n")
            tmp.replace(zip_path)

    def _wanted(info: zipfile.ZipInfo) -> bool:
        # The official zip lays files out as
        # ``databases/<db_id>/<db_id>.sqlite``. We tolerate alternate
        # layouts by looking for any ``<x>.sqlite`` whose parent
        # directory matches a referenced db_id.
        parts = info.filename.split("/")
        if not parts:
            return False
        if not refs:
            return True
        # Find the segment immediately before the .sqlite filename;
        # that's the db_id directory in the standard layout.
        for i, seg in enumerate(parts):
            if seg in refs:
                return True
            if (
                seg.endswith(".sqlite")
                and i > 0
                and parts[i - 1] in refs
            ):
                return True
        return False

    extracted_count = 0
    extracted_bytes = 0
    print(
        f"[synsql] extracting {zip_path} into {dest}"
        + (f" (filtering to {len(refs)} referenced db_id(s))" if refs else "")
    )
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if _wanted(info):
                zf.extract(info, dest)
                extracted_count += 1
                extracted_bytes += info.file_size
    print(
        f"[synsql] extract done: {extracted_count} entries, "
        f"{_human_bytes(extracted_bytes)}"
    )

    # Flatten the leading ``databases/`` directory so the final
    # layout is ``<dest>/<db_id>/<db_id>.sqlite``.
    nested = dest / "databases"
    if nested.is_dir():
        for child in nested.iterdir():
            target = dest / child.name
            if not target.exists():
                shutil.move(str(child), str(target))
        try:
            nested.rmdir()
        except OSError:
            pass

    if not os.environ.get("SYNSQL_KEEP_ZIP"):
        try:
            zip_path.unlink()
        except OSError:
            pass


def _human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.0f}KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / 1024 / 1024:.0f}MB"
    return f"{n / 1024 / 1024 / 1024:.1f}GB"


# ---------------------------------------------------------------------------
# Schema prompt (system-prompt slot)
# ---------------------------------------------------------------------------

_SYNSQL_SCHEMA_PROMPT = (
    "Each task targets ONE synthetic database from the SynSQL-2.5M "
    "corpus (16k+ generated SQLite schemas, varying domain shapes). "
    "The relevant database's tables -- with sanitized column names "
    "(``c_<lowercase_alnum>``), original headers, types, and a few "
    "sample rows -- are provided in the user message; reference that "
    "block, not a global schema. Tables are namespaced "
    "``<db_id>__<table>`` so they're unique across the whole training "
    "corpus.\n\n"
    "DATE/TIME COLUMNS: SQLite stores dates and timestamps as TEXT, "
    "and the catalog preserves that storage shape. Any column whose "
    "original SQLite type was DATE / DATETIME / TIMESTAMP is exposed "
    "as a TEXT column whose values are ISO-formatted strings. The "
    "per-table column listing tags these with ``(TEXT, was DATETIME)`` "
    "etc. so they are identifiable. To use ``EXTRACT`` / ``DATE_PART`` "
    "/ ``DATE_TRUNC`` / ``DATE_DIFF`` on such a column, CAST it first "
    "(e.g. ``EXTRACT(YEAR FROM CAST(c_hire_date AS DATE))``); without "
    "the CAST the engine will reject the call because the underlying "
    "type is still text. Direct string ordering "
    "(``c_hire_date >= '2020-01-01'``) works without a CAST because "
    "ISO format sorts lexicographically."
)


# ---------------------------------------------------------------------------
# Task generator
# ---------------------------------------------------------------------------


@dataclass
class SynSqlTaskConfig:
    """Configuration for :class:`SynSqlTaskGenerator`.

    Pass a pre-built ``catalog`` to share across multiple dialect
    runtimes (the recommended path for multi-dialect curricula).
    Otherwise the generator constructs its own with the same
    n_samples/split/seed/complexities.

    ``drop_empty`` filters out tasks whose gold SQL returns zero
    rows. SynSQL has a small fraction of such cases (often when a
    filter excludes everything in the synthetic DB); empty-result
    tasks are a poor training signal because "any SELECT that returns
    nothing" matches.

    ``max_rows_per_table`` is inherited from BIRD's safety logic;
    SynSQL DBs are typically tiny (<<1MB) so it rarely triggers, but
    it protects against pathological generated DBs. 0 / None disables.
    """

    target_dialect: str
    n_samples: int = 1000
    split: str = "train"  # 'train' | 'dev' | 'test'
    seed: int = 0
    complexities: tuple[str, ...] = _DEFAULT_COMPLEXITIES
    start_index: int | None = None
    sample_rows: int = 3
    db_dir: str | None = None
    auto_download: bool = True
    catalog: SynSqlCatalog | None = None
    drop_empty: bool = True
    max_rows_per_table: int | None = _DEFAULT_MAX_ROWS_PER_TABLE
    cache_dir: str | None = None
    extra_meta: dict[str, Any] = field(default_factory=dict)


class SynSqlTaskGenerator(TaskGenerator):
    """Build NL->SQL tasks from SynSQL-2.5M.

    Each task pairs a natural-language question (plus the
    ``external_knowledge`` field as a hint when present) with the
    relevant database's schema and an executable canonical gold SQL.
    Gold rows are computed once via stdlib ``sqlite3`` against the
    source ``.sqlite`` file; the resulting :class:`SqlTask` list is
    dialect-tagged via ``config.target_dialect`` but the rows are
    dialect-independent -- higher layers (e.g.
    :func:`train.grpo_sql.build_runtimes_and_tasks`) clone tasks
    across additional dialects without re-running SQLite.
    """

    name = "synsql"

    def __init__(self, config: SynSqlTaskConfig) -> None:
        self.config = config
        self.catalog = config.catalog or SynSqlCatalog(
            n_samples=config.n_samples,
            split=config.split,
            seed=config.seed,
            complexities=config.complexities,
            start_index=config.start_index,
            db_dir=config.db_dir,
            auto_download=config.auto_download,
            sample_rows=config.sample_rows,
            max_rows_per_table=config.max_rows_per_table,
            cache_dir=config.cache_dir,
        )
        self._tasks: list[SqlTask] = []
        self._built = False

    def build(self) -> None:
        if self._built:
            return
        self.catalog.build()

        conns: dict[str, sqlite3.Connection] = {}
        try:
            for entry in self.catalog.entries():
                conn = conns.get(entry.db_path)
                if conn is None:
                    conn = sqlite3.connect(entry.db_path)
                    conns[entry.db_path] = conn
                rows = _execute_gold(conn, entry.sql)
                if rows is None:
                    continue
                if self.config.drop_empty and not rows:
                    continue
                meta = TaskMeta(
                    task_id=f"synsql_{entry.db_id}_{entry.item_index}",
                    dialect=self.config.target_dialect,
                    generator=self.name,
                    meta={
                        "db_id": entry.db_id,
                        "split": self.config.split,
                        "sql_complexity": entry.sql_complexity,
                        "question_style": entry.question_style,
                        "n_tables": len(entry.tables),
                        "external_knowledge_present": bool(
                            entry.external_knowledge
                        ),
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
                            f"SynSQL {self.config.split}/{entry.db_id}"
                            f"/idx={entry.item_index}/"
                            f"{entry.sql_complexity}"
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


def _render_user_prompt(entry: SynSqlEntry, *, max_value_chars: int = 40) -> str:
    """Format the per-task user prompt.

    Layout mirrors :func:`train.env.bird._render_user_prompt`:

        Question: <q>
        External knowledge: <ek (if present)>
        Database: <db_id>
        Tables (in this database):
          <table1> (catalog name: <db_id>__<table1>)
            Columns (sanitized <- original; type):
              c_xxx  <-  Original Heading  (TEXT)
              ...
            Sample rows (3):
              ...
            (Total rows: <n>)
          <table2> ...

        Write a SELECT in the target dialect that answers the
        question. Reference each table by its catalog name and use
        the sanitized column names.
    """
    lines: list[str] = []
    lines.append(f"Question: {entry.question}")
    if entry.external_knowledge:
        lines.append(f"External knowledge: {entry.external_knowledge}")
    lines.append("")
    lines.append(f"Database: {entry.db_id}")
    if entry.sql_complexity:
        lines.append(f"SQL complexity: {entry.sql_complexity}")
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
        "Write a SELECT in the target dialect that answers the "
        "question. Reference each table by its catalog name (the "
        "``<db_id>__<table>`` identifier shown above) and use the "
        "sanitized column names (the c_* identifiers). Apply the "
        "external-knowledge hints when relevant."
    )
    return "\n".join(lines)


def _format_type_label(ir_type: str, sqlite_type: str) -> str:
    """Render the column's IR type, annotating TEXT-stored dates.

    Same contract as the BIRD prompt: SQLite stores DATE/DATETIME as
    TEXT, so we surface the original affinity to tell the model "this
    Utf8 column actually holds date strings, CAST before
    EXTRACT/DATE_DIFF/etc."
    """
    if ir_type != "TEXT":
        return ir_type
    src = (sqlite_type or "").upper()
    if not src:
        return ir_type
    if "DATE" in src or "TIME" in src:
        import re  # noqa: PLC0415
        clean = re.sub(r"\s*\([^)]*\)\s*", "", src).strip()
        return f"TEXT, was {clean}"
    return ir_type


def _format_cell(v: Any, *, max_chars: int = 40) -> str:
    if v is None:
        return "NULL"
    s = str(v)
    s = s.replace("|", "\\|").replace("\n", " ")
    import re  # noqa: PLC0415
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_chars:
        s = s[: max_chars - 3] + "..."
    return s


__all__ = [
    "SynSqlCatalog",
    "SynSqlEntry",
    "SynSqlTableInfo",
    "SynSqlTaskConfig",
    "SynSqlTaskGenerator",
    "iter_json_array_items",
]
