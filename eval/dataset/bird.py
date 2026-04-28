"""BIRD-SQL question loader for the eval harness.

Wires a subset of the BIRD-SQL benchmark
(`Li et al. 2023 <https://bird-bench.github.io/>`_) into the eval
runner. Each BIRD example targets a different multi-table SQLite
database, so we deviate from the github-events flow in two ways:

* The candidate SQL must run against that question's specific
  ``.sqlite`` file. We surface this via :class:`Question.db_path`,
  which the runner threads to ``BirdSqliteExecutor.execute(...,
  question=q)`` to pick the right backing connection.
* The schema can't sit in the system prompt because it varies per
  question. We inline the relevant DB's tables (original column
  names + SQLite type declarations + a few sample rows + the BIRD
  ``evidence`` field) directly into ``Question.prompt`` and keep
  the system prompt's ``schema_prompt()`` as a generic "schema is in
  the user message below" blurb.

Important: this loader is intentionally *not* the same as
:mod:`train.env.bird`'s. Training sanitizes column names
(``c_free_meal_count_k_12``) so SQL parses through grammar-strict
synthetic dialects; eval against stdlib SQLite uses the **real**
column names so candidate SQL can hit the BIRD ``.sqlite`` file
directly. We still reuse train's HF/auto-download helpers (lazily
imported) to avoid duplicating the multi-GB train.zip handler.

Data sources:

* Questions: HuggingFace ``birdsql/bird23-train-filtered`` (train) or
  ``birdsql/bird_sql_dev_20251106`` (dev). Loading needs
  ``datasets>=2.20``; lazily imported.
* DB files: NOT on HuggingFace. Train ships as a ~5GB zip on a public
  Beijing OSS bucket; we auto-download into
  ``~/.cache/manysql/bird/<split>/`` on first use. Dev is gated
  behind a Google-Drive page so it requires manual download or
  ``--bird-db-dir``.
"""

from __future__ import annotations

import os
import random
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.dataset.questions import Question

_DEFAULT_DIFFICULTIES: tuple[str, ...] = ("simple", "moderate")
_VALID_DIFFICULTIES: frozenset[str] = frozenset(
    ("simple", "moderate", "challenging")
)
_HF_DATASET_TRAIN = "birdsql/bird23-train-filtered"
_HF_DATASET_DEV = "birdsql/bird_sql_dev_20251106"
_DEV_DB_GDRIVE_HINT = (
    "https://drive.google.com/file/d/13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG/view"
)

_BIRD_EVAL_SCHEMA_PROMPT = (
    "Each question targets ONE BIRD-SQL database (real Kaggle-derived "
    "multi-table SQLite schemas, ~5-25 tables per DB). The relevant "
    "database's tables -- with their ORIGINAL column names, SQLite type "
    "declarations, and a few sample rows -- are inlined in the user "
    "message; reference that block, not a global schema.\n\n"
    "When a column name contains spaces, mixed case, or special "
    "characters, quote it with double-quotes (e.g. "
    "``\"Free Meal Count (K-12)\"``). Bare identifiers parse as "
    "lowercase in SQLite, so quoted-as-shown is the safest default.\n\n"
    "DATE/TIME COLUMNS: SQLite stores dates and timestamps as TEXT; the "
    "per-table column listing tags these (e.g. ``DATETIME``) so they "
    "are identifiable. Use ``strftime('%Y', col)``, ``date(col)``, "
    "``julianday(col)``, etc. to extract date components -- there is no "
    "``EXTRACT`` / ``DATE_PART`` in SQLite. ISO format sorts "
    "lexicographically, so ``col >= '2020-01-01'`` works without a CAST."
)

# Sentinel pattern: a column / table name that is safe as a bare
# identifier in SQLite (lower-case alnum + underscore). Anything else
# gets double-quoted in the prompt's schema rendering so the LLM
# learns to quote it back.
_BARE_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


# ---------------------------------------------------------------------------
# Per-question entry
# ---------------------------------------------------------------------------


@dataclass
class _BirdEvalTable:
    name: str
    columns: list[str]
    types: list[str]  # sqlite type declarations as-is (e.g. "INTEGER", "DATETIME")
    sample_rows: list[tuple[Any, ...]]
    n_rows: int


@dataclass
class _BirdEvalEntry:
    question_id: int
    db_id: str
    db_path: Path
    question: str
    evidence: str
    sql: str
    difficulty: str
    tables: list[_BirdEvalTable] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def select_bird(
    *,
    n_samples: int = 50,
    split: str = "dev",
    seed: int = 0,
    difficulties: tuple[str, ...] | list[str] = _DEFAULT_DIFFICULTIES,
    db_dir: str | Path | None = None,
    sample_rows: int = 3,
    max_value_chars: int = 40,
    auto_download: bool = True,
    drop_unrunnable: bool = True,
) -> list[Question]:
    """Build a list of :class:`Question` objects from a BIRD subset.

    Args:
        n_samples: how many questions to sample from the (filtered)
            HF split. The sample is reproducible given ``seed``.
        split: ``"dev"`` (1.5k community-reviewed questions, default
            since dev is the canonical leaderboard split) or
            ``"train"`` (~6.6k filtered training questions).
        seed: PRNG seed for the random subset.
        difficulties: subset of ``{"simple", "moderate", "challenging"}``
            to keep before sampling. Defaults to
            ``("simple", "moderate")`` -- challenging is significantly
            harder and adds noise to small-N evals.
        db_dir: optional override for the directory containing
            ``<db_id>/<db_id>.sqlite``. Defaults to ``$BIRD_DB_DIR`` or
            ``~/.cache/manysql/bird/<split>/``.
        sample_rows: rows to include per table in the prompt.
        max_value_chars: per-cell truncation in the sample-row table.
        auto_download: when ``True`` (and ``split == "train"``), fetch
            the official train.zip into the cache on first use. Dev
            cannot be auto-downloaded (Google Drive blocks scripted
            access).
        drop_unrunnable: when ``True``, silently drop questions whose
            gold SQL fails to execute on the DB (rare in the filtered
            train split, more common in dev). When ``False``, keep
            them and surface the SQLite error in the runner output.

    Returns:
        A list of :class:`Question`. Each item's ``prompt`` is the
        per-question NL question plus that DB's full schema with
        sample rows; ``reference_sql={"sqlite": gold_sql}`` is the
        BIRD ``SQL`` field; ``db_path`` points at the actual
        ``.sqlite`` file the runner should query against.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if split not in {"train", "dev"}:
        raise ValueError(f"split must be 'train' | 'dev', got {split!r}")
    diffs = tuple(difficulties)
    bad = set(diffs) - _VALID_DIFFICULTIES
    if bad:
        raise ValueError(
            f"unknown difficulty levels {sorted(bad)}; "
            f"valid: {sorted(_VALID_DIFFICULTIES)}"
        )
    if sample_rows < 0:
        raise ValueError(f"sample_rows must be non-negative, got {sample_rows}")

    raw_questions = _load_questions_from_hf(split=split, difficulties=diffs)
    if not raw_questions:
        raise RuntimeError(
            f"No BIRD questions matched difficulties={diffs} on split={split!r}"
        )

    n = min(n_samples, len(raw_questions))
    rng = random.Random(seed)
    sampled = rng.sample(raw_questions, n)
    sampled.sort(key=lambda q: int(q.get("question_id", 0)))

    referenced_db_ids = {q["db_id"] for q in sampled}
    resolved_db_dir = _resolve_db_dir(
        split=split,
        db_dir_override=db_dir,
        referenced_db_ids=referenced_db_ids,
        auto_download=auto_download,
    )

    questions: list[Question] = []
    schema_cache: dict[str, list[_BirdEvalTable]] = {}
    skipped_missing = 0
    skipped_unrunnable = 0

    for raw in sampled:
        db_id = raw["db_id"]
        db_path = _db_path_for(resolved_db_dir, db_id)
        if db_path is None:
            skipped_missing += 1
            continue

        if db_id not in schema_cache:
            try:
                schema_cache[db_id] = _introspect_db(
                    db_path, sample_rows=sample_rows
                )
            except sqlite3.Error as exc:
                # Bad DB file -- skip every question that targets it.
                schema_cache[db_id] = []
                print(
                    f"[bird] WARN: failed to introspect {db_id}: {exc}",
                    flush=True,
                )
        tables = schema_cache[db_id]
        if not tables:
            skipped_missing += 1
            continue

        entry = _BirdEvalEntry(
            question_id=int(raw.get("question_id", 0)),
            db_id=db_id,
            db_path=db_path,
            question=raw["question"],
            evidence=raw.get("evidence") or "",
            sql=raw["SQL"],
            difficulty=raw.get("difficulty") or "",
            tables=tables,
        )

        if drop_unrunnable and not _gold_sql_runs(entry):
            skipped_unrunnable += 1
            continue

        questions.append(_entry_to_question(entry, max_value_chars=max_value_chars))

    if skipped_missing:
        print(
            f"[bird] dropped {skipped_missing} question(s) with missing / "
            f"unreadable DBs (referenced db_dir={resolved_db_dir})",
            flush=True,
        )
    if skipped_unrunnable:
        print(
            f"[bird] dropped {skipped_unrunnable} question(s) whose gold SQL "
            f"failed to execute (pass drop_unrunnable=False to keep them)",
            flush=True,
        )
    if not questions:
        raise RuntimeError(
            "BIRD subset is empty after filtering. Common causes: no DBs "
            "at the resolved location, all gold SQL failed, or the chosen "
            "(seed, difficulties) sampled only DBs that aren't downloaded."
        )
    return questions


# ---------------------------------------------------------------------------
# HuggingFace + DB loading
# ---------------------------------------------------------------------------


def _load_questions_from_hf(
    *,
    split: str,
    difficulties: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Pull the BIRD HF dataset and return rows as plain dicts."""
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import-time failure
        raise RuntimeError(
            "BIRD eval requires the `datasets` package "
            "(install with `pip install datasets`)"
        ) from exc

    if split == "train":
        ds_name = _HF_DATASET_TRAIN
        split_name = "train"
    else:
        ds_name = _HF_DATASET_DEV
        split_name = "dev_20251106"

    ds = load_dataset(ds_name, split=split_name)
    rows: list[dict[str, Any]] = []
    diff_set = set(difficulties)
    for idx, r in enumerate(ds):
        difficulty = r.get("difficulty") or "simple"
        if difficulty not in diff_set:
            continue
        row = dict(r)
        row.setdefault("question_id", idx)
        row.setdefault("difficulty", difficulty)
        rows.append(row)
    return rows


def _resolve_db_dir(
    *,
    split: str,
    db_dir_override: str | Path | None,
    referenced_db_ids: set[str],
    auto_download: bool,
) -> Path:
    """Find or fetch the directory holding ``<db_id>/<db_id>.sqlite``.

    Resolution order:
        1. ``db_dir_override`` if set.
        2. ``$BIRD_DB_DIR`` env var if set.
        3. ``~/.cache/manysql/bird/<split>/`` if it already exists.
        4. Auto-download (train only, when ``auto_download``). Reuses
           the heavy fetcher from :mod:`train.env.bird` so we don't
           duplicate the ~600 LOC zip-handling pipeline. Lazy-imported
           so eval doesn't pay the cost when the cache is already
           populated.
    """
    candidates: list[Path] = []
    if db_dir_override:
        candidates.append(Path(db_dir_override).expanduser())
    env_root = os.environ.get("BIRD_DB_DIR")
    if env_root:
        candidates.append(Path(env_root).expanduser() / split)
        candidates.append(Path(env_root).expanduser())
    cache_root = Path.home() / ".cache" / "manysql" / "bird" / split
    candidates.append(cache_root)

    for cand in candidates:
        if _looks_like_bird_db_dir(cand, referenced_db_ids):
            return cand

    if split == "train" and auto_download:
        cache_root.mkdir(parents=True, exist_ok=True)
        try:
            from train.env.bird import (  # noqa: PLC0415
                _download_and_extract_train_dbs,
            )
        except ImportError as exc:  # pragma: no cover - source-tree dep
            raise RuntimeError(
                "BIRD train auto-download relies on `train.env.bird` "
                "helpers that aren't on the import path. Either install "
                "the BIRD train DBs manually under "
                f"{cache_root} (see {_TRAIN_DB_HINT}), or pass "
                "--bird-db-dir / set $BIRD_DB_DIR."
            ) from exc
        _download_and_extract_train_dbs(
            cache_root, referenced_db_ids=referenced_db_ids
        )
        if _looks_like_bird_db_dir(cache_root, referenced_db_ids):
            return cache_root
        raise RuntimeError(
            f"BIRD train auto-download finished but {cache_root} "
            f"doesn't have the expected layout (<db_id>/<db_id>.sqlite)."
        )

    if split == "dev":
        raise RuntimeError(
            "BIRD dev databases not found and auto-download is not "
            "supported for the dev split (Google Drive blocks scripted "
            f"access). Download manually from\n  {_DEV_DB_GDRIVE_HINT}\n"
            "and unzip into ~/.cache/manysql/bird/dev/ "
            "(or pass --bird-db-dir / set $BIRD_DB_DIR)."
        )
    raise RuntimeError(
        "BIRD train databases not found and auto-download is disabled. "
        "Either pass --bird-db-dir / set $BIRD_DB_DIR, or rerun with "
        "auto_download=True."
    )


_TRAIN_DB_HINT = (
    "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"
)


def _looks_like_bird_db_dir(path: Path, referenced_db_ids: set[str]) -> bool:
    """True iff at least half of the referenced DBs live under ``path``."""
    if not path.is_dir():
        return False
    if not referenced_db_ids:
        return any((path / sub).is_dir() for sub in path.iterdir())
    hits = 0
    for db_id in referenced_db_ids:
        if (
            (path / db_id / f"{db_id}.sqlite").is_file()
            or (path / db_id / f"{db_id}.db").is_file()
        ):
            hits += 1
    return hits >= max(1, len(referenced_db_ids) // 2)


def _db_path_for(db_dir: Path, db_id: str) -> Path | None:
    candidate = db_dir / db_id / f"{db_id}.sqlite"
    if candidate.is_file():
        return candidate
    alt = db_dir / db_id / f"{db_id}.db"
    if alt.is_file():
        return alt
    return None


def _introspect_db(db_path: Path, *, sample_rows: int) -> list[_BirdEvalTable]:
    """Walk every user table, capture columns/types and a row sample."""
    out: list[_BirdEvalTable] = []
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
        names = [r[0] for r in cur.fetchall()]
        for tbl in names:
            cur.execute(f'PRAGMA table_info("{tbl}")')
            info = cur.fetchall()
            if not info:
                continue
            cols = [c[1] for c in info]
            types = [(c[2] or "").upper() for c in info]
            try:
                cur.execute(f'SELECT COUNT(*) FROM "{tbl}"')
                n_rows = int(cur.fetchone()[0])
            except sqlite3.Error:
                n_rows = 0
            try:
                quoted = ", ".join(f'"{c}"' for c in cols)
                cur.execute(
                    f'SELECT {quoted} FROM "{tbl}" LIMIT {max(0, sample_rows)}'
                )
                rows = cur.fetchall() if sample_rows > 0 else []
            except sqlite3.Error:
                rows = []
            out.append(
                _BirdEvalTable(
                    name=tbl,
                    columns=cols,
                    types=types,
                    sample_rows=rows,
                    n_rows=n_rows,
                )
            )
    finally:
        conn.close()
    return out


def _gold_sql_runs(entry: _BirdEvalEntry) -> bool:
    """True iff the BIRD gold SQL executes without error on the DB."""
    conn = sqlite3.connect(f"file:{entry.db_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute(entry.sql)
        cur.fetchall()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def _entry_to_question(
    entry: _BirdEvalEntry,
    *,
    max_value_chars: int,
) -> Question:
    """Turn a single BIRD entry into a runner-ready :class:`Question`."""
    return Question(
        name=f"bird_{entry.db_id}_{entry.question_id}",
        prompt=_render_user_prompt(entry, max_value_chars=max_value_chars),
        reference_sql={"sqlite": entry.sql.strip()},
        notes=(
            f"BIRD/{entry.db_id}/qid={entry.question_id}/{entry.difficulty}"
            if entry.difficulty
            else f"BIRD/{entry.db_id}/qid={entry.question_id}"
        ),
        db_path=str(entry.db_path),
    )


def _render_user_prompt(entry: _BirdEvalEntry, *, max_value_chars: int) -> str:
    """Inline the per-question schema + sample rows into the user prompt.

    Identifiers are rendered exactly as they appear in the underlying
    DB (no sanitization). Column names that aren't bare-identifier
    safe are shown double-quoted; the LLM should mirror that
    quotation in its SQL.
    """
    lines: list[str] = []
    lines.append(f"Question: {entry.question}")
    if entry.evidence:
        lines.append(f"Evidence: {entry.evidence}")
    lines.append("")
    lines.append(f"Database: {entry.db_id}  (SQLite)")
    if entry.difficulty:
        lines.append(f"Difficulty: {entry.difficulty}")
    lines.append("")
    lines.append("Tables (in this database):")
    for tbl in entry.tables:
        lines.append("")
        lines.append(f"  {_quote_ident(tbl.name)}  ({tbl.n_rows} rows)")
        lines.append("    Columns (name : SQLite type):")
        for c, t in zip(tbl.columns, tbl.types, strict=False):
            lines.append(f"      {_quote_ident(c)} : {t or 'TEXT'}")
        if tbl.sample_rows:
            lines.append(f"    Sample rows ({len(tbl.sample_rows)}):")
            header = " | ".join(_quote_ident(c) for c in tbl.columns)
            sep = " | ".join("---" for _ in tbl.columns)
            lines.append(f"      | {header} |")
            lines.append(f"      | {sep} |")
            for row in tbl.sample_rows:
                cells = [
                    _format_cell(v, max_chars=max_value_chars) for v in row
                ]
                lines.append("      | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Write a single SQLite SELECT (or WITH ... SELECT) that answers the "
        "question against this database. Use the table and column names "
        "exactly as shown above (double-quote any identifier that is "
        "shown quoted). Apply the evidence hint when relevant."
    )
    return "\n".join(lines)


def _quote_ident(name: str) -> str:
    """Render a SQLite identifier as the LLM should write it."""
    if _BARE_IDENT_RE.match(name):
        return name
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def _format_cell(v: Any, *, max_chars: int) -> str:
    if v is None:
        return "NULL"
    s = str(v)
    s = s.replace("|", "\\|").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_chars:
        s = s[: max_chars - 3] + "..."
    return s


__all__ = [
    "_BIRD_EVAL_SCHEMA_PROMPT",
    "select_bird",
]
