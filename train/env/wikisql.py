"""WikiSQL prompt / table / answer pairs as RL training data.

WikiSQL ships ~80k (question, table, SQL) triples scraped from Wikipedia.
Each row carries its own small relational table (~17 rows on average) and
a canonical SQL query that produces the answer. That's a natural fit for
this env: the LLM has to write a SELECT in the chosen manysql dialect
that returns the same rows the reference SQL returns on the example's
table.

Wiring:

* :class:`WikiSqlCatalog` -- a :class:`CatalogProvider` that loads a
  reproducible random subset of N examples from ``Salesforce/wikisql``,
  materializes each one as a uniquely-named Polars table
  (``wikisql_<safe_id>``), and packs them all into a single
  :class:`CatalogSnapshot`. Column names are sanitized to ``c_<safe>``
  so they're valid identifiers in every dialect's grammar regardless of
  reserved words / special characters in the original WikiSQL header.
  ``build()`` is memoized so multiple :class:`DialectRuntime` instances
  can share one catalog without re-downloading.

* :class:`WikiSqlTaskGenerator` -- emits one :class:`SqlTask` per
  example. The user prompt embeds the example's table schema + a few
  sample rows so the model has the same context the question implicitly
  references; the system prompt still gets the dialect card via
  ``runtime.system_prompt()`` (full dialect priors, no grammar dump).

* Gold rows are computed once via the **reference dialect** on the same
  catalog. Data is dialect-independent, so the gold rows are universal
  across dialects -- the same catalog instance can be shared across
  dialect runtimes when building multi-dialect curricula, and the
  cross-product / partition expansion in :mod:`train.grpo_sql` clones
  tasks rather than recomputing.

Loading the dataset requires ``datasets>=2.20``. We import lazily inside
``build()`` so this module is import-cheap on stripped-down environments
(CPU-only smoke boxes that haven't installed ``datasets``).

Why we rebuild SQL from the structured triple instead of using
``sql.human_readable``:

WikiSQL's ``human_readable`` field uses arbitrary column names with
spaces, dots, and ``#`` characters; quoting is inconsistent across the
~80k rows. The structured triple (``sel`` / ``agg`` / ``conds``) is the
canonical form WikiSQL was generated from -- recomposing SQL from it
using our sanitized identifiers guarantees the result parses in every
dialect's grammar. The reference SQL is then run through the
``_reference`` engine to get gold rows; any example whose recomposed
SQL fails to execute is dropped (a small percentage of malformed
WikiSQL entries).
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from train.env.catalog import CatalogProvider, CatalogSnapshot
from train.env.engine import DialectRuntime
from train.env.tasks import SqlTask, TaskGenerator
from train.env.types import TaskMeta

# WikiSQL aggregator opcodes -> SQL function names. Index 0 = no aggregate.
# Source: https://github.com/salesforce/WikiSQL/blob/master/lib/query.py
_WIKISQL_AGG_OPS: tuple[str, ...] = ("", "MAX", "MIN", "COUNT", "SUM", "AVG")
# WikiSQL operator opcodes -> SQL operators. Same source.
_WIKISQL_COND_OPS: tuple[str, ...] = ("=", ">", "<")


# ---------------------------------------------------------------------------
# Sanitization helpers
# ---------------------------------------------------------------------------


def _safe_ident(raw: str, *, fallback: str = "x") -> str:
    """Map an arbitrary string to a safe SQL identifier.

    Strips diacritics, replaces non-alphanumeric runs with underscores,
    lowercases, and prepends ``c_`` so reserved-word collisions are
    avoided regardless of dialect. Empty results fall back to
    ``c_<fallback>`` (e.g. when the original heading was a single
    non-ASCII glyph that gets stripped entirely).
    """
    norm = unicodedata.normalize("NFKD", raw or "")
    norm = norm.encode("ascii", "ignore").decode("ascii")
    norm = re.sub(r"[^a-zA-Z0-9]+", "_", norm).strip("_").lower()
    if not norm:
        norm = fallback
    return f"c_{norm}"


def _safe_table_name(raw: str) -> str:
    norm = re.sub(r"[^a-zA-Z0-9]+", "_", raw or "").strip("_").lower()
    if not norm:
        norm = "anon"
    return f"wikisql_{norm}"


def _dedupe_columns(names: list[str]) -> list[str]:
    """Resolve duplicate sanitized column names by suffixing _1, _2, ...

    WikiSQL occasionally has tables with two columns whose original
    headers collapse to the same sanitized form (e.g. "Score (1)" and
    "Score (2)" both become ``c_score``). Deduping keeps the first
    occurrence and suffixes the rest so the IR schema stays unique.
    """
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen[name] = 0
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name}_{seen[name]}")
    return out


# ---------------------------------------------------------------------------
# WikiSQL catalog
# ---------------------------------------------------------------------------


@dataclass
class WikiSqlEntry:
    """One materialized WikiSQL example (post-sanitization).

    Held by :class:`WikiSqlCatalog` after ``build()``; consumed by
    :class:`WikiSqlTaskGenerator` to render the per-task prompt and
    rebuild canonical SQL.
    """

    raw_id: str
    table_name: str
    original_header: list[str]
    safe_header: list[str]
    types: list[str]
    sample_rows: list[dict[str, Any]]
    n_rows: int
    question: str
    sel: int
    agg: int
    conds: list[dict[str, Any]]


class WikiSqlCatalog(CatalogProvider):
    """Catalog provider backed by N WikiSQL examples.

    All sampled tables are loaded into one :class:`CatalogSnapshot` with
    unique table names ``wikisql_<safe_id>``. The snapshot's
    ``schema_prompt`` is a brief placeholder -- the real per-task schema
    lives in the user message (each task references one specific
    table), not the system prompt.

    Memoization: ``build()`` caches its snapshot, so multiple
    :class:`DialectRuntime` instances created with the same catalog
    instance share one materialized copy. Important for multi-dialect
    cross-product training, where N dialects would otherwise re-download
    + re-materialize WikiSQL N times.
    """

    name = "wikisql"

    def __init__(
        self,
        *,
        n_samples: int = 1000,
        split: str = "train",
        seed: int = 0,
        dataset_name: str = "Salesforce/wikisql",
        # The Salesforce/wikisql repo ships a legacy loading script
        # (``wikisql.py``) that ``datasets >= 4.0`` no longer supports.
        # HF auto-converts every script-based dataset to parquet on the
        # ``refs/convert/parquet`` branch; pinning to that revision is
        # the recommended workaround.
        revision: str | None = "refs/convert/parquet",
        sample_rows: int = 3,
    ) -> None:
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if split not in {"train", "validation", "test"}:
            raise ValueError(
                f"split must be 'train' | 'validation' | 'test', got {split!r}"
            )
        if sample_rows < 0:
            raise ValueError(f"sample_rows must be non-negative, got {sample_rows}")
        self.n_samples = n_samples
        self.split = split
        self.seed = seed
        self.dataset_name = dataset_name
        self.revision = revision
        self.sample_rows_n = sample_rows
        self._snapshot: CatalogSnapshot | None = None
        self._entries: list[WikiSqlEntry] = []

    def build(self) -> CatalogSnapshot:
        if self._snapshot is not None:
            return self._snapshot

        from datasets import load_dataset  # noqa: PLC0415

        from manysql.ir.plan import ColumnSchema  # noqa: PLC0415

        load_kwargs: dict[str, Any] = {"split": self.split}
        if self.revision is not None:
            load_kwargs["revision"] = self.revision
        ds = load_dataset(self.dataset_name, **load_kwargs)
        # Reproducible sampling: deterministic random subset of indices.
        # WikiSQL is on the order of 80k rows, so a Python rng.sample is
        # fine; HF Dataset.shuffle would also work but is slightly more
        # opaque about which rows ended up selected.
        n = min(self.n_samples, len(ds))
        rng = random.Random(self.seed)
        idxs = rng.sample(range(len(ds)), n)

        tables: dict[str, Any] = {}
        schemas: dict[str, tuple[ColumnSchema, ...]] = {}
        entries: list[WikiSqlEntry] = []
        used_table_names: dict[str, int] = {}

        for idx in idxs:
            row = ds[idx]
            table = row["table"]
            sql = row["sql"]
            raw_id = table.get("id") or f"i{idx}"
            base_name = _safe_table_name(raw_id)
            # Different examples can share a table id (e.g. multiple
            # questions over the same Wikipedia table). Suffix on
            # collision so each task gets a uniquely named table.
            if base_name in used_table_names:
                used_table_names[base_name] += 1
                table_name = f"{base_name}_{used_table_names[base_name]}"
            else:
                used_table_names[base_name] = 0
                table_name = base_name

            df, safe_header = self._materialize_table(table)
            if df is None:
                continue
            ir_cols = tuple(
                ColumnSchema(name=col, type=_ir_type_for(t))
                for col, t in zip(safe_header, table["types"], strict=False)
            )
            tables[table_name] = df
            schemas[table_name] = ir_cols

            sample = (
                df.head(self.sample_rows_n).to_dicts()
                if self.sample_rows_n > 0
                else []
            )
            entries.append(
                WikiSqlEntry(
                    raw_id=raw_id,
                    table_name=table_name,
                    original_header=list(table["header"]),
                    safe_header=safe_header,
                    types=list(table["types"]),
                    sample_rows=sample,
                    n_rows=df.height,
                    question=row["question"],
                    sel=int(sql["sel"]),
                    agg=int(sql["agg"]),
                    conds=_normalize_conds(sql.get("conds", {})),
                )
            )

        self._entries = entries
        self._snapshot = CatalogSnapshot(
            tables=tables,
            schemas=schemas,
            schema_prompt=_WIKISQL_SCHEMA_PROMPT,
        )
        return self._snapshot

    def entries(self) -> list[WikiSqlEntry]:
        if self._snapshot is None:
            self.build()
        return list(self._entries)

    def _materialize_table(
        self, table: dict[str, Any]
    ) -> tuple[Any, list[str]]:
        """Build a Polars frame for one WikiSQL example.

        Returns ``(df, safe_header)`` or ``(None, [])`` if the table is
        empty / malformed (header missing). Cells in ``real``-typed
        columns are coerced to ``Float64`` with ``strict=False``;
        unparseable values become null instead of crashing the build.
        """
        import polars as pl  # noqa: PLC0415

        header = table["header"]
        types = table["types"]
        rows = table["rows"]
        if not header:
            return None, []
        safe_header = _dedupe_columns(
            [_safe_ident(h, fallback=f"col{i}") for i, h in enumerate(header)]
        )

        cols: dict[str, list[Any]] = {n: [] for n in safe_header}
        for row in rows:
            for i, name in enumerate(safe_header):
                cell = row[i] if i < len(row) else None
                cols[name].append(cell)

        series: list[Any] = []
        for name, t in zip(safe_header, types, strict=False):
            raw = cols[name]
            if t == "real":
                # WikiSQL numeric cells arrive as a mix of int / float /
                # str / None depending on the source row. Coerce in
                # Python before building the Series so polars doesn't
                # have to fight a heterogeneous Object column; bad
                # values become None instead of aborting the whole
                # build for one bad row in 80k.
                coerced = [_coerce_float(v) for v in raw]
                series.append(pl.Series(name, coerced, dtype=pl.Float64))
            else:
                series.append(
                    pl.Series(
                        name,
                        [None if v is None else str(v) for v in raw],
                        dtype=pl.Utf8,
                    )
                )

        df = pl.DataFrame(series)
        return df, safe_header


def _ir_type_for(wikisql_type: str) -> Any:
    from manysql.ir.types import FLOAT, TEXT  # noqa: PLC0415

    return FLOAT if wikisql_type == "real" else TEXT


def _coerce_float(v: Any) -> float | None:
    """Coerce a WikiSQL ``real`` cell to ``float`` or ``None``."""
    if v is None:
        return None
    if isinstance(v, bool):  # bool subclasses int; not useful as a real
        return float(v)
    if isinstance(v, int | float):
        return float(v)
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def _normalize_conds(conds: Any) -> list[dict[str, Any]]:
    """Normalize WikiSQL conditions to a list of row dicts.

    HF's loader exposes the conds either as a columnar struct
    (``{column_index: [...], operator_index: [...], condition: [...]}``)
    or already-unwrapped to a list of dicts depending on version. Handle
    both shapes.
    """
    if isinstance(conds, list):
        return list(conds)
    col_idx = conds.get("column_index", []) or []
    op_idx = conds.get("operator_index", []) or []
    cond = conds.get("condition", []) or []
    out: list[dict[str, Any]] = []
    for c, o, v in zip(col_idx, op_idx, cond, strict=False):
        out.append({"column_index": int(c), "operator_index": int(o), "condition": v})
    return out


_WIKISQL_SCHEMA_PROMPT = (
    "Each task targets one Wikipedia-derived table from the WikiSQL "
    "corpus. The relevant table's name, columns, types, and a few "
    "sample rows are provided in the user message; reference that "
    "block, not a global schema. All column names have been sanitized "
    "to ``c_<lowercase_alnum>`` form so they're safe identifiers in "
    "every dialect's grammar."
)


# ---------------------------------------------------------------------------
# WikiSQL task generator
# ---------------------------------------------------------------------------


@dataclass
class WikiSqlTaskConfig:
    """Configuration for :class:`WikiSqlTaskGenerator`.

    ``catalog`` can be passed in pre-built so it's shared across
    dialect runtimes (the recommended path for multi-dialect curricula:
    build the catalog once, build N dialect runtimes against it, clone
    tasks per dialect). Pass ``None`` to let the generator construct
    its own with the same n_samples/split/seed.

    ``drop_empty`` filters out tasks whose recomposed gold SQL returns
    no rows on the example's table. WikiSQL has a small fraction of
    such cases (typically when the conditions reference values absent
    from the table after string-vs-real normalization); empty-result
    tasks are a poor training signal because "any SELECT that returns
    nothing" matches.
    """

    target_dialect: str
    n_samples: int = 1000
    split: str = "train"
    seed: int = 0
    sample_rows: int = 3
    reference_dialect: str = "_reference"
    catalog: WikiSqlCatalog | None = None
    drop_empty: bool = True
    extra_meta: dict[str, Any] = field(default_factory=dict)


class WikiSqlTaskGenerator(TaskGenerator):
    """Build NL->SQL tasks from WikiSQL.

    Each task pairs a natural-language question with its table schema
    (in the user prompt) and an executable canonical gold SQL. Gold
    rows are computed once via the reference dialect on the shared
    catalog; the resulting :class:`SqlTask` list is dialect-tagged via
    ``config.target_dialect`` but the rows are dialect-independent --
    higher layers (e.g. :func:`train.grpo_sql.build_runtimes_and_tasks`)
    clone tasks across additional dialects without re-running the
    reference engine.
    """

    name = "wikisql"

    def __init__(self, config: WikiSqlTaskConfig) -> None:
        self.config = config
        self.catalog = config.catalog or WikiSqlCatalog(
            n_samples=config.n_samples,
            split=config.split,
            seed=config.seed,
            sample_rows=config.sample_rows,
        )
        self._tasks: list[SqlTask] = []
        self._built = False

    def build(self) -> None:
        if self._built:
            return
        self.catalog.build()
        ref_runtime = DialectRuntime(
            dialect=self.config.reference_dialect, catalog=self.catalog
        )
        ref_runtime.setup()
        try:
            for entry in self.catalog.entries():
                gold_sql = _build_canonical_sql(entry)
                run = ref_runtime.run(gold_sql)
                if not run.exec_result.success:
                    continue
                if self.config.drop_empty and not run.exec_result.rows:
                    continue
                meta = TaskMeta(
                    task_id=f"wikisql_{entry.raw_id}",
                    dialect=self.config.target_dialect,
                    generator=self.name,
                    meta={
                        "table_name": entry.table_name,
                        "n_rows": entry.n_rows,
                        "split": self.config.split,
                        "reference_dialect": self.config.reference_dialect,
                        **self.config.extra_meta,
                    },
                )
                prompt = _render_user_prompt(entry)
                self._tasks.append(
                    SqlTask(
                        meta=meta,
                        prompt=prompt,
                        gold_rows=run.exec_result.rows,
                        gold_sql=gold_sql,
                        catalog=self.catalog,
                    )
                )
        finally:
            ref_runtime.teardown()
        self._built = True

    def all_tasks(self) -> list[SqlTask]:
        if not self._built:
            self.build()
        return list(self._tasks)


# ---------------------------------------------------------------------------
# SQL builder + prompt renderer
# ---------------------------------------------------------------------------


def _build_canonical_sql(entry: WikiSqlEntry) -> str:
    """Reconstruct a canonical SQL for one WikiSQL example.

    WikiSQL's ``human_readable`` field is unreliable -- it uses
    arbitrary column names with spaces, dots, and ``#`` characters and
    its quoting is inconsistent across the corpus. Instead we rebuild
    SQL from the structured ``sel`` / ``agg`` / ``conds`` triple using
    the sanitized identifiers, which guarantees the result parses in
    every dialect's grammar.
    """
    if not (0 <= entry.sel < len(entry.safe_header)):
        # Out-of-bounds sel index -- malformed example. Return a
        # syntactically-valid but semantically-empty SQL so the
        # reference engine fails it cleanly and the generator drops it.
        return f"SELECT * FROM {entry.table_name} WHERE 1 = 0"

    sel_col = entry.safe_header[entry.sel]
    if 0 < entry.agg < len(_WIKISQL_AGG_OPS):
        sel_expr = f"{_WIKISQL_AGG_OPS[entry.agg]}({sel_col})"
    else:
        sel_expr = sel_col

    parts = [f"SELECT {sel_expr} FROM {entry.table_name}"]
    cond_parts: list[str] = []
    for cond in entry.conds:
        col_idx = cond["column_index"]
        op_idx = cond["operator_index"]
        if not (0 <= col_idx < len(entry.safe_header)):
            continue
        if not (0 <= op_idx < len(_WIKISQL_COND_OPS)):
            continue
        col = entry.safe_header[col_idx]
        op = _WIKISQL_COND_OPS[op_idx]
        val = cond["condition"]
        col_type = entry.types[col_idx] if col_idx < len(entry.types) else "text"
        if col_type == "real":
            try:
                lit = str(float(val))
            except (TypeError, ValueError):
                # Occasional non-numeric strings in real-typed condition
                # slots. Quote them; reference engine will likely fail
                # the cast and we'll drop the task.
                lit = "'" + str(val).replace("'", "''") + "'"
        else:
            lit = "'" + str(val).replace("'", "''") + "'"
        cond_parts.append(f"{col} {op} {lit}")
    if cond_parts:
        parts.append("WHERE " + " AND ".join(cond_parts))
    return " ".join(parts)


def _render_user_prompt(entry: WikiSqlEntry) -> str:
    """Format the per-task user prompt: question + schema + sample rows."""
    lines: list[str] = []
    lines.append(f"Question: {entry.question}")
    lines.append("")
    lines.append(f"Table: {entry.table_name}")
    lines.append("Columns (sanitized name <- original heading; type):")
    for safe, original, t in zip(
        entry.safe_header, entry.original_header, entry.types, strict=False
    ):
        lines.append(f"  {safe}  <-  {original}  ({_pretty_type(t)})")
    if entry.sample_rows:
        lines.append("")
        lines.append("Sample rows:")
        cols = entry.safe_header
        header = " | ".join(cols)
        sep = " | ".join("---" for _ in cols)
        lines.append(f"| {header} |")
        lines.append(f"| {sep} |")
        for row in entry.sample_rows:
            cells = [_format_cell(row.get(c)) for c in cols]
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"(Total rows in {entry.table_name}: {entry.n_rows})")
    lines.append("")
    lines.append(
        f"Write a SELECT in the target dialect that answers the question. "
        f"Reference the table as `{entry.table_name}` and use the "
        f"sanitized column names (the c_* identifiers) above."
    )
    return "\n".join(lines)


def _pretty_type(t: str) -> str:
    return {"real": "FLOAT", "text": "TEXT"}.get(t, t.upper() or "TEXT")


def _format_cell(v: Any) -> str:
    if v is None:
        return "NULL"
    s = str(v)
    s = s.replace("|", "\\|").replace("\n", " ")
    if len(s) > 30:
        s = s[:27] + "..."
    return s


__all__ = [
    "WikiSqlCatalog",
    "WikiSqlEntry",
    "WikiSqlTaskConfig",
    "WikiSqlTaskGenerator",
]
