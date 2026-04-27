"""SQL tasks and task generators.

A :class:`SqlTask` is the unit of work the RL env hands an agent: a
prompt, a target dialect, and a reference result-set the env scores
the agent's answer against.

Two task generators ship out of the box:

* :class:`GoldenTaskGenerator` - cross-dialect translation tasks built
  from ``manysql.golden.queries.GOLDEN_QUERIES``. Gold rows are
  computed by running each canonical query through the **reference**
  dialect on the same catalog the candidate dialect will see, so the
  env compares apples to apples (same engine, same data, only the
  surface differs). The agent's job is "rewrite this near-ANSI SQL so
  it parses + executes in the target dialect and yields the same
  rows." Best signal for training a model to *speak* a dialect.

* :class:`EvalSuiteTaskGenerator` - NL-question tasks lifted from
  ``eval.dataset.questions``. Gold rows come from running the
  question's SQLite-flavored reference SQL through an in-memory
  SQLite executor (the same one ``eval/runner.py`` uses for ground
  truth on synthetic dialects). Best signal for end-to-end NL->SQL
  benchmarking against a synthetic dialect.

Both generators precompute gold rows up front so ``SqlEnv.step()``
never has to call a reference engine on the hot path.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from train.env.catalog import CatalogProvider, GithubEventsCatalog, GoldenCatalog
from train.env.engine import DialectRuntime
from train.env.types import TaskMeta

if TYPE_CHECKING:
    pass


@dataclass
class SqlTask:
    """One RL episode: a prompt + a target dialect + gold rows.

    Fields:
        meta            identity / provenance (dialect, generator, task_id)
        prompt          user-message text for the LLM (NL question OR
                        "translate this reference SQL" instruction)
        gold_rows       reference result rows the env compares against
        gold_sql        the canonical/reference SQL that produced
                        ``gold_rows`` (kept for analysis + as a
                        teacher-forcing target)
        catalog         the catalog provider this task was built for;
                        SqlEnv reuses it for the candidate dialect's
                        runtime so columns/dtypes line up
        notes           free-form annotation surfaced in transcripts
    """

    meta: TaskMeta
    prompt: str
    gold_rows: list[dict[str, Any]]
    gold_sql: str
    catalog: CatalogProvider
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta.to_dict(),
            "prompt": self.prompt,
            "gold_rows": self.gold_rows,
            "gold_sql": self.gold_sql,
            "catalog": self.catalog.name,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Task generator ABC
# ---------------------------------------------------------------------------


class TaskGenerator(ABC):
    """ABC for task generators.

    Subclasses materialize tasks lazily. The env calls ``build()`` once
    to load any heavy state (gold-row reference engine, etc.), then
    ``sample()`` per episode.
    """

    name: str = "abstract"

    @abstractmethod
    def build(self) -> None:
        """Heavy-lifting: spin up reference engines, materialize gold rows."""

    @abstractmethod
    def all_tasks(self) -> list[SqlTask]:
        """Return every task this generator can emit, in deterministic order."""

    def sample(self, *, seed: int | None = None) -> SqlTask:
        rng = random.Random(seed)
        tasks = self.all_tasks()
        if not tasks:
            raise RuntimeError(f"{self.name}: no tasks available")
        return rng.choice(tasks)

    def get(self, task_id: str) -> SqlTask:
        for t in self.all_tasks():
            if t.meta.task_id == task_id:
                return t
        raise KeyError(f"{self.name}: no task with id {task_id!r}")


# ---------------------------------------------------------------------------
# GoldenTaskGenerator: cross-dialect translation tasks
# ---------------------------------------------------------------------------


@dataclass
class GoldenTaskConfig:
    """Configuration knobs for GoldenTaskGenerator.

    ``categories`` filters by the golden query's bucket (scan / filter /
    project / join / aggregate / sort_limit / distinct / set_op / cte /
    subquery / window / semantic). Default: all categories whose SQL is
    cross-dialect (``cross_dialect=True``).

    ``include_non_cross_dialect`` opts into the small set of golden
    queries flagged ``cross_dialect=False`` (e.g. FILTER aggregates,
    FULL OUTER JOIN). They still execute on the reference dialect, so
    gold rows exist; whether the candidate dialect's grammar accepts
    them depends on the dialect.
    """

    target_dialect: str
    reference_dialect: str = "_reference"
    categories: list[str] | None = None
    include_non_cross_dialect: bool = False
    prompt_template: str = (
        "Translate the following reference SQL to the target dialect described "
        "above. The result rows must be identical.\n\n"
        "Reference SQL:\n{ref_sql}"
    )


class GoldenTaskGenerator(TaskGenerator):
    """Build cross-dialect translation tasks from the golden corpus.

    For each ``GoldenQuery``:
      1. The canonical SQL is parsed + executed by the **reference**
         dialect on the manysql 5-table catalog.
      2. The resulting rows become ``gold_rows``.
      3. ``prompt`` is the canonical SQL wrapped in a "translate this"
         instruction (see ``GoldenTaskConfig.prompt_template``).

    The agent is then asked to produce SQL in the **target** dialect
    that, when executed on the same catalog through the candidate
    dialect's engine, returns the same rows.

    Why this is good RL signal: the dialect card already tells the LLM
    *how* the surface differs; the gold result tells it whether it got
    the translation right; failures come back as parse / runtime errors
    that the model can iterate on across turns.
    """

    name = "golden"

    def __init__(self, config: GoldenTaskConfig) -> None:
        self.config = config
        self.catalog = GoldenCatalog()
        self._tasks: list[SqlTask] = []
        self._built = False

    def build(self) -> None:
        if self._built:
            return
        from manysql.golden.queries import GOLDEN_QUERIES  # noqa: PLC0415

        ref_runtime = DialectRuntime(
            dialect=self.config.reference_dialect, catalog=self.catalog
        )
        ref_runtime.setup()
        try:
            for gq in GOLDEN_QUERIES:
                if (
                    self.config.categories is not None
                    and gq.category not in self.config.categories
                ):
                    continue
                if not gq.cross_dialect and not self.config.include_non_cross_dialect:
                    continue
                # Materialize gold rows by running the canonical SQL through
                # the reference dialect engine on the same catalog the env
                # will hand the candidate. This keeps everything (engine,
                # data, semantics) the same except the dialect surface.
                run = ref_runtime.run(gq.sql)
                if not run.exec_result.success:
                    # A query the reference dialect can't run is not a
                    # tractable training task; skip it loudly enough to
                    # surface codegen regressions, but don't crash the
                    # whole generator.
                    continue
                meta = TaskMeta(
                    task_id=gq.id,
                    dialect=self.config.target_dialect,
                    generator=self.name,
                    meta={
                        "category": gq.category,
                        "cross_dialect": gq.cross_dialect,
                        "notes": gq.notes,
                        "reference_dialect": self.config.reference_dialect,
                    },
                )
                prompt = self.config.prompt_template.format(ref_sql=gq.sql)
                self._tasks.append(
                    SqlTask(
                        meta=meta,
                        prompt=prompt,
                        gold_rows=run.exec_result.rows,
                        gold_sql=gq.sql,
                        catalog=self.catalog,
                        notes=gq.notes,
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
# EvalSuiteTaskGenerator: NL-question tasks
# ---------------------------------------------------------------------------


@dataclass
class EvalSuiteTaskConfig:
    """Configuration knobs for EvalSuiteTaskGenerator.

    ``names`` and ``limit`` mirror ``eval.dataset.questions.select`` so
    the same subset selection works in both eval and training contexts.

    ``catalog`` defaults to a fresh ``GithubEventsCatalog`` with the
    eval defaults (5000 rows, seed 0xDB). Override to bump rows for
    harder grouping queries or change the seed.
    """

    target_dialect: str
    names: list[str] | None = None
    limit: int | None = None
    catalog: GithubEventsCatalog = field(default_factory=GithubEventsCatalog)


class EvalSuiteTaskGenerator(TaskGenerator):
    """Build NL->SQL tasks from the eval question suite.

    Reuses the question + reference-SQL pairs from
    ``eval.dataset.questions``. Gold rows come from an in-memory
    SQLite executor running each question's reference SQL - same
    ground-truth strategy ``eval/runner.py`` uses when the candidate
    backend is a synthetic manysql dialect.

    The catalog provider is :class:`GithubEventsCatalog`, which seeds
    the same Polars table with the same seed/row-count contract.
    """

    name = "eval_suite"

    def __init__(self, config: EvalSuiteTaskConfig) -> None:
        self.config = config
        self.catalog = config.catalog
        self._tasks: list[SqlTask] = []
        self._built = False

    def build(self) -> None:
        if self._built:
            return
        from eval.dataset.questions import select  # noqa: PLC0415
        from eval.executors.sqlite_executor import SqliteExecutor  # noqa: PLC0415

        questions = select(self.config.names, limit=self.config.limit)

        # Spin up a SQLite reference executor with the *same* seed/row
        # count as the catalog provider so gold rows align with what
        # the candidate dialect's runtime sees.
        sqlite_exec = SqliteExecutor(
            seed=self.catalog.seed, n_rows=self.catalog.n_rows
        )
        sqlite_exec.setup()
        try:
            for q in questions:
                ref_sql = q.reference_sql.get("sqlite")
                if not ref_sql:
                    continue
                ref = sqlite_exec.execute(ref_sql)
                if not ref.success:
                    continue
                meta = TaskMeta(
                    task_id=q.name,
                    dialect=self.config.target_dialect,
                    generator=self.name,
                    meta={
                        "reference_backend": "sqlite",
                        "notes": q.notes,
                    },
                )
                self._tasks.append(
                    SqlTask(
                        meta=meta,
                        prompt=q.prompt,
                        gold_rows=ref.rows,
                        gold_sql=ref_sql,
                        catalog=self.catalog,
                        notes=q.notes,
                    )
                )
        finally:
            sqlite_exec.teardown()
        self._built = True

    def all_tasks(self) -> list[SqlTask]:
        if not self._built:
            self.build()
        return list(self._tasks)


__all__ = [
    "EvalSuiteTaskConfig",
    "EvalSuiteTaskGenerator",
    "GoldenTaskConfig",
    "GoldenTaskGenerator",
    "SqlTask",
    "TaskGenerator",
]
