"""``train.env``: an RL environment over manysql synthetic SQL dialects.

Public surface, in roughly the order callers reach for things:

* :class:`SqlTask`, :class:`TaskGenerator`,
  :class:`GoldenTaskGenerator`, :class:`EvalSuiteTaskGenerator`,
  :class:`GoldenTaskConfig`, :class:`EvalSuiteTaskConfig`
    -- define what an episode is (a (dialect, prompt, gold_rows) tuple).

* :class:`CatalogProvider`, :class:`GoldenCatalog`,
  :class:`GithubEventsCatalog`, :class:`CatalogSnapshot`
    -- supply the in-memory tables + schemas + a schema prompt.

* :class:`DialectRuntime`, :class:`RunResult`
    -- parse + lower + execute candidate SQL through a chosen dialect.

* :class:`SqlEnv`, :class:`InitialObservation`
    -- the gym-style episode driver. Owns reset/step/done/observation
       and produces an :class:`EpisodeResult` at the end.

* :class:`RewardConfig`, :class:`RewardBreakdown`, :func:`compute_reward`
    -- map a transcript + final comparison to a scalar (with components).

* :class:`Policy`, :class:`FixedSqlPolicy`, :class:`LLMPolicy`,
  :class:`RolloutResult`, :func:`run_episode`
    -- glue for driving an env with an agent and producing a chat-style
       transcript that GRPO trainers can consume.

* :class:`WikiSqlCatalog`, :class:`WikiSqlTaskGenerator`,
  :class:`WikiSqlTaskConfig`, :class:`WikiSqlEntry`
    -- WikiSQL prompt / table / answer triples as RL tasks. Each
       example brings its own small Wikipedia table; the catalog packs
       N of them into one snapshot. See :mod:`train.env.wikisql`.

* :class:`BirdCatalog`, :class:`BirdTaskGenerator`,
  :class:`BirdTaskConfig`, :class:`BirdEntry`, :class:`BirdTableInfo`
    -- BIRD-SQL prompt / multi-table-database / answer triples as RL
       tasks. Strictly harder than WikiSQL: real Kaggle-style schemas
       (5-25 tables per DB) plus an "evidence" field carrying domain
       semantics. See :mod:`train.env.bird`.

Importing this module loads only stub deps (dataclasses, typing); the
heavy work (Lark, polars, the dialect package, ``datasets`` for
WikiSQL/BIRD, stdlib ``sqlite3`` for BIRD database files) happens
when you call ``DialectRuntime.setup()`` or ``CatalogProvider.build()``.
"""

from __future__ import annotations

from train.env.catalog import (
    CatalogProvider,
    CatalogSnapshot,
    GithubEventsCatalog,
    GoldenCatalog,
)
from train.env.engine import DialectRuntime, RunResult
from train.env.rewards import (
    RewardBreakdown,
    RewardConfig,
    RewardMode,
    compute_reward,
)
from train.env.rollout import (
    FixedSqlPolicy,
    LLMPolicy,
    Policy,
    RolloutResult,
    run_episode,
)
from train.env.sql_env import InitialObservation, SqlEnv
from train.env.tasks import (
    EvalSuiteTaskConfig,
    EvalSuiteTaskGenerator,
    GoldenTaskConfig,
    GoldenTaskGenerator,
    SqlTask,
    TaskGenerator,
)
from train.env.trl import (
    REWARD_COMPONENTS,
    TRL_AGENT_BASE_RULES,
    make_reward_funcs,
    make_run_sql_tool,
    reconstruct_turns,
    score_completion,
    tasks_to_dataset,
    trl_agent_system_prompt,
)
from train.env.bird import (
    BirdCatalog,
    BirdEntry,
    BirdTableInfo,
    BirdTaskConfig,
    BirdTaskGenerator,
)
from train.env.types import EpisodeResult, StepResult, TaskMeta, Turn
from train.env.wikisql import (
    WikiSqlCatalog,
    WikiSqlEntry,
    WikiSqlTaskConfig,
    WikiSqlTaskGenerator,
)

__all__ = [
    "REWARD_COMPONENTS",
    "TRL_AGENT_BASE_RULES",
    "BirdCatalog",
    "BirdEntry",
    "BirdTableInfo",
    "BirdTaskConfig",
    "BirdTaskGenerator",
    "CatalogProvider",
    "CatalogSnapshot",
    "DialectRuntime",
    "EpisodeResult",
    "EvalSuiteTaskConfig",
    "EvalSuiteTaskGenerator",
    "FixedSqlPolicy",
    "GithubEventsCatalog",
    "GoldenCatalog",
    "GoldenTaskConfig",
    "GoldenTaskGenerator",
    "InitialObservation",
    "LLMPolicy",
    "Policy",
    "RewardBreakdown",
    "RewardConfig",
    "RewardMode",
    "RolloutResult",
    "RunResult",
    "SqlEnv",
    "SqlTask",
    "StepResult",
    "TaskGenerator",
    "TaskMeta",
    "Turn",
    "WikiSqlCatalog",
    "WikiSqlEntry",
    "WikiSqlTaskConfig",
    "WikiSqlTaskGenerator",
    "compute_reward",
    "make_reward_funcs",
    "make_run_sql_tool",
    "reconstruct_turns",
    "run_episode",
    "score_completion",
    "tasks_to_dataset",
    "trl_agent_system_prompt",
]
