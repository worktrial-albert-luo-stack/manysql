# `train/env/`: a SQL RL environment over manysql synthetic dialects

Drop-in environment abstraction for training LLMs to write SQL in any
synthetic dialect produced by `manysql.codegen`. One agent, one task,
one dialect, multi-turn: the agent reads a system prompt + a question,
emits SQL, and gets back either a row-count summary (success) or a
parse / runtime error trace it can iterate on next turn. Reward is a
function of correctness *and* the number of turns it took to get there.

## Architecture

```
            ┌────────────────────────────────────────────────────────────┐
            │                    train.env (this dir)                   │
            │                                                            │
  Task gen  │   GoldenTaskGenerator ──┐                                  │
  (preload) │                         ├──► SqlTask                       │
            │   EvalSuiteTaskGenerator┘   (prompt, gold_rows, dialect)   │
            │                                                            │
            │                                                            │
  Catalogs  │   GoldenCatalog          ──► CatalogSnapshot               │
            │   GithubEventsCatalog        (tables, schemas, prompt)     │
            │                                                            │
            │                                                            │
  Runtime   │   DialectRuntime ── parse(grammar.lark)                    │
            │                  ── lower(tree, semantics, schemas)        │
            │                  ── execute(plan, catalog, overrides,...)  │
            │                  → ExecResult + error_class                │
            │                                                            │
            │                                                            │
  Episode   │   SqlEnv:  reset() ─┬─► InitialObservation                 │
            │            step(s) ─┴─► StepResult (turn, observation, …)  │
            │                                                            │
            │   compute_reward(transcript, comparison, …)                │
            │                                                            │
            │                                                            │
  Rollout   │   Policy → FixedSqlPolicy / LLMPolicy                      │
            │   run_episode(env, policy) → RolloutResult                 │
            └────────────────────────────────────────────────────────────┘
                              ▲                  ▲
                              │                  │
                         manysql.dialects   manysql.executor
                         (DialectEngine)    (Polars IR engine)
```

The env reuses three things from existing code (no copies, no rewrites):

* **`manysql.dialects.DialectRegistry`** loads the per-dialect grammar /
  lowering / semantics / overrides / passes / effects bundle.
* **`manysql.executor`** runs the lowered IR plan against the in-memory
  catalog using the same code path that `eval` and the oracle harness use.
* **`eval.executors.base.ExecResult`** is the result shape — kept identical
  so reward functions, validators, and transcript renderers don't fork.

## Public surface

```python
from train.env import (
    # data shapes
    SqlTask, TaskMeta, Turn, StepResult, EpisodeResult,
    # catalogs
    CatalogProvider, GoldenCatalog, GithubEventsCatalog,
    WikiSqlCatalog, WikiSqlEntry,
    # runtime
    DialectRuntime,
    # tasks
    GoldenTaskGenerator, GoldenTaskConfig,
    EvalSuiteTaskGenerator, EvalSuiteTaskConfig,
    WikiSqlTaskGenerator, WikiSqlTaskConfig,
    # env + rewards
    SqlEnv, InitialObservation,
    RewardConfig, RewardBreakdown, compute_reward,
    # offline rollouts (eval / debug; NOT used at training time)
    Policy, FixedSqlPolicy, LLMPolicy, run_episode,
    # TRL GRPOTrainer adapter (see "Plugging into TRL" below)
    make_run_sql_tool, make_reward_funcs, reconstruct_turns,
    score_completion, tasks_to_dataset, trl_agent_system_prompt,
)
```

## Quick start

```python
from train.env import (
    DialectRuntime, GoldenTaskGenerator, GoldenTaskConfig,
    SqlEnv, FixedSqlPolicy, run_episode,
)

# 1. Build tasks for a target dialect (precomputes gold rows once).
gen = GoldenTaskGenerator(GoldenTaskConfig(target_dialect="aggressive_alien"))
gen.build()
task = gen.all_tasks()[0]

# 2. Spin up the dialect runtime once; reuse it across many episodes.
runtime = DialectRuntime(dialect="aggressive_alien", catalog=task.catalog)
runtime.setup()

# 3. Run an episode.
env = SqlEnv(task=task, runtime=runtime, max_turns=3)
policy = FixedSqlPolicy(task.gold_sql)  # or LLMPolicy(eval.llm.LLMClient(...))
result = run_episode(env=env, policy=policy)

print(result.episode.matched, result.episode.reward, result.episode.reward_components)
```

CLI smoke test:

```bash
python -m train.env --dialect aggressive_alien --generator golden --policy gold --task-index 0
```

## Task generators

| Generator | Prompt | Gold rows from | Best for |
|---|---|---|---|
| `GoldenTaskGenerator` | "Translate this reference SQL: …" | reference dialect runtime on golden 5-table catalog | Teaching the model to *speak* a dialect (mechanical translation, no NL ambiguity). |
| `EvalSuiteTaskGenerator` | NL question from `eval.dataset.questions` | SQLite running the question's reference SQL on `github_events` | End-to-end NL→SQL benchmarking against a synthetic dialect. |
| `WikiSqlTaskGenerator` | NL question from `Salesforce/wikisql` + per-task table schema | reference dialect runtime on a `WikiSqlCatalog` of N sampled tables | NL→SQL on real Wikipedia tables with diverse schemas; large-scale variety. |

All subclass `TaskGenerator(ABC)`; rolling your own (e.g. for a Spider-style
multi-table benchmark or a synthesized augmentation pipeline) means
implementing `build()` + `all_tasks()`.

### WikiSQL specifics

`WikiSqlCatalog` pulls a reproducible random subset of N examples from
`Salesforce/wikisql` (HF `datasets`), materializes each one as a uniquely
named Polars table (`wikisql_<safe_id>`), and packs them all into a single
`CatalogSnapshot`. Column headers are sanitized to `c_<lowercase_alnum>`
form so they're valid identifiers in every dialect's grammar; the schema
prompt for the catalog is a brief placeholder because the actual per-task
schema (table name, columns, types, sample rows) is embedded in the
**user message**, not the system prompt — embedding 1000 schemas in the
system prompt would blow context for no reason.

Gold SQL is reconstructed from the WikiSQL structured triple
(`sel` / `agg` / `conds`) using the sanitized identifiers — WikiSQL's
`human_readable` field is too inconsistent to use directly. Gold rows
come from running that reconstructed SQL through the `_reference`
dialect on the same catalog; tasks where the reference engine fails or
returns no rows are dropped. Because the data + reference SQL are
dialect-independent, **`WikiSqlCatalog` can be shared across multiple
`DialectRuntime` instances** — `build()` is memoized so multi-dialect
curricula don't re-download / re-materialize WikiSQL N times.

```python
from train.env import (
    DialectRuntime, WikiSqlCatalog, WikiSqlTaskGenerator, WikiSqlTaskConfig,
)

cat = WikiSqlCatalog(n_samples=2000, split="train", seed=0)
gen = WikiSqlTaskGenerator(
    WikiSqlTaskConfig(target_dialect="aggressive_alien", catalog=cat)
)
gen.build()
tasks = gen.all_tasks()  # list[SqlTask]
```

## Multi-dialect curricula

The single-dialect path stays the simple one: pass a `DialectRuntime`
to the tool factory, the reward factory, and the dataset builder; the
trainer sees `run_sql(sql_command)` and the system prompt mentions one
dialect. Most experiments live here.

For **multi-dialect** training in one run, swap the runtime for a
`dict[str, DialectRuntime]`:

```python
from train.env import DialectRuntime, GoldenCatalog
from train.env.trl import (
    make_run_sql_tool, make_reward_funcs, trl_agent_system_prompt,
)

cat = GoldenCatalog()
cat.build()
runtimes = {
    d: DialectRuntime(dialect=d, catalog=cat).__enter__()
    for d in ["aggressive_alien", "tsql_ish", "mild_postgres_ish"]
}

# Tool now has signature run_sql(sql_command, dialect) and the
# docstring lists the supported dialects + tells the model to copy
# the prompt's "Dialect: X" tag into the dialect= argument.
run_sql = make_run_sql_tool(runtimes)

# Reward dispatches per row using the dataset's `dialect` column.
# The model's claimed dialect= argument is IGNORED for scoring; the
# reward always re-executes against ground truth.
reward_funcs = make_reward_funcs(runtimes=runtimes)

# Build per-dialect system prompts (each carries its own dialect card)
# and concatenate the resulting datasets — see train/grpo_sql.py.
```

Two coverage modes for combining tasks with multiple dialects, surfaced
in `train/grpo_sql.py` as `--coverage-mode`:

| Mode | Effect | Use when |
|---|---|---|
| `partition` (default) | Round-robin assign each task to one dialect (N rows total). | N is large, you want per-row variety. |
| `cross_product` | Emit each task once per dialect (N × M rows; `task_id` suffixed `__<dialect>`). | N is small, you want maximum dialect coverage per question. |

Each dialect gets its own system prompt — same dialect card / schema
body, but with a `Dialect: <name>` tag at the end and a rule telling
the model to copy that string into every `run_sql` call. The trainer
sees one merged dataset; reward functions look up the runtime per-row
via the `dialect` column.

## Reward shape

```
total = correctness                       # mode-dependent (see below)
      + turn_bonus                        # linear-mode only; 0 in discounted mode
      + sum(per-turn error_shaping)       # parse < runtime < unmatched < empty
      + format_penalty                    # if the agent never produced a parseable query
      + terminal_penalty                  # if truncated AND last turn was parse/empty
```

Two modes ship; the rest of the components are identical between them.

### Linear mode (default, `RewardConfig()`)

* `correctness` = 5.0 on match, 0 (or up to 40% of full on partial credit) on miss.
* `turn_bonus` = `2.0 * (max_turns - turns_used + 1) / max_turns` on match, else 0.
* Interpretable component-by-component; easy to tune individual weights.

### Discounted mode (`RewardConfig.discounted(discount_factor=0.9)`)

* `correctness` = `gamma**n` if matched, else 0. `n` = 0-based index of the
  matching turn, so first-turn correct = 1.0, second-turn = 0.9, third-turn = 0.81.
* `turn_bonus` = 0 (the discount IS the turn-efficiency signal).
* Binary correctness only — no partial credit.
* Bounded in `[0, 1]`. Doesn't need re-tuning when `max_turns` changes.
* Standard RL discounted-return semantics; recommended for GRPO.

Per-turn shaping, the format penalty, and the terminal-invalid penalty
apply identically in both modes, so a trainer can A/B them by swapping
the config without touching the rest of the pipeline. Sample reward
trajectories at default settings, max_turns=3:

| episode                              | linear total | discounted total |
|---|---|---|
| match on turn 1 (no errors)          | **+7.00**    | **+1.00** |
| match on turn 2 (1 runtime error)    | +5.83        | +0.40     |
| match on turn 3 (2 parse errors)     | +3.67        | -1.19     |
| no match, all parsed-but-wrong       | -0.75        | -0.75     |
| no match, all parse errors           | -6.00        | -6.00     |
| no match, parsed → parsed → parse-fail | -3.50      | -3.50     |

The breakdown is logged separately in `EpisodeResult.reward_components`
so each component shows up as its own metric in W&B/whatever.

## Plugging into TRL `GRPOTrainer` (Unsloth)

TRL's recent `GRPOTrainer` ships first-class agent / multi-turn support
via `tools=[...]`: the trainer drives the rollout loop, vLLM does the
generation, and the model interleaves `tool_calls` with `tool` results
until it stops. We don't run `SqlEnv` during training — the trainer
owns the rollout — but every other piece of `train.env` carries over
unchanged. The bridge module is `train/env/trl.py`; the end-to-end
training script is `train/grpo_sql.py` (see also `--dry-run` for a
CPU-only smoke test).

Three adapter pieces wire into the trainer:

```python
from train.env.trl import (
    make_run_sql_tool, make_reward_funcs,
    tasks_to_dataset, trl_agent_system_prompt,
)
from train.env import (
    DialectRuntime, GoldenTaskGenerator, GoldenTaskConfig, RewardConfig,
)

# 1. Build the runtime + task list (process-wide; ~100ms one-shot setup).
gen = GoldenTaskGenerator(GoldenTaskConfig(target_dialect="aggressive_alien"))
gen.build()
runtime = DialectRuntime(dialect="aggressive_alien", catalog=gen.catalog)
runtime.setup()

# 2. Dataset shaped for GRPOTrainer (chat-template prompt + reward kwargs).
ds = tasks_to_dataset(
    tasks=gen.all_tasks(),
    runtime=runtime,
    system_prompt=trl_agent_system_prompt(runtime),
    tokenizer=tokenizer,
    max_prompt_tokens=2048,
)

# 3. The tool the model calls (typed signature + Google-style docstring).
run_sql = make_run_sql_tool(runtime, preview_limit=50)

# 4. Per-component reward functions (each shows up as its own W&B panel).
reward_funcs = make_reward_funcs(
    runtimes=runtime,  # or a dict[str, DialectRuntime] for multi-dialect
    reward_config=RewardConfig.discounted(discount_factor=0.9),
    max_turns=5,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    tools=[run_sql],
    reward_funcs=reward_funcs,
    train_dataset=ds,
    args=grpo_config,
)
trainer.train()
```

What each piece does:

* **`make_run_sql_tool(runtime_or_dict)`** returns a closure-bound
  `run_sql` whose docstring TRL auto-parses into the JSON schema the
  model sees. Single-runtime input → `run_sql(sql_command: str)`;
  dict input → `run_sql(sql_command, dialect)` with dispatch on the
  `dialect` argument and a structured error payload for unknown
  dialect strings. Return payload is *model-facing* (capped row
  preview + error fields).
* **`reconstruct_turns(completion, runtime)`** walks a TRL completion
  (assistant + tool turns) and rebuilds our `Turn` shape by
  *re-executing each `sql_command`* against the runtime. Re-executing
  (instead of trusting the tool preview) buys deterministic scoring
  and decouples preview UX from scoring fidelity. Used internally by
  the reward functions; exposed for custom rewards.
* **`make_reward_funcs(runtimes, reward_config, components)`** returns
  one TRL-shaped reward function per `RewardBreakdown` field
  (`correctness`, `turn_bonus`, `error_shaping`, `format_penalty`,
  `terminal_penalty`). GRPO logs each independently, so the breakdown
  shows up component-by-component. Pass `components=["total"]` if you'd
  rather log a single scalar. `runtimes` accepts either a single
  `DialectRuntime` (single-dialect run) or a `dict[str,
  DialectRuntime]` (multi-dialect curriculum, dispatches per row using
  the dataset's `dialect` column).
* **`tasks_to_dataset(tasks, runtime, system_prompt, ...)`** returns a
  HF `Dataset` with chat-message `prompt` rows plus `task_id`,
  `dialect`, `gold_sql`, and `gold_rows_json` (JSON-encoded to dodge
  Arrow's struct-schema unification across heterogeneous gold queries).
* **`trl_agent_system_prompt(runtime, with_dialect_arg=False)`** is
  `runtime.system_prompt()` with a different rules block: tells the
  model to call `run_sql` rather than emit raw SQL in chat. **Use this
  when wiring TRL** — with the default raw-SQL prompt the model never
  invokes the tool and the reward function sees nothing. Pass
  `with_dialect_arg=True` for multi-dialect runs; this appends a rule
  telling the model to copy the prompt's `Dialect: <name>` tag into
  every `run_sql` call's `dialect=` argument.

Constraints:

* **Single-dialect mode** closes over one runtime and registers
  `run_sql(sql_command)`. **Multi-dialect mode** (pass a `dict` of
  runtimes) registers `run_sql(sql_command, dialect)` and dispatches
  per call. Reward functions always re-execute against the row's
  ground-truth dialect (the dataset's `dialect` column), so a model
  that emits the wrong `dialect=` argument and gets misleading tool
  feedback still receives the correct learning signal.
* **No hard turn budget at the trainer level.** TRL caps on
  `max_completion_length` (token budget). The reward function clips
  the transcript at `max_turns` post-hoc and stops at the first match
  (mirroring `SqlEnv` semantics); excess tool calls are ignored, not
  rewarded, not penalized.
* **`SqlEnv` / `LLMPolicy` / `run_episode` are NOT used at training
  time.** They stay useful for offline eval (e.g. against an OpenRouter
  endpoint), CLI smoke tests (`python -m train.env`), and replaying
  transcripts to debug reward shaping.

End-to-end smoke test:

```bash
# CPU only -- no Unsloth / torch / vLLM needed.

# Single dialect, golden corpus.
python -m train.grpo_sql --dialects aggressive_alien --dry-run

# Multi-dialect WikiSQL (downloads a small subset on first run).
python -m train.grpo_sql \
    --generator wikisql --wikisql-size 32 \
    --dialects aggressive_alien,tsql_ish \
    --coverage-mode partition --dry-run

# GPU box: see train/grpo_sql.py docstring for the full uv install incantation.
python train/grpo_sql.py --dialects aggressive_alien --max-steps 200
```

## Lifecycle + threading

* `DialectRuntime.setup()` is the expensive call (loads the dialect
  package, builds the Lark parser, materializes the catalog). Reuse one
  runtime across many tasks and episodes.
* `DialectRuntime` is **not thread-safe**. Use one per worker.
* `SqlEnv` is cheap — one per episode.
* `TaskGenerator.build()` precomputes gold rows up front. The hot path
  (`SqlEnv.step()`) never calls a reference engine.

## Layering rules (please respect)

* `train/env/` does **not** import from `eval/runner.py`,
  `eval/executors/synthetic_executor.py`, or any heavy CLI module.
  It only touches `eval/executors/base.py`,
  `eval/executors/sqlite_executor.py`, `eval/dataset/{questions,github_events}.py`,
  `eval/prompt.py`, `eval/llm.py`, and `eval/validator.py` —
  all stable, dependency-light leaves.
* Shared LLM-prompt logic (the dialect "card") lives in
  `manysql/dialects/card.py`. Both `eval/executors/synthetic_executor.py`
  and `train/env/engine.py` import from there.
* If you find yourself wanting a third copy of any of this glue,
  promote it to `manysql/` instead of cross-importing between `eval/`
  and `train/`.
