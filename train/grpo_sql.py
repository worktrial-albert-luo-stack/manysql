"""GRPO fine-tune of an LLM on one or more manysql synthetic SQL dialects.

The SQL counterpart to ``train/grpo_gsm8k.py``. Same Unsloth + TRL +
vLLM stack, same single-file shape, but the dataset, system prompt,
tool, and reward functions all come from :mod:`train.env`.

Task sources (``--generator``)
------------------------------

* ``golden`` (default) -- :class:`GoldenTaskGenerator`: cross-dialect
  translations of the manysql golden corpus on the canonical 5-table
  catalog. Best signal for teaching a model to *speak* a dialect.

* ``eval_suite`` -- :class:`EvalSuiteTaskGenerator`: NL->SQL benchmark
  questions over the synthetic ``github_events`` corpus.

* ``wikisql`` -- :class:`WikiSqlTaskGenerator`: pulls
  ``--wikisql-size`` examples from the ``Salesforce/wikisql`` HF
  dataset (each row carries its own small Wikipedia table). Column
  names are sanitized to ``c_<name>``; gold SQL is rebuilt from the
  WikiSQL structured triple to dodge the corpus's inconsistent
  human-readable strings; gold rows are computed by running that SQL
  through the reference dialect engine.

Multi-dialect curricula (``--dialects`` / ``--coverage-mode``)
--------------------------------------------------------------

Pass ``--dialects aggressive_alien,mild_postgres_ish,tsql_ish`` to
train one model on three dialects at once. Coverage modes:

* ``partition`` (default) -- round-robin assign each task to one
  dialect (N rows total). Best when N is large and you want per-row
  diversity rather than per-question dialect coverage.

* ``cross_product`` -- emit each task once per dialect (N samples * M
  dialects rows). Best when N is small and you want maximum dialect
  exposure per question.

Multi-dialect dispatch is implemented at the **tool** layer: the
``run_sql`` tool gains a ``dialect`` argument and the system prompt
tags every row with ``Dialect: <name>`` and tells the model to copy
that string verbatim into the call. Reward functions look up each
row's runtime via the dataset's ``dialect`` column and re-execute
against ground truth (so a model that misuses the dialect arg still
gets the correct reward signal). See ``train/env/trl.py`` for the
adapter mechanics.

Informative priors
------------------

Each dialect's system prompt includes the **dialect card** rendered by
``manysql.dialects.card.render_dialect_card``: surface divergences,
canonical patterns, function aliases, semantic divergences, and a
trimmed view of the codegen-emitted ``examples.sql``. This is exactly
what eval uses, by design -- the goal is to give the model the same
shape of context it would have facing a closed-source dialect (no full
grammar dump, just the divergences and a Rosetta stone of working
queries).

Usage
-----

GPU box (real training, single dialect, golden corpus)::

    uv venv .venv-train --python 3.11
    source .venv-train/bin/activate
    uv pip install --upgrade pip
    uv pip install "unsloth" "vllm" \\
        "transformers==4.56.2" "datasets>=2.20" "torch>=2.4" \\
        "accelerate" "bitsandbytes" "wandb" "python-dotenv"
    uv pip install --no-deps "trl==0.22.2"

    python train/grpo_sql.py --dialects aggressive_alien --max-steps 200

GPU box (multi-dialect WikiSQL, partition mode)::

    python train/grpo_sql.py \\
        --generator wikisql --wikisql-size 2000 \\
        --dialects aggressive_alien,mild_postgres_ish,tsql_ish \\
        --coverage-mode partition --max-steps 500

CPU box (dry-run, no GPU)::

    uv pip install "transformers" "datasets"
    python train/grpo_sql.py --dialects aggressive_alien --dry-run

Differences vs ``grpo_gsm8k.py``
--------------------------------

* No regex-based answer extraction; correctness comes from running the
  model's tool calls through the dialect engine and comparing rows to
  the precomputed gold rows.
* No ``SOLUTION_START``/``SOLUTION_END`` template; the model speaks
  through the ``run_sql`` tool, not freeform text.
* ``num_generations`` rollouts of the same prompt share the same
  dialect runtime (or the same dispatch dict in multi-dialect mode).
  Group-relative advantages are computed within a single environment.

W&B / trackio logging
---------------------

If ``WANDB_API_KEY`` is found (in env or ``.env``) the run logs to W&B
under project ``manysql-grpo`` with run name
``<model>-<dialects>-<steps>``. Override with ``--wandb-project`` /
``--wandb-run-name`` or disable with ``--no-wandb``. Each reward-
function component (correctness, turn_bonus, ...) gets its own panel
automatically.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

# UNSLOTH_VLLM_STANDBY=1 unlocks ~30% more context length when vLLM and
# training share GPU memory. Must be set before importing unsloth.
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

# Pull WANDB_API_KEY (and OPENAI_*, etc) from .env if present.
try:
    from dotenv import load_dotenv  # python-dotenv, already in pyproject.toml

    load_dotenv(override=False)
except ImportError:
    pass

if TYPE_CHECKING:
    from datasets import Dataset

    from train.env.engine import DialectRuntime
    from train.env.tasks import SqlTask


# Heavy ML deps (unsloth / torch / trl / vllm) are imported inside
# main() so this module imports on a CPU-only box for dry-run + tests.


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
@dataclass
class TrainArgs:
    # ---- model / unsloth ----
    model_name: str = "unsloth/Qwen3-4B-Instruct-2507"
    max_seq_length: int = 4096  # higher than GSM8K: SQL prompts are larger
    lora_rank: int = 32
    load_in_4bit: bool = True
    gpu_memory_utilization: float = 0.85
    # ---- env / task generator ----
    dialects: list[str] = field(default_factory=lambda: ["aggressive_alien"])
    coverage_mode: str = "partition"  # 'partition' | 'cross_product'
    generator: str = "golden"  # 'golden' | 'eval_suite' | 'wikisql'
    eval_suite_limit: int | None = None
    eval_suite_names: list[str] | None = None
    wikisql_size: int = 1000
    wikisql_split: str = "train"
    wikisql_seed: int = 0
    wikisql_sample_rows: int = 3
    # ---- rewards ----
    reward_mode: str = "discounted"  # 'linear' | 'discounted'
    discount_factor: float = 0.9
    max_turns: int = 5
    # ---- trainer ----
    learning_rate: float = 5e-6
    num_generations: int = 4
    max_steps: int = 200
    save_steps: int = 100
    output_dir: str = "outputs/grpo_qwen3_4b_sql"
    seed: int = 3407
    dry_run: bool = False
    # ---- logging ----
    wandb_project: str = "manysql-grpo"
    wandb_run_name: str | None = None
    no_wandb: bool = False


def parse_args() -> TrainArgs:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    defaults = TrainArgs()
    p.add_argument("--model-name", default=defaults.model_name)
    p.add_argument("--max-seq-length", type=int, default=defaults.max_seq_length)
    p.add_argument("--lora-rank", type=int, default=defaults.lora_rank)
    p.add_argument(
        "--load-in-4bit",
        type=lambda s: s.lower() != "false",
        default=defaults.load_in_4bit,
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=defaults.gpu_memory_utilization,
    )
    p.add_argument(
        "--dialects",
        default=",".join(defaults.dialects),
        help=(
            "Comma-separated list of manysql dialect ids "
            "(e.g. 'aggressive_alien,mild_postgres_ish'). One name = "
            "single-dialect run; >1 name = multi-dialect curriculum, "
            "see --coverage-mode."
        ),
    )
    p.add_argument(
        "--coverage-mode",
        default=defaults.coverage_mode,
        choices=["partition", "cross_product"],
        help=(
            "How to combine tasks with multiple dialects. "
            "partition (default) = round-robin assign one dialect per "
            "task (N rows); cross_product = each task once per dialect "
            "(N*M rows)."
        ),
    )
    p.add_argument(
        "--generator",
        default=defaults.generator,
        choices=["golden", "eval_suite", "wikisql"],
        help=(
            "Task source. golden = cross-dialect translation on the 5-table "
            "manysql catalog; eval_suite = NL->SQL benchmark questions on "
            "github_events; wikisql = NL->SQL on Wikipedia tables from "
            "Salesforce/wikisql (use --wikisql-size to pick a subset)."
        ),
    )
    p.add_argument(
        "--eval-suite-limit",
        type=int,
        default=defaults.eval_suite_limit,
        help="Limit eval-suite questions (only used with --generator=eval_suite).",
    )
    p.add_argument(
        "--wikisql-size",
        type=int,
        default=defaults.wikisql_size,
        help="Number of WikiSQL examples to sample (only used with --generator=wikisql).",
    )
    p.add_argument(
        "--wikisql-split",
        default=defaults.wikisql_split,
        choices=["train", "validation", "test"],
        help="WikiSQL HF split to draw from.",
    )
    p.add_argument(
        "--wikisql-seed",
        type=int,
        default=defaults.wikisql_seed,
        help="Seed for the WikiSQL random subset selection.",
    )
    p.add_argument(
        "--wikisql-sample-rows",
        type=int,
        default=defaults.wikisql_sample_rows,
        help="How many sample rows to embed in each WikiSQL user prompt.",
    )
    p.add_argument(
        "--reward-mode",
        default=defaults.reward_mode,
        choices=["linear", "discounted"],
    )
    p.add_argument(
        "--discount-factor",
        type=float,
        default=defaults.discount_factor,
        help="Gamma for discounted reward mode (ignored otherwise).",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=defaults.max_turns,
        help="Tool-call cap per completion considered for scoring.",
    )
    p.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    p.add_argument("--num-generations", type=int, default=defaults.num_generations)
    p.add_argument("--max-steps", type=int, default=defaults.max_steps)
    p.add_argument("--save-steps", type=int, default=defaults.save_steps)
    p.add_argument("--output-dir", default=defaults.output_dir)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip GPU model loading; only exercise dataset + reward funcs (CPU).",
    )
    p.add_argument("--wandb-project", default=defaults.wandb_project)
    p.add_argument("--wandb-run-name", default=defaults.wandb_run_name)
    p.add_argument("--no-wandb", action="store_true")
    ns = p.parse_args()
    parsed = vars(ns)
    parsed["dialects"] = _split_csv(parsed["dialects"])
    if not parsed["dialects"]:
        p.error("--dialects must contain at least one dialect id")
    return TrainArgs(**parsed)


def _split_csv(s: str | list[str]) -> list[str]:
    if isinstance(s, list):
        return [item.strip() for item in s if item and item.strip()]
    return [item.strip() for item in s.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Env / dataset construction
# ---------------------------------------------------------------------------
def build_runtimes_and_tasks(
    args: TrainArgs,
) -> tuple[dict[str, DialectRuntime], list[SqlTask]]:
    """Build the dialect runtimes and the merged task list.

    Returns a ``(runtimes, tasks)`` pair:

    * ``runtimes`` maps each dialect id from ``args.dialects`` to a
      *set-up* :class:`DialectRuntime`. All runtimes share the same
      catalog instance (catalogs are dialect-independent) so memory +
      load time scale with O(catalog) not O(catalog * dialects).

    * ``tasks`` is the flat task list ready to feed into the dataset
      builder. In single-dialect mode each task already targets the one
      dialect; in multi-dialect mode tasks are produced once with a
      "base" dialect and cloned-with-relabeling per the coverage mode:

        - ``cross_product``: each task is duplicated for every dialect
          (N tasks * M dialects = N*M rows), with ``task_id`` suffixed
          ``__<dialect>`` for uniqueness.
        - ``partition``: tasks are round-robin assigned, one dialect
          per task (N rows total).

    Gold rows are dialect-independent (data is the same across dialect
    runtimes) so we compute them once via the task generator and reuse
    the cached ``SqlTask.gold_rows`` across clones.
    """
    from train.env.engine import DialectRuntime  # noqa: PLC0415
    from train.env.tasks import (  # noqa: PLC0415
        EvalSuiteTaskConfig,
        EvalSuiteTaskGenerator,
        GoldenTaskConfig,
        GoldenTaskGenerator,
    )

    base_dialect = args.dialects[0]

    if args.generator == "golden":
        gen = GoldenTaskGenerator(GoldenTaskConfig(target_dialect=base_dialect))
    elif args.generator == "eval_suite":
        gen = EvalSuiteTaskGenerator(
            EvalSuiteTaskConfig(
                target_dialect=base_dialect,
                names=args.eval_suite_names,
                limit=args.eval_suite_limit,
            )
        )
    elif args.generator == "wikisql":
        from train.env.wikisql import (  # noqa: PLC0415
            WikiSqlTaskConfig,
            WikiSqlTaskGenerator,
        )

        gen = WikiSqlTaskGenerator(
            WikiSqlTaskConfig(
                target_dialect=base_dialect,
                n_samples=args.wikisql_size,
                split=args.wikisql_split,
                seed=args.wikisql_seed,
                sample_rows=args.wikisql_sample_rows,
            )
        )
    else:  # pragma: no cover - argparse choices guard this
        raise ValueError(f"unknown generator {args.generator!r}")

    gen.build()
    base_tasks = gen.all_tasks()

    runtimes: dict[str, DialectRuntime] = {}
    for d in args.dialects:
        rt = DialectRuntime(dialect=d, catalog=gen.catalog)
        rt.setup()
        runtimes[d] = rt

    if len(args.dialects) == 1:
        # Tasks already target the one dialect via the generator's
        # base_dialect. Nothing to expand.
        return runtimes, base_tasks

    if args.coverage_mode == "cross_product":
        tasks = _expand_cross_product(base_tasks, args.dialects)
    elif args.coverage_mode == "partition":
        tasks = _expand_partition(base_tasks, args.dialects)
    else:  # pragma: no cover - argparse choices guard this
        raise ValueError(f"unknown coverage_mode {args.coverage_mode!r}")
    return runtimes, tasks


def _expand_cross_product(
    base_tasks: list[SqlTask], dialects: list[str]
) -> list[SqlTask]:
    """Each base task -> one task per dialect (suffix ``__<dialect>`` on id)."""
    out: list[SqlTask] = []
    for task in base_tasks:
        for d in dialects:
            out.append(_relabel_task(task, dialect=d, suffix_id=True))
    return out


def _expand_partition(
    base_tasks: list[SqlTask], dialects: list[str]
) -> list[SqlTask]:
    """Round-robin: task i -> dialects[i % len(dialects)]."""
    out: list[SqlTask] = []
    for i, task in enumerate(base_tasks):
        d = dialects[i % len(dialects)]
        out.append(_relabel_task(task, dialect=d, suffix_id=False))
    return out


def _relabel_task(task: SqlTask, *, dialect: str, suffix_id: bool) -> SqlTask:
    """Clone a task with a different target dialect.

    Gold rows / gold SQL / prompt / catalog are unchanged (data is
    dialect-independent). Only ``meta.dialect`` and optionally
    ``meta.task_id`` change. We use ``dataclasses.replace`` so any
    future fields on :class:`SqlTask` keep round-tripping.
    """
    new_id = f"{task.meta.task_id}__{dialect}" if suffix_id else task.meta.task_id
    new_meta = replace(task.meta, dialect=dialect, task_id=new_id)
    return replace(task, meta=new_meta)


def build_dataset(
    args: TrainArgs,
    runtimes: dict[str, DialectRuntime],
    tasks: list[SqlTask],
    tokenizer: Any | None = None,
) -> Dataset:
    """Translate the task list into a HF Dataset for GRPOTrainer.

    For multi-dialect runs, builds one sub-dataset per dialect with
    that dialect's tool-aware system prompt (different dialect cards =
    different system prompts) and concatenates them. The trainer sees
    one dataset; reward functions dispatch per-row using the ``dialect``
    column.

    Each row carries ``{"prompt", "task_id", "dialect", "generator",
    "gold_sql", "gold_rows_json"}``. The ``prompt`` column is in
    chat-message form so ``GRPOTrainer`` auto-applies the tokenizer's
    chat template; the rest are forwarded to reward functions.
    """
    from datasets import concatenate_datasets  # noqa: PLC0415

    from train.env.trl import (  # noqa: PLC0415
        tasks_to_dataset,
        trl_agent_system_prompt,
    )

    cap_tokens: int | None = None
    if tokenizer is not None:
        cap_tokens = args.max_seq_length // 2  # mirrors grpo_gsm8k.py

    multi = len(runtimes) > 1
    by_dialect: dict[str, list[SqlTask]] = defaultdict(list)
    for t in tasks:
        by_dialect[t.meta.dialect].append(t)

    parts = []
    for dialect, dt in by_dialect.items():
        if dialect not in runtimes:
            raise RuntimeError(
                f"task targets dialect {dialect!r} but no matching runtime "
                f"was built (have {sorted(runtimes)})"
            )
        rt = runtimes[dialect]
        sys_prompt = trl_agent_system_prompt(rt, with_dialect_arg=multi)
        parts.append(
            tasks_to_dataset(
                tasks=dt,
                runtime=rt,
                system_prompt=sys_prompt,
                tokenizer=tokenizer,
                max_prompt_tokens=cap_tokens,
            )
        )
    if not parts:
        raise RuntimeError("build_dataset: no tasks to convert")
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------
def make_reward_funcs(args: TrainArgs, runtimes: dict[str, DialectRuntime]) -> list[Any]:
    """Build the per-component reward functions for GRPOTrainer.

    Reward functions look up each row's runtime via the dataset's
    ``dialect`` column and re-execute the model's SQL against ground
    truth. In single-dialect mode the dispatch is a no-op (one entry).
    """
    from train.env.rewards import RewardConfig  # noqa: PLC0415
    from train.env.trl import make_reward_funcs as _make  # noqa: PLC0415

    if args.reward_mode == "discounted":
        cfg = RewardConfig.discounted(discount_factor=args.discount_factor)
    else:
        cfg = RewardConfig()
    return _make(
        runtimes=runtimes,
        reward_config=cfg,
        max_turns=args.max_turns,
    )


# ---------------------------------------------------------------------------
# Dry-run (CPU-only smoke test)
# ---------------------------------------------------------------------------
def _dry_run(args: TrainArgs) -> None:
    """Exercise dataset + reward pipeline without unsloth / torch / vllm.

    Builds the runtime(s), generates the task list, formats the dataset,
    and scores a handful of synthetic completions through every reward
    function so you can eyeball the rubric before burning GPU time.

    For multi-dialect runs the dry-run scores the same fake completions
    once per distinct dialect appearing in the dataset, so you can see
    that the dispatch is hooked up correctly (different reward
    breakdowns per dialect because each parses + executes the same SQL
    differently).
    """
    import json  # noqa: PLC0415

    print(
        f"[dry-run] building runtimes for dialects={args.dialects}, "
        f"generator={args.generator}, coverage={args.coverage_mode}"
    )
    runtimes, tasks = build_runtimes_and_tasks(args)
    print(f"[dry-run] generated {len(tasks)} tasks")

    ds = build_dataset(args, runtimes, tasks, tokenizer=None)
    print(f"[dry-run] built dataset with {len(ds)} rows")
    sample = ds[0]
    print("[dry-run] sample row:")
    for msg in sample["prompt"]:
        head = msg["content"][:200].replace("\n", " ")
        suffix = "..." if len(msg["content"]) > 200 else ""
        print(f"  {msg['role']}: {head}{suffix}")
    print(f"  task_id: {sample['task_id']}")
    print(f"  dialect: {sample['dialect']}")
    print(f"  gold_sql: {sample['gold_sql']}")

    gold_sql = sample["gold_sql"]
    fake_completions = _build_fake_completions(gold_sql)
    fake_prompts = [sample["prompt"]] * len(fake_completions)
    gold_rows_json = [sample["gold_rows_json"]] * len(fake_completions)
    fake_dialects = [sample["dialect"]] * len(fake_completions)

    funcs = make_reward_funcs(args, runtimes)

    print("[dry-run] reward breakdown:")
    header = ["#", *(fn.__name__ for fn in funcs), "sum"]
    print("  " + "  ".join(f"{h:>26}" for h in header))
    all_scores = [
        fn(
            prompts=fake_prompts,
            completions=fake_completions,
            gold_rows_json=gold_rows_json,
            dialect=fake_dialects,
        )
        for fn in funcs
    ]
    for i in range(len(fake_completions)):
        row: list[str] = [str(i)]
        total = 0.0
        for col_scores in all_scores:
            row.append(f"{col_scores[i]:+.3f}")
            total += col_scores[i]
        row.append(f"{total:+.3f}")
        print("  " + "  ".join(f"{c:>26}" for c in row))

    if args.reward_mode == "discounted" and all_scores:
        correctness_scores = next(
            scores for scores, fn in zip(all_scores, funcs, strict=False)
            if fn.__name__ == "sql_correctness_reward"
        )
        assert abs(correctness_scores[0] - 1.0) < 1e-6, (
            f"expected match-on-turn-1 correctness=1.0, got {correctness_scores[0]}"
        )

    # Multi-dialect spot-check: re-score the gold-SQL completion against
    # every dialect in the run and confirm dispatch routes to each one.
    if len(runtimes) > 1:
        print("[dry-run] multi-dialect dispatch check:")
        gold_completion = fake_completions[0]
        correctness_fn = next(
            fn for fn in funcs if fn.__name__ == "sql_correctness_reward"
        )
        for d in runtimes:
            score = correctness_fn(
                prompts=[sample["prompt"]],
                completions=[gold_completion],
                gold_rows_json=[sample["gold_rows_json"]],
                dialect=[d],
            )
            print(f"  dialect={d}: correctness={score[0]:+.3f}")

    json.loads(sample["gold_rows_json"])
    print("\n[dry-run] OK -- dataset + tool + rewards look healthy. "
          "Run without --dry-run on a GPU box to actually train.")


def _build_fake_completions(gold_sql: str) -> list[list[dict[str, Any]]]:
    """Synthetic completions that exercise the reward branches.

    Returns a list of TRL-shaped completions:

    1. One-shot correct: tool calls gold SQL.
    2. Two-step recovery: bad SQL, then gold SQL.
    3. All parse errors: never produces valid SQL.
    4. No tool calls at all: free-text response.

    Same pattern as the GSM8K dry-run; here the SQL is real.
    """
    return [
        [_assistant_call(gold_sql)],
        [
            _assistant_call("SELECT this_is_garbage"),
            _tool_response("{}"),
            _assistant_call(gold_sql),
        ],
        [
            _assistant_call("this is not sql"),
            _tool_response("{}"),
            _assistant_call("still not sql"),
        ],
        [{"role": "assistant", "content": "I don't know."}],
    ]


def _assistant_call(sql: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "run_sql",
                    "arguments": {"sql_command": sql},
                }
            }
        ],
    }


def _tool_response(content: str) -> dict[str, Any]:
    return {"role": "tool", "content": content}


# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------
def _configure_wandb(args: TrainArgs) -> str:
    if args.no_wandb:
        print("[wandb] disabled via --no-wandb")
        return "none"
    if not os.environ.get("WANDB_API_KEY"):
        print("[wandb] WANDB_API_KEY not set; logging disabled")
        return "none"

    os.environ["WANDB_PROJECT"] = args.wandb_project
    dialects_tag = (
        "-".join(args.dialects)
        if len(args.dialects) <= 3
        else f"{len(args.dialects)}dialects"
    )
    run_name = args.wandb_run_name or (
        f"{args.model_name.split('/')[-1]}-{dialects_tag}-{args.max_steps}steps"
    )
    os.environ["WANDB_NAME"] = run_name
    os.environ.setdefault("WANDB_WATCH", "false")
    print(f"[wandb] enabled: project={args.wandb_project} run={run_name}")
    return "wandb"


# ---------------------------------------------------------------------------
# Main (real training, GPU required)
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    if args.dry_run:
        _dry_run(args)
        return

    report_to = _configure_wandb(args)

    import torch  # noqa: PLC0415
    from trl import GRPOConfig, GRPOTrainer  # noqa: PLC0415
    from unsloth import FastLanguageModel  # noqa: PLC0415  GPU-only import
    from vllm import SamplingParams  # noqa: PLC0415

    print(
        f"[grpo] loading {args.model_name} "
        f"(4bit={args.load_in_4bit}, lora_rank={args.lora_rank})"
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=True,
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    print(
        f"[grpo] building runtimes for dialects={args.dialects}, "
        f"generator={args.generator}, coverage={args.coverage_mode}"
    )
    runtimes, tasks = build_runtimes_and_tasks(args)
    print(f"[grpo] generated {len(tasks)} tasks across {len(runtimes)} dialect(s)")

    print("[grpo] building dataset")
    train_ds = build_dataset(args, runtimes, tasks, tokenizer=tokenizer)
    print(f"[grpo] kept {len(train_ds)} examples after length filter")

    prompt_lens = [
        len(
            tokenizer.apply_chat_template(
                r["prompt"], add_generation_prompt=True, tokenize=True
            )
        )
        for r in train_ds
    ]
    max_prompt_length = max(prompt_lens) + 1
    max_completion_length = args.max_seq_length - max_prompt_length
    if max_completion_length <= 256:
        raise ValueError(
            f"max_completion_length={max_completion_length} too small for SQL "
            f"multi-turn rollouts; raise --max-seq-length "
            f"(currently {args.max_seq_length})"
        )

    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=args.seed,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )

    grpo_config = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=1.0,
        learning_rate=args.learning_rate,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        report_to=report_to,
        run_name=os.environ.get("WANDB_NAME") if report_to == "wandb" else None,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    from train.env.trl import make_run_sql_tool  # noqa: PLC0415

    # In single-dialect mode we hand the tool factory the lone runtime
    # directly (back-compat run_sql signature). In multi-dialect mode
    # we hand it the dict so the model emits run_sql(sql_command, dialect=...)
    # and dispatch happens at call time.
    tool_target = (
        next(iter(runtimes.values())) if len(runtimes) == 1 else runtimes
    )
    run_sql_tool = make_run_sql_tool(tool_target)
    reward_funcs = make_reward_funcs(args, runtimes)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        tools=[run_sql_tool],
        args=grpo_config,
        train_dataset=train_ds,
    )

    print(f"[grpo] starting training for {args.max_steps} steps -> {args.output_dir}")
    trainer.train()

    lora_dir = os.path.join(args.output_dir, "lora")
    print(f"[grpo] saving LoRA adapter to {lora_dir}")
    model.save_lora(lora_dir)

    # Quick smoke-test generation on the first task with the trained adapter.
    sample = train_ds[0]
    text = tokenizer.apply_chat_template(
        sample["prompt"], add_generation_prompt=True, tokenize=False
    )
    sampling = SamplingParams(temperature=0.7, top_k=50, max_tokens=512)
    out = (
        model.fast_generate(
            text, sampling_params=sampling, lora_request=model.load_lora(lora_dir)
        )[0]
        .outputs[0]
        .text
    )
    print("[grpo] smoke-test response:\n" + out)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
