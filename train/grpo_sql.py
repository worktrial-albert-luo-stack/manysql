"""GRPO fine-tune of an LLM on a manysql synthetic SQL dialect.

The SQL counterpart to ``train/grpo_gsm8k.py``. Same Unsloth + TRL +
vLLM stack, same single-file shape, but the dataset, system prompt,
tool, and reward functions all come from :mod:`train.env`. Specifically:

* **Dataset**: ``GoldenTaskGenerator`` (default) or
  ``EvalSuiteTaskGenerator`` produce :class:`SqlTask`s, which
  :func:`train.env.trl.tasks_to_dataset` formats into the chat-message
  shape ``GRPOTrainer`` consumes (one row = one task; ``num_generations``
  rollouts per row are scored group-relative).

* **Tool**: :func:`train.env.trl.make_run_sql_tool` exposes a typed
  ``run_sql(sql_command)`` to the model. TRL handles the multi-turn
  loop (assistant tool_call -> tool result -> next assistant turn)
  natively; we do NOT call ``SqlEnv`` during training. The env stays
  useful for offline eval, smoke tests, and transcript replay.

* **System prompt**: ``trl_agent_system_prompt(runtime)`` -- same
  dialect card + schema, but the rules tell the model to use ``run_sql``
  rather than emit raw SQL in chat.

* **Rewards**: :func:`train.env.trl.make_reward_funcs` returns one
  TRL ``reward_func`` per :class:`RewardBreakdown` field
  (``correctness``, ``turn_bonus``, ``error_shaping``,
  ``format_penalty``, ``terminal_penalty``). Each shows up as its own
  panel in W&B / trackio so you can watch the breakdown in flight.

Usage
-----

GPU box (real training)::

    uv venv .venv-train --python 3.11
    source .venv-train/bin/activate
    uv pip install --upgrade pip
    uv pip install "unsloth" "vllm" \\
        "transformers==4.56.2" "datasets>=2.20" "torch>=2.4" "accelerate" "bitsandbytes" \\
        "wandb" "python-dotenv"
    uv pip install --no-deps "trl==0.22.2"

    python train/grpo_sql.py --dialect aggressive_alien --max-steps 200

CPU box (dry-run, no GPU required)::

    uv pip install "transformers" "datasets"
    python train/grpo_sql.py --dialect aggressive_alien --dry-run

Differences vs ``grpo_gsm8k.py``
--------------------------------

* No regex-based answer extraction; correctness comes from running the
  model's tool calls through the dialect engine and comparing rows to
  the precomputed gold rows.
* No ``SOLUTION_START``/``SOLUTION_END`` template; the model speaks
  through the ``run_sql`` tool, not freeform text. The system prompt
  reflects that.
* ``num_generations`` rollouts of the same prompt all share the same
  dialect runtime (we hold one process-wide). Group-relative advantages
  are computed within a single environment, as it should be.
* One dialect per training run for v0; multi-dialect curricula need a
  thread-local runtime selector that we haven't wired yet (see
  ``train/env/trl.py`` constraints docstring).

W&B / trackio logging
---------------------

If ``WANDB_API_KEY`` is found (in env or ``.env``) the run logs to W&B
under project ``manysql-grpo`` with run name ``<model>-<dialect>-<steps>``.
Override with ``--wandb-project`` / ``--wandb-run-name`` or disable with
``--no-wandb``. Each reward-function component (correctness, turn_bonus,
...) gets its own panel automatically.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
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
    dialect: str = "aggressive_alien"
    generator: str = "golden"  # 'golden' | 'eval_suite'
    eval_suite_limit: int | None = None
    eval_suite_names: list[str] | None = None
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
        "--dialect",
        default=defaults.dialect,
        help="manysql dialect id (must be loadable by DialectRegistry).",
    )
    p.add_argument(
        "--generator",
        default=defaults.generator,
        choices=["golden", "eval_suite"],
        help="Task source. golden = cross-dialect translation; "
        "eval_suite = NL->SQL benchmark questions.",
    )
    p.add_argument(
        "--eval-suite-limit",
        type=int,
        default=defaults.eval_suite_limit,
        help="Limit eval-suite questions (only used with --generator=eval_suite).",
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
    return TrainArgs(**vars(ns))


# ---------------------------------------------------------------------------
# Env / dataset construction
# ---------------------------------------------------------------------------
def build_runtime_and_tasks(
    args: TrainArgs,
) -> tuple[DialectRuntime, list[SqlTask]]:
    """Build the shared ``DialectRuntime`` and pre-materialize the task list.

    The runtime is expensive (loads dialect package, builds Lark parser,
    materializes Polars catalog) and cheap to call. We keep one
    process-wide and let TRL's tool dispatch + our reward functions both
    close over it.

    Catalog choice depends on the task generator: ``golden`` uses the
    canonical 5-table catalog; ``eval_suite`` uses ``github_events``.
    """
    from train.env.engine import DialectRuntime  # noqa: PLC0415
    from train.env.tasks import (  # noqa: PLC0415
        EvalSuiteTaskConfig,
        EvalSuiteTaskGenerator,
        GoldenTaskConfig,
        GoldenTaskGenerator,
    )

    if args.generator == "golden":
        gen = GoldenTaskGenerator(GoldenTaskConfig(target_dialect=args.dialect))
    elif args.generator == "eval_suite":
        gen = EvalSuiteTaskGenerator(
            EvalSuiteTaskConfig(
                target_dialect=args.dialect,
                names=args.eval_suite_names,
                limit=args.eval_suite_limit,
            )
        )
    else:  # pragma: no cover - argparse choices guard this
        raise ValueError(f"unknown generator {args.generator!r}")

    gen.build()
    runtime = DialectRuntime(dialect=args.dialect, catalog=gen.catalog)
    runtime.setup()
    return runtime, gen.all_tasks()


def build_dataset(
    args: TrainArgs,
    runtime: DialectRuntime,
    tasks: list[SqlTask],
    tokenizer: Any | None = None,
) -> Dataset:
    """Translate the task list into a HF Dataset for GRPOTrainer.

    Each row becomes ``{"prompt", "task_id", "dialect", "generator",
    "gold_sql", "gold_rows_json"}``. ``GRPOTrainer`` auto-applies the
    tokenizer's chat template to the ``prompt`` column; everything else
    is forwarded to reward functions as kwargs (zipped per-row).
    """
    from train.env.trl import (  # noqa: PLC0415
        tasks_to_dataset,
        trl_agent_system_prompt,
    )

    cap_tokens: int | None = None
    if tokenizer is not None:
        cap_tokens = args.max_seq_length // 2  # mirrors grpo_gsm8k.py
    return tasks_to_dataset(
        tasks=tasks,
        runtime=runtime,
        # Tool-aware system prompt: tells the model to call run_sql
        # rather than emit raw SQL. Without this the rollout never goes
        # through the tool and the reward function sees nothing.
        system_prompt=trl_agent_system_prompt(runtime),
        tokenizer=tokenizer,
        max_prompt_tokens=cap_tokens,
    )


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------
def make_reward_funcs(args: TrainArgs, runtime: DialectRuntime) -> list[Any]:
    """Build the per-component reward functions for GRPOTrainer."""
    from train.env.rewards import RewardConfig  # noqa: PLC0415
    from train.env.trl import make_reward_funcs as _make  # noqa: PLC0415

    if args.reward_mode == "discounted":
        cfg = RewardConfig.discounted(discount_factor=args.discount_factor)
    else:
        cfg = RewardConfig()
    return _make(
        runtime=runtime,
        reward_config=cfg,
        max_turns=args.max_turns,
    )


# ---------------------------------------------------------------------------
# Dry-run (CPU-only smoke test)
# ---------------------------------------------------------------------------
def _dry_run(args: TrainArgs) -> None:
    """Exercise dataset + reward pipeline without unsloth / torch / vllm.

    Builds the runtime, generates the task list, formats the dataset,
    and scores a handful of synthetic completions through every reward
    function so you can eyeball the rubric before burning GPU time.
    """
    import json  # noqa: PLC0415

    print(
        f"[dry-run] building runtime for dialect={args.dialect}, "
        f"generator={args.generator}"
    )
    runtime, tasks = build_runtime_and_tasks(args)
    print(f"[dry-run] generated {len(tasks)} tasks")

    ds = build_dataset(args, runtime, tasks, tokenizer=None)
    print(f"[dry-run] built dataset with {len(ds)} rows")
    sample = ds[0]
    print("[dry-run] sample row:")
    for msg in sample["prompt"]:
        head = msg["content"][:200].replace("\n", " ")
        suffix = "..." if len(msg["content"]) > 200 else ""
        print(f"  {msg['role']}: {head}{suffix}")
    print(f"  task_id: {sample['task_id']}")
    print(f"  gold_sql: {sample['gold_sql']}")

    # Build synthetic completions to exercise the reward branches.
    gold_sql = sample["gold_sql"]
    fake_completions = [
        # 1. One-shot correct: tool calls gold SQL.
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_sql",
                            "arguments": {"sql_command": gold_sql},
                        }
                    }
                ],
            },
        ],
        # 2. Two-step recovery: bad SQL, then gold SQL.
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_sql",
                            "arguments": {"sql_command": "SELECT this_is_garbage"},
                        }
                    }
                ],
            },
            {"role": "tool", "content": "{}"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_sql",
                            "arguments": {"sql_command": gold_sql},
                        }
                    }
                ],
            },
        ],
        # 3. All parse errors: never produces valid SQL.
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_sql",
                            "arguments": {"sql_command": "this is not sql"},
                        }
                    }
                ],
            },
            {"role": "tool", "content": "{}"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_sql",
                            "arguments": {"sql_command": "still not sql"},
                        }
                    }
                ],
            },
        ],
        # 4. No tool calls at all: free-text response.
        [{"role": "assistant", "content": "I don't know."}],
    ]
    fake_prompts = [sample["prompt"]] * len(fake_completions)
    gold_rows_json = [sample["gold_rows_json"]] * len(fake_completions)

    funcs = make_reward_funcs(args, runtime)

    print("[dry-run] reward breakdown:")
    header = ["#", *(fn.__name__ for fn in funcs), "sum"]
    print("  " + "  ".join(f"{h:>26}" for h in header))
    all_scores = [
        fn(
            prompts=fake_prompts,
            completions=fake_completions,
            gold_rows_json=gold_rows_json,
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

    # Sanity: discount-mode "match on turn 1" should be 1.0.
    if args.reward_mode == "discounted" and all_scores:
        correctness_scores = next(
            scores for scores, fn in zip(all_scores, funcs, strict=False)
            if fn.__name__ == "sql_correctness_reward"
        )
        assert abs(correctness_scores[0] - 1.0) < 1e-6, (
            f"expected match-on-turn-1 correctness=1.0, got {correctness_scores[0]}"
        )

    # Round-trip the gold_rows_json to make sure it parses cleanly.
    json.loads(sample["gold_rows_json"])
    print("\n[dry-run] OK -- dataset + tool + rewards look healthy. "
          "Run without --dry-run on a GPU box to actually train.")


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
    run_name = args.wandb_run_name or (
        f"{args.model_name.split('/')[-1]}-{args.dialect}-{args.max_steps}steps"
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
        f"[grpo] building runtime for dialect={args.dialect}, "
        f"generator={args.generator}"
    )
    runtime, tasks = build_runtime_and_tasks(args)
    print(f"[grpo] generated {len(tasks)} tasks")

    print("[grpo] building dataset")
    train_ds = build_dataset(args, runtime, tasks, tokenizer=tokenizer)
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

    run_sql_tool = make_run_sql_tool(runtime)
    reward_funcs = make_reward_funcs(args, runtime)

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
