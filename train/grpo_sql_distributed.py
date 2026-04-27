"""GRPO fine-tune of an LLM on manysql SQL dialects -- DDP / multi-GPU.

The multi-GPU counterpart to :mod:`train.grpo_sql`. Same env / dataset /
reward layer (reuses ``build_runtimes_and_tasks``, ``build_dataset``,
``make_reward_funcs`` from :mod:`train.grpo_sql`), same ``<SQL>...</SQL>``
tag-mode SQL protocol, but drops Unsloth in favour of plain Hugging Face
``transformers`` + ``peft`` so the trainer composes with
``accelerate launch --num_processes N``.

Stack pin (matches ``train/GPU_SETUP.md``)
------------------------------------------

This script is built against the same pinned stack as the single-GPU
script -- there is no compatible newer combination today (see
``train/GPU_SETUP.md`` for the full breakdown):

- torch 2.10.0+cu128
- vllm 0.17.0  (caps transformers <5)
- transformers 4.57.6
- trl 0.22.2  (with the two source patches: ``GuidedDecodingParams``
  stub + ``prepare_peft_model`` ``dataclasses.replace`` fix)
- datasets >=2.20,<4
- peft, bitsandbytes (optional 4-bit base weights)

Why no Unsloth
--------------

Unsloth's vLLM engine is single-process by design -- its monkey-patch
into ``trl.trainer.grpo_trainer`` hijacks the rollout into a single
in-process inference engine. Under ``accelerate launch`` with
``num_processes > 1`` each rank tries to spin up its own and OOMs
against the others. Dropping Unsloth costs ~30% throughput from its
fused kernels but lets us scale across N GPUs.

vLLM rollouts under trl 0.22 (split mode)
-----------------------------------------

trl 0.22 doesn't have the colocate / server modes that landed in 0.26.
The supported pattern is **split mode**: ``use_vllm=True`` plus
``vllm_device`` pointing at one dedicated GPU. Rank 0 owns the vLLM
instance, generates the completions, and broadcasts them to the other
training ranks. With N GPUs the recommended layout is::

    accelerate launch --num_processes N-1 \\
        train/grpo_sql_distributed.py \\
        --vllm-device cuda:N-1

i.e. training on GPUs ``0..N-2``, vLLM on GPU ``N-1``. The
``run_grpo_sql_ddp.sh`` launcher autodetects ``CUDA_VISIBLE_DEVICES``
and picks this layout for you.

Tag mode (no ``tools=``)
------------------------

trl 0.26.2 added ``GRPOTrainer(tools=...)`` but requires
``transformers>=5``, which ``vllm 0.17.0`` forbids. Like
:mod:`train.grpo_sql`, we use the ``<SQL>...</SQL>`` tag protocol:
the model writes SQL between tags, reward functions extract +
re-execute it through the dialect engine. The shared
:func:`train.grpo_sql.build_dataset` already wires the tag-mode
system prompt (:func:`train.env.trl.trl_tag_system_prompt`), and the
shared :func:`train.grpo_sql.make_reward_funcs` produces reward
callables whose transcript reconstruction handles both tool-call and
tag formats (see :func:`train.env.trl.reconstruct_turns`).

Usage
-----

CPU dry-run (no torch / trl / vllm)::

    python -m train.grpo_sql_distributed \\
        --dialects aggressive_alien --dry-run

Pod launcher (recommended; sets HF caches + cd /tmp, autodetects GPUs)::

    bash train/run_grpo_sql_ddp.sh \\
        --dialects aggressive_alien --max-steps 200

Single node, 8 GPUs, 7 train + 1 vLLM, expanded form::

    accelerate launch --num_processes 7 \\
        train/grpo_sql_distributed.py \\
        --dialects aggressive_alien --max-steps 200 \\
        --vllm-device cuda:7

Single GPU sanity (still no Unsloth -- useful for smoke-testing the
plain-HF path before scaling up)::

    python train/grpo_sql_distributed.py \\
        --dialects aggressive_alien --max-steps 50 \\
        --vllm-device cuda:0 --vllm-gpu-memory-utilization 0.4

Notes / gotchas
---------------

* ``per_device_train_batch_size * num_processes * gradient_accumulation_steps``
  must be divisible by ``num_generations`` (trl 0.22 invariant -- see
  ``train/GPU_SETUP.md``). Defaults below keep that divisible for
  ``num_processes`` in ``{1, 2, 4}`` with ``num_generations=4``.
* Bf16 LoRA is the default. ``--load-in-4bit`` switches to bnb nf4 +
  paged_adamw_8bit; saves VRAM, ~10% slower.
* Both trl 0.22 patches in ``train/GPU_SETUP.md`` are required.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# Pull WANDB_API_KEY (and OPENAI_*, etc) from .env if present.
try:
    from dotenv import load_dotenv  # python-dotenv, already in pyproject.toml

    load_dotenv(override=False)
except ImportError:
    pass

# Reuse the env / dataset / reward / dry-run plumbing from the
# single-GPU script. The shared helpers are duck-typed against the
# overlapping fields on :class:`DistributedTrainArgs` and
# :class:`train.grpo_sql.TrainArgs`, so keeping the field names in sync
# is what makes the reuse safe.
from train.grpo_sql import (  # noqa: E402  (after dotenv on purpose)
    _configure_wandb,
    _dry_run,
    _split_csv,
    build_dataset,
    build_runtimes_and_tasks,
    make_reward_funcs,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
@dataclass
class DistributedTrainArgs:
    """Args for the DDP / no-Unsloth GRPO script.

    Field names that overlap with :class:`train.grpo_sql.TrainArgs` keep
    the same semantics so the shared helpers (``build_runtimes_and_tasks``,
    ``build_dataset``, ``make_reward_funcs``, ``_configure_wandb``,
    ``_dry_run``) duck-type cleanly across both scripts.
    """

    # ---- model ----
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_seq_length: int = 4096
    lora_rank: int = 32
    load_in_4bit: bool = False  # bf16 by default; opt-in to bnb nf4
    bf16: bool = True
    flash_attn: bool = False
    # ---- env / task generator ----
    dialects: list[str] = field(default_factory=lambda: ["aggressive_alien"])
    coverage_mode: str = "partition"
    generator: str = "golden"
    eval_suite_limit: int | None = None
    eval_suite_names: list[str] | None = None
    wikisql_size: int = 1000
    wikisql_split: str = "train"
    wikisql_seed: int = 0
    wikisql_sample_rows: int = 3
    # ---- rewards ----
    reward_mode: str = "discounted"
    discount_factor: float = 0.9
    max_turns: int = 5
    # ---- trainer ----
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 4  # >= num_generations (trl 0.22 invariant)
    gradient_accumulation_steps: int = 1
    num_generations: int = 4
    max_steps: int = 200
    save_steps: int = 100
    gradient_checkpointing: bool = True
    output_dir: str = "outputs/grpo_qwen3_4b_sql_ddp"
    seed: int = 3407
    dry_run: bool = False
    # ---- vLLM rollout (trl 0.22 split mode) ----
    vllm_device: str = "auto"  # 'auto' = pick last visible GPU
    vllm_gpu_memory_utilization: float = 0.85
    vllm_max_model_len: int | None = None  # default: trainer.max_seq_length
    vllm_dtype: str = "auto"
    # ---- logging ----
    wandb_project: str = "manysql-grpo"
    wandb_run_name: str | None = None
    no_wandb: bool = False


def parse_args() -> DistributedTrainArgs:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    defaults = DistributedTrainArgs()
    # ---- model ----
    p.add_argument("--model-name", default=defaults.model_name)
    p.add_argument("--max-seq-length", type=int, default=defaults.max_seq_length)
    p.add_argument("--lora-rank", type=int, default=defaults.lora_rank)
    p.add_argument(
        "--load-in-4bit",
        type=lambda s: s.lower() != "false",
        default=defaults.load_in_4bit,
        help="Load base weights in 4-bit (bitsandbytes nf4). Default: bf16.",
    )
    p.add_argument(
        "--bf16",
        type=lambda s: s.lower() != "false",
        default=defaults.bf16,
        help="Train in bf16 (recommended on Ampere+/H100).",
    )
    p.add_argument(
        "--flash-attn",
        action="store_true",
        help="Use flash-attention-2 for the base model (requires the package).",
    )
    # ---- env / task generator ----
    p.add_argument(
        "--dialects",
        default=",".join(defaults.dialects),
        help=(
            "Comma-separated list of manysql dialect ids "
            "(e.g. 'aggressive_alien,mild_postgres_ish'). One = single-"
            "dialect run; >1 = multi-dialect curriculum (see "
            "--coverage-mode)."
        ),
    )
    p.add_argument(
        "--coverage-mode",
        default=defaults.coverage_mode,
        choices=["partition", "cross_product"],
        help=(
            "How to combine tasks across dialects. partition (default) "
            "= round-robin one dialect per task (N rows); cross_product "
            "= each task once per dialect (N*M rows)."
        ),
    )
    p.add_argument(
        "--generator",
        default=defaults.generator,
        choices=["golden", "eval_suite", "wikisql"],
        help=(
            "Task source. golden = cross-dialect translation on the "
            "5-table manysql catalog; eval_suite = NL->SQL benchmark "
            "questions on github_events; wikisql = NL->SQL on Wikipedia "
            "tables (use --wikisql-size to subset)."
        ),
    )
    p.add_argument(
        "--eval-suite-limit",
        type=int,
        default=defaults.eval_suite_limit,
        help="Limit eval-suite questions (only with --generator=eval_suite).",
    )
    p.add_argument(
        "--wikisql-size",
        type=int,
        default=defaults.wikisql_size,
        help="Number of WikiSQL examples to sample (only with --generator=wikisql).",
    )
    p.add_argument(
        "--wikisql-split",
        default=defaults.wikisql_split,
        choices=["train", "validation", "test"],
    )
    p.add_argument("--wikisql-seed", type=int, default=defaults.wikisql_seed)
    p.add_argument(
        "--wikisql-sample-rows",
        type=int,
        default=defaults.wikisql_sample_rows,
        help="How many sample rows to embed in each WikiSQL user prompt.",
    )
    # ---- rewards ----
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
    # ---- trainer ----
    p.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    p.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=defaults.per_device_train_batch_size,
        help=(
            "Per-rank train batch. trl 0.22 requires "
            "per_device_train_batch_size * num_processes * grad_accum to "
            "be divisible by num_generations (default 4)."
        ),
    )
    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=defaults.gradient_accumulation_steps,
    )
    p.add_argument(
        "--num-generations",
        type=int,
        default=defaults.num_generations,
        help=(
            "Rollouts per prompt. Effective batch (per_device_bs * "
            "num_processes * grad_accum) must be divisible by this."
        ),
    )
    p.add_argument("--max-steps", type=int, default=defaults.max_steps)
    p.add_argument("--save-steps", type=int, default=defaults.save_steps)
    p.add_argument(
        "--gradient-checkpointing",
        type=lambda s: s.lower() != "false",
        default=defaults.gradient_checkpointing,
        help="Enable gradient checkpointing (use_reentrant=False; DDP-friendly).",
    )
    p.add_argument("--output-dir", default=defaults.output_dir)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip GPU model loading; only exercise dataset + reward funcs (CPU).",
    )
    # ---- vLLM rollout ----
    p.add_argument(
        "--vllm-device",
        default=defaults.vllm_device,
        help=(
            "GPU dedicated to vLLM rollouts (split mode). 'auto' picks "
            "the last visible GPU and assumes you launched accelerate "
            "with --num_processes = (n_gpus - 1). Pass e.g. cuda:7 "
            "explicitly to override."
        ),
    )
    p.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=defaults.vllm_gpu_memory_utilization,
        help=(
            "Mem fraction on the vLLM GPU. Higher is fine in split mode "
            "since the vLLM GPU isn't training; lower in single-GPU "
            "sanity mode where training and vLLM share the device."
        ),
    )
    p.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=defaults.vllm_max_model_len,
        help="vLLM max_model_len. Defaults to --max-seq-length.",
    )
    p.add_argument(
        "--vllm-dtype",
        default=defaults.vllm_dtype,
        help="vLLM dtype (auto / bfloat16 / float16).",
    )
    # ---- logging ----
    p.add_argument("--wandb-project", default=defaults.wandb_project)
    p.add_argument("--wandb-run-name", default=defaults.wandb_run_name)
    p.add_argument("--no-wandb", action="store_true")

    ns = p.parse_args()
    parsed = vars(ns)
    parsed["dialects"] = _split_csv(parsed["dialects"])
    if not parsed["dialects"]:
        p.error("--dialects must contain at least one dialect id")
    return DistributedTrainArgs(**parsed)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
def _rank_print(is_main: bool, msg: str) -> None:
    if is_main:
        print(msg)


def _resolve_vllm_device(arg: str) -> str:
    """Resolve ``--vllm-device auto`` -> ``cuda:<last visible GPU>``.

    Honors ``CUDA_VISIBLE_DEVICES`` -- accelerate launch already remaps
    the visible devices for each rank, so "the last visible GPU" is
    the one we want to dedicate to vLLM regardless of the absolute
    device id.
    """
    if arg != "auto":
        return arg
    import torch  # noqa: PLC0415

    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError(
            "--vllm-device=auto but no CUDA devices visible; pass "
            "--vllm-device cuda:N explicitly or unset CUDA_VISIBLE_DEVICES."
        )
    return f"cuda:{n - 1}"


# ---------------------------------------------------------------------------
# Main (real training, GPU required)
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    if args.dry_run:
        # Dry-run is rank-agnostic by construction (no torch / GPU).
        _dry_run(args)
        return

    # accelerate.PartialState works under `accelerate launch`,
    # `torchrun`, and bare `python` (single-process). is_main_process
    # is True on rank 0 / when there's no distributed init at all.
    from accelerate import PartialState  # noqa: PLC0415

    state = PartialState()
    is_main = state.is_main_process
    world_size = state.num_processes

    # Configure W&B env vars on every rank (HF Trainer gates actual
    # logging to rank 0 itself); only print the banner on rank 0.
    if is_main:
        report_to = _configure_wandb(args)
    else:
        # Mirror env-var setup quietly so HF Trainer sees identical
        # config on all ranks (it reads WANDB_PROJECT / WANDB_NAME).
        report_to = "wandb" if (
            not args.no_wandb and os.environ.get("WANDB_API_KEY")
        ) else "none"
        if report_to == "wandb":
            os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
            os.environ.setdefault("WANDB_WATCH", "false")

    import torch  # noqa: PLC0415
    from peft import LoraConfig  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    from trl import GRPOConfig, GRPOTrainer  # noqa: PLC0415

    vllm_device = _resolve_vllm_device(args.vllm_device)
    vllm_max_model_len = args.vllm_max_model_len or args.max_seq_length

    _rank_print(
        is_main,
        f"[grpo] world_size={world_size} vllm_device={vllm_device} "
        f"loading {args.model_name} (4bit={args.load_in_4bit}, "
        f"lora_rank={args.lora_rank})",
    )

    # ---- divisibility guard (trl 0.22 invariant) ----
    effective = args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps
    if effective % args.num_generations != 0:
        raise ValueError(
            f"trl 0.22 invariant violated: per_device_train_batch_size "
            f"({args.per_device_train_batch_size}) * num_processes "
            f"({world_size}) * gradient_accumulation_steps "
            f"({args.gradient_accumulation_steps}) = {effective} is not "
            f"divisible by num_generations ({args.num_generations}). "
            f"Adjust --per-device-train-batch-size or "
            f"--gradient-accumulation-steps so the product divides."
        )

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        # Most chat-tuned models leave pad unset; reuse eos so DDP
        # padded batches don't break attention masks.
        tokenizer.pad_token = tokenizer.eos_token

    # ---- base model ----
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
    }
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    if args.load_in_4bit:
        from peft import prepare_model_for_kbit_training  # noqa: PLC0415

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    # ---- LoRA ----
    # PEFT is wired through GRPOTrainer(peft_config=...) below. The trl
    # 0.22 patch in GPU_SETUP.md (relaxing the prepare_peft_model
    # dataclasses.replace guard) is required for this to not crash
    # mid-init.
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    # ---- env / dataset ----
    _rank_print(
        is_main,
        f"[grpo] building runtimes for dialects={args.dialects}, "
        f"generator={args.generator}, coverage={args.coverage_mode}",
    )
    runtimes, tasks = build_runtimes_and_tasks(args)
    _rank_print(
        is_main,
        f"[grpo] generated {len(tasks)} tasks across {len(runtimes)} dialect(s)",
    )

    train_ds = build_dataset(args, runtimes, tasks, tokenizer=tokenizer)
    _rank_print(is_main, f"[grpo] kept {len(train_ds)} examples after length filter")

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

    # ---- GRPOConfig ----
    # trl 0.22 surface only -- no top-level top_p/top_k/min_p, no
    # generation_kwargs, no vllm_mode. Sampling temperature is the
    # one knob we set; vLLM defaults handle the rest. If we need more
    # control later, it lands by upgrading the stack (blocked by
    # vllm <-> transformers compatibility today).
    grpo_config = GRPOConfig(
        # ---- generation ----
        temperature=1.0,
        # ---- optim ----
        learning_rate=args.learning_rate,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        bf16=args.bf16,
        # ---- batching / GRPO ----
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=1,
        # ---- DDP / memory ----
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if args.gradient_checkpointing else None
        ),
        ddp_find_unused_parameters=False,
        # ---- vLLM rollout (trl 0.22 split mode) ----
        use_vllm=True,
        vllm_device=vllm_device,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=vllm_max_model_len,
        vllm_dtype=args.vllm_dtype,
        # ---- bookkeeping ----
        report_to=report_to,
        run_name=os.environ.get("WANDB_NAME") if report_to == "wandb" else None,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    # ---- rewards ----
    # No tools= arg: tag mode (the dataset's system prompt instructs
    # the model to wrap SQL in <SQL>...</SQL>; reward functions parse
    # the tag and re-execute through the dialect engine). Each rank
    # holds its own runtimes (DuckDB is per-process), so reward
    # execution parallelises naturally across ranks.
    reward_funcs = make_reward_funcs(args, runtimes)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_ds,
        peft_config=peft_config,
    )

    _rank_print(
        is_main,
        f"[grpo] starting training for {args.max_steps} steps -> {args.output_dir}",
    )
    trainer.train()

    # Only rank 0 writes the final adapter; other ranks wait so we
    # don't race on the output dir or exit before the save completes.
    if is_main:
        lora_dir = os.path.join(args.output_dir, "lora")
        print(f"[grpo] saving LoRA adapter to {lora_dir}")
        trainer.save_model(lora_dir)
    state.wait_for_everyone()

    if is_main:
        print(
            "[grpo] done. To smoke-test the adapter, load "
            f"{args.model_name} + the LoRA at "
            f"{os.path.join(args.output_dir, 'lora')} in a separate "
            "process (this script doesn't run inference inline because "
            "the vLLM lifetime is owned by the trainer)."
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
