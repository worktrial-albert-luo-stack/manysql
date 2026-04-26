"""GRPO fine-tune of Qwen3-4B on GSM8K with Unsloth.

This script is intentionally a *single-file, end-to-end example*. It is the
template we will adapt to RL on the synthetic SQL dialect environments produced
by `manysql/`. Every block that is GSM8K-specific is marked with

    # ---- SQL ADAPTATION HOOK ----

so the seam between "math task" and "manysql SQL task" is obvious.

What it does
------------
1. Loads `unsloth/Qwen3-4B-Instruct-2507` in 4-bit with vLLM-backed fast
   inference + a LoRA adapter (the standard Unsloth GRPO setup).
2. Loads the `openai/gsm8k` train split and reformats each row into a
   `prompt`/`answer` pair where `prompt` is the chat-templated
   system+user messages and `answer` is the gold final number.
3. Defines a small RLVR reward rubric (format-exact, format-soft, integer
   shape, exact-answer match, fuzzy-numeric match) using regex extraction
   on a `<start_working_out>...<end_working_out><SOLUTION>...</SOLUTION>`
   response template.
4. Runs `trl.GRPOTrainer` for `--max-steps` steps and saves the LoRA
   adapter to `outputs/grpo_qwen3_4b_gsm8k/`.

Usage
-----
We don't put the heavy ML deps in `pyproject.toml` (CUDA-pinned torch +
unsloth + vllm doesn't compose well with the rest of the project). Two
modes:

GPU box (real training)::

    uv venv .venv-train --python 3.11
    source .venv-train/bin/activate
    uv pip install --upgrade pip
    uv pip install "unsloth" "vllm" \\
        "transformers==4.56.2" "datasets>=2.20" "torch>=2.4" "accelerate" "bitsandbytes" \\
        "wandb" "python-dotenv"
    uv pip install --no-deps "trl==0.22.2"

    python train/grpo_gsm8k.py --max-steps 200

CPU box (dry-run, no GPU required)::

    uv pip install "transformers" "datasets"
    python train/grpo_gsm8k.py --dry-run --max-examples 8

W&B logging
-----------
If ``WANDB_API_KEY`` is found (in env or `.env`), training logs to W&B
automatically under project ``manysql-grpo`` with a run name derived from
the model/dataset/step count. Override with ``--wandb-project``,
``--wandb-run-name``, or disable entirely with ``--no-wandb``. GRPOTrainer
emits per-reward-function totals as separate metrics, so each reward
component gets its own panel in the W&B UI.

`--dry-run` skips unsloth / torch / trl / vllm entirely. It loads a tiny
slice of GSM8K, runs it through `build_dataset` + the chat template, then
scores a handful of synthetic completions against every reward function
and prints the breakdown. Useful for iterating on dataset format and
reward shaping before you have GPU access.

Hardware note
-------------
Qwen3-4B in 4-bit + LoRA + vLLM fast inference fits in ~16-20 GB VRAM at
2k context with 4 generations per prompt. Drop `--num-generations` and/or
`--max-seq-length` if you OOM. LoRA-16bit (`--load-in-4bit=False`) needs
roughly 2-4x more VRAM but trains a bit better.

Adapting to manysql
-------------------
The four hooks below are everything we need to swap:

* Dataset: `build_dataset` returns `{prompt, answer}` rows. For SQL we will
  return `{prompt, schema, expected_rows, dialect_id}` where the dialect's
  grammar/system-prompt is rendered into the prompt and `expected_rows` is
  the ground-truth result table from running the canonical query through
  `manysql.executor` on the seed Parquet data.
* Reward: `check_answer` / `check_numbers` will be replaced by a single
  `execute_and_compare` reward that (a) extracts the SQL between
  `<SOLUTION>...</SOLUTION>`, (b) parses + executes it through the
  per-dialect engine in `manysql.dialects.<id>`, (c) compares the result
  table to the gold rows via `manysql.oracle`. Format rewards stay as-is.
* System prompt: include the dialect's grammar summary + function list so
  the model knows which keywords/types are legal for this rollout.
* Sampling: keep one dialect per *prompt* (all `num_generations` rollouts
  of the same prompt share the dialect) so group-relative advantages stay
  on a single environment.

Everything else (Unsloth setup, GRPOConfig, trainer loop, LoRA save) is
identical between the math and SQL versions.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

# UNSLOTH_VLLM_STANDBY=1 unlocks ~30% more context length when vLLM and
# training share GPU memory. Must be set before importing unsloth.
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

# Pull WANDB_API_KEY (and OPENAI_*, etc) from .env if present. We use
# override=False so existing environment vars take precedence -- handy when
# CI or a pod injects keys directly.
try:
    from dotenv import load_dotenv  # python-dotenv, already in pyproject.toml
    load_dotenv(override=False)
except ImportError:
    pass

if TYPE_CHECKING:  # only for type hints; never imported at runtime on CPU
    from datasets import Dataset


# Heavy ML deps (unsloth / torch / trl / vllm) are deliberately imported
# *inside* `main()` so this module can be imported on a CPU-only machine
# for dry-run / unit-testing the reward functions and dataset pipeline.


# ---------------------------------------------------------------------------
# Response template
# ---------------------------------------------------------------------------
# We force the model to emit:
#   <start_working_out> ...chain of thought... <end_working_out>
#   <SOLUTION> final answer </SOLUTION>
#
# Using bespoke tags (rather than <think>) means the format rewards are
# unambiguous and the answer extractor is a single regex.
REASONING_START = "<start_working_out>"
REASONING_END = "<end_working_out>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

SYSTEM_PROMPT = (
    "You are given a problem.\n"
    "Think about the problem and provide your working out.\n"
    f"Place it between {REASONING_START} and {REASONING_END}.\n"
    f"Then, provide your final answer between {SOLUTION_START} and {SOLUTION_END}."
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
@dataclass
class TrainArgs:
    model_name: str = "unsloth/Qwen3-4B-Instruct-2507"
    max_seq_length: int = 2048
    lora_rank: int = 32
    load_in_4bit: bool = True
    gpu_memory_utilization: float = 0.85
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    dataset_split: str = "train"
    max_examples: int | None = None
    learning_rate: float = 5e-6
    num_generations: int = 4
    max_steps: int = 200
    save_steps: int = 100
    output_dir: str = "outputs/grpo_qwen3_4b_gsm8k"
    seed: int = 3407
    dry_run: bool = False
    wandb_project: str = "manysql-grpo"
    wandb_run_name: str | None = None
    no_wandb: bool = False


def parse_args() -> TrainArgs:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    defaults = TrainArgs()
    p.add_argument("--model-name", default=defaults.model_name)
    p.add_argument("--max-seq-length", type=int, default=defaults.max_seq_length)
    p.add_argument("--lora-rank", type=int, default=defaults.lora_rank)
    p.add_argument("--load-in-4bit", type=lambda s: s.lower() != "false", default=defaults.load_in_4bit)
    p.add_argument("--gpu-memory-utilization", type=float, default=defaults.gpu_memory_utilization)
    p.add_argument("--dataset-name", default=defaults.dataset_name)
    p.add_argument("--dataset-config", default=defaults.dataset_config)
    p.add_argument("--dataset-split", default=defaults.dataset_split)
    p.add_argument("--max-examples", type=int, default=defaults.max_examples)
    p.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    p.add_argument("--num-generations", type=int, default=defaults.num_generations)
    p.add_argument("--max-steps", type=int, default=defaults.max_steps)
    p.add_argument("--save-steps", type=int, default=defaults.save_steps)
    p.add_argument("--output-dir", default=defaults.output_dir)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip GPU model loading; only exercise dataset + reward funcs (CPU only).",
    )
    p.add_argument(
        "--wandb-project",
        default=defaults.wandb_project,
        help="W&B project name (used when WANDB_API_KEY is set).",
    )
    p.add_argument(
        "--wandb-run-name",
        default=defaults.wandb_run_name,
        help="W&B run name; defaults to <model>-<dataset>-<max_steps>steps.",
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Force-disable W&B logging even if WANDB_API_KEY is set.",
    )
    ns = p.parse_args()
    return TrainArgs(**vars(ns))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def extract_gsm8k_answer(answer_field: str) -> str:
    """GSM8K stores answers as `<reasoning>\\n#### <number>`. Take what's after `####`."""
    if "####" not in answer_field:
        return answer_field.strip()
    return answer_field.split("####", 1)[1].strip().replace(",", "")


def build_dataset(args: TrainArgs, tokenizer) -> "Dataset":
    """Load + format the training dataset.

    Each row becomes ``{"prompt": [chat messages], "answer": gold_string}``.
    `GRPOTrainer` will auto-apply the tokenizer's chat template to `prompt`.

    # ---- SQL ADAPTATION HOOK ----
    For manysql we will replace this with something like::

        from manysql.dialects.registry import sample_dialect
        from manysql.codegen.tasks import sample_task

        def build_dataset(args, tokenizer):
            rows = []
            for _ in range(args.max_examples):
                dialect = sample_dialect()
                task = sample_task(dialect)  # NL question + schema + gold rows
                rows.append({
                    "prompt": [
                        {"role": "system",
                         "content": SQL_SYSTEM_PROMPT.format(
                             grammar_summary=dialect.grammar_summary(),
                             function_list=dialect.function_list(),
                         )},
                        {"role": "user",
                         "content": task.render_user_message()},
                    ],
                    "dialect_id": dialect.id,
                    "schema": task.schema_json,
                    "expected_rows": task.expected_rows,  # arrow-serializable
                })
            return Dataset.from_list(rows)
    """
    from datasets import load_dataset

    ds = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    if args.max_examples is not None:
        ds = ds.select(range(min(args.max_examples, len(ds))))

    def format_row(row: dict) -> dict:
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["question"]},
            ],
            "answer": extract_gsm8k_answer(row["answer"]),
        }

    ds = ds.map(format_row, remove_columns=ds.column_names)

    # Filter out prompts that don't fit so vLLM doesn't waste cycles.
    def prompt_token_count(row: dict) -> int:
        return len(
            tokenizer.apply_chat_template(
                row["prompt"], add_generation_prompt=True, tokenize=True
            )
        )

    ds = ds.map(lambda r: {"_n": prompt_token_count(r)})
    cap = args.max_seq_length // 2
    ds = ds.filter(lambda r: r["_n"] <= cap).remove_columns(["_n"])
    return ds


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------
# `match_format`: matches a complete <end_working_out>...<SOLUTION>X</SOLUTION>
# block at end of the response. The opening <start_working_out> is auto-prepended
# by the chat template, so we don't require it here.
SOLUTION_END_RE = (
    re.escape(SOLUTION_END) + r"[\s]{0,}"
)  # tokenizer.eos_token is added in build_match_format

MATCH_NUMBER = re.compile(
    re.escape(SOLUTION_START) + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
    flags=re.MULTILINE | re.DOTALL,
)


def build_match_format(eos_token: str) -> re.Pattern:
    end_re = SOLUTION_END_RE + r"(?:" + re.escape(eos_token) + r")?"
    return re.compile(
        re.escape(REASONING_END) + r".*?"
        + re.escape(SOLUTION_START) + r"(.+?)" + end_re
        + r"[\s]{0,}$",
        flags=re.MULTILINE | re.DOTALL,
    )


def make_reward_funcs(tokenizer):
    match_format = build_match_format(tokenizer.eos_token)

    def match_format_exactly(completions, **_):
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            scores.append(3.0 if match_format.search(response) is not None else 0.0)
        return scores

    def match_format_approximately(completions, **_):
        """Reward seeing each tag exactly once. Opening tag is prepended for us."""
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            score = 0.0
            score += 0.5 if response.count(REASONING_END) == 1 else -1.0
            score += 0.5 if response.count(SOLUTION_START) == 1 else -1.0
            score += 0.5 if response.count(SOLUTION_END) == 1 else -1.0
            scores.append(score)
        return scores

    # ---- SQL ADAPTATION HOOK ----
    # The two reward funcs below (`check_answer`, `check_numbers`) are the
    # GSM8K-specific verifiers. For manysql, replace them with a single
    # execution-based reward, e.g.:
    #
    #   def execute_and_compare(prompts, completions, dialect_id, schema,
    #                           expected_rows, **_):
    #       scores = []
    #       for completion, did, sch, gold in zip(
    #           completions, dialect_id, schema, expected_rows
    #       ):
    #           sql = match_format.search(completion[0]["content"])
    #           if sql is None:
    #               scores.append(-2.0); continue
    #           try:
    #               result = manysql.executor.run(
    #                   manysql.dialects.get(did).parse(sql.group(1)),
    #                   schema=sch,
    #               )
    #               scores.append(5.0 if result == gold else -1.0)
    #           except manysql.executor.ExecError:
    #               scores.append(-3.0)
    #       return scores
    #
    # The differential oracle in manysql/oracle/ can also be plugged in to
    # cross-check generated SQL against DuckDB/SQLite for extra reward signal.
    def check_answer(prompts, completions, answer, **_):
        responses = [c[0]["content"] for c in completions]
        extracted = [
            (m.group(1) if (m := match_format.search(r)) is not None else None)
            for r in responses
        ]
        scores = []
        for guess, true_answer in zip(extracted, answer):
            if guess is None:
                scores.append(-2.0)
                continue
            score = 0.0
            if guess == true_answer:
                score += 5.0
            elif guess.strip() == true_answer.strip():
                score += 3.5
            else:
                try:
                    ratio = float(guess) / float(true_answer)
                    if 0.9 <= ratio <= 1.1:
                        score += 2.0
                    elif 0.8 <= ratio <= 1.2:
                        score += 1.5
                    else:
                        score -= 2.5
                except (ValueError, ZeroDivisionError):
                    score -= 4.5
            scores.append(score)
        return scores

    printed = {"n": 0}

    def check_numbers(prompts, completions, answer, **_):
        question = prompts[0][-1]["content"]
        responses = [c[0]["content"] for c in completions]
        extracted = [
            (m.group(1) if (m := MATCH_NUMBER.search(r)) is not None else None)
            for r in responses
        ]
        if printed["n"] % 5 == 0:
            print(
                "*" * 20,
                f"\nQuestion:\n{question}",
                f"\nAnswer:\n{answer[0]}",
                f"\nResponse:\n{responses[0]}",
                f"\nExtracted:\n{extracted[0]}",
            )
        printed["n"] += 1

        scores = []
        for guess, true_answer in zip(extracted, answer):
            if guess is None:
                scores.append(-2.5)
                continue
            try:
                t = float(str(true_answer).strip().replace(",", ""))
                g = float(str(guess).strip().replace(",", ""))
                scores.append(3.5 if g == t else -1.5)
            except ValueError:
                scores.append(0.0)
        return scores

    return [
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ]


# ---------------------------------------------------------------------------
# Dry-run (CPU-only smoke test)
# ---------------------------------------------------------------------------
def _dry_run(args: TrainArgs) -> None:
    """Exercise dataset + reward pipeline without unsloth / torch / vllm.

    Requires only ``transformers`` and ``datasets`` (CPU-only). Loads the
    tokenizer from HF, builds a small slice of GSM8K, then scores a fixed
    set of synthetic completions through every reward function so you can
    eyeball the rubric.
    """
    from transformers import AutoTokenizer

    print(f"[dry-run] loading tokenizer for {args.model_name} (CPU only)")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"

    n = args.max_examples or 4
    dry_args = TrainArgs(**{**args.__dict__, "max_examples": n})
    train_ds = build_dataset(dry_args, tokenizer)
    print(f"[dry-run] built dataset with {len(train_ds)} rows")
    sample = train_ds[0]
    print("[dry-run] sample row:")
    for msg in sample["prompt"]:
        print(f"  {msg['role']}: {msg['content'][:120]}{'...' if len(msg['content']) > 120 else ''}")
    print(f"  answer: {sample['answer']}")

    rendered = tokenizer.apply_chat_template(
        sample["prompt"], add_generation_prompt=True, tokenize=False
    )
    print(f"[dry-run] rendered prompt ({len(rendered)} chars):\n{rendered}\n")

    gold = sample["answer"]
    eos = tokenizer.eos_token
    fake_completions = [
        # 1. Perfect: well-formed + correct answer
        [{"role": "assistant",
          "content": f"step by step...{REASONING_END}{SOLUTION_START}{gold}{SOLUTION_END}{eos}"}],
        # 2. Well-formed but wrong number
        [{"role": "assistant",
          "content": f"step by step...{REASONING_END}{SOLUTION_START}999999{SOLUTION_END}{eos}"}],
        # 3. Off-by-a-bit (tests fuzzy ratio reward)
        [{"role": "assistant",
          "content": (
              f"step by step...{REASONING_END}{SOLUTION_START}"
              f"{int(float(gold)) + 1 if gold.lstrip('-').replace('.', '', 1).isdigit() else gold}"
              f"{SOLUTION_END}{eos}"
          )}],
        # 4. Missing tags entirely
        [{"role": "assistant", "content": f"The answer is {gold}."}],
        # 5. Tags duplicated (format approximate should penalise)
        [{"role": "assistant",
          "content": (
              f"step{REASONING_END}{REASONING_END}{SOLUTION_START}{gold}{SOLUTION_END}"
              f"{SOLUTION_START}{gold}{SOLUTION_END}{eos}"
          )}],
    ]
    fake_prompts = [sample["prompt"]] * len(fake_completions)
    fake_answers = [gold] * len(fake_completions)

    reward_funcs = make_reward_funcs(tokenizer)

    # TRL's GRPOTrainer calls reward funcs with kwargs sourced from the
    # dataset columns, so we mirror that here. Each func absorbs unknown
    # kwargs via `**_`.
    def call(fn, prompts, completions, answers):
        return fn(prompts=prompts, completions=completions, answer=answers)

    print("[dry-run] reward breakdown:")
    header = ["#"] + [fn.__name__ for fn in reward_funcs] + ["total"]
    print("  " + "  ".join(f"{h:>28}" for h in header))
    all_scores = [
        call(fn, fake_prompts, fake_completions, fake_answers) for fn in reward_funcs
    ]
    for i in range(len(fake_completions)):
        row = [str(i)]
        total = 0.0
        for col_scores in all_scores:
            row.append(f"{col_scores[i]:.2f}")
            total += col_scores[i]
        row.append(f"{total:.2f}")
        print("  " + "  ".join(f"{c:>28}" for c in row))

    print("\n[dry-run] OK — dataset + rewards look healthy. "
          "Run without --dry-run on a GPU box to actually train.")


# ---------------------------------------------------------------------------
# Main (real training, GPU required)
# ---------------------------------------------------------------------------
def _configure_wandb(args: TrainArgs) -> str:
    """Decide report_to backend and configure W&B env vars.

    Returns the value to pass to ``GRPOConfig(report_to=...)`` -- either
    ``"wandb"`` (when a key is available and not disabled) or ``"none"``.
    HF Trainer reads ``WANDB_PROJECT`` / ``WANDB_NAME`` from env, so we set
    them here before constructing the trainer.
    """
    if args.no_wandb:
        print("[wandb] disabled via --no-wandb")
        return "none"
    if not os.environ.get("WANDB_API_KEY"):
        print("[wandb] WANDB_API_KEY not set (.env or environment); logging disabled")
        return "none"

    # CLI wins over env so re-runs with a different --wandb-project
    # actually take effect.
    os.environ["WANDB_PROJECT"] = args.wandb_project
    run_name = args.wandb_run_name or (
        f"{args.model_name.split('/')[-1]}-"
        f"{args.dataset_name.split('/')[-1]}-{args.max_steps}steps"
    )
    os.environ["WANDB_NAME"] = run_name
    # Param/grad logging is noisy at 4B params; users can override.
    os.environ.setdefault("WANDB_WATCH", "false")
    print(f"[wandb] enabled: project={args.wandb_project} run={run_name}")
    return "wandb"


def main() -> None:
    args = parse_args()

    if args.dry_run:
        _dry_run(args)
        return

    report_to = _configure_wandb(args)

    from unsloth import FastLanguageModel  # noqa: PLC0415  GPU-only import
    import torch  # noqa: PLC0415
    from trl import GRPOConfig, GRPOTrainer  # noqa: PLC0415
    from vllm import SamplingParams  # noqa: PLC0415

    print(f"[grpo] loading {args.model_name} (4bit={args.load_in_4bit}, lora_rank={args.lora_rank})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=True,  # vLLM rollout backend
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

    print(f"[grpo] building dataset {args.dataset_name}/{args.dataset_config}:{args.dataset_split}")
    train_ds = build_dataset(args, tokenizer)
    print(f"[grpo] kept {len(train_ds)} examples after length filter")

    # Size the prompt/completion budgets from actual data.
    prompt_lens = [
        len(tokenizer.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=True))
        for r in train_ds
    ]
    max_prompt_length = max(prompt_lens) + 1
    max_completion_length = args.max_seq_length - max_prompt_length
    if max_completion_length <= 128:
        raise ValueError(
            f"max_completion_length={max_completion_length} too small; "
            f"raise --max-seq-length (currently {args.max_seq_length})"
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

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=make_reward_funcs(tokenizer),
        args=grpo_config,
        train_dataset=train_ds,
    )

    print(f"[grpo] starting training for {args.max_steps} steps -> {args.output_dir}")
    trainer.train()

    lora_dir = os.path.join(args.output_dir, "lora")
    print(f"[grpo] saving LoRA adapter to {lora_dir}")
    model.save_lora(lora_dir)

    # Quick smoke-test generation with the trained adapter.
    smoke_q = "Janet has 3 apples. She gives 1 to Bob and buys 5 more. How many apples does she have?"
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": smoke_q},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    sampling = SamplingParams(temperature=0.7, top_k=50, max_tokens=512)
    out = model.fast_generate(
        text, sampling_params=sampling, lora_request=model.load_lora(lora_dir)
    )[0].outputs[0].text
    print("[grpo] smoke-test response:\n" + out)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
