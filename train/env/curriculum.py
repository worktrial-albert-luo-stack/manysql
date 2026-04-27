"""Rollout-based curriculum filter for GRPO training.

Prunes training prompts that contribute zero learning signal -- those
where every base-model rollout gets the same correctness score (all
perfect, all failed, or all stuck on the same partial). GRPO computes
group-relative advantages across the N rollouts of a single prompt,
so when ``min(correctness) == max(correctness)`` the advantage is zero
and the prompt burns step budget without moving the policy.

Run once before training to tighten the dataset around the
gradient-bearing slice. Reuses ``score_completion`` from
:mod:`train.env.trl` so the filter sees exactly the score signal the
trainer will see at training time.

Performance note:
    Costs ``len(dataset) * num_rollouts`` extra vLLM generations up
    front. On an H100 with a 4B-parameter model this is roughly
    1-2 minutes per 1000 prompts at ``num_rollouts=4``, vs. tens of
    minutes saved later by skipping zero-advantage steps.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from train.env.rewards import RewardConfig
from train.env.trl import score_completion

if TYPE_CHECKING:
    from datasets import Dataset

    from train.env.engine import DialectRuntime


def filter_dataset_by_rollout_variance(
    dataset: Dataset,
    *,
    model: Any,
    tokenizer: Any,
    runtimes: dict[str, DialectRuntime],
    sampling_params: Any,
    reward_config: RewardConfig | None = None,
    batch_size: int = 16,
    progress_every: int = 100,
) -> Dataset:
    """Drop rows whose base-model rollouts all score the same correctness.

    Args:
        dataset: HF Dataset built by :func:`tasks_to_dataset`. Must have
            ``prompt``, ``dialect``, ``gold_rows_json`` columns.
        model: Unsloth FastLanguageModel (must expose ``.fast_generate``).
        tokenizer: HF tokenizer or multimodal processor matching
            ``model``. Used to render the chat-form prompt to text.
        runtimes: Dict from dialect name to set-up
            :class:`~train.env.engine.DialectRuntime`. The row's
            ``dialect`` column picks which runtime scores its rollouts.
        sampling_params: ``vllm.SamplingParams`` with ``n=num_rollouts``
            set to the desired group size. Should generally match (or
            roughly match) what training will use, so the filter
            doesn't drop prompts that would have been tractable under
            the trainer's own sampling.
        reward_config: Forwarded to ``score_completion``. Defaults to
            ``RewardConfig()``.
        batch_size: How many prompts to send to vLLM per call. Each call
            yields ``batch_size * sampling_params.n`` completions.
        progress_every: Print progress after this many rows scanned.

    Returns:
        A subset of the input dataset, in original row order, containing
        only rows where rollout correctness has within-group variance.
    """
    cfg = reward_config or RewardConfig()
    n = len(dataset)
    n_rollouts = getattr(sampling_params, "n", None) or 1
    if n_rollouts < 2:
        raise ValueError(
            "filter_dataset_by_rollout_variance: sampling_params.n must be "
            f">= 2 to detect variance (got {n_rollouts})"
        )

    # Pre-render prompts. Multimodal processors return strings here too
    # (tokenize=False), so the same call works for Qwen3 and Gemma3.
    rendered = [
        tokenizer.apply_chat_template(
            row["prompt"], add_generation_prompt=True, tokenize=False
        )
        for row in dataset
    ]

    keep_mask: list[bool] = [False] * n
    n_all_perfect = 0
    n_all_failed = 0
    n_all_partial = 0  # all rollouts hit same non-zero non-one correctness

    t0 = time.time()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = rendered[start:end]
        outs = model.fast_generate(batch, sampling_params=sampling_params)
        for j, req_out in enumerate(outs):
            i = start + j
            row = dataset[i]
            gold_rows = json.loads(row["gold_rows_json"])
            runtime = runtimes[row["dialect"]]
            scores: list[float] = []
            for cout in req_out.outputs:
                completion = [{"role": "assistant", "content": cout.text}]
                breakdown = score_completion(
                    completion=completion,
                    gold_rows=gold_rows,
                    runtime=runtime,
                    cfg=cfg,
                )
                scores.append(breakdown.correctness)
            if min(scores) != max(scores):
                keep_mask[i] = True
            else:
                v = scores[0]
                if v >= 0.999:
                    n_all_perfect += 1
                elif v <= 0.001:
                    n_all_failed += 1
                else:
                    n_all_partial += 1

        if (end % progress_every) < batch_size or end == n:
            kept = sum(keep_mask)
            elapsed = time.time() - t0
            rate = end / elapsed if elapsed > 0 else 0.0
            print(
                f"[curriculum] scanned {end}/{n} prompts ({rate:.1f}/s), "
                f"kept {kept}, dropped: perfect={n_all_perfect} "
                f"failed={n_all_failed} partial={n_all_partial}",
                flush=True,
            )

    kept = sum(keep_mask)
    print(
        f"[curriculum] DONE: kept {kept}/{n} ({100*kept/max(n,1):.1f}%); "
        f"dropped {n_all_perfect} all-perfect, {n_all_failed} all-failed, "
        f"{n_all_partial} all-same-partial in {time.time()-t0:.1f}s",
        flush=True,
    )
    return dataset.select([i for i, k in enumerate(keep_mask) if k])


__all__ = ["filter_dataset_by_rollout_variance"]
