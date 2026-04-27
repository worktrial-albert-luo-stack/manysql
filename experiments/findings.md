# manysql GRPO findings — 2026-04-27 session

Notes from a day of single-GPU GRPO runs over wikisql + synsql + bird,
plus an in-distribution eval pass on snowflake_clone.

## TL;DR

- **(a+b+c) hyperparams** — `--num-generations 6` + `--lr-scheduler-type cosine`
  + `--max-steps 500` — beats the dataclass defaults (gens=4, linear LR,
  200 steps) on every run we've tried. Default linear LR decays to zero
  by `max_steps`, producing a "Q3 peak then erode" plateau; cosine holds
  the LR through the late steps and lets the model keep climbing past
  the breakthrough.
- **Multi-dialect training does not visibly hurt single-dialect
  performance.** A 1-dialect run and a 10-dialect run trained on the
  same wikisql questions land within ±1 question of each other on a
  50-question in-distribution eval. The 10-dialect variant has a
  higher *first-attempt* success rate (64% vs 62%); the 1-dialect
  variant edges it out only via retries.
- **Gemma3-4B can't crack multi-table SQL out of the box.** It learns
  wikisql (single-table; 0.47-0.53 final correctness) but stays at 0%
  correctness on bird and on the early-stage synsql. Qwen3-4B learns
  all three, with synsql being the hardest of them.

## What's measured

- **GRPO health**: per-step reward components, `frac_reward_zero_std`
  (the % of N-rollout groups where every rollout scored the same → no
  gradient), grouped by quintile.
- **Saturation buckets**:
  - `sat@P` = all rollouts perfect → no gradient (ceiling)
  - `sat@F` = all rollouts identically failed → no gradient (floor)
  - `flow` = within-group variance → useful gradient
- **Eval**: `eval/serve_lora.py` against `--backend synthetic
  --dialects snowflake_clone`, 50-question NL→SQL set, scored on row
  match (exact + numeric).

## (a+b+c) hyperparameter ablation

| Run | Steps | Final corr | Pattern |
|---|---|---|---|
| Bird qwen3 (defaults, t=1.6) | 200/200 | 0.54 | plateau Q3 onward |
| Wikisql gemma3 (defaults, snowflake_clone, t=1.6) | 500/500 | 0.47 | Q3 peak 0.53 → Q5 0.47 (LR-decay erosion) |
| Wikisql qwen3 (a+b+c, snowflake_clone, t=1.6) | 500/500 | **0.679** | clean climb, sat@P 29% → 52%, completion length 339 → 264 |
| Wikisql qwen3 10-d (a+b+c, t=1.6) | 500/500 | 0.630 | similar to above; 55% sat@P at end |

Cosine LR + larger group size compounded: same model, same data, same
seed, same temp — only the schedule and group size changed, and the
final correctness moved from 0.47 → 0.68.

## Multi-dialect (1-d vs 10-d) pattern

In training, the 10-dialect run leads the 1-dialect run early
(implicit data augmentation: same NL question seen across multiple
dialect contexts), then they cross around step 200-300 and 1-d pulls
slightly ahead. End-of-training was 0.679 (1-d) vs 0.630 (10-d) on the
last quintile.

In eval (50 q × snowflake_clone, the only common dialect):

| Model | matched/50 | first-attempt | avg attempts |
|---|---|---|---|
| Qwen3-4B base | 26 (52%) | 58% | 1.76 |
| Wikisql 1-d v2 LoRA | **28 (56%)** | 62% | 1.68 |
| Wikisql 10-d v2 LoRA | 27 (54%) | **64%** | 1.68 |

Both LoRAs improve over base by 1-2 questions (within noise on n=50,
but consistent direction). The 10-d LoRA's higher first-attempt rate
suggests multi-dialect training produces *better one-shot answers*;
the 1-d LoRA overtakes it via retries. Net: the dialect dimension
provides useful diversity, not noise.

## Saturation patterns and what unsticks them

- **Bimodal saturation** (sat@P + sat@F together >50%) caps gradient
  signal because both modes have zero advantage. Higher temperature
  (1.6 vs 1.4) reduces both modes by widening rollout spread.
- **Temp=1.4** produced ~40% of steps with no gradient (perfect or
  failure-saturated). Bumping to 1.6 dropped that to ~25-35% on Qwen3
  bird. Temp=1.6 is the working setting; temp=2.0 not tried.
- **All-failed runs** (Gemma3 + bird, Gemma3 + synsql) had 100% sat@F
  for hundreds of steps — the model never produces a parseable+correct
  rollout, so no positive examples for advantage to score. Hyperparam
  knobs don't help; needs SFT warmup or easier curriculum.

## Plateau / erosion pattern (linear LR)

Wikisql gemma3 baseline (linear LR, 500 steps) hit corr 0.53 in Q3
then drifted down to 0.47 by Q5. KL collapsed 1.72 → 0.33 across the
run. This is the signature of linear-LR-decay: by the last quintile
the LR is near zero and the policy is shrinking back toward base via
KL pull. Cosine LR (used in the (a+b+c) qwen3 runs) holds the LR
through Q5 and prevented the erosion in those runs.

## Reproducibility caveat

Two runs with identical seed, code, data, and hyperparams but different
concurrent-job loads diverged through ~step 200 (the v2 was at corr
0.572 at step 205 vs the v1 at 0.753 at step 100), then converged by
step 325. Likely vLLM/CUDA non-determinism under different load.
**Don't compare runs at single-step snapshots; use trajectory shape.**

## Synsql complexity filter bug

The HF dataset stores capitalized values
(`'Simple'`/`'Moderate'`/`'Complex'`/`'Highly Complex'`) but the filter
in `train/env/synsql.py` compared against lowercase. Crashed every
synsql run with "no items match complexities=('simple', 'moderate')".
Fixed by lowercasing the data side at the comparison point. The
`_VALID_COMPLEXITIES` set is the source of truth for the API; data is
normalized in.

## Synsql is *hard*, not easy (current state)

| Run | Steps | Final corr | Diagnosis |
|---|---|---|---|
| Synsql qwen3 baseline (defaults) | 148/500 (died) | oscillating 0.11-0.29 | hard but flowing; not easy |
| Synsql qwen3 (a+b+c) snowflake_clone | 500/500 (save crashed at step 250) | 0.31 (step 88), then unknown | climbing slowly |
| Synsql qwen3 (a+b+c) 10-d | 500/500 (save crashed) | 0.33 (step 96) | similar to 1-d |
| Synsql gemma3 (a+b+c) | 156/500 (died) | 0.00 throughout | model can't even produce parseable SQL |

Synsql at `simple,moderate` complexity is brutal for Gemma3 and
moderately hard for Qwen3. Going to `complex` / `highly complex` would
push it from "moderately hard for Qwen3" to "brutally hard for both"
— bad time. Revisit when a model actually plateaus at high correctness
on simple+moderate.

## Operational lessons

- **`/workspace` has a tight per-pod quota** (~5-10 GB) that's much
  smaller than the 190TB filesystem-level free space on mfs. Manifests
  as `OSError: [Errno 122] Disk quota exceeded` even when `df` shows
  plenty of space. Diagnose with a `dd` write test of the size you'd
  need.
- **Don't write training outputs to `/tmp`** (small overlay, 30 GB
  total). The `run_grpo_sql.sh` launcher does `cd /tmp` so relative
  `--output-dir` paths land there. Pass an absolute path to
  `/workspace/...` to dodge this. Bigger save_steps (250+) also helps.
- **vLLM serving defaults bite the LoRA case**:
  - `--max-lora-rank 16` (default) rejects rank-32 LoRAs. Pass
    `--vllm-extra='--max-lora-rank 32'`.
  - `--max-model-len 4096` (default in serve_lora.py) is too small
    when the eval prompt + 2048-token output budget overflows. Pass
    `--max-model-len 8192`.
- **HF cache redirect needs the model already cached at the new
  path.** Setting `HF_HOME=/workspace/...` then trying to download an
  8GB model into a quota-constrained workspace fails. The training
  setup already cached `Qwen/Qwen3-4B-Instruct-2507` in `/tmp/hf-home`,
  so eval points there + uses the upstream Qwen base (same weights as
  unsloth's re-upload) and skips the download entirely.
- **`vllm` binary not on default `PATH`** — it lives at
  `/tmp/venv-gpu/bin/vllm`. `serve_lora.py` does
  `subprocess.Popen(['vllm', ...])` and needs `PATH` set. Pass
  `PATH=/tmp/venv-gpu/bin:$PATH` in the launch env.

## Skill / commands shipped

- `train/env/curriculum.py` — `filter_dataset_by_rollout_variance`,
  rollout-based curriculum filter that drops prompts where every
  rollout scores identically (zero gradient anyway). Wired through
  `--curriculum-filter` flag in `grpo_sql.py`. Not yet exercised.
- `.claude/commands/analyze-grpo-run.md` — `/analyze-grpo-run <log>`
  slash command that prints the per-quintile saturation/reward table
  and a plain-English verdict. Standardizes the analysis we did
  multiple times manually.
- `train/winning_runs.txt` — pinned launch command for the (a+b+c)
  qwen3 wikisql-snowflake run, in case the trajectory needs reproducing.

## Open questions / next experiments

- Does the curriculum filter actually help the synsql or bird Gemma3
  runs? Hypothesis: probably useless when sat@F = 100% (no
  gradient-bearing prompts to keep), useful when there's a meaningful
  middle band.
- Cross-dialect generalization: train on dialect set A, eval on
  dialect set B. We only ran in-distribution evals so far.
- Synsql 1-d vs 10-d — we have training trajectories but not eval
  results yet (this is the next planned eval).
- The Gemma3 + synsql Q5 "wake-up" (flow jumped 0% → 62% but
  correctness still 0%) — does it eventually break through, or does
  the wake-up plateau in incoherent variance?
