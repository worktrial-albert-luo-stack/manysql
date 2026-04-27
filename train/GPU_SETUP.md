# GPU training setup (H100 pod)

This file captures the working stack for `train/grpo_gsm8k.py` and
`train/grpo_sql.py` on the project H100 pod, plus every gotcha hit while
getting it stable. Read this *before* re-installing.

## Quick start

```bash
# launchers do "cd /tmp" + set HF caches + invoke the python entrypoint
bash train/run_grpo_gsm8k.sh --max-steps 200 --no-wandb
bash train/run_grpo_sql.sh   --dialects aggressive_alien --max-steps 200 --no-wandb
```

Training caches and the `grpo_trainer_lora_model/` temp dir are written
relative to `cwd`, so the launchers `cd /tmp` first to dodge the
`/workspace` NFS quota.

## Working stack

| Component       | Pinned version            | Notes                                   |
| --------------- | ------------------------- | --------------------------------------- |
| python          | 3.12 (system)             | venv at `/tmp/venv-gpu`                 |
| torch           | 2.10.0+cu128              | NOT 2.11/cu130 (driver is 12.8)         |
| torchvision     | 0.25.0+cu128              | 0.26 is for torch 2.11 — crashes nms    |
| vllm            | 0.17.0                    | newer wheels need transformers>=5       |
| transformers    | 4.57.6                    | <5 because vllm 0.17 requires it        |
| trl             | 0.22.2                    | 0.26+ needs transformers>=5             |
| datasets        | >=2.20,<4                 | 4.x broke arrow writer API              |
| unsloth         | latest                    | imported before trl/transformers        |

## Fresh install

```bash
# 1. Build venv outside /workspace (workspace has a per-user NFS quota)
python3.12 -m venv --system-site-packages /tmp/venv-gpu
export UV_CACHE_DIR=/tmp/uv-cache
PIP="/tmp/venv-gpu/bin/python -m pip"

# 2. Torch matched to driver (cu128 for driver 12.8)
$PIP install --index-url https://download.pytorch.org/whl/cu128 \
    "torch>=2.9,<2.11" "torchvision==0.25.0+cu128"

# 3. Inference + training stack
$PIP install "vllm==0.17.0" "transformers==4.57.6" "trl==0.22.2" \
             "datasets>=2.20,<4" unsloth wandb

# 4. Project deps + editable install (no-deps avoids resolver dragging in CPU torch)
$PIP install polars duckdb lark pydantic httpx rich
$PIP install -e /workspace/manysql --no-deps
```

## Critical gotchas

### `/workspace` quota (~17GB)
Pip installs and HF caches must NOT land in `/workspace`. Set:
```
export HF_HOME=/tmp/hf-home
export HF_DATASETS_CACHE=/tmp/hf-datasets
export UV_CACHE_DIR=/tmp/uv-cache
```
EDQUOT errors during dataset writes look like a pyarrow bug
(`I/O operation on closed file`) — it's actually the disk full.

### `UNSLOTH_VLLM_STANDBY=0` (mandatory on H100)
Sleep mode + CUDA graphs causes `cudaErrorIllegalAddress` around step 3.
The launchers export this. Don't unset.

### `cd /tmp` before training
Unsloth writes `grpo_trainer_lora_model/` relative to cwd. Inside
`/workspace` the LoRA write blows the quota mid-run.

### torch / vllm ABI mismatch
vllm wheels >=0.6.x are compiled against torch >=2.9 (the
`c10_cuda_check_implementation` symbol changed `int` -> `unsigned int`).
System torch 2.8 produces `undefined symbol: ...S2_jb`. Always install
torch >=2.9 *inside* the venv even when using `--system-site-packages`.

### torchvision pin
`torchvision 0.26.0+cu128` against torch 2.10 crashes with
`torchvision::nms does not exist`. Use 0.25.0+cu128.

### datasets 4.x breaks arrow writer
`datasets>=4` raises `ValueError: I/O operation on closed file` on
`Dataset.from_list`. Pin `<4`.

### trl 0.22.2 + vllm 0.17.0: GuidedDecodingParams import
trl 0.22.2's `grpo_trainer.py` imports `GuidedDecodingParams` from
`vllm.sampling_params` at module load. vllm 0.17.0 doesn't ship that
symbol, so `from trl import GRPOTrainer` raises ImportError. Patch
`/tmp/venv-gpu/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py`
to wrap the import in `try/except ImportError: GuidedDecodingParams =
None`. We don't use guided decoding so the stub is harmless.

### Unsloth must import BEFORE trl (or vllm OOMs)
Unsloth dynamically monkey-patches `trl.trainer.grpo_trainer` so GRPOTrainer
shares unsloth's vllm engine. If `from trl import GRPOConfig, GRPOTrainer`
runs first, the patches don't take effect — trl spins up a *second* vllm
engine, instantly OOMing against unsloth's. Always:

```python
from unsloth import FastLanguageModel  # FIRST
from trl import GRPOConfig, GRPOTrainer  # SECOND
```

The patched GRPOConfig also accepts `vllm_sampling_params=SamplingParams(...)`
as a kwarg even though stock trl 0.22.2 doesn't. Use it — passing
`use_vllm=False` won't help because unsloth re-enables vllm anyway.

### trl 0.22.2 + dataclasses.replace() in prepare_peft_model
`trl/models/utils.py:545` runs `dataclasses.replace(args, gradient_checkpointing=False)`,
which re-runs `GRPOConfig.__post_init__` with both `generation_batch_size` and
`steps_per_generation` already set, hitting the "can not be both configured"
guard. Patch `trl/trainer/grpo_config.py` line ~628: relax the else branch to
allow consistent values (both auto-derived from the same `per_device_train_batch_size
* num_processes * steps_per_generation`).

### per_device_train_batch_size must be ≥ num_generations
trl 0.22's `__post_init__` computes `generation_batch_size =
per_device_train_batch_size * num_processes * gradient_accumulation_steps` and
asserts divisibility by `num_generations`. With 1 GPU and grad_accum=1, the
batch size must equal num_generations (default 4).

### SQL prompt-length filter
`grpo_sql.py` builds dataset prompts that include the full schema card (~2.1k
tokens for `aggressive_alien`). The gsm8k-style cap of `max_seq_length // 2`
drops every example. Use `max_seq_length - 1024` instead — leaves a 1024-token
completion budget.

### Tools API conflict (resolved by tag-based SQL)
trl 0.26.2 added `tools=` to GRPOTrainer but requires
`transformers>=5.0`. vllm 0.17.0 caps `transformers<5`. The triangle
is unsatisfiable, so `train/grpo_sql.py` switched to a `<SQL>...</SQL>`
tag protocol — the model emits SQL inside tags, reward functions parse
and execute it. No tools= needed.

### Zero-variance reward batches
GRPO computes advantage as `(reward - mean) / std`. If all rollouts in
a batch score identically, std=0 and the loss spikes to ~150k for one
step. It's self-correcting — don't intervene unless it persists.

## Observed perf

GSM8K Qwen3 4B LoRA on H100, vllm rollouts:
- startup: ~60s for CUDA graph capture
- 4-13s/step at batch 4, 200 steps in ~15-20 minutes
- vllm rollout was visibly faster than HF generation in isolated tests
  (no number recorded — kept vllm regardless because the gap widens
  with model size).
