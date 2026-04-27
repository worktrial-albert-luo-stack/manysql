#!/usr/bin/env bash
# Multi-GPU DDP launcher for SQL GRPO -- run from anywhere.
# Venv: /tmp/venv-gpu  |  All HF/dataset caches and CWD redirected to /tmp.
#
# Layout (split mode, trl 0.22 pattern): N total GPUs ->
#   - training on GPUs 0..N-2 via accelerate launch --num_processes N-1
#   - vLLM on GPU N-1 (rank 0 owns the engine, broadcasts to other ranks)
#
# Override GPU layout via env:
#   NGPU=8 bash train/run_grpo_sql_ddp.sh --dialects ...   # explicit total
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4 bash train/...          # mask devices first
#
# UNSLOTH_VLLM_STANDBY is intentionally unset: this script doesn't use
# Unsloth (DDP requires plain HF). HF caches still go to /tmp because
# /workspace has a per-user NFS quota (see train/GPU_SETUP.md).
set -euo pipefail
cd /tmp

export HF_HOME=/tmp/hf-home
export HF_DATASETS_CACHE=/tmp/hf-datasets

PY="/tmp/venv-gpu/bin/python"

# Detect GPU count after CUDA_VISIBLE_DEVICES masking.
NGPU="${NGPU:-$("$PY" -c 'import torch; print(torch.cuda.device_count())')}"
if [ "$NGPU" -lt 2 ]; then
    echo "[run_grpo_sql_ddp] need >= 2 GPUs for split-mode DDP (got $NGPU)." >&2
    echo "[run_grpo_sql_ddp] for single-GPU sanity, run grpo_sql_distributed.py" >&2
    echo "[run_grpo_sql_ddp] directly: $PY train/grpo_sql_distributed.py \\" >&2
    echo "[run_grpo_sql_ddp]   --vllm-device cuda:0 --vllm-gpu-memory-utilization 0.4 ..." >&2
    exit 1
fi

TRAIN_PROCS=$((NGPU - 1))
VLLM_DEV="cuda:$((NGPU - 1))"

echo "[run_grpo_sql_ddp] NGPU=$NGPU train_procs=$TRAIN_PROCS vllm_device=$VLLM_DEV"

exec "$PY" -m accelerate.commands.launch \
    --num_processes "$TRAIN_PROCS" \
    -m train.grpo_sql_distributed \
    --vllm-device "$VLLM_DEV" \
    "$@"
