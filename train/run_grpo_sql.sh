#!/usr/bin/env bash
# GPU training launcher for SQL — run from repo root.
# Venv: /tmp/venv-gpu  |  All HF/dataset caches and CWD redirected to /tmp.
set -euo pipefail
cd /tmp

export UNSLOTH_VLLM_STANDBY=0
export HF_HOME=/tmp/hf-home
export HF_DATASETS_CACHE=/tmp/hf-datasets

/tmp/venv-gpu/bin/python -m train.grpo_sql "$@"
