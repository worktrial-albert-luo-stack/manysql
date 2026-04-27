#!/usr/bin/env bash
set -e
cd /tmp
/tmp/venv-gpu/bin/python /workspace/manysql/train/grpo_gsm8k.py "$@"
