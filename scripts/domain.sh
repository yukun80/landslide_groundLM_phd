#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export XIAOMI_LENS_DECODER_ATTN=sdpa
exec /home/yukun80/miniconda3/envs/qwen3vl/bin/python "$@"
