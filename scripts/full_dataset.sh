#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
# 可选加速包只装在本项目输出目录，不升级或复制原Conda环境。
if [[ -d "$PWD/outputs/full-dataset-runtime/fla" ]]; then
  export PYTHONPATH="$PWD/outputs/full-dataset-runtime:$PYTHONPATH"
fi
export XIAOMI_LENS_DECODER_ATTN=sdpa
exec /home/yukun80/miniconda3/envs/qwen3vl/bin/python "$@"
