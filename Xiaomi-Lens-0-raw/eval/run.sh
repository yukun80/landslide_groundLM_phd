#!/bin/bash
# 【中文导读】阅读第 6 步：选择普通预测或 vLLM 预测，再把模型、数据、设备与输出位置交给 eval/run.py；本文件不计算评分。
# 从路径设置 → 设备设置 → 启动命令阅读。完整解释见仓库最外层 docs/xiaomi-lens-0/README.md 与 docs/xiaomi-lens-0/config/README.md。
# Copyright 2026 Xiaomi Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Evaluate Xiaomi-Lens-0 on LensBench with the Hugging Face or vLLM backend.
#
# MODEL and DATASET accept Hugging Face IDs or local paths (a checkpoint
# directory; a test.parquet file or its directory).
#
# Usage:
#   bash eval/run.sh hf   [extra run.py args...]   # one HF shard per GPU in DEVICES
#   bash eval/run.sh vllm [extra run.py args...]   # text and mask engines on two GPUs
#
# Examples:
#   DEVICES="cuda:0 cuda:1 cuda:2 cuda:3" bash eval/run.sh hf
#   TEXT_DEVICE=cuda:0 MASK_DEVICE=cuda:1 bash eval/run.sh vllm --limit 16
#   bash eval/run.sh hf --resume
# 步骤：遇到命令失败、未设置变量或管道错误就停止，避免继续使用不完整设置。
set -euo pipefail

# 步骤：第一个位置参数选择 hf 或 vllm，shift 后剩余参数继续交给评测程序。
BACKEND="${1:?usage: run.sh <hf|vllm> [extra run.py args...]}"
shift

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-SeerRay-Lab/Xiaomi-Lens-0}"
DATASET="${DATASET:-SeerRay-Lab/LensBench}"
# Unset MAX_NEW_TOKENS keeps run.py's default (DEFAULT_MAX_NEW_TOKENS).
COMMON_ARGS=()
if [[ -n "${MAX_NEW_TOKENS:-}" ]]; then
    COMMON_ARGS=(--max-new-tokens "${MAX_NEW_TOKENS}")
fi

# 步骤：按计算方式分支：hf 可按卡拆分样本；vllm 分别给文字和区域引擎指定设备。
case "${BACKEND}" in
    hf)
        # Space-separated devices; a single device runs without sharding.
        DEVICES="${DEVICES:-cuda:0}"
        OUTPUT_DIR="${OUTPUT_DIR:-outputs/lensbench_hf}"
        read -r -a DEVICE_LIST <<< "${DEVICES}"
        if (( ${#DEVICE_LIST[@]} > 1 )); then
            DEVICE_ARGS=(--devices "${DEVICE_LIST[@]}")
        else
            DEVICE_ARGS=(--device "${DEVICE_LIST[0]}")
        fi
        exec python "${ROOT}/eval/run.py" \
            --backend hf \
            --model-path "${MODEL}" \
            --dataset-path "${DATASET}" \
            --output-dir "${OUTPUT_DIR}" \
            ${COMMON_ARGS[@]+"${COMMON_ARGS[@]}"} \
            "${DEVICE_ARGS[@]}" \
            "$@"
        ;;
    vllm)
        OUTPUT_DIR="${OUTPUT_DIR:-outputs/lensbench_vllm}"
        exec python "${ROOT}/eval/run.py" \
            --backend vllm \
            --model-path "${MODEL}" \
            --dataset-path "${DATASET}" \
            --output-dir "${OUTPUT_DIR}" \
            ${COMMON_ARGS[@]+"${COMMON_ARGS[@]}"} \
            --vllm-text-device "${TEXT_DEVICE:-cuda:0}" \
            --vllm-mask-device "${MASK_DEVICE:-cuda:1}" \
            --vllm-max-model-len "${MAX_MODEL_LEN:-32768}" \
            --vllm-gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.8}" \
            "$@"
        ;;
    *)
        echo "Unknown backend: ${BACKEND} (expected hf or vllm)" >&2
        exit 2
        ;;
esac
