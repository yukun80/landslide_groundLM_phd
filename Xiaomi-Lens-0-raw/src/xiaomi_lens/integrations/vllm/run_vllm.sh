#!/bin/bash
# 【中文导读】阅读第 7 步：启动负责生成文字的服务，默认端口 8000；区域生成需要另外运行 run_vllm_mask.sh。
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

# Launch vLLM to serve Xiaomi-Lens-0's TEXT phase (bboxes + <|mask|> tokens).
#
# vLLM's --runner is exclusive (one process is generative OR pooling, never both),
# so this generative serve handles text only; the MASK phase runs in a SECOND,
# same-venv (transformers 4.57) vLLM instance via run_vllm_mask.sh (--runner
# pooling). Together they replace the old vLLM-text + HF-mask_service(5.4) hybrid
# with a single-ENV, all-vLLM deployment (two processes).
#
# Serves the checkpoint as a vanilla Qwen3.5-VL, WITHOUT mutating it. A prep step
# (prepare_served.py) materializes a symlink "served view" that fixes two
# checkpoint quirks so stock vLLM loads it:
#   1) Architecture — `--hf-overrides` swaps the custom
#      XiaomiLensQwen3_5ForConditionalGeneration arch for the base
#      `Qwen3_5ForConditionalGeneration` that vLLM registers.
#   2) Tokenizer + weights — prepare_served.py corrects the transformers-5.4
#      tokenizer_config (AutoTokenizer can't load `TokenizersBackend` / a list
#      `extra_special_tokens`) and strips the `mask_decoder.*` tensors that vLLM's
#      strict loader would reject (the mask head runs in run_vllm_mask.sh, not here).
# `<|mask|>` and its delimiters survive — they live in tokenizer.json / the
# resized embedding, not the mask head.
#
# Usage:
#   bash run_vllm.sh SeerRay-Lab/Xiaomi-Lens-0 [extra vllm args...]   # or a local checkpoint dir
# 步骤：遇到命令失败、未设置变量或管道错误就停止，避免继续使用不完整设置。
set -euo pipefail

# 步骤：读取模型位置；本地目录不存在时，下方代码会把它当在线模型名称下载。
CKPT="${1:?usage: run_vllm.sh <checkpoint_dir|hf_repo_id> [extra vllm args...]}"
shift || true

# Accept a Hugging Face repo ID (e.g. SeerRay-Lab/Xiaomi-Lens-0): download the
# snapshot, or reuse the cached one, and serve from the cache directory.
if [[ ! -d "${CKPT}" ]]; then
    CKPT="$(python3 -c 'import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))' "${CKPT}")"
fi

# 步骤：设置对外模型名称和端口；客户端请求使用此名称，不是直接传模型目录。
SERVED_NAME="${SERVED_NAME:-xiaomi-lens}"
PORT="${PORT:-8000}"
# By default vLLM reads max_pixels from the checkpoint's processor_config.json.
# Set MAX_PIXELS only for an intentional override; both serves must use the same
# value.
# 步骤：可选地覆盖图片像素上限；两个服务必须一致，否则区域位置可能无法与文字对应。
MAX_PIXELS="${MAX_PIXELS:-}"
MM_PROCESSOR_ARGS=()
if [[ -n "${MAX_PIXELS}" ]]; then
    MM_PROCESSOR_ARGS=(--mm-processor-kwargs "{\"max_pixels\": ${MAX_PIXELS}}")
    echo "[run_vllm] max_pixels override: ${MAX_PIXELS}"
else
    echo "[run_vllm] max_pixels: checkpoint processor config"
fi

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SRC_ROOT="$( cd "${HERE}/../../.." && pwd )"   # .../src, so `xiaomi_lens` is importable

# Build the served view once (idempotent; reuse across launches). Delete it to
# force a rebuild after a checkpoint change.
#
# The cache key includes a hash of the RESOLVED checkpoint path, not just its
# basename — different checkpoints often share a basename, so a basename-only key
# would make two different checkpoints collide on one served dir and silently
# serve the first one's weights.
# 步骤：把模型目录变成明确的真实路径，并用其摘要区分缓存，避免同名模型目录互相混淆。
CKPT_ABS="$( cd "${CKPT}" && pwd -P )"
CKPT_HASH="$(printf '%s' "${CKPT_ABS}" | cksum | cut -d' ' -f1)"
SERVED_DIR="${SERVED_DIR:-${TMPDIR:-/tmp}/xiaomi_lens_served_$(basename "${CKPT_ABS}")_${CKPT_HASH}}"
if [ ! -f "${SERVED_DIR}/model.safetensors.index.json" ]; then
    PYTHONPATH="${SRC_ROOT}" python3 -m xiaomi_lens.integrations.vllm.prepare_served \
        --checkpoint "${CKPT}" --out "${SERVED_DIR}"
fi
echo "[run_vllm] served view: ${SERVED_DIR}"

# 步骤：启动服务并由服务进程接管当前脚本；下方参数分别控制名称、端口和运行方式。
exec vllm serve "${SERVED_DIR}" \
    --served-model-name "${SERVED_NAME}" \
    --port "${PORT}" \
    --hf-overrides '{"architectures":["Qwen3_5ForConditionalGeneration"]}' \
    "${MM_PROCESSOR_ARGS[@]}" \
    --trust-remote-code \
    "$@"
