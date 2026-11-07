#!/bin/bash
# 【中文导读】阅读第 7 步：启动负责生成像素区域的服务，默认端口 8001；与文字服务采用相同图片处理大小。
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

# Launch a vLLM instance that serves Xiaomi-Lens-0's MASK phase via /pooling.
#
# vLLM's --runner is exclusive: one process is EITHER generative OR pooling, never
# both. So the unified-env deployment is TWO same-venv (transformers 4.57) vLLM
# instances — this one (mask, --runner pooling) plus a generative one (text, see
# run_vllm.sh). Both load the same weights; the env-split (4.57 vs the old HF
# mask_service's 5.4) is gone because PixelMaskDecoder is pure torch and runs here.
#
# The mask head runs in-engine via a custom pooling model + io_processor plugin:
#   - XiaomiLensForVLLM (vllm_model.py) subclasses vLLM's
#     Qwen3_5ForConditionalGeneration, taps the ViT pre-merge features + the
#     <|mask|> hidden state, and runs PixelMaskDecoder in a "plugin" Pooler.
#   - XiaomiLensMaskIOProcessor (io_processor.py) turns a mask request into that
#     pooling forward and serializes the dense mask (PNG + overlay) back out.
# Both are registered via entry points in the package pyproject.toml. Installing
# the package is preferred for normal serving environments. For immutable
# source-mounted runtimes, this launcher supplies equivalent temporary entry-point
# metadata without modifying the shared venv.
#
# prepare_served.py --tokenizer-only materializes a pure-symlink served view that
# only fixes the transformers-5.4 tokenizer_config; the arch name
# (XiaomiLensQwen3_5ForConditionalGeneration, which the plugin registers) and ALL
# weights — including mask_decoder.* — are preserved. load_weights loads the
# mask head. No offline shard rewrite, no arch override.
#
# Disable chunked prefill and prefix caching. The pooler needs the <|mask|>
# token's hidden state, but chunking may place that token in another pooler call.
# Also disable multimodal preprocessing caching: a cache hit bypasses the ViT
# forward hook, preventing the pooler from receiving pre-merge patch features.
#
# Usage:
#   PORT=8001 bash run_vllm_mask.sh SeerRay-Lab/Xiaomi-Lens-0 [extra vllm args...]   # or a local checkpoint dir
# 步骤：遇到命令失败、未设置变量或管道错误就停止，避免继续使用不完整设置。
set -euo pipefail

# 步骤：读取模型位置；本地目录不存在时，下方代码会把它当在线模型名称下载。
CKPT="${1:?usage: run_vllm_mask.sh <checkpoint_dir|hf_repo_id> [extra vllm args...]}"
shift || true

# Accept a Hugging Face repo ID (e.g. SeerRay-Lab/Xiaomi-Lens-0): download the
# snapshot, or reuse the cached one, and serve from the cache directory.
if [[ ! -d "${CKPT}" ]]; then
    CKPT="$(python3 -c 'import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))' "${CKPT}")"
fi

# 步骤：设置对外模型名称和端口；客户端请求使用此名称，不是直接传模型目录。
SERVED_NAME="${SERVED_NAME:-xiaomi-lens}"
PORT="${PORT:-8001}"          # default off :8000 so it can coexist with the text serve
# By default vLLM reads max_pixels from the checkpoint's processor_config.json.
# Set MAX_PIXELS only for an intentional override; both serves must use the same
# value.
# 步骤：可选地覆盖图片像素上限；两个服务必须一致，否则区域位置可能无法与文字对应。
MAX_PIXELS="${MAX_PIXELS:-}"
MM_PROCESSOR_ARGS=()
if [[ -n "${MAX_PIXELS}" ]]; then
    MM_PROCESSOR_ARGS=(--mm-processor-kwargs "{\"max_pixels\": ${MAX_PIXELS}}")
    echo "[run_vllm_mask] max_pixels override: ${MAX_PIXELS}"
else
    echo "[run_vllm_mask] max_pixels: checkpoint processor config"
fi

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SRC_ROOT="$( cd "${HERE}/../../.." && pwd )"   # .../src, so `xiaomi_lens` is importable

# vLLM discovers out-of-tree models and IO processors from distribution entry
# points in every spawned process. When the source tree is mounted into a shared,
# read-only venv instead of installed, make the source importable and add a tiny
# per-launch metadata overlay only when the installed environment lacks our entry
# points. EngineCore children inherit PYTHONPATH and discover the same overlay.
# 步骤：让当前进程和后续子进程都能找到本项目模块。
export PYTHONPATH="${SRC_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# 步骤：检查框架能否发现自定义模型和输入输出处理器；缺失时在临时目录补登记信息。
if ! python3 - <<'PY'
from importlib.metadata import entry_points

required = {
    "vllm.general_plugins": (
        "xiaomi_lens_register",
        "xiaomi_lens.integrations.vllm.plugin:register",
    ),
    "vllm.io_processor_plugins": (
        "xiaomi_lens_mask",
        "xiaomi_lens.integrations.vllm.io_processor:make_io_processor",
    ),
}
for group, expected in required.items():
    available = {(item.name, item.value) for item in entry_points(group=group)}
    if expected not in available:
        raise SystemExit(1)
PY
then
    PLUGIN_META_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/xiaomi-lens-vllm-plugin.XXXXXX")"
    DIST_INFO="${PLUGIN_META_ROOT}/xiaomi_lens-0.1.0.dist-info"
    mkdir -p "${DIST_INFO}"
    cat > "${DIST_INFO}/METADATA" <<'EOF'
Metadata-Version: 2.1
Name: xiaomi-lens
Version: 0.1.0
EOF
    cat > "${DIST_INFO}/entry_points.txt" <<'EOF'
[vllm.general_plugins]
xiaomi_lens_register = xiaomi_lens.integrations.vllm.plugin:register

[vllm.io_processor_plugins]
xiaomi_lens_mask = xiaomi_lens.integrations.vllm.io_processor:make_io_processor
EOF
    export PYTHONPATH="${PLUGIN_META_ROOT}:${PYTHONPATH}"
    echo "[run_vllm_mask] bootstrapped Xiaomi-Lens-0 vLLM plugin metadata: ${PLUGIN_META_ROOT}"
fi

# Pure-symlink served view (tokenizer-only). Cache key = hash of the RESOLVED
# checkpoint path; `pool_` prefix keeps it distinct from the generative path's
# `xiaomi_lens_served_*` dir (rewritten shards, mask stripped).
# 步骤：把模型目录变成明确的真实路径，并用其摘要区分缓存，避免同名模型目录互相混淆。
CKPT_ABS="$( cd "${CKPT}" && pwd -P )"
CKPT_HASH="$(printf '%s' "${CKPT_ABS}" | cksum | cut -d' ' -f1)"
SERVED_DIR="${SERVED_DIR:-${TMPDIR:-/tmp}/xiaomi_lens_pool_$(basename "${CKPT_ABS}")_${CKPT_HASH}}"
if [ ! -e "${SERVED_DIR}/tokenizer_config.json" ]; then
    PYTHONPATH="${SRC_ROOT}" python3 -m xiaomi_lens.integrations.vllm.prepare_served \
        --checkpoint "${CKPT}" --out "${SERVED_DIR}" --tokenizer-only
fi
echo "[run_vllm_mask] served view (tokenizer-only symlinks): ${SERVED_DIR}"

# 步骤：启动服务并由服务进程接管当前脚本；下方参数分别控制名称、端口和运行方式。
exec vllm serve "${SERVED_DIR}" \
    --runner pooling \
    --io-processor-plugin xiaomi_lens_mask \
    --served-model-name "${SERVED_NAME}" \
    --port "${PORT}" \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    "${MM_PROCESSOR_ARGS[@]}" \
    --trust-remote-code \
    "$@"
