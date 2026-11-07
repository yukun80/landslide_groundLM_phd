#!/bin/bash
# 【中文导读】阅读第 5 步：第一阶段只学习区域解码器和区域符号表示，关闭文字损失；这里是启动配方，实际训练在 train.py。
# 从路径设置 → 设备设置 → 启动命令阅读。完整解释见仓库最外层 docs/xiaomi-lens-0/README.md 与 docs/xiaomi-lens-0/config/README.md。
# Stage 1: train the PixelMaskDecoder and the <|mask|> embedding row on
# mask loss only. The ViT, LLM, lm_head and all other embedding rows stay frozen.
#
# Usage:
#   MODEL_PATH=/path/to/prepared/Qwen3.5-4B DATA_CONFIG=/path/to/stage1.yaml \
#       bash scripts/train_stage1.sh [extra train.py args...]
# 步骤：遇到命令失败、未设置变量或管道错误就停止，避免继续使用不完整设置。
set -euo pipefail

# 步骤：定位代码根目录并进入，后面的相对路径由此解释。
REPO_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "${REPO_HOME}"

# ── Paths ──
# MODEL_PATH is a base Qwen3.5 checkpoint prepared by tools/prepare_base_model.py.
: "${MODEL_PATH:?Set MODEL_PATH to a base Qwen3.5 checkpoint prepared by tools/prepare_base_model.py}"
: "${DATA_CONFIG:?Set DATA_CONFIG to a YAML data manifest (see examples/train/README.md)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_HOME}/checkpoints/xiaomi-lens-stage1}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_HOME}/src/xiaomi_lens/train/configs/zero1.json}"
mkdir -p "${OUTPUT_DIR}"

# ── Distributed ──
# Single node by default with all visible GPUs. For multi-node, set NNODES,
# NODE_RANK and MASTER_ADDR on each node (and optionally NPROC_PER_NODE,
# MASTER_PORT).
# 步骤：设置每台机器的进程数和机器编号；通常一张 GPU 对应一个训练进程。
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l | tr -d ' ')}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_PORT="${MASTER_PORT:-29500}"
MASTER_ADDR="${MASTER_ADDR:-}"

if [[ "${NNODES}" -gt 1 && -z "${MASTER_ADDR}" ]]; then
    echo "MASTER_ADDR must be set when NNODES > 1" >&2
    exit 1
fi
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

# 步骤：将多进程启动选项保存为数组，保证带空格的参数不会被错误拆开。
LAUNCHER_ARGS=(
    --nproc_per_node "${NPROC_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

# ── Launch ──
# <|mask|>, <mask> and </mask> start from the same vocabulary-mean embedding.
# Training the <|mask|> row here (--unfreeze_new_tokens) breaks that symmetry
# before Stage 2. Its gradient flows back through the frozen LLM, so keep
# --gradient_checkpointing on. On tied-embedding models lm_head is
# embed_tokens, so keep --unfreeze_language_model_head false.
# --lm_loss_weight -1 disables the LM loss for this stage.
# 步骤：开始训练：下方连续多行实际属于同一条命令，反斜杠表示续行。
# 不要在这些续行中间插入普通注释，否则可能截断命令。各选项详解见根目录配置说明。
python -m torch.distributed.run "${LAUNCHER_ARGS[@]}" train.py \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path "${MODEL_PATH}" \
    --attn_implementation flash_attention_2 \
    --bf16 \
    --data_config "${DATA_CONFIG}" \
    --max_pixels 4194304 \
    --min_pixels 65536 \
    --model_max_length 8192 \
    --group_by_modality_length true \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers true \
    --dataloader_prefetch_factor 4 \
    --dataloader_drop_last true \
    --unfreeze_all_parameters false \
    --unfreeze_vision_tower false \
    --unfreeze_vision_projection false \
    --unfreeze_language_model_backbone false \
    --unfreeze_language_model_head false \
    --unfreeze_embed_tokens false \
    --unfreeze_new_tokens true \
    --unfreeze_mask_decoder true \
    --lm_loss_weight -1.0 \
    --mask_loss_weight 1.0 \
    --mask_loss_chunk_size 8 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --warmup_steps 0.03 \
    --lr_scheduler_type cosine \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --output_dir "${OUTPUT_DIR}" \
    --auto_resume true \
    --save_steps 2000 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --report_to tensorboard \
    --run_name "xiaomi-lens-stage1" \
    "$@"
