#!/usr/bin/env bash
# 使用已准备的一个多模态训练样本演示调用；效果评价以独立测试为准。
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/domain.sh infer.py --images datasets/landslide_lens_positive_v1/samples/multimodal_landslide_sic_003370/view_0.png datasets/landslide_lens_positive_v1/samples/multimodal_landslide_sic_003370/view_1.png datasets/landslide_lens_positive_v1/samples/multimodal_landslide_sic_003370/view_2.png --checkpoint "${1:-outputs/multiview/best.pt}" --output outputs/example-prediction
