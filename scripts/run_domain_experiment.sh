#!/usr/bin/env bash
# 前置短训练通过后执行完整实验；任一步失败立即停止并保留现场。
set -euo pipefail
cd "$(dirname "$0")/.."
run() {
  local stage="$1"
  shift
  echo "$stage" > outputs/current-stage.txt
  scripts/domain.sh "$@" > "outputs/${stage}.log" 2>&1
}
run reload-check tools/check_domain_reload.py
run baseline-val eval/domain.py --split val --output outputs/baseline-val
run multiview-training -m xiaomi_lens.train.domain_train --output outputs/multiview
run rgb-training -m xiaomi_lens.train.domain_train --rgb-only --output outputs/rgb
run auxiliary-check tools/domain_auxiliary_check.py
run lock-models tools/lock_domain_models.py
run baseline-test eval/domain.py --split test --output outputs/baseline-test
run multiview-test eval/domain.py --checkpoint outputs/multiview/best.pt --split test --output outputs/multiview-test
run rgb-test eval/domain.py --checkpoint outputs/rgb/best.pt --rgb-only --split test --output outputs/rgb-test
run comparison tools/report_domain_results.py
echo 'finished-numerical-evaluation-awaiting-visual-review' > outputs/current-stage.txt
