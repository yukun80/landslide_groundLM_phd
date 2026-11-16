#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec scripts/domain.sh tools/run_stable_finetuning.py --output "${1:-outputs/finetune-20261115}"
