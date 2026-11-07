#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-multiview}"
case "$mode" in
 multiview) scripts/domain.sh -m xiaomi_lens.train.domain_train --output outputs/multiview ;;
 rgb) scripts/domain.sh -m xiaomi_lens.train.domain_train --rgb-only --output outputs/rgb ;;
 *) echo '请选择 multiview 或 rgb' >&2; exit 2 ;;
esac
