#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"
exec scripts/domain.sh -m xiaomi_lens.local_demo.app
