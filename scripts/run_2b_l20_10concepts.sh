#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export EASYSTEER_VLLM="${EASYSTEER_VLLM:-$ROOT/.venv-easysteer/bin/vllm}"
exec "$ROOT/.venv/bin/python" -m steerscope.sweep.paper.scheduler \
    --config "steerscope/sweep/paper/scheduler_configs/2b_l20_10concepts.yaml" "$@"
