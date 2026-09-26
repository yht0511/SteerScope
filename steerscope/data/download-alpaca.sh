#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

output="alpaca_eval.json"
if [[ -s "$output" ]]; then
    echo "Already downloaded: $output"
    exit 0
fi

curl --fail --location --retry 5 --retry-all-errors \
    --continue-at - --output "${output}.part" \
    "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/2edc6fad8be6b14ea7230aabfd08188da6b8b814/alpaca_eval.json"
mv "${output}.part" "$output"
