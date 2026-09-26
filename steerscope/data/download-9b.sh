#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

for layer in 20 31; do
    output="gemma-2-9b-it_${layer}-gemmascope-res-131k.json"
    if [[ -s "$output" ]]; then
        sha256sum --check <(awk -v name="$output" '$2 == name' concept_checksums.sha256)
        echo "Already downloaded: $output"
        continue
    fi
    curl --fail --location --retry 5 --retry-all-errors \
        --continue-at - --output "${output}.part" \
        "https://neuronpedia-exports.s3.amazonaws.com/explanations-only/${output}"
    mv "${output}.part" "$output"
    sha256sum --check <(awk -v name="$output" '$2 == name' concept_checksums.sha256)
done
