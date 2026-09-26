#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# The paper evaluates layers 10 and 20 with the 16k GemmaScope release.
BASE_URL="https://neuronpedia-exports.s3.amazonaws.com/explanations-only/gemma-2-"

for layer in 10 20; do
    output="gemma-2-2b_${layer}-gemmascope-res-16k.json"
    if [[ -s "$output" ]]; then
        sha256sum --check <(awk -v name="$output" '$2 == name' concept_checksums.sha256)
        echo "Already downloaded: $output"
        continue
    fi
    url="${BASE_URL}${output#gemma-2-}"
    echo "Downloading ${url}..."
    curl --fail --location --retry 5 --retry-all-errors \
        --continue-at - --output "${output}.part" "$url"
    mv "${output}.part" "$output"
    sha256sum --check <(awk -v name="$output" '$2 == name' concept_checksums.sha256)
done

echo "Download completed."
