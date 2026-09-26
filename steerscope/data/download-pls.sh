#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

download_file() {
    local url="$1"
    local output="$2"
    local partial="${output}.part"

    if [[ -s "$output" ]]; then
        echo "Already downloaded: $output"
        return
    fi

    curl --fail --location --retry 5 --retry-all-errors \
        --continue-at - --output "$partial" "$url"
    mv "$partial" "$output"
}

download_file \
    "https://neuronpedia-exports.s3.amazonaws.com/explanations-only/gemma-2-2b_20-gemmascope-res-65k.json" \
    "gemma-2-2b_20-gemmascope-res-65k.json"
download_file \
    "https://neuronpedia-exports.s3.amazonaws.com/explanations-only/gemma-2-9b-it_20-gemmascope-res-131k.json" \
    "gemma-2-9b-it_20-gemmascope-res-131k.json"

LLAMA_OUTPUT="llama3.1-8b_20-llamascope-res-131k.json"
LLAMA_ARCHIVE_URL="https://neuronpedia-exports.s3.amazonaws.com/explanations-only/llama3.1-8b_llamascope-res-131k.zip"

if [[ -s "$LLAMA_OUTPUT" ]]; then
    echo "Already downloaded: $LLAMA_OUTPUT"
else
    temporary_dir="$(mktemp -d)"
    trap 'rm -rf -- "$temporary_dir"' EXIT
    archive="$temporary_dir/llamascope-res-131k.zip"

    curl --fail --location --retry 5 --retry-all-errors \
        --output "$archive" "$LLAMA_ARCHIVE_URL"
    member="$(unzip -Z1 "$archive" | awk -F/ -v target="$LLAMA_OUTPUT" '$NF == target { print; exit }')"
    if [[ -z "$member" ]]; then
        echo "Archive does not contain $LLAMA_OUTPUT" >&2
        exit 1
    fi
    unzip -j "$archive" "$member" -d "$temporary_dir/extracted"
    mv "$temporary_dir/extracted/$LLAMA_OUTPUT" "$LLAMA_OUTPUT"
    echo "Extracted $LLAMA_OUTPUT"
fi
