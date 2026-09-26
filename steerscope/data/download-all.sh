#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# hf-mirror currently serves repository metadata but fails file resolution for
# several dataset repositories used here.  The Hugging Face origin works via
# the configured proxy, while callers can still override it explicitly.
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

run_with_manifest() {
    local manifest="$1"
    local script="$2"
    if [[ -s "$manifest" ]]; then
        echo "Already complete: $manifest"
    else
        "$PYTHON_BIN" "$script" --overwrite
    fi
}

bash download-alpaca.sh
run_with_manifest mmlu/manifest.json download-mmlu.py
run_with_manifest superglue/manifest.json download-superglue.py
run_with_manifest math/manifest.json download-math.py
run_with_manifest ifeval/manifest.json download-ifeval.py
run_with_manifest truthfulqa/manifest.json download-truthfulqa.py
run_with_manifest bbq/manifest.json download-bbq.py

if [[ -s jailbreakbench/JailBreakBench_Harmful.parquet && -s jailbreakbench/JailBreakBench_Benign.parquet ]]; then
    echo "Already complete: jailbreakbench"
else
    "$PYTHON_BIN" download-jailbreakbench.py --overwrite
fi

if [[ -s seed_sentences/dataset_dict.json && -s seed_instructions/dataset_dict.json ]]; then
    echo "Already complete: seed datasets"
else
    "$PYTHON_BIN" download-seed-sentences.py
fi

run_with_manifest x_alpaca_eval/manifest.json download-x-alpaca-eval.py
bash download-2b.sh
bash download-9b.sh
