#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
if ! command -v uv >/dev/null; then
    python3 -m pip install --user uv
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install the benchmark environment
uv sync --frozen --python 3.12

# IFEval sentence tokenization resources.
"$ROOT/.venv/bin/python" -m nltk.downloader --exit-on-error punkt punkt_tab

# Install the EasySteer environment.
if [ -e "$ROOT/.git" ]; then
    git submodule sync --recursive
    git submodule update --init --recursive --depth 1 reference/EasySteer
else
    EASYSTEER_COMMIT=fff1f61837e34af0f70351d46756187a69530f7e
    if [ ! -e reference/EasySteer/.git ]; then
        git clone --no-checkout --depth 1 \
            https://github.com/ZJU-REAL/EasySteer.git reference/EasySteer
    fi
    git -C reference/EasySteer fetch --depth 1 origin "$EASYSTEER_COMMIT"
    git -C reference/EasySteer checkout --detach "$EASYSTEER_COMMIT"
    git -C reference/EasySteer submodule sync --recursive
    git -C reference/EasySteer submodule update --init --recursive --depth 1
fi
uv venv --python 3.10 .venv-easysteer
export VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_PRECOMPILED_WHEEL_COMMIT:-95c0f928cdeeaa21c4906e73cee6a156e1b3b995}"
VLLM_USE_PRECOMPILED=1 uv pip install \
    --python .venv-easysteer/bin/python \
    --requirement easysteer-requirements.txt
.venv-easysteer/bin/python -c 'import easysteer, vllm'
.venv-easysteer/bin/vllm --help >/dev/null

# Download evaluator datasets and the four released GemmaScope concept files.
PYTHON_BIN="$ROOT/.venv/bin/python" bash steerscope/data/download-all.sh
