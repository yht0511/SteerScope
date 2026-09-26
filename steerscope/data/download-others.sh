#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -d Feature-Descriptions/.git ]]; then
    echo "Updating existing Feature-Descriptions repository"
    git -C Feature-Descriptions pull --ff-only
elif [[ -e Feature-Descriptions ]]; then
    echo "Feature-Descriptions exists but is not a Git repository" >&2
    exit 1
else
    git clone https://github.com/yoavgur/Feature-Descriptions.git
fi
