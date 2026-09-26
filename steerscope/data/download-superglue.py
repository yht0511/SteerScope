#!/usr/bin/env python3
"""Download the labeled SuperGLUE validation splits used by the evaluator."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi


DATASET_ID = "aps/super_glue"
DEFAULT_REVISION = "3de24cf8022e94f4ee4b9d55a6f539891524d646"
TASK_CONFIGS = {
    "boolq": "boolq",
    "cb": "cb",
    "copa": "copa",
    "multirc": "multirc",
    "record": "record",
    "rte": "rte",
    "wic": "wic",
    "wsc": "wsc.fixed",
}


def parse_args():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=directory / "superglue")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    paths = {task: output_dir / f"{task}_validation.parquet" for task in TASK_CONFIGS}
    existing = [path for path in [*paths.values(), manifest_path] if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output files already exist; pass --overwrite to replace them:\n  "
            + "\n  ".join(map(str, existing))
        )
    revision = HfApi().dataset_info(DATASET_ID, revision=args.revision).sha
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for task, config in TASK_CONFIGS.items():
        dataset = load_dataset(
            DATASET_ID,
            config,
            split="validation",
            revision=revision,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )
        frame = dataset.to_pandas()
        path = paths[task]
        temporary = path.with_suffix(".parquet.tmp")
        try:
            frame.to_parquet(temporary, index=False)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        files[task] = {
            "config": config,
            "rows": len(frame),
            "sha256": sha256_file(path),
        }
        print(f"Wrote {len(frame)} {task} validation rows to {path}")
    manifest = {
        "dataset": DATASET_ID,
        "split": "validation",
        "requested_revision": args.revision,
        "resolved_revision": revision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    try:
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
