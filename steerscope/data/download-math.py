#!/usr/bin/env python3
"""Download the 5,000-example official MATH test split."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi


DATASET_ID = "EleutherAI/hendrycks_math"
DEFAULT_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
CONFIGS = (
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
)
EXPECTED_ROWS = 5000


def last_boxed(string):
    index = max(string.rfind("\\boxed"), string.rfind("\\fbox"))
    if index < 0:
        return None
    depth = 0
    opened = False
    for position in range(index, len(string)):
        if string[position] == "{":
            depth += 1
            opened = True
        elif string[position] == "}" and opened:
            depth -= 1
            if depth == 0:
                boxed = string[index:position + 1]
                return boxed[boxed.index("{") + 1:-1]
    return None


def parse_args():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=directory / "math")
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
    output_path = output_dir / "MATH_test.parquet"
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in (output_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Output files already exist; pass --overwrite to replace them:\n  " + "\n  ".join(map(str, existing)))
    revision = HfApi().dataset_info(DATASET_ID, revision=args.revision).sha
    rows = []
    counts = {}
    for config in CONFIGS:
        dataset = load_dataset(DATASET_ID, config, split="test", revision=revision, cache_dir=str(args.cache_dir) if args.cache_dir else None)
        counts[config] = len(dataset)
        for source_index, example in enumerate(dataset):
            solution = str(example["solution"])
            gold = last_boxed(solution)
            if gold is None:
                raise ValueError(f"MATH {config} test row {source_index} has no boxed answer.")
            level_text = str(example["level"])
            try:
                level = int(level_text.rsplit(" ", 1)[-1])
            except ValueError as error:
                raise ValueError(f"Invalid MATH level: {level_text!r}") from error
            rows.append({
                "source_config": config, "source_index": source_index,
                "problem": str(example["problem"]), "solution": solution,
                "gold_answer": gold, "level": level, "subject": str(example["type"]),
            })
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"MATH test has {len(rows)} rows; expected {EXPECTED_ROWS}.")
    for input_id, row in enumerate(rows):
        row["input_id"] = input_id
    frame = pd.DataFrame(rows)[["input_id", "source_config", "source_index", "problem", "solution", "gold_answer", "level", "subject"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".parquet.tmp")
    try:
        try:
            frame.to_parquet(temporary, index=False)
        except ImportError as error:
            raise RuntimeError("Writing MATH requires pyarrow. Activate the steerscope environment or install it with: python -m pip install pyarrow") from error
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "dataset": DATASET_ID, "split": "test", "requested_revision": args.revision,
        "resolved_revision": revision, "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(frame), "config_counts": counts, "output_sha256": sha256_file(output_path),
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    try:
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    print(f"Wrote {len(frame)} official MATH test examples to {output_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
