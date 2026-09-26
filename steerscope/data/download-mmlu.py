#!/usr/bin/env python3
"""Download the MMLU train, dev, validation, and test splits as Parquet files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


DATASET_ID = "cais/mmlu"
DATASET_CONFIG = "all"
REQUIRED_COLUMNS = ("question", "subject", "choices", "answer")
SPLITS = {
    "auxiliary_train": {
        "filename": "MMLU_train.parquet",
        "expected_rows": 99_842,
    },
    "dev": {
        "filename": "MMLU_dev.parquet",
        "expected_rows": 285,
    },
    "validation": {
        "filename": "MMLU_val.parquet",
        "expected_rows": 1_531,
    },
    "test": {
        "filename": "MMLU_test.parquet",
        "expected_rows": 14_042,
    },
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "mmlu",
        help="Directory for MMLU_*.parquet files (default: %(default)s).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Hugging Face datasets cache directory. By default, use the "
            "datasets library's standard HF_HOME-based cache."
        ),
    )
    parser.add_argument(
        "--revision",
        default="c30699e8356da336a370243923dbaf21066bb9fe",
        help="Hugging Face dataset revision to resolve and download.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_revision(requested_revision: str) -> str:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(DATASET_ID, revision=requested_revision)
    if not info.sha:
        raise RuntimeError(
            f"Hugging Face did not return a commit SHA for {DATASET_ID}."
        )
    return info.sha


def validate_split(dataset, source_split: str, expected_rows: int) -> None:
    missing = sorted(set(REQUIRED_COLUMNS).difference(dataset.column_names))
    if missing:
        raise ValueError(f"MMLU {source_split} is missing columns: {missing}")
    if len(dataset) != expected_rows:
        raise ValueError(
            f"MMLU {source_split} has {len(dataset):,} rows; "
            f"expected {expected_rows:,}."
        )

    invalid_answers = [answer for answer in dataset["answer"] if answer not in range(4)]
    if invalid_answers:
        raise ValueError(
            f"MMLU {source_split} contains answers outside the range 0-3."
        )

    invalid_choices = sum(len(choices) != 4 for choices in dataset["choices"])
    if invalid_choices:
        raise ValueError(
            f"MMLU {source_split} contains {invalid_choices} rows without four choices."
        )


def ensure_outputs_available(output_dir: Path, overwrite: bool) -> None:
    output_paths = [
        output_dir / split_config["filename"]
        for split_config in SPLITS.values()
    ]
    output_paths.append(output_dir / "manifest.json")
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Output files already exist; pass --overwrite to replace them:\n"
            f"{formatted}"
        )


def write_parquet_atomic(dataset, output_path: Path) -> None:
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        dataset.to_parquet(temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_json_atomic(payload: dict, output_path: Path) -> None:
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.cache_dir is not None:
        args.cache_dir = args.cache_dir.expanduser().resolve()
    ensure_outputs_available(args.output_dir, args.overwrite)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Import after parsing so callers can control cache paths through the CLI.
    from datasets import load_dataset

    resolved_revision = resolve_revision(args.revision)
    manifest = {
        "dataset": DATASET_ID,
        "config": DATASET_CONFIG,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "splits": {},
    }

    for source_split, split_config in SPLITS.items():
        output_path = args.output_dir / split_config["filename"]
        print(f"Downloading {DATASET_ID}/{DATASET_CONFIG}:{source_split}")
        load_kwargs = {
            "split": source_split,
            "revision": resolved_revision,
        }
        if args.cache_dir is not None:
            load_kwargs["cache_dir"] = str(args.cache_dir)
        dataset = load_dataset(
            DATASET_ID,
            DATASET_CONFIG,
            **load_kwargs,
        )
        validate_split(dataset, source_split, split_config["expected_rows"])
        dataset = dataset.select_columns(list(REQUIRED_COLUMNS))
        write_parquet_atomic(dataset, output_path)

        manifest["splits"][source_split] = {
            "filename": output_path.name,
            "rows": len(dataset),
            "sha256": sha256_file(output_path),
        }
        print(f"Wrote {len(dataset):,} rows to {output_path}")

    manifest_path = args.output_dir / "manifest.json"
    write_json_atomic(manifest, manifest_path)
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
