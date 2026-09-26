#!/usr/bin/env python3
"""Download the harmful and benign JailbreakBench behavior datasets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DATASET_ID = "JailbreakBench/JBB-Behaviors"
DATASET_CONFIG = "behaviors"
REQUIRED_COLUMNS = ("Behavior", "Goal", "Target", "Category", "Source")
SPLITS = {
    "harmful": {
        "filename": "JailBreakBench_Harmful.parquet",
        "expected_rows": 100,
    },
    "benign": {
        "filename": "JailBreakBench_Benign.parquet",
        "expected_rows": 100,
    },
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "jailbreakbench",
        help="Directory for JailBreakBench_*.parquet files (default: %(default)s).",
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
        default="886acc352a31533ffbcf4ef22c744658688086fc",
        help="Hugging Face dataset revision to resolve and download.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files.",
    )
    return parser.parse_args()


def resolve_revision(requested_revision: str) -> str:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(DATASET_ID, revision=requested_revision)
    if not info.sha:
        raise RuntimeError(
            f"Hugging Face did not return a commit SHA for {DATASET_ID}."
        )
    return info.sha


def validate_split(dataset, split: str, expected_rows: int) -> None:
    missing = sorted(set(REQUIRED_COLUMNS).difference(dataset.column_names))
    if missing:
        raise ValueError(f"JailbreakBench {split} is missing columns: {missing}")
    if len(dataset) != expected_rows:
        raise ValueError(
            f"JailbreakBench {split} has {len(dataset):,} rows; "
            f"expected {expected_rows:,}."
        )

    behaviors = [str(value).strip() for value in dataset["Behavior"]]
    goals = [str(value).strip() for value in dataset["Goal"]]
    if any(not value for value in behaviors):
        raise ValueError(f"JailbreakBench {split} contains empty behaviors.")
    if len(set(behaviors)) != len(behaviors):
        raise ValueError(f"JailbreakBench {split} contains duplicate behaviors.")
    if any(not value for value in goals):
        raise ValueError(f"JailbreakBench {split} contains empty goals.")


def ensure_outputs_available(output_dir: Path, overwrite: bool) -> None:
    output_paths = [
        output_dir / split_config["filename"]
        for split_config in SPLITS.values()
    ]
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


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.cache_dir is not None:
        args.cache_dir = args.cache_dir.expanduser().resolve()
    ensure_outputs_available(args.output_dir, args.overwrite)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    resolved_revision = resolve_revision(args.revision)

    for split, split_config in SPLITS.items():
        print(f"Downloading {DATASET_ID}/{DATASET_CONFIG}:{split}")
        load_kwargs = {
            "split": split,
            "revision": resolved_revision,
        }
        if args.cache_dir is not None:
            load_kwargs["cache_dir"] = str(args.cache_dir)
        dataset = load_dataset(DATASET_ID, DATASET_CONFIG, **load_kwargs)
        validate_split(dataset, split, split_config["expected_rows"])
        dataset = dataset.select_columns(list(REQUIRED_COLUMNS))

        output_path = args.output_dir / split_config["filename"]
        write_parquet_atomic(dataset, output_path)
        print(f"Wrote {len(dataset):,} rows to {output_path}")


if __name__ == "__main__":
    main()
