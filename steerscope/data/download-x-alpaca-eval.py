#!/usr/bin/env python3
"""Download the aligned multilingual X-AlpacaEval benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DATASET_ID = "zhihz0535/X-AlpacaEval"
DEFAULT_REVISION = "03d79e702ac75fc1c7a1fe773b1c168653193c69"
LANGUAGES = ("english", "chinese", "korean", "italian", "spanish")
LANGUAGE_SUFFIXES = {
    "english": "en",
    "chinese": "cn",
    "korean": "ko",
    "italian": "it",
    "spanish": "es",
}
REQUIRED_COLUMNS = ("id", "dataset", "instruction")
EXPECTED_ROWS_PER_LANGUAGE = 805


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "x_alpaca_eval",
        help="Directory for XAlpacaEval.parquet and manifest.json.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face datasets cache directory.",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Pinned Hugging Face dataset revision to download.",
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


def ensure_outputs_available(output_dir: Path, overwrite: bool) -> None:
    outputs = (
        output_dir / "XAlpacaEval.parquet",
        output_dir / "manifest.json",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Output files already exist; pass --overwrite to replace them:\n"
            f"{formatted}"
        )


def validate_language(frame: pd.DataFrame, language: str) -> None:
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"X-AlpacaEval {language} is missing columns: {missing}")
    if len(frame) != EXPECTED_ROWS_PER_LANGUAGE:
        raise ValueError(
            f"X-AlpacaEval {language} has {len(frame):,} rows; expected "
            f"{EXPECTED_ROWS_PER_LANGUAGE:,}."
        )
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError(f"X-AlpacaEval {language} contains invalid IDs.")
    if frame["instruction"].isna().any() or not frame[
        "instruction"
    ].astype(str).str.strip().all():
        raise ValueError(f"X-AlpacaEval {language} contains empty instructions.")


def validate_alignment(frames: dict[str, pd.DataFrame]) -> None:
    reference = frames["english"].set_index("id")["dataset"].sort_index()
    for language, frame in frames.items():
        metadata = frame.set_index("id")["dataset"].sort_index()
        if not metadata.index.equals(reference.index):
            raise ValueError(
                f"X-AlpacaEval {language} IDs do not align with English."
            )
        if not metadata.equals(reference):
            raise ValueError(
                f"X-AlpacaEval {language} source metadata does not align "
                "with English."
            )


def write_parquet_atomic(frame: pd.DataFrame, output_path: Path) -> None:
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        try:
            frame.to_parquet(temporary_path, index=False)
        except ImportError as error:
            raise RuntimeError(
                "Writing X-AlpacaEval requires pyarrow. Activate the steerscope "
                "environment or install it with: python -m pip install pyarrow"
            ) from error
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
    output_dir = args.output_dir.expanduser().resolve()
    cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir is not None
        else None
    )
    ensure_outputs_available(output_dir, args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    resolved_revision = resolve_revision(args.revision)
    load_kwargs = {"revision": resolved_revision}
    if cache_dir is not None:
        load_kwargs["cache_dir"] = str(cache_dir)
    corpus = load_dataset(
        DATASET_ID,
        data_files={
            language: f"{language}.json" for language in LANGUAGES
        },
        **load_kwargs,
    )
    missing_languages = sorted(set(LANGUAGES).difference(corpus.keys()))
    if missing_languages:
        raise ValueError(
            f"X-AlpacaEval is missing language splits: {missing_languages}"
        )

    frames = {}
    for language in LANGUAGES:
        frame = corpus[language].to_pandas()[list(REQUIRED_COLUMNS)].copy()
        validate_language(frame, language)
        frames[language] = frame
    validate_alignment(frames)

    combined = frames["english"][["id", "dataset"]].sort_values(
        "id", kind="stable"
    ).reset_index(drop=True)
    for language in LANGUAGES:
        instruction_column = f"instruction_{LANGUAGE_SUFFIXES[language]}"
        instructions = frames[language].set_index("id")["instruction"]
        combined[instruction_column] = combined["id"].map(instructions)

    output_path = output_dir / "XAlpacaEval.parquet"
    write_parquet_atomic(combined, output_path)
    manifest = {
        "dataset": DATASET_ID,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "license": "CC-BY-NC-4.0",
        "languages": list(LANGUAGES),
        "rows_per_language": {
            language: len(frames[language]) for language in LANGUAGES
        },
        "rows": len(combined),
        "columns": combined.columns.tolist(),
        "output": output_path.name,
        "output_sha256": sha256_file(output_path),
    }
    manifest_path = output_dir / "manifest.json"
    write_json_atomic(manifest, manifest_path)
    print(
        f"Wrote {len(combined):,} aligned X-AlpacaEval rows to {output_path}"
    )
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
