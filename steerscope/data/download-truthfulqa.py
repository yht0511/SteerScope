#!/usr/bin/env python3
"""Download the official 2025 binary-choice TruthfulQA dataset."""

import argparse
import csv
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd


REPOSITORY = "sylinrl/TruthfulQA"
UPSTREAM_PATH = "TruthfulQA.csv"
DEFAULT_REVISION = "d71c110897f5d31c5d7f309e7bc316c152f6f031"
EXPECTED_ROWS = 790


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=script_dir / "truthfulqa")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fetch(url):
    response = httpx.get(url, follow_redirects=True, timeout=60.0)
    response.raise_for_status()
    return response.content


def resolve_revision(revision):
    payload = json.loads(
        fetch(f"https://api.github.com/repos/{REPOSITORY}/commits/{revision}")
    )
    if not payload.get("sha"):
        raise RuntimeError(f"GitHub did not resolve {REPOSITORY}@{revision}.")
    return payload["sha"]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_path = output_dir / "TruthfulQA_binary.parquet"
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in (output_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output files already exist; pass --overwrite to replace them:\n  "
            + "\n  ".join(map(str, existing))
        )
    revision = resolve_revision(args.revision)
    payload = fetch(
        f"https://raw.githubusercontent.com/{REPOSITORY}/{revision}/{UPSTREAM_PATH}"
    )
    source_rows = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    required = {
        "Type", "Category", "Question", "Best Answer",
        "Best Incorrect Answer", "Source",
    }
    if source_rows.fieldnames is None:
        raise ValueError("TruthfulQA CSV has no header.")
    missing = sorted(required.difference(source_rows.fieldnames))
    if missing:
        raise ValueError(f"TruthfulQA CSV is missing columns: {missing}")
    rows = []
    seen_questions = set()
    for input_id, row in enumerate(source_rows):
        question = str(row["Question"]).strip()
        correct = str(row["Best Answer"]).strip()
        incorrect = str(row["Best Incorrect Answer"]).strip()
        if not question or not correct or not incorrect:
            raise ValueError(f"TruthfulQA row {input_id} has an empty binary field.")
        if question in seen_questions:
            raise ValueError(f"Duplicate TruthfulQA question: {question!r}")
        seen_questions.add(question)
        if correct == incorrect:
            raise ValueError(f"TruthfulQA row {input_id} has identical answers.")
        rows.append({
            "input_id": input_id,
            "type": str(row["Type"]).strip(),
            "category": str(row["Category"]).strip(),
            "question": question,
            "best_answer": correct,
            "best_incorrect_answer": incorrect,
            "source": str(row["Source"]).strip(),
        })
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"TruthfulQA has {len(rows)} rows; expected {EXPECTED_ROWS}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".parquet.tmp")
    try:
        try:
            pd.DataFrame(rows).to_parquet(temporary, index=False)
        except ImportError as error:
            raise RuntimeError(
                "Writing TruthfulQA requires the project's pyarrow dependency. "
                "Activate the steerscope environment or install it with: "
                "python -m pip install pyarrow"
            ) from error
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "dataset": REPOSITORY,
        "task": "binary-choice",
        "upstream_path": UPSTREAM_PATH,
        "requested_revision": args.revision,
        "resolved_revision": revision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "output_sha256": sha256_file(output_path),
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    try:
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    print(f"Wrote {len(rows)} TruthfulQA binary examples to {output_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
