#!/usr/bin/env python3
"""Download the official BBQ benchmark and write one validated Parquet file."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


REPOSITORY = "nyu-mll/BBQ"
CATEGORIES = (
    "Age", "Disability_status", "Gender_identity", "Nationality",
    "Physical_appearance", "Race_ethnicity", "Race_x_gender",
    "Race_x_SES", "Religion", "SES", "Sexual_orientation",
)
OUTPUT_COLUMNS = (
    "example_id", "question_index", "question_polarity", "context_condition",
    "category", "context", "question", "ans0", "ans1", "ans2", "label",
    "target_loc", "unknown_loc",
)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=script_dir / "bbq")
    parser.add_argument("--revision", default="bea11bd97d79217245b5871acd247b9d6eb24598")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fetch(url):
    with urlopen(url) as response:
        return response.read()


def resolve_revision(revision):
    payload = json.loads(fetch(
        f"https://api.github.com/repos/{REPOSITORY}/commits/{revision}"
    ))
    sha = payload.get("sha")
    if not sha:
        raise RuntimeError(f"GitHub did not resolve {REPOSITORY}@{revision}.")
    return sha


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def answer_kind(answer_info, index):
    value = answer_info[f"ans{index}"]
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"Invalid BBQ answer_info entry: {value!r}")
    return str(value[1])


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_path = output_dir / "BBQ.parquet"
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in (output_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output files already exist; pass --overwrite to replace them:\n  "
            + "\n  ".join(map(str, existing))
        )
    revision = resolve_revision(args.revision)
    base = f"https://raw.githubusercontent.com/{REPOSITORY}/{revision}"
    metadata_bytes = fetch(f"{base}/analysis_scripts/additional_metadata.csv")
    metadata_rows = csv.DictReader(io.StringIO(metadata_bytes.decode("utf-8-sig")))
    target_locations = {
        (row["category"], str(row["question_index"]), int(row["example_id"])):
            int(row["target_loc"])
        for row in metadata_rows if row.get("target_loc") not in (None, "", "NA")
    }
    rows = []
    source_hashes = {}
    for category in CATEGORIES:
        payload = fetch(f"{base}/data/{category}.jsonl")
        source_hashes[f"{category}.jsonl"] = hashlib.sha256(payload).hexdigest()
        for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            key = (category, str(item["question_index"]), int(item["example_id"]))
            if key not in target_locations:
                # The official scorer also excludes examples lacking target metadata.
                continue
            unknown = [
                index for index in range(3)
                if answer_kind(item["answer_info"], index).lower() == "unknown"
            ]
            if len(unknown) != 1:
                raise ValueError(
                    f"{category}.jsonl:{line_number} has {len(unknown)} UNKNOWN options."
                )
            row = {column: item[column] for column in OUTPUT_COLUMNS[:11]}
            row["target_loc"] = target_locations[key]
            row["unknown_loc"] = unknown[0]
            if int(row["label"]) not in range(3) or row["target_loc"] not in range(3):
                raise ValueError(f"Invalid BBQ answer index at {category}:{item['example_id']}")
            if row["context_condition"] == "ambig" and int(row["label"]) != unknown[0]:
                raise ValueError(
                    f"Ambiguous BBQ gold answer is not UNKNOWN at {category}:{item['example_id']}"
                )
            rows.append(row)
    data = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if data.empty or set(data["context_condition"]) != {"ambig", "disambig"}:
        raise ValueError("Downloaded BBQ data is empty or lacks a context condition.")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".parquet.tmp")
    try:
        data.to_parquet(temporary, index=False)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "dataset": REPOSITORY,
        "requested_revision": args.revision,
        "resolved_revision": revision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(data),
        "categories": list(CATEGORIES),
        "source_sha256": source_hashes,
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
    print(f"Wrote {len(data):,} BBQ examples to {output_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
