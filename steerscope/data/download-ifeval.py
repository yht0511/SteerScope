#!/usr/bin/env python3
"""Download Google's official IFEval prompts as validated Parquet data."""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd


REPOSITORY = "google-research/google-research"
UPSTREAM_PATH = "instruction_following_eval/data/input_data.jsonl"
EXPECTED_ROWS = 541


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=script_dir / "ifeval")
    parser.add_argument("--revision", default="95e3a1da2d27cb9c8289f6fd3076cfed608c3c94")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fetch(url):
    response = httpx.get(url, follow_redirects=True, timeout=60.0)
    response.raise_for_status()
    return response.content


def resolve_revision(revision):
    payload = json.loads(fetch(f"https://api.github.com/repos/{REPOSITORY}/commits/{revision}"))
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
    output_path = output_dir / "IFEval.parquet"
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
    rows = []
    seen_keys = set()
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        required = {"key", "prompt", "instruction_id_list", "kwargs"}
        missing = sorted(required.difference(item))
        if missing:
            raise ValueError(f"IFEval line {line_number} is missing fields: {missing}")
        key = int(item["key"])
        if key in seen_keys:
            raise ValueError(f"Duplicate IFEval key {key}.")
        seen_keys.add(key)
        if not str(item["prompt"]).strip():
            raise ValueError(f"IFEval key {key} has an empty prompt.")
        if not item["instruction_id_list"] or len(item["instruction_id_list"]) != len(item["kwargs"]):
            raise ValueError(f"IFEval key {key} has invalid instruction metadata.")
        rows.append({
            "key": key,
            "prompt": str(item["prompt"]),
            "instruction_id_list_json": json.dumps(
                item["instruction_id_list"], ensure_ascii=False, separators=(",", ":")
            ),
            "kwargs_json": json.dumps(
                item["kwargs"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        })
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"IFEval has {len(rows)} rows; expected {EXPECTED_ROWS}.")
    # Prefer the checker vendored next to this script.  The active environment
    # may contain another editable package checkout, which must not decide
    # which IFEval rules validate this dataset.
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    try:
        from steerscope.evaluators.ifeval_official.instructions_registry import (
            INSTRUCTION_DICT,
        )
    except ImportError as error:
        raise RuntimeError(
            "Install the project dependencies before downloading IFEval."
        ) from error
    configured_ids = {
        instruction_id
        for row in rows
        for instruction_id in json.loads(row["instruction_id_list_json"])
    }
    unknown_ids = sorted(configured_ids.difference(INSTRUCTION_DICT))
    if unknown_ids:
        raise ValueError(
            f"IFEval data contains instruction IDs absent from the pinned checker: {unknown_ids}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".parquet.tmp")
    try:
        pd.DataFrame(rows).to_parquet(temporary, index=False)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "dataset": REPOSITORY,
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
    print(f"Wrote {len(rows)} IFEval examples to {output_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
