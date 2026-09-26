"""Stable training-recipe identifiers and per-task random seeds."""

from __future__ import annotations

import hashlib


TRAINING_RECIPE_VERSION = 3
TRAINING_SEED_DERIVATION = "sha256_method_concept_v1"


def derive_training_seed(
    base_seed: int,
    method: str,
    concept_id: int | None = None,
) -> int:
    """Derive a process-stable seed independent of loop and resume order."""
    scope = "all_concepts" if concept_id is None else f"concept:{int(concept_id)}"
    payload = (
        f"{TRAINING_SEED_DERIVATION}\0{int(base_seed)}\0{method}\0{scope}"
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % (2**31 - 1)
