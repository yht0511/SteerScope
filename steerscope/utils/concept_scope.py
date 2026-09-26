"""Deterministic concept-panel selection shared by training and evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


def select_concept_ids(
    concept_ids: Iterable[int],
    *,
    count: int,
    seed: int = 42,
) -> list[int]:
    """Select a stable, order-independent concept panel.

    Hash ranking avoids dependence on dataframe order and gives training and
    evaluator scopes exactly the same semantics on the same concept universe.
    """
    available = sorted({int(value) for value in concept_ids})
    count = int(count)
    seed = int(seed)
    if count < 1:
        raise ValueError("Concept count must be at least 1.")
    if count > len(available):
        raise ValueError(
            f"Requested {count} concepts, but only {len(available)} are available."
        )
    selected = sorted(
        available,
        key=lambda concept_id: (
            hashlib.sha256(
                f"concept-scope-v1:{seed}:{concept_id}".encode("utf-8")
            ).digest(),
            concept_id,
        ),
    )[:count]
    return sorted(selected)
