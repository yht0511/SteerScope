"""Deterministic, paired subsampling for training-data studies."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd


SUBSET_POLICY_VERSION = 1


def _stable_order(values, seed: int, group_key: Any) -> list:
    """Order values reproducibly without relying on pandas/Python RNG details."""
    prefix = f"v{SUBSET_POLICY_VERSION}:{int(seed)}:{group_key}:"
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"{prefix}{value}".encode("utf-8")
        ).digest(),
    )


def select_balanced_subset(
    positive_df: pd.DataFrame,
    negative_df: pd.DataFrame,
    max_num_of_examples: int | None,
    subset_seed: int | None,
    group_key: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select a deterministic, nested positive/negative subset at pair level when pair IDs are available."""
    positive_df = positive_df.copy()
    negative_df = negative_df.copy()
    available_per_class = min(len(positive_df), len(negative_df))
    if max_num_of_examples is None:
        requested_per_class = available_per_class
    else:
        max_num_of_examples = int(max_num_of_examples)
        if max_num_of_examples <= 0:
            raise ValueError("train.max_num_of_examples must be positive.")
        requested_per_class = max_num_of_examples // 2
    if requested_per_class > available_per_class:
        raise ValueError(
            f"Requested {requested_per_class} positive/negative training pairs, "
            f"but only {available_per_class} are available for {group_key}."
        )

    use_pair_ids = (
        "pair_id" in positive_df.columns
        and "pair_id" in negative_df.columns
        and positive_df["pair_id"].notna().all()
        and negative_df["pair_id"].notna().all()
    )
    if use_pair_ids:
        positive_ids = list(dict.fromkeys(positive_df["pair_id"].tolist()))
        negative_ids = set(negative_df["pair_id"].tolist())
        pair_ids = [pair_id for pair_id in positive_ids if pair_id in negative_ids]
        if len(pair_ids) < requested_per_class:
            raise ValueError(
                f"Only {len(pair_ids)} complete positive/negative pairs are "
                f"available for {group_key}."
            )
        ordered = (
            pair_ids
            if subset_seed is None
            else _stable_order(pair_ids, subset_seed, group_key)
        )
        selected_ids = ordered[:requested_per_class]
        selected_set = set(selected_ids)
        positive_subset = positive_df[positive_df["pair_id"].isin(selected_set)]
        negative_subset = negative_df[negative_df["pair_id"].isin(selected_set)]
        selected_keys = selected_ids
    else:
        positions = list(range(available_per_class))
        ordered = (
            positions
            if subset_seed is None
            else _stable_order(positions, subset_seed, group_key)
        )
        selected_positions = ordered[:requested_per_class]
        # Apply the same positions to both sides to preserve legacy paired data.
        positive_subset = positive_df.iloc[selected_positions]
        negative_subset = negative_df.iloc[selected_positions]
        selected_keys = selected_positions

    selected_digest = hashlib.sha256(
        "\n".join(str(value) for value in selected_keys).encode("utf-8")
    ).hexdigest()
    provenance = {
        "policy_version": SUBSET_POLICY_VERSION,
        "group_key": str(group_key),
        "subset_seed": None if subset_seed is None else int(subset_seed),
        "max_num_of_examples": max_num_of_examples,
        "selected_pairs": int(requested_per_class),
        "selected_pair_ids": list(selected_keys),
        "selected_pair_ids_sha256": selected_digest,
    }
    return positive_subset.copy(), negative_subset.copy(), provenance


def select_unpaired_subset(
    data: pd.DataFrame,
    max_rows: int | None,
    subset_seed: int | None,
    group_key: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select a deterministic nested subset when no negative pair exists."""
    requested = len(data) if max_rows is None else int(max_rows)
    if requested < 0:
        raise ValueError("Requested training subset size cannot be negative.")
    if requested > len(data):
        raise ValueError(
            f"Requested {requested} training rows, but only {len(data)} are "
            f"available for {group_key}."
        )
    values = (
        list(dict.fromkeys(data["pair_id"].tolist()))
        if "pair_id" in data.columns and data["pair_id"].notna().all()
        else list(range(len(data)))
    )
    ordered = values if subset_seed is None else _stable_order(
        values, subset_seed, group_key
    )
    selected = ordered[:requested]
    if "pair_id" in data.columns and data["pair_id"].notna().all():
        result = data[data["pair_id"].isin(set(selected))]
    else:
        result = data.iloc[selected]
    digest = hashlib.sha256(
        "\n".join(str(value) for value in selected).encode("utf-8")
    ).hexdigest()
    return result.copy(), {
        "policy_version": SUBSET_POLICY_VERSION,
        "group_key": str(group_key),
        "subset_seed": None if subset_seed is None else int(subset_seed),
        "max_rows": max_rows,
        "selected_rows": int(len(result)),
        "selected_ids": list(selected),
        "selected_ids_sha256": digest,
    }
