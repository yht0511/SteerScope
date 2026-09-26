"""Dataset-agnostic helpers for evaluator-owned data pipelines."""

import hashlib
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import pandas as pd


CONCEPT_SEED_VERSION = "concept-seed-sha256-v1"


def concept_seed(
    base_seed: int,
    concept_id: int,
    namespace: str,
) -> int:
    """Return a deterministic dataset/concept seed shared across methods and factors."""
    payload = (
        f"{CONCEPT_SEED_VERSION}:{str(namespace)}:"
        f"{int(base_seed)}:{int(concept_id)}"
    )
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:4],
        byteorder="big",
    )


def require_dataset_type(
    config: Mapping[str, Any],
    supported: str | Iterable[str],
) -> str:
    """Validate a dataset name inside the evaluator that understands it."""
    configured = config.get("type", config.get("name"))
    if configured is None:
        raise ValueError("Evaluator dataset configuration must define 'type'.")
    supported_types = (
        (supported,) if isinstance(supported, str) else tuple(supported)
    )
    if not supported_types:
        raise ValueError("Evaluator must declare at least one supported dataset type.")
    if configured not in supported_types:
        raise ValueError(
            f"Unsupported dataset type '{configured}'; expected one of "
            f"{list(supported_types)}."
        )
    return str(configured)


def require_num_examples(config: Mapping[str, Any]) -> int:
    """Read a positive sample count without assigning dataset semantics to it."""
    value = config.get("num_examples", config.get("samples_per_concept"))
    if value is None:
        raise ValueError("Evaluator dataset configuration must define num_examples.")
    value = int(value)
    if value < 1:
        raise ValueError("Evaluator dataset num_examples must be at least 1.")
    return value


def expand_factors(
    examples: pd.DataFrame,
    factors: Iterable[float],
    *,
    dataset_factor: Callable[[float], float] | None = None,
) -> pd.DataFrame:
    """Cross examples with model factors while preserving dataset factor semantics."""
    frames = []
    transform = dataset_factor or (lambda value: value)
    for factor in factors:
        factor = float(factor)
        frame = examples.copy()
        frame["factor"] = float(transform(factor))
        frame["model_factor"] = factor
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def split_by_input_id(
    examples: pd.DataFrame,
    split: str | None,
    ratio: float | None,
) -> pd.DataFrame:
    """Apply the common deterministic validation/test partition by input ID."""
    if split in {None, "all"}:
        return examples.reset_index(drop=True)
    split = {"steering": "validation", "steering_test": "test"}.get(
        split, split
    )
    if split not in {"validation", "test"}:
        raise ValueError(f"Unsupported evaluator dataset split '{split}'.")
    if "input_id" not in examples:
        raise ValueError(f"Dataset split '{split}' requires an input_id column.")
    ratio = 0.5 if ratio is None else float(ratio)
    if not 0 < ratio < 1:
        raise ValueError("Evaluator dataset split_ratio must be between 0 and 1.")
    unique_ids = sorted(examples["input_id"].unique())
    boundary = len(unique_ids) - round(len(unique_ids) * ratio)
    selected = (
        unique_ids[:boundary] if split == "validation" else unique_ids[boundary:]
    )
    return examples[examples["input_id"].isin(selected)].reset_index(drop=True)
