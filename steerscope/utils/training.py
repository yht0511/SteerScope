"""Small, framework-independent helpers for hand-written training loops."""

from __future__ import annotations

import operator


# Fingerprint the corrected final partial accumulation window.
GRADIENT_ACCUMULATION_SEMANTICS = "epoch_local_actual_window_v1"


def _nonnegative_integer(value, name: str) -> int:
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer, got {value!r}.") from exc
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value}.")
    return value


def _positive_integer(value, name: str) -> int:
    value = _nonnegative_integer(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive.")
    return value


def optimizer_steps_per_epoch(
    num_microbatches: int,
    gradient_accumulation_steps: int,
) -> int:
    """Return the number of optimizer updates, including a partial tail."""

    num_microbatches = _nonnegative_integer(
        num_microbatches, "num_microbatches"
    )
    gradient_accumulation_steps = _positive_integer(
        gradient_accumulation_steps, "gradient_accumulation_steps"
    )
    return (
        num_microbatches + gradient_accumulation_steps - 1
    ) // gradient_accumulation_steps


def accumulation_window_size(
    microbatch_index: int,
    num_microbatches: int,
    gradient_accumulation_steps: int,
) -> int:
    """Return the epoch-local accumulation-window size, including a final partial window."""

    num_microbatches = _nonnegative_integer(
        num_microbatches, "num_microbatches"
    )
    gradient_accumulation_steps = _positive_integer(
        gradient_accumulation_steps, "gradient_accumulation_steps"
    )
    microbatch_index = _nonnegative_integer(
        microbatch_index, "microbatch_index"
    )
    if microbatch_index >= num_microbatches:
        raise IndexError(
            "microbatch_index must be smaller than num_microbatches, got "
            f"{microbatch_index} >= {num_microbatches}."
        )
    window_start = (
        microbatch_index // gradient_accumulation_steps
    ) * gradient_accumulation_steps
    return min(
        gradient_accumulation_steps,
        num_microbatches - window_start,
    )


def normalize_loss_for_accumulation(
    loss,
    microbatch_index: int,
    num_microbatches: int,
    gradient_accumulation_steps: int,
):
    """Scale a scalar loss by its actual epoch-local accumulation window."""

    return loss / accumulation_window_size(
        microbatch_index,
        num_microbatches,
        gradient_accumulation_steps,
    )


def is_optimizer_step(
    microbatch_index: int,
    num_microbatches: int,
    gradient_accumulation_steps: int,
) -> bool:
    """Return whether this microbatch closes a full or partial window."""

    # Calling accumulation_window_size performs all argument validation.
    accumulation_window_size(
        microbatch_index,
        num_microbatches,
        gradient_accumulation_steps,
    )
    return (
        (microbatch_index + 1) % gradient_accumulation_steps == 0
        or microbatch_index + 1 == num_microbatches
    )
