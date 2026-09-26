"""Configuration shared by native SteerScope FLAS training and inference."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FLASConfig:
    """The public FLAS recipe, using the defaults from the released code."""

    num_blocks: int = 1
    n_steps: int = 3
    t_min: float = 0.5
    t_max: float = 2.0
    div_weight: float = 0.1
    total_steps: int = 80_000
    warmup_steps: int = 2_000
    max_length: int = 256
    concept_max_length: int = 64
    n_val_samples: int = 100
    val_n_concepts: int = 0
    val_every: int = 500
    val_batches: int = 100
    patience: int = 30
    num_workers: int = 4

    def validate(self) -> None:
        if self.num_blocks < 1:
            raise ValueError("flas_num_blocks must be at least one.")
        if self.n_steps < 1:
            raise ValueError("flas_n_steps must be at least one.")
        if not math.isfinite(self.t_min) or not math.isfinite(self.t_max):
            raise ValueError("FLAS flow-time bounds must be finite.")
        if self.t_min < 0 or self.t_max < self.t_min:
            raise ValueError("FLAS requires 0 <= t_min <= t_max.")
        if not math.isfinite(self.div_weight) or self.div_weight < 0:
            raise ValueError("flas_div_weight must be finite and non-negative.")
        for name in (
            "total_steps",
            "max_length",
            "concept_max_length",
            "val_every",
            "val_batches",
            "patience",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"flas_{name} must be at least one.")
        if self.warmup_steps < 0 or self.warmup_steps > self.total_steps:
            raise ValueError(
                "flas_warmup_steps must be between zero and flas_total_steps."
            )
        if self.n_val_samples < 0 or self.val_n_concepts < 0:
            raise ValueError("FLAS validation sizes must be non-negative.")
        if self.num_workers < 0:
            raise ValueError("flas_num_workers must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_training_args(cls, args: Any) -> FLASConfig:
        values = {}
        for field, default in cls().__dict__.items():
            value = getattr(args, f"flas_{field}", None)
            values[field] = default if value is None else value
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> FLASConfig:
        """Read either native SteerScope keys or the released FLAS config keys."""
        defaults = cls()
        aliases = {
            "T_min": "t_min",
            "T_max": "t_max",
            "max_len": "max_length",
            "concept_max_len": "concept_max_length",
            "n_val_samples": "n_val_samples",
            "val_n_concepts": "val_n_concepts",
            "val_every": "val_every",
            "val_batches": "val_batches",
            "patience": "patience",
            "num_workers": "num_workers",
        }
        normalized = {}
        for key, value in values.items():
            key = key.removeprefix("flas_")
            normalized[aliases.get(key, key)] = value
        payload = {}
        for name, default in defaults.__dict__.items():
            value = normalized.get(name)
            payload[name] = default if value is None else value
        config = cls(**payload)
        config.validate()
        return config
