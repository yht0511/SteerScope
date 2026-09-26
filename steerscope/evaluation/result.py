from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd


@dataclass
class EvaluationResult:
    """Complete output produced by one evaluator node."""

    inference: pd.DataFrame | None = None
    samples: pd.DataFrame | None = None
    metrics: pd.DataFrame | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def result_kinds(self) -> list[str]:
        return [
            name
            for name in ("inference", "samples", "metrics")
            if getattr(self, name) is not None
        ]
