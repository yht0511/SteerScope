from dataclasses import dataclass
from pathlib import Path
from typing import Any


def context_for_node(context, node_id):
    """Combine shared execution inputs with one evaluator's private inputs."""
    values = dict(context or {})
    evaluator_contexts = values.pop("evaluators", {})
    values.update(dict(evaluator_contexts.get(node_id, {})))
    return values


@dataclass(frozen=True)
class EvaluationContext:
    """Runtime services available to one evaluator node."""

    args: Any
    root_dump_dir: Path
    output_dir: Path
    results: Any = None
    progress: Any = None
