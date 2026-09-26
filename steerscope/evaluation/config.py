import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


NODE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class EvaluatorNode:
    node_id: str
    evaluator_type: str
    depends_on: tuple[str, ...] = ()
    requires_inference: bool = True
    models: tuple[str, ...] = ()
    concepts: Mapping[str, Any] = field(default_factory=dict)
    dataset: Mapping[str, Any] = field(default_factory=dict)
    inference: Mapping[str, Any] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)
    input: Mapping[str, Any] = field(default_factory=dict)
    report: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.evaluator_type,
            "depends_on": list(self.depends_on),
            "requires_inference": self.requires_inference,
            "models": list(self.models),
            "concepts": dict(self.concepts),
            "dataset": dict(self.dataset),
            "inference": dict(self.inference),
            "params": dict(self.params),
            "input": dict(self.input),
            "report": dict(self.report),
        }

    def config_hash(
        self,
        dependency_hashes: Mapping[str, str] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        payload = self.as_dict()
        # Reporting changes presentation only and must not invalidate inference.
        payload.pop("report", None)
        payload["dependency_hashes"] = dict(sorted((dependency_hashes or {}).items()))
        payload["context"] = dict(context or {})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def parse_evaluator_nodes(
    evaluators: Mapping[str, Any] | Sequence[Any] | None,
    evaluator_resolver=None,
) -> list[EvaluatorNode]:
    if not evaluators:
        raise ValueError("'evaluate.evaluators' must configure at least one evaluator node.")

    if isinstance(evaluators, Mapping):
        entries = list(evaluators.items())
    elif isinstance(evaluators, Sequence) and not isinstance(evaluators, (str, bytes)):
        entries = []
        for index, value in enumerate(evaluators):
            if isinstance(value, Mapping):
                node_id = value.get("id")
                if not node_id:
                    raise ValueError(f"Evaluator entry {index} is missing an 'id'.")
                entries.append((node_id, value))
            else:
                raise TypeError(f"Evaluator entry {index} must be a mapping.")
    else:
        raise TypeError("'evaluators' must be a mapping or a list.")

    nodes = []
    seen = set()
    for node_id, value in entries:
        if not NODE_ID_PATTERN.fullmatch(str(node_id)):
            raise ValueError(
                f"Invalid evaluator id '{node_id}'. Use letters, numbers, '.', '_' or '-'."
            )
        node_id = str(node_id)
        if node_id in seen:
            raise ValueError(f"Duplicate evaluator id '{node_id}'.")
        seen.add(node_id)

        if isinstance(value, str):
            value = {"type": value}
        if not isinstance(value, Mapping):
            raise TypeError(f"Configuration for evaluator '{node_id}' must be a mapping.")
        evaluator_type = value.get("type")
        if not evaluator_type:
            raise ValueError(f"Evaluator '{node_id}' is missing a 'type'.")

        try:
            evaluator_class = evaluator_resolver(evaluator_type) if evaluator_resolver else None
        except (AttributeError, KeyError) as error:
            raise ValueError(
                f"Unknown evaluator type '{evaluator_type}' for node '{node_id}'."
            ) from error

        class_requires_inference = getattr(
            evaluator_class, "requires_inference", True
        )
        configured_requires_inference = bool(value.get(
            "requires_inference", class_requires_inference
        ))
        if configured_requires_inference != class_requires_inference:
            raise ValueError(
                f"Evaluator '{node_id}' cannot override requires_inference for "
                f"type '{evaluator_type}'."
            )

        depends_on = value.get(
            "depends_on",
            value.get("dependencies", ()),
        ) or ()
        if isinstance(depends_on, str):
            depends_on = (depends_on,)
        models = value.get("models", ()) or ()
        if isinstance(models, str):
            models = (models,)

        nodes.append(
            EvaluatorNode(
                node_id=node_id,
                evaluator_type=str(evaluator_type),
                depends_on=tuple(depends_on),
                requires_inference=configured_requires_inference,
                models=tuple(models),
                concepts=_normalize_section(value.get("concepts")),
                dataset=_normalize_section(value.get("dataset"), scalar_key="name"),
                inference=_normalize_section(value.get("inference")),
                params=_normalize_section(value.get("params")),
                input=_normalize_section(value.get("input"), scalar_key="from"),
                report=_normalize_section(value.get("report")),
            )
        )
    return nodes


def apply_node_overrides(args: Any, node: EvaluatorNode) -> SimpleNamespace:
    values = copy.deepcopy(vars(args))
    if node.models:
        values["models"] = list(node.models)

    inference_aliases = {
        "batch_size": "steering_batch_size",
        "output_length": "steering_output_length",
        "layers": "steering_layers",
        "layer": "steering_layer",
        "intervention_type": "steering_intervention_type",
    }
    for key, value in node.inference.items():
        values[inference_aliases.get(key, key)] = value

    values["evaluation_node_id"] = node.node_id
    values["evaluation_node_type"] = node.evaluator_type
    values["evaluation_node_hash"] = node.config_hash()
    return SimpleNamespace(**values)


def _normalize_section(value: Any, scalar_key: str | None = None) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if scalar_key is not None:
        return {scalar_key: value}
    raise TypeError("Evaluator configuration sections must be mappings.")
