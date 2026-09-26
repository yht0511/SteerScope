from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Concept:
    concept_id: int
    text: str
    ref: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Artifact:
    kind: str
    path: Path | None = None
    fingerprint: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def signature(self) -> Mapping[str, Any]:
        """Return stable artifact information suitable for cache manifests."""
        if self.fingerprint is not None:
            return {"kind": self.kind, "fingerprint": self.fingerprint}
        if self.path is None:
            return {"kind": self.kind, "path": None}
        path = Path(self.path)
        if not path.exists():
            return {"kind": self.kind, "path": str(path), "missing": True}
        if path.is_dir():
            files = []
            for root, directories, names in os.walk(path, followlinks=True):
                directories.sort()
                for name in sorted(names):
                    child = Path(root) / name
                    files.append({
                        "path": str(child.relative_to(path)),
                        "sha256": _file_sha256(child),
                    })
            return {"kind": self.kind, "path": str(path), "files": files}
        return {
            "kind": self.kind,
            "path": str(path),
            "sha256": _file_sha256(path),
        }


@dataclass(frozen=True)
class EvaluationTarget:
    target_id: str
    method: str
    concept: Concept
    base_model: str
    artifact: Artifact


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def targets_from_metadata(
    metadata: Iterable[Mapping[str, Any]],
    methods: Iterable[str],
    base_model: str,
    artifact_dir: str | Path | None = None,
) -> list[EvaluationTarget]:
    artifact_root = Path(artifact_dir) if artifact_dir is not None else None
    targets = []
    for item in metadata:
        concept = Concept(
            concept_id=int(item["concept_id"]),
            text=str(item["concept"]),
            ref=item.get("ref"),
            metadata=dict(item),
        )
        for method in methods:
            targets.append(
                EvaluationTarget(
                    target_id=f"{method}/concept-{concept.concept_id}",
                    method=method,
                    concept=concept,
                    base_model=base_model,
                    artifact=Artifact(kind="checkpoint", path=artifact_root),
                )
            )
    return targets


def parse_evaluation_targets(
    targets: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    default_base_model: str,
    base_dir: str | Path | None = None,
) -> list[EvaluationTarget]:
    """Parse explicitly configured checkpoints or standalone steering vectors."""
    if isinstance(targets, Mapping):
        entries = list(targets.items())
    elif isinstance(targets, Sequence) and not isinstance(targets, (str, bytes)):
        entries = []
        for index, value in enumerate(targets):
            if not isinstance(value, Mapping):
                raise TypeError(f"Evaluation target {index} must be a mapping.")
            target_id = value.get("id", value.get("target_id"))
            if not target_id:
                raise ValueError(f"Evaluation target {index} is missing an 'id'.")
            entries.append((target_id, value))
    else:
        raise TypeError("'targets' must be a mapping or list of mappings.")

    root = Path(base_dir) if base_dir is not None else None
    parsed = []
    seen = set()
    for target_id, value in entries:
        target_id = str(target_id)
        if target_id in seen:
            raise ValueError(f"Duplicate evaluation target id '{target_id}'.")
        seen.add(target_id)
        if not isinstance(value, Mapping):
            raise TypeError(f"Evaluation target '{target_id}' must be a mapping.")

        method = value.get("method")
        if not method:
            raise ValueError(f"Evaluation target '{target_id}' is missing 'method'.")
        concept_value = value.get("concept")
        if isinstance(concept_value, str):
            concept_value = {"text": concept_value}
        if not isinstance(concept_value, Mapping):
            raise ValueError(
                f"Evaluation target '{target_id}' must configure a concept mapping."
            )
        concept_id = concept_value.get("id", concept_value.get("concept_id", 0))
        concept_text = concept_value.get("text", concept_value.get("concept"))
        if not concept_text:
            raise ValueError(
                f"Evaluation target '{target_id}' concept is missing 'text'."
            )

        artifact_value = value.get("artifact", {})
        if isinstance(artifact_value, (str, Path)):
            artifact_value = {"kind": "steering_vector", "path": artifact_value}
        if not isinstance(artifact_value, Mapping):
            raise TypeError(
                f"Evaluation target '{target_id}' artifact must be a mapping or path."
            )
        artifact_path = artifact_value.get("path")
        if artifact_path is not None:
            artifact_path = Path(artifact_path).expanduser()
            if root is not None and not artifact_path.is_absolute():
                artifact_path = root / artifact_path
            artifact_path = artifact_path.resolve()
        artifact = Artifact(
            kind=str(artifact_value.get("kind", "checkpoint")),
            path=artifact_path,
            fingerprint=artifact_value.get("fingerprint"),
            metadata=dict(artifact_value.get("metadata") or {}),
        )
        if artifact.kind not in {"checkpoint", "steering_vector"}:
            raise ValueError(
                f"Evaluation target '{target_id}' has unsupported artifact kind "
                f"'{artifact.kind}'."
            )
        if artifact.path is None:
            raise ValueError(
                f"Evaluation target '{target_id}' artifact is missing 'path'."
            )
        if not artifact.path.exists():
            raise FileNotFoundError(
                f"Evaluation target '{target_id}' artifact does not exist: {artifact.path}"
            )
        if artifact.kind == "checkpoint" and not artifact.path.is_dir():
            raise ValueError(
                f"Evaluation target '{target_id}' checkpoint artifact must be a directory."
            )
        if artifact.kind == "steering_vector" and not artifact.path.is_file():
            raise ValueError(
                f"Evaluation target '{target_id}' steering_vector artifact must be a file."
            )
        if artifact.kind == "steering_vector" and method != "SteeringVector":
            raise ValueError(
                f"Evaluation target '{target_id}' must use method 'SteeringVector' "
                "for a standalone steering_vector artifact."
            )

        parsed.append(EvaluationTarget(
            target_id=target_id,
            method=str(method),
            concept=Concept(
                concept_id=int(concept_id),
                text=str(concept_text),
                ref=concept_value.get("ref"),
                metadata=dict(concept_value.get("metadata") or {}),
            ),
            base_model=str(value.get("base_model") or default_base_model),
            artifact=artifact,
        ))
    return parsed
