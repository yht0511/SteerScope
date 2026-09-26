import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pandas as pd


class ResultStore:
    IDENTITY_MIGRATION_ENV = "STEERSCOPE_EVALUATION_IDENTITY_MIGRATION_FILE"

    def __init__(
        self,
        root_dir: str | Path,
        run_id: str = "default",
        progress_root: str | Path | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.run_id = str(run_id)
        self.run_dir = self.root_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        configured_progress_root = (
            Path(progress_root) if progress_root is not None else self.root_dir
        )
        self.progress_root = configured_progress_root
        self.progress_run_dir = self.progress_root / self.run_id
        self.mirrored_progress = (
            self.progress_root.resolve() != self.root_dir.resolve()
        )

    def view(self, dependencies: Iterable[str] = ()) -> "ResultView":
        return ResultView(self, tuple(dependencies))

    def progress(self, node_id: str) -> "NodeProgressStore":
        manifest = self.manifest(node_id)
        if manifest is None or not manifest.get("execution_hash"):
            raise ValueError(
                f"Evaluator '{node_id}' must be marked running before opening progress."
            )
        return NodeProgressStore(
            self, node_id=node_id, execution_hash=manifest["execution_hash"]
        )

    def node_dir(self, node_id: str) -> Path:
        path = self.run_dir / "evaluators" / node_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def progress_node_dir(self, node_id: str) -> Path:
        if not self.mirrored_progress:
            return self.node_dir(node_id)
        path = self.progress_run_dir / "evaluators" / node_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def progress_execution_dir(self, node_id: str, execution_hash: str) -> Path:
        return self.progress_node_dir(node_id) / "progress" / str(execution_hash)

    def save_inference(self, node_id: str, data: pd.DataFrame) -> Path:
        return self._save_frame(node_id, "inference", data)

    def save_samples(self, node_id: str, data: pd.DataFrame) -> Path:
        return self._save_frame(node_id, "samples", data)

    def save_metrics(self, node_id: str, data: pd.DataFrame) -> Path:
        return self._save_frame(node_id, "metrics", data)

    def load(self, node_id: str, kind: str) -> pd.DataFrame:
        path = self._frame_path(node_id, kind)
        if not path.exists():
            raise FileNotFoundError(f"No {kind} results found for evaluator '{node_id}'.")
        return pd.read_parquet(path)

    def load_result(self, node_id: str):
        """Load a completed evaluator result without rerunning the evaluator."""
        from .result import EvaluationResult

        manifest = self.manifest(node_id)
        if not manifest or manifest.get("status") != "complete":
            raise ValueError(f"Evaluator '{node_id}' does not have complete results.")
        metadata = dict(manifest.get("metadata") or {})
        result_kinds = metadata.get("result_kinds", ())
        frames = {}
        for kind in result_kinds:
            frames[kind] = self.load(node_id, kind)
        return EvaluationResult(
            inference=frames.get("inference"),
            samples=frames.get("samples"),
            metrics=frames.get("metrics"),
            metadata=metadata,
        )

    def record_report(
        self,
        node_id: str,
        paths: Iterable[str] = (),
        error: str | None = None,
    ) -> None:
        manifest = self.manifest(node_id)
        if not manifest or manifest.get("status") != "complete":
            raise ValueError(
                f"Evaluator '{node_id}' must be complete before recording a report."
            )
        metadata = dict(manifest.get("metadata") or {})
        report = {
            "status": "failed" if error is not None else "complete",
            "paths": list(paths),
        }
        if error is not None:
            report["error"] = error
        metadata["report"] = report
        manifest["metadata"] = metadata
        self._write_manifest(node_id, manifest)

    def query(
        self,
        node_id: str,
        kind: str = "metrics",
        filters: Mapping[str, Any] | None = None,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        data = self.load(node_id, kind)
        for column, expected in (filters or {}).items():
            if column not in data.columns:
                raise KeyError(f"Column '{column}' is not available in {kind} results for '{node_id}'.")
            if isinstance(expected, (list, tuple, set, frozenset)):
                data = data[data[column].isin(expected)]
            else:
                data = data[data[column] == expected]
        if columns is not None:
            data = data[list(columns)]
        return data.copy()

    def manifest(self, node_id: str) -> dict[str, Any] | None:
        path = self.node_dir(node_id) / "manifest.json"
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as file:
                manifest = json.load(file)
        except (OSError, json.JSONDecodeError):
            return None
        if self.mirrored_progress and manifest.get("execution_hash"):
            progress = self._read_progress_metadata(
                node_id, str(manifest["execution_hash"])
            )
            if progress is not None:
                metadata = dict(manifest.get("metadata") or {})
                metadata["progress"] = progress
                manifest["metadata"] = metadata
        return manifest

    @classmethod
    def semantic_cache_identity(
        cls,
        config: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the experiment identity used to reuse completed results.

        This identity records the evaluator inputs. Completed results also
        require the exact execution hash before they can be reused.
        """

        def normalize(value):
            if isinstance(value, Mapping):
                return {
                    str(key): normalize(item)
                    for key, item in sorted(
                        value.items(), key=lambda pair: str(pair[0])
                    )
                    if key not in {"source_fingerprint", "pipeline_version"}
                }
            if isinstance(value, (list, tuple)):
                return [normalize(item) for item in value]
            return value

        normalized_config = cls._normalized_config(config)
        identity = {
            "config": normalize(normalized_config),
            "context": normalize(dict(context or {})),
        }
        return json.loads(json.dumps(identity, sort_keys=True, default=str))

    def is_complete(
        self,
        node_id: str,
        execution_hash: str,
        config: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        manifest = self.manifest(node_id)
        if not (
            manifest
            and manifest.get("status") == "complete"
        ):
            return False
        required_results = manifest.get("metadata", {}).get(
            "result_kinds", ("metrics", "samples")
        )
        if not all(
            self._frame_path(node_id, kind).exists() for kind in required_results
        ):
            return False

        if manifest.get("execution_hash") != execution_hash:
            return False

        if config is None:
            return True
        current_identity = self.semantic_cache_identity(config, context)
        stored_identity = manifest.get("cache_identity")
        if not stored_identity:
            # Results written before semantic identities were introduced can
            # still be reused when their complete evaluator configuration is
            # unchanged.
            if self._normalized_config(manifest.get("config") or {}) != (
                self._normalized_config(config)
            ):
                return False
        elif stored_identity != current_identity:
            return False

        return True

    def adopt_allowlisted_completion(
        self,
        node_id: str,
        execution_hash: str,
        config: Mapping[str, Any],
    ) -> bool:
        """Adopt a completed result only when its identity, config, and file digests match the allowlist."""
        migration_path = os.environ.get(self.IDENTITY_MIGRATION_ENV)
        if not migration_path:
            return False
        try:
            payload = json.loads(Path(migration_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if payload.get("version") != 1:
            return False
        entry = (
            ((payload.get("entries") or {}).get(self.run_id) or {})
            .get(node_id)
        )
        if not isinstance(entry, Mapping):
            return False
        manifest = self.manifest(node_id)
        if not manifest or manifest.get("status") != "complete":
            return False
        old_hash = manifest.get("execution_hash")
        if (
            entry.get("from_execution_hash") != old_hash
            or entry.get("to_execution_hash") != execution_hash
            or old_hash == execution_hash
        ):
            return False
        normalized_manifest = self._normalized_config(
            manifest.get("config") or {}
        )
        normalized_current = self._normalized_config(config)
        if normalized_manifest != normalized_current:
            return False
        if entry.get("config_sha256") != self._config_sha256(config):
            return False
        expected_results = entry.get("result_signatures")
        if not isinstance(expected_results, Mapping):
            return False
        try:
            current_results = self.result_signatures(node_id, manifest)
        except (FileNotFoundError, OSError):
            return False
        if dict(expected_results) != current_results:
            return False

        metadata = dict(manifest.get("metadata") or {})
        metadata["cache_identity_migration"] = {
            "from_execution_hash": old_hash,
            "to_execution_hash": execution_hash,
            "reason": "exact_allowlist",
            "allowlist": str(Path(migration_path).resolve()),
        }
        self._write_manifest(node_id, {
            "node_id": node_id,
            "run_id": self.run_id,
            "status": "complete",
            "execution_hash": execution_hash,
            "config": dict(config),
            "metadata": metadata,
        })
        return True

    @staticmethod
    def _normalized_config(value: Mapping[str, Any]) -> dict[str, Any]:
        normalized = json.loads(json.dumps(value, sort_keys=True, default=str))
        # Reporting is presentation-only and is excluded from config_hash.
        normalized.pop("report", None)
        return normalized

    @classmethod
    def _config_sha256(cls, value: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            cls._normalized_config(value),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def result_signatures(
        self,
        node_id: str,
        manifest: Mapping[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        manifest = dict(manifest or self.manifest(node_id) or {})
        signatures = {}
        for kind in manifest.get("metadata", {}).get(
            "result_kinds", ("metrics", "samples")
        ):
            path = self._frame_path(node_id, kind)
            if not path.exists():
                raise FileNotFoundError(
                    f"Evaluator '{node_id}' is missing its {kind} result."
                )
            signatures[kind] = {
                "size": path.stat().st_size,
                "sha256": self._file_sha256(path),
            }
        return signatures

    def completion_hash(self, node_id: str, execution_hash: str) -> str:
        manifest = self.manifest(node_id)
        if not manifest or manifest.get("status") != "complete":
            raise ValueError(f"Evaluator '{node_id}' does not have complete results.")
        result_signatures = self.result_signatures(node_id, manifest)
        payload = {
            "execution_hash": execution_hash,
            "results": result_signatures,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def mark_running(
        self,
        node_id: str,
        execution_hash: str,
        config: Mapping[str, Any],
        reset_progress: bool = False,
        cache_identity: Mapping[str, Any] | None = None,
    ) -> None:
        existing = self.manifest(node_id)
        same_execution = bool(
            existing and existing.get("execution_hash") == execution_hash
        )
        # Retain the previous result until its atomic replacement completes.
        if reset_progress:
            progress_dir = self.progress_execution_dir(node_id, execution_hash)
            if progress_dir.exists():
                shutil.rmtree(progress_dir)
            if self.mirrored_progress:
                # An explicit reset must not be undone by legacy-progress
                # seeding when the progress store is opened later.
                progress_dir.mkdir(parents=True, exist_ok=True)
                self._write_seed_marker(progress_dir, seeded_from=None)
        payload = {
            "node_id": node_id,
            "run_id": self.run_id,
            "status": "running",
            "execution_hash": execution_hash,
            "config": dict(config),
            "metadata": (
                dict(existing.get("metadata") or {})
                if same_execution and not reset_progress
                else {}
            ),
        }
        identity = cache_identity or (existing or {}).get("cache_identity")
        if identity:
            payload["cache_identity"] = dict(identity)
        self._write_manifest(node_id, payload)

    def _seed_legacy_progress(self, node_id: str, execution_hash: str) -> None:
        """Copy an old in-output checkpoint tree into mirrored progress once."""
        if not self.mirrored_progress:
            return
        destination = self.progress_execution_dir(node_id, execution_hash)
        marker = destination / ".legacy_seed_complete.json"
        if marker.is_file():
            return
        source = self.node_dir(node_id) / "progress" / str(execution_hash)
        destination.mkdir(parents=True, exist_ok=True)
        if source.is_dir() and source.resolve() != destination.resolve():
            shutil.copytree(source, destination, dirs_exist_ok=True)
            seeded_from = str(source.resolve())
        else:
            seeded_from = None
        self._write_seed_marker(destination, seeded_from=seeded_from)

    @staticmethod
    def _write_seed_marker(path: Path, seeded_from: str | None) -> None:
        NodeProgressStore._write_json_atomic(
            path / ".legacy_seed_complete.json",
            {"status": "complete", "seeded_from": seeded_from},
        )

    def _progress_metadata_path(self, node_id: str, execution_hash: str) -> Path:
        return self.progress_execution_dir(node_id, execution_hash) / "_progress.json"

    def _read_progress_metadata(
        self, node_id: str, execution_hash: str
    ) -> dict[str, Any] | None:
        if not self.mirrored_progress:
            return None
        return NodeProgressStore._read_json(
            self._progress_metadata_path(node_id, execution_hash)
        )

    def _write_progress_metadata(
        self,
        node_id: str,
        execution_hash: str,
        progress: Mapping[str, Any],
    ) -> None:
        NodeProgressStore._write_json_atomic(
            self._progress_metadata_path(node_id, execution_hash), progress
        )

    def mark_complete(
        self,
        node_id: str,
        execution_hash: str,
        config: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
        cache_identity: Mapping[str, Any] | None = None,
    ) -> None:
        existing = self.manifest(node_id)
        combined_metadata = (
            dict(existing.get("metadata") or {})
            if existing and existing.get("execution_hash") == execution_hash
            else {}
        )
        combined_metadata.update(dict(metadata or {}))
        payload = {
            "node_id": node_id,
            "run_id": self.run_id,
            "status": "complete",
            "execution_hash": execution_hash,
            "config": dict(config),
            "metadata": combined_metadata,
        }
        identity = cache_identity or (existing or {}).get("cache_identity")
        if identity:
            payload["cache_identity"] = dict(identity)
        self._write_manifest(node_id, payload)

    def mark_failed(
        self,
        node_id: str,
        execution_hash: str,
        config: Mapping[str, Any],
        error: str,
    ) -> None:
        existing = self.manifest(node_id)
        self._write_manifest(node_id, {
            "node_id": node_id,
            "run_id": self.run_id,
            "status": "failed",
            "execution_hash": execution_hash,
            "config": dict(config),
            "error": error,
            "metadata": (
                dict(existing.get("metadata") or {})
                if existing and existing.get("execution_hash") == execution_hash
                else {}
            ),
        })

    def _save_frame(self, node_id: str, kind: str, data: pd.DataFrame) -> Path:
        path = self._frame_path(node_id, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            delete=False, dir=path.parent, suffix=".parquet.tmp"
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            data.to_parquet(temporary_path, index=False)
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return path

    def _frame_path(self, node_id: str, kind: str) -> Path:
        if kind not in {"inference", "samples", "metrics"}:
            raise ValueError(f"Unsupported result kind '{kind}'.")
        return self.node_dir(node_id) / f"{kind}.parquet"

    def _write_manifest(self, node_id: str, payload: Mapping[str, Any]) -> None:
        path = self.node_dir(node_id) / "manifest.json"
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, dir=path.parent, suffix=".json.tmp", encoding="utf-8"
        ) as temporary:
            json.dump(payload, temporary, indent=2, sort_keys=True, default=str)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)


class NodeProgressStore:
    """Atomic per-target checkpoints for one evaluator execution hash."""

    def __init__(self, store: ResultStore, node_id: str, execution_hash: str):
        self.store = store
        self.node_id = node_id
        self.execution_hash = execution_hash
        if store.mirrored_progress:
            store._seed_legacy_progress(node_id, execution_hash)
        self.root = store.progress_execution_dir(node_id, self.execution_hash)
        self.root.mkdir(parents=True, exist_ok=True)

    def begin(self, target_ids: Iterable[str]) -> None:
        target_ids = list(target_ids)
        self._update_node_manifest(expected_target_ids=target_ids)

    def load_target(self, target_id: str):
        from .result import EvaluationResult

        manifest = self._target_manifest(target_id)
        if not self._valid_target_manifest(manifest, target_id):
            return None
        frames = {}
        for kind in manifest["result_kinds"]:
            path = self._target_dir(target_id) / f"{kind}.parquet"
            if not path.exists():
                return None
            try:
                frames[kind] = pd.read_parquet(path)
            except Exception:
                return None
        return EvaluationResult(
            inference=frames.get("inference"),
            samples=frames.get("samples"),
            metrics=frames.get("metrics"),
            metadata=dict(manifest.get("metadata") or {}),
        )

    def save_target(self, target_id: str, result) -> None:
        target_dir = self._target_dir(target_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(
            target_dir / "manifest.json",
            {
                "status": "running",
                "execution_hash": self.execution_hash,
                "target_id": target_id,
            },
        )
        for kind in result.result_kinds:
            self._save_target_frame(
                target_dir / f"{kind}.parquet", getattr(result, kind)
            )
        self._write_json_atomic(
            target_dir / "manifest.json",
            {
                "status": "complete",
                "execution_hash": self.execution_hash,
                "target_id": target_id,
                "result_kinds": result.result_kinds,
                "metadata": dict(result.metadata),
            },
        )
        self._update_node_manifest(completed_target_id=target_id)

    def combine(
        self,
        target_ids: Iterable[str],
        metadata_combiner: Callable[[Iterable[Mapping[str, Any]]], Mapping[str, Any]]
        | None = None,
    ):
        """Combine validated checkpoints for the engine's one final write."""
        from .result import EvaluationResult

        results = []
        for target_id in target_ids:
            result = self.load_target(target_id)
            if result is None:
                raise ValueError(
                    f"Evaluator '{self.node_id}' is missing completed target "
                    f"'{target_id}'."
                )
            results.append(result)
        return EvaluationResult(
            inference=self._concat_kind(results, "inference"),
            samples=self._concat_kind(results, "samples"),
            metrics=self._concat_kind(results, "metrics"),
            metadata=(
                metadata_combiner(result.metadata for result in results)
                if metadata_combiner is not None
                else {}
            ),
        )

    def materialize(self) -> None:
        results = self._completed_results()
        if not results:
            return
        for kind in ("inference", "samples", "metrics"):
            frame = self._concat_kind(results, kind)
            if frame is not None:
                self.store._save_frame(self.node_id, kind, frame)

    def completed_target_ids(self) -> list[str]:
        return [
            result[0]
            for result in self._completed_target_entries()
        ]

    def expected_target_ids(self) -> list[str]:
        """Return the targets recorded when this execution first started.

        The durable list records which targets belong to the current evaluator
        execution. An empty list means target scheduling has not started.
        """
        manifest = self.store.manifest(self.node_id) or {}
        if manifest.get("execution_hash") != self.execution_hash:
            return []
        progress = dict((manifest.get("metadata") or {}).get("progress") or {})
        return [str(value) for value in progress.get("expected_target_ids") or ()]

    def _completed_results(self) -> list[Any]:
        return [entry[1] for entry in self._completed_target_entries()]

    def _completed_target_entries(self) -> list[tuple[str, Any]]:
        entries = []
        for target_dir in sorted(self.root.iterdir() if self.root.exists() else []):
            if not target_dir.is_dir():
                continue
            manifest = self._read_json(target_dir / "manifest.json")
            target_id = (manifest or {}).get("target_id")
            if not target_id:
                continue
            result = self.load_target(target_id)
            if result is not None:
                entries.append((target_id, result))
        return sorted(entries, key=lambda item: item[0])

    def _update_node_manifest(
        self,
        expected_target_ids: Iterable[str] | None = None,
        completed_target_id: str | None = None,
    ) -> None:
        manifest = self.store.manifest(self.node_id)
        if not manifest or manifest.get("execution_hash") != self.execution_hash:
            raise ValueError(
                f"Evaluator '{self.node_id}' progress no longer matches its manifest."
            )
        metadata = dict(manifest.get("metadata") or {})
        previous = dict(metadata.get("progress") or {})
        expected = (
            list(expected_target_ids)
            if expected_target_ids is not None
            else list(previous.get("expected_target_ids") or [])
        )
        if expected_target_ids is not None:
            # Recovery validates every checkpoint once.  A target whose manifest
            # or parquet frames were interrupted/corrupted is deliberately not
            # advertised as complete and will be evaluated again.
            completed = self.completed_target_ids()
        else:
            # save_target has just atomically committed this target.  Reusing the
            # recovered manifest state avoids rereading every prior target after
            # every checkpoint (quadratic parquet I/O).
            completed = list(previous.get("completed_target_ids") or [])
            if completed_target_id is not None:
                completed.append(completed_target_id)
            completed = sorted(set(completed))
        progress = {
            "completed_targets": len(completed),
            "total_targets": len(expected) if expected else None,
            "completed_target_ids": completed,
            "expected_target_ids": expected,
        }
        if self.store.mirrored_progress:
            self.store._write_progress_metadata(
                self.node_id, self.execution_hash, progress
            )
        else:
            metadata["progress"] = progress
            manifest["metadata"] = metadata
            self.store._write_manifest(self.node_id, manifest)

    def _target_dir(self, target_id: str) -> Path:
        digest = hashlib.sha256(target_id.encode("utf-8")).hexdigest()[:24]
        return self.root / digest

    def _target_manifest(self, target_id: str) -> dict[str, Any] | None:
        return self._read_json(self._target_dir(target_id) / "manifest.json")

    def _valid_target_manifest(
        self, manifest: Mapping[str, Any] | None, target_id: str
    ) -> bool:
        return bool(
            manifest
            and manifest.get("status") == "complete"
            and manifest.get("execution_hash") == self.execution_hash
            and manifest.get("target_id") == target_id
            and manifest.get("result_kinds")
        )

    @staticmethod
    def _concat_kind(results: Iterable[Any], kind: str) -> pd.DataFrame | None:
        frames = [
            getattr(result, kind)
            for result in results
            if getattr(result, kind) is not None
        ]
        return pd.concat(frames, ignore_index=True) if frames else None

    @staticmethod
    def _save_target_frame(path: Path, data: pd.DataFrame) -> None:
        with tempfile.NamedTemporaryFile(
            delete=False, dir=path.parent, suffix=".parquet.tmp"
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            data.to_parquet(temporary_path, index=False)
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as file:
                return json.load(file)
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, dir=path.parent, suffix=".json.tmp", encoding="utf-8"
        ) as temporary:
            json.dump(payload, temporary, indent=2, sort_keys=True, default=str)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)


class ResultView:
    def __init__(self, store: ResultStore, dependencies: tuple[str, ...]):
        self._store = store
        self.dependencies = dependencies

    def query(
        self,
        node_id: str,
        kind: str = "metrics",
        filters: Mapping[str, Any] | None = None,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        self._check_dependency(node_id)
        return self._store.query(node_id, kind=kind, filters=filters, columns=columns)

    def inference(self, node_id: str, **kwargs: Any) -> pd.DataFrame:
        return self.query(node_id, kind="inference", **kwargs)

    def samples(self, node_id: str, **kwargs: Any) -> pd.DataFrame:
        return self.query(node_id, kind="samples", **kwargs)

    def metrics(self, node_id: str, **kwargs: Any) -> pd.DataFrame:
        return self.query(node_id, kind="metrics", **kwargs)

    def manifest(self, node_id: str) -> dict[str, Any] | None:
        self._check_dependency(node_id)
        return self._store.manifest(node_id)

    def _check_dependency(self, node_id: str) -> None:
        if node_id not in self.dependencies:
            raise ValueError(
                f"Evaluator '{node_id}' is not a declared dependency. "
                f"Available dependencies: {list(self.dependencies)}"
            )
