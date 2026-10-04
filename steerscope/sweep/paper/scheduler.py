#!/usr/bin/env python3
"""Run one complete SteerScope paper sweep."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import html
import json
import logging
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

import yaml
from steerscope.utils.api_clients import apply_api_model_overrides
from tqdm.auto import tqdm

from steerscope.utils.training_seed import TRAINING_RECIPE_VERSION
from steerscope.utils.concept_scope import select_concept_ids
from steerscope.inference.easysteer import EASYSTEER_METHODS

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
STATE_VERSION = 2
TASK_SIGNATURE_SCHEMA = 2
EXCLUDED_YAMLS = {"generalization.yaml", "study.yaml"}
DEPENDENCY_BLOCKED_ERROR = "blocked by failed dependency"
LOGGER = logging.getLogger("steerscope.paper.scheduler")
LOGGER.addHandler(logging.NullHandler())
LOGGER.propagate = False

# Previous factor grids validate only allowlisted cached study receipts.
_PRE_SPSR_STUDY_FACTORS = {
    "SphericalSteering": [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    "HiDRA": [0.5, 1.0, 1.2, 1.5, 2.0],
    "AUSteer": [2.5, 5.0, 7.5, 10.0, 15.0],
    "ODESteer": [1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0],
    "StepODESteer": [
        1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0,
    ],
}
_FACTOR_GRID_MIGRATION_METHODS = frozenset(_PRE_SPSR_STUDY_FACTORS)


def configure_scheduler_logging(output_dir: Path) -> Path:
    """Log scheduler control-plane events to both the terminal and disk."""
    log_path = output_dir / "logs/scheduler.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # main() configures logging once, but replacing handlers makes repeated
    # in-process invocations (for example from a test harness) deterministic.
    for handler in list(LOGGER.handlers):
        LOGGER.removeHandler(handler)
        if not isinstance(handler, logging.NullHandler):
            handler.close()

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03dZ %(levelname)s "
        "[scheduler pid=%(process)d] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)
    LOGGER.setLevel(logging.INFO)
    return log_path


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = yaml.safe_load(file) or {}
    if not isinstance(value, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return apply_api_model_overrides(value)


def atomic_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        yaml.safe_dump(value, file, sort_keys=False, allow_unicode=True)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _training_task_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only fields that can change a training artifact."""
    evaluate = config.get("evaluate") or {}
    return {
        "generate": copy.deepcopy(config.get("generate") or {}),
        "train": copy.deepcopy(config.get("train") or {}),
        "evaluation_models": [
            str(method) for method in (evaluate.get("models") or ())
        ],
    }


def _study_source_for_method(
    source: Mapping[str, Any], method: str
) -> dict[str, Any]:
    """Return the part of a shared study config that can affect one method."""
    scoped = copy.deepcopy(dict(source))
    train = scoped.setdefault("train", {})
    recipes = train.get("models") or {}
    recipe = copy.deepcopy(recipes.get(method))
    train["models"] = {method: recipe} if recipe is not None else {}

    evaluate = scoped.setdefault("evaluate", {})
    evaluate["models"] = [method]
    for node in (evaluate.get("evaluators") or {}).values():
        if not isinstance(node, dict):
            continue
        if "models" in node:
            node["models"] = [method]
        inference = node.get("inference")
        if not isinstance(inference, dict):
            continue
        strengths = inference.get("strengths_by_model")
        if isinstance(strengths, Mapping):
            inference["strengths_by_model"] = {
                method: copy.deepcopy(strengths[method])
            } if method in strengths else {}
    return scoped


def _study_evaluation_source_for_method(
    source: Mapping[str, Any], method: str, variant: Any
) -> dict[str, Any]:
    """Build the method/variant-scoped source used by evaluation receipts."""
    from steerscope.studies.training_data import apply_variant_inference_settings

    scoped = _study_source_for_method(source, method)
    study = scoped.pop("study", None) or {}
    apply_variant_inference_settings(scoped, variant, study)
    return {
        "config": scoped,
        "result": copy.deepcopy(study.get("result") or {}),
    }


def _pre_spsr_study_source(
    source: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Reconstruct the exact shared source used by pre-expansion receipts."""
    evaluate = source.get("evaluate") or {}
    if "SPSR" not in (evaluate.get("models") or ()):
        return None
    legacy = copy.deepcopy(dict(source))
    train = legacy.get("train") or {}
    (train.get("models") or {}).pop("SPSR", None)
    legacy_evaluate = legacy.get("evaluate") or {}
    legacy_evaluate["models"] = [
        method for method in (legacy_evaluate.get("models") or ())
        if method != "SPSR"
    ]
    for node in (legacy_evaluate.get("evaluators") or {}).values():
        if not isinstance(node, dict):
            continue
        if "models" in node:
            node["models"] = [
                method for method in (node.get("models") or ())
                if method != "SPSR"
            ]
        inference = node.get("inference")
        if not isinstance(inference, dict):
            continue
        strengths = inference.get("strengths_by_model")
        if not isinstance(strengths, dict):
            continue
        strengths.pop("SPSR", None)
        for method, factors in _PRE_SPSR_STUDY_FACTORS.items():
            if method in strengths:
                strengths[method] = list(factors)
    return legacy


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generated_data_signature(generate_dir: Path) -> str:
    required = (
        generate_dir / "metadata.jsonl",
        generate_dir / "train_data.parquet",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Generated data signature is missing files: {missing}."
        )
    return canonical_hash({path.name: _sha256_file(path) for path in required})


def _configured_training_concept_count(config: Mapping[str, Any]) -> int:
    """Return the concept count owned by this method's training artifact."""
    train = config.get("train") or {}
    concept_ids = train.get("concept_ids")
    if concept_ids is not None:
        return len({int(value) for value in concept_ids})
    for key in ("num_concepts", "max_concepts"):
        value = train.get(key)
        if value is not None:
            return int(value)
    return int((config.get("generate") or {}).get("max_concepts") or 1)


def _training_concept_scope(config: Mapping[str, Any]) -> dict[str, int] | None:
    """Return a method-level random concept panel declaration, if configured."""
    train = config.get("train") or {}
    count = train.get("num_concepts")
    if count is None:
        return None
    return {
        "count": int(count),
        "seed": int(train.get("seed", 42)),
    }


def _generated_concept_metadata(generate_dir: Path) -> list[dict[str, Any]]:
    path = Path(generate_dir) / "metadata.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _scope_evaluators_to_training_panel(
    config: dict[str, Any],
    *,
    generate_dir: Path,
    count: int,
    seed: int,
) -> tuple[int, ...]:
    """Intersect evaluator-specific filters with the method's trained concept panel."""
    metadata = _generated_concept_metadata(generate_dir)
    all_ids = [int(item["concept_id"]) for item in metadata]
    panel = select_concept_ids(all_ids, count=int(count), seed=int(seed))
    panel_set = set(panel)
    genres_by_id: dict[int, set[str]] = {}
    for item in metadata:
        concept_id = int(item["concept_id"])
        concept = str(item.get("concept"))
        mapping = item.get("concept_genres_map")
        if isinstance(mapping, Mapping) and concept in mapping:
            genres = mapping[concept]
            if isinstance(genres, str):
                genres = [genres]
            genres_by_id[concept_id] = {
                str(value).strip().lower()
                for value in genres
                if str(value).strip()
            }

    evaluators = (config.get("evaluate") or {}).get("evaluators") or {}
    remove: set[str] = set()
    for node_id, node in evaluators.items():
        if not isinstance(node, dict):
            continue
        scope = dict(node.get("concepts") or {})
        node_universe = list(all_ids)
        configured_genres = scope.get("genres")
        if configured_genres is not None:
            if isinstance(configured_genres, str):
                configured_genres = [configured_genres]
            required = {
                str(value).strip().lower()
                for value in configured_genres
                if str(value).strip()
            }
            node_universe = [
                concept_id for concept_id in node_universe
                if required.intersection(genres_by_id.get(concept_id, set()))
            ]
        configured_ids = scope.get("ids")
        if configured_ids is not None:
            allowed = {int(value) for value in configured_ids}
            node_universe = [value for value in node_universe if value in allowed]
        configured_count = scope.get("count")
        if configured_count is not None:
            node_universe = select_concept_ids(
                node_universe,
                count=int(configured_count),
                seed=int(scope.get("seed", 42)),
            )
        selected = sorted(panel_set.intersection(node_universe))
        if not selected:
            remove.add(str(node_id))
            continue
        materialized = {
            key: copy.deepcopy(value)
            for key, value in scope.items()
            if key not in {"ids", "count", "seed"}
        }
        materialized["ids"] = selected
        node["concepts"] = materialized

    changed = True
    while changed:
        changed = False
        for node_id, node in evaluators.items():
            if node_id in remove or not isinstance(node, dict):
                continue
            if set(node.get("depends_on") or ()).intersection(remove):
                remove.add(str(node_id))
                changed = True
    for node_id in remove:
        evaluators.pop(node_id, None)
    return tuple(panel)


def _run_checked(command: list[str], *, cwd: Path) -> None:
    print("Running:", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, check=False)
    result.check_returncode()


def _configure_fixed_factor_self_baseline(
    config: dict[str, Any],
    *,
    scan_id: str,
    method: str,
    factor: float,
) -> None:
    """Use LMJudge's base-model inference for one fixed-factor method."""
    evaluate = config.setdefault("evaluate", {})
    evaluators = evaluate.get("evaluators") or {}
    scan = evaluators.get(scan_id)
    if not isinstance(scan, dict):
        raise ValueError(f"Missing fixed-factor LM judge node '{scan_id}'.")

    baseline_factors = set()
    for node in evaluators.values():
        if not isinstance(node, dict) or node.get("type") != "BestFactorEvaluator":
            continue
        source = (node.get("input") or {}).get("from")
        if source is None:
            dependencies = node.get("depends_on") or []
            source = dependencies[0] if len(dependencies) == 1 else None
        if source != scan_id:
            continue
        params = node.setdefault("params", {})
        baseline_factor = params.get("baseline_factor")
        if baseline_factor is not None:
            baseline_factors.add(float(baseline_factor))
        params.pop("fallback_baseline_method", None)
    if len(baseline_factors) > 1:
        raise ValueError(
            f"Fixed-factor selectors for '{scan_id}' disagree on baseline: "
            f"{sorted(baseline_factors)}."
        )
    baseline_factor = next(iter(baseline_factors), 0.0)
    if float(factor) == baseline_factor:
        raise ValueError("Fixed method factor must differ from its baseline factor.")

    evaluate["models"] = [method]
    scan["models"] = [method]
    inference = scan.setdefault("inference", {})
    inference.pop("strengths", None)
    inference.pop("factors", None)
    inference["strengths_by_model"] = {method: [float(factor)]}
    params = scan.setdefault("params", {})
    params["include_baseline"] = True
    params["baseline_factor"] = baseline_factor


def _validate_merged_metric_identities(
    frame: "pd.DataFrame", evaluator_id: str
) -> None:
    """Reject duplicate metric rows without collapsing evaluator dimensions."""
    if not {"method", "concept_id"}.issubset(frame.columns):
        return
    identity = ["method", "concept_id"]
    for dimension in ("factor", "scope"):
        if dimension in frame.columns:
            identity.append(dimension)
    if frame.duplicated(identity).any():
        raise RuntimeError(
            f"Merged {evaluator_id} has duplicate metric identities {identity}."
        )


def run_study_worker(job_path: Path) -> int:
    """Train and/or judge one method at one study size/seed.

    Study jobs are split so training and online judging can use independent
    scheduler slots.
    The default ``all`` stage preserves compatibility with older job files.
    """
    import pandas as pd

    from steerscope.studies.training_data import (
        StudyVariant,
        derive_variant_config,
    )

    job = json.loads(job_path.read_text(encoding="utf-8"))
    stage = str(job.get("stage", "all"))
    if stage not in {"all", "train", "evaluate"}:
        raise ValueError(f"Unknown study worker stage: {stage!r}")
    method = str(job["method"])
    variant = StudyVariant(**job["variant"])
    source = load_yaml(Path(job["study_config"]))
    shared_generate = Path(job["shared_generate"]).resolve()
    output = Path(job["task_output"]).resolve()

    derived = derive_variant_config(source, variant, shared_generate)
    train_config = copy.deepcopy(derived)
    train = train_config.setdefault("train", {})
    # A method-owned scope (currently SFT's random 20-concept panel) takes
    # precedence over the Study-wide panel.  Hand-written/legacy jobs may omit
    # the explicit field, so derive the Study default from its source config.
    training_scope = (
        job.get("training_concept_scope")
        or _training_concept_scope(derived)
    )
    if training_scope is not None:
        train["num_concepts"] = int(training_scope["count"])
        train["seed"] = int(training_scope["seed"])
    recipes = train.get("models") or {}
    recipe = copy.deepcopy(recipes.get(method))
    train["models"] = {method: recipe} if recipe is not None else {}
    train_config.setdefault("evaluate", {})["models"] = [method]

    runtime_dir = output / "runtime"
    train_path = runtime_dir / "train.yaml"
    atomic_yaml(train_path, train_config)

    requires_training = recipe is not None or method == "GemmaScopeSAE"
    if stage in {"all", "train"} and requires_training:
        _run_checked([
            str(job["torchrun"]),
            "--standalone",
            f"--nproc_per_node={int(job['nproc_per_node'])}",
            str(REPO_ROOT / "steerscope/scripts/train.py"),
            "--config",
            str(train_path),
            "--dump_dir",
            str(output),
        ], cwd=REPO_ROOT)
    if stage == "train":
        return 0

    reference_path = Path(job["reference_metrics"]).resolve()
    reference = pd.read_parquet(reference_path)
    rows = reference[reference["method"].astype(str) == method]
    if len(rows) != 1 or "factor" not in rows:
        raise ValueError(
            f"Study reference needs one factor row for {method}; got {len(rows)}."
        )
    factor = float(rows.iloc[0]["factor"])

    evaluate_config = copy.deepcopy(derived)
    evaluate_train = evaluate_config.setdefault("train", {})
    if training_scope is not None:
        evaluate_train["num_concepts"] = int(training_scope["count"])
        evaluate_train["seed"] = int(training_scope["seed"])
    evaluate_train["models"] = (
        {method: copy.deepcopy(recipe)} if recipe is not None else {}
    )
    evaluate = evaluate_config.setdefault("evaluate", {})
    evaluate["models"] = [method]
    evaluate["generate_reports"] = False
    _apply_job_hot_paths(evaluate, job)
    evaluate["easysteer_url"] = str(job["easysteer_url"])
    evaluate["artifact_dirs_by_model"] = (
        {method: str((output / "train").resolve())}
        if requires_training
        else {}
    )
    evaluators = evaluate.get("evaluators") or {}
    selector_id = str(
        ((source.get("study") or {}).get("result") or {}).get(
            "evaluator", "best_factor"
        )
    )
    selector = evaluators.get(selector_id)
    if not isinstance(selector, dict):
        raise ValueError(f"Study selector '{selector_id}' is missing.")
    scan_id = str(
        (selector.get("input") or {}).get("from")
        or (selector.get("depends_on") or ["id_lm_judge"])[0]
    )
    _configure_fixed_factor_self_baseline(
        evaluate_config,
        scan_id=scan_id,
        method=method,
        factor=factor,
    )
    evaluate["evaluators"] = {
        node_id: node
        for node_id, node in evaluators.items()
        if node_id in {scan_id, selector_id}
    }
    scan = evaluate["evaluators"][scan_id]
    scan.setdefault("params", {})["judge_concurrency"] = int(
        job.get("judge_concurrency", 8)
    )
    if training_scope is not None:
        _scope_evaluators_to_training_panel(
            evaluate_config,
            generate_dir=shared_generate,
            count=int(training_scope["count"]),
            seed=int(training_scope["seed"]),
        )

    evaluate_path = runtime_dir / "evaluate.yaml"
    atomic_yaml(evaluate_path, evaluate_config)

    _run_checked([
        str(job["python"]),
        str(REPO_ROOT / "steerscope/scripts/evaluate.py"),
        "--config",
        str(evaluate_path),
        "--mode",
        "steering",
        "--dump_dir",
        str(output),
    ], cwd=REPO_ROOT)
    return 0


def _external_study_reference(
    raw_metrics,
    *,
    method: str,
    evaluator_id: str,
    train_examples: int,
    subset_seed: int,
):
    selected = raw_metrics[raw_metrics["method"].astype(str) == method].copy()
    if len(selected) != 1:
        raise ValueError(
            f"External study reference needs one {method} row, got "
            f"{len(selected)}."
        )
    selected.insert(0, "source_evaluator", evaluator_id)
    selected.insert(0, "sensitivity", False)
    selected.insert(0, "efficiency", True)
    selected.insert(0, "factor_reference", True)
    selected.insert(0, "subset_seed", int(subset_seed))
    selected.insert(0, "train_examples", int(train_examples))
    selected.insert(0, "study_run_id", "external-main-reference")
    return selected


def _atomic_parquet(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def wandb_launch_config(
    config: Mapping[str, Any],
    default_name: str,
    launched_at: dt.datetime | None = None,
) -> dict[str, Any]:
    """Give every scheduler process its own timestamped W&B run."""
    launched_at = launched_at or dt.datetime.now(dt.UTC)
    launched_at = launched_at.astimezone(dt.UTC)
    timestamp = (
        launched_at.strftime("%Y%m%d-%H%M%S-")
        + f"{launched_at.microsecond // 1000:03d}Z"
    )
    resolved = dict(config or {})
    base_name = str(resolved.get("name") or default_name)
    base_id = str(resolved.get("run_id") or base_name)
    resolved["name"] = f"{base_name}-{timestamp}"
    resolved["run_id"] = f"{base_id}-{timestamp}"
    # Local scheduler state handles job resumption. A new scheduler process
    # should never append low step numbers to an older W&B run.
    resolved["resume"] = "never"
    return resolved


def resolve_repo_path(value: str | Path | None, default: Path | None = None) -> Path:
    if value is None:
        if default is None:
            raise ValueError("A required path is missing.")
        return default.resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def default_cache_root() -> Path:
    """Return the user-configurable local cache root."""
    configured = os.environ.get("STEERSCOPE_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return (Path(xdg_cache).expanduser() / "steerscope").resolve()
    return (Path.home() / ".cache" / "steerscope").resolve()


def hot_data_namespace(
    output_dir: Path,
    configured: str | Path | None = None,
) -> Path:
    """Return a stable relative namespace for one scheduler output root."""
    if configured is not None:
        namespace = Path(configured)
        if namespace.is_absolute() or ".." in namespace.parts:
            raise ValueError(
                "runtime.hot_data_namespace must be a relative path without '..'."
            )
        return namespace
    canonical_output = (REPO_ROOT / "outputs").resolve()
    try:
        return output_dir.resolve().relative_to(canonical_output)
    except ValueError:
        digest = hashlib.sha256(
            str(output_dir.resolve()).encode("utf-8")
        ).hexdigest()[:12]
        return Path("custom") / f"{output_dir.name}-{digest}"


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _evaluator_components(
    evaluators: Mapping[str, Any],
) -> tuple[tuple[str, ...], ...]:
    """Return dependency-connected evaluator components in configuration order."""
    node_ids = tuple(str(value) for value in evaluators)
    known = set(node_ids)
    adjacency = {node_id: set() for node_id in node_ids}
    for node_id, raw_node in evaluators.items():
        node = raw_node if isinstance(raw_node, Mapping) else {}
        dependencies = node.get("depends_on", node.get("dependencies", ())) or ()
        if isinstance(dependencies, str):
            dependencies = (dependencies,)
        missing = [str(value) for value in dependencies if str(value) not in known]
        if missing:
            raise ValueError(
                f"Evaluator '{node_id}' has unknown dependencies: {missing}"
            )
        for dependency in dependencies:
            dependency = str(dependency)
            adjacency[str(node_id)].add(dependency)
            adjacency[dependency].add(str(node_id))

    remaining = set(node_ids)
    components = []
    for first in node_ids:
        if first not in remaining:
            continue
        pending = [first]
        connected = set()
        while pending:
            current = pending.pop()
            if current in connected:
                continue
            connected.add(current)
            pending.extend(adjacency[current].difference(connected))
        remaining.difference_update(connected)
        components.append(tuple(value for value in node_ids if value in connected))
    return tuple(components)


def _apply_job_hot_paths(
    evaluate: dict[str, Any], job: Mapping[str, Any]
) -> None:
    if job.get("shared_request_cache_dir"):
        evaluate["shared_request_cache_dir"] = str(
            Path(job["shared_request_cache_dir"]).resolve()
        )
    if job.get("progress_root"):
        evaluate["progress_root"] = str(Path(job["progress_root"]).resolve())
    if job.get("progress_source_root"):
        evaluate["progress_source_root"] = str(
            Path(job["progress_source_root"]).resolve()
        )


def executable(value: str | None, fallback: Path) -> str:
    path = Path(os.path.expandvars(value)).expanduser() if value else fallback
    if not path.is_absolute():
        candidate = REPO_ROOT / path
        if candidate.exists():
            path = candidate
        else:
            discovered = shutil.which(str(path))
            if discovered is not None:
                path = Path(discovered)
    if not path.exists():
        raise FileNotFoundError(f"Executable not found: {path}")
    # Do not resolve venv/Conda symlinks: invoking the real system interpreter
    # path would lose the selected environment's sys.prefix and packages.
    return str(path.absolute())


def easysteer_executable(value: str | None) -> str:
    """Locate vLLM without tying scheduler configs to one machine."""
    if value:
        return executable(value, Path(value))

    configured = os.environ.get("EASYSTEER_VLLM")
    if configured:
        return executable(configured, Path(configured))

    discovered = shutil.which("vllm")
    if discovered:
        return executable(discovered, Path(discovered))

    # The two Conda environments normally share an ``envs`` directory.  Keep
    # this inference after the explicit environment variable/PATH so unusual
    # installations can override it cleanly.
    python_path = Path(sys.executable).absolute()
    candidates: list[Path] = []
    if len(python_path.parents) >= 3:
        candidates.append(
            python_path.parents[2] / "easysteer" / "bin" / "vllm"
        )
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        conda_root = Path(conda_exe).expanduser().absolute().parent.parent
        candidates.append(conda_root / "envs/easysteer/bin/vllm")
    candidates.extend(
        [
            Path.home() / ".conda/envs/easysteer/bin/vllm",
            Path.home() / "miniconda3/envs/easysteer/bin/vllm",
            *(
                ancestor / "miniconda3/envs/easysteer/bin/vllm"
                for ancestor in REPO_ROOT.parents
            ),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return executable(str(candidate), candidate)

    raise FileNotFoundError(
        "EasySteer vLLM executable not found. Activate/expose the easysteer "
        "environment or set EASYSTEER_VLLM to its vllm executable, e.g. "
        "export EASYSTEER_VLLM=$(conda run -n easysteer which vllm)."
    )


@dataclass
class MethodTask:
    stem: str
    method: str
    source_config: Path
    runtime_config: Path
    output_dir: Path
    train_models: tuple[str, ...]
    evaluator_ids: tuple[str, ...]
    gpu_count: int
    nproc_per_node: int
    vram_gb: float
    exclusive: bool
    priority: int
    max_retries: int
    evaluator_workers: int = 1
    config_signature: str = ""
    signature: str = ""
    status: str = "queued"
    phase: str = "queued"
    attempts: int = 0
    process: subprocess.Popen | None = None
    log_file: Any = None
    allocated_gpus: tuple[int, ...] = field(default_factory=tuple)
    progress: float = 0.0
    error: str | None = None

    @property
    def train_dir(self) -> Path:
        return self.output_dir / "train"

    @property
    def requires_training(self) -> bool:
        # GemmaScopeSAE has no learned model entry, but train.py materializes
        # its selected SAE and calibration scale from metadata.
        return bool(self.train_models) or self.method == "GemmaScopeSAE"


@dataclass
class PipelineTask:
    """One resumable node in the two-phase train/evaluate/study DAG."""

    stem: str
    kind: str
    method: str | None
    command: tuple[str, ...]
    output_dir: Path
    evaluator_ids: tuple[str, ...] = field(default_factory=tuple)
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    gpu_count: int = 1
    vram_gb: float = 16.0
    exclusive: bool = False
    priority: int = 0
    max_retries: int = 0
    uses_api: bool = False
    signature: str = ""
    compatible_signatures: tuple[str, ...] = field(default_factory=tuple)
    accept_legacy_signature: bool = False
    receipt_path: Path | None = None
    required_artifacts: tuple[Path, ...] = field(default_factory=tuple)
    status: str = "queued"
    phase: str = "queued"
    attempts: int = 0
    process: subprocess.Popen | None = None
    log_file: Any = None
    allocated_gpus: tuple[int, ...] = field(default_factory=tuple)
    progress: float = 0.0
    error: str | None = None
    config_path: Path | None = None
    run_id: str = ""
    training_concepts: int = 0
    concurrency_key: str = ""
    concurrency_limit: int = 0


@dataclass(frozen=True)
class GPUInfo:
    index: int
    total_gb: float
    free_gb: float


class GPUAllocator:
    def __init__(
        self,
        gpu_ids: list[int],
        safety_margin_gb: float,
        reserve_initial_usage: bool = False,
    ):
        self.gpu_ids = tuple(int(value) for value in gpu_ids)
        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("workers.gpus contains duplicate GPU IDs.")
        self.safety_margin_gb = float(safety_margin_gb)
        # Reserve memory owned by persistent services before scheduling workers.
        self.reserve_initial_usage = bool(reserve_initial_usage)
        self.initial_used_gb: dict[int, float] | None = None
        self.allocations: dict[str, tuple[tuple[int, ...], float, bool]] = {}

    @staticmethod
    def query() -> dict[int, GPUInfo]:
        command = [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        infos = {}
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            index, total_mb, free_mb = [part.strip() for part in line.split(",")]
            info = GPUInfo(
                index=int(index),
                total_gb=float(total_mb) / 1024.0,
                free_gb=float(free_mb) / 1024.0,
            )
            infos[info.index] = info
        return infos

    def validate(self) -> None:
        available = self.query()
        self._capture_initial_usage(available)
        missing = sorted(set(self.gpu_ids).difference(available))
        if missing:
            raise ValueError(f"Configured worker GPUs do not exist: {missing}")

    def _capture_initial_usage(self, infos: Mapping[int, GPUInfo]) -> None:
        if self.initial_used_gb is not None:
            return
        self.initial_used_gb = {
            gpu: (
                max(0.0, infos[gpu].total_gb - infos[gpu].free_gb)
                if self.reserve_initial_usage and gpu in infos
                else 0.0
            )
            for gpu in self.gpu_ids
        }
        if self.reserve_initial_usage:
            LOGGER.info(
                "GPU allocator reserved initial persistent usage GiB=%s",
                {
                    gpu: round(value, 2)
                    for gpu, value in self.initial_used_gb.items()
                    if value >= 0.01
                },
            )

    def _persistent_used(self, gpu: int) -> float:
        return float((self.initial_used_gb or {}).get(gpu, 0.0))

    def validate_tasks(self, tasks: list[MethodTask | PipelineTask]) -> None:
        infos = self.query()
        self._capture_initial_usage(infos)
        for task in tasks:
            eligible = [
                gpu
                for gpu in self.gpu_ids
                if (
                    infos[gpu].total_gb
                    - self._persistent_used(gpu)
                    >= task.vram_gb + self.safety_margin_gb
                )
            ]
            if len(eligible) < task.gpu_count:
                raise ValueError(
                    f"Task '{task.stem}' requests {task.gpu_count} GPU(s) with "
                    f"{task.vram_gb:g} GB each plus a "
                    f"{self.safety_margin_gb:g} GB safety margin, but only "
                    f"{len(eligible)} configured worker GPU(s) can fit it."
                )

    def _booked(self, gpu: int) -> tuple[float, bool, int]:
        reservations = [
            (vram, exclusive)
            for gpus, vram, exclusive in self.allocations.values()
            if gpu in gpus
        ]
        return (
            sum(vram for vram, _ in reservations),
            any(exclusive for _, exclusive in reservations),
            len(reservations),
        )

    def allocate(
        self, task: MethodTask | PipelineTask
    ) -> tuple[int, ...] | None:
        infos = self.query()
        self._capture_initial_usage(infos)
        candidates = []
        for gpu in self.gpu_ids:
            info = infos[gpu]
            booked, has_exclusive, jobs = self._booked(gpu)
            if has_exclusive or (task.exclusive and jobs):
                continue
            reservation_capacity = (
                info.total_gb
                - self._persistent_used(gpu)
                - self.safety_margin_gb
            )
            if booked + task.vram_gb > reservation_capacity:
                continue
            if info.free_gb < task.vram_gb + self.safety_margin_gb:
                continue
            # Prefer the device with the greatest conservative headroom.
            # The first term observes real allocations; the second protects
            # reservations that launched processes have not consumed yet.
            headroom = min(
                info.free_gb - self.safety_margin_gb,
                reservation_capacity - booked,
            )
            candidates.append((gpu, headroom))
        candidates.sort(key=lambda item: item[1], reverse=True)
        if len(candidates) < task.gpu_count:
            return None
        selected = tuple(gpu for gpu, _ in candidates[: task.gpu_count])
        self.allocations[task.stem] = (
            selected,
            task.vram_gb,
            task.exclusive,
        )
        return selected

    def release(self, task: MethodTask | PipelineTask) -> None:
        self.allocations.pop(task.stem, None)

    def release_persistent_reservations(self, gpu_ids: Sequence[int]) -> None:
        """Release the reservation ledger after a persistent service exits."""
        if self.initial_used_gb is None:
            return
        released = {}
        for gpu in gpu_ids:
            gpu = int(gpu)
            reserved = float(self.initial_used_gb.get(gpu, 0.0))
            if reserved > 0:
                released[gpu] = round(reserved, 2)
            self.initial_used_gb[gpu] = 0.0
        if released:
            LOGGER.info(
                "GPU allocator released persistent reservations GiB=%s",
                released,
            )


class ProgressReporter:
    def __init__(self, config: Mapping[str, Any], output_dir: Path, run_config):
        self.config = dict(config or {})
        self.output_dir = output_dir
        self.last_html = 0.0
        self.step = 0
        self.run = None
        self.bar = tqdm(
            total=1000,
            desc="SteerScope sweep",
            unit="‰",
            dynamic_ncols=True,
            disable=not parse_bool(self.config.get("terminal"), True),
        )
        if not parse_bool(self.config.get("enabled"), False):
            return
        try:
            import wandb

            wandb_dir = output_dir / "wandb"
            wandb_dir.mkdir(parents=True, exist_ok=True)
            mode = self.config.get("mode")
            if mode:
                os.environ["WANDB_MODE"] = str(mode)
            self.run = wandb.init(
                project=self.config.get("project", "steerscope-paper"),
                entity=self.config.get("entity"),
                name=self.config.get("name"),
                group=self.config.get("group"),
                tags=list(self.config.get("tags") or []),
                dir=str(wandb_dir),
                config=run_config,
                resume=self.config.get("resume", "never"),
                id=self.config.get("run_id"),
            )
        except Exception as error:
            if parse_bool(self.config.get("required"), False):
                raise
            tqdm.write(f"W&B disabled after initialization error: {error}")
            self.run = None

    @staticmethod
    def _dashboard(progress: Mapping[str, float], methods, stage: str) -> str:
        rows = [
            ("Overall", progress.get("overall", 0.0)),
            ("Generate", progress.get("generate", 0.0)),
            ("Method YAMLs", progress.get("methods", 0.0)),
            ("Study", progress.get("study", 0.0)),
            ("Generalization", progress.get("generalization", 0.0)),
        ]
        rows.extend(
            (
                (
                    f"{task.stem} [{task.status}/{task.phase}; GPU "
                    f"{','.join(map(str, task.allocated_gpus)) or '-'}]"
                ),
                task.progress,
            )
            for task in methods
        )
        bars = []
        for label, value in rows:
            percent = max(0.0, min(100.0, float(value) * 100.0))
            bars.append(
                "<div style='margin:7px 0'><div style='display:flex;"
                "justify-content:space-between'><span>"
                f"{html.escape(label)}</span><span>{percent:.1f}%</span></div>"
                "<div style='height:14px;background:#e5e7eb;border-radius:7px;"
                "overflow:hidden'><div style='height:100%;background:#2563eb;"
                f"width:{percent:.3f}%'></div></div></div>"
            )
        return (
            "<div style='font-family:system-ui;max-width:900px'>"
            f"<h3>Current stage: {html.escape(stage)}</h3>{''.join(bars)}</div>"
        )

    def update(
        self, progress, methods, stage, counts, statuses, force_html=False
    ) -> None:
        overall = max(0.0, min(1.0, float(progress.get("overall", 0.0))))
        desired = max(self.bar.n, round(overall * 1000))
        self.bar.update(desired - self.bar.n)
        self.bar.set_postfix_str(
            f"{stage} | yaml {counts['complete']}/{counts['total']} "
            f"running={counts['running']} "
            f"failed={counts['failed']} "
            f"study={statuses['study']}({statuses.get('study_attempts', 0)}) "
            f"gen={statuses['generalization']}"
        )
        self.bar.refresh()
        if self.run is None:
            return
        import wandb
        payload = {
            "progress/overall": overall,
            "progress/generate": progress.get("generate", 0.0),
            "progress/method_yamls": progress.get("methods", 0.0),
            "progress/study": progress.get("study", 0.0),
            "progress/generalization": progress.get("generalization", 0.0),
            "counts/method_total": counts["total"],
            "counts/method_complete": counts["complete"],
            "counts/method_running": counts["running"],
            "counts/method_failed": counts["failed"],
            "scheduler/current_stage": stage,
            "scheduler/generate_status": statuses["generate"],
            "scheduler/study_status": statuses["study"],
            "scheduler/study_attempts": statuses.get("study_attempts", 0),
            "scheduler/study_error": statuses.get("study_error") or "",
            "scheduler/generalization_status": statuses["generalization"],
            "scheduler/running_yamls": ",".join(
                task.stem for task in methods if task.status == "running"
            ),
        }
        for task in methods:
            payload[f"yaml_progress/{task.stem}"] = task.progress
        now = time.monotonic()
        html_interval = float(self.config.get("html_interval_seconds", 30))
        if force_html or now - self.last_html >= html_interval:
            payload["scheduler/progress_dashboard"] = wandb.Html(
                self._dashboard(progress, methods, stage)
            )
            self.last_html = now
        self.run.log(payload, step=self.step)
        self.step += 1
        self.run.summary["current_stage"] = stage
        self.run.summary["overall_progress"] = overall

    def finish(self, status: str, methods=()) -> None:
        self.bar.close()
        if self.run is not None:
            self.run.summary["scheduler_status"] = status
            for task in methods:
                self.run.summary[f"yaml_status/{task.stem}"] = task.status
            self.run.finish(exit_code=0 if status == "complete" else 1)


class Scheduler:
    """Schedule one model/layer sweep through generation, training, evaluation, and data-dependence studies."""

    def __init__(self, config_path: Path, cli):
        self.config_path = Path(config_path).resolve()
        self.config = load_yaml(self.config_path)
        self.cli = cli
        experiment = self.config.get("experiment") or {}
        self.model_key = str(experiment.get("model_key") or "").strip()
        if not self.model_key:
            raise ValueError("experiment.model_key is required.")
        self.layer = int(experiment.get("layer"))
        self.source_dir = resolve_repo_path(
            experiment.get("config_dir"),
            SCRIPT_DIR / self.model_key / f"l{self.layer}",
        )
        self.output_dir = resolve_repo_path(
            experiment.get("output_dir"),
            REPO_ROOT / "outputs/paper" / self.model_key / f"l{self.layer}",
        )

        runtime = self.config.get("runtime") or {}
        self.python = executable(runtime.get("python"), Path(sys.executable))
        self.torchrun = executable(
            runtime.get("torchrun"), Path(self.python).with_name("torchrun")
        )
        self.hot_data_root = resolve_repo_path(
            runtime.get("hot_data_root"), default_cache_root()
        )
        self.hot_data_namespace = hot_data_namespace(
            self.output_dir, runtime.get("hot_data_namespace")
        )
        self.progress_root = (
            self.hot_data_root / "progress" / self.hot_data_namespace
        ).resolve()
        self.request_cache_dir = (
            self.hot_data_root / "request_cache" / self.hot_data_namespace
        ).resolve()
        self.lm_cache_dir = resolve_repo_path(
            runtime.get("lm_cache_dir"), self.hot_data_root / "lm_cache"
        )
        self.lm_cache_min_free_gb = float(
            runtime.get("lm_cache_min_free_gb", 10)
        )
        if self.lm_cache_min_free_gb < 0:
            raise ValueError("runtime.lm_cache_min_free_gb cannot be negative.")

        scheduler = self.config.get("scheduler") or {}
        self.poll_seconds = float(scheduler.get("poll_interval_seconds", 10))
        if self.poll_seconds <= 0:
            raise ValueError("scheduler.poll_interval_seconds must be positive.")
        self.max_concurrent_api_runs = int(
            scheduler.get("max_concurrent_api_runs", 8)
        )
        if self.max_concurrent_api_runs < 1:
            raise ValueError("scheduler.max_concurrent_api_runs must be positive.")
        self.resume = parse_bool(scheduler.get("resume"), True) and not cli.no_resume
        self.sample_sensitivity_enabled = parse_bool(
            (self.config.get("study") or {}).get(
                "sample_sensitivity_enabled"
            ),
            True,
        )
        study_runtime = self.config.get("study") or {}
        self.sensitivity_only = parse_bool(
            study_runtime.get("sensitivity_only"), False
        )
        self.study_config_path = resolve_repo_path(
            study_runtime.get("config"), self.source_dir / "study.yaml"
        )
        self.generate_dir = resolve_repo_path(
            study_runtime.get("shared_generate_dir"),
            self.output_dir / "generate",
        )
        reference_output = study_runtime.get("reference_output_dir")
        self.reference_output_dir = (
            resolve_repo_path(reference_output) if reference_output else None
        )
        reference_metrics = study_runtime.get("reference_metrics")
        self.reference_metrics_source = (
            resolve_repo_path(reference_metrics) if reference_metrics else None
        )
        self.study_reference_metrics_path: Path | None = None
        self.study_judge_concurrency = int(
            study_runtime.get("judge_concurrency", 8)
        )
        if self.study_judge_concurrency < 1:
            raise ValueError("study.judge_concurrency must be positive.")

        self.state_path = self.output_dir / "state.json"
        self.state = self._load_state()
        self.stage = str(self.state.get("stage") or "initialize")
        self.current_stage = self.stage
        self.generate_status = "queued"
        self.study_status = "disabled"
        self.generalization_status = "disabled"
        self.stop_requested = False
        self.received_signal: int | None = None
        self.service_process: subprocess.Popen | None = None
        self.service_log = None
        self.children: set[subprocess.Popen] = set()
        self.lock_file = None
        self.reporter: ProgressReporter | None = None

        self.methods: list[MethodTask] = []
        self.providers: list[MethodTask] = []
        self.tasks = self.methods  # ProgressReporter/backward-compatible alias.
        self.source_configs: dict[str, dict[str, Any]] = {}
        self.training_provider: dict[str, MethodTask] = {}
        self.pipeline_tasks: list[PipelineTask] = []
        self.pipeline_task_map: dict[str, PipelineTask] = {}
        self.runtime_configs_prepared = False
        self.generate_signature = ""
        self._discover()

    def _study_variants(self, source: Mapping[str, Any]):
        """Return study runs enabled by scheduler policy."""
        from steerscope.studies.training_data import build_study_variants

        variants = build_study_variants(source)
        if self.sensitivity_only:
            return [variant for variant in variants if variant.sensitivity]
        if self.sample_sensitivity_enabled:
            return variants
        return [
            variant for variant in variants
            if variant.efficiency or variant.factor_reference
        ]

    def _load_state(self) -> dict[str, Any]:
        if not self.resume or not self.state_path.is_file():
            return {"version": STATE_VERSION, "pipeline_tasks": {}}
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": STATE_VERSION, "pipeline_tasks": {}}
        if int(state.get("version", -1)) != STATE_VERSION:
            LOGGER.warning(
                "Ignoring scheduler state version=%s; expected=%s",
                state.get("version"), STATE_VERSION,
            )
            return {"version": STATE_VERSION, "pipeline_tasks": {}}
        return state

    def _save_state(self) -> None:
        if self.cli.dry_run:
            return
        payload = {
            "version": STATE_VERSION,
            "updated_at": utc_now(),
            "model_key": self.model_key,
            "layer": self.layer,
            "stage": self.stage,
            "generate": {
                "signature": self.generate_signature,
                "status": self.generate_status,
            },
            "pipeline_tasks": {
                task.stem: {
                    "kind": task.kind,
                    "method": task.method,
                    "signature": task.signature,
                    "signature_schema": TASK_SIGNATURE_SCHEMA,
                    "status": task.status,
                    "phase": task.phase,
                    "progress": task.progress,
                    "attempts": task.attempts,
                    "gpus": list(task.allocated_gpus),
                    "error": task.error,
                }
                for task in self.pipeline_tasks
            },
        }
        atomic_json(self.state_path, payload)

    def _discover(self) -> None:
        if not self.source_dir.is_dir():
            raise FileNotFoundError(
                f"Sweep config directory not found: {self.source_dir}"
            )
        selected = set(self.cli.methods or ())
        method_section = self.config.get("methods") or {}
        included = set(method_section.get("include") or ())
        excluded = set(method_section.get("exclude") or ())
        providers = set(method_section.get("providers") or ())
        defaults = dict(method_section.get("defaults") or {})
        overrides = dict(method_section.get("overrides") or {})
        for path in sorted(self.source_dir.glob("*.yaml")):
            if path.name in EXCLUDED_YAMLS:
                continue
            source = load_yaml(path)
            models = list((source.get("evaluate") or {}).get("models") or ())
            if len(models) != 1:
                raise ValueError(
                    f"Method YAML must identify one primary method: {path}"
                )
            method = str(models[0])
            stem = path.stem
            is_provider = stem in providers or method in providers
            if (
                selected
                and stem not in selected
                and method not in selected
                and not is_provider
            ):
                continue
            if (
                not selected
                and included
                and stem not in included
                and method not in included
                and not is_provider
            ):
                continue
            if stem in excluded or method in excluded:
                continue
            settings = {**defaults, **dict(overrides.get(method) or {})}
            settings.update(dict(overrides.get(stem) or {}))
            gpu_count = int(settings.get("gpu_count", 1))
            nproc = int(settings.get("nproc_per_node", gpu_count))
            evaluator_workers = int(settings.get("evaluator_workers", 1))
            if gpu_count < 1 or nproc < 1 or nproc > gpu_count:
                raise ValueError(
                    f"Invalid gpu_count/nproc_per_node for {stem}: "
                    f"{gpu_count}/{nproc}"
                )
            if evaluator_workers < 1:
                raise ValueError(
                    f"Invalid evaluator_workers for {stem}: "
                    f"{evaluator_workers}"
                )
            train_models = tuple((source.get("train") or {}).get("models") or ())
            method_task = MethodTask(
                stem=stem,
                method=method,
                source_config=path,
                runtime_config=(
                    self.output_dir / "runtime_configs/phase1/methods"
                    / stem / "evaluate.yaml"
                ),
                output_dir=self.output_dir / "methods" / stem,
                train_models=train_models,
                evaluator_ids=tuple(
                    (source.get("evaluate") or {}).get("evaluators") or ()
                ),
                gpu_count=gpu_count,
                nproc_per_node=nproc,
                vram_gb=float(settings.get("vram_reservation_gb", 16)),
                exclusive=parse_bool(settings.get("exclusive_gpu"), False),
                priority=int(settings.get("priority", 0)),
                max_retries=int(settings.get("max_retries", 0)),
                evaluator_workers=evaluator_workers,
            )
            (self.providers if is_provider else self.methods).append(method_task)
            self.source_configs[stem] = source
            for trained_method in train_models:
                if trained_method in self.training_provider:
                    raise ValueError(
                        f"Method {trained_method} has multiple training providers."
                    )
                self.training_provider[str(trained_method)] = method_task
        if not self.methods:
            raise ValueError("No method YAMLs matched the scheduler selection.")
        self.methods.sort(key=lambda task: (-task.priority, task.stem))
        self.providers.sort(key=lambda task: (-task.priority, task.stem))
        self._validate_sources()
        self._validate_easysteer()
        self._validate_gpu_pool()

    def _validate_sources(self) -> None:
        signatures = {
            canonical_hash(source.get("generate") or {})
            for source in self.source_configs.values()
        }
        required_shared = (
            ["generalization.yaml", "study.yaml"]
            if not self.sensitivity_only
            else []
        )
        for name in required_shared:
            path = self.source_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing sweep config: {path}")
            signatures.add(canonical_hash(load_yaml(path).get("generate") or {}))
        if not self.study_config_path.is_file():
            raise FileNotFoundError(
                f"Missing study config: {self.study_config_path}"
            )
        study = load_yaml(self.study_config_path)
        signatures.add(canonical_hash(study.get("generate") or {}))
        if len(signatures) != 1:
            raise ValueError("Every YAML must have the same generate section.")

        for method in self.methods:
            evaluators = (
                self.source_configs[method.stem].get("evaluate") or {}
            ).get("evaluators") or {}
            if "id_lm_judge" not in evaluators:
                raise ValueError(f"{method.source_config} lacks id_lm_judge.")
            if "prompt_generalization" in evaluators:
                raise ValueError(
                    f"Method YAML must not contain generalization: "
                    f"{method.source_config}"
                )

        if not self.sensitivity_only:
            generalization = load_yaml(self.source_dir / "generalization.yaml")
            gen_nodes = (generalization.get("evaluate") or {}).get("evaluators") or {}
            if "id_lm_judge" in gen_nodes:
                raise ValueError("generalization.yaml must not rescan id_lm_judge.")
            if set(gen_nodes) != {"best_factor", "prompt_generalization"}:
                raise ValueError(
                    "generalization.yaml must contain only best_factor and "
                    "prompt_generalization."
                )
        study_nodes = (study.get("evaluate") or {}).get("evaluators") or {}
        if "id_lm_judge" in study_nodes:
            raise ValueError("study.yaml must not contain id_lm_judge.")
        result_id = str(
            ((study.get("study") or {}).get("result") or {}).get("evaluator")
            or ""
        )
        if result_id not in study_nodes:
            raise ValueError("study.result.evaluator is missing from study.yaml.")
        if self.sensitivity_only:
            study_section = study.get("study") or {}
            if not parse_bool(
                (self.config.get("study") or {}).get("enabled"), True
            ):
                raise ValueError("sensitivity_only requires study.enabled=true.")
            if "sample_sensitivity" not in study_section:
                raise ValueError(
                    "sensitivity_only study config needs sample_sensitivity."
                )
            if not parse_bool(
                (study_section.get("factor_selection") or {}).get("external"),
                False,
            ):
                raise ValueError(
                    "sensitivity_only requires external frozen factor selection."
                )
            if (
                self.reference_output_dir is None
                and self.reference_metrics_source is None
            ):
                raise ValueError(
                    "sensitivity_only requires study.reference_output_dir or "
                    "study.reference_metrics."
                )

    def _validate_easysteer(self) -> None:
        easy = self.config.get("easysteer") or {}
        if not parse_bool(easy.get("enabled"), True):
            return
        gpus = [int(value) for value in easy.get("gpus") or ()]
        if not gpus:
            raise ValueError("easysteer.gpus cannot be empty.")
        tp = int(easy.get("tensor_parallel_size", len(gpus)))
        dp = int(easy.get("data_parallel_size", 1))
        if tp * dp != len(gpus):
            raise ValueError(
                "easysteer.gpus must equal tensor_parallel_size * "
                "data_parallel_size."
            )
        try:
            easysteer_executable(easy.get("executable"))
        except FileNotFoundError:
            if not self.cli.dry_run:
                raise
            print("EasySteer runtime is not installed; this dry-run validates the plan only. Run scripts/setup.sh before launching.")

    def _validate_gpu_pool(self) -> None:
        workers = {
            int(value) for value in (self.config.get("workers") or {}).get("gpus", ())
        }
        if not workers:
            raise ValueError("workers.gpus cannot be empty.")
        easy = {
            int(value) for value in (self.config.get("easysteer") or {}).get("gpus", ())
        }
        overlap = workers.intersection(easy)
        allow = parse_bool(
            (self.config.get("easysteer") or {}).get("allow_worker_overlap"),
            False,
        )
        if overlap and not allow:
            raise ValueError(
                f"EasySteer/worker GPU overlap requires allow_worker_overlap: "
                f"{sorted(overlap)}"
            )

    def _acquire_lock(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / ".scheduler.lock"
        self.lock_file = path.open("w", encoding="utf-8")
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another scheduler is already using {self.output_dir}"
            ) from error
        self.lock_file.write(str(os.getpid()))
        self.lock_file.flush()
        LOGGER.info("Acquired scheduler lock path=%s", path)

    @staticmethod
    def _set_factors(node: dict[str, Any], factors: Mapping[str, list[float]]) -> None:
        inference = node.get("inference")
        if not isinstance(inference, dict):
            return
        # Node factor maps define routing; evaluate.models remains the artifact union.
        node["models"] = [str(method) for method in factors]
        inference.pop("strengths", None)
        inference.pop("factors", None)
        inference["strengths_by_model"] = {
            str(method): [float(value) for value in values]
            for method, values in factors.items()
        }

    def _bind_common_runtime(self, config: dict[str, Any]) -> None:
        generated = str(self.generate_dir.resolve())
        train = config.setdefault("train", {})
        train["overwrite_data_dir"] = generated
        train["overwrite_metadata_dir"] = generated
        evaluate = config.setdefault("evaluate", {})
        evaluate["shared_request_cache_dir"] = str(self.request_cache_dir)
        evaluate["progress_root"] = str(self.progress_root)
        evaluate["progress_source_root"] = str(self.output_dir)
        easy = self.config.get("easysteer") or {}
        evaluate["easysteer_url"] = (
            f"http://{easy.get('host', '127.0.0.1')}:"
            f"{int(easy.get('port', 8017))}"
        )

    def _method_scope(self, method: MethodTask) -> dict[str, int] | None:
        return _training_concept_scope(self.source_configs[method.stem])

    def _study_method_scope(
        self,
        study_source: Mapping[str, Any],
        method: MethodTask,
    ) -> dict[str, int] | None:
        """Resolve the deterministic training/evaluation panel for a method."""
        scopes = (study_source.get("study") or {}).get("concept_scopes") or {}
        by_method = scopes.get("by_method") or {}
        configured = by_method.get(method.method, scopes.get("default"))
        if configured is not None:
            if not isinstance(configured, Mapping):
                raise TypeError(
                    "study.concept_scopes entries must be mappings."
                )
            count = int(configured.get("count", 0))
            seed = int(configured.get("seed", 42))
            if count < 1:
                raise ValueError(
                    f"Invalid study concept count for {method.method}: {count}."
                )
            return {"count": count, "seed": seed}
        return self._method_scope(method) or _training_concept_scope(study_source)

    def _main_id_metrics_path(
        self, method: MethodTask, output_root: Path
    ) -> Path:
        source = self.source_configs[method.stem]
        run_id = str(
            (source.get("evaluate") or {}).get("evaluation_run_id")
            or method.stem
        )
        return (
            output_root / "methods" / method.stem / "evaluate" / "runs"
            / run_id / "evaluators" / "id_lm_judge" / "metrics.parquet"
        )

    def _materialize_study_reference(
        self, study_source: Mapping[str, Any]
    ) -> Path:
        """Freeze one completed main-sweep factor per selected method for downstream studies."""
        import pandas as pd
        from steerscope.evaluators.best_factor import BestFactorEvaluator

        source_paths: list[Path] = []
        selected_methods = {method.method for method in self.methods}
        reference = None
        if self.reference_metrics_source is not None:
            source_paths = [self.reference_metrics_source]
            if not self.reference_metrics_source.is_file():
                raise FileNotFoundError(
                    f"Frozen factor reference is missing: "
                    f"{self.reference_metrics_source}"
                )
            reference = pd.read_parquet(self.reference_metrics_source)

        if reference is not None:
            if not {"method", "factor"}.issubset(reference.columns):
                raise ValueError(
                    "Frozen factor reference must contain method and factor."
                )
            reference = reference[
                reference["method"].astype(str).isin(selected_methods)
            ].copy()
            counts = reference["method"].astype(str).value_counts().to_dict()
            invalid = {
                method: int(counts.get(method, 0))
                for method in sorted(selected_methods)
                if int(counts.get(method, 0)) != 1
            }
            if invalid:
                raise ValueError(
                    "Frozen factor reference needs exactly one row per method; "
                    f"invalid={invalid}."
                )
        else:
            if self.reference_output_dir is None:
                raise ValueError("No reference output directory was configured.")
            frames = []
            for method in self.methods:
                path = self._main_id_metrics_path(
                    method, self.reference_output_dir
                )
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Completed main ID metrics are missing: {path}"
                    )
                try:
                    manifest = json.loads(
                        path.with_name("manifest.json").read_text(
                            encoding="utf-8"
                        )
                    )
                except (OSError, json.JSONDecodeError) as error:
                    raise RuntimeError(
                        f"Cannot validate main ID metrics: {path}"
                    ) from error
                if manifest.get("status") != "complete":
                    raise RuntimeError(f"Main ID metrics are incomplete: {path}")
                source_paths.append(path)
                frame = pd.read_parquet(path)
                frames.append(
                    frame[frame["method"].astype(str) == method.method]
                )
            raw = pd.concat(frames, ignore_index=True, sort=False)
            result_id = str(
                ((study_source.get("study") or {}).get("result") or {}).get(
                    "evaluator", "study_result"
                )
            )
            selector_node = (
                (study_source.get("evaluate") or {}).get("evaluators") or {}
            ).get(result_id) or {}
            selector = object.__new__(BestFactorEvaluator)
            selector_params = copy.deepcopy(selector_node.get("params") or {})
            fallback = selector_params.get("fallback_baseline_method")
            if fallback is not None and str(fallback) not in set(
                raw["method"].astype(str)
            ):
                # A CLI-selected subset may omit DiffMean.  Every current ID
                # scan owns its factor-zero baseline, so selecting without the
                # unused fallback remains exactly equivalent.
                selector_params.pop("fallback_baseline_method", None)
            selector.node_config = {"params": selector_params}
            reference = pd.DataFrame(selector.compute_metrics(raw))
            reference.insert(0, "evaluator_type", "BestFactorEvaluator")
            reference.insert(0, "evaluator_id", "frozen_main_best_factor")

        reference["factor"] = pd.to_numeric(reference["factor"], errors="coerce")
        if reference["factor"].isna().any() or not reference["factor"].map(
            math.isfinite
        ).all():
            raise ValueError("Frozen factor reference contains non-finite factors.")
        reference = reference.sort_values("method").reset_index(drop=True)
        destination = self.output_dir / "studies/reference_best_factors.parquet"
        _atomic_parquet(reference, destination)
        atomic_json(destination.with_suffix(".provenance.json"), {
            "version": 1,
            "created_at": utc_now(),
            "selected_methods": sorted(selected_methods),
            "sources": [
                {"path": str(path.resolve()), "sha256": _sha256_file(path)}
                for path in source_paths
            ],
            "snapshot_sha256": _sha256_file(destination),
        })
        self.study_reference_metrics_path = destination
        return destination

    def _main_configs(self, method: MethodTask) -> tuple[Path, Path, dict, dict]:
        source = self.source_configs[method.stem]
        train_config = copy.deepcopy(source)
        self._bind_common_runtime(train_config)
        evaluate_config = copy.deepcopy(train_config)
        evaluate = evaluate_config.setdefault("evaluate", {})
        evaluate["models"] = [method.method]
        artifacts = {}
        if method.requires_training:
            artifacts[method.method] = str(method.train_dir.resolve())

        # Ordinary metrics share DiffMean factor zero; ID LM judge uses the base model.
        diff = next(
            (
                item for item in (*self.methods, *self.providers)
                if item.method == "DiffMean"
            ),
            None,
        )
        ordinary_needs_diff = False
        for node_id, node in (evaluate.get("evaluators") or {}).items():
            if not isinstance(node, dict) or not isinstance(node.get("inference"), dict):
                continue
            if node_id == "id_lm_judge":
                self._set_factors(
                    node,
                    {method.method: list(
                        (node["inference"].get("strengths_by_model") or {}).get(
                            method.method,
                            node["inference"].get("strengths") or (),
                        )
                    )},
                )
                continue
            inference = node["inference"]
            factors = list(
                (inference.get("strengths_by_model") or {}).get(method.method)
                or inference.get("strengths")
                or inference.get("factors")
                or ()
            )
            if method.method != "DiffMean" and factors:
                if diff is None:
                    raise ValueError(
                        f"Method {method.method} requires DiffMean baseline."
                    )
                scan_factors = [
                    float(value)
                    for value in factors
                    if float(value) != 0.0
                ]
                if not scan_factors:
                    raise ValueError(
                        f"Method {method.method} configures only factor zero for "
                        f"ordinary evaluator {node_id}."
                    )
                ordinary_needs_diff = True
                self._set_factors(
                    node, {method.method: scan_factors, "DiffMean": [0.0]}
                )
        if ordinary_needs_diff and diff is not None:
            evaluate["models"].append("DiffMean")
            artifacts["DiffMean"] = str(diff.train_dir.resolve())
            diff_recipe = (
                self.source_configs[diff.stem].get("train") or {}
            ).get("models", {}).get("DiffMean")
            evaluate_config.setdefault("train", {}).setdefault("models", {})[
                "DiffMean"
            ] = copy.deepcopy(diff_recipe)

        # Bind each inference node only to methods in its factor map.
        for node_id, node in (evaluate.get("evaluators") or {}).items():
            if not isinstance(node, dict):
                continue
            inference = node.get("inference")
            factors = (
                (inference.get("strengths_by_model") or {})
                if isinstance(inference, dict)
                else {}
            )
            if factors:
                node["models"] = list(factors)
            else:
                node["models"] = [method.method]
            if node_id == "id_lm_judge":
                node["models"] = [method.method]
        evaluate["artifact_dirs_by_model"] = artifacts

        scope = self._method_scope(method)
        if scope is not None and (self.output_dir / "generate/metadata.jsonl").is_file():
            _scope_evaluators_to_training_panel(
                evaluate_config,
                generate_dir=self.output_dir / "generate",
                **scope,
            )
        root = self.output_dir / "runtime_configs/phase1/methods" / method.stem
        train_path = root / "train.yaml"
        evaluate_path = root / "evaluate.yaml"
        atomic_yaml(train_path, train_config)
        atomic_yaml(evaluate_path, evaluate_config)
        method.runtime_config = evaluate_path
        method.evaluator_ids = tuple(
            (evaluate.get("evaluators") or {}).keys()
        )
        return train_path, evaluate_path, train_config, evaluate_config

    def _generalization_config(
        self, method: MethodTask, *, include_prompt: bool
    ) -> tuple[Path, dict[str, Any]]:
        config = load_yaml(self.source_dir / "generalization.yaml")
        self._bind_common_runtime(config)
        train = config.setdefault("train", {})
        recipe = copy.deepcopy((train.get("models") or {}).get(method.method))
        train["models"] = {method.method: recipe} if recipe is not None else {}
        evaluate = config.setdefault("evaluate", {})
        evaluate["models"] = [method.method]
        evaluate["artifact_dirs_by_model"] = (
            {method.method: str(method.train_dir.resolve())}
            if method.requires_training else {}
        )
        nodes = evaluate.get("evaluators") or {}
        selector = nodes["best_factor"]
        selector["models"] = [method.method]
        selector["depends_on"] = []
        selector["input"] = {
            "path": str((
                method.output_dir / "evaluate/runs"
                / str((self.source_configs[method.stem].get("evaluate") or {}).get(
                    "evaluation_run_id", method.stem
                ))
                / "evaluators/id_lm_judge/metrics.parquet"
            ).resolve()),
            "filters": {"method": method.method},
        }
        prompt = nodes.get("prompt_generalization")
        if include_prompt:
            prompt["models"] = [method.method]
            strengths = (prompt.get("inference") or {}).get("strengths_by_model") or {}
            self._set_factors(prompt, {method.method: strengths[method.method]})
        else:
            nodes.pop("prompt_generalization", None)
        evaluate["evaluators"] = nodes
        scope = self._method_scope(method)
        if scope is not None and (self.output_dir / "generate/metadata.jsonl").is_file():
            _scope_evaluators_to_training_panel(
                config, generate_dir=self.output_dir / "generate", **scope
            )
        path = (
            self.output_dir / "runtime_configs/phase2/generalization"
            / f"{method.stem}.yaml"
        )
        atomic_yaml(path, config)
        return path, config

    def _task_receipt_path(self, stem: str) -> Path:
        safe = stem.replace(":", "__").replace("/", "_")
        return self.output_dir / "runtime_configs/receipts" / f"{safe}.json"

    def _receipt_complete(self, task: PipelineTask) -> bool:
        path = task.receipt_path
        if path is None or not path.is_file():
            return False
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        exact_signature = receipt.get("signature") in {
            task.signature, *task.compatible_signatures
        }
        legacy_signature = (
            task.accept_legacy_signature
            and receipt.get("signature_schema") != TASK_SIGNATURE_SCHEMA
        )
        valid_receipt = receipt.get("status") == "complete" and (
            exact_signature or legacy_signature
        )
        if not valid_receipt:
            return False
        if task.kind in {"main_train", "study_train"}:
            complete = (
                task.output_dir / "train/artifact_manifest.json"
            ).is_file()
        else:
            complete = True
        if complete and task.evaluator_ids:
            if not task.run_id:
                return False
            for evaluator_id in task.evaluator_ids:
                root = (
                    task.output_dir / "evaluate/runs" / task.run_id
                    / "evaluators" / evaluator_id
                )
                try:
                    manifest = json.loads(
                        (root / "manifest.json").read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    return False
                if manifest.get("status") != "complete":
                    return False
                for kind in (manifest.get("metadata") or {}).get(
                    "result_kinds", ()
                ):
                    if not (root / f"{kind}.parquet").is_file():
                        return False
        signature_migrated = receipt.get("signature") != task.signature
        if (
            complete
            and (legacy_signature or signature_migrated)
            and not self.cli.dry_run
        ):
            self._write_receipt(task)
        return complete

    def _write_receipt(self, task: PipelineTask) -> None:
        if task.receipt_path is None:
            raise ValueError(f"Task {task.stem} has no receipt path.")
        atomic_json(task.receipt_path, {
            "version": 1,
            "signature_schema": TASK_SIGNATURE_SCHEMA,
            "status": "complete",
            "task": task.stem,
            "kind": task.kind,
            "method": task.method,
            "signature": task.signature,
            "completed_at": utc_now(),
        })

    def _add_task(self, task: PipelineTask) -> None:
        if task.stem in self.pipeline_task_map:
            raise ValueError(f"Duplicate scheduler task: {task.stem}")
        task.receipt_path = self._task_receipt_path(task.stem)
        prior = (self.state.get("pipeline_tasks") or {}).get(task.stem) or {}
        exact_signature = prior.get("signature") in {
            task.signature, *task.compatible_signatures
        }
        legacy_signature = (
            task.accept_legacy_signature
            and prior.get("signature_schema") != TASK_SIGNATURE_SCHEMA
        )
        same = exact_signature or legacy_signature
        if self.resume and self._receipt_complete(task):
            task.status = "complete"
            task.phase = "complete"
            task.progress = 1.0
        elif self.resume and same:
            prior_status = str(prior.get("status") or "queued")
            task.attempts = int(prior.get("attempts", 0))
            task.progress = float(prior.get("progress", 0.0))
            task.error = prior.get("error")
            if prior_status == "failed" and not self.cli.retry_failed:
                task.status = "failed"
                task.phase = "failed"
            else:
                # Running children never survive the scheduler process.
                task.status = "queued"
                task.phase = "queued"
        self.pipeline_tasks.append(task)
        self.pipeline_task_map[task.stem] = task

    def _prepare_tasks(self) -> None:
        self.pipeline_tasks = []
        self.pipeline_task_map = {}
        try:
            generated_signature = _generated_data_signature(
                self.generate_dir
            )
        except FileNotFoundError:
            generated_signature = "dry-run-not-generated"

        main_methods = [] if self.sensitivity_only else [
            *self.providers,
            *self.methods,
        ]
        # Register all providers before wiring cross-method evaluation dependencies.
        main_train_ids = {
            method.method: f"phase1:train:{method.stem}"
            for method in main_methods
            if method.requires_training
        }
        main_eval_ids: dict[str, str] = {}
        main_configs: dict[str, tuple[Path, Path, dict, dict]] = {}
        diff = next(
            (item for item in main_methods if item.method == "DiffMean"), None
        )
        for method in main_methods:
            train_path, evaluate_path, train_config, evaluate_config = (
                self._main_configs(method)
            )
            main_configs[method.method] = (
                train_path, evaluate_path, train_config, evaluate_config
            )
            concept_count = _configured_training_concept_count(train_config)
            if method.requires_training:
                stem = main_train_ids[method.method]
                signature = canonical_hash({
                    "kind": "main_train",
                    "config": _training_task_config(train_config),
                    "generated": generated_signature,
                    "recipe": TRAINING_RECIPE_VERSION,
                })
                self._add_task(PipelineTask(
                    stem=stem,
                    kind="main_train",
                    method=method.method,
                    command=(
                        self.torchrun,
                        "--standalone",
                        f"--nproc_per_node={method.nproc_per_node}",
                        str(REPO_ROOT / "steerscope/scripts/train.py"),
                        "--config", str(train_path),
                        "--dump_dir", str(method.output_dir),
                    ),
                    output_dir=method.output_dir,
                    gpu_count=method.gpu_count,
                    vram_gb=method.vram_gb,
                    exclusive=method.exclusive,
                    priority=1200 + method.priority,
                    max_retries=method.max_retries,
                    signature=signature,
                    config_path=train_path,
                    training_concepts=concept_count,
                ))

        for method in ([] if self.sensitivity_only else self.methods):
            train_path, evaluate_path, train_config, evaluate_config = (
                main_configs[method.method]
            )

            dependencies = []
            if method.method in main_train_ids:
                dependencies.append(main_train_ids[method.method])
            pre_dependency_fix_dependencies = list(dependencies)
            eval_models = list(
                (evaluate_config.get("evaluate") or {}).get("models") or ()
            )
            if (
                "DiffMean" in eval_models
                and method.method != "DiffMean"
                and diff is not None
                and "DiffMean" in main_train_ids
            ):
                dependencies.append(main_train_ids["DiffMean"])
            run_id = str(
                (evaluate_config.get("evaluate") or {}).get("evaluation_run_id")
                or method.stem
            )
            stem = f"phase1:evaluate:{method.stem}"
            main_eval_ids[method.method] = stem
            signature = canonical_hash({
                "kind": "main_evaluate",
                "config": evaluate_config,
                "generated": generated_signature,
                "dependencies": dependencies,
            })
            compatible_signatures = ()
            if dependencies != pre_dependency_fix_dependencies:
                # Allowlist receipts created before shared-baseline ordering was fixed.
                compatible_signatures = (canonical_hash({
                    "kind": "main_evaluate",
                    "config": evaluate_config,
                    "generated": generated_signature,
                    "dependencies": pre_dependency_fix_dependencies,
                }),)
            evaluators = dict(
                (evaluate_config.get("evaluate") or {}).get("evaluators") or {}
            )
            components = _evaluator_components(evaluators)
            if method.evaluator_workers <= 1 or len(components) <= 1:
                self._add_task(PipelineTask(
                    stem=stem,
                    kind="main_evaluate",
                    method=method.method,
                    command=(
                        self.python,
                        str(REPO_ROOT / "steerscope/scripts/evaluate.py"),
                        "--config", str(evaluate_path),
                        "--mode", "steering",
                        "--dump_dir", str(method.output_dir),
                    ),
                    output_dir=method.output_dir,
                    evaluator_ids=tuple(evaluators),
                    dependencies=tuple(dict.fromkeys(dependencies)),
                    gpu_count=method.gpu_count,
                    vram_gb=method.vram_gb,
                    exclusive=method.exclusive,
                    priority=1000 + method.priority,
                    max_retries=method.max_retries,
                    uses_api=True,
                    signature=signature,
                    compatible_signatures=compatible_signatures,
                    config_path=evaluate_path,
                    run_id=run_id,
                    training_concepts=concept_count,
                ))
                continue

            worker_limit = min(method.evaluator_workers, len(components))
            shard_stems = []
            shard_root = evaluate_path.parent / "evaluate_shards"
            for component in components:
                shard_name = component[0]
                shard_stem = f"{stem}:evaluator:{shard_name}"
                shard_stems.append(shard_stem)
                shard_config = copy.deepcopy(evaluate_config)
                shard_config["evaluate"]["evaluators"] = {
                    node_id: copy.deepcopy(evaluators[node_id])
                    for node_id in component
                }
                configured_models = list(
                    shard_config["evaluate"].get("models") or ()
                )
                component_methods = {
                    str(model)
                    for node_id in component
                    for model in (
                        (
                            evaluators[node_id].get("models")
                            if isinstance(evaluators[node_id], Mapping)
                            else None
                        )
                        or configured_models
                    )
                }
                shard_config["evaluate"]["models"] = [
                    model for model in configured_models
                    if model in component_methods
                ]
                artifacts = dict(
                    shard_config["evaluate"].get("artifact_dirs_by_model")
                    or {}
                )
                shard_config["evaluate"]["artifact_dirs_by_model"] = {
                    model: path for model, path in artifacts.items()
                    if model in component_methods
                }
                train_models = dict(
                    (shard_config.get("train") or {}).get("models") or {}
                )
                shard_config.setdefault("train", {})["models"] = {
                    model: recipe for model, recipe in train_models.items()
                    if model in component_methods
                }
                shard_path = shard_root / f"{shard_name}.yaml"
                atomic_yaml(shard_path, shard_config)
                shard_signature = canonical_hash({
                    "kind": "main_evaluate_shard",
                    "config": shard_config,
                    "generated": generated_signature,
                    "dependencies": dependencies,
                })
                self._add_task(PipelineTask(
                    stem=shard_stem,
                    kind="main_evaluate",
                    method=method.method,
                    command=(
                        self.python,
                        str(REPO_ROOT / "steerscope/scripts/evaluate.py"),
                        "--config", str(shard_path),
                        "--mode", "steering",
                        "--dump_dir", str(method.output_dir),
                    ),
                    output_dir=method.output_dir,
                    evaluator_ids=component,
                    dependencies=tuple(dict.fromkeys(dependencies)),
                    gpu_count=method.gpu_count,
                    vram_gb=method.vram_gb,
                    exclusive=method.exclusive,
                    priority=1000 + method.priority,
                    max_retries=method.max_retries,
                    uses_api=True,
                    signature=shard_signature,
                    config_path=shard_path,
                    run_id=run_id,
                    training_concepts=concept_count,
                    concurrency_key=stem,
                    concurrency_limit=worker_limit,
                ))

            # Preserve the historical method-level task ID as a zero-GPU
            # completion barrier.  Downstream tasks and state reporting can
            # keep depending on one stable ID while evaluator components run
            # and resume independently.
            self._add_task(PipelineTask(
                stem=stem,
                kind="main_evaluate_barrier",
                method=method.method,
                command=(self.python, "-c", "pass"),
                output_dir=method.output_dir,
                evaluator_ids=tuple(evaluators),
                dependencies=tuple(shard_stems),
                gpu_count=0,
                vram_gb=0.0,
                exclusive=False,
                priority=999 + method.priority,
                max_retries=0,
                uses_api=False,
                signature=canonical_hash({
                    "kind": "main_evaluate_barrier",
                    "config": evaluate_config,
                    "generated": generated_signature,
                    "dependencies": shard_stems,
                }),
                config_path=evaluate_path,
                run_id=run_id,
                training_concepts=concept_count,
            ))

        study_enabled = (
            parse_bool((self.config.get("study") or {}).get("enabled"), True)
            and not self.cli.skip_study
        )
        generalization_enabled = (
            parse_bool(
                (self.config.get("generalization") or {}).get("enabled"), True
            )
            and not self.cli.skip_generalization
        )
        if self.sensitivity_only and generalization_enabled:
            raise ValueError(
                "sensitivity_only requires generalization.enabled=false."
            )
        self.study_status = "queued" if study_enabled else "disabled"
        self.generalization_status = (
            "queued" if generalization_enabled else "disabled"
        )

        study_source = load_yaml(self.study_config_path)
        legacy_study_source = _pre_spsr_study_source(study_source)
        variants = []
        if study_enabled:
            variants = self._study_variants(study_source)
        study_train_ids: dict[tuple[str, str], str] = {}
        for variant in variants:
            variant_payload = {
                "train_examples": variant.train_examples,
                "subset_seed": variant.subset_seed,
                "efficiency": variant.efficiency,
                "sensitivity": variant.sensitivity,
                "factor_reference": variant.factor_reference,
            }
            for method in self.methods:
                training_scope = self._study_method_scope(study_source, method)
                recipe = (
                    (study_source.get("train") or {}).get("models") or {}
                ).get(method.method)
                requires_training = recipe is not None or method.method == "GemmaScopeSAE"
                if not requires_training:
                    continue
                output = (
                    self.output_dir / "studies/runs" / variant.run_id / method.stem
                )
                job = {
                    "python": self.python,
                    "torchrun": self.torchrun,
                    "nproc_per_node": method.nproc_per_node,
                    "method": method.method,
                    "variant": variant_payload,
                    "task_output": str(output.resolve()),
                    "shared_generate": str(self.generate_dir.resolve()),
                    "shared_request_cache_dir": str(self.request_cache_dir),
                    "progress_root": str(self.progress_root),
                    "progress_source_root": str(self.output_dir),
                    "study_config": str(self.study_config_path.resolve()),
                    "easysteer_url": (
                        f"http://{(self.config.get('easysteer') or {}).get('host', '127.0.0.1')}:"
                        f"{int((self.config.get('easysteer') or {}).get('port', 8017))}"
                    ),
                    "judge_concurrency": self.study_judge_concurrency,
                    "training_concept_scope": training_scope,
                    "stage": "train",
                }
                stem = f"phase1:study-train:{variant.run_id}:{method.stem}"
                signature = canonical_hash({
                    "kind": "study_train",
                    "source": _training_task_config(
                        _study_source_for_method(study_source, method.method)
                    ),
                    "variant": variant_payload,
                    "method": method.method,
                    "generated": generated_signature,
                    "scope": training_scope,
                    "recipe": TRAINING_RECIPE_VERSION,
                })
                compatible_signatures = ()
                if legacy_study_source is not None:
                    legacy_variant_payload = dict(variant_payload)
                    if (
                        variant.train_examples == 36
                        and variant.subset_seed == 42
                        and variant.efficiency
                    ):
                        # Before this migration, n=36/seed=42 was shared by
                        # efficiency and sensitivity.  Moving sensitivity to
                        # n=24 must not retrain the identical n=36 subset.
                        legacy_variant_payload["sensitivity"] = True
                    compatible_signatures = (canonical_hash({
                        "kind": "study_train",
                        "source": _training_task_config(legacy_study_source),
                        "variant": legacy_variant_payload,
                        "method": method.method,
                        "generated": generated_signature,
                        "scope": training_scope,
                        "recipe": TRAINING_RECIPE_VERSION,
                    }),)
                job["task_signature"] = signature
                job_path = (
                    self.output_dir / "runtime_configs/phase1/study"
                    / variant.run_id / f"{method.stem}.train.json"
                )
                atomic_json(job_path, job)
                study_train_ids[(variant.run_id, method.method)] = stem
                self._add_task(PipelineTask(
                    stem=stem,
                    kind="study_train",
                    method=method.method,
                    command=(
                        self.python, str(Path(__file__).resolve()),
                        "--study-worker", str(job_path),
                    ),
                    output_dir=output,
                    gpu_count=method.gpu_count,
                    vram_gb=method.vram_gb,
                    exclusive=method.exclusive,
                    priority=700 + method.priority,
                    max_retries=int(
                        (self.config.get("study") or {}).get("max_retries", 3)
                    ),
                    signature=signature,
                    compatible_signatures=compatible_signatures,
                    config_path=job_path,
                    training_concepts=int(
                        training_scope["count"]
                        if training_scope is not None
                        else _configured_training_concept_count(study_source)
                    ),
                ))

        if getattr(self.cli, "phase1_only", False):
            self.runtime_configs_prepared = True
            LOGGER.info(
                "Prepared phase-one-only tasks total=%d",
                len(self.pipeline_tasks),
            )
            return

        # One downstream generalization job per method keeps per-concept
        # artifacts independently scoped while retaining one source YAML.
        include_prompt = generalization_enabled
        factor_outputs: dict[str, Path] = {}
        factor_signatures: dict[str, str] = {}
        if self.sensitivity_only and study_enabled:
            frozen_reference = self._materialize_study_reference(study_source)
            factor_outputs = {
                method.method: frozen_reference for method in self.methods
            }
        downstream_methods = (
            [] if self.sensitivity_only else self.methods
        ) if (generalization_enabled or study_enabled) else []
        for method in downstream_methods:
            path, runtime = self._generalization_config(
                method, include_prompt=include_prompt
            )
            output = self.output_dir / "generalization/methods" / method.stem
            run_id = str(
                (runtime.get("evaluate") or {}).get("evaluation_run_id")
                or "generalization"
            )
            factor_outputs[method.method] = (
                output / "evaluate/runs" / run_id
                / "evaluators/best_factor/metrics.parquet"
            )
            stem = f"phase2:generalization:{method.stem}"
            signature = canonical_hash({
                "kind": "generalization" if include_prompt else "factor_select",
                "config": runtime,
                "generated": generated_signature,
                "main_reference": main_eval_ids[method.method],
            })
            factor_signatures[method.method] = signature
            self._add_task(PipelineTask(
                stem=stem,
                kind="generalization" if include_prompt else "factor_select",
                method=method.method,
                command=(
                    self.python,
                    str(REPO_ROOT / "steerscope/scripts/evaluate.py"),
                    "--config", str(path),
                    "--mode", "steering",
                    "--dump_dir", str(output),
                ),
                output_dir=output,
                evaluator_ids=tuple(
                    (runtime.get("evaluate") or {}).get("evaluators") or ()
                ),
                dependencies=(main_eval_ids[method.method],),
                gpu_count=method.gpu_count if include_prompt else 0,
                vram_gb=method.vram_gb if include_prompt else 0.0,
                exclusive=method.exclusive if include_prompt else False,
                priority=500 + method.priority,
                max_retries=method.max_retries,
                uses_api=include_prompt,
                signature=signature,
                config_path=path,
                run_id=run_id,
                training_concepts=_configured_training_concept_count(
                    self.source_configs[method.stem]
                ),
            ))

        if study_enabled:
            for variant in variants:
                variant_payload = {
                    "train_examples": variant.train_examples,
                    "subset_seed": variant.subset_seed,
                    "efficiency": variant.efficiency,
                    "sensitivity": variant.sensitivity,
                    "factor_reference": variant.factor_reference,
                }
                for method in self.methods:
                    training_scope = self._study_method_scope(
                        study_source, method
                    )
                    output = (
                        self.output_dir / "studies/runs" / variant.run_id
                        / method.stem
                    )
                    reference = factor_outputs[method.method]
                    job = {
                        "python": self.python,
                        "torchrun": self.torchrun,
                        "nproc_per_node": method.nproc_per_node,
                        "method": method.method,
                        "variant": variant_payload,
                        "task_output": str(output.resolve()),
                        "shared_generate": str(self.generate_dir.resolve()),
                        "shared_request_cache_dir": str(self.request_cache_dir),
                        "progress_root": str(self.progress_root),
                        "progress_source_root": str(self.output_dir),
                        "study_config": str(self.study_config_path.resolve()),
                        "easysteer_url": (
                            f"http://{(self.config.get('easysteer') or {}).get('host', '127.0.0.1')}:"
                            f"{int((self.config.get('easysteer') or {}).get('port', 8017))}"
                        ),
                        "judge_concurrency": self.study_judge_concurrency,
                        "training_concept_scope": training_scope,
                        "stage": "evaluate",
                        "reference_metrics": str(reference.resolve()),
                    }
                    stem = f"phase2:study-evaluate:{variant.run_id}:{method.stem}"
                    dependency = study_train_ids.get((variant.run_id, method.method))
                    dependencies = tuple(value for value in (dependency,) if value)
                    signature = canonical_hash({
                        "kind": "study_evaluate",
                        "source": _study_evaluation_source_for_method(
                            study_source, method.method, variant
                        ),
                        "variant": variant_payload,
                        "method": method.method,
                        "generated": generated_signature,
                        "factor_reference": (
                            {
                                "path": str(reference.resolve()),
                                "sha256": _sha256_file(reference),
                            }
                            if self.sensitivity_only
                            else {
                                "task": f"phase2:generalization:{method.stem}",
                                "signature": factor_signatures[method.method],
                            }
                        ),
                        "scope": training_scope,
                    })
                    compatible_signatures = ()
                    accept_legacy_signature = False
                    if (
                        legacy_study_source is not None
                        and method.method not in _FACTOR_GRID_MIGRATION_METHODS
                        and method.method != "SPSR"
                    ):
                        # Preserve allowlisted manifests across report-only config edits.
                        accept_legacy_signature = True
                        compatible_signatures = (canonical_hash({
                            "kind": "study_evaluate",
                            "source": legacy_study_source,
                            "variant": variant_payload,
                            "method": method.method,
                            "generated": generated_signature,
                            "factor_reference": (
                                {
                                    "path": str(reference.resolve()),
                                    "sha256": _sha256_file(reference),
                                }
                                if self.sensitivity_only
                                else f"phase2:generalization:{method.stem}"
                            ),
                            "scope": training_scope,
                        }),)
                    job["task_signature"] = signature
                    job_path = (
                        self.output_dir / "runtime_configs/phase2/study"
                        / variant.run_id / f"{method.stem}.evaluate.json"
                    )
                    atomic_json(job_path, job)
                    study_run_id = str(
                        (study_source.get("evaluate") or {}).get(
                            "evaluation_run_id"
                        ) or "study"
                    )
                    self._add_task(PipelineTask(
                        stem=stem,
                        kind="study_evaluate",
                        method=method.method,
                        command=(
                            self.python, str(Path(__file__).resolve()),
                            "--study-worker", str(job_path),
                        ),
                        output_dir=output,
                        evaluator_ids=tuple(
                            (study_source.get("evaluate") or {}).get("evaluators") or ()
                        ),
                        dependencies=dependencies,
                        required_artifacts=(reference,),
                        gpu_count=method.gpu_count,
                        vram_gb=method.vram_gb,
                        exclusive=method.exclusive,
                        priority=300 + method.priority,
                        max_retries=int(
                            (self.config.get("study") or {}).get("max_retries", 3)
                        ),
                        uses_api=True,
                        signature=signature,
                        compatible_signatures=compatible_signatures,
                        accept_legacy_signature=accept_legacy_signature,
                        config_path=job_path,
                        run_id=study_run_id,
                        training_concepts=int(
                            training_scope["count"]
                            if training_scope is not None
                            else _configured_training_concept_count(study_source)
                        ),
                    ))
        self.runtime_configs_prepared = True
        LOGGER.info(
            "Prepared two-phase tasks total=%d phase1=%d phase2=%d",
            len(self.pipeline_tasks),
            sum(task.stem.startswith("phase1:") for task in self.pipeline_tasks),
            sum(task.stem.startswith("phase2:") for task in self.pipeline_tasks),
        )

    @staticmethod
    def _filesystem_type(path: Path) -> str | None:
        try:
            mounts = []
            for line in Path("/proc/mounts").read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) >= 3:
                    mounts.append((Path(fields[1]), fields[2]))
        except OSError:
            return None
        resolved = path.resolve()
        matches = []
        for mount, fs_type in mounts:
            try:
                resolved.relative_to(mount)
            except ValueError:
                continue
            matches.append((len(mount.parts), fs_type))
        return max(matches, default=(0, None))[1]

    def _prepare_hot_storage(self) -> None:
        paths = {
            "progress": self.progress_root,
            "request cache": self.request_cache_dir,
            "LM SQLite cache": self.lm_cache_dir,
        }
        unsafe = {"nfs", "nfs4", "cifs", "smbfs", "sshfs", "fuse.sshfs"}
        for label, path in paths.items():
            path.mkdir(parents=True, exist_ok=True)
            fs_type = self._filesystem_type(path)
            if fs_type in unsafe:
                raise ValueError(
                    f"Local {label} must not use a network filesystem: "
                    f"{path} is {fs_type}."
                )
        stat = os.statvfs(self.hot_data_root)
        free_gb = stat.f_bavail * stat.f_frsize / (1024 ** 3)
        if free_gb < self.lm_cache_min_free_gb:
            raise OSError(
                f"Hot-data filesystem has {free_gb:.1f} GiB free; "
                f"requires {self.lm_cache_min_free_gb:.1f} GiB."
            )
        atomic_json(self.output_dir / "runtime_configs/hot_data.json", {
            "version": 2,
            "output_root": str(self.output_dir),
            "hot_data_root": str(self.hot_data_root),
            "namespace": str(self.hot_data_namespace),
            "progress_root": str(self.progress_root),
            "request_cache_dir": str(self.request_cache_dir),
            "lm_cache_dir": str(self.lm_cache_dir),
            "updated_at": utc_now(),
        })
        LOGGER.info(
            "Prepared local hot storage root=%s free_gib=%.1f",
            self.hot_data_root, free_gb,
        )

    def _command_env(
        self, gpus: Sequence[int], task_id: str | None = None
    ) -> dict[str, str]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in gpus)
        env["PYTHONUNBUFFERED"] = "1"
        env["STEERSCOPE_LM_CACHE_DIR"] = str(self.lm_cache_dir)
        migration = (self.config.get("runtime") or {}).get(
            "evaluation_identity_migration_file"
        )
        if migration and task_id and task_id.startswith("phase1:evaluate:"):
            path = resolve_repo_path(migration)
            if path.is_file():
                env["STEERSCOPE_EVALUATION_IDENTITY_MIGRATION_FILE"] = str(path)
        if task_id:
            env["STEERSCOPE_SCHEDULER_TASK_ID"] = task_id
        return env

    def _start_process(
        self,
        command: Sequence[str],
        log_path: Path,
        gpus: Sequence[int],
        *,
        task_id: str | None = None,
    ) -> tuple[subprocess.Popen, Any]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a", encoding="utf-8", buffering=1)
        log.write(f"\n[{utc_now()}] COMMAND: {shlex.join(command)}\n")
        try:
            process = subprocess.Popen(
                list(command),
                cwd=REPO_ROOT,
                env=self._command_env(gpus, task_id),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError:
            log.close()
            raise
        self.children.add(process)
        LOGGER.info(
            "Started task=%s pid=%d gpus=%s log=%s command=%s",
            task_id, process.pid, list(gpus), log_path,
            shlex.join(str(value) for value in command),
        )
        return process, log

    @staticmethod
    def _terminate_process(process: subprocess.Popen | None, grace: int = 15) -> None:
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        except ProcessLookupError:
            return

    @staticmethod
    def _tail(path: Path, limit: int = 16000) -> str:
        try:
            with path.open("rb") as file:
                file.seek(0, os.SEEK_END)
                size = file.tell()
                file.seek(max(0, size - limit))
                return file.read().decode("utf-8", errors="replace")
        except OSError:
            return ""

    def _start_easysteer(self) -> None:
        easy = self.config.get("easysteer") or {}
        if not parse_bool(easy.get("enabled"), True):
            LOGGER.info("EasySteer disabled")
            return
        if not self._needs_easysteer():
            LOGGER.info(
                "Skipping EasySteer startup: no unfinished main evaluation "
                "can use the EasySteer backend"
            )
            return
        self.current_stage = "start_easysteer"
        executable_path = easysteer_executable(easy.get("executable"))
        gpus = [int(value) for value in easy.get("gpus") or ()]
        tp = int(easy.get("tensor_parallel_size", len(gpus)))
        dp = int(easy.get("data_parallel_size", 1))
        model = str(
            (next(iter(self.source_configs.values())).get("train") or {}).get(
                "model_name"
            )
        )
        port = int(easy.get("port", 8017))
        bind_host = str(easy.get("bind_host", easy.get("host", "127.0.0.1")))
        command = [
            executable_path, "serve", model,
            "--enable-steer-vector",
            "--host", bind_host,
            "--port", str(port),
            "--tensor-parallel-size", str(tp),
            "--data-parallel-size", str(dp),
            "--gpu-memory-utilization",
            str(float(easy.get("gpu_memory_utilization", 0.9))),
        ]
        if parse_bool(easy.get("enforce_eager"), True):
            command.append("--enforce-eager")
        command.extend(str(value) for value in easy.get("extra_args") or ())
        self.service_process, self.service_log = self._start_process(
            command, self.output_dir / "logs/easysteer.log", gpus,
            task_id="easysteer",
        )
        health_host = str(easy.get("health_host", easy.get("host", "127.0.0.1")))
        url = f"http://{health_host}:{port}/v1/models"
        deadline = time.monotonic() + float(
            easy.get("startup_timeout_seconds", 600)
        )
        opener = build_opener(ProxyHandler({}))
        while time.monotonic() < deadline:
            if self.stop_requested:
                raise KeyboardInterrupt
            if self.service_process.poll() is not None:
                raise RuntimeError(
                    "EasySteer exited during startup; see logs/easysteer.log"
                )
            try:
                with opener.open(Request(url, method="GET"), timeout=5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if model in {str(item.get("id")) for item in payload.get("data", ())}:
                    LOGGER.info("EasySteer healthy url=%s model=%s", url, model)
                    return
            except (URLError, TimeoutError, OSError, json.JSONDecodeError):
                pass
            self._report()
            time.sleep(min(5, self.poll_seconds))
        raise TimeoutError(f"EasySteer did not become healthy: {url}")

    def _stop_easysteer(self, allocator: GPUAllocator) -> None:
        """Stop EasySteer once no remaining task can use it."""
        process = self.service_process
        if process is None:
            return
        LOGGER.info(
            "Stopping EasySteer after all main method inference finished"
        )
        self._terminate_process(process)
        self.children.discard(process)
        self.service_process = None
        if self.service_log is not None:
            self.service_log.close()
            self.service_log = None
        easy_gpus = [
            int(value)
            for value in (self.config.get("easysteer") or {}).get("gpus", ())
        ]
        allocator.release_persistent_reservations(easy_gpus)

    def _generate_progress(self) -> float:
        source = next(iter(self.source_configs.values()))
        total = int((source.get("generate") or {}).get("max_concepts") or 1)
        try:
            completed = sum(
                bool(line.strip())
                for line in (self.generate_dir / "metadata.jsonl")
                .read_text(encoding="utf-8").splitlines()
            )
        except OSError:
            completed = 0
        return 1.0 if self.generate_status == "complete" else min(
            1.0, completed / max(1, total)
        )

    def _validate_generate_pool(self) -> tuple[bool, str]:
        generate = next(iter(self.source_configs.values())).get("generate") or {}
        expected_rows = int(generate.get("num_of_examples") or 0)
        expected_concepts = int(generate.get("max_concepts") or 0)
        root = self.generate_dir
        metadata_path = root / "metadata.jsonl"
        parquet_paths = sorted(root.glob("train_data*.parquet"))
        if expected_rows <= 0 or expected_concepts <= 0:
            return False, "generate counts must be positive"
        if not metadata_path.is_file() or not parquet_paths:
            return False, "metadata/parquet files are missing"
        try:
            metadata_ids = [
                int(json.loads(line)["concept_id"])
                for line in metadata_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            import pandas as pd
            counts: dict[int, int] = {}
            for path in parquet_paths:
                frame = pd.read_parquet(path)
                # Both supported pool layouts count concept-positive rows.
                if "category" in frame.columns:
                    frame = frame[
                        frame["category"].astype(str).str.lower() == "positive"
                    ]
                for concept_id, count in frame["concept_id"].value_counts().items():
                    key = int(concept_id)
                    counts[key] = counts.get(key, 0) + int(count)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            return False, f"cannot inspect generated pool: {error}"
        if len(metadata_ids) != expected_concepts or len(set(metadata_ids)) != expected_concepts:
            return False, (
                f"metadata has {len(metadata_ids)} rows and "
                f"{len(set(metadata_ids))} unique IDs; expected {expected_concepts}"
            )
        # The generated pool stores half of num_of_examples as concept-positive rows.
        expected_positive_rows = expected_rows // 2
        bad = {
            concept_id: counts.get(concept_id, 0)
            for concept_id in metadata_ids
            if counts.get(concept_id, 0) != expected_positive_rows
        }
        unexpected = sorted(set(counts).difference({-1, *metadata_ids}))
        if bad or unexpected:
            return False, (
                f"expected {expected_positive_rows} positive rows/concept; "
                f"bad={list(bad.items())[:5]} "
                f"unexpected={unexpected[:5]}"
            )
        return True, "ok"

    def _run_generate(self) -> None:
        source = self.methods[0].source_config
        generate = next(iter(self.source_configs.values())).get("generate") or {}
        self.generate_signature = canonical_hash(generate)
        valid, reason = self._validate_generate_pool()
        if self.generate_dir != (self.output_dir / "generate").resolve():
            if not valid:
                raise RuntimeError(
                    f"Shared generated pool validation failed at "
                    f"{self.generate_dir}: {reason}"
                )
            self.generate_status = "complete"
            LOGGER.info("Reusing external generated pool: %s", self.generate_dir)
            return
        if self.resume and valid:
            self.generate_status = "complete"
            LOGGER.info("Reusing generated pool: %s", reason)
            return
        self.stage = self.current_stage = "generate"
        self.generate_status = "running"
        process, log = self._start_process(
            [
                self.python,
                str(REPO_ROOT / "steerscope/scripts/generate.py"),
                "--config", str(source),
                "--dump_dir", str(self.output_dir),
            ],
            self.output_dir / "logs/generate.log",
            (),
            task_id="generate",
        )
        try:
            while process.poll() is None:
                if self.stop_requested:
                    raise KeyboardInterrupt
                self._report()
                time.sleep(self.poll_seconds)
        finally:
            log.close()
            if process.poll() is not None:
                self.children.discard(process)
        if process.returncode != 0:
            self.generate_status = "failed"
            raise RuntimeError("Generate failed; see logs/generate.log")
        valid, reason = self._validate_generate_pool()
        if not valid:
            self.generate_status = "failed"
            raise RuntimeError(f"Generated pool validation failed: {reason}")
        self.generate_status = "complete"
        LOGGER.info("Generation complete")

    def _evaluation_progress(self, task: PipelineTask) -> float:
        if not task.evaluator_ids or not task.run_id:
            return 0.0
        values = []
        for evaluator_id in task.evaluator_ids:
            manifest_path = (
                task.output_dir / "evaluate/runs" / task.run_id
                / "evaluators" / evaluator_id / "manifest.json"
            )
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                values.append(0.0)
                continue
            if manifest.get("status") == "complete":
                values.append(1.0)
                continue
            progress = (manifest.get("metadata") or {}).get("progress") or {}
            execution_hash = manifest.get("execution_hash")
            if execution_hash:
                try:
                    relative = manifest_path.parent.relative_to(self.output_dir)
                    local = (
                        self.progress_root / relative / "progress"
                        / str(execution_hash) / "_progress.json"
                    )
                    if local.is_file():
                        progress = json.loads(local.read_text(encoding="utf-8"))
                except (ValueError, OSError, json.JSONDecodeError):
                    pass
            total = int(progress.get("total_targets") or 0)
            completed = int(progress.get("completed_targets") or 0)
            values.append(min(0.99, completed / total) if total else 0.0)
        return sum(values) / len(values) if values else 0.0

    def _training_progress(self, task: PipelineTask) -> float:
        if not task.method:
            return 0.0
        train_dir = task.output_dir / "train"
        if task.method == "GemmaScopeSAE":
            required = (
                train_dir / "GemmaScopeSAE.pt",
                train_dir / "GemmaScopeSAE_scale.pt",
            )
            return 1.0 if all(path.is_file() for path in required) else 0.0
        joint = (
            train_dir / "checkpoints/all_concepts" / task.method / ".complete.json"
        )
        if joint.is_file():
            return 1.0
        markers = list(
            (train_dir / "checkpoints").glob(
                f"rank_*/concept_*/{task.method}/.complete.json"
            )
        )
        return min(1.0, len(markers) / max(1, task.training_concepts))

    def _update_task_progress(self, task: PipelineTask) -> None:
        if task.status == "complete":
            task.progress = 1.0
            return
        if task.kind in {"main_train", "study_train"}:
            current = self._training_progress(task)
        else:
            current = self._evaluation_progress(task)
        task.progress = max(task.progress, current)

    def _phase_tasks(self, phase: int) -> list[PipelineTask]:
        prefix = f"phase{phase}:"
        return [task for task in self.pipeline_tasks if task.stem.startswith(prefix)]

    def _main_inference_ready(self) -> bool:
        """Return whether main method training and evaluation are complete."""
        tasks = [
            task for task in self.pipeline_tasks
            if task.kind in {"main_train", "main_evaluate"}
        ]
        evaluations = [task for task in tasks if task.kind == "main_evaluate"]
        return bool(evaluations) and all(task.status == "complete" for task in tasks)

    def _needs_easysteer(self) -> bool:
        """Return whether an unfinished main evaluation can use EasySteer."""
        return any(
            task.kind == "main_evaluate"
            and task.method in EASYSTEER_METHODS
            and task.status != "complete"
            for task in self.pipeline_tasks
        )

    def _sync_method_status(self) -> None:
        for method in self.methods:
            train = self.pipeline_task_map.get(f"phase1:train:{method.stem}")
            evaluations = [
                task for task in self.pipeline_tasks
                if task.method == method.method
                and task.kind in {"main_evaluate", "main_evaluate_barrier"}
            ]
            values = [value for value in (train, *evaluations) if value is not None]
            if self.sensitivity_only:
                values = [
                    task for task in self.pipeline_tasks
                    if task.method == method.method
                    and task.kind in {"study_train", "study_evaluate"}
                ]
            # Generate runs before the phase-one task graph is materialized.
            # Reporter updates during that stage must keep methods queued
            # instead of averaging an empty task list.
            if not values:
                method.progress = 0.0
                method.status = "queued"
                method.phase = "queued"
                method.allocated_gpus = ()
                continue
            progress_values = [
                value for value in values
                if value.kind != "main_evaluate_barrier"
            ]
            method.progress = (
                sum(value.progress for value in progress_values)
                / len(progress_values)
            )
            if any(value.status == "failed" for value in values):
                method.status = "failed"
            elif all(value.status == "complete" for value in values):
                method.status = "complete"
            elif any(value.status == "running" for value in values):
                method.status = "running"
            else:
                method.status = "queued"
            active = next(
                (
                    value for value in values
                    if value.status == "running"
                ),
                None,
            )
            method.phase = (
                active.phase if active is not None else method.status
            )
            method.allocated_gpus = (
                next(
                    (value.allocated_gpus for value in values if value.allocated_gpus),
                    (),
                )
            )

    def _stage_progress(self, kinds: set[str]) -> float:
        tasks = [task for task in self.pipeline_tasks if task.kind in kinds]
        return (
            sum(task.progress for task in tasks) / len(tasks)
            if tasks else 1.0
        )

    def _progress_values(self) -> dict[str, float]:
        values = {
            "generate": self._generate_progress(),
            "methods": self._stage_progress({"main_train", "main_evaluate"}),
            "study": self._stage_progress({"study_train", "study_evaluate"}),
            "generalization": self._stage_progress(
                {"generalization", "factor_select"}
            ),
        }
        if not self.runtime_configs_prepared:
            values.update({"methods": 0.0, "study": 0.0, "generalization": 0.0})
        configured = (self.config.get("progress") or {}).get("weights") or {
            "generate": 0.05,
            "methods": 0.45,
            "study": 0.35,
            "generalization": 0.15,
        }
        weights = {key: float(configured.get(key, 0.0)) for key in values}
        if self.study_status == "disabled":
            weights["study"] = 0.0
        if self.generalization_status == "disabled":
            weights["generalization"] = 0.0
        denominator = sum(weights.values()) or 1.0
        values["overall"] = sum(
            weights[key] * values[key] for key in weights
        ) / denominator
        return values

    def _counts(self) -> dict[str, int]:
        return {
            "total": len(self.pipeline_tasks),
            "complete": sum(task.status == "complete" for task in self.pipeline_tasks),
            "running": sum(task.status == "running" for task in self.pipeline_tasks),
            "failed": sum(task.status == "failed" for task in self.pipeline_tasks),
        }

    def _report(self, force_html: bool = False) -> None:
        if self.reporter is None:
            return
        self._sync_method_status()
        study_tasks = [
            task for task in self.pipeline_tasks
            if task.kind in {"study_train", "study_evaluate"}
        ]
        gen_tasks = [
            task for task in self.pipeline_tasks
            if task.kind in {"generalization", "factor_select"}
        ]
        if study_tasks and self.study_status != "disabled":
            self.study_status = (
                "complete" if all(task.status == "complete" for task in study_tasks)
                else "failed" if any(task.status == "failed" for task in study_tasks)
                else "running"
            )
        if gen_tasks and self.generalization_status != "disabled":
            self.generalization_status = (
                "complete" if all(task.status == "complete" for task in gen_tasks)
                else "failed" if any(task.status == "failed" for task in gen_tasks)
                else "running"
            )
        self.reporter.update(
            self._progress_values(),
            self.methods,
            self.current_stage,
            self._counts(),
            {
                "generate": self.generate_status,
                "study": self.study_status,
                "generalization": self.generalization_status,
            },
            force_html=force_html,
        )
        self._save_state()

    def _concurrency_slot_available(self, task: PipelineTask) -> bool:
        if not task.concurrency_key or task.concurrency_limit <= 0:
            return True
        running = sum(
            value.status == "running"
            and value.concurrency_key == task.concurrency_key
            for value in self.pipeline_tasks
        )
        return running < task.concurrency_limit

    def _start_task(self, task: PipelineTask, allocator: GPUAllocator) -> bool:
        gpus = allocator.allocate(task)
        if gpus is None:
            return False
        safe = task.stem.replace(":", "__").replace("/", "_")
        task.process, task.log_file = self._start_process(
            task.command,
            self.output_dir / "logs/pipeline" / f"{safe}.log",
            gpus,
            task_id=task.stem,
        )
        task.allocated_gpus = tuple(gpus)
        task.status = "running"
        task.phase = task.kind
        task.attempts += 1
        task.error = None
        return True

    def _finish_task(self, task: PipelineTask, allocator: GPUAllocator) -> None:
        if task.process is None or task.process.poll() is None:
            return
        returncode = int(task.process.returncode)
        if task.log_file is not None:
            task.log_file.close()
        self.children.discard(task.process)
        task.process = None
        task.log_file = None
        allocator.release(task)
        task.allocated_gpus = ()
        if returncode == 0:
            self._write_receipt(task)
            task.status = "complete"
            task.phase = "complete"
            task.progress = 1.0
            LOGGER.info("Completed task=%s attempts=%d", task.stem, task.attempts)
            return
        safe = task.stem.replace(":", "__").replace("/", "_")
        log_path = self.output_dir / "logs/pipeline" / f"{safe}.log"
        tail = self._tail(log_path).lower()
        oom = "out of memory" in tail or "cuda error: out of memory" in tail
        retry_oom = parse_bool(
            (self.config.get("workers") or {}).get("retry_exclusive_on_oom"),
            True,
        )
        if oom and retry_oom and not task.exclusive:
            task.exclusive = True
            task.status = "queued"
            task.phase = "queued"
            task.error = "OOM; retrying on an exclusive GPU"
        elif task.attempts <= task.max_retries:
            task.status = "queued"
            task.phase = "queued"
            task.error = f"exit {returncode}; retrying"
        else:
            task.status = "failed"
            task.phase = "failed"
            task.error = f"exit {returncode}"
        LOGGER.warning(
            "Task exited task=%s returncode=%d status=%s attempts=%d/%d",
            task.stem, returncode, task.status, task.attempts,
            task.max_retries + 1,
        )

    def _dependencies_ready(self, task: PipelineTask) -> bool:
        dependencies = [self.pipeline_task_map[value] for value in task.dependencies]
        if any(value.status == "failed" for value in dependencies):
            task.status = "failed"
            task.phase = "blocked"
            task.error = DEPENDENCY_BLOCKED_ERROR
            return False
        return all(value.status == "complete" for value in dependencies)

    def _phase_complete(self, phase: int) -> bool:
        tasks = self._phase_tasks(phase)
        return all(task.status == "complete" for task in tasks)

    def _phase_failed(self, phase: int) -> list[str]:
        return [
            task.stem for task in self._phase_tasks(phase)
            if task.status == "failed"
        ]

    def _run_pipeline(self) -> None:
        workers = self.config.get("workers") or {}
        allocator = GPUAllocator(
            [int(value) for value in workers.get("gpus") or ()],
            float(workers.get("safety_margin_gb", 6)),
            reserve_initial_usage=self.service_process is not None and parse_bool(
                (self.config.get("easysteer") or {}).get(
                    "allow_worker_overlap"
                ),
                False,
            ),
        )
        allocator.validate()
        allocator.validate_tasks(self.pipeline_tasks)

        while True:
            if self.stop_requested:
                raise KeyboardInterrupt
            for task in self.pipeline_tasks:
                if task.status == "running":
                    self._finish_task(task, allocator)
                self._update_task_progress(task)
            if not self._needs_easysteer():
                self._stop_easysteer(allocator)
            elif (
                self.service_process is not None
                and self.service_process.poll() is not None
            ):
                raise RuntimeError(
                    "EasySteer exited unexpectedly; see logs/easysteer.log"
                )

            phase1_failures = self._phase_failed(1)
            if phase1_failures:
                raise RuntimeError(
                    "Phase one failed: " + ", ".join(phase1_failures)
                )
            phase1_complete = self._phase_complete(1)
            if not phase1_complete:
                self.stage = self.current_stage = "phase1"
                allowed_phase = 1
            else:
                if getattr(self.cli, "phase1_only", False):
                    self.stage = self.current_stage = "phase1_complete"
                    self._report(force_html=True)
                    return
                phase2_failures = self._phase_failed(2)
                if phase2_failures:
                    raise RuntimeError(
                        "Phase two failed: " + ", ".join(phase2_failures)
                    )
                if self._phase_complete(2):
                    self.stage = self.current_stage = "finalize"
                    self._report(force_html=True)
                    return
                self.stage = self.current_stage = "phase2"
                allowed_phase = 2

            running_api = sum(
                task.status == "running" and task.uses_api
                for task in self.pipeline_tasks
            )
            for task in sorted(
                self._phase_tasks(allowed_phase),
                key=lambda value: (-value.priority, value.stem),
            ):
                if task.status != "queued":
                    continue
                if not self._dependencies_ready(task):
                    continue
                if not all(path.is_file() for path in task.required_artifacts):
                    continue
                if not self._concurrency_slot_available(task):
                    continue
                if (
                    task.uses_api
                    and running_api >= self.max_concurrent_api_runs
                ):
                    continue
                if self._start_task(task, allocator) and task.uses_api:
                    running_api += 1
            self._report()
            time.sleep(self.poll_seconds)

    @staticmethod
    def _atomic_frame(frame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)

    def _aggregate_generalization(self) -> None:
        tasks = [
            task for task in self.pipeline_tasks
            if task.kind in {"generalization", "factor_select"}
        ]
        if not tasks:
            return
        import pandas as pd
        destination_root = (
            self.output_dir / "generalization/evaluate/runs/generalization/evaluators"
        )
        evaluator_ids = sorted({
            evaluator_id for task in tasks for evaluator_id in task.evaluator_ids
        })
        for evaluator_id in evaluator_ids:
            frames = {kind: [] for kind in ("inference", "samples", "metrics")}
            manifests = []
            for task in tasks:
                source = (
                    task.output_dir / "evaluate/runs" / task.run_id
                    / "evaluators" / evaluator_id
                )
                manifest_path = source / "manifest.json"
                if not manifest_path.is_file():
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") != "complete":
                    raise RuntimeError(f"Incomplete evaluator: {manifest_path}")
                manifests.append(manifest)
                for kind in frames:
                    path = source / f"{kind}.parquet"
                    if path.is_file():
                        frame = pd.read_parquet(path)
                        if not frame.empty:
                            frames[kind].append(frame)
            if not manifests:
                continue
            destination = destination_root / evaluator_id
            result_kinds = []
            for kind, values in frames.items():
                if not values:
                    continue
                combined = pd.concat(values, ignore_index=True, sort=False)
                if kind == "metrics":
                    _validate_merged_metric_identities(combined, evaluator_id)
                self._atomic_frame(combined, destination / f"{kind}.parquet")
                result_kinds.append(kind)
            atomic_json(destination / "manifest.json", {
                "node_id": evaluator_id,
                "run_id": "generalization",
                "status": "complete",
                "execution_hash": canonical_hash({
                    "sources": [item.get("execution_hash") for item in manifests]
                }),
                "config": manifests[0].get("config") or {},
                "metadata": {
                    "result_kinds": result_kinds,
                    "combined_methods": True,
                },
            })
        atomic_json(destination_root.parent / "manifest.json", {
            "status": "complete",
            "evaluators": evaluator_ids,
            "methods": [method.method for method in self.methods],
            "updated_at": utc_now(),
        })

    def _aggregate_study(self) -> None:
        if self.study_status == "disabled":
            return
        import pandas as pd
        from steerscope.studies.training_data import (
            aggregate_study_metrics,
            collect_variant_metrics,
        )
        source = load_yaml(self.study_config_path)
        variants = self._study_variants(source)
        run_id = str(
            (source.get("evaluate") or {}).get("evaluation_run_id") or "study"
        )
        frames = []
        for variant in variants:
            for method in self.methods:
                root = (
                    self.output_dir / "studies/runs" / variant.run_id
                    / method.stem
                )
                frames.append(collect_variant_metrics(root, run_id, variant))
        factor_selection = (
            (source.get("study") or {}).get("factor_selection") or {}
        )
        selector_id = str(
            ((source.get("study") or {}).get("result") or {}).get(
                "evaluator"
            ) or "study_result"
        )
        shared_reference = None
        if self.sensitivity_only:
            reference_path = self.study_reference_metrics_path or (
                self.output_dir / "studies/reference_best_factors.parquet"
            )
            shared_reference = pd.read_parquet(reference_path)
        for method in self.methods:
            if shared_reference is not None:
                reference = shared_reference
            else:
                reference = pd.read_parquet(
                    self.output_dir / "generalization/methods" / method.stem
                    / "evaluate/runs/generalization/evaluators"
                    / "best_factor/metrics.parquet"
                )
            frames.append(_external_study_reference(
                reference,
                method=method.method,
                evaluator_id=selector_id,
                train_examples=int(factor_selection["train_examples"]),
                subset_seed=int(factor_selection.get("subset_seed", 42)),
            ))
        metrics = pd.concat(frames, ignore_index=True, sort=False)
        from steerscope.studies.statistics import load_study_concept_scores
        summary_config = copy.deepcopy(source)
        if self.sensitivity_only:
            summary_config["study"].pop("sample_efficiency", None)
        if not self.sample_sensitivity_enabled:
            metrics.loc[:, "sensitivity"] = False
        concept_scores = load_study_concept_scores(
            metrics, summary_config, self.output_dir / "studies",
            {method.method: self.output_dir / "methods" / method.stem / "evaluate/runs"
             for method in self.methods},
        )
        efficiency, sensitivity = aggregate_study_metrics(
            metrics, summary_config, concept_scores=concept_scores,
        )
        if self.sensitivity_only:
            efficiency = efficiency.iloc[0:0].copy()
        if not self.sample_sensitivity_enabled:
            sensitivity = sensitivity.iloc[0:0].copy()
        root = self.output_dir / "studies"
        self._atomic_frame(metrics, root / "metrics.parquet")
        self._atomic_frame(efficiency, root / "sample_efficiency.parquet")
        self._atomic_frame(sensitivity, root / "sample_sensitivity.parquet")
        atomic_json(root / "manifest.json", {
            "status": "complete",
            "variants": [variant.run_id for variant in variants],
            "methods": [method.method for method in self.methods],
            "sample_sensitivity_enabled": self.sample_sensitivity_enabled,
            "updated_at": utc_now(),
        })

    def _finalize(self) -> None:
        self.current_stage = self.stage = "finalize"
        self._aggregate_generalization()
        self._aggregate_study()
        self.stage = self.current_stage = "complete"
        atomic_json(self.output_dir / "scheduler_complete.json", {
            "status": "complete",
            "completed_at": utc_now(),
            "methods": [method.method for method in self.methods],
        })
        self._report(force_html=True)

    def _dry_run(self) -> int:
        valid, reason = self._validate_generate_pool()
        print(f"Config:          {self.config_path}")
        print(f"Source configs:  {self.source_dir}")
        print(f"Generated pool:  {self.generate_dir}")
        print(f"Output:          {self.output_dir}")
        print(f"Methods:         {len(self.methods)}")
        print(f"Generate reuse:  {valid} ({reason})")
        if valid:
            self.generate_status = "complete"
            self._prepare_tasks()
            print(f"Phase 1 tasks:   {len(self._phase_tasks(1))}")
            print(f"Phase 2 tasks:   {len(self._phase_tasks(2))}")
            for phase in (1, 2):
                counts: dict[str, int] = {}
                for task in self._phase_tasks(phase):
                    counts[task.kind] = counts.get(task.kind, 0) + 1
                summary = ", ".join(
                    f"{kind}={count}" for kind, count in sorted(counts.items())
                )
                print(f"  Phase {phase}: {summary}")
        else:
            study_enabled = (
                parse_bool(
                    (self.config.get("study") or {}).get("enabled"), True
                )
                and not self.cli.skip_study
            )
            variants = (
                self._study_variants(load_yaml(self.study_config_path))
                if study_enabled else []
            )
            trainable = sum(method.requires_training for method in self.methods)
            generalization_enabled = (
                parse_bool(
                    (self.config.get("generalization") or {}).get("enabled"),
                    True,
                )
                and not self.cli.skip_generalization
            )
            factor_tasks = len(self.methods) if (
                generalization_enabled or study_enabled
            ) and not self.sensitivity_only else 0
            main_tasks = 0 if self.sensitivity_only else (
                len(self.methods) + trainable
            )
            print(
                "Projected after generation: phase1="
                f"{main_tasks + len(variants) * trainable}, "
                "phase2="
                f"{factor_tasks + len(variants) * len(self.methods)}"
            )
        return 0

    def run(self) -> int:
        if self.cli.dry_run:
            return self._dry_run()
        self._acquire_lock()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        wandb_config = dict(self.config.get("wandb") or {})
        if self.cli.no_wandb:
            wandb_config["enabled"] = False
        wandb_config = wandb_launch_config(
            wandb_config, f"steerscope-{self.model_key}-l{self.layer}"
        )
        self.reporter = ProgressReporter(
            wandb_config,
            self.output_dir,
            {"scheduler": self.config, "config_path": str(self.config_path)},
        )
        status = "failed"
        try:
            self._prepare_hot_storage()
            self._run_generate()
            self._prepare_tasks()
            self._start_easysteer()
            self._run_pipeline()
            if not getattr(self.cli, "phase1_only", False):
                self._finalize()
            status = "complete"
            return 0
        except KeyboardInterrupt:
            status = "stopped"
            LOGGER.warning("Scheduler stopping after signal/user interrupt")
            return 130
        finally:
            for task in self.pipeline_tasks:
                self._terminate_process(task.process)
                if task.log_file is not None:
                    task.log_file.close()
            self._terminate_process(self.service_process)
            if self.service_log is not None:
                self.service_log.close()
            self._save_state()
            if self.reporter is not None:
                self.reporter.finish(status, self.methods)
            if self.lock_file is not None:
                self.lock_file.close()
                self.lock_file = None
            LOGGER.info("Scheduler cleanup status=%s", status)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Scheduler YAML path.")
    parser.add_argument(
        "--study-worker",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Only queue matching YAML stems or method class names.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--phase1-only",
        action="store_true",
        help="Run training and main inference through phase one, then stop.",
    )
    parser.add_argument("--skip-study", action="store_true")
    parser.add_argument("--skip-generalization", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    if cli.study_worker:
        return run_study_worker(
            Path(cli.study_worker).expanduser().resolve()
        )
    if not cli.config:
        raise ValueError("--config is required for the scheduler.")
    config_path = Path(cli.config).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if cli.dry_run:
        return Scheduler(config_path, cli).run()

    # Resolve the output directory before full scheduler discovery so config
    # or discovery failures can also leave a traceback in scheduler.log.
    preliminary_config = load_yaml(config_path)
    preliminary_experiment = preliminary_config.get("experiment") or {}
    preliminary_model_key = str(
        preliminary_experiment.get("model_key") or ""
    ).strip()
    preliminary_layer = int(preliminary_experiment.get("layer"))
    if not preliminary_model_key:
        raise ValueError("experiment.model_key is required.")
    preliminary_output = resolve_repo_path(
        preliminary_experiment.get("output_dir"),
        REPO_ROOT
        / "outputs/paper"
        / preliminary_model_key
        / f"l{preliminary_layer}",
    )
    log_path = configure_scheduler_logging(preliminary_output)
    session_id = uuid.uuid4().hex
    LOGGER.info(
        "Scheduler session started session=%s cwd=%s config=%s output=%s "
        "command=%s",
        session_id,
        Path.cwd(),
        config_path.resolve(),
        preliminary_output,
        shlex.join(sys.argv),
    )
    LOGGER.info("Scheduler log path=%s", log_path)
    scheduler: Scheduler | None = None
    try:
        scheduler = Scheduler(config_path, cli)
        for task in scheduler.tasks:
            LOGGER.info(
                "Discovered task=%s method=%s priority=%d gpu_count=%d "
                "vram_reservation_gb=%g exclusive=%s max_retries=%d",
                task.stem,
                task.method,
                task.priority,
                task.gpu_count,
                task.vram_gb,
                task.exclusive,
                task.max_retries,
            )

        def request_stop(signum, _frame):
            signal_name = signal.Signals(signum).name
            if scheduler.received_signal is None:
                scheduler.received_signal = signum
                LOGGER.warning(
                    "Received exit signal=%s(%d); requesting graceful shutdown",
                    signal_name,
                    signum,
                )
            else:
                LOGGER.warning(
                    "Received additional exit signal=%s(%d) while stopping",
                    signal_name,
                    signum,
                )
            scheduler.stop_requested = True

        exit_signals = [signal.SIGTERM, signal.SIGINT]
        for optional_name in ("SIGHUP", "SIGQUIT"):
            optional_signal = getattr(signal, optional_name, None)
            if optional_signal is not None:
                exit_signals.append(optional_signal)
        for exit_signal in exit_signals:
            signal.signal(exit_signal, request_stop)

        exit_code = scheduler.run()
    except Exception:
        stage = (
            scheduler.current_stage
            if scheduler is not None
            else "initialization"
        )
        LOGGER.critical(
            "Unhandled scheduler exception session=%s stage=%s",
            session_id,
            stage,
            exc_info=True,
        )
        exit_code = 1
    final_stage = (
        scheduler.current_stage if scheduler is not None else "initialization"
    )
    LOGGER.info(
        "Scheduler session ended session=%s exit_code=%d stage=%s",
        session_id,
        exit_code,
        final_stage,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
