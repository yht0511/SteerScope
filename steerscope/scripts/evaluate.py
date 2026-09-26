"""Run evaluator-owned steering inference and scoring as a dependency graph."""

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path


_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path[:] = [
    entry
    for entry in sys.path
    if Path(entry or os.curdir).resolve() != _SCRIPTS_DIR
]

import torch.distributed as dist

import steerscope
from steerscope.evaluation import (
    Artifact,
    Concept,
    EvaluationContext,
    EvaluationEngine,
    EvaluationTarget,
    ResultStore,
    apply_node_overrides,
    context_for_node,
    parse_evaluation_targets,
    parse_evaluator_nodes,
)
from steerscope.inference import SteeringModel, SteeringModelConfig, SteeringRuntime
from steerscope.inference.utils import load_metadata_flatten
from steerscope.evaluation.version import (
    EVALUATION_PIPELINE_VERSION,
    evaluation_source_fingerprint,
    file_signature,
    generated_data_identity,
    path_signature,
)
from steerscope.scripts.args.eval_args import EvalArgs
from steerscope.scripts.args.training_args import TrainingArgs

import logging


logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.WARN,
)
logger = logging.getLogger(__name__)


def _method_requires_checkpoint(method):
    """Whether inference loads learned state from the target artifact."""
    model_class = getattr(steerscope, method, None)
    return bool(
        model_class is None
        or getattr(model_class, "load_trained_weights", True)
    )


def ensure_single_process_environment():
    """Reject multi-process evaluation without initializing distributed state."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise RuntimeError(
            "evaluate.py writes shared outputs and must run as one process."
        )
    if dist.is_available() and dist.is_initialized():
        raise RuntimeError(
            "evaluate.py must run without an initialized torch.distributed "
            "process group. Run it with plain python, not torchrun."
        )


def _file_signature(path):
    return file_signature(path)


def _artifact_dirs_by_model(args, methods, train_dir):
    configured = getattr(args, "artifact_dirs_by_model", None) or {}
    if not isinstance(configured, Mapping):
        raise TypeError("evaluate.artifact_dirs_by_model must be a mapping.")
    unknown = sorted(set(configured).difference(methods))
    if unknown:
        raise ValueError(
            "evaluate.artifact_dirs_by_model configures methods that are not "
            f"evaluated: {unknown}"
        )

    config_dir = Path(args.config_file).resolve().parent
    resolved = {}
    for method in methods:
        if method in configured:
            path = Path(configured[method]).expanduser()
            if not path.is_absolute():
                path = config_dir / path
        else:
            path = Path(train_dir).expanduser()
        if method in configured:
            path = path.resolve()
            if _method_requires_checkpoint(method) and not path.exists():
                raise FileNotFoundError(
                    f"Checkpoint directory for method '{method}' does not exist: {path}"
                )
            if path.exists() and not path.is_dir():
                raise ValueError(
                    f"Checkpoint path for method '{method}' is not a directory: {path}"
                )
        resolved[method] = path
    return resolved


def _required_artifact_concept_ids(args, nodes, method, all_concept_ids):
    """Return the concept IDs that the routed artifact must cover."""
    relevant = []
    default_methods = tuple(args.models or ())
    for node in nodes:
        if not node.requires_inference:
            continue
        methods = tuple(node.models or default_methods)
        if method in methods:
            relevant.append(node)
    if not relevant:
        return set(all_concept_ids)

    required = set()
    for node in relevant:
        configured_ids = (node.concepts or {}).get("ids")
        if configured_ids is None:
            return set(all_concept_ids)
        if isinstance(configured_ids, int):
            configured_ids = [configured_ids]
        if isinstance(configured_ids, (str, bytes)):
            raise TypeError(
                f"Evaluator '{node.node_id}' concepts.ids must be a list of "
                "integer concept IDs."
            )
        required.update(int(value) for value in configured_ids)
    return required


def _validate_explicit_artifact_dirs(
    args,
    nodes,
    targets,
    root_dump_dir,
    training_args,
):
    """Validate scheduler-routed checkpoints before any model is loaded."""
    configured = getattr(args, "artifact_dirs_by_model", None) or {}
    if not configured:
        return

    targets_by_method = {}
    for target in targets:
        targets_by_method.setdefault(target.method, []).append(target)

    data_dir = Path(
        training_args.overwrite_data_dir
        or (Path(root_dump_dir) / "generate")
    )
    metadata_dir = Path(
        training_args.overwrite_metadata_dir or data_dir
    )
    expected_generated_data = generated_data_identity(
        data_dir,
        metadata_dir=metadata_dir,
        use_dpo_loss=bool(getattr(training_args, "use_dpo_loss", False)),
    )
    expected_component = getattr(training_args, "component", None)
    configured_layers = set(int(value) for value in (args.steering_layers or ()))
    if args.steering_layer is not None:
        configured_layers.add(int(args.steering_layer))

    for method in sorted(configured):
        # Prompt-only/stateless methods intentionally own no checkpoint or
        # artifact manifest.  Use the model contract rather than a method-name
        # special case so future stateless methods follow the same path.
        if not _method_requires_checkpoint(method):
            continue
        method_targets = targets_by_method.get(method) or []
        if not method_targets:
            continue
        artifact_dir = Path(method_targets[0].artifact.path)
        manifest_path = artifact_dir / "artifact_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Explicit checkpoint directory for method '{method}' is "
                f"missing artifact_manifest.json: {artifact_dir}"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError) as error:
            raise ValueError(
                f"Invalid artifact manifest for method '{method}' at "
                f"{manifest_path}: {error}"
            ) from error
        if manifest.get("version") != 2:
            raise ValueError(
                f"Artifact manifest for method '{method}' has unsupported "
                f"version {manifest.get('version')!r}; expected 2."
            )

        expected_base_models = {target.base_model for target in method_targets}
        if manifest.get("base_model") not in expected_base_models:
            raise ValueError(
                f"Artifact/base-model mismatch for method '{method}': "
                f"manifest has {manifest.get('base_model')!r}, evaluation "
                f"requires {sorted(expected_base_models)!r}."
            )
        if configured_layers and set([int(manifest.get("layer", -1))]) != configured_layers:
            raise ValueError(
                f"Artifact/layer mismatch for method '{method}': manifest "
                f"has {manifest.get('layer')!r}, evaluation requires "
                f"{sorted(configured_layers)!r}."
            )
        if (
            expected_component is not None
            and manifest.get("component") != expected_component
        ):
            raise ValueError(
                f"Artifact/component mismatch for method '{method}': "
                f"manifest has {manifest.get('component')!r}, training "
                f"configuration requires {expected_component!r}."
            )

        method_manifest = (manifest.get("methods") or {}).get(method)
        if not isinstance(method_manifest, Mapping) or not method_manifest.get(
            "fingerprint"
        ):
            raise ValueError(
                f"Artifact manifest at {manifest_path} does not contain a "
                f"fingerprinted entry for method '{method}'."
            )

        all_ids = {
            int(target.concept.concept_id) for target in method_targets
        }
        required_ids = _required_artifact_concept_ids(
            args, nodes, method, all_ids
        )
        artifact_ids = {
            int(value) for value in (manifest.get("concept_ids") or ())
        }
        missing_ids = sorted(required_ids.difference(artifact_ids))
        if missing_ids:
            raise ValueError(
                f"Artifact/concept mismatch for method '{method}': checkpoint "
                f"does not cover required concept IDs {missing_ids}."
            )
        if manifest.get("generated_data") != expected_generated_data:
            raise ValueError(
                f"Artifact/generated-data mismatch for method '{method}': "
                "the checkpoint was trained from a different metadata or "
                "training parquet pool."
            )


def _implicit_targets(args, nodes, metadata, train_dir):
    methods = sorted({
        method
        for node in nodes
        for method in (node.models or tuple(args.models or ()))
    })
    if not methods:
        raise ValueError("Evaluator nodes must configure models, or evaluate.models must be set.")
    base_model = args.steering_model_name or args.model_name
    if not base_model:
        raise ValueError("evaluate.model_name or evaluate.steering_model_name is required.")
    artifact_dirs = _artifact_dirs_by_model(args, methods, train_dir)
    return [
        EvaluationTarget(
            target_id=f"{method}/concept-{int(item['concept_id'])}",
            method=method,
            concept=Concept(
                concept_id=int(item["concept_id"]),
                text=str(item["concept"]),
                ref=item.get("ref"),
                metadata=dict(item),
            ),
            base_model=base_model,
            artifact=Artifact(kind="checkpoint", path=artifact_dirs[method]),
        )
        for item in metadata
        for method in methods
    ]


def _load_targets(args, nodes, root_dump_dir, training_args):
    train_dir = Path(root_dump_dir) / "train"
    if args.targets:
        targets = parse_evaluation_targets(
            args.targets,
            default_base_model=args.steering_model_name or args.model_name,
            base_dir=Path(args.config_file).resolve().parent,
        )
    else:
        metadata_dir = (
            Path(training_args.overwrite_metadata_dir)
            if training_args.overwrite_metadata_dir
            else Path(root_dump_dir) / "generate"
        )
        targets = _implicit_targets(
            args,
            nodes,
            load_metadata_flatten(metadata_dir),
            train_dir,
        )

    base_models = {target.base_model for target in targets}
    if len(base_models) != 1:
        raise ValueError("All targets in one evaluation run must use one base model.")
    keys = [(target.method, target.concept.concept_id) for target in targets]
    if len(keys) != len(set(keys)):
        raise ValueError("Targets must have unique (method, concept.id) pairs.")
    configured_methods = {target.method for target in targets}
    referenced_methods = {
        method
        for node in nodes
        for method in (node.models or tuple(args.models or ()))
    }
    unknown = referenced_methods.difference(configured_methods)
    if unknown:
        raise ValueError(
            f"Evaluator nodes reference methods without targets: {sorted(unknown)}"
        )
    _validate_explicit_artifact_dirs(
        args,
        nodes,
        targets,
        root_dump_dir,
        training_args,
    )
    return targets


def _engine_context(args, targets, root_dump_dir, training_args, nodes=()):
    train_dir = Path(root_dump_dir) / "train"
    training_models = getattr(training_args, "models", {})
    method_artifact_roots = {}
    for target in targets:
        if target.artifact.kind != "checkpoint":
            continue
        existing = method_artifact_roots.setdefault(target.method, target.artifact.path)
        if existing != target.artifact.path:
            method_artifact_roots[target.method] = None
    artifact_signature_cache = {}
    checkpoint_signature_cache = {}
    method_artifact_signature_cache = {}

    def artifact_signature(target):
        key = (
            target.artifact.kind,
            target.artifact.path,
            target.artifact.fingerprint,
        )
        if key not in artifact_signature_cache:
            artifact_signature_cache[key] = target.artifact.signature()
        return artifact_signature_cache[key]

    def checkpoint_signature(method):
        path = method_artifact_roots.get(method)
        if path is None:
            return None
        if path not in checkpoint_signature_cache:
            manifest = Path(path) / "artifact_manifest.json"
            checkpoint_signature_cache[path] = (
                _file_signature(manifest)
                if manifest.is_file()
                else None
            )
        return checkpoint_signature_cache[path]

    def method_artifact_signatures(method):
        """Use training manifests as cache identities without hashing large checkpoint trees."""
        if method in method_artifact_signature_cache:
            return method_artifact_signature_cache[method]
        manifest_signature = checkpoint_signature(method)
        if manifest_signature is not None:
            signatures = {
                "weight": None,
                "bias": None,
                "scale": None,
                "model": None,
                "top_features": None,
                "prompts": None,
                "lora": None,
                "checkpoint": manifest_signature,
            }
        else:
            signatures = {
                "weight": _file_signature(train_dir / f"{method}_weight.pt"),
                "bias": _file_signature(train_dir / f"{method}_bias.pt"),
                "scale": _file_signature(train_dir / f"{method}_scale.pt"),
                "model": _file_signature(train_dir / f"{method}.pt"),
                "top_features": _file_signature(
                    train_dir / f"{method}_top_features.json"
                ),
                "prompts": _file_signature(
                    train_dir / f"{method}_prompts.json"
                ),
                "lora": _path_signature(
                    train_dir / _method_artifact_directory(method)
                ),
                "checkpoint": None,
            }
        method_artifact_signature_cache[method] = signatures
        return signatures

    shared = {
        "pipeline_version": EVALUATION_PIPELINE_VERSION,
        "source_fingerprint": evaluation_source_fingerprint(),
        "base_model": targets[0].base_model,
        "evaluation": {
            "steering_layers": list(args.steering_layers or ()),
            "steering_layer": args.steering_layer,
            "temperature": args.temperature,
            "steering_output_length": args.steering_output_length,
            "steering_batch_size": args.steering_batch_size,
            "steering_intervention_type": args.steering_intervention_type,
            "seed": args.seed,
            "use_bf16": args.use_bf16,
            "intervene_on_prompt": args.intervene_on_prompt,
            "disable_neuronpedia_max_act": args.disable_neuronpedia_max_act,
            "runtime_backend": getattr(
                args, "runtime_backend", "legacy"
            ),
            "easysteer_url": getattr(args, "easysteer_url", None),
        },
    }
    evaluator_contexts = {}
    for node in nodes:
        node_methods = (
            set(node.models or tuple(args.models or ()))
            if node.requires_inference
            else set()
        )
        node_targets = [
            target for target in targets if target.method in node_methods
        ]
        evaluator_contexts[node.node_id] = {
            "inputs": getattr(steerscope, node.evaluator_type).execution_context(
                node, args
            ),
            "training": {
                method: (
                    vars(training_models[method])
                    if method in training_models
                    else None
                )
                for method in sorted(node_methods)
            },
            "artifacts": {
                method: method_artifact_signatures(method)
                for method in sorted(node_methods)
            },
            "targets": {
                target.target_id: {
                    "method": target.method,
                    "concept_id": target.concept.concept_id,
                    "concept": target.concept.text,
                    "ref": target.concept.ref,
                    "metadata": dict(target.concept.metadata),
                    "artifact": (
                        None
                        if target.artifact.path
                        == method_artifact_roots.get(target.method)
                        else artifact_signature(target)
                    ),
                }
                for target in node_targets
            },
        }
    shared["evaluators"] = evaluator_contexts
    return shared


def _method_artifact_directory(method):
    model_class = getattr(steerscope, method)
    return model_class.artifact_directory or f"__unused__/{method}"


def _path_signature(path):
    return path_signature(path)


def _configured_factors(node, method=None):
    inference = dict(node.inference)
    method_factors = inference.get("strengths_by_model", {})
    if method is not None and method in method_factors:
        factors = method_factors[method]
        if not factors:
            raise ValueError(
                f"Evaluator '{node.node_id}' configures no strengths for "
                f"model '{method}'."
            )
        return [float(factor) for factor in factors]
    factors = inference.get("strengths", inference.get("factors"))
    if factors:
        return [float(factor) for factor in factors]

    if inference.get("select"):
        raise ValueError(
            f"Evaluator '{node.node_id}' uses inference.select and must also "
            "configure the candidate inference.strengths."
        )
    raise ValueError(
        f"Evaluator '{node.node_id}' must configure its own "
        "inference.strengths (or inference.factors)."
    )


def _configured_batch_size(args, method):
    overrides = getattr(args, "batch_size_by_model", {}) or {}
    if not isinstance(overrides, dict):
        raise TypeError("inference.batch_size_by_model must be a mapping.")
    value = overrides.get(method, args.steering_batch_size)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"Inference batch size for model '{method}' must be an integer."
        )
    if value < 1:
        raise ValueError(
            f"Inference batch size for model '{method}' must be positive."
        )
    return value


def _request_cache_dir(args, store: ResultStore) -> Path:
    """Return the optional cross-run request cache without moving run state."""
    configured = getattr(args, "shared_request_cache_dir", None)
    if not configured:
        return store.run_dir / "cache"
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = Path(args.config_file).resolve().parent / path
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _progress_store_root(args, evaluate_dump_dir: Path) -> Path | None:
    """Resolve an optional local mirror for evaluator checkpoint progress."""
    configured = getattr(args, "progress_root", None)
    if not configured:
        return None
    root = Path(configured).expanduser()
    if not root.is_absolute():
        root = Path(args.config_file).resolve().parent / root
    root = root.resolve()

    source = getattr(args, "progress_source_root", None)
    if not source:
        root.mkdir(parents=True, exist_ok=True)
        return root
    source_root = Path(source).expanduser()
    if not source_root.is_absolute():
        source_root = Path(args.config_file).resolve().parent / source_root
    source_root = source_root.resolve()
    evaluate_root = Path(evaluate_dump_dir).resolve()
    try:
        relative = evaluate_root.relative_to(source_root)
    except ValueError as error:
        raise ValueError(
            "evaluate.progress_source_root must contain the evaluation output: "
            f"{evaluate_root} is not below {source_root}."
        ) from error
    mirrored = root / relative / "runs"
    mirrored.mkdir(parents=True, exist_ok=True)
    return mirrored


def _model_config(args, evaluator_class, factor, method):
    layers = args.steering_layers or ()
    return SteeringModelConfig(
        factor=float(factor),
        layer=int(args.steering_layer) if args.steering_layer is not None else None,
        layers=tuple(int(layer) for layer in layers),
        temperature=float(args.temperature),
        do_sample=bool(getattr(args, "do_sample", True)),
        max_new_tokens=int(args.steering_output_length),
        batch_size=_configured_batch_size(args, method),
        seed=int(args.seed),
        intervention_type=args.steering_intervention_type or "addition",
        intervene_on_prompt=bool(args.intervene_on_prompt),
        disable_neuronpedia_max_act=bool(args.disable_neuronpedia_max_act),
        compute_perplexity=bool(evaluator_class.compute_perplexity),
    )


def run_evaluation_graph(
    args,
    root_dump_dir,
    evaluate_dump_dir,
    training_args,
):
    nodes = parse_evaluator_nodes(
        args.evaluators,
        evaluator_resolver=lambda name: getattr(steerscope, name),
    )
    run_id = args.evaluation_run_id or args.run_name or "default"
    store = ResultStore(
        Path(evaluate_dump_dir) / "runs",
        run_id=run_id,
        progress_root=_progress_store_root(args, Path(evaluate_dump_dir)),
    )
    request_cache_dir = _request_cache_dir(args, store)
    report_only = bool(getattr(args, "report_only", False))
    if report_only:
        targets = []
        context = {}
    else:
        targets = _load_targets(args, nodes, root_dump_dir, training_args)
        args.models = sorted({target.method for target in targets})
        args.steering_model_name = targets[0].base_model
        if args.model_name is None:
            args.model_name = targets[0].base_model
        context = _engine_context(
            args, targets, root_dump_dir, training_args, nodes=nodes
        )

    def evaluator_factory(node, result_view):
        node_args = apply_node_overrides(args, node)
        node_args.evaluation_cache_context = context_for_node(
            context, node.node_id
        )
        evaluator_class = getattr(steerscope, node.evaluator_type)
        return evaluator_class(
            node=node,
            context=EvaluationContext(
                args=node_args,
                root_dump_dir=Path(root_dump_dir),
                output_dir=store.node_dir(node.node_id),
                results=result_view,
                progress=None if report_only else store.progress(node.node_id),
            ),
            **dict(node.params),
        )

    def targets_factory(node):
        methods = set(node.models or tuple(args.models))
        node_targets = sorted(
            (target for target in targets if target.method in methods),
            key=lambda target: (target.method, target.concept.concept_id),
        )
        if not node_targets:
            raise ValueError(f"Evaluator '{node.node_id}' has no matching targets.")
        concepts = {
            int(target.concept.concept_id): target.concept
            for target in node_targets
        }
        ordered_concepts = [concepts[key] for key in sorted(concepts)]
        return node_targets, ordered_concepts

    def models_factory(node):
        for existing_runtime in runtimes.values():
            existing_runtime.close()
        runtimes.clear()
        node_args = apply_node_overrides(args, node)
        node_args.evaluation_cache_context = context_for_node(
            context, node.node_id
        )
        node_targets, ordered_concepts = targets_factory(node)
        if not node.requires_inference:
            # Result-only evaluators can still own a concept scope.  They need
            # the run's concept collection to apply that scope to dependency
            # rows, but must not construct model wrappers or a runtime.
            return [], ordered_concepts
        evaluator_class = getattr(steerscope, node.evaluator_type)
        node_args.compute_perplexity = bool(evaluator_class.compute_perplexity)
        runtime = SteeringRuntime(
            args=node_args,
            training_args=training_args,
            root_dump_dir=root_dump_dir,
            output_dir=store.run_dir / "cache",
            cache_dir=store.run_dir / "cache",
            targets=node_targets,
            long_format=True,
        )
        runtimes[node.node_id] = runtime
        models = [
            SteeringModel(
                target=target,
                args=node_args,
                training_args=training_args,
                root_dump_dir=root_dump_dir,
                cache_dir=request_cache_dir,
                config=_model_config(
                    node_args, evaluator_class, factor, target.method
                ),
                runtime=runtime,
            )
            for target in node_targets
            for factor in _configured_factors(node, target.method)
        ]
        return models, ordered_concepts

    runtimes = {}
    engine = EvaluationEngine.for_evaluators(
        nodes=nodes,
        result_store=store,
        evaluator_factory=evaluator_factory,
        models_factory=models_factory,
        targets_factory=targets_factory,
        overwrite=bool(args.overwrite_cache),
        context=context,
        generate_reports=bool(args.generate_reports) or report_only,
        report_only=report_only,
    )
    try:
        return engine.run()
    finally:
        for runtime in runtimes.values():
            runtime.close()


def main():
    custom_args = [{
        "args": ["--mode"],
        "kwargs": {"type": str, "default": "steering"},
    }]
    args = EvalArgs(custom_args=custom_args, section="evaluate", ignore_unknown=True)
    ensure_single_process_environment()
    if args.mode != "steering":
        raise ValueError("The evaluator-owned pipeline only supports --mode steering.")
    training_args = TrainingArgs(
        custom_args=custom_args, section="train", ignore_unknown=True
    )
    if not args.evaluators:
        raise ValueError("evaluate.evaluators is required; the legacy flow was removed.")

    root_dump_dir = Path(args.dump_dir)
    evaluate_dump_dir = (
        Path(args.overwrite_evaluate_dump_dir)
        if args.overwrite_evaluate_dump_dir
        else root_dump_dir / "evaluate"
    )
    evaluate_dump_dir.mkdir(parents=True, exist_ok=True)
    run_evaluation_graph(
        args,
        root_dump_dir=root_dump_dir,
        evaluate_dump_dir=evaluate_dump_dir,
        training_args=training_args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
