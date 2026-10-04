"""Run sample-efficiency and subset-sensitivity studies from one YAML."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml
from steerscope.utils.api_clients import apply_api_model_overrides

from steerscope.studies.training_data import (
    StudyVariant,
    aggregate_study_metrics,
    apply_reference_factors,
    build_study_variants,
    collect_variant_metrics,
    derive_variant_config,
    extract_reference_factors,
    render_study_reports,
)


REUSABLE_REFERENCE_TRAIN_ARTIFACTS = (
    "GemmaScopeSAE.pt",
    "GemmaScopeSAE_scale.pt",
)


def _run(command: list[str], cwd: Path) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _reuse_reference_training_artifacts(
    reference_run_dir: Path,
    destination_run_dir: Path,
) -> list[Path]:
    """Seed run-invariant SAE artifacts from the completed reference run."""
    source_dir = Path(reference_run_dir) / "train"
    destination_dir = Path(destination_run_dir) / "train"
    reused = []
    for filename in REUSABLE_REFERENCE_TRAIN_ARTIFACTS:
        source = source_dir / filename
        if not source.is_file():
            continue
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / filename
        if destination.exists():
            print(f"Keeping existing study artifact: {destination}", flush=True)
            reused.append(destination)
            continue
        temporary = destination.with_name(
            f".{destination.name}.shared-{os.getpid()}.tmp"
        )
        temporary.unlink(missing_ok=True)
        try:
            try:
                os.link(source, temporary)
                mode = "hard-linked"
            except OSError:
                shutil.copy2(source, temporary)
                mode = "copied"
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        print(
            f"Reused reference study artifact ({mode}): "
            f"{source} -> {destination}",
            flush=True,
        )
        reused.append(destination)
    return reused


def _write_yaml(path: Path, config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
    os.replace(temporary, path)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _torchrun() -> str:
    beside_python = Path(sys.executable).with_name("torchrun")
    if beside_python.exists():
        return str(beside_python)
    executable = shutil.which("torchrun")
    if executable is None:
        raise FileNotFoundError("torchrun was not found in the active environment.")
    return executable


def _prepare_shared_generate_dir(
    external_dir: str | None,
    study_root: Path,
    config_path: Path,
    repo_root: Path,
) -> Path:
    shared_root = study_root / "shared"
    shared_root.mkdir(parents=True, exist_ok=True)
    if external_dir:
        shared_generate_dir = Path(external_dir).expanduser().resolve()
        metadata_path = shared_generate_dir / "metadata.jsonl"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                "External shared generate directory is incomplete; expected "
                f"metadata at {metadata_path}"
            )
        print(f"Reusing external generated pool: {shared_generate_dir}")
        return shared_generate_dir

    # Generate the largest configured pool exactly once. generate.py is
    # resumable, so restarting the study does not regenerate completed concepts.
    _run(
        [
            sys.executable,
            str(repo_root / "steerscope/scripts/generate.py"),
            "--config",
            str(config_path),
            "--dump_dir",
            str(shared_root),
        ],
        repo_root,
    )
    return shared_root / "generate"


def _write_study_reports(metrics, config, study_root: Path, main_results_root=None) -> list[Path]:
    from steerscope.studies.statistics import load_study_concept_scores
    from steerscope.sweep.paper.generate_configs import METHOD_FILES
    main_root = Path(main_results_root) if main_results_root else study_root.parent
    roots = {method: main_root / "methods" / Path(METHOD_FILES[method]).stem / "evaluate/runs"
             for method in metrics.method.unique()}
    scores = load_study_concept_scores(metrics, config, study_root, roots)
    efficiency, sensitivity = aggregate_study_metrics(metrics, config, concept_scores=scores)
    efficiency_path = study_root / "sample_efficiency.parquet"
    sensitivity_path = study_root / "sample_sensitivity.parquet"
    efficiency.to_parquet(efficiency_path, index=False)
    sensitivity.to_parquet(sensitivity_path, index=False)
    report_paths = render_study_reports(
        efficiency, sensitivity, study_root / "reports"
    )
    print(f"Sample efficiency: {efficiency_path}")
    print(f"Sample sensitivity: {sensitivity_path}")
    for path in report_paths:
        print(f"Report: {path}")
    return report_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dump_dir", required=True)
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument(
        "--shared_generate_dir",
        help=(
            "Reuse an existing generate directory instead of generating a "
            "study-local copy. The directory must contain metadata.jsonl."
        ),
    )
    parser.add_argument(
        "--reference_metrics",
        help=(
            "Completed main-run BestFactorEvaluator metrics used as the "
            "external full-data factor reference."
        ),
    )
    parser.add_argument(
        "--reports_only",
        action="store_true",
        help="Rebuild study summaries and plots from the existing metrics.parquet.",
    )
    parser.add_argument("--main_results_root", help="Main sweep root with raw ID judge samples for the full-data reference.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with config_path.open(encoding="utf-8") as file:
        config = apply_api_model_overrides(yaml.safe_load(file) or {})
    variants = build_study_variants(config)

    repo_root = Path(__file__).resolve().parents[2]
    output_root = Path(args.dump_dir).resolve()
    study_root = output_root / "studies"
    if args.reports_only:
        metrics_path = study_root / "metrics.parquet"
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Combined study metrics not found: {metrics_path}"
            )
        metrics = pd.read_parquet(metrics_path)
        _write_study_reports(metrics, config, study_root, args.main_results_root)
        return

    configs_dir = study_root / "configs"
    runs_root = study_root / "runs"
    study_root.mkdir(parents=True, exist_ok=True)
    shared_generate_dir = _prepare_shared_generate_dir(
        args.shared_generate_dir,
        study_root,
        config_path,
        repo_root,
    )

    metric_frames = []
    evaluate = config.get("evaluate") or {}
    evaluation_run_id = str(
        evaluate.get("evaluation_run_id")
        or evaluate.get("run_name")
        or "default"
    )
    reference_variants = [
        variant for variant in variants if variant.factor_reference
    ]
    if len(reference_variants) > 1:
        raise ValueError("A study can configure only one factor reference run.")
    reference_variant = reference_variants[0] if reference_variants else None
    reference_factors = None
    factor_selection = (config.get("study") or {}).get("factor_selection") or {}
    if bool(factor_selection.get("external", False)):
        if not args.reference_metrics:
            raise ValueError(
                "study.factor_selection.external requires --reference_metrics."
            )
        reference_path = Path(args.reference_metrics).expanduser().resolve()
        if not reference_path.is_file():
            raise FileNotFoundError(
                f"External factor reference metrics not found: {reference_path}"
            )
        reference_metrics = pd.read_parquet(reference_path)
        external_variant = StudyVariant(
            train_examples=int(factor_selection["train_examples"]),
            subset_seed=int(factor_selection.get("subset_seed", 42)),
            efficiency=True,
            factor_reference=True,
        )
        reference_metrics = reference_metrics.copy()
        reference_metrics.insert(0, "source_evaluator", config["study"]["result"]["evaluator"])
        reference_metrics.insert(0, "sensitivity", False)
        reference_metrics.insert(0, "efficiency", True)
        reference_metrics.insert(0, "factor_reference", True)
        reference_metrics.insert(0, "subset_seed", external_variant.subset_seed)
        reference_metrics.insert(0, "train_examples", external_variant.train_examples)
        reference_metrics.insert(0, "study_run_id", "external-main-reference")
        reference_factors = extract_reference_factors(
            reference_metrics, config
        )
        metric_frames.append(reference_metrics)
        factor_path = study_root / "reference_best_factors.json"
        _write_json(factor_path, {
            "reference_run_id": "external-main-reference",
            "train_examples": external_variant.train_examples,
            "subset_seed": external_variant.subset_seed,
            "source": str(reference_path),
            "factors": reference_factors,
        })
        print(f"External reference factors: {factor_path}")
    for index, variant in enumerate(variants, start=1):
        print(
            f"Study run {index}/{len(variants)}: {variant.run_id}", flush=True
        )
        variant_config = config
        if factor_selection and not variant.factor_reference:
            if reference_factors is None:
                raise RuntimeError(
                    "The factor reference run must complete before fixed-factor runs."
                )
            variant_config = apply_reference_factors(
                config, reference_factors
            )
        derived = derive_variant_config(
            variant_config, variant, shared_generate_dir
        )
        runtime_config = configs_dir / f"{variant.run_id}.yaml"
        run_dir = runs_root / variant.run_id
        _write_yaml(runtime_config, derived)
        if reference_variant is not None and not variant.factor_reference:
            _reuse_reference_training_artifacts(
                runs_root / reference_variant.run_id,
                run_dir,
            )
        _run(
            [
                _torchrun(),
                "--standalone",
                f"--nproc_per_node={int(args.nproc_per_node)}",
                str(repo_root / "steerscope/scripts/train.py"),
                "--config",
                str(runtime_config),
                "--dump_dir",
                str(run_dir),
            ],
            repo_root,
        )
        _run(
            [
                sys.executable,
                str(repo_root / "steerscope/scripts/evaluate.py"),
                "--config",
                str(runtime_config),
                "--mode",
                "steering",
                "--dump_dir",
                str(run_dir),
            ],
            repo_root,
        )
        variant_metrics = collect_variant_metrics(
            run_dir, evaluation_run_id, variant
        )
        metric_frames.append(variant_metrics)
        if variant.factor_reference:
            reference_factors = extract_reference_factors(
                variant_metrics, config
            )
            factor_path = study_root / "reference_best_factors.json"
            _write_json(factor_path, {
                "reference_run_id": variant.run_id,
                "train_examples": variant.train_examples,
                "subset_seed": variant.subset_seed,
                "factors": reference_factors,
            })
            print(f"Reference factors: {factor_path}")

    metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
    metrics_path = study_root / "metrics.parquet"
    metrics.to_parquet(metrics_path, index=False)
    print(f"Combined metrics: {metrics_path}")
    _write_study_reports(metrics, config, study_root, args.main_results_root)


if __name__ == "__main__":
    main()
