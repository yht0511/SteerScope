"""Configuration and reporting for training-data studies."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class StudyVariant:
    train_examples: int
    subset_seed: int
    efficiency: bool = False
    sensitivity: bool = False
    factor_reference: bool = False

    @property
    def run_id(self) -> str:
        return f"n-{self.train_examples:04d}_subset-{self.subset_seed}"


def _positive_even(value: Any, name: str) -> int:
    value = int(value)
    if value <= 0 or value % 2:
        raise ValueError(f"{name} must be a positive even integer.")
    return value


def build_study_variants(config: Mapping[str, Any]) -> list[StudyVariant]:
    """Expand efficiency and sensitivity declarations into unique runs."""
    study = config.get("study") or {}
    efficiency = study.get("sample_efficiency") or {}
    sensitivity = study.get("sample_sensitivity") or {}
    factor_selection = study.get("factor_selection") or {}
    if not efficiency and not sensitivity:
        raise ValueError(
            "study must configure sample_efficiency and/or sample_sensitivity."
        )

    variants: dict[tuple[int, int], dict[str, bool]] = {}
    if efficiency:
        sizes = efficiency.get("sizes") or []
        if not sizes:
            raise ValueError("study.sample_efficiency.sizes cannot be empty.")
        seed = int(efficiency.get("subset_seed", 42))
        for raw_size in sizes:
            size = _positive_even(raw_size, "sample_efficiency size")
            variants.setdefault((size, seed), {}).update(efficiency=True)

    if sensitivity:
        size = _positive_even(
            sensitivity.get("size"), "sample_sensitivity size"
        )
        seeds = sensitivity.get("subset_seeds") or []
        if len(seeds) < 2:
            raise ValueError(
                "study.sample_sensitivity.subset_seeds needs at least two seeds."
            )
        for raw_seed in seeds:
            seed = int(raw_seed)
            variants.setdefault((size, seed), {}).update(sensitivity=True)

    if factor_selection:
        size = _positive_even(
            factor_selection.get("train_examples"),
            "factor_selection train_examples",
        )
        seed = int(factor_selection.get("subset_seed", 42))
        if not bool(factor_selection.get("external", False)):
            variants.setdefault((size, seed), {}).update(factor_reference=True)

    pool_size = (config.get("generate") or {}).get("num_of_examples")
    if pool_size is None:
        raise ValueError(
            "generate.num_of_examples is required for a training-data study."
        )
    pool_size = int(pool_size)
    largest = max(size for size, _ in variants)
    if largest > pool_size:
        raise ValueError(
            f"Largest study size is {largest}, but generate.num_of_examples "
            f"only creates {pool_size} rows per concept."
        )

    return [
        StudyVariant(
            train_examples=size,
            subset_seed=seed,
            efficiency=flags.get("efficiency", False),
            sensitivity=flags.get("sensitivity", False),
            factor_reference=flags.get("factor_reference", False),
        )
        for (size, seed), flags in sorted(
            variants.items(),
            key=lambda item: (
                not item[1].get("factor_reference", False),
                item[0],
            ),
        )
    ]


def derive_variant_config(
    config: Mapping[str, Any],
    variant: StudyVariant,
    shared_generate_dir: str | Path,
) -> dict[str, Any]:
    """Create the ordinary one-run YAML consumed by train.py/evaluate.py."""
    derived = copy.deepcopy(dict(config))
    study = derived.pop("study", None) or {}
    apply_variant_inference_settings(derived, variant, study)
    train = derived.setdefault("train", {})
    train["max_num_of_examples"] = int(variant.train_examples)
    train["subset_seed"] = int(variant.subset_seed)
    shared_generate_dir = str(Path(shared_generate_dir).resolve())
    train["overwrite_data_dir"] = shared_generate_dir
    train["overwrite_metadata_dir"] = shared_generate_dir
    return derived


def apply_variant_inference_settings(
    config: dict[str, Any],
    variant: StudyVariant,
    study: Mapping[str, Any],
) -> None:
    """Apply sampling controls that belong only to one study variant."""
    if variant.sensitivity and variant.efficiency:
        raise ValueError("Efficiency and sensitivity cannot share a run with different prompt/decoding settings.")
    kind = "sample_sensitivity" if variant.sensitivity else "sample_efficiency"
    settings = study.get(kind) or {}
    count = int(settings.get("num_examples", 20 if variant.sensitivity else 10))
    if count < 1:
        raise ValueError("Study num_examples must be positive.")
    for node in (config.get("evaluate", {}).get("evaluators") or {}).values():
        if isinstance(node, dict) and node.get("dataset"):
            node["dataset"]["num_examples"] = count
    temperature = settings.get("temperature")
    do_sample = settings.get("do_sample")
    if temperature is not None:
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError(
                "study.sample_sensitivity.temperature must be finite and "
                "non-negative."
            )
    if do_sample is not None and not isinstance(do_sample, bool):
        raise TypeError("study.sample_sensitivity.do_sample must be boolean.")

    evaluate = config.setdefault("evaluate", {})
    if temperature is not None:
        evaluate["temperature"] = temperature
    for node in (evaluate.get("evaluators") or {}).values():
        if not isinstance(node, dict):
            continue
        inference = node.get("inference")
        if not isinstance(inference, dict):
            continue
        if temperature is not None:
            inference["temperature"] = temperature
        if do_sample is not None:
            inference["do_sample"] = do_sample


def extract_reference_factors(
    metrics: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Read one method-level factor per method from the reference run."""
    evaluator_id, _, _, _, expected_methods = _factor_scan_details(config)
    selected = metrics[metrics["source_evaluator"] == evaluator_id].copy()
    required = {"method", "factor"}
    missing = sorted(required.difference(selected.columns))
    if missing:
        raise ValueError(
            f"Reference evaluator '{evaluator_id}' is missing columns: {missing}."
        )
    if selected.empty:
        raise ValueError(
            f"Reference evaluator '{evaluator_id}' produced no factor rows."
        )
    if selected["method"].duplicated().any():
        duplicated = sorted(
            selected.loc[
                selected["method"].duplicated(keep=False), "method"
            ].astype(str).unique()
        )
        raise ValueError(
            "Reference factor selection requires one row per method; "
            f"duplicates found for {duplicated}."
        )
    selected["factor"] = pd.to_numeric(selected["factor"], errors="coerce")
    if (
        selected["factor"].isna().any()
        or not selected["factor"].map(math.isfinite).all()
    ):
        raise ValueError(
            "Reference factor selection contains non-finite factors."
        )
    factors = dict(zip(selected["method"].astype(str), selected["factor"]))
    expected_method_set = set(expected_methods)
    missing_methods = sorted(expected_method_set.difference(factors))
    if missing_methods:
        raise ValueError(
            "Reference factor selection is missing methods: "
            f"{missing_methods}."
        )
    unexpected_methods = sorted(set(factors).difference(expected_method_set))
    if unexpected_methods:
        raise ValueError(
            "Reference factor selection contains unexpected methods: "
            f"{unexpected_methods}."
        )
    return {method: float(factors[method]) for method in sorted(expected_methods)}


def apply_reference_factors(
    config: Mapping[str, Any],
    factors: Mapping[str, float],
) -> dict[str, Any]:
    """Restrict inference to reference factors plus the required baseline."""
    derived = copy.deepcopy(dict(config))
    _, selector, _, scan, methods = _factor_scan_details(derived)
    missing_methods = sorted(set(methods).difference(factors))
    if missing_methods:
        raise ValueError(f"Fixed factors are missing methods: {missing_methods}.")
    strengths_by_model = {
        method: [float(factors[method])]
        for method in methods
    }

    params = selector.get("params") or {}
    baseline_factor = params.get("baseline_factor")
    fallback_method = params.get("fallback_baseline_method")
    if baseline_factor is not None:
        baseline_factor = float(baseline_factor)
        if fallback_method is None:
            for method in methods:
                strengths_by_model[method] = list(dict.fromkeys([
                    baseline_factor,
                    *strengths_by_model[method],
                ]))
        else:
            fallback_method = str(fallback_method)
            if fallback_method not in strengths_by_model:
                raise ValueError(
                    "fallback_baseline_method is absent from evaluate.models: "
                    f"{fallback_method}."
                )
            strengths_by_model[fallback_method] = list(dict.fromkeys([
                baseline_factor,
                *strengths_by_model[fallback_method],
            ]))
    inference = scan.setdefault("inference", {})
    # Full-grid fallbacks must not survive into child runs: complete method
    # coverage below is deliberate and makes accidental rescans impossible.
    inference.pop("strengths", None)
    inference.pop("factors", None)
    inference["strengths_by_model"] = strengths_by_model
    return derived


def _factor_scan_details(config: Mapping[str, Any]):
    study = config.get("study") or {}
    evaluator_id, _ = _configured_result(study, config)
    evaluate = config.get("evaluate") or {}
    evaluators = evaluate.get("evaluators") or {}
    selector = evaluators.get(evaluator_id)
    if not isinstance(selector, Mapping):
        raise ValueError(
            f"Study result evaluator '{evaluator_id}' is not configured."
        )
    group_by = (selector.get("params") or {}).get("group_by")
    if isinstance(group_by, str):
        group_by = [group_by]
    if list(group_by or ()) != ["method"]:
        raise ValueError(
            "Fixed reference factors require BestFactorEvaluator "
            "params.group_by: [method]."
        )
    source = (selector.get("input") or {}).get("from")
    if source is None:
        dependencies = selector.get("depends_on") or []
        if len(dependencies) != 1:
            raise ValueError(
                f"Study result evaluator '{evaluator_id}' must identify one "
                "factor-scan input."
            )
        source = dependencies[0]
    scan = evaluators.get(source)
    if not isinstance(scan, Mapping):
        raise ValueError(f"Factor-scan evaluator '{source}' is not configured.")
    methods = scan.get("models") or evaluate.get("models") or []
    methods = [str(method) for method in methods]
    if not methods or len(methods) != len(set(methods)):
        raise ValueError(
            f"Factor-scan evaluator '{source}' requires unique models."
        )
    return evaluator_id, selector, str(source), scan, methods


def _configured_result(study: Mapping[str, Any], config: Mapping[str, Any]):
    result = study.get("result") or {}
    evaluator_id = result.get("evaluator")
    evaluators = (config.get("evaluate") or {}).get("evaluators") or {}
    if evaluator_id is None:
        evaluator_id = next(
            (
                node_id
                for node_id, node in evaluators.items()
                if isinstance(node, Mapping)
                and node.get("type") == "BestFactorEvaluator"
            ),
            None,
        )
    if evaluator_id is None:
        raise ValueError(
            "study.result.evaluator is required when no BestFactorEvaluator "
            "is configured."
        )
    node = evaluators.get(evaluator_id) or {}
    default_metric = (
        "selected_improvement"
        if node.get("type") == "BestFactorEvaluator"
        else "lm_judge_rating"
    )
    return str(evaluator_id), str(result.get("metric", default_metric))


def collect_variant_metrics(
    run_dir: str | Path,
    evaluation_run_id: str,
    variant: StudyVariant,
) -> pd.DataFrame:
    root = (
        Path(run_dir)
        / "evaluate"
        / "runs"
        / evaluation_run_id
        / "evaluators"
    )
    frames = []
    if not root.exists():
        raise FileNotFoundError(f"Evaluation result directory not found: {root}")
    for path in sorted(root.glob("*/metrics.parquet")):
        frame = pd.read_parquet(path)
        frame.insert(0, "source_evaluator", path.parent.name)
        frame.insert(0, "sensitivity", variant.sensitivity)
        frame.insert(0, "efficiency", variant.efficiency)
        frame.insert(0, "factor_reference", variant.factor_reference)
        frame.insert(0, "subset_seed", variant.subset_seed)
        frame.insert(0, "train_examples", variant.train_examples)
        frame.insert(0, "study_run_id", variant.run_id)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No evaluator metrics found below {root}")
    return pd.concat(frames, ignore_index=True, sort=False)


def aggregate_study_metrics(metrics, config, *, concept_scores=None):
    """Aggregate paired concept effects with the same definitions as analysis."""
    from .statistics import align_efficiency_concepts, summarize_sensitivity

    evaluator_id, metric = _configured_result(config.get("study") or {}, config)
    if concept_scores is None:
        selected = metrics[metrics["source_evaluator"] == evaluator_id].copy()
        if "concept_id" not in selected or selected["concept_id"].isna().any():
            raise ValueError("Study summaries require paired concept scores, not method-level averages.")
        concept_scores = selected.rename(columns={metric: "score"})
    rows = concept_scores.copy()
    if "factor_reference" not in rows:
        rows["factor_reference"] = False
    efficiency_rows = rows[rows["efficiency"].fillna(False)].copy()
    efficiency_rows = align_efficiency_concepts(efficiency_rows)
    per_run = efficiency_rows.groupby(
        ["study_run_id", "train_examples", "subset_seed", "method", "factor_reference"],
        as_index=False,
    ).agg(score=("score", "mean"), n_concepts=("concept_id", "nunique"))
    efficiency = per_run.groupby(["train_examples", "method"], as_index=False).agg(
        score=("score", "mean"), score_std=("score", "std"),
        runs=("score", "size"), n_concepts=("n_concepts", "min"),
    )
    efficiency["score_std"] = efficiency["score_std"].fillna(0.)
    relative = ((config.get("study") or {}).get("sample_efficiency") or {}).get("relative_report")
    if relative and not efficiency.empty:
        reference = per_run[per_run.factor_reference].copy()
        if reference.empty or reference.method.duplicated().any():
            raise ValueError("Relative sample efficiency requires one full-data reference per method.")
        reference = reference[["method", "train_examples", "score"]].rename(columns={
            "train_examples": "reference_train_examples", "score": "reference_improvement",
        })
        efficiency = efficiency.merge(reference, on="method", how="left", validate="many_to_one")
        if efficiency.reference_improvement.isna().any():
            raise ValueError("Missing full-data reference for one or more study methods.")
        minimum = float(relative.get("min_reference_improvement", .1))
        if not math.isfinite(minimum) or minimum < 0:
            raise ValueError("min_reference_improvement must be finite and non-negative.")
        efficiency["relative_eligible"] = efficiency.reference_improvement.ge(minimum)
        efficiency["relative_improvement_pct"] = (
            100 * efficiency.score / efficiency.reference_improvement.replace(0., float("nan"))
        )
        efficiency["relative_min_reference_improvement"] = minimum
    sensitivity_rows = rows[rows["sensitivity"].fillna(False)]
    sensitivity_config = (config.get("study") or {}).get("sample_sensitivity") or {}
    _, sensitivity = summarize_sensitivity(
        sensitivity_rows, sensitivity_config.get("subset_seeds", [42, 43, 44, 45, 46]),
    )
    efficiency["metric"] = metric
    sensitivity["metric"] = metric
    return efficiency, sensitivity


def render_study_reports(
    efficiency: pd.DataFrame,
    sensitivity: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Render compact cross-run plots; parquet remains the source of truth."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    if not efficiency.empty:
        figure, axis = plt.subplots(figsize=(8.0, 5.0))
        for method, group in efficiency.groupby("method", sort=True):
            group = group.sort_values("train_examples")
            axis.plot(
                group["train_examples"], group["score"], marker="o", label=method
            )
        axis.set_title("Sample efficiency")
        axis.set_xlabel("Training examples per concept")
        axis.set_ylabel(str(efficiency["metric"].iloc[0]))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
        figure.tight_layout()
        path = output_dir / "sample_efficiency.png"
        figure.savefig(path, dpi=300)
        plt.close(figure)
        paths.append(path)

        if "relative_improvement_pct" in efficiency.columns:
            relative = efficiency[efficiency["relative_eligible"]].copy()
            if not relative.empty:
                figure, axis = plt.subplots(figsize=(8.0, 5.0))
                for method, group in relative.groupby("method", sort=True):
                    group = group.sort_values("train_examples")
                    axis.plot(
                        group["train_examples"],
                        group["relative_improvement_pct"],
                        marker="o",
                        label=method,
                    )
                reference_sizes = sorted(
                    relative["reference_train_examples"].unique()
                )
                reference_label = ", ".join(
                    str(int(value)) for value in reference_sizes
                )
                axis.axhline(100.0, color="black", linestyle="--", alpha=0.4)
                axis.set_title(
                    "Sample efficiency relative to full-data improvement"
                )
                axis.set_xlabel("Training examples per concept")
                axis.set_ylabel(
                    f"Percentage of {reference_label}-example improvement (%)"
                )
                axis.grid(alpha=0.25)
                axis.legend(fontsize=8, ncol=2)
                figure.tight_layout()
                path = output_dir / "sample_efficiency_relative.png"
                figure.savefig(path, dpi=300)
                plt.close(figure)
                paths.append(path)

    if not sensitivity.empty:
        for train_examples, group in sensitivity.groupby("train_examples"):
            group = group.sort_values("score_mean")
            figure, axis = plt.subplots(
                figsize=(8.0, max(4.0, 0.45 * len(group)))
            )
            axis.errorbar(
                group["score_mean"],
                group["method"],
                xerr=group["score_std"],
                fmt="o",
                capsize=3,
            )
            axis.set_title(
                f"Sample sensitivity ({int(train_examples)} training examples)"
            )
            axis.set_xlabel(
                f"{sensitivity['metric'].iloc[0]} (mean ± subset std)"
            )
            axis.grid(axis="x", alpha=0.25)
            figure.tight_layout()
            path = output_dir / f"sample_sensitivity_n-{int(train_examples):04d}.png"
            figure.savefig(path, dpi=300)
            plt.close(figure)
            paths.append(path)
    return paths
