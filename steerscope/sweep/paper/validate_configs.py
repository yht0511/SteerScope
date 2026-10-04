#!/usr/bin/env python3
"""Validate the four canonical SteerScope paper configuration suites."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
SCHEDULER_DIR = ROOT / "scheduler_configs"
_GENERATOR_SPEC = importlib.util.spec_from_file_location(
    "steerscope_paper_generate_configs", ROOT / "generate_configs.py"
)
if _GENERATOR_SPEC is None or _GENERATOR_SPEC.loader is None:
    raise RuntimeError("Could not load the paper configuration generator")
generator = importlib.util.module_from_spec(_GENERATOR_SPEC)
_GENERATOR_SPEC.loader.exec_module(generator)
EXTENDED_FACTOR_METHODS = {
    "SphericalSteering",
    "HiDRA",
    "AUSteer",
    "ODESteer",
    "StepODESteer",
    "SPSR",
}


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _expected_suite(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    training = generator.training_models(spec)
    expected = {
        filename: generator.method_config(method, spec, training)
        for method, filename in generator.METHOD_FILES.items()
    }
    expected["generalization.yaml"] = generator.generalization_config(spec, training)
    expected["study.yaml"] = generator.study_config(spec, training)
    return expected


def _factor_grid(config: dict[str, Any], method: str) -> list[float]:
    node = config["evaluate"]["evaluators"]["id_lm_judge"]
    strengths = node["inference"]["strengths_by_model"]
    return list(strengths[method])


def validate() -> list[str]:
    errors: list[str] = []
    profiles = set(generator.MODEL_LAYERS)
    if profiles != {"2b/l10", "2b/l20", "9b/l20", "9b/l31"}:
        errors.append(f"Unexpected generator profiles: {sorted(profiles)}")

    suites = {name: _expected_suite(spec) for name, spec in generator.MODEL_LAYERS.items()}
    suites["2b/l20_10concepts"] = generator.ten_concept_configs()
    for relative, expected in sorted(suites.items()):
        directory = ROOT / relative
        actual_names = {path.name for path in directory.glob("*.yaml")}
        if actual_names != set(expected):
            errors.append(
                f"{relative}: files differ; missing={sorted(set(expected) - actual_names)}, "
                f"extra={sorted(actual_names - set(expected))}"
            )
        for filename, expected_config in expected.items():
            path = directory / filename
            if not path.is_file():
                continue
            actual = _load(path)
            if actual != expected_config:
                errors.append(f"{relative}/{filename}: differs from generate_configs.py")
            if actual.get("generate", {}).get("lm_model") != "DeepSeek-V3.2-Instruct":
                errors.append(f"{relative}/{filename}: wrong generation model")
            if actual.get("evaluate", {}).get("lm_model") != "gpt-4o-mini":
                errors.append(f"{relative}/{filename}: wrong judge model")
            for value in _strings(actual):
                if value.startswith("/"):
                    errors.append(f"{relative}/{filename}: absolute path {value!r}")
                if "/home/" in value or "/mnt/" in value:
                    errors.append(f"{relative}/{filename}: nonportable value {value!r}")

        model_key = relative.split("/", 1)[0]
        for method in sorted(EXTENDED_FACTOR_METHODS):
            filename = generator.METHOD_FILES[method]
            path = directory / filename
            if not path.is_file():
                continue
            factors = _factor_grid(_load(path), method)
            expected_factors = generator.factors_for(method, model_key=model_key)
            if factors != expected_factors:
                errors.append(f"{relative}/{filename}: factor grid differs from generator")
            if len(factors) != 14:
                errors.append(f"{relative}/{filename}: expected 14 factors, got {len(factors)}")

    expected_scheduler_names = {
        f"{profile.replace('/', '_')}.yaml" for profile in suites
    }
    actual_scheduler_names = {
        path.name for path in SCHEDULER_DIR.glob("*.yaml")
    }
    if actual_scheduler_names != expected_scheduler_names:
        errors.append(
            "Scheduler configs differ; "
            f"missing={sorted(expected_scheduler_names - actual_scheduler_names)}, "
            f"extra={sorted(actual_scheduler_names - expected_scheduler_names)}"
        )
    for profile in sorted(suites):
        path = SCHEDULER_DIR / f"{profile.replace('/', '_')}.yaml"
        if not path.is_file():
            continue
        config = _load(path)
        experiment = config.get("experiment", {})
        model_key, layer_name = profile.split("/")
        expected_layer = int(layer_name.split("_")[0].removeprefix("l"))
        if experiment.get("model_key") != model_key:
            errors.append(f"{path.name}: wrong model_key")
        if experiment.get("layer") != expected_layer:
            errors.append(f"{path.name}: wrong layer")
        if experiment.get("config_dir") != f"steerscope/sweep/paper/{profile}":
            errors.append(f"{path.name}: wrong config_dir")
        if experiment.get("output_dir") != f"outputs/paper/{profile}":
            errors.append(f"{path.name}: wrong output_dir")
        if config.get("wandb", {}).get("enabled") is not False:
            errors.append(f"{path.name}: W&B must be opt-in")
        for value in _strings(config):
            if value.startswith("/"):
                errors.append(f"{path.name}: nonportable value {value!r}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    errors = validate()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Validated four paper profiles and the 10-concept suite: 127 experiment YAMLs and 5 scheduler YAMLs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
