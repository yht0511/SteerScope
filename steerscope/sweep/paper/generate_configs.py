"""Generate the four canonical SteerScope paper configurations."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent

BASE_FACTORS = [
    0.2,
    0.4,
    0.6,
    0.8,
    1.0,
    1.2,
    1.4,
    1.6,
    1.8,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
]

METHOD_FACTORS = {
    "PromptSteering": [1.0],
    "SimplePromptSteering": [1.0],
    "FLAS": [1.0, 1.5, 2.0, 2.5, 3.0],
    "SphericalSteering": [
        0.04,
        0.06,
        0.08,
        0.1,
        0.12,
        0.16,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        1.0,
    ],
    "HiDRA": [
        0.12,
        0.24,
        0.36,
        0.5,
        0.6,
        0.72,
        0.84,
        1.0,
        1.2,
        1.5,
        1.8,
        2.0,
        2.4,
        3.0,
    ],
    "AUSteer": [
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
        15.0,
        20.0,
        30.0,
        40.0,
        50.0,
        75.0,
        100.0,
        150.0,
        200.0,
    ],
    "PreferenceVector": [
        2.0,
        4.0,
        6.0,
        8.0,
        10.0,
        12.0,
        14.0,
        16.0,
        18.0,
        20.0,
        25.0,
        30.0,
        40.0,
        50.0,
    ],
    "LoRA": [1.0],
    "LoReFT": [1.0],
    "SFT": [1.0],
}

# ODE time is measured in the model's activation scale, so use the calibrated
# 2B/9B grids instead of pretending that one numeric grid is model agnostic.
MODEL_METHOD_FACTORS = {
    "2b": {
        "ODESteer": [
            5.0, 10.0, 15.0, 20.0, 40.0, 60.0, 80.0,
            100.0, 120.0, 140.0, 180.0, 250.0, 350.0, 500.0,
        ],
        "StepODESteer": [
            10.0, 15.0, 20.0, 44.0, 88.0, 132.0, 176.0,
            220.0, 308.0, 440.0, 550.0, 660.0, 880.0, 1100.0,
        ],
    },
    "9b": {
        "ODESteer": [
            10.0, 15.0, 18.0, 20.0, 36.0, 54.0, 72.0,
            90.0, 108.0, 144.0, 180.0, 225.0, 360.0, 450.0,
        ],
        "StepODESteer": [
            10.0, 15.0, 20.0, 29.0, 58.0, 87.0, 116.0,
            145.0, 174.0, 232.0, 261.0, 290.0, 435.0, 725.0,
        ],
    },
}
MODEL_SPECIFIC_FACTOR_METHODS = frozenset(
    method
    for configured in MODEL_METHOD_FACTORS.values()
    for method in configured
)

METHOD_FILES = {
    "PromptSteering": "prompt_steering.yaml",
    "SimplePromptSteering": "simple_prompt_steering.yaml",
    "DiffMean": "diff_mean.yaml",
    "PCA": "pca.yaml",
    "LAT": "lat.yaml",
    "Random": "random.yaml",
    "LinearProbe": "linear_probe.yaml",
    "SteeringVector": "steering_vector.yaml",
    "LsReFT": "lsreft.yaml",
    "HyperSteer": "hypersteer.yaml",
    "GemmaScopeSAE": "gemmascope_sae.yaml",
    "GemmaScopeSAEMaxAUC": "gemmascope_sae_max_auc.yaml",
    "LoRA": "lora.yaml",
    "LoReFT": "loreft.yaml",
    "SFT": "sft.yaml",
    "FLAS": "flas.yaml",
    "APSR": "apsr.yaml",
    "SPSR": "spsr.yaml",
    "SphericalSteering": "spherical_steering.yaml",
    "HiDRA": "hidra.yaml",
    "AUSteer": "austeer.yaml",
    "ODESteer": "odesteer.yaml",
    "StepODESteer": "step_odesteer.yaml",
    "PreferenceVector": "preference_vector.yaml",
}

EASYSTEER_METHODS = {
    "DiffMean",
    "PCA",
    "LAT",
    "Random",
    "LinearProbe",
    "SteeringVector",
    "LsReFT",
    "GemmaScopeSAE",
    "GemmaScopeSAEMaxAUC",
}

FIXED_FACTOR_METHODS = {
    "PromptSteering",
    "SimplePromptSteering",
    "LoRA",
    "LoReFT",
    "SFT",
}

SLOW_EVALUATION_CONCEPTS = {
    "count": 50,
    "seed": 42,
}

SFT_CONCEPTS = {
    "count": 20,
    "seed": 42,
}

ALL_CONCEPT_METHODS = {"HyperSteer", "FLAS"}
TEN_CONCEPT_SENSITIVITY_EXCLUDED = {
    *ALL_CONCEPT_METHODS,
    "SFT",
}

MODEL_LAYERS = {
    "2b/l10": {
        "model_name": "google/gemma-2-2b-it",
        "hypernet_name": "google/gemma-2-2b",
        "layer": 10,
        "concept_path": "steerscope/data/gemma-2-2b_10-gemmascope-res-16k.json",
        "adapter_layers": [5, 10, 15, 20],
        "steering_batch_size": 10,
        "flas_batch_size": 32,
        "flas_gradient_accumulation_steps": 1,
        "sft_gradient_accumulation_steps": 24,
    },
    "2b/l20": {
        "model_name": "google/gemma-2-2b-it",
        "hypernet_name": "google/gemma-2-2b",
        "layer": 20,
        "concept_path": "steerscope/data/gemma-2-2b_20-gemmascope-res-16k.json",
        "adapter_layers": [5, 10, 15, 20],
        "steering_batch_size": 10,
        "flas_batch_size": 32,
        "flas_gradient_accumulation_steps": 1,
        "sft_gradient_accumulation_steps": 24,
    },
    "9b/l20": {
        "model_name": "google/gemma-2-9b-it",
        "hypernet_name": "google/gemma-2-9b",
        "layer": 20,
        "concept_path": (
            "steerscope/data/gemma-2-9b-it_20-gemmascope-res-131k.json"
        ),
        "adapter_layers": [12, 20, 31, 39],
        "steering_batch_size": 5,
        "flas_batch_size": 16,
        "flas_gradient_accumulation_steps": 2,
        "sft_batch_size": 1,
        "sft_gradient_accumulation_steps": 36,
    },
    "9b/l31": {
        "model_name": "google/gemma-2-9b-it",
        "hypernet_name": "google/gemma-2-9b",
        "layer": 31,
        "concept_path": (
            "steerscope/data/gemma-2-9b-it_31-gemmascope-res-131k.json"
        ),
        "adapter_layers": [12, 20, 31, 39],
        "steering_batch_size": 5,
        "flas_batch_size": 16,
        "flas_gradient_accumulation_steps": 2,
        "sft_batch_size": 1,
        "sft_gradient_accumulation_steps": 36,
    },
}


def model_key_for_spec(spec: dict) -> str:
    model_name = str(spec["model_name"])
    if "2b" in model_name:
        return "2b"
    if "9b" in model_name:
        return "9b"
    raise ValueError(f"Unsupported model scale for factor grid: {model_name}")


def factors_for(
    method: str,
    *,
    include_baseline: bool = False,
    model_key: str | None = None,
) -> list[float]:
    if (
        method in MODEL_SPECIFIC_FACTOR_METHODS
        and model_key not in MODEL_METHOD_FACTORS
    ):
        raise ValueError(
            f"{method} requires a model_key in "
            f"{sorted(MODEL_METHOD_FACTORS)}."
        )
    model_factors = MODEL_METHOD_FACTORS.get(model_key or "", {})
    factors = list(
        model_factors.get(method, METHOD_FACTORS.get(method, BASE_FACTORS))
    )
    if include_baseline and method not in FIXED_FACTOR_METHODS:
        factors = list(dict.fromkeys([0.0, *factors]))
    return factors


def training_models(spec: dict) -> dict:
    adapter_layers = list(spec["adapter_layers"])
    is_9b = spec["model_name"] == "google/gemma-2-9b-it"
    return {
        "PromptSteering": {
            "lm_model": "DeepSeek-V3.2-Instruct",
            "prompt_temperature": 0.0,
        },
        "LsReFT": {
            "batch_size": 6,
            "gradient_accumulation_steps": 1,
            "n_epochs": 3,
            "lr": 0.005 if is_9b else 0.01,
            "weight_decay": 0.0,
            "topk": 8,
            "coeff_latent_l1_loss": 0.005,
            "intervention_positions": "all",
            "intervention_type": "addition",
            "binarize_dataset": False,
            "train_on_negative": True,
            "exclude_bos": True,
        },
        "HyperSteer": {
            "batch_size": 12,
            "gradient_accumulation_steps": 1,
            "n_epochs": 10,
            "lr": 0.00008,
            "weight_decay": 0.0,
            "low_rank_dimension": 1,
            "intervention_positions": "all",
            "intervention_type": "addition",
            "binarize_dataset": False,
            "train_on_negative": False,
            "exclude_bos": True,
            "hypernet_name_or_path": spec["hypernet_name"],
            "num_hidden_layers": 4,
            "hypernet_initialize_from_pretrained": True,
        },
        "SteeringVector": {
            "batch_size": 6,
            "gradient_accumulation_steps": 1,
            "n_epochs": 3,
            "lr": 0.01,
            "weight_decay": 0.0,
            "topk": 8,
            "coeff_latent_l1_loss": 0.005,
            "intervention_positions": "all",
            "intervention_type": "addition",
            "binarize_dataset": False,
            "train_on_negative": True,
            "exclude_bos": True,
        },
        "DiffMean": {
            "batch_size": 6,
            "n_epochs": 1,
            "binarize_dataset": True,
        },
        "PCA": {
            "batch_size": 6,
            "n_epochs": 1,
            "binarize_dataset": True,
        },
        "LAT": {
            "batch_size": 6,
            "n_epochs": 1,
            "binarize_dataset": True,
        },
        "Random": {
            "batch_size": 6,
            "n_epochs": 1,
            "binarize_dataset": True,
        },
        "LinearProbe": {
            "batch_size": 12 if is_9b else 6,
            "gradient_accumulation_steps": 4 if is_9b else 8,
            "n_epochs": 24,
            "lr": 0.001 if is_9b else 0.005,
            "weight_decay": 0.0001 if is_9b else 0.001,
            "coeff_l1_loss": 0.0,
            "binarize_dataset": True,
        },
        "GemmaScopeSAEMaxAUC": {
            "batch_size": 6,
            "n_epochs": 1,
            "binarize_dataset": True,
        },
        "LoRA": {
            "batch_size": 18 if is_9b else 3,
            "gradient_accumulation_steps": 2 if is_9b else 12,
            "n_epochs": 24,
            "lr": 0.005 if is_9b else 0.0009,
            "weight_decay": 0.0,
            "low_rank_dimension": 4,
            "lora_layers": adapter_layers,
            "lora_components": ["o_proj"],
            "lora_alpha": 32,
            "binarize_dataset": False,
            "train_on_negative": False,
            "exclude_bos": True,
        },
        "LoReFT": {
            "batch_size": 9 if is_9b else 3,
            "gradient_accumulation_steps": 4 if is_9b else 12,
            "n_epochs": 24,
            "lr": 0.0004 if is_9b else 0.0009,
            "weight_decay": 0.0,
            "low_rank_dimension": 4,
            "reft_layers": adapter_layers,
            "reft_positions": "f5+l5",
            "reft_type": "Loreft",
            "binarize_dataset": False,
            "train_on_negative": False,
            "exclude_bos": True,
        },
        "SFT": {
            "batch_size": spec.get("sft_batch_size", 3),
            "gradient_accumulation_steps": spec[
                "sft_gradient_accumulation_steps"
            ],
            "n_epochs": 8,
            "lr": 0.00004,
            "weight_decay": 0.0,
            "binarize_dataset": False,
            "train_on_negative": False,
            "exclude_bos": True,
        },
        "FLAS": {
            # FLAS's diversity loss is computed within each microbatch, so
            # batch=4 with accumulation=8 is not equivalent to the paper's
            # batch=32 recipe.  Keep the Appendix A per-model settings.
            "batch_size": spec["flas_batch_size"],
            "gradient_accumulation_steps": spec[
                "flas_gradient_accumulation_steps"
            ],
            "lr": 0.00005,
            "weight_decay": 0.01,
            "binarize_dataset": False,
            "train_on_negative": False,
            "intervention_positions": "all",
            "flas_num_blocks": 1,
            "flas_n_steps": 3,
            "flas_t_min": 0.5,
            "flas_t_max": 2.0,
            "flas_div_weight": 0.1,
            "flas_total_steps": 80000,
            "flas_warmup_steps": 2000,
            "flas_max_length": 256,
            "flas_concept_max_length": 64,
            "flas_n_val_samples": 100,
            "flas_val_n_concepts": 0,
            "flas_val_every": 500,
            "flas_val_batches": 100,
            "flas_patience": 30,
            "flas_num_workers": 4,
        },
        "APSR": {
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "n_epochs": 15,
            "lr": 0.001,
            "weight_decay": 0.000001,
            "binarize_dataset": False,
            "train_on_negative": False,
            "lm_model": "DeepSeek-V3.2-Instruct",
            "prompt_temperature": 0.0,
        },
        # Official single-layer PSR uses the same optimization recipe as
        # A-PSR; only the set of imitated/intervened layers differs in the
        # model implementation.
        "SPSR": {
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "n_epochs": 15,
            "lr": 0.001,
            "weight_decay": 0.000001,
            "binarize_dataset": False,
            "train_on_negative": False,
            "lm_model": "DeepSeek-V3.2-Instruct",
            "prompt_temperature": 0.0,
        },
        "SphericalSteering": {
            "batch_size": 16,
            "binarize_dataset": True,
            "train_on_negative": True,
            "spherical_kappa": 20.0,
            "spherical_beta": 0.1,
        },
        "HiDRA": {
            "batch_size": 16,
            "binarize_dataset": True,
            "train_on_negative": True,
            "hidra_projected_dim": 8192,
            "hidra_negative_slope": 0.7,
            "hidra_projection_seed": 42,
            "hidra_normalize_direction": False,
        },
        "AUSteer": {
            "batch_size": 16,
            "binarize_dataset": True,
            "train_on_negative": True,
            "austeer_topk": 10,
        },
        "ODESteer": {
            "batch_size": 16,
            "binarize_dataset": True,
            "train_on_negative": True,
            "ode_degree": 2,
            "ode_n_components": 8000,
            "ode_gamma": 0.1,
            "ode_coef0": 1.0,
            "ode_linear_classifier": "lr",
            "ode_sketch_seed": 42,
            "ode_solver": "euler",
            "ode_steps": 10,
        },
        "StepODESteer": {
            "batch_size": 16,
            "binarize_dataset": True,
            "train_on_negative": True,
            "ode_degree": 2,
            "ode_n_components": 8000,
            "ode_gamma": 0.1,
            "ode_coef0": 1.0,
            "ode_linear_classifier": "lr",
            "ode_sketch_seed": 42,
            "ode_solver": "euler",
            "ode_steps": 10,
        },
        "PreferenceVector": {
            "batch_size": 6,
            "gradient_accumulation_steps": 1,
            "n_epochs": 18,
            "lr": 0.08 if is_9b else 0.04,
            "weight_decay": 0.0,
            "low_rank_dimension": 1,
            "intervention_positions": "all",
            "intervention_type": "addition",
            "binarize_dataset": False,
            "train_on_negative": True,
            "exclude_bos": True,
            "loss_type": "scaled_simpo",
            "beta": 1.0,
            "gemma": 0.0,
            "simpo_scaler": 1.0,
            "reference_free": True,
            "label_smoothing": 0.0,
            "dropout": 0.1 if is_9b else 0.0,
            "intervention_positions_dropout": 0.0,
            "steering_factors": [
                2.0,
                4.0,
                6.0,
                8.0,
                10.0,
                12.0,
                14.0,
                16.0,
                18.0,
                20.0,
            ],
            "preference_pairs": ["orig_add", "orig_sub"],
            "steering_prompt_type": "blend_in",
            "substraction_type": "null_it_out",
        },
    }


def generate_config(spec: dict) -> dict:
    return {
        "lm_model": "DeepSeek-V3.2-Instruct",
        "output_length": 128,
        "num_of_examples": 144,
        "concept_path": spec["concept_path"],
        "max_concepts": 500,
        "api_concurrency": 64,
        "dataset_category": "instruction",
        "master_data_dir": "steerscope/data",
        "seed": 42,
        "keep_legacy_output_format": True,
    }


def train_config(spec: dict, models: dict) -> dict:
    config = {
        "model_name": spec["model_name"],
        "layer": spec["layer"],
        "component": "res",
        "output_length": 128,
        "seed": 42,
        "use_bf16": True,
    }
    if models:
        config["models"] = copy.deepcopy(models)
    return config


def evaluate_base(spec: dict, models: list[str], run_id: str) -> dict:
    return {
        "use_bf16": True,
        "generate_reports": False,
        "evaluation_run_id": run_id,
        "models": list(models),
        "model_name": spec["model_name"],
        "output_length": 128,
        "steering_intervention_type": "addition",
        "steering_model_name": spec["model_name"],
        "steering_batch_size": spec["steering_batch_size"],
        "steering_output_length": 128,
        "steering_layers": [spec["layer"]],
        "runtime_backend": "legacy",
        "easysteer_url": "http://127.0.0.1:8017",
        "easysteer_timeout": 180,
        "master_data_dir": "steerscope/data",
        "seed": 42,
        "lm_model": "gpt-4o-mini",
        "temperature": 1.0,
    }


def inference_config(
    method: str,
    *,
    model_key: str | None = None,
    batch_size: int,
    output_length: int,
    temperature: float = 1.0,
    do_sample: bool | None = None,
    runtime_backend: str | None = None,
) -> dict:
    config = {
        # Ordinary capability/side-effect evaluation needs an explicit
        # unsteered row so retention and delta metrics share the same prompts.
        "strengths": factors_for(
            method, include_baseline=True, model_key=model_key
        ),
        "batch_size": int(batch_size),
        "output_length": int(output_length),
        "temperature": float(temperature),
    }
    if do_sample is not None:
        config["do_sample"] = bool(do_sample)
    if runtime_backend is not None:
        config["runtime_backend"] = runtime_backend
    return config


def long_batch_size(method: str, spec: dict) -> int:
    if method in EASYSTEER_METHODS:
        return 100
    if spec["model_name"] == "google/gemma-2-9b-it":
        return 8
    return 20


def method_evaluators(method: str, spec: dict) -> dict:
    model_key = model_key_for_spec(spec)
    short_batch_size = 8 if spec["model_name"].endswith("2b-it") else 4
    superglue_batch_size = 4 if spec["model_name"].endswith("2b-it") else 2
    long_batch = long_batch_size(method, spec)
    evaluators = {
        "mmlu": {
            "type": "MMLUEvaluator",
            "report": {"enabled": True},
            "dataset": {
                "type": "MMLU_test",
                "split": "all",
                "num_examples": 20,
                "seed": 42,
            },
            "inference": inference_config(
                method,
                model_key=model_key,
                batch_size=short_batch_size,
                output_length=1,
            ),
        },
        "bbq": {
            "type": "BBQEvaluator",
            "report": {"enabled": True},
            "dataset": {
                "type": "BBQ",
                "split": "all",
                "num_examples": 20,
                "seed": 42,
            },
            "inference": inference_config(
                method,
                model_key=model_key,
                batch_size=short_batch_size,
                output_length=1,
            ),
        },
        "truthfulqa": {
            "type": "TruthfulQAEvaluator",
            "report": {"enabled": True},
            "dataset": {
                "type": "TruthfulQA_binary",
                "split": "all",
                "num_examples": 20,
                "seed": 42,
            },
            "inference": inference_config(
                method,
                model_key=model_key,
                batch_size=short_batch_size,
                output_length=1,
            ),
        },
        "superglue": {
            "type": "SuperGLUEEvaluator",
            "concepts": copy.deepcopy(SLOW_EVALUATION_CONCEPTS),
            "report": {"enabled": True},
            "dataset": {
                "type": "SuperGLUE",
                "split": "all",
                "tasks": [
                    "boolq",
                    "cb",
                    "copa",
                    "multirc",
                    "record",
                    "rte",
                    "wic",
                    "wsc",
                ],
                "num_examples_per_task": 20,
                "seed": 42,
            },
            "inference": inference_config(
                method,
                model_key=model_key,
                batch_size=superglue_batch_size,
                output_length=1,
            ),
        },
        "math": {
            "type": "MATHEvaluator",
            "concepts": copy.deepcopy(SLOW_EVALUATION_CONCEPTS),
            "report": {"enabled": True},
            "dataset": {
                "type": "MATH",
                "split": "all",
                "num_examples": 20,
                "seed": 42,
                "levels": [1],
            },
            "inference": inference_config(
                method,
                model_key=model_key,
                batch_size=long_batch,
                output_length=1024,
                runtime_backend="auto",
            ),
        },
        "math_output_length": {
            "type": "OutputLengthEvaluator",
            "depends_on": ["math"],
            "concepts": copy.deepcopy(SLOW_EVALUATION_CONCEPTS),
            "report": {"enabled": True},
            "input": {"from": "math", "kind": "inference"},
        },
        "ifeval": {
            "type": "IFEvalEvaluator",
            "concepts": copy.deepcopy(SLOW_EVALUATION_CONCEPTS),
            "report": {"enabled": True},
            "dataset": {
                "type": "IFEval",
                "split": "all",
                "num_examples": 20,
                "seed": 42,
            },
            "inference": inference_config(
                method,
                model_key=model_key,
                batch_size=long_batch,
                output_length=1024,
                runtime_backend="auto",
            ),
        },
        "jailbreakbench": {
            "type": "JailBreakBenchEvaluator",
            "concepts": copy.deepcopy(SLOW_EVALUATION_CONCEPTS),
            "report": {"enabled": True},
            "params": {
                "harmful_judge_model_name": (
                    "meta-llama/Llama-3.1-8B-Instruct"
                ),
                "benign_judge_model_name": (
                    "meta-llama/Llama-3.1-8B-Instruct"
                ),
                "harmful_judge_batch_size": 64,
                "benign_judge_batch_size": 64,
                "judge_max_new_tokens": 4,
                "judge_dtype": "bfloat16",
            },
            "dataset": {
                "type": "JailBreakBench",
                "split": "all",
                "num_examples": 20,
                "seed": 42,
            },
            "inference": inference_config(
                method,
                model_key=model_key,
                batch_size=long_batch,
                output_length=150,
                runtime_backend="auto",
            ),
        },
    }
    if method == "SFT":
        for node in evaluators.values():
            if isinstance(node, dict):
                node["concepts"] = copy.deepcopy(SFT_CONCEPTS)
    return evaluators


def method_config(method: str, spec: dict, all_training: dict) -> dict:
    configured_training = (
        {method: all_training[method]} if method in all_training else {}
    )
    evaluate = evaluate_base(
        spec,
        [method],
        Path(METHOD_FILES[method]).stem,
    )
    evaluators = method_evaluators(method, spec)
    id_lm_judge = lm_judge_node(
        [method],
        include_baseline=False,
        model_key=model_key_for_spec(spec),
    )
    id_lm_judge.setdefault("params", {}).update({
        "include_baseline": True,
        "baseline_factor": 0.0,
    })
    evaluators["id_lm_judge"] = id_lm_judge
    if method == "SFT":
        for node in evaluators.values():
            if isinstance(node, dict):
                node["concepts"] = copy.deepcopy(SFT_CONCEPTS)
    evaluate["evaluators"] = evaluators
    train = train_config(spec, configured_training)
    if method == "SFT":
        train["num_concepts"] = int(SFT_CONCEPTS["count"])
        train["seed"] = int(SFT_CONCEPTS["seed"])
    return {
        "generate": generate_config(spec),
        "train": train,
        "evaluate": evaluate,
    }


def factor_map(
    methods: list[str],
    *,
    include_baseline: bool,
    model_key: str | None = None,
) -> dict:
    mapping = {
        method: factors_for(method, model_key=model_key) for method in methods
    }
    if include_baseline:
        mapping["DiffMean"] = [0.0, *mapping["DiffMean"]]
    return mapping


def lm_judge_node(
    methods: list[str],
    *,
    include_baseline: bool,
    num_examples: int = 10,
    model_key: str | None = None,
) -> dict:
    return {
        "type": "LMJudgeEvaluator",
        "report": {"enabled": True},
        "params": {
            "judge_concurrency": 8,
            "judge_pipeline_enabled": True,
        },
        "dataset": {
            "type": "AlpacaEval",
            "split": "all",
            "num_examples": int(num_examples),
            "seed": 42,
        },
        "inference": {
            "strengths_by_model": factor_map(
                methods,
                include_baseline=include_baseline,
                model_key=model_key,
            ),
            "batch_size": 10,
            "output_length": 128,
            "temperature": 1.0,
            "do_sample": True,
        },
    }


def best_factor_node(
    *,
    split: bool = False,
    genres: list[str] | None = None,
) -> dict:
    node = {
        "type": "BestFactorEvaluator",
        "depends_on": ["id_lm_judge"],
        "input": {"from": "id_lm_judge"},
        "report": {"enabled": True},
        "params": {
            "metric": "lm_judge_rating",
            "group_by": ["method"],
            "strategy": "argmax",
            "aggregation": "mean",
            "baseline_factor": 0.0,
            "fallback_baseline_method": "DiffMean",
        },
    }
    if genres is not None:
        node["concepts"] = {"genres": list(genres)}
    if split:
        node["input"]["kind"] = "samples"
        node["params"].update({
            "metric": "raw_aggregated_ratings",
            "group_by": ["method", "concept_id"],
            "split": {
                "selection": "validation",
                "evaluation": "test",
                "ratio": 0.5,
            },
            "report_metrics": {
                "lm_judge_rating": "raw_aggregated_ratings",
                "relevance_concept_ratings": (
                    "raw_relevance_concept_ratings"
                ),
                "relevance_instruction_ratings": (
                    "raw_relevance_instruction_ratings"
                ),
                "fluency_ratings": "raw_fluency_ratings",
            },
        })
    return node


def prompt_generalization_node(
    methods: list[str], *, model_key: str | None = None
) -> dict:
    return {
        "type": "PromptGeneralizationEvaluator",
        "depends_on": ["best_factor", "study_reference_best_factor"],
        "concepts": {
            "genres": ["text"],
            **copy.deepcopy(SLOW_EVALUATION_CONCEPTS),
        },
        "report": {"enabled": True},
        "params": {
            "judge_concurrency": 8,
            "judge_pipeline_enabled": True,
            "min_id_effect": 0.1,
        },
        "dataset": {
            "type": "XAlpacaEval",
            "path": "x_alpaca_eval/XAlpacaEval.parquet",
            "num_examples": 20,
            "seed": 42,
            "reference_column": "instruction_en",
            "augmenters": [
                {
                    "id": "chinese",
                    "name": "LanguageAugmenter",
                    "kwargs": {"column": "instruction_cn"},
                },
                {
                    "id": "korean",
                    "name": "LanguageAugmenter",
                    "kwargs": {"column": "instruction_ko"},
                },
                {
                    "id": "italian",
                    "name": "LanguageAugmenter",
                    "kwargs": {"column": "instruction_it"},
                },
                {
                    "id": "spanish",
                    "name": "LanguageAugmenter",
                    "kwargs": {"column": "instruction_es"},
                },
            ],
        },
        "inference": {
            "strengths_by_model": factor_map(
                methods,
                include_baseline=False,
                model_key=model_key,
            ),
            "select": {
                "from": "best_factor",
                "metric": "selected_score",
            },
            "batch_size": 10,
            "output_length": 128,
            "temperature": 1.0,
            "do_sample": True,
        },
    }


def generalization_config(spec: dict, all_training: dict) -> dict:
    methods = list(METHOD_FILES)
    evaluate = evaluate_base(spec, methods, "generalization")
    best_factor = best_factor_node()
    best_factor["depends_on"] = []
    best_factor["input"] = {
        # The scheduler replaces this explicit placeholder with the matching
        # method's completed phase-one ID metrics parquet.
        "path": "__STEERSCOPE_PHASE1_ID_METRICS__",
    }
    best_factor.setdefault("params", {}).pop(
        "fallback_baseline_method", None
    )
    prompt_generalization = prompt_generalization_node(
        methods, model_key=model_key_for_spec(spec)
    )
    prompt_generalization["depends_on"] = ["best_factor"]
    evaluate["evaluators"] = {
        "best_factor": best_factor,
        "prompt_generalization": prompt_generalization,
    }
    return {
        "generate": generate_config(spec),
        "train": train_config(spec, all_training),
        "evaluate": evaluate,
    }


def study_config(spec: dict, all_training: dict) -> dict:
    methods = list(METHOD_FILES)
    evaluate = evaluate_base(spec, methods, "study")
    study_lm_judge = lm_judge_node(
        methods,
        include_baseline=False,
        num_examples=20,
        model_key=model_key_for_spec(spec),
    )
    study_lm_judge["concepts"] = copy.deepcopy(SLOW_EVALUATION_CONCEPTS)
    study_result = best_factor_node()
    # Result-only nodes must request the same panel as their scored input.
    study_result["concepts"] = copy.deepcopy(study_lm_judge["concepts"])
    study_result["depends_on"] = ["study_lm_judge"]
    study_result["input"] = {"from": "study_lm_judge"}
    evaluate["evaluators"] = {
        "study_lm_judge": study_lm_judge,
        "study_result": study_result,
    }
    return {
        "generate": generate_config(spec),
        # Study variants use the full generated concept pool.
        "train": train_config(spec, all_training),
        "evaluate": evaluate,
        "study": {
            "factor_selection": {
                "train_examples": 144,
                "subset_seed": 42,
                "external": True,
            },
            "sample_efficiency": {
                "sizes": [6, 12, 36, 72],
                "num_examples": 10,
                "subset_seed": 42,
                "relative_report": {
                    "min_reference_improvement": 0.1,
                },
            },
            "sample_sensitivity": {
                # max_num_of_examples counts individual rows, so this is
                # twelve positive/negative pairs per concept.
                "size": 24,
                "subset_seeds": [42, 43, 44, 45, 46],
                "num_examples": 20,
                "temperature": 0.0,
                "do_sample": False,
            },
            "result": {
                "evaluator": "study_result",
                "metric": "selected_improvement",
            },
            "concept_scopes": {
                "default": copy.deepcopy(SLOW_EVALUATION_CONCEPTS),
                "by_method": {
                    "HyperSteer": {"count": 500, "seed": 42},
                    "FLAS": {"count": 500, "seed": 42},
                    "SFT": copy.deepcopy(SFT_CONCEPTS),
                },
            },
        },
    }


def sensitivity_study_config(
    spec: dict,
    all_training: dict,
    *,
    ten_concepts: bool,
) -> dict:
    """Build a deterministic, fixed-factor sample-sensitivity study."""
    config = study_config(spec, all_training)
    excluded = TEN_CONCEPT_SENSITIVITY_EXCLUDED if ten_concepts else set()
    methods = [method for method in METHOD_FILES if method not in excluded]

    train = config["train"]
    if ten_concepts:
        # Keep the compact profile directly comparable with the legacy
        # 10-concept runs and compatible with globally indexed artifacts.
        train["max_concepts"] = 10
    train["models"] = {
        method: recipe
        for method, recipe in (train.get("models") or {}).items()
        if method in methods
    }
    evaluate = config["evaluate"]
    evaluate["models"] = methods
    evaluate["temperature"] = 0.0
    evaluators = evaluate["evaluators"]
    for node in evaluators.values():
        if not isinstance(node, dict):
            continue
        node["models"] = methods
        inference = node.get("inference")
        if isinstance(inference, dict):
            strengths = inference.get("strengths_by_model") or {}
            inference["strengths_by_model"] = {
                method: values
                for method, values in strengths.items()
                if method in methods
            }
            inference["temperature"] = 0.0
            inference["do_sample"] = False
        scope = node.get("concepts")
        if isinstance(scope, dict):
            if ten_concepts:
                scope.clear()
                scope["ids"] = list(range(10))
            else:
                scope["count"] = 50
                scope["seed"] = 42

    study = config["study"]
    study.pop("sample_efficiency", None)
    study["sample_sensitivity"] = {
        # train.max_num_of_examples counts individual rows: 24 rows are
        # twelve positive/negative pairs for each concept.
        "size": 24,
        "subset_seeds": [42, 43, 44, 45, 46],
                "num_examples": 20,
        "temperature": 0.0,
        "do_sample": False,
    }
    if ten_concepts:
        study.pop("concept_scopes", None)
    else:
        study["concept_scopes"] = {
            "default": {"count": 50, "seed": 42},
        }
        study["concept_scopes"]["by_method"] = {
            "HyperSteer": {"count": 500, "seed": 42},
            "FLAS": {"count": 500, "seed": 42},
            "SFT": copy.deepcopy(SFT_CONCEPTS),
        }
    return config


def write_yaml(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
            allow_unicode=True,
            width=88,
        )


def _ten_concept_profile(
    config: dict,
    *,
    sft_concepts: int | None = None,
) -> dict:
    """Use all ten eligible concepts while preserving semantic genre scopes."""
    if sft_concepts is not None and not 1 <= int(sft_concepts) <= 10:
        raise ValueError("sft_concepts must be between 1 and 10.")
    profiled = copy.deepcopy(config)
    profiled["generate"]["max_concepts"] = 10
    train = profiled.get("train") or {}
    if train.get("num_concepts") is not None:
        train["num_concepts"] = min(10, int(train["num_concepts"]))
    # The smoke variant needs enough rows for FLAS batch_size=32 with drop_last.
    sample_efficiency = (
        (profiled.get("study") or {}).get("sample_efficiency") or {}
    )
    if sample_efficiency.get("sizes"):
        sample_efficiency["sizes"] = [
            8 if int(size) == 6 else size
            for size in sample_efficiency["sizes"]
        ]
    for node in (
        (profiled.get("evaluate") or {}).get("evaluators") or {}
    ).values():
        if not isinstance(node, dict):
            continue
        scope = node.get("concepts")
        if not isinstance(scope, dict):
            continue
        # Smoke runs drop count limits but retain evaluator genre constraints.
        genres = scope.get("genres")
        if genres is None:
            node.pop("concepts", None)
        else:
            node["concepts"] = {"genres": copy.deepcopy(genres)}

    # A small full-SFT panel keeps the smoke sweep affordable.  This remains a
    # normal method config: the scheduler derives the same deterministic panel
    # for main evaluation, generalization, and study from train.num_concepts.
    models = list((profiled.get("evaluate") or {}).get("models") or ())
    if sft_concepts is not None and models == ["SFT"]:
        count = int(sft_concepts)
        seed = int((profiled.get("train") or {}).get("seed", 42))
        profiled.setdefault("train", {})["num_concepts"] = count
        for node in (
            (profiled.get("evaluate") or {}).get("evaluators") or {}
        ).values():
            if isinstance(node, dict):
                node["concepts"] = {"count": count, "seed": seed}
    return profiled


def write_model_layer(
    relative_dir: str,
    spec: dict,
    output_dir: Path,
    *,
    ten_concept_test: bool = False,
    sft_concepts: int | None = None,
) -> None:
    all_training = training_models(spec)
    configs = {
        filename: method_config(method, spec, all_training)
        for method, filename in METHOD_FILES.items()
    }
    configs["generalization.yaml"] = generalization_config(spec, all_training)
    configs["study.yaml"] = study_config(spec, all_training)
    for filename, config in configs.items():
        if ten_concept_test:
            config = _ten_concept_profile(
                config,
                sft_concepts=sft_concepts,
            )
        write_yaml(output_dir / filename, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ten-concept-test",
        choices=sorted(MODEL_LAYERS),
        help=(
            "Generate one isolated test sweep whose every evaluator uses all "
            "ten concepts."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Required external output directory for --ten-concept-test. "
            "Temporary profiles are not written into the canonical config tree."
        ),
    )
    parser.add_argument(
        "--sft-concepts",
        type=int,
        help="SFT concept count for --ten-concept-test (1-10).",
    )
    args = parser.parse_args()
    if args.ten_concept_test:
        relative_dir = args.ten_concept_test
        if not args.output_dir:
            parser.error("--ten-concept-test requires --output-dir.")
        output_dir = Path(args.output_dir).expanduser().resolve()
        write_model_layer(
            relative_dir,
            MODEL_LAYERS[relative_dir],
            output_dir,
            ten_concept_test=True,
            sft_concepts=args.sft_concepts,
        )
        return

    if args.output_dir or args.sft_concepts is not None:
        parser.error(
            "--output-dir and --sft-concepts require --ten-concept-test."
        )
    for relative_dir, spec in MODEL_LAYERS.items():
        write_model_layer(relative_dir, spec, ROOT / relative_dir)


if __name__ == "__main__":
    main()
