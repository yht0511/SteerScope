import hashlib
import json
import logging
import os
import tempfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

import steerscope
from steerscope.evaluation.version import (
    PERPLEXITY_SCORING_VERSION,
)
from steerscope.inference.easysteer import (
    EASYSTEER_METHODS,
    EasySteerRuntime,
    EasySteerUnavailable,
    EasySteerUnsupported,
    EasySteerVectorResolver,
)
from steerscope.inference.utils import load_config, load_metadata_flatten
from steerscope.utils.constants import CHAT_MODELS
from steerscope.utils.model_utils import get_prefix_length


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SteeringModelConfig:
    """Immutable inference configuration bound to one model wrapper."""

    factor: float
    layer: int | None = None
    layers: tuple[int, ...] = ()
    temperature: float = 1.0
    do_sample: bool = True
    max_new_tokens: int = 128
    batch_size: int = 8
    seed: int = 42
    intervention_type: str = "addition"
    intervene_on_prompt: bool = True
    disable_neuronpedia_max_act: bool = False
    compute_perplexity: bool = False


class SteeringModel:
    """A fully configured target model that an evaluator can invoke directly."""

    def __init__(
        self,
        target,
        args,
        training_args,
        root_dump_dir,
        cache_dir,
        config: SteeringModelConfig,
        device=None,
        runtime=None,
    ):
        self.target = target
        self.method = target.method
        self.concept = target.concept
        self.config = config
        self.factor = config.factor
        self._args = args
        self._training_args = training_args
        self._root_dump_dir = Path(root_dump_dir)
        self._cache_dir = Path(cache_dir)
        self._device = device
        self._runtime = runtime

    def generate(self, examples: pd.DataFrame) -> pd.DataFrame:
        """Infer over evaluator-owned examples using the configuration bound at init."""
        if examples.empty:
            raise ValueError(f"Target '{self.target.target_id}' received no examples.")
        if "factor" not in examples and "model_factor" not in examples:
            raise ValueError(
                "Evaluator datasets must contain 'factor' or 'model_factor'."
            )
        factor_column = "model_factor" if "model_factor" in examples else "factor"
        factors = {float(value) for value in examples[factor_column].unique()}
        if factors != {self.factor}:
            raise ValueError(
                f"Target '{self.target.target_id}' wrapper is bound to factor "
                f"{self.factor}, but received {factor_column} values {sorted(factors)}."
            )
        cache_file = self._cache_file(examples)
        if cache_file.exists() and not bool(getattr(self._args, "overwrite_cache", False)):
            try:
                cached = pd.read_parquet(cache_file)
                if self._valid_request_cache(cached, examples):
                    return cached
                logger.warning("Ignoring mismatched request cache %s", cache_file)
            except Exception as error:
                logger.warning("Ignoring invalid model cache %s: %s", cache_file, error)
        node_args = SimpleNamespace(**vars(self._args))
        node_args.models = [self.method]
        node_args.steering_factors = [self.config.factor]
        node_args.steering_layer = self.config.layer
        node_args.steering_layers = list(self.config.layers) or None
        node_args.temperature = self.config.temperature
        node_args.do_sample = self.config.do_sample
        node_args.steering_output_length = self.config.max_new_tokens
        node_args.steering_batch_size = self.config.batch_size
        node_args.seed = self.config.seed
        node_args.steering_intervention_type = self.config.intervention_type
        node_args.intervene_on_prompt = self.config.intervene_on_prompt
        node_args.disable_neuronpedia_max_act = self.config.disable_neuronpedia_max_act
        node_args.compute_perplexity = self.config.compute_perplexity

        if self._runtime is None:
            runner = SteeringTargetRunner(
                args=node_args,
                training_args=self._training_args,
                root_dump_dir=self._root_dump_dir,
                cache_dir=self._cache_dir,
                target=self.target,
                device=self._device,
            )
            generated = runner.generate(examples)
        else:
            generated = self._runtime.generate_target(
                self.target,
                examples,
                batch_size=self.config.batch_size,
            )
        SteeringInferenceWrapper._save_cache(cache_file, generated)
        return generated

    def generate_baseline(self, examples: pd.DataFrame) -> pd.DataFrame:
        """Generate with the shared, completely unsteered base model."""
        if examples.empty:
            raise ValueError("Baseline inference received no examples.")
        cache_file = self._baseline_cache_file(examples)
        if cache_file.exists() and not bool(
            getattr(self._args, "overwrite_cache", False)
        ):
            try:
                cached = pd.read_parquet(cache_file)
                if self._valid_baseline_cache(cached, examples):
                    return cached
                logger.warning(
                    "Ignoring mismatched baseline request cache %s", cache_file
                )
            except Exception as error:
                logger.warning(
                    "Ignoring invalid baseline request cache %s: %s",
                    cache_file,
                    error,
                )
        node_args = SimpleNamespace(**vars(self._args))
        node_args.temperature = self.config.temperature
        node_args.do_sample = self.config.do_sample
        node_args.steering_output_length = self.config.max_new_tokens
        node_args.steering_batch_size = int(
            getattr(
                self._args,
                "steering_batch_size",
                self.config.batch_size,
            )
        )
        node_args.seed = self.config.seed
        if self._runtime is None:
            runner = SteeringTargetRunner(
                args=node_args,
                training_args=self._training_args,
                root_dump_dir=self._root_dump_dir,
                cache_dir=self._cache_dir,
                target=self.target,
                device=self._device,
            )
            generated = runner.generate_baseline(examples)
        else:
            generated = self._runtime.generate_baseline(examples)
        SteeringInferenceWrapper._save_cache(cache_file, generated)
        return generated

    def _valid_request_cache(
        self, cached: pd.DataFrame, examples: pd.DataFrame
    ) -> bool:
        output_columns = {
            f"{self.method}_steered_generation",
            f"{self.method}_choice_logits",
            f"{self.method}_choice_loglikelihoods",
        }
        if len(cached) != len(examples) or not output_columns.intersection(cached):
            return False
        if self.config.compute_perplexity and f"{self.method}_perplexity" not in cached:
            return False
        if any(column not in cached for column in examples.columns):
            return False
        if "target_id" not in cached or set(cached["target_id"]) != {
            self.target.target_id
        }:
            return False
        if "method" not in cached or set(cached["method"]) != {self.method}:
            return False
        expected = examples.reset_index(drop=True)
        actual = cached[list(expected.columns)].reset_index(drop=True)
        return self._frame_payload(actual) == self._frame_payload(expected)

    @staticmethod
    def _frame_payload(frame: pd.DataFrame) -> str:
        return frame.to_json(
            orient="split", date_format="iso", default_handler=str
        )

    def _cache_file(self, examples: pd.DataFrame) -> Path:
        cache_dir = self._cache_dir / "requests"
        cache_dir.mkdir(parents=True, exist_ok=True)
        training_models = getattr(self._training_args, "models", {})
        training_model_args = None
        if self.method in training_models:
            training_model_args = vars(training_models[self.method])
        payload = {
            "target": self._target_cache_context(),
            "config": asdict(self.config),
            "training_model_args": training_model_args,
            "use_bf16": bool(getattr(self._args, "use_bf16", False)),
            "runtime_backend": getattr(
                self._args, "runtime_backend", "legacy"
            ),
            "easysteer_url": getattr(
                self._args, "easysteer_url", None
            ),
            "demo_interactive": bool(
                getattr(self._args, "demo_interactive", False)
            ),
            "demo_output": getattr(self._args, "demo_output", None),
            "rows": json.loads(examples.to_json(orient="split", date_format="iso")),
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        key = hashlib.sha256(encoded).hexdigest()[:24]
        return cache_dir / f"{self.method}_{key}.parquet"

    def _baseline_cache_file(self, examples: pd.DataFrame) -> Path:
        cache_dir = self._cache_dir / "requests"
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "base_model": self.target.base_model,
            "temperature": self.config.temperature,
            "do_sample": self.config.do_sample,
            "max_new_tokens": self.config.max_new_tokens,
            "batch_size": int(
                getattr(
                    self._args,
                    "steering_batch_size",
                    self.config.batch_size,
                )
            ),
            "seed": self.config.seed,
            "use_bf16": bool(getattr(self._args, "use_bf16", False)),
            "runtime_backend": getattr(
                self._args, "runtime_backend", "legacy"
            ),
            "rows": json.loads(
                examples.to_json(orient="split", date_format="iso")
            ),
        }
        encoded = json.dumps(
            payload, sort_keys=True, default=str
        ).encode("utf-8")
        key = hashlib.sha256(encoded).hexdigest()[:24]
        return cache_dir / f"Baseline_{key}.parquet"

    @classmethod
    def _valid_baseline_cache(
        cls, cached: pd.DataFrame, examples: pd.DataFrame
    ) -> bool:
        if len(cached) != len(examples) or "baseline_generation" not in cached:
            return False
        if any(column not in cached for column in examples.columns):
            return False
        expected = examples.reset_index(drop=True)
        actual = cached[list(expected.columns)].reset_index(drop=True)
        return cls._frame_payload(actual) == cls._frame_payload(expected)

    def _target_cache_context(self) -> Mapping[str, Any]:
        train_dir = self._root_dump_dir / "train"
        engine_context = getattr(self._args, "evaluation_cache_context", None) or {}
        if self.target.artifact.kind == "checkpoint" and self.target.artifact.path == train_dir:
            artifact = (engine_context.get("artifacts") or {}).get(self.method)
        else:
            artifact = self.target.artifact.signature()
        return {
            "target_id": self.target.target_id,
            "method": self.method,
            "concept_id": self.concept.concept_id,
            "concept": self.concept.text,
            "base_model": self.target.base_model,
            "artifact": artifact,
        }


class SteeringTargetRunner:
    """Load one evaluation target and generate an evaluator-provided dataset."""

    def __init__(
        self,
        args,
        training_args,
        root_dump_dir,
        cache_dir,
        target,
        device=None,
    ):
        self.args = args
        self.training_args = training_args
        self.root_dump_dir = Path(root_dump_dir)
        self.cache_dir = Path(cache_dir)
        self.target = target
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def generate(self, examples: pd.DataFrame) -> pd.DataFrame:
        runtime = _SteeringTargetRuntime(
            args=self.args,
            training_args=self.training_args,
            root_dump_dir=self.root_dump_dir,
            output_dir=self.cache_dir,
            device=self.device,
            cache_dir=self.cache_dir,
            targets=[self.target],
            long_format=True,
        )
        try:
            return runtime.generate_target(self.target, examples)
        finally:
            runtime.close()

    def generate_baseline(self, examples: pd.DataFrame) -> pd.DataFrame:
        runtime = _SteeringTargetRuntime(
            args=self.args,
            training_args=self.training_args,
            root_dump_dir=self.root_dump_dir,
            output_dir=self.cache_dir,
            device=self.device,
            cache_dir=self.cache_dir,
            targets=[self.target],
            long_format=True,
        )
        try:
            return runtime.generate_baseline(examples)
        finally:
            runtime.close()


class SteeringInferenceWrapper:
    def __init__(
        self,
        model_name,
        benchmark_model,
        training_args,
        inference_args,
        prefix_length,
        cache_dir,
        cache_context=None,
    ):
        self.model_name = model_name
        self.benchmark_model = benchmark_model
        self.training_args = training_args
        self.inference_args = inference_args
        self.prefix_length = prefix_length
        self.cache_dir = Path(cache_dir)
        self.cache_context = dict(cache_context or {})
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def predict(self, examples, concept_id, metadata):
        model_args = self._model_args()
        cache_key = self._cache_key(examples, concept_id)
        cache_file = self.cache_dir / f"{self.model_name}_{cache_key}.parquet"
        if cache_file.exists() and not bool(self.inference_args.overwrite_cache):
            try:
                cached = pd.read_parquet(cache_file)
                if self._valid_cache(cached, len(examples)):
                    results = {
                        column: cached[column].tolist()
                        for column in cached.columns
                    }
                    if self._needs_perplexity(results):
                        results["perplexity"] = self._perplexities(
                            results["steered_generation"]
                        )
                        self._save_cache(cache_file, results)
                    return results
                logger.warning("Ignoring mismatched inference cache %s", cache_file)
            except Exception as error:
                logger.warning("Ignoring invalid inference cache %s: %s", cache_file, error)

        request_seed = int(
            hashlib.sha256(
                f"{int(self.inference_args.seed or 0)}:{cache_key}".encode("utf-8")
            ).hexdigest()[:8],
            16,
        )
        set_seed(request_seed)
        inference_examples = self.benchmark_model.prepare_inference_examples(
            examples,
            concept_id=concept_id,
        )
        inference_modes = set(
            inference_examples.get("inference_mode", pd.Series(dtype=str))
        )
        supported_modes = {"choice_logits", "choice_loglikelihood"}
        if len(inference_modes) > 1 or inference_modes.difference(supported_modes):
            raise ValueError(
                f"Unsupported or mixed inference modes: {sorted(inference_modes)}"
            )
        if inference_modes == {"choice_logits"}:
            predict = self.benchmark_model.predict_choice_logits
        elif inference_modes == {"choice_loglikelihood"}:
            predict = self.benchmark_model.predict_choice_loglikelihoods
        else:
            predict = self.benchmark_model.predict_steer
        results = predict(
            inference_examples,
            concept_id=concept_id,
            sae_link=None,
            sae_id=None,
            batch_size=int(self.inference_args.steering_batch_size),
            eval_output_length=int(self.inference_args.steering_output_length),
            temperature=float(self.inference_args.temperature),
            do_sample=bool(getattr(self.inference_args, "do_sample", True)),
            prefix_length=self.prefix_length,
            positions=model_args.intervention_positions if self.benchmark_model.uses_intervention_positions and model_args is not None else None,
            use_synergy=False,
            disable_neuronpedia_max_act=self.inference_args.disable_neuronpedia_max_act,
            intervene_on_prompt=self.inference_args.intervene_on_prompt if self.inference_args.intervene_on_prompt is not None else True,
            return_vector=False,
            show_progress=False,
            demo_interactive=bool(
                getattr(self.inference_args, "demo_interactive", False)
            ),
            demo_output=getattr(
                self.inference_args, "demo_output", "Demo response"
            ),
        )
        if getattr(self.benchmark_model, "records_model_input", False) is True:
            results["model_input"] = inference_examples["input"].tolist()
        # Method implementations historically used different sequences and
        # models for PPL. Always replace those values with the common scorer.
        if bool(getattr(self.inference_args, "compute_perplexity", False)):
            results["perplexity"] = self._perplexities(results["steered_generation"])
        else:
            results.pop("perplexity", None)
        self._save_cache(cache_file, results)
        return results

    @staticmethod
    def _valid_cache(cached: pd.DataFrame, expected_rows: int) -> bool:
        output_columns = {
            "steered_generation", "choice_logits", "choice_loglikelihoods",
        }
        if len(cached) != expected_rows or not output_columns.intersection(cached):
            return False
        return all(len(cached[column]) == expected_rows for column in cached.columns)

    def _needs_perplexity(self, results):
        return (
            bool(getattr(self.inference_args, "compute_perplexity", False))
            and "perplexity" not in results
        )

    @staticmethod
    def _save_cache(cache_file, results):
        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=cache_file.parent,
            suffix=".parquet.tmp",
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            pd.DataFrame(results).to_parquet(temporary_path, index=False)
            os.replace(temporary_path, cache_file)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    @torch.no_grad()
    def _perplexities(self, generations):
        """Score only generated continuations with the unsteered base model."""
        scores = []
        batch_size = int(self.inference_args.steering_batch_size)
        model = self.benchmark_model.model
        tokenizer = self.benchmark_model.tokenizer
        adapter_model = getattr(self.benchmark_model, "ax_model", None)
        adapter_context = (
            adapter_model.disable_adapter()
            if hasattr(adapter_model, "disable_adapter")
            else nullcontext()
        )
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "right"
        try:
            with adapter_context:
                model.eval()
                for start in range(0, len(generations), batch_size):
                    batch = generations[start:start + batch_size]
                    encoded = tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(self.benchmark_model.device)
                    labels = encoded.input_ids.masked_fill(
                        encoded.attention_mask == 0, -100
                    )
                    logits = model(**encoded).logits[:, :-1].float()
                    targets = labels[:, 1:]
                    token_losses = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        reduction="none",
                        ignore_index=-100,
                    ).view(targets.shape)
                    valid_tokens = targets.ne(-100)
                    token_counts = valid_tokens.sum(dim=1)
                    sequence_losses = token_losses.sum(dim=1) / token_counts.clamp_min(1)
                    perplexities = torch.exp(sequence_losses)
                    perplexities = perplexities.masked_fill(token_counts == 0, torch.nan)
                    scores.extend(perplexities.cpu().tolist())
        finally:
            tokenizer.padding_side = original_padding_side
        return scores

    def _model_args(self):
        if not self.benchmark_model.requires_training_args:
            return None
        if self.model_name not in self.training_args.models.keys():
            return None
        return self.training_args.models[self.model_name]

    def _cache_key(self, examples, concept_id):
        model_args = self._model_args()
        payload = {
            "perplexity_scoring_version": PERPLEXITY_SCORING_VERSION,
            "model_name": self.model_name,
            "concept_id": concept_id,
            "temperature": self.inference_args.temperature,
            "do_sample": bool(getattr(self.inference_args, "do_sample", True)),
            "steering_output_length": self.inference_args.steering_output_length,
            "steering_batch_size": self.inference_args.steering_batch_size,
            "steering_intervention_type": self.inference_args.steering_intervention_type,
            "intervention_positions": getattr(model_args, "intervention_positions", None),
            "intervene_on_prompt": self.inference_args.intervene_on_prompt,
            "disable_neuronpedia_max_act": self.inference_args.disable_neuronpedia_max_act,
            "base_model": self.inference_args.steering_model_name or self.inference_args.model_name,
            "steering_layers": self.inference_args.steering_layers,
            "steering_layer": self.inference_args.steering_layer,
            "seed": self.inference_args.seed,
            "model_args": vars(model_args) if model_args is not None else None,
            "demo_interactive": bool(
                getattr(self.inference_args, "demo_interactive", False)
            ),
            "demo_output": getattr(self.inference_args, "demo_output", None),
            "target_context": self.cache_context,
            "rows": json.loads(examples.to_json(orient="split", date_format="iso")),
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:24]


class _SteeringTargetRuntime:
    def __init__(
        self,
        args,
        training_args,
        root_dump_dir,
        output_dir,
        device=None,
        cache_dir=None,
        targets=None,
        long_format=False,
    ):
        self.args = args
        self.training_args = training_args
        self.root_dump_dir = Path(root_dump_dir)
        self.output_dir = Path(output_dir)
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.output_dir / "inference_cache"
        self.model_cache_dir = self.cache_dir / "models"
        self.targets = {
            (target.method, target.concept.concept_id): target
            for target in (targets or [])
        }
        self.long_format = bool(long_format)
        self._tokenizer = None
        self._base_model = None
        self._prefix_length = None
        self._benchmark_model = None
        self._benchmark_key = None
        self._benchmark_method = None
        self._mean_activations_ready = False
        self._easysteer_runtime = None
        self._runtime_backend = str(
            getattr(self.args, "runtime_backend", "legacy") or "legacy"
        ).lower()
        if self._runtime_backend not in {"legacy", "auto", "easysteer"}:
            raise ValueError(
                "runtime_backend must be one of: legacy, auto, easysteer; "
                f"got {self._runtime_backend!r}."
            )

    def generate_target(
        self,
        target,
        examples: pd.DataFrame,
        batch_size: int | None = None,
    ) -> pd.DataFrame:
        """Generate an evaluator-owned dataframe for exactly one target."""
        request_batch_size = int(
            self.args.steering_batch_size
            if batch_size is None
            else batch_size
        )
        if request_batch_size < 1:
            raise ValueError("steering_batch_size must be positive.")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        set_seed(int(self.args.seed or 0))
        examples = examples.copy().reset_index(drop=True)
        examples["concept_id"] = int(target.concept.concept_id)
        examples["sae_link"] = target.concept.ref or ""
        if "sae_id" not in examples:
            try:
                examples["sae_id"] = int((target.concept.ref or "").split("/")[-1])
            except (TypeError, ValueError):
                examples["sae_id"] = 0

        train_dir = self.root_dump_dir / "train"
        metadata = self._metadata()
        config = load_config(train_dir)
        if self.args.steering_layer is not None:
            layer = int(self.args.steering_layer)
        elif self.args.steering_layers:
            layer = int(self.args.steering_layers[0])
        else:
            layer = int(config["layer"]) if config else 0
        steering_layers = self.args.steering_layers or [layer]

        if self._should_use_easysteer(target.method):
            try:
                results = self._easy_runtime(target).predict(
                    target,
                    examples,
                    target_layers=steering_layers,
                    batch_size=request_batch_size,
                    max_new_tokens=int(
                        self.args.steering_output_length
                    ),
                    temperature=float(self.args.temperature),
                    seed=int(self.args.seed or 0),
                    intervention_type=(
                        self.args.steering_intervention_type or "addition"
                    ),
                    intervene_on_prompt=(
                        self.args.intervene_on_prompt
                        if self.args.intervene_on_prompt is not None
                        else True
                    ),
                    disable_neuronpedia_max_act=bool(
                        self.args.disable_neuronpedia_max_act
                    ),
                    compute_perplexity=bool(
                        getattr(self.args, "compute_perplexity", False)
                    ),
                )
                return self._result_frame(target, examples, results)
            except (EasySteerUnsupported, EasySteerUnavailable) as error:
                if self._runtime_backend == "easysteer":
                    raise
                logger.warning(
                    "EasySteer cannot serve %s; falling back to the legacy "
                    "runtime: %s",
                    target.target_id,
                    error,
                )

        model_class = self._model_class(target.method)
        if getattr(model_class, "lightweight_runtime", False):
            tokenizer, base_model, prefix_length = None, None, 0
        else:
            tokenizer, base_model, prefix_length = self._ensure_runtime_resources()
        benchmark_key = self._benchmark_cache_key(
            target, layer, steering_layers
        )
        if benchmark_key != getattr(self, "_benchmark_key", None):
            self._release_benchmark_model()
            benchmark_model = self._load_concept_model(
                target.method,
                base_model,
                tokenizer,
                metadata,
                train_dir,
                layer,
                steering_layers,
                target.concept.concept_id,
            )
            self._benchmark_model = benchmark_model
            self._benchmark_key = benchmark_key
            self._benchmark_method = target.method
            self._mean_activations_ready = False
        benchmark_model = self._benchmark_model

        if (
            benchmark_model.requires_mean_activations
            and not self._mean_activations_ready
        ):
            inference_dir = self.root_dump_dir / "inference"
            inference_dir.mkdir(parents=True, exist_ok=True)
            benchmark_model.pre_compute_mean_activations(
                str(inference_dir),
                master_data_dir=self.args.master_data_dir,
                disable_neuronpedia_max_act=self.args.disable_neuronpedia_max_act,
                metadata=metadata,
            )
            self._mean_activations_ready = True

        # Standalone vectors contain one local row; trained checkpoints retain
        # their global concept-indexed weight table.
        if target.artifact.kind == "steering_vector":
            benchmark_model.concept_id_map = {target.concept.concept_id: 0}

        inference_args = self.args
        if request_batch_size != int(self.args.steering_batch_size):
            inference_args = SimpleNamespace(**vars(self.args))
            inference_args.steering_batch_size = request_batch_size
        wrapper = SteeringInferenceWrapper(
            target.method,
            benchmark_model,
            self.training_args,
            inference_args,
            prefix_length,
            self.model_cache_dir,
            cache_context=self._target_cache_context(
                target.method, target.concept.concept_id
            ),
        )
        results = wrapper.predict(
            examples,
            concept_id=target.concept.concept_id,
            metadata=metadata,
        )
        return self._result_frame(target, examples, results)

    @torch.no_grad()
    def generate_baseline(
        self,
        examples: pd.DataFrame,
        batch_size: int | None = None,
    ) -> pd.DataFrame:
        """Generate once with the local base model and no steering machinery."""
        if self._runtime_backend != "legacy":
            raise RuntimeError(
                "Shared baseline inference currently requires "
                "runtime_backend='legacy' so baseline and steered generations "
                "use the same backend."
            )
        if examples.empty:
            raise ValueError("Baseline inference received no examples.")
        if "input" not in examples:
            raise KeyError("Baseline inference requires an 'input' column.")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        set_seed(int(self.args.seed or 0))
        self._release_benchmark_model()
        tokenizer, base_model, _ = self._ensure_runtime_resources()
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        generations = []
        batch_size = int(
            self.args.steering_batch_size
            if batch_size is None
            else batch_size
        )
        max_new_tokens = int(self.args.steering_output_length)
        temperature = float(self.args.temperature)
        do_sample = bool(getattr(self.args, "do_sample", temperature > 0))
        if batch_size < 1:
            raise ValueError("steering_batch_size must be positive.")
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            if temperature <= 0:
                raise ValueError(
                    "Sampling generation requires a positive temperature."
                )
            generation_kwargs["temperature"] = temperature
        try:
            for start in range(0, len(examples), batch_size):
                prompts = examples.iloc[
                    start:start + batch_size
                ]["input"].astype(str).tolist()
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)
                output_ids = base_model.generate(
                    **inputs,
                    **generation_kwargs,
                )
                prompt_width = inputs.input_ids.shape[1]
                generations.extend(
                    tokenizer.decode(
                        output[prompt_width:],
                        skip_special_tokens=True,
                    )
                    for output in output_ids
                )
        finally:
            tokenizer.padding_side = original_padding_side

        result = examples.copy().reset_index(drop=True)
        result["baseline_generation"] = generations
        return result

    @staticmethod
    def _result_frame(target, examples, results):
        for key, value in results.items():
            examples[f"{target.method}_{key}"] = value
        examples.insert(0, "target_id", target.target_id)
        examples.insert(0, "method", target.method)
        return examples

    def _should_use_easysteer(self, method):
        backend = getattr(
            self,
            "_runtime_backend",
            str(
                getattr(getattr(self, "args", None), "runtime_backend", "legacy")
                or "legacy"
            ).lower(),
        )
        return (
            backend in {"auto", "easysteer"}
            and method in EASYSTEER_METHODS
        )

    def _easy_runtime(self, target):
        if getattr(self, "_easysteer_runtime", None) is None:
            resolver = EasySteerVectorResolver(
                cache_dir=self.model_cache_dir / "easysteer_vectors",
                default_checkpoint_dir=self.root_dump_dir / "train",
                master_data_dir=getattr(
                    self.args, "master_data_dir", "steerscope/data"
                ),
            )
            self._easysteer_runtime = EasySteerRuntime(
                base_url=(
                    getattr(self.args, "easysteer_url", None)
                    or "http://127.0.0.1:8017"
                ),
                model_name=target.base_model,
                resolver=resolver,
                timeout=float(
                    getattr(self.args, "easysteer_timeout", 120.0)
                ),
            )
        elif self._easysteer_runtime.model_name != target.base_model:
            raise EasySteerUnsupported(
                "One EasySteer runtime cannot mix base models: "
                f"{self._easysteer_runtime.model_name!r} and "
                f"{target.base_model!r}."
            )
        return self._easysteer_runtime

    def _ensure_runtime_resources(self):
        if getattr(self, "_tokenizer", None) is None:
            self._tokenizer = self._load_tokenizer()
        if getattr(self, "_base_model", None) is None:
            self._base_model = self._load_base_model()
            self._ensure_padding(self._tokenizer, self._base_model)
        if getattr(self, "_prefix_length", None) is None:
            base_model_name = self.args.steering_model_name or self.args.model_name
            self._prefix_length = (
                get_prefix_length(self._tokenizer)
                if base_model_name in CHAT_MODELS
                else 1
            )
        return self._tokenizer, self._base_model, self._prefix_length

    def _benchmark_cache_key(self, target, layer, steering_layers):
        model_class = self._model_class(target.method)
        concept_id = (
            target.concept.concept_id
            if model_class.inference_instance_scope == "per_concept"
            else None
        )
        return (
            target.method,
            concept_id,
            target.artifact.kind,
            str(target.artifact.path),
            target.artifact.fingerprint,
            int(layer),
            tuple(int(value) for value in steering_layers),
            self.args.steering_intervention_type,
        )

    def _release_benchmark_model(self):
        benchmark_model = getattr(self, "_benchmark_model", None)
        if benchmark_model is None:
            return
        if getattr(self, "_benchmark_method", None) == "LoRA":
            adapter_model = getattr(benchmark_model, "ax_model", None)
            if hasattr(adapter_model, "unload"):
                self._base_model = adapter_model.unload()
                self._base_model.eval()
        self._benchmark_model = None
        self._benchmark_key = None
        self._benchmark_method = None
        self._mean_activations_ready = False
        torch.cuda.empty_cache()

    def close(self):
        self._release_benchmark_model()
        self._easysteer_runtime = None
        self._base_model = None
        self._tokenizer = None
        torch.cuda.empty_cache()

    def _metadata_dir(self):
        if self.training_args.overwrite_metadata_dir is not None and os.path.exists(self.training_args.overwrite_metadata_dir):
            return Path(self.training_args.overwrite_metadata_dir)
        return self.root_dump_dir / "generate"

    def _metadata(self):
        if not self.targets:
            return load_metadata_flatten(self._metadata_dir())
        by_concept = {}
        for target in self.targets.values():
            concept = target.concept
            by_concept.setdefault(concept.concept_id, {
                "concept": concept.text,
                "ref": concept.ref,
                "concept_genres_map": concept.metadata.get(
                    "concept_genres_map", {concept.text: ["text"]}
                ),
                "concept_id": concept.concept_id,
            })
        return [by_concept[concept_id] for concept_id in sorted(by_concept)]

    def _target_id(self, model_name, concept_id):
        target = self.targets.get((model_name, concept_id))
        return target.target_id if target is not None else f"{model_name}/concept-{concept_id}"

    def _target_cache_context(self, model_name, concept_id):
        target = self.targets.get((model_name, concept_id))
        engine_context = getattr(self.args, "evaluation_cache_context", None) or {}
        if target is None:
            target_id = f"{model_name}/concept-{concept_id}"
            artifact = (engine_context.get("artifacts") or {}).get(model_name)
        else:
            target_id = target.target_id
            implicit_train_dir = self.root_dump_dir / "train"
            configured_artifacts = (
                engine_context.get("artifacts") or {}
            ).get(model_name) or {}
            if (
                target.artifact.kind == "checkpoint"
                and target.artifact.path == implicit_train_dir
            ):
                artifact = configured_artifacts
            elif (
                target.artifact.kind == "checkpoint"
                and configured_artifacts.get("checkpoint") is not None
            ):
                artifact = configured_artifacts["checkpoint"]
            else:
                artifact = target.artifact.signature()
        return {
            "target_id": target_id,
            "base_model": self.args.steering_model_name or self.args.model_name,
            "concept_id": int(concept_id),
            "artifact": artifact,
        }

    def _load_tokenizer(self):
        model_name = self.args.steering_model_name or self.args.model_name
        max_length = 128000 if "google/gemma-3" in model_name else 1024
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, model_max_length=max_length)
        tokenizer.padding_side = "right"
        return tokenizer

    def _load_base_model(self):
        model_name = self.args.steering_model_name or self.args.model_name
        dtype = torch.bfloat16 if self.args.use_bf16 else None
        if "gemma-3" in model_name:
            from transformers import Gemma3ForCausalLM
            return Gemma3ForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=self.device).eval()
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=self.device).eval()

    def _ensure_padding(self, tokenizer, model_instance):
        if tokenizer.unk_token is None and tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model_instance.resize_token_embeddings(len(tokenizer))

    def _load_concept_model(self, model_name, model_instance, tokenizer, metadata, train_dir, layer, steering_layers, concept_id):
        model_class = self._model_class(model_name)
        model_args = self._training_model_args(model_name) if model_class.requires_training_args else None
        target = self.targets.get((model_name, concept_id))
        is_standalone_vector = (
            target is not None and target.artifact.kind == "steering_vector"
        )
        if model_class.requires_training_args and model_args is None and not is_standalone_vector:
            raise ValueError(f"Training config for model '{model_name}' is required for steering evaluation.")
        benchmark_model = model_class(
            model_instance,
            tokenizer,
            layer=layer,
            training_args=model_args,
            device=self.device,
            steering_layers=steering_layers,
            lm_model_name=(
                target.base_model
                if target is not None
                else self.args.steering_model_name or self.args.model_name
            ),
        )
        if is_standalone_vector:
            self._load_steering_vector(
                benchmark_model,
                target.artifact.path,
                model_name,
                concept_id,
                self.args.steering_intervention_type,
            )
            benchmark_model.to(self.device)
            self._prepare_ax_dtype(benchmark_model, model_name)
            return benchmark_model
        if model_class.load_trained_weights:
            load_kwargs = dict(
                dump_dir=(
                    target.artifact.path
                    if target is not None and target.artifact.kind == "checkpoint"
                    else train_dir
                ),
                sae_path=metadata[0]["ref"],
                mode="steering",
                priority_mode="compute_priority",
                intervention_type=self.args.steering_intervention_type,
                concept_id=concept_id,
            )
            configured_rank = (
                getattr(model_args, "low_rank_dimension", None)
                if model_args is not None
                else None
            )
            if configured_rank is not None:
                load_kwargs["low_rank_dimension"] = configured_rank
            for key in (
                "hypernet_initialize_from_pretrained",
                "hypernet_name_or_path",
                "num_hidden_layers",
            ):
                value = getattr(model_args, key, None) if model_args is not None else None
                if value is not None:
                    load_kwargs[key] = value
            benchmark_model.load(**load_kwargs)
        elif hasattr(benchmark_model, "make_model"):
            benchmark_model.make_model(
                mode="steering",
                intervention_type=self.args.steering_intervention_type or "addition",
                concept_id=concept_id,
            )
        benchmark_model.to(self.device)
        self._prepare_ax_dtype(benchmark_model, model_name)
        return benchmark_model

    def _load_steering_vector(
        self,
        benchmark_model,
        vector_path,
        model_name,
        concept_id,
        intervention_type,
    ):
        if model_name != "SteeringVector":
            raise ValueError(
                "Standalone steering_vector artifacts currently require "
                "method: SteeringVector."
            )
        vector_path = Path(vector_path)
        if not vector_path.exists():
            raise FileNotFoundError(f"Steering vector not found: {vector_path}")
        raw = torch.load(vector_path, map_location="cpu", weights_only=True)
        if isinstance(raw, dict):
            for key in ("steering_vector", "vector", "weight"):
                if key in raw:
                    raw = raw[key]
                    break
        vector = torch.as_tensor(raw).detach()
        if vector.ndim == 2:
            if vector.shape[0] == 1:
                vector = vector[0]
            elif concept_id < vector.shape[0]:
                vector = vector[concept_id]
            else:
                raise ValueError(
                    f"Steering vector at {vector_path} has {vector.shape[0]} rows, "
                    f"so it cannot select concept ID {concept_id}."
                )
        if vector.ndim != 1:
            raise ValueError(
                f"Steering vector at {vector_path} must have shape [hidden_size] "
                "or [num_concepts, hidden_size]."
            )
        hidden_size = int(self._base_model_hidden_size(benchmark_model))
        if vector.numel() != hidden_size:
            raise ValueError(
                f"Steering vector at {vector_path} has {vector.numel()} values; "
                f"the base model expects {hidden_size}."
            )
        benchmark_model.make_model(
            mode="steering",
            intervention_type=intervention_type or "addition",
            low_rank_dimension=1,
        )
        weight = benchmark_model.ax.proj.weight.data
        weight[0].copy_(vector.to(device=weight.device, dtype=weight.dtype))
        benchmark_model.ax.proj.bias.data.zero_()
        benchmark_model.concept_id_map = {concept_id: 0}

    @staticmethod
    def _base_model_hidden_size(benchmark_model):
        return benchmark_model.model.config.hidden_size

    def _model_class(self, model_name):
        return getattr(steerscope, model_name)

    def _training_model_args(self, model_name):
        if model_name not in self.training_args.models.keys():
            return None
        return self.training_args.models[model_name]

    def _prepare_ax_dtype(self, benchmark_model, model_name):
        if not hasattr(benchmark_model, "ax") or not self.args.use_bf16:
            return
        if model_name in {"PreferenceLoReFT", "ConceptLoReFT"}:
            return
        if isinstance(benchmark_model.ax, list):
            for ax in benchmark_model.ax:
                ax.eval()
                ax.to(torch.bfloat16)
        else:
            benchmark_model.ax.eval()
            benchmark_model.ax.to(torch.bfloat16)
