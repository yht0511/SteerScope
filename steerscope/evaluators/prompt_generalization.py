"""Evaluate prompt-shift generalization on aligned X-AlpacaEval prompts."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from steerscope.evaluation.dataset import (
    concept_seed,
    expand_factors,
    require_dataset_type,
    require_num_examples,
)
from steerscope.evaluation.version import (
    evaluator_source_fingerprint,
    file_signature,
)
from steerscope.utils.constants import HAS_SYSTEM_PROMPT_MODELS

from .lm_judge import LMJudgeEvaluator
from .prompt_templates import (
    UNIDIRECTIONAL_PAIRWISE_EVALUATION_CONCEPT_RELEVANCE_TEMPLATE,
    UNIDIRECTIONAL_PAIRWISE_EVALUATION_FLUENCY_TEMPLATE,
    UNIDIRECTIONAL_PAIRWISE_EVALUATION_INSTRUCTION_RELEVANCE_TEMPLATE,
)


logger = logging.getLogger(__name__)

__all__ = ["PromptGeneralizationEvaluator"]

DATASET_TYPE = "XAlpacaEval"
PROMPT_BANK_VERSION = "x-alpaca-generalization-v2-concept-seeded"
DEFAULT_DATASET_PATH = "x_alpaca_eval/XAlpacaEval.parquet"
DEFAULT_BASE64_WRAPPER = (
    "The following instruction is Base64-encoded. Decode it carefully and "
    "follow the decoded instruction. Output only the response to the decoded "
    "instruction. Do not explain the decoding process.\n\n"
    "Encoded instruction:\n{payload}"
)
REQUIRED_PROMPT_BANK_COLUMNS = {
    "dataset_name",
    "input_id",
    "prompt_id",
    "source_input_id",
    "source_prompt",
    "original_prompt",
    "raw_input",
    "system_prompt",
    "is_reference",
    "augmenter",
}


@dataclass(frozen=True)
class _PromptVariant:
    user_prompt: str
    system_prompt: str | None = None


class _LanguageAugmenter:
    def __init__(self, *, column: str):
        if not str(column).strip():
            raise ValueError("LanguageAugmenter requires a non-empty column.")
        self.column = str(column)

    @property
    def required_columns(self) -> set[str]:
        return {self.column}

    def apply(self, row: pd.Series) -> _PromptVariant:
        return _PromptVariant(user_prompt=str(row[self.column]))


class _Base64Augmenter:
    def __init__(self, *, source_column: str, wrapper: str = DEFAULT_BASE64_WRAPPER):
        if not str(source_column).strip():
            raise ValueError("Base64Augmenter requires a non-empty source_column.")
        if "{payload}" not in str(wrapper):
            raise ValueError("Base64Augmenter wrapper must contain '{payload}'.")
        self.source_column = str(source_column)
        self.wrapper = str(wrapper)

    @property
    def required_columns(self) -> set[str]:
        return {self.source_column}

    def apply(self, row: pd.Series) -> _PromptVariant:
        payload = base64.b64encode(
            str(row[self.source_column]).encode("utf-8")
        ).decode("ascii")
        return _PromptVariant(user_prompt=self.wrapper.format(payload=payload))


_AUGMENTER_TYPES = {
    "LanguageAugmenter": _LanguageAugmenter,
    "Base64Augmenter": _Base64Augmenter,
}


class PromptGeneralizationEvaluator(LMJudgeEvaluator):
    """Measure OOD joint-effect retention relative to English prompts."""

    supports_factor_pipeline = False
    dataset_type = DATASET_TYPE
    source_dependencies = ("evaluators/prompt_templates.py",)

    def __init__(self, node, context, **params):
        super().__init__(node, context, **params)
        require_dataset_type(self.node.dataset, self.dataset_type)
        require_num_examples(self.node.dataset)
        self.reference_column = str(
            self.node.dataset.get("reference_column", "instruction_en")
        )
        self.source_id_column = str(
            self.node.dataset.get("source_id_column", "id")
        )
        self.source_dataset_column = str(
            self.node.dataset.get("source_dataset_column", "dataset")
        )
        self.min_id_effect = float(self.params.get("min_id_effect", 0.1))
        if self.min_id_effect < 0.0:
            raise ValueError("min_id_effect must be non-negative.")
        configured_minimum = self.params.get("min_selection_improvement")
        self.min_selection_improvement = (
            None
            if configured_minimum is None
            else float(configured_minimum)
        )
        if (
            self.min_selection_improvement is not None
            and self.min_selection_improvement < 0.0
        ):
            raise ValueError(
                "min_selection_improvement must be non-negative."
            )

        configured = self.node.dataset.get("augmenters")
        if not isinstance(configured, list) or not configured:
            raise ValueError(
                "PromptGeneralizationEvaluator requires dataset.augmenters."
            )
        augmenter_configs = []
        labels = set()
        for entry in configured:
            if isinstance(entry, str):
                entry = {"name": entry}
            if not isinstance(entry, dict) or not entry.get("name"):
                raise TypeError(
                    "Each generalization augmenter must be a name or mapping."
                )
            name = str(entry["name"])
            augmenter_type = _AUGMENTER_TYPES.get(name)
            if augmenter_type is None:
                raise ValueError(
                    f"Unknown generalization augmenter '{name}'. Choose from "
                    f"{sorted(_AUGMENTER_TYPES)}."
                )
            label = str(entry.get("id") or self._augmenter_label(name))
            if label in {"identity", "overall"} or label in labels:
                raise ValueError(f"Duplicate or reserved augmenter id '{label}'.")
            labels.add(label)
            kwargs = dict(entry.get("kwargs") or {})
            if name == "Base64Augmenter":
                kwargs.setdefault("source_column", self.reference_column)
            try:
                augmenter = augmenter_type(**kwargs)
            except TypeError as error:
                raise TypeError(
                    f"Invalid configuration for generalization augmenter "
                    f"'{label}': {error}"
                ) from error
            augmenter_configs.append({
                "id": label,
                "name": name,
                "augmenter": augmenter,
            })
        self.augmenter_configs = tuple(augmenter_configs)
        self.augmenter_ids = tuple(
            config["id"] for config in self.augmenter_configs
        )
        self._prompt_banks = {}
        self._baseline_inference = {}

    @classmethod
    def execution_context(cls, node, args):
        require_dataset_type(node.dataset, cls.dataset_type)
        require_num_examples(node.dataset)
        path = cls._configured_dataset_path(node.dataset, args)
        return {
            "source_fingerprint": evaluator_source_fingerprint(cls),
            "x_alpaca_eval": file_signature(path),
            "prompt_bank_version": PROMPT_BANK_VERSION,
        }

    def __str__(self):
        return "PromptGeneralizationEvaluator"

    def evaluation_models(self, models):
        models = list(super().evaluation_models(models))
        if self.min_selection_improvement is None:
            return models
        eligible_methods = self._eligible_methods_from_best_factor(models)
        return [
            model for model in models
            if model.method in eligible_methods
        ]

    def _eligible_methods_from_best_factor(self, models) -> set[str]:
        selection = self.node_config.get("inference", {}).get("select")
        if not selection:
            raise ValueError(
                "min_selection_improvement requires inference.select."
            )
        selected = self._selection_data(self.results, selection)
        required = {"method", "selected_improvement"}
        missing = sorted(required.difference(selected.columns))
        if missing:
            raise KeyError(
                "Prompt generalization factor selection is missing columns: "
                f"{missing}"
            )
        if selected["method"].duplicated().any():
            duplicated = sorted(
                selected.loc[
                    selected["method"].duplicated(keep=False), "method"
                ].astype(str).unique()
            )
            raise ValueError(
                "Prompt generalization requires one method-level best-factor "
                f"row per method; duplicates found for {duplicated}."
            )

        improvements = pd.to_numeric(
            selected["selected_improvement"], errors="coerce"
        )
        if improvements.isna().any() or not np.isfinite(improvements).all():
            raise ValueError(
                "Prompt generalization requires finite selected_improvement "
                "values from BestFactorEvaluator."
            )
        available_methods = set(selected["method"].astype(str))
        expected_methods = {str(model.method) for model in models}
        missing_methods = sorted(expected_methods.difference(available_methods))
        if missing_methods:
            raise ValueError(
                "BestFactorEvaluator has no method-level result for methods: "
                f"{missing_methods}"
            )
        return set(selected.loc[
            improvements >= self.min_selection_improvement,
            "method",
        ].astype(str))

    def open_resources(self, models) -> None:
        if str(getattr(self.args, "runtime_backend", "legacy")).lower() != "legacy":
            raise ValueError(
                "PromptGeneralizationEvaluator requires runtime_backend='legacy' "
                "so baseline and steered generations use one backend."
            )
        super().open_resources(models)
        base_models = {model.target.base_model for model in models}
        if len(base_models) != 1:
            raise ValueError(
                "Prompt generalization requires one shared base model."
            )
        self._prompt_banks = {}
        self._baseline_inference = {}

    def close_resources(self) -> None:
        self._prompt_banks = {}
        self._baseline_inference = {}
        super().close_resources()

    def build_dataset(self, model, factors) -> pd.DataFrame:
        self._ensure_concept_resources(model)
        examples = self._model_ready_bank(model)
        examples["concept_id"] = int(model.concept.concept_id)
        examples["input_concept"] = str(model.concept.text)
        return expand_factors(examples, factors)

    def prepare_examples(self, examples, results=None, config=None):
        """Attach baseline generations so pending inference is self-contained."""
        prepared = super().prepare_examples(
            examples,
            results=results,
            config=config,
        )
        if prepared.empty or "baseline_generation" in prepared:
            return prepared
        if "concept_id" not in prepared:
            raise KeyError(
                "Prompt generalization inference is missing 'concept_id'."
            )
        concept_ids = pd.to_numeric(
            prepared["concept_id"], errors="raise"
        ).astype(int).unique()
        if len(concept_ids) != 1:
            raise ValueError(
                "Prompt generalization target inference must contain exactly "
                "one concept."
            )
        concept_id = int(concept_ids[0])
        baseline = self._baseline_inference.get(concept_id)
        if baseline is None:
            raise RuntimeError(
                f"Baseline inference for concept {concept_id} is not available."
            )
        self._validate_baseline(baseline)
        baseline_rows = baseline[[
            "prompt_id", "baseline_generation"
        ]].drop_duplicates("prompt_id")
        merged = prepared.merge(
            baseline_rows,
            on="prompt_id",
            how="left",
            sort=False,
            validate="many_to_one",
        )
        if merged["baseline_generation"].isna().any():
            missing = sorted(
                merged.loc[
                    merged["baseline_generation"].isna(), "prompt_id"
                ].astype(int).unique()
            )
            raise ValueError(
                "Baseline inference is missing prompt IDs: "
                f"{missing[:10]}"
            )
        return merged

    def release_target_inference_resources(self, representative) -> None:
        concept_id = int(representative.concept.concept_id)
        self._prompt_banks.pop(concept_id, None)
        self._baseline_inference.pop(concept_id, None)

    def _ensure_concept_resources(self, model) -> None:
        concept_id = int(model.concept.concept_id)
        if concept_id not in self._prompt_banks:
            self._prompt_banks[concept_id] = (
                self._load_or_build_prompt_bank(concept_id)
            )
        if concept_id not in self._baseline_inference:
            baseline = model.generate_baseline(self._model_ready_bank(model))
            self._validate_baseline(baseline)
            self._baseline_inference[concept_id] = baseline

    def _model_ready_bank(self, model) -> pd.DataFrame:
        concept_id = int(model.concept.concept_id)
        if concept_id not in self._prompt_banks:
            raise RuntimeError(
                f"Prompt bank for concept {concept_id} is not available."
            )
        bank = self._prompt_banks[concept_id].copy().reset_index(drop=True)
        bank["input"] = [
            self._format_variant(
                model.target.base_model,
                str(row.raw_input),
                None if pd.isna(row.system_prompt) else str(row.system_prompt),
            )
            for row in bank.itertuples(index=False)
        ]
        return bank

    def _format_variant(
        self,
        model_name: str,
        user_prompt: str,
        system_prompt: str | None,
    ) -> str:
        messages = []
        if system_prompt is not None:
            if model_name not in HAS_SYSTEM_PROMPT_MODELS:
                raise ValueError(
                    f"Model '{model_name}' does not support system prompts; "
                    "place the augmenter wrapper in the user prompt."
                )
            messages.append({"role": "system", "content": system_prompt})
        elif model_name in HAS_SYSTEM_PROMPT_MODELS:
            messages.append({
                "role": "system",
                "content": "You are a helpful assistant.",
            })
        messages.append({"role": "user", "content": user_prompt})
        tokens = self._dataset_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        bos_token_id = self._dataset_tokenizer.bos_token_id
        if tokens and bos_token_id is not None and tokens[0] == bos_token_id:
            tokens = tokens[1:]
        return self._dataset_tokenizer.decode(tokens)

    def _load_or_build_prompt_bank(self, concept_id: int) -> pd.DataFrame:
        cache_path = self._prompt_bank_cache_path(concept_id)
        if (
            cache_path.is_file()
            and not bool(getattr(self.args, "overwrite_cache", False))
        ):
            try:
                cached = pd.read_parquet(cache_path)
                self._validate_prompt_bank(cached)
                logger.warning("Using cached prompt bank %s", cache_path)
                return cached
            except Exception as error:
                logger.warning(
                    "Ignoring invalid prompt bank cache %s: %s",
                    cache_path,
                    error,
                )

        prompt_bank = self._build_prompt_bank(concept_id)
        self._validate_prompt_bank(prompt_bank)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=cache_path.parent,
            suffix=".parquet.tmp",
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            prompt_bank.to_parquet(temporary_path, index=False)
            os.replace(temporary_path, cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return prompt_bank

    def _build_prompt_bank(self, concept_id: int) -> pd.DataFrame:
        source = self._load_fixed_source(concept_id)
        rows = []
        prompt_id = 0
        for _, source_row in source.iterrows():
            source_id = int(source_row[self.source_id_column])
            source_prompt = str(source_row[self.reference_column])
            source_dataset = str(source_row[self.source_dataset_column])
            rows.append(self._prompt_row(
                prompt_id=prompt_id,
                source_id=source_id,
                source_dataset=source_dataset,
                source_prompt=source_prompt,
                variant=_PromptVariant(source_prompt),
                is_reference=True,
                augmenter="identity",
            ))
            prompt_id += 1
            for config in self.augmenter_configs:
                variant = config["augmenter"].apply(source_row)
                rows.append(self._prompt_row(
                    prompt_id=prompt_id,
                    source_id=source_id,
                    source_dataset=source_dataset,
                    source_prompt=source_prompt,
                    variant=variant,
                    is_reference=False,
                    augmenter=config["id"],
                ))
                prompt_id += 1
        return pd.DataFrame(rows)

    def _load_fixed_source(self, concept_id: int) -> pd.DataFrame:
        path = self._dataset_path()
        if not path.is_file():
            raise FileNotFoundError(f"X-AlpacaEval data not found at {path}.")
        data = pd.read_parquet(path)
        required = {
            self.source_id_column,
            self.source_dataset_column,
            self.reference_column,
        }
        for config in self.augmenter_configs:
            required.update(config["augmenter"].required_columns)
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(
                f"X-AlpacaEval data is missing columns: {missing}"
            )
        if data[self.source_id_column].isna().any() or data[
            self.source_id_column
        ].duplicated().any():
            raise ValueError("X-AlpacaEval source IDs must be unique and non-null.")
        for column in required.difference({self.source_id_column}):
            if data[column].isna().any() or not data[column].astype(
                str
            ).str.strip().all():
                raise ValueError(
                    f"X-AlpacaEval column '{column}' contains empty values."
                )
        num_examples = require_num_examples(self.node.dataset)
        if num_examples > len(data):
            raise ValueError(
                f"X-AlpacaEval requested {num_examples} examples, but only "
                f"{len(data)} are available."
            )
        base_seed = int(
            self.node.dataset.get("seed", getattr(self.args, "seed", 42))
        )
        return data.sample(
            n=num_examples,
            random_state=concept_seed(
                base_seed,
                concept_id,
                DATASET_TYPE,
            ),
        ).reset_index(drop=True)

    @staticmethod
    def _prompt_row(
        *,
        prompt_id: int,
        source_id: int,
        source_dataset: str,
        source_prompt: str,
        variant: _PromptVariant,
        is_reference: bool,
        augmenter: str,
    ) -> dict:
        if not variant.user_prompt.strip():
            raise ValueError(
                f"Generalization augmenter '{augmenter}' produced an empty prompt."
            )
        return {
            "dataset_name": DATASET_TYPE,
            "input_id": int(prompt_id),
            "prompt_id": int(prompt_id),
            "source_input_id": int(source_id),
            "source_dataset": str(source_dataset),
            "source_prompt": str(source_prompt),
            "original_prompt": str(variant.user_prompt),
            "raw_input": str(variant.user_prompt),
            "system_prompt": variant.system_prompt,
            "is_reference": bool(is_reference),
            "augmenter": str(augmenter),
            "suppress_original": "",
            "suppress_rewrite": "",
            "steered_prompt": "",
            "defense": [],
        }

    def _prompt_bank_cache_path(self, concept_id: int) -> Path:
        payload = {
            "version": PROMPT_BANK_VERSION,
            "concept_id": int(concept_id),
            "dataset": dict(self.node.dataset),
            "source": file_signature(self._dataset_path()),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:24]
        return (
            self.output_dir
            / "generalization_prompt_bank"
            / f"concept-{int(concept_id)}-{digest}.parquet"
        )

    def _dataset_path(self) -> Path:
        return self._configured_dataset_path(self.node.dataset, self.args)

    @staticmethod
    def _configured_dataset_path(dataset_config, args) -> Path:
        configured = Path(
            str(dataset_config.get("path", DEFAULT_DATASET_PATH))
        ).expanduser()
        if configured.is_absolute():
            return configured.resolve()
        master_data_dir = getattr(args, "master_data_dir", None)
        if not master_data_dir:
            raise ValueError(
                "X-AlpacaEval requires evaluate.master_data_dir when dataset.path "
                "is relative."
            )
        return (Path(master_data_dir) / configured).resolve()

    def _validate_prompt_bank(self, prompt_bank: pd.DataFrame) -> None:
        missing = sorted(
            REQUIRED_PROMPT_BANK_COLUMNS.difference(prompt_bank.columns)
        )
        if missing:
            raise KeyError(f"Prompt bank is missing columns: {missing}")
        if prompt_bank.empty:
            raise ValueError("Prompt bank is empty.")
        if prompt_bank["prompt_id"].duplicated().any():
            raise ValueError("Prompt bank contains duplicate prompt_id values.")
        expected_scopes = {"identity", *self.augmenter_ids}
        for source_id, group in prompt_bank.groupby("source_input_id"):
            counts = group["augmenter"].value_counts().to_dict()
            if set(counts) != expected_scopes or any(
                count != 1 for count in counts.values()
            ):
                raise ValueError(
                    f"Source prompt {source_id} must contain exactly one row "
                    f"for each scope {sorted(expected_scopes)}."
                )
            references = group[group["is_reference"].astype(bool)]
            if len(references) != 1 or references.iloc[0]["augmenter"] != "identity":
                raise ValueError(
                    f"Source prompt {source_id} must contain one identity reference."
                )

    @staticmethod
    def _validate_baseline(baseline: pd.DataFrame) -> None:
        required = {"prompt_id", "source_prompt", "baseline_generation"}
        missing = sorted(required.difference(baseline.columns))
        if missing:
            raise KeyError(f"Baseline inference is missing columns: {missing}")
        if baseline["prompt_id"].duplicated().any():
            raise ValueError(
                "Baseline inference contains duplicate prompt_id values."
            )

    def compute_metrics(self, data, write_to_dir=None):
        required = {
            "factor",
            "prompt_id",
            "source_input_id",
            "source_prompt",
            "is_reference",
            "augmenter",
            f"{self.model_name}_steered_generation",
        }
        missing = sorted(required.difference(data.columns))
        if missing:
            raise KeyError(
                f"Prompt generalization inference is missing columns: {missing}"
            )
        scored = data.copy().reset_index(drop=True)
        steered_ratings = self._joint_ratings(
            scored,
            f"{self.model_name}_steered_generation",
            api_name=f"{self.model_name}_steered",
        )
        if "baseline_generation" in scored:
            baseline_values = scored[[
                "prompt_id", "baseline_generation"
            ]].drop_duplicates()
            if baseline_values["prompt_id"].duplicated().any():
                raise ValueError(
                    "Prompt generalization inference contains conflicting "
                    "baseline generations for one prompt_id."
                )
            baseline_rows = scored[[
                "prompt_id", "source_prompt", "baseline_generation"
            ]].drop_duplicates("prompt_id")
        else:
            # Backward-compatible path for direct callers and old in-memory
            # fixtures.  Pipeline checkpoints always use the self-contained
            # column above.
            concept_id = int(self.target.concept.concept_id)
            baseline_inference = self._baseline_inference.get(concept_id)
            if baseline_inference is None:
                raise RuntimeError(
                    f"Baseline inference for concept {concept_id} is not "
                    "available."
                )
            baseline_rows = baseline_inference[
                ["prompt_id", "source_prompt", "baseline_generation"]
            ].drop_duplicates("prompt_id")
        self._validate_baseline(baseline_rows)
        baseline_ratings = self._joint_ratings(
            baseline_rows,
            "baseline_generation",
            api_name="baseline",
        )
        baseline_by_prompt = {
            component: dict(zip(
                baseline_rows["prompt_id"].astype(int),
                scores,
            ))
            for component, scores in baseline_ratings["scores"].items()
        }
        missing_baselines = sorted(
            set(scored["prompt_id"].astype(int)).difference(
                baseline_by_prompt["overall"]
            )
        )
        if missing_baselines:
            raise ValueError(
                "Baseline inference is missing prompt IDs: "
                f"{missing_baselines[:10]}"
            )

        for component in ("concept", "instruction", "fluency", "overall"):
            scored[f"_steered_{component}_score"] = np.asarray(
                steered_ratings["scores"][component],
                dtype=float,
            )
            scored[f"_baseline_{component}_score"] = [
                float(baseline_by_prompt[component][int(prompt_id)])
                for prompt_id in scored["prompt_id"]
            ]
        scored["_net_steering_effect"] = (
            scored["_steered_overall_score"]
            - scored["_baseline_overall_score"]
        )
        self._validate_factor_pairs(scored)

        group_metrics = self._group_metrics(scored)
        metrics = {
            column: group_metrics[column].tolist()
            for column in group_metrics.columns
        }
        for component in ("concept", "instruction", "fluency", "overall"):
            metrics[f"raw_steered_{component}_score"] = scored[
                f"_steered_{component}_score"
            ].tolist()
            metrics[f"raw_baseline_{component}_score"] = scored[
                f"_baseline_{component}_score"
            ].tolist()
        metrics.update({
            "raw_net_steering_effect": scored[
                "_net_steering_effect"
            ].tolist(),
        })
        for component in ("concept", "instruction", "fluency"):
            metrics[f"steered_{component}_completions"] = (
                steered_ratings["completions"][component]
            )
            metrics[f"baseline_{component}_completion_count"] = len(
                baseline_ratings["completions"][component]
            )
        return metrics

    def _joint_ratings(self, data, generation_column, api_name):
        generations = data[generation_column].astype(str).tolist()
        concept_prompts = [
            UNIDIRECTIONAL_PAIRWISE_EVALUATION_CONCEPT_RELEVANCE_TEMPLATE.format(
                concept=self.target.concept.text,
                sentence=generation,
            )
            for generation in generations
        ]
        instruction_prompts = [
            UNIDIRECTIONAL_PAIRWISE_EVALUATION_INSTRUCTION_RELEVANCE_TEMPLATE.format(
                instruction=source_prompt,
                sentence=generation,
            )
            for source_prompt, generation in zip(
                data["source_prompt"].astype(str),
                generations,
            )
        ]
        fluency_prompts = [
            UNIDIRECTIONAL_PAIRWISE_EVALUATION_FLUENCY_TEMPLATE.format(
                sentence=generation,
            )
            for generation in generations
        ]
        (
            (concept_scores, concept_completions),
            (instruction_scores, instruction_completions),
            (fluency_scores, fluency_completions),
        ) = self._get_rating_groups([
            (f"{api_name}_concept", concept_prompts),
            (f"{api_name}_instruction", instruction_prompts),
            (f"{api_name}_fluency", fluency_prompts),
        ])
        overall_scores = [
            self._harmonic_mean(list(scores))
            for scores in zip(
                concept_scores,
                instruction_scores,
                fluency_scores,
            )
        ]
        return {
            "scores": {
                "concept": concept_scores,
                "instruction": instruction_scores,
                "fluency": fluency_scores,
                "overall": overall_scores,
            },
            "completions": {
                "concept": concept_completions,
                "instruction": instruction_completions,
                "fluency": fluency_completions,
            },
        }

    def _group_metrics(self, scored: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for factor, factor_rows in scored.groupby("factor", sort=True):
            reference = self._effects_by_source(factor_rows, "identity")
            ood_effects = {
                scope: self._effects_by_source(factor_rows, scope)
                for scope in self.augmenter_ids
            }
            for scope, effects in ood_effects.items():
                rows.append(self._metric_row(
                    factor=float(factor),
                    scope=scope,
                    reference=reference,
                    ood=effects,
                ))
            overall = pd.concat(ood_effects, axis=1).mean(axis=1)
            rows.append(self._metric_row(
                factor=float(factor),
                scope="overall",
                reference=reference,
                ood=overall,
            ))
        if not rows:
            raise ValueError(
                "Prompt generalization data contains no reportable rows."
            )
        return pd.DataFrame(rows)

    def _metric_row(
        self,
        *,
        factor: float,
        scope: str,
        reference: pd.Series,
        ood: pd.Series,
    ) -> dict:
        if not reference.index.equals(ood.index):
            raise ValueError(
                f"Generalization scope '{scope}' is not paired with identity."
            )
        id_effect = float(reference.mean())
        ood_effect = float(ood.mean())
        return {
            "factor": factor,
            "scope": scope,
            "id_effect": id_effect,
            "ood_effect": ood_effect,
            "num_prompts": int(len(reference)),
        }

    @staticmethod
    def _effects_by_source(
        factor_rows: pd.DataFrame,
        scope: str,
    ) -> pd.Series:
        selected = factor_rows[factor_rows["augmenter"] == scope]
        if selected.empty:
            raise ValueError(
                f"Generalization factor is missing scope '{scope}'."
            )
        if selected["source_input_id"].duplicated().any():
            raise ValueError(
                f"Generalization scope '{scope}' has duplicate source prompts."
            )
        return selected.set_index("source_input_id")[
            "_net_steering_effect"
        ].astype(float).sort_index()

    @staticmethod
    def _validate_factor_pairs(scored: pd.DataFrame) -> None:
        keys = ["factor", "prompt_id"]
        if scored.duplicated(keys).any():
            raise ValueError(
                "Prompt generalization inference contains duplicate "
                "factor/prompt rows."
            )
        expected = None
        for factor, group in scored.groupby("factor", sort=False):
            prompt_ids = set(group["prompt_id"].astype(int))
            if expected is None:
                expected = prompt_ids
            elif prompt_ids != expected:
                raise ValueError(
                    "Every factor must use the same fixed prompt bank; "
                    f"factor {factor} has mismatched prompt IDs."
                )

    def render_report(self, result, output_dir=None):
        if result.metrics is None or result.metrics.empty:
            raise ValueError("Prompt generalization has no metrics to report.")
        if "scope" not in result.metrics:
            raise KeyError(
                "Prompt generalization metrics are missing 'scope'."
            )
        method_metrics = self._aggregate_method_metrics(result.metrics)
        metric_labels = {
            "id_effect": "ID net effect",
            "ood_effect": "OOD net effect",
            "retention": "OOD / ID retention",
        }
        summaries = []
        for scope in ("overall", *self.augmenter_ids):
            scoped = method_metrics[method_metrics["scope"] == scope]
            if scoped.empty:
                continue
            summary = self._curve_summary(scoped, metric_labels.keys())
            metadata = scoped[[
                "method",
                "factor",
                "retention_valid",
                "num_concepts",
                "num_prompts",
            ]].drop_duplicates()
            summary = summary.merge(
                metadata,
                on=["method", "factor"],
                how="left",
                validate="many_to_one",
            )
            summary.insert(0, "scope", scope)
            summaries.append(summary)
        if not summaries:
            raise ValueError(
                "Prompt generalization has no reportable scopes."
            )
        summary_path = self._save_report_summary(
            pd.concat(summaries, ignore_index=True),
            output_dir,
        )
        paths = [summary_path]
        for summary in summaries:
            scope = str(summary.iloc[0]["scope"])
            scoped_metrics = method_metrics[
                method_metrics["scope"] == scope
            ]
            paths.extend(self._render_scope_figure(
                summary,
                scoped_metrics,
                scope,
                metric_labels,
                output_dir,
            ))
        return paths

    def _aggregate_method_metrics(self, metrics: pd.DataFrame) -> pd.DataFrame:
        required = {
            "method",
            "concept_id",
            "factor",
            "scope",
            "id_effect",
            "ood_effect",
        }
        missing = sorted(required.difference(metrics.columns))
        if missing:
            raise KeyError(
                "Prompt generalization metrics cannot be aggregated without "
                f"columns: {missing}"
            )

        data = metrics.copy()
        for column in ("factor", "id_effect", "ood_effect"):
            data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[["factor", "id_effect", "ood_effect"]].isna().any().any():
            raise ValueError(
                "Prompt generalization aggregation requires numeric factor, "
                "ID effect, and OOD effect values."
            )

        keys = ["method", "concept_id", "factor", "scope"]
        if data.duplicated(keys).any():
            duplicates = data.loc[data.duplicated(keys, keep=False), keys]
            raise ValueError(
                "Prompt generalization contains duplicate method/concept/"
                f"factor/scope rows: {duplicates.head().to_dict('records')}"
            )

        expected_concepts = frozenset(data["concept_id"].unique())
        group_keys = ["method", "factor", "scope"]
        concept_sets = data.groupby(
            group_keys, dropna=False, sort=True
        )["concept_id"].agg(lambda values: frozenset(values))
        incomplete = concept_sets[concept_sets != expected_concepts]
        if not incomplete.empty:
            identity = incomplete.index[0]
            missing_concepts = sorted(expected_concepts.difference(
                incomplete.iloc[0]
            ))
            raise ValueError(
                "Every method/factor/scope must use the complete concept set; "
                f"{identity} is missing concepts {missing_concepts}."
            )

        aggregations = {
            "id_effect": ("id_effect", "mean"),
            "ood_effect": ("ood_effect", "mean"),
            "num_concepts": ("concept_id", "nunique"),
        }
        if "num_prompts" in data:
            aggregations["num_prompts"] = ("num_prompts", "sum")
        aggregated = (
            data.groupby(group_keys, dropna=False, sort=True)
            .agg(**aggregations)
            .reset_index()
        )
        if "num_prompts" not in aggregated:
            aggregated["num_prompts"] = 0
        aggregated["retention_valid"] = (
            (aggregated["factor"] != 0.0)
            & (aggregated["id_effect"] >= self.min_id_effect)
        )
        aggregated["retention"] = np.where(
            aggregated["retention_valid"],
            aggregated["ood_effect"] / aggregated["id_effect"],
            np.nan,
        )
        return aggregated

    def _render_scope_figure(
        self,
        summary,
        metrics,
        scope,
        metric_labels,
        output_dir,
    ):
        plt = self._pyplot()
        figure, axes = plt.subplots(
            2,
            3,
            figsize=(18.0, 9.0),
            squeeze=False,
        )
        methods = (
            summary["method"].drop_duplicates().tolist()
            if "method" in summary
            else [None]
        )
        styles = self._method_plot_styles(methods, plt)
        for axis, (metric, label) in zip(
            axes[0], metric_labels.items()
        ):
            metric_data = summary[summary["metric"] == metric]
            for method in methods:
                curve = self._method_rows(metric_data, method).sort_values(
                    "factor"
                )
                if curve.empty:
                    continue
                style = styles[self._method_plot_key(method)]
                line = axis.plot(
                    curve["factor"].astype(float),
                    curve["mean"].astype(float),
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=1.8,
                    markersize=3.5,
                    label=str(method) if method is not None else None,
                )[0]
                axis.fill_between(
                    curve["factor"].astype(float).to_numpy(),
                    curve["ci_lower"].astype(float).to_numpy(),
                    curve["ci_upper"].astype(float).to_numpy(),
                    color=line.get_color(),
                    alpha=0.16,
                    linewidth=0,
                )
            axis.set_title(f"{label} vs factor")
            axis.set_xlabel("Steering factor")
            axis.set_ylabel("Score")
            axis.grid(alpha=0.25)

        score_axes = (
            (axes[1, 0], "ood_effect", "OOD net effect"),
            (axes[1, 1], "retention", "OOD / ID retention"),
        )
        for axis, metric, label in score_axes:
            for method in methods:
                method_rows = self._method_rows(metrics, method)
                if method_rows.empty:
                    continue
                curve = (
                    method_rows.groupby("factor", sort=True)[
                        ["id_effect", metric]
                    ]
                    .mean()
                    .dropna()
                    .reset_index()
                    .sort_values("factor")
                )
                if curve.empty:
                    continue
                style = styles[self._method_plot_key(method)]
                axis.plot(
                    curve["id_effect"].astype(float),
                    curve[metric].astype(float),
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=1.8,
                    markersize=3.5,
                    label=str(method) if method is not None else None,
                )
            axis.set_title(f"{label} vs ID effect")
            axis.set_xlabel("ID net effect")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
        axes[1, 2].set_visible(False)

        if methods != [None]:
            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                figure.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.965),
                    ncol=min(6, len(labels)),
                    frameon=False,
                )
                figure.subplots_adjust(top=0.88)
        figure.suptitle(
            f"{self.node_id}: {scope}",
            fontsize=14,
            y=0.995,
        )
        return self._save_report_figure(
            figure,
            output_dir,
            stem=self._safe_stem(scope),
        )

    @staticmethod
    def _method_rows(data: pd.DataFrame, method):
        if method is None or "method" not in data:
            return data
        return data[
            data["method"].isna()
            if pd.isna(method)
            else data["method"] == method
        ]

    @staticmethod
    def _augmenter_label(name: str) -> str:
        value = re.sub(r"Augmenter$", "", name)
        value = re.sub(r"(?<!^)(?=[A-Z])", "_", value)
        return value.lower()

    @staticmethod
    def _safe_stem(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
