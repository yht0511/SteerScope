from .alpaca import AlpacaEvaluator
from .judge import JudgeEvaluatorMixin
from .prompt_templates import (
    UNIDIRECTIONAL_PAIRWISE_EVALUATION_CONCEPT_RELEVANCE_TEMPLATE,
    UNIDIRECTIONAL_PAIRWISE_EVALUATION_FLUENCY_TEMPLATE,
    UNIDIRECTIONAL_PAIRWISE_EVALUATION_INSTRUCTION_RELEVANCE_TEMPLATE,
)

import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import pandas as pd

from steerscope.evaluation.result import EvaluationResult
from steerscope.evaluation.version import evaluator_source_fingerprint

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


class _InferenceSpool:
    """Atomic, execution-scoped disk queue for generated judge inputs."""

    def __init__(self, progress):
        self.execution_hash = progress.execution_hash
        self.root = progress.root / "judge_pending_inference"
        self.root.mkdir(parents=True, exist_ok=True)

    def load(self, target_id: str) -> pd.DataFrame | None:
        target_dir = self._target_dir(target_id)
        manifest = self._read_json(target_dir / "manifest.json")
        if not (
            manifest
            and manifest.get("status") == "complete"
            and manifest.get("execution_hash") == self.execution_hash
            and manifest.get("target_id") == target_id
        ):
            return None
        path = target_dir / "inference.parquet"
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            return None

    def save(self, target_id: str, inference: pd.DataFrame) -> None:
        target_dir = self._target_dir(target_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = target_dir / "manifest.json"
        self._write_json_atomic(
            manifest_path,
            {
                "status": "running",
                "execution_hash": self.execution_hash,
                "target_id": target_id,
            },
        )
        self._save_frame_atomic(target_dir / "inference.parquet", inference)
        self._write_json_atomic(
            manifest_path,
            {
                "status": "complete",
                "execution_hash": self.execution_hash,
                "target_id": target_id,
                "rows": int(len(inference)),
            },
        )

    def clear(self, target_id: str) -> None:
        target_dir = self._target_dir(target_id)
        for name in (
            "inference.parquet",
            "manifest.json",
        ):
            try:
                (target_dir / name).unlink(missing_ok=True)
            except OSError:
                # A completed target is authoritative; a stale spool entry is
                # harmless and will be ignored before any future scoring.
                pass
        try:
            target_dir.rmdir()
        except OSError:
            pass

    def _target_dir(self, target_id: str) -> Path:
        digest = hashlib.sha256(target_id.encode("utf-8")).hexdigest()[:24]
        return self.root / digest

    @staticmethod
    def _save_frame_atomic(path: Path, data: pd.DataFrame) -> None:
        with tempfile.NamedTemporaryFile(
            delete=False, dir=path.parent, suffix=".parquet.tmp"
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            data.to_parquet(temporary_path, index=False)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _read_json(path: Path):
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as file:
                return json.load(file)
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_json_atomic(path: Path, payload) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            dir=path.parent,
            suffix=".json.tmp",
            encoding="utf-8",
        ) as temporary:
            json.dump(payload, temporary, indent=2, sort_keys=True)
            temporary_path = Path(temporary.name)
        try:
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)


class LMJudgeEvaluator(JudgeEvaluatorMixin, AlpacaEvaluator):
    source_dependencies = ("evaluators/prompt_templates.py",)
    DEFAULT_RATING = 0.0
    supports_factor_pipeline = True

    @classmethod
    def execution_context(cls, node, args):
        context = super().execution_context(node, args)
        context["source_fingerprint"] = evaluator_source_fingerprint(cls)
        return context

    def evaluate(self, models, concepts) -> EvaluationResult:
        """Pipeline target generation into asynchronous judging while checkpointing each completed stage."""
        enabled = self.params.get("judge_pipeline_enabled", True)
        if not isinstance(enabled, bool):
            raise TypeError("judge_pipeline_enabled must be a boolean.")
        if not enabled:
            return super().evaluate(models, concepts)
        if not self.node.requires_inference:
            raise NotImplementedError(
                f"Result-only evaluator '{self.__class__.__name__}' must own "
                "its dependency queries by implementing evaluate()."
            )
        if not models:
            raise ValueError(
                f"Evaluator '{self.node_id}' received no model wrappers."
            )

        self._validate_concepts(models, concepts)
        self.prepare_evaluation(models, concepts)
        scheduled_models = list(self.evaluation_models(models))
        if not scheduled_models:
            raise ValueError(
                f"Evaluator '{self.node_id}' selected no models for evaluation."
            )
        target_groups = list(self._models_by_target(scheduled_models))
        target_ids = [group[0].target.target_id for group in target_groups]
        self._selected_models_by_target = {
            group[0].target.target_id: self.select_models(group, self.results)
            for group in target_groups
        }
        if self.progress is not None:
            self.progress.begin(target_ids)
        spool = (
            _InferenceSpool(self.progress)
            if self.progress is not None
            else None
        )
        self._open_model_progress(target_groups)

        restored_results = {}
        futures = {}
        in_memory_inference = {}
        metadata = {}
        try:
            self.open_resources(scheduled_models)
            scorer = copy.copy(self)
            # tqdm owns its own lock and may safely receive producer updates
            # and judge usage postfix updates from the consumer thread.
            scorer._model_progress_bar = self._model_progress_bar
            with ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"{self.node_id}-judge",
            ) as executor:
                for target_models in target_groups:
                    target_id = target_models[0].target.target_id
                    restored = (
                        self.progress.load_target(target_id)
                        if self.progress is not None
                        else None
                    )
                    if restored is not None:
                        logger.warning(
                            "Evaluator %s restoring completed target %s",
                            self.node_id,
                            target_id,
                        )
                        restored_results[target_id] = restored
                        spool.clear(target_id)
                        self._advance_model_progress(
                            self.model_evaluation_count(target_models),
                            description=(
                                f"{self.node_id} | cached {target_id}"
                            ),
                        )
                        continue

                    inference = (
                        spool.load(target_id)
                        if spool is not None
                        else None
                    )
                    representative = self._selected_models(target_models)[0]
                    if inference is None:
                        representative, inference = (
                            self._generate_target_inference(target_models)
                        )
                        if spool is not None:
                            spool.save(target_id, inference)
                        else:
                            in_memory_inference[target_id] = inference
                        self.release_target_inference_resources(representative)
                    else:
                        logger.warning(
                            "Evaluator %s restoring pending inference for "
                            "target %s",
                            self.node_id,
                            target_id,
                        )
                        self._advance_model_progress(
                            self.model_evaluation_count(target_models),
                            description=(
                                f"{self.node_id} | inference cached {target_id}"
                            ),
                        )

                    futures[target_id] = executor.submit(
                        self._score_pipeline_target,
                        scorer,
                        representative,
                        target_id,
                        spool,
                        (
                            None
                            if self.progress is not None
                            else in_memory_inference.pop(target_id)
                        ),
                    )

                # This is the evaluator's only barrier: every inference target
                # has now been produced or restored.  Await judging in stable
                # target order before downstream evaluators are allowed to run.
                for target_id in target_ids:
                    future = futures.get(target_id)
                    if future is None:
                        continue
                    result = future.result()
                    if result is not None:
                        restored_results[target_id] = result
            metadata = self.evaluation_metadata()
        finally:
            try:
                self.close_resources()
            finally:
                self._close_model_progress()

        if self.progress is not None:
            return self.progress.combine(
                target_ids,
                metadata_combiner=self.combine_metadata,
            )
        ordered = [restored_results[target_id] for target_id in target_ids]
        return EvaluationResult(
            inference=self._concat([
                result.inference
                for result in ordered
                if result.inference is not None
            ]),
            samples=self._concat([
                result.samples
                for result in ordered
                if result.samples is not None
            ]),
            metrics=self._concat([
                result.metrics
                for result in ordered
                if result.metrics is not None
            ]),
            metadata=metadata,
        )

    def _score_pipeline_target(
        self,
        scorer,
        representative,
        target_id,
        spool,
        inference,
    ):
        if inference is None:
            inference = spool.load(target_id)
            if inference is None:
                raise RuntimeError(
                    f"Evaluator '{self.node_id}' lost pending inference for "
                    f"target '{target_id}'."
                )
        scorer.begin_target(target_id)
        metrics, samples = scorer.score(representative, inference)
        result = EvaluationResult(
            inference=inference,
            samples=samples,
            metrics=metrics,
            metadata=scorer.target_metadata(),
        )
        if self.progress is None:
            return result
        scorer.checkpoint_resources()
        self.progress.save_target(target_id, result)
        spool.clear(target_id)
        return None

    def evaluate_target(self, target_models) -> EvaluationResult:
        """Evaluate one target serially for direct/programmatic callers."""
        representative, inference = self._generate_target_inference(
            target_models
        )
        metrics, samples = self.score(representative, inference)
        return EvaluationResult(
            inference=inference,
            samples=samples,
            metrics=metrics,
            metadata=self.target_metadata(),
        )

    def _generate_target_inference(self, target_models):
        """Generate every configured factor for one target without judging."""
        selected_models = self._selected_models(target_models)
        baseline_only = self.params.get("baseline_only", False)
        if not isinstance(baseline_only, bool):
            raise TypeError("baseline_only must be a boolean.")
        if baseline_only and not self.supports_factor_pipeline:
            raise ValueError(
                "baseline_only is supported only by the base LMJudgeEvaluator."
            )
        if baseline_only and len(selected_models) != 1:
            raise ValueError(
                "baseline_only requires exactly one representative model."
            )
        include_baseline = self.params.get("include_baseline", False)
        if not isinstance(include_baseline, bool):
            raise TypeError("include_baseline must be a boolean.")
        include_baseline = include_baseline or baseline_only

        representative = selected_models[0]
        factors = [model.factor for model in selected_models]
        baseline_factor = None
        if include_baseline:
            baseline_factor = float(self.params.get("baseline_factor", 0.0))
            if not math.isfinite(baseline_factor):
                raise ValueError("baseline_factor must be finite.")
            if baseline_factor in factors:
                raise ValueError(
                    "include_baseline requires baseline_factor to differ from "
                    "every configured model factor."
                )
        dataset_factors = (
            [baseline_factor, *factors]
            if baseline_factor is not None
            else factors
        )
        examples = self.build_dataset(representative, dataset_factors)
        inference_frames = []

        jobs = []
        if baseline_factor is not None:
            jobs.append(("baseline", baseline_factor, representative))
        if not baseline_only:
            jobs.extend(("model", model.factor, model) for model in selected_models)

        for job_kind, factor, model in jobs:
            if job_kind == "baseline":
                factor_column = (
                    "model_factor" if "model_factor" in examples else "factor"
                )
                model_examples = examples[
                    examples[factor_column] == factor
                ].copy().reset_index(drop=True)
            else:
                model_examples = self.examples_for_model(examples, model)
            self._set_current_model(model)
            if job_kind == "baseline":
                generated = model.generate_baseline(model_examples)
                if "baseline_generation" not in generated:
                    raise KeyError(
                        "Baseline inference did not return "
                        "'baseline_generation'."
                    )
                generated = generated.copy()
                generated[
                    f"{representative.method}_steered_generation"
                ] = generated.pop("baseline_generation")
                if "method" not in generated:
                    generated.insert(0, "method", representative.method)
                if "target_id" not in generated:
                    generated.insert(
                        0, "target_id", representative.target.target_id
                    )
            else:
                generated = model.generate(model_examples)
                self._advance_model_progress()
            if job_kind == "baseline" and baseline_only:
                self._advance_model_progress()
            inference_frames.append(generated)

        inference = self.prepare_examples(
            self._concat(inference_frames),
            results=self.results,
            config=self.node_config,
        )
        if inference.empty:
            raise ValueError(
                f"Evaluator '{self.node_id}' produced no examples for "
                f"target '{representative.target.target_id}'."
            )
        return representative, inference

    def release_target_inference_resources(self, representative) -> None:
        """Release producer-only per-target state after inference is durable."""

    def render_report(self, result, output_dir=None):
        return self._render_curve_report(
            result,
            {
                "lm_judge_rating": "Overall",
                "relevance_concept_ratings": "Concept relevance",
                "relevance_instruction_ratings": "Instruction relevance",
                "fluency_ratings": "Fluency",
            },
            output_dir,
            y_limits=(0.0, 2.0),
        )

    def __str__(self):
        return 'LMJudgeEvaluator'

    def _get_rating_from_completion(self, completion):
        if "Rating:" not in completion:
            raise ValueError("Cannot find rating value.")
        rating_text = completion.split("Rating:")[-1].strip()
        rating_text = rating_text.split('\n')[0].strip()
        rating_text = rating_text.replace('[', '').replace(']', '')
        rating_text = rating_text.rstrip('.').strip('"').strip("'").strip("*").strip()
        return float(rating_text)

    def _get_ratings_from_prompts(self, prompts, api_name, min_rating=0.0, max_rating=2.0):
        return self._get_judge_ratings(
            prompts,
            f"{api_name}_{self.model_name}_LMJudgeEvaluator",
            self._get_rating_from_completion,
            min_rating=min_rating,
            max_rating=max_rating,
            default_rating=self.DEFAULT_RATING,
        )

    def _get_rating_groups(self, prompt_groups):
        lengths = [len(prompts) for _, prompts in prompt_groups]
        prompts = [prompt for _, group in prompt_groups for prompt in group]
        api_names = [
            f"{api_name}_{self.model_name}_LMJudgeEvaluator"
            for api_name, group in prompt_groups
            for _ in group
        ]

        ratings, completions = self._get_judge_ratings(
            prompts,
            api_names,
            self._get_rating_from_completion,
            default_rating=self.DEFAULT_RATING,
        )
        groups = []
        offset = 0
        for length in lengths:
            group_ratings = ratings[offset:offset + length]
            group_completions = completions[offset:offset + length]
            groups.append((
                group_ratings,
                group_completions,
            ))
            offset += length
        return groups

    @staticmethod
    def _harmonic_mean(scores):
        """Aggregate judge dimensions while making any zero score decisive."""
        if 0 in scores:
            return 0.0
        return len(scores) / sum(1 / score for score in scores)

    def _get_all_ratings_from_data(self, data, column_name):
        model_relevance_concept_prompts = []
        model_relevance_instruction_prompts = []
        model_fluency_prompts = []
        dataset_names = []
        # This is a generation dataset.
        for idx, row in data.iterrows():
            dataset_name = row["dataset_name"]
            input_concept = row["input_concept"]
            original_prompt = row["original_prompt"]
            generation = row[f"{column_name}_steered_generation"]
            model_relevance_concept_prompts += [UNIDIRECTIONAL_PAIRWISE_EVALUATION_CONCEPT_RELEVANCE_TEMPLATE.format(
                concept=input_concept,
                sentence=generation
            )]
                
            model_relevance_instruction_prompts += [UNIDIRECTIONAL_PAIRWISE_EVALUATION_INSTRUCTION_RELEVANCE_TEMPLATE.format(
                instruction=original_prompt,
                sentence=generation
            )]
            model_fluency_prompts += [UNIDIRECTIONAL_PAIRWISE_EVALUATION_FLUENCY_TEMPLATE.format(
                sentence=generation
            )]
            dataset_names += [dataset_name]
        (
            (model_relevance_concept_ratings, model_relevance_concept_completions),
            (model_relevance_instruction_ratings, model_relevance_instruction_completions),
            (model_fluency_ratings, model_fluency_completions),
        ) = self._get_rating_groups([
            (f"{column_name}_concept", model_relevance_concept_prompts),
            (f"{column_name}_instruction", model_relevance_instruction_prompts),
            (f"{column_name}_fluency", model_fluency_prompts),
        ])
        return list(zip(model_relevance_concept_prompts, model_relevance_concept_ratings)), \
               list(zip(model_relevance_instruction_prompts, model_relevance_instruction_ratings)), \
               list(zip(model_fluency_prompts, model_fluency_ratings)), \
               model_relevance_concept_completions, model_relevance_instruction_completions, model_fluency_completions, dataset_names

    def compute_metrics(self, data, write_to_dir=None):
        """Score concept relevance, instruction relevance, and fluency. Overall is their sum when both relevance scores are positive, and zero otherwise."""
        logger.warning(
            f"Starting task for concept_id: {self.concept_id}, "
            f"model: {self.model_name}, evaluator: {self.__str__()}")
        data_copy = data.copy()
        
        model_relevance_concept_ratings, model_relevance_instruction_ratings, model_fluency_ratings, \
            model_relevance_concept_completions, model_relevance_instruction_completions, model_fluency_completions, dataset_names = \
            self._get_all_ratings_from_data(data_copy, self.model_name)
        
        all_relevance_concept_ratings = []
        all_relevance_instruction_ratings = []
        all_fluency_ratings = []
        all_aggregated_ratings = []

        for i in range(len(model_relevance_concept_ratings)):
            all_relevance_concept_ratings += [model_relevance_concept_ratings[i][-1]]
            all_relevance_instruction_ratings += [model_relevance_instruction_ratings[i][-1]]
            all_fluency_ratings += [model_fluency_ratings[i][-1]]

            if dataset_names[i] == "AlpacaEvalSuppress":
                model_scores = [
                    2-model_relevance_concept_ratings[i][-1],
                    model_relevance_instruction_ratings[i][-1],
                    model_fluency_ratings[i][-1]
                ]
            else:
                model_scores = [
                    model_relevance_concept_ratings[i][-1],
                    model_relevance_instruction_ratings[i][-1],
                    model_fluency_ratings[i][-1]
                ]
                
            model_score = self._harmonic_mean(model_scores)
            all_aggregated_ratings += [model_score]

        metrics = {
            "lm_judge_rating": [],
            "relevance_concept_ratings": [],
            "relevance_instruction_ratings": [],
            "fluency_ratings": [],
            "factor": [],
            "raw_relevance_concept_ratings": all_relevance_concept_ratings,
            "raw_relevance_instruction_ratings": all_relevance_instruction_ratings,
            "raw_fluency_ratings": all_fluency_ratings,
            "raw_aggregated_ratings": all_aggregated_ratings,
            "relevance_concept_completions": model_relevance_concept_completions,
            "relevance_instruction_completions": model_relevance_instruction_completions,
            "fluency_completions": model_fluency_completions
        }
        data_copy[f"{self.model_name}_lm_judge_rating"] = all_aggregated_ratings
        data_copy[f"{self.model_name}_relevance_concept_ratings"] = all_relevance_concept_ratings
        data_copy[f"{self.model_name}_relevance_instruction_ratings"] = all_relevance_instruction_ratings
        data_copy[f"{self.model_name}_fluency_ratings"] = all_fluency_ratings

        # group by factor only and compute means
        grouped = data_copy.groupby("factor")
        for factor, group in grouped:
            metrics["lm_judge_rating"].append(group[f"{self.model_name}_lm_judge_rating"].mean())
            metrics["relevance_concept_ratings"].append(group[f"{self.model_name}_relevance_concept_ratings"].mean())
            metrics["relevance_instruction_ratings"].append(group[f"{self.model_name}_relevance_instruction_ratings"].mean())
            metrics["fluency_ratings"].append(group[f"{self.model_name}_fluency_ratings"].mean())
            metrics["factor"].append(factor)

        return metrics
