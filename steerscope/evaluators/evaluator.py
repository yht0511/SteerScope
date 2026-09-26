from abc import ABC, abstractmethod
from collections.abc import Mapping
import logging
import os
from pathlib import Path
import tempfile

import pandas as pd

from steerscope.utils.concept_scope import select_concept_ids
import numpy as np
from tqdm.auto import tqdm

from steerscope.evaluation.config import EvaluatorNode
from steerscope.evaluation.context import EvaluationContext
from steerscope.evaluation.result import EvaluationResult
from steerscope.evaluation.version import evaluator_source_fingerprint


logger = logging.getLogger(__name__)


class Evaluator(ABC):
    """Own a complete evaluation: dataset, inference, scoring and aggregation."""

    requires_inference = True
    compute_perplexity = False

    def __init__(
        self,
        node: EvaluatorNode,
        context: EvaluationContext,
        **params,
    ):
        self.node = node
        self.context = context
        self.node_id = node.node_id
        self.node_config = node.as_dict()
        self.args = context.args
        self.results = context.results
        self.progress = context.progress
        self.output_dir = Path(context.output_dir)
        # Explicit kwargs override the canonical node parameters.
        self.params = {**dict(node.params), **params}
        self.model_name = None
        self.target = None
        self.concept_id = None
        self._selected_models_by_target = {}
        self._model_progress_bar = None
        self._concept_scope = None

    def evaluate(self, models, concepts) -> EvaluationResult:
        """Construct this evaluator's data and invoke each target-bound model."""
        if not self.node.requires_inference:
            raise NotImplementedError(
                f"Result-only evaluator '{self.__class__.__name__}' must own "
                "its dependency queries by implementing evaluate()."
            )
        if not models:
            raise ValueError(f"Evaluator '{self.node_id}' received no model wrappers.")

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
        self._open_model_progress(target_groups)
        inference_frames = []
        sample_frames = []
        metric_frames = []
        try:
            self.open_resources(scheduled_models)
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
                    inference_frames.append(restored.inference)
                    sample_frames.append(restored.samples)
                    metric_frames.append(restored.metrics)
                    self._advance_model_progress(
                        self.model_evaluation_count(target_models),
                        description=f"{self.node_id} | cached {target_id}",
                    )
                    continue

                self.begin_target(target_id)
                target_result = self.evaluate_target(target_models)
                if not isinstance(target_result, EvaluationResult):
                    raise TypeError(
                        f"Evaluator '{self.node_id}' evaluate_target() must return "
                        "EvaluationResult."
                    )
                if self.progress is not None:
                    self.checkpoint_resources()
                    self.progress.save_target(target_id, target_result)
                inference_frames.append(target_result.inference)
                sample_frames.append(target_result.samples)
                metric_frames.append(target_result.metrics)
        finally:
            try:
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
        return EvaluationResult(
            inference=self._concat(inference_frames),
            samples=self._concat(sample_frames),
            metrics=self._concat(metric_frames),
            metadata=metadata,
        )

    def prepare_evaluation(self, models, concepts) -> None:
        """Validate or initialize evaluator-owned state for one invocation."""
        self._prepare_concept_scope(concepts)
        self._validate_dependency_concept_scopes()

    def evaluation_models(self, models):
        """Return the model wrappers that define target scheduling."""
        if self._concept_scope is None:
            concepts = self._unique_model_concepts(models)
            self._prepare_concept_scope(concepts)
        selected_ids = set(self._concept_scope["concept_ids"])
        return [
            model
            for model in models
            if int(model.concept.concept_id) in selected_ids
        ]

    def evaluate_target(self, target_models) -> EvaluationResult:
        """Build data, run inference, and score one target."""
        selected_models = self._selected_models(target_models)
        representative = selected_models[0]
        factors = [model.factor for model in selected_models]
        examples = self.build_dataset(representative, factors)
        generated = []
        for model in selected_models:
            model_examples = self.examples_for_model(examples, model)
            self._set_current_model(model)
            generated.append(model.generate(model_examples))
            self._advance_model_progress()
        inference = self._concat(generated)
        inference = self.prepare_examples(
            inference,
            results=self.results,
            config=self.node_config,
        )
        if inference.empty:
            raise ValueError(
                f"Evaluator '{self.node_id}' produced no examples for "
                f"target '{representative.target.target_id}'."
            )
        metrics, samples = self.score(representative, inference)
        return EvaluationResult(
            inference=inference,
            samples=samples,
            metrics=metrics,
            metadata=self.target_metadata(),
        )

    def _selected_models(self, target_models):
        target_id = target_models[0].target.target_id
        selected = self._selected_models_by_target.get(target_id)
        if selected is None:
            selected = self.select_models(target_models, self.results)
            self._selected_models_by_target[target_id] = selected
        return selected

    def model_evaluation_count(self, target_models) -> int:
        """Number of model.generate calls represented by one target."""
        return len(self._selected_models(target_models))

    def _open_model_progress(self, target_groups) -> None:
        total = sum(self.model_evaluation_count(group) for group in target_groups)
        self._model_progress_bar = tqdm(
            total=total,
            desc=f"{self.node_id} | model evaluations",
            unit="model",
            position=1,
            dynamic_ncols=True,
            leave=True,
        )

    def _set_current_model(self, model, role=None) -> None:
        if self._model_progress_bar is None:
            return
        concept_id = model.concept.concept_id
        role_text = f" | {role}" if role else ""
        self._model_progress_bar.set_description(
            f"{self.node_id} | {model.method} | concept={concept_id} "
            f"| factor={model.factor:g}{role_text}"
        )

    def _advance_model_progress(self, count=1, description=None) -> None:
        if self._model_progress_bar is None:
            return
        if description is not None:
            self._model_progress_bar.set_description(description)
        self._model_progress_bar.update(int(count))

    def _close_model_progress(self) -> None:
        if self._model_progress_bar is not None:
            self._model_progress_bar.close()
            self._model_progress_bar = None
        self._selected_models_by_target = {}

    def examples_for_model(self, examples, model) -> pd.DataFrame:
        """Select the rows owned by one factor-bound model wrapper."""
        factor_column = "model_factor" if "model_factor" in examples else "factor"
        selected = examples[examples[factor_column] == model.factor]
        if selected.empty:
            raise ValueError(
                f"Evaluator '{self.node_id}' dataset has no rows for factor "
                f"{model.factor} and target '{model.target.target_id}'."
            )
        return selected.reset_index(drop=True)

    @classmethod
    def execution_context(cls, node, args):
        """Return evaluator-owned inputs that must participate in cache keys."""
        return {"source_fingerprint": evaluator_source_fingerprint(cls)}

    def open_resources(self, models) -> None:
        """Open resources owned by this evaluator invocation."""

    def checkpoint_resources(self) -> None:
        """Persist evaluator resources after one target completes."""

    def close_resources(self) -> None:
        """Close resources owned by this evaluator invocation."""

    def begin_target(self, target_id: str) -> None:
        """Start evaluator-owned accounting for one target."""

    def target_metadata(self) -> dict:
        """Return metadata accumulated for the current target."""
        return self._concept_scope_metadata()

    def evaluation_metadata(self) -> dict:
        """Return metadata accumulated across this evaluator invocation."""
        return self._concept_scope_metadata()

    def combine_metadata(self, target_metadata) -> dict:
        """Combine evaluator-owned metadata restored from target checkpoints."""
        metadata = tuple(target_metadata)
        scopes = [
            item.get("concept_scope")
            for item in metadata
            if item.get("concept_scope") is not None
        ]
        if not scopes:
            return self._concept_scope_metadata()
        reference = scopes[0]
        if any(scope != reference for scope in scopes[1:]):
            raise ValueError(
                f"Evaluator '{self.node_id}' restored inconsistent concept scopes."
            )
        return {"concept_scope": reference}

    def filter_concept_rows(self, examples, concepts) -> pd.DataFrame:
        """Restrict a dependency table to this evaluator's concept scope."""
        if not self.node.concepts:
            return examples.copy()
        if self._concept_scope is None:
            self._prepare_concept_scope(concepts)
        if "concept_id" not in examples.columns:
            raise KeyError(
                f"Evaluator '{self.node_id}' cannot apply a concept filter to "
                "dependency rows without a 'concept_id' column."
            )
        concept_ids = pd.to_numeric(examples["concept_id"], errors="coerce")
        selected_ids = set(self._concept_scope["concept_ids"])
        filtered = examples[concept_ids.isin(selected_ids)].copy()
        if filtered.empty:
            raise ValueError(
                f"Evaluator '{self.node_id}' concept filter selected no "
                "dependency rows."
            )
        return filtered

    def _prepare_concept_scope(self, concepts) -> None:
        concepts = list(concepts)
        if not concepts:
            if not self.node.concepts:
                self._concept_scope = None
                return
            raise ValueError(
                f"Evaluator '{self.node_id}' received no concepts."
            )
        configured = dict(self.node.concepts)
        unsupported = sorted(
            set(configured).difference({"genres", "ids", "count", "seed"})
        )
        if unsupported:
            raise ValueError(
                f"Evaluator '{self.node_id}' has unsupported concept filters: "
                f"{unsupported}."
            )
        configured_genres = configured.get("genres")
        if configured_genres is None:
            required_genres = None
        else:
            if isinstance(configured_genres, str):
                configured_genres = [configured_genres]
            if not isinstance(configured_genres, (list, tuple, set, frozenset)):
                raise TypeError(
                    f"Evaluator '{self.node_id}' concepts.genres must be a list."
                )
            required_genres = {
                str(genre).strip().lower()
                for genre in configured_genres
                if str(genre).strip()
            }
            if not required_genres:
                raise ValueError(
                    f"Evaluator '{self.node_id}' concepts.genres cannot be empty."
                )

        selected_ids = []
        for concept in concepts:
            concept_id = int(concept.concept_id)
            if required_genres is None:
                selected_ids.append(concept_id)
                continue
            if required_genres.intersection(self._concept_genres(concept)):
                selected_ids.append(concept_id)
        selected_ids = sorted(set(selected_ids))

        configured_ids = configured.get("ids")
        if configured_ids is not None:
            if isinstance(configured_ids, (str, bytes)):
                raise TypeError(
                    f"Evaluator '{self.node_id}' concepts.ids must be a list "
                    "of integer concept IDs."
                )
            if isinstance(configured_ids, int):
                configured_ids = [configured_ids]
            if not isinstance(configured_ids, (list, tuple, set, frozenset)):
                raise TypeError(
                    f"Evaluator '{self.node_id}' concepts.ids must be a list "
                    "of integer concept IDs."
                )
            requested_ids = []
            for value in configured_ids:
                if isinstance(value, bool):
                    raise TypeError(
                        f"Evaluator '{self.node_id}' concepts.ids contains a "
                        "boolean instead of an integer concept ID."
                    )
                try:
                    requested_ids.append(int(value))
                except (TypeError, ValueError) as error:
                    raise TypeError(
                        f"Evaluator '{self.node_id}' concepts.ids contains "
                        f"invalid concept ID {value!r}."
                    ) from error
            requested_ids = sorted(set(requested_ids))
            if not requested_ids:
                raise ValueError(
                    f"Evaluator '{self.node_id}' concepts.ids cannot be empty."
                )
            available_ids = set(selected_ids)
            missing_ids = sorted(set(requested_ids).difference(available_ids))
            if missing_ids:
                raise ValueError(
                    f"Evaluator '{self.node_id}' concepts.ids requests "
                    f"unavailable concept IDs {missing_ids}."
                )
            selected_ids = requested_ids

        configured_count = configured.get("count")
        configured_seed = configured.get("seed")
        if configured_count is None:
            if configured_seed is not None:
                raise ValueError(
                    f"Evaluator '{self.node_id}' concepts.seed requires "
                    "concepts.count."
                )
        else:
            if isinstance(configured_count, bool):
                raise TypeError(
                    f"Evaluator '{self.node_id}' concepts.count must be a "
                    "positive integer."
                )
            try:
                configured_count = int(configured_count)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"Evaluator '{self.node_id}' concepts.count must be a "
                    "positive integer."
                ) from error
            if configured_count < 1:
                raise ValueError(
                    f"Evaluator '{self.node_id}' concepts.count must be at "
                    "least 1."
                )
            if configured_count > len(selected_ids):
                raise ValueError(
                    f"Evaluator '{self.node_id}' requested {configured_count} "
                    f"concepts, but only {len(selected_ids)} are available after "
                    "applying its other concept filters."
                )
            seed = 42 if configured_seed is None else int(configured_seed)
            selected_ids = select_concept_ids(
                selected_ids,
                count=configured_count,
                seed=seed,
            )
        if not selected_ids:
            description = sorted(required_genres) if required_genres else "all"
            raise ValueError(
                f"Evaluator '{self.node_id}' selected no concepts for genres "
                f"{description}."
            )
        self._concept_scope = {
            "genres": (
                sorted(required_genres) if required_genres is not None else None
            ),
            "concept_ids": selected_ids,
        }

    @staticmethod
    def _concept_genres(concept) -> set[str]:
        mapping = concept.metadata.get("concept_genres_map")
        if not isinstance(mapping, Mapping):
            raise ValueError(
                f"Concept {concept.concept_id} is missing concept_genres_map "
                "metadata."
            )
        genres = mapping.get(concept.text)
        if genres is None:
            raise ValueError(
                f"Concept {concept.concept_id} has no genre entry for "
                f"{concept.text!r}."
            )
        if isinstance(genres, str):
            genres = [genres]
        if not isinstance(genres, (list, tuple, set, frozenset)):
            raise TypeError(
                f"Concept {concept.concept_id} genres must be a list."
            )
        normalized = {
            str(genre).strip().lower()
            for genre in genres
            if str(genre).strip()
        }
        if not normalized:
            raise ValueError(f"Concept {concept.concept_id} has no genres.")
        return normalized

    @staticmethod
    def _unique_model_concepts(models):
        concepts = {}
        for model in models:
            concepts[int(model.concept.concept_id)] = model.concept
        return list(concepts.values())

    def _concept_scope_metadata(self) -> dict:
        if self._concept_scope is None:
            return {}
        return {
            "concept_scope": {
                "genres": self._concept_scope["genres"],
                "concept_ids": list(self._concept_scope["concept_ids"]),
            }
        }

    def _validate_dependency_concept_scopes(self) -> None:
        if self.results is None or self._concept_scope is None:
            return
        expected_ids = set(self._concept_scope["concept_ids"])
        for dependency in self.node.depends_on:
            manifest = self.results.manifest(dependency)
            if not isinstance(manifest, Mapping):
                continue
            scope = (manifest.get("metadata") or {}).get("concept_scope")
            if scope is None:
                continue
            dependency_ids = {int(value) for value in scope.get("concept_ids", ())}
            missing_ids = expected_ids.difference(dependency_ids)
            if missing_ids:
                raise ValueError(
                    f"Evaluator '{self.node_id}' concept scope "
                    f"requests concept IDs {sorted(missing_ids)} that are absent "
                    f"from dependency '{dependency}' scope "
                    f"{sorted(dependency_ids)}."
                )

    def build_dataset(self, model, factors) -> pd.DataFrame:
        raise NotImplementedError(
            f"Inference evaluator '{self.__class__.__name__}' must implement "
            "build_dataset()."
        )

    def select_models(self, models, results):
        config = self.node_config.get("inference", {})
        selection = config.get("select")
        if not selection:
            return list(models)

        model = models[0]
        metric = selection.get("metric")
        if not metric:
            raise ValueError(
                f"Evaluator '{self.node_id}' inference.select must set 'metric'."
            )
        data = self._selection_rows(model, results, selection)
        factor_column = "model_factor" if "model_factor" in data else "factor"
        required = {factor_column, metric}
        missing = sorted(required.difference(data.columns))
        if missing:
            raise KeyError(f"Upstream results are missing selection columns: {missing}")
        scores = data.groupby(factor_column, sort=True)[metric].mean().dropna()
        if scores.empty:
            raise ValueError(
                f"No factor scores found for target '{model.target.target_id}'."
            )
        strategy = selection.get("strategy", "argmax")
        if strategy == "argmax":
            selected_factor = float(scores.idxmax())
        elif strategy == "argmin":
            selected_factor = float(scores.idxmin())
        else:
            raise ValueError(f"Unsupported factor selection strategy '{strategy}'.")

        included_by_model = selection.get("include_factors_by_model", {}) or {}
        if not isinstance(included_by_model, dict):
            raise TypeError(
                f"Evaluator '{self.node_id}' inference.select."
                "include_factors_by_model must be a mapping."
            )
        included = included_by_model.get(
            model.method,
            selection.get("include_factors", ()),
        ) or ()
        if isinstance(included, (int, float, str)):
            included = [included]
        requested_factors = list(dict.fromkeys([
            selected_factor,
            *(float(factor) for factor in included),
        ]))
        available_factors = {float(item.factor) for item in models}
        missing_factors = sorted(
            set(requested_factors).difference(available_factors)
        )
        if missing_factors:
            raise ValueError(
                f"Evaluator '{self.node_id}' was not given a model wrapper for "
                f"selected factor(s) {missing_factors} and target "
                f"'{model.target.target_id}'."
            )
        return [
            item for item in models
            if float(item.factor) in requested_factors
        ]

    def _selection_rows(self, model, results, selection) -> pd.DataFrame:
        data = self._selection_data(results, selection)
        identities = (
            ("target_id", model.target.target_id),
            ("method", model.method),
            ("concept_id", model.concept.concept_id),
        )
        matched_columns = []
        for column, value in identities:
            if column in data.columns:
                data = data[data[column] == value]
                matched_columns.append(column)
        if data.empty:
            identity = ", ".join(
                f"{column}={value!r}"
                for column, value in identities
                if column in matched_columns
            ) or "global selection"
            raise ValueError(
                f"No factor selection rows found for {identity}."
            )
        return data.copy()

    def _selection_data(self, results, selection) -> pd.DataFrame:
        source = selection.get("from")
        if source is None:
            dependencies = tuple(self.node_config.get("depends_on") or ())
            if len(dependencies) != 1:
                raise ValueError(
                    f"Evaluator '{self.node_id}' inference.select must set 'from'."
                )
            source = dependencies[0]
        if results is None:
            raise ValueError(
                f"Evaluator '{self.node_id}' cannot select a factor without results."
            )
        return results.query(
            source,
            kind=selection.get("kind", "metrics"),
            filters=selection.get("filters"),
        )

    @staticmethod
    def _models_by_target(models):
        grouped = {}
        for model in models:
            grouped.setdefault(model.target.target_id, []).append(model)
        for target_id, target_models in grouped.items():
            factors = [model.factor for model in target_models]
            if len(factors) != len(set(factors)):
                raise ValueError(
                    f"Target '{target_id}' has duplicate model wrappers for a factor."
                )
            yield target_models

    def _validate_concepts(self, models, concepts):
        available = {
            (concept.concept_id, concept.text)
            for concept in concepts
        }
        missing = sorted({
            (model.concept.concept_id, model.concept.text)
            for model in models
            if (model.concept.concept_id, model.concept.text) not in available
        })
        if missing:
            raise ValueError(
                f"Evaluator '{self.node_id}' received models whose concepts are "
                f"not in the concept collection: {missing}"
            )

    def score(self, model, inference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.model_name = model.method
        self.target = model.target
        self.concept_id = model.concept.concept_id
        result = self.compute_metrics(inference)
        samples = self._sample_rows(model, inference, result)
        metrics = self._metric_rows(model, result, inference=inference)
        return metrics, samples

    def prepare_examples(self, examples, results=None, config=None):
        return examples

    def render_report(
        self,
        result: EvaluationResult,
        output_dir: str | Path | None = None,
    ) -> list[Path]:
        """Render evaluator-specific artifacts from persisted evaluation results."""
        return []

    def _report_directory(self, output_dir: str | Path | None = None) -> Path:
        report_dir = Path(output_dir or self.output_dir) / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir

    def _report_formats(self) -> tuple[str, ...]:
        report_config = self.node_config.get("report", {})
        configured = report_config.get(
            "formats", getattr(self.args, "report_formats", ("png", "pdf"))
        )
        if configured is None:
            raise ValueError("Report formats must be a string or a list of strings.")
        if isinstance(configured, str):
            configured = configured.split(",")
        formats = tuple(
            dict.fromkeys(
                str(value).strip().lower().lstrip(".")
                for value in configured
                if str(value).strip()
            )
        )
        supported = {"png", "pdf", "svg"}
        unsupported = sorted(set(formats).difference(supported))
        if unsupported:
            raise ValueError(
                f"Unsupported report formats {unsupported}; choose from {sorted(supported)}."
            )
        return formats

    def _save_report_summary(
        self,
        summary: pd.DataFrame,
        output_dir: str | Path | None = None,
    ) -> Path:
        report_dir = self._report_directory(output_dir)
        path = report_dir / "summary.parquet"
        with tempfile.NamedTemporaryFile(
            delete=False, dir=report_dir, suffix=".parquet.tmp"
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            summary.to_parquet(temporary_path, index=False)
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return path

    def _save_report_figure(
        self,
        figure,
        output_dir: str | Path | None = None,
        stem: str = "metrics",
    ) -> list[Path]:
        report_dir = self._report_directory(output_dir)
        report_config = self.node_config.get("report", {})
        dpi = int(report_config.get("dpi", getattr(self.args, "report_dpi", 300)))
        if dpi < 1:
            raise ValueError("Report DPI must be at least 1.")
        paths = []
        try:
            for file_format in self._report_formats():
                path = report_dir / f"{stem}.{file_format}"
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    dir=report_dir,
                    suffix=f".{file_format}.tmp",
                ) as temporary:
                    temporary_path = Path(temporary.name)
                try:
                    figure.savefig(
                        temporary_path,
                        format=file_format,
                        dpi=dpi,
                        bbox_inches="tight",
                    )
                    os.replace(temporary_path, path)
                finally:
                    if temporary_path.exists():
                        temporary_path.unlink()
                paths.append(path)
        finally:
            self._pyplot().close(figure)
        return paths

    @staticmethod
    def _pyplot():
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt

        return plt

    @staticmethod
    def _curve_summary(
        metrics: pd.DataFrame,
        value_columns,
        factor_column: str = "factor",
    ) -> pd.DataFrame:
        if metrics is None or metrics.empty:
            raise ValueError("Cannot render a report without metric rows.")
        value_columns = tuple(value_columns)
        required = {factor_column, *value_columns}
        missing = sorted(required.difference(metrics.columns))
        if missing:
            raise KeyError(f"Report metrics are missing columns: {missing}")

        id_columns = [
            column for column in ("method", factor_column) if column in metrics.columns
        ]
        values = metrics[id_columns + list(value_columns)].melt(
            id_vars=id_columns,
            value_vars=list(value_columns),
            var_name="metric",
            value_name="value",
        )
        values["value"] = pd.to_numeric(values["value"], errors="coerce")
        values = values.dropna(subset=["value"])
        if values.empty:
            raise ValueError("Report metrics contain no numeric values.")
        values[factor_column] = pd.to_numeric(
            values[factor_column], errors="coerce"
        )
        if values[factor_column].isna().any():
            raise ValueError(
                f"Report factor column '{factor_column}' contains non-numeric values."
            )
        finite_values = values["value"].map(
            lambda value: float("-inf") < value < float("inf")
        )
        finite_factors = values[factor_column].map(
            lambda value: float("-inf") < value < float("inf")
        )
        if not finite_values.all() or not finite_factors.all():
            raise ValueError("Report metrics and factors must contain finite values.")

        group_columns = [*id_columns, "metric"]
        summary = (
            values.groupby(group_columns, dropna=False, sort=True)["value"]
            .agg(mean="mean", std="std", count="count")
            .reset_index()
        )
        margin = 1.96 * summary["std"].fillna(0.0) / summary["count"].pow(0.5)
        summary["ci_lower"] = summary["mean"] - margin
        summary["ci_upper"] = summary["mean"] + margin
        return summary

    def _render_curve_report(
        self,
        result: EvaluationResult,
        metric_labels,
        output_dir: str | Path | None = None,
        *,
        factor_column: str = "factor",
        columns: int = 2,
        y_limits: tuple[float, float] | None = None,
        metric_y_limits: dict[str, tuple[float, float]] | None = None,
        y_scale: str = "linear",
        y_axis_label: str = "Score",
    ) -> list[Path]:
        summary = self._curve_summary(
            result.metrics,
            metric_labels.keys(),
            factor_column=factor_column,
        )
        positive_floor = None
        if y_scale == "log":
            positive_means = summary.loc[summary["mean"] > 0.0, "mean"]
            if len(positive_means) != len(summary):
                raise ValueError("Log-scale reports require strictly positive values.")
            positive_floor = float(positive_means.min()) * 0.1
            if y_limits is not None and y_limits[0] <= 0.0:
                raise ValueError("Log-scale report limits must be strictly positive.")
        summary_path = self._save_report_summary(summary, output_dir)

        plt = self._pyplot()
        metric_count = len(metric_labels)
        columns = min(columns, metric_count)
        rows = (metric_count + columns - 1) // columns
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(6.0 * columns, 4.0 * rows),
            squeeze=False,
            sharex=True,
        )
        methods = (
            summary["method"].drop_duplicates().tolist()
            if "method" in summary
            else [None]
        )
        method_styles = self._method_plot_styles(methods, plt)
        for axis, (metric, label) in zip(axes.flat, metric_labels.items()):
            metric_data = summary[summary["metric"] == metric]
            for method in methods:
                curve = metric_data
                if method is not None:
                    curve = curve[
                        curve["method"].isna()
                        if pd.isna(method)
                        else curve["method"] == method
                    ]
                curve = curve.sort_values(factor_column)
                if curve.empty:
                    continue
                x = curve[factor_column].astype(float).to_numpy()
                mean = curve["mean"].astype(float).to_numpy()
                lower = curve["ci_lower"].astype(float).to_numpy()
                upper = curve["ci_upper"].astype(float).to_numpy()
                if positive_floor is not None:
                    lower = lower.clip(min=positive_floor)
                    upper = upper.clip(min=positive_floor)
                style = method_styles[self._method_plot_key(method)]
                line = axis.plot(
                    x,
                    mean,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=1.8,
                    markersize=3.5,
                    label=(
                        "Unknown"
                        if method is not None and pd.isna(method)
                        else str(method) if method is not None else None
                    ),
                )[0]
                axis.fill_between(
                    x,
                    lower,
                    upper,
                    color=line.get_color(),
                    alpha=0.16,
                    linewidth=0,
                )
            axis.set_title(label)
            axis.set_xlabel("Steering factor")
            axis.set_ylabel(y_axis_label)
            axis.set_yscale(y_scale)
            limits = (
                metric_y_limits.get(metric, y_limits)
                if metric_y_limits is not None
                else y_limits
            )
            if limits is not None:
                axis.set_ylim(*limits)
            axis.grid(alpha=0.25)
        for axis in axes.flat[metric_count:]:
            axis.set_visible(False)
        if methods != [None]:
            handles, labels = axes.flat[0].get_legend_handles_labels()
            if handles:
                figure.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.945),
                    ncol=min(6, len(labels)),
                    frameon=False,
                )
                figure.subplots_adjust(top=0.86)
        figure.suptitle(f"{self.node_id} evaluation", fontsize=14, y=0.99)
        return [summary_path, *self._save_report_figure(figure, output_dir)]

    @staticmethod
    def _method_plot_key(method):
        return "__unknown__" if method is not None and pd.isna(method) else str(method)

    @classmethod
    def _method_plot_styles(cls, methods, plt):
        """Assign every displayed method a distinct color and redundant line style."""
        keys = sorted({cls._method_plot_key(method) for method in methods})
        if not keys:
            return {}
        positions = np.linspace(0.03, 0.97, len(keys)) if len(keys) > 1 else [0.5]
        colors = plt.get_cmap("turbo")(positions)
        markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*", "p")
        linestyles = ("-", "--", "-.", ":")
        return {
            key: {
                "color": tuple(float(channel) for channel in colors[index]),
                "marker": markers[index % len(markers)],
                "linestyle": linestyles[(index // len(markers)) % len(linestyles)],
            }
            for index, key in enumerate(keys)
        }

    def _sample_rows(self, model, inference, result):
        samples = inference.copy().reset_index(drop=True)
        samples["evaluator_id"] = self.node_id
        samples["evaluator_type"] = self.__class__.__name__
        for name, values in result.items():
            if isinstance(values, (list, tuple)) and len(values) == len(samples):
                samples[name] = list(values)
        return samples

    def _metric_rows(self, model, result, inference=None):
        aggregate = {
            key: value
            for key, value in result.items()
            if not key.startswith("raw_") and "completions" not in key
        }
        lengths = [len(value) for value in aggregate.values() if isinstance(value, (list, tuple))]
        count = max(lengths, default=1)
        rows = []
        for index in range(count):
            row = {
                "evaluator_id": self.node_id,
                "evaluator_type": self.__class__.__name__,
                "target_id": model.target.target_id,
                "method": model.method,
                "concept_id": model.concept.concept_id,
            }
            for name, value in aggregate.items():
                row[name] = (
                    value[index] if isinstance(value, (list, tuple)) and index < len(value)
                    else value if not isinstance(value, (list, tuple))
                    else None
                )
            if inference is not None and "model_factor" in inference:
                if "factor" in row:
                    matching = inference[inference["factor"] == row["factor"]]
                else:
                    matching = inference
                model_factors = matching["model_factor"].dropna().unique()
                if len(model_factors) == 1:
                    row["model_factor"] = float(model_factors[0])
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _concat(frames):
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def fit(self, examples):
        pass

    @abstractmethod
    def compute_metrics(self, examples):
        pass
