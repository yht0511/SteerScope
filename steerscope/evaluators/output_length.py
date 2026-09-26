"""Result-only evaluator for token lengths of upstream generations."""

import math

import pandas as pd
from transformers import AutoTokenizer

from steerscope.evaluation.result import EvaluationResult

from .evaluator import Evaluator


class OutputLengthEvaluator(Evaluator):
    """Measure completion length without running model inference again."""

    requires_inference = False

    def evaluate(self, models, concepts):
        if self.results is None:
            raise ValueError(
                f"Evaluator '{self.node_id}' requires an upstream result view."
            )
        self.prepare_evaluation(models, concepts)
        input_config = dict(self.node.input)
        source = input_config.get("from")
        if source is None:
            if len(self.node.depends_on) != 1:
                raise ValueError(
                    f"Evaluator '{self.node_id}' input.from is required with "
                    "zero or multiple dependencies."
                )
            source = self.node.depends_on[0]
        examples = self.results.query(
            source,
            kind=input_config.get("kind", "inference"),
            filters=input_config.get("filters"),
        )
        examples = self.filter_concept_rows(examples, concepts)
        base_model = getattr(self.args, "steering_model_name", None) or getattr(
            self.args, "model_name", None
        )
        if not base_model:
            raise ValueError(
                f"Evaluator '{self.node_id}' requires a base model name."
            )
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
        metrics, samples = self._measure(examples, tokenizer, str(base_model))
        metrics.insert(0, "evaluator_type", self.__class__.__name__)
        metrics.insert(0, "evaluator_id", self.node_id)
        return EvaluationResult(
            samples=samples,
            metrics=metrics,
            metadata=self.evaluation_metadata(),
        )

    def _measure(self, examples, tokenizer, base_model):
        required = {"method", "concept_id", "factor"}
        missing = sorted(required.difference(examples.columns))
        if missing:
            raise KeyError(
                f"OutputLengthEvaluator input is missing columns: {missing}"
            )
        if examples.empty:
            raise ValueError("OutputLengthEvaluator received no upstream rows.")

        samples = examples.copy().reset_index(drop=True)
        lengths = pd.Series(index=samples.index, dtype="int64")
        batch_size = int(self.params.get("tokenize_batch_size", 512))
        if batch_size < 1:
            raise ValueError(
                "OutputLengthEvaluator tokenize_batch_size must be at least 1."
            )
        for method, indices in samples.groupby("method", sort=False).groups.items():
            column = f"{method}_steered_generation"
            if column not in samples:
                raise KeyError(
                    f"OutputLengthEvaluator cannot find generation column "
                    f"'{column}' for method '{method}'."
                )
            generations = samples.loc[indices, column]
            if generations.isna().any():
                raise ValueError(
                    f"OutputLengthEvaluator found null generations for "
                    f"method '{method}'."
                )
            method_lengths = []
            texts = generations.astype(str).tolist()
            for start in range(0, len(texts), batch_size):
                encoded = tokenizer(
                    texts[start:start + batch_size],
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                )["input_ids"]
                method_lengths.extend(len(token_ids) for token_ids in encoded)
            lengths.loc[indices] = method_lengths

        samples["base_model"] = base_model
        samples["output_tokens"] = lengths.astype(int)
        group_columns = [
            column
            for column in (
                "base_model", "target_id", "method", "concept_id",
                "input_concept", "factor", "model_factor",
            )
            if column in samples.columns
        ]
        metrics = (
            samples.groupby(group_columns, dropna=False, sort=True)["output_tokens"]
            .agg(
                mean_output_tokens="mean",
                std_output_tokens="std",
                min_output_tokens="min",
                max_output_tokens="max",
                output_length_num_examples="count",
            )
            .reset_index()
        )
        metrics["std_output_tokens"] = metrics["std_output_tokens"].fillna(0.0)
        return metrics, samples

    def render_report(self, result, output_dir=None):
        return self._render_curve_report(
            result,
            {"mean_output_tokens": "Mean completion length"},
            output_dir,
            columns=1,
            y_limits=(0.0, self._upper_plot_limit(result.metrics)),
            y_axis_label="Mean completion tokens",
        )

    @staticmethod
    def _upper_plot_limit(metrics):
        if metrics is None or metrics.empty or "mean_output_tokens" not in metrics:
            raise ValueError("OutputLengthEvaluator has no token lengths to report.")
        values = pd.to_numeric(metrics["mean_output_tokens"], errors="coerce")
        values = values[values.map(lambda value: pd.notna(value) and math.isfinite(value))]
        if values.empty:
            raise ValueError("OutputLengthEvaluator has no finite token lengths.")
        maximum = float(values.max())
        return max(1.0, maximum * 1.1)

    def compute_metrics(self, examples):
        raise NotImplementedError(
            "OutputLengthEvaluator computes token counts in evaluate()."
        )

    def __str__(self):
        return "OutputLengthEvaluator"
