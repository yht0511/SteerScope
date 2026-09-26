from .alpaca import AlpacaEvaluator
from .judge import JudgeEvaluatorMixin
from .prompt_templates import (
    UNIDIRECTIONAL_PAIRWISE_EVALUATION_CONCEPT_RELEVANCE_TEMPLATE,
    UNIDIRECTIONAL_PAIRWISE_EVALUATION_FLUENCY_TEMPLATE,
    UNIDIRECTIONAL_PAIRWISE_EVALUATION_INSTRUCTION_RELEVANCE_TEMPLATE,
)
from collections import Counter

from steerscope.evaluation import EvaluationResult
import pandas as pd

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


class WinRateEvaluator(JudgeEvaluatorMixin, AlpacaEvaluator):
    source_dependencies = ("evaluators/prompt_templates.py",)
    DEFAULT_RATING = 0.0
    def __init__(self, node, context, **params):
        super().__init__(node, context, **params)
        self.winrate_baseline = params.get("baseline", "PromptSteering")

    def __str__(self):
        return 'WinRateEvaluator'

    def render_report(self, result, output_dir=None):
        metrics = result.metrics
        if metrics is None or metrics.empty:
            raise ValueError("WinRateEvaluator has no metric rows to report.")
        factor_column = "model_factor" if "model_factor" in metrics else "factor"
        labels = {
            "win_rate": "Win",
            "loss_rate": "Loss",
            "tie_rate": "Tie",
        }
        summary = self._curve_summary(metrics, labels, factor_column=factor_column)
        summary_path = self._save_report_summary(summary, output_dir)

        plt = self._pyplot()
        methods = (
            summary["method"].drop_duplicates().tolist()
            if "method" in summary
            else [None]
        )
        columns = min(2, len(methods))
        rows = (len(methods) + columns - 1) // columns
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(6.0 * columns, 4.0 * rows),
            squeeze=False,
            sharey=True,
        )
        colors = {"win_rate": "#2ca02c", "loss_rate": "#d62728", "tie_rate": "#7f7f7f"}
        for axis, method in zip(axes.flat, methods):
            method_data = summary
            if method is not None:
                method_data = method_data[
                    method_data["method"].isna()
                    if pd.isna(method)
                    else method_data["method"] == method
                ]
            factors = sorted(method_data[factor_column].astype(float).unique())
            bottom = [0.0] * len(factors)
            for metric, label in labels.items():
                values = (
                    method_data[method_data["metric"] == metric]
                    .set_index(factor_column)["mean"]
                    .reindex(factors, fill_value=0.0)
                    .astype(float)
                    .tolist()
                )
                axis.bar(
                    range(len(factors)),
                    values,
                    bottom=bottom,
                    label=label,
                    color=colors[metric],
                )
                bottom = [left + value for left, value in zip(bottom, values)]
            axis.set_title(
                "Unknown"
                if method is not None and pd.isna(method)
                else str(method) if method is not None else self.node_id
            )
            axis.set_xticks(
                range(len(factors)),
                [f"{factor:g}" for factor in factors],
                rotation=45,
            )
            axis.set_xlabel("Steering factor")
            axis.set_ylabel("Rate")
            axis.set_ylim(0.0, 1.0)
            axis.grid(axis="y", alpha=0.25)
        for axis in axes.flat[len(methods):]:
            axis.set_visible(False)
        handles, legend_labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=3,
            frameon=False,
        )
        figure.suptitle(f"{self.node_id} evaluation", fontsize=14, y=0.99)
        figure.subplots_adjust(top=0.84)
        return [summary_path, *self._save_report_figure(figure, output_dir)]

    def prepare_evaluation(self, models, concepts) -> None:
        """Resolve the evaluator-owned baseline before target scheduling."""
        super().prepare_evaluation(models, concepts)
        baseline_name = self.params.get("baseline")
        if baseline_name is None:
            baseline_name = (
                getattr(self.args, "winrate_baseline", None)
                or self.winrate_baseline
            )
        baseline_models = [model for model in models if model.method == baseline_name]
        baselines_by_concept = {}
        for model in baseline_models:
            baselines_by_concept.setdefault(model.concept.concept_id, []).append(model)
        candidates = [model for model in models if model.method != baseline_name]
        if not baselines_by_concept:
            raise ValueError(
                f"WinRateEvaluator requires baseline model '{baseline_name}' in its models."
            )
        if not candidates:
            raise ValueError("WinRateEvaluator requires at least one non-baseline model.")
        self.winrate_baseline = baseline_name
        self._baselines_by_concept = baselines_by_concept
        self._candidate_models = candidates

    def evaluation_models(self, models):
        return self._candidate_models

    def model_evaluation_count(self, target_models):
        # Each selected candidate also invokes one factor-matched baseline.
        return 2 * len(self._selected_models(target_models))

    def evaluate_target(self, target_models) -> EvaluationResult:
        """Generate candidate and baseline responses for one candidate target."""
        target_inference = []
        target_samples = []
        target_metrics = []
        selected = self._selected_models(target_models)
        representative = selected[0]
        factors = [model.factor for model in selected]
        examples = self.build_dataset(representative, factors)
        for candidate in selected:
            baseline = self._select_baseline(
                self._baselines_by_concept.get(candidate.concept.concept_id, ()),
                candidate.factor,
            )
            if baseline is None:
                raise ValueError(
                    f"No '{self.winrate_baseline}' wrapper exists for concept "
                    f"{candidate.concept.concept_id}."
                )
            model_examples = self.examples_for_model(examples, candidate)
            self._set_current_model(candidate, role="candidate")
            candidate_data = candidate.generate(model_examples)
            self._advance_model_progress()
            baseline_examples = model_examples.copy()
            if "model_factor" in baseline_examples:
                baseline_examples["model_factor"] = baseline.factor
            else:
                baseline_examples["factor"] = baseline.factor
            self._set_current_model(baseline, role="baseline")
            baseline_data = baseline.generate(baseline_examples)
            self._advance_model_progress()
            baseline_column = f"{self.winrate_baseline}_steered_generation"
            if baseline_column not in baseline_data:
                raise KeyError(
                    f"Win-rate baseline inference did not produce "
                    f"'{baseline_column}'."
                )
            if len(candidate_data) != len(baseline_data):
                raise ValueError(
                    "Win-rate candidate and baseline inference returned different "
                    "row counts."
                )
            candidate_data[baseline_column] = baseline_data[baseline_column].tolist()
            self.model_name = candidate.method
            result = self.compute_metrics(candidate_data)
            target_samples.append(
                self._sample_rows(candidate, candidate_data, result)
            )
            target_metrics.append(
                self._metric_rows(candidate, result, inference=candidate_data)
            )
            target_inference.append(candidate_data)
        return EvaluationResult(
            inference=self._concat(target_inference),
            samples=self._concat(target_samples),
            metrics=self._concat(target_metrics),
            metadata=self.target_metadata(),
        )

    @staticmethod
    def _select_baseline(baselines, candidate_factor):
        """Prefer a matching factor, otherwise allow one fixed baseline."""
        exact = [model for model in baselines if model.factor == candidate_factor]
        if exact:
            return exact[0]
        if len(baselines) == 1:
            return baselines[0]
        return None

    def _get_rating_from_completion(self, completion):
        if "Rating:" not in completion:
            raise ValueError("Cannot find rating value.")
        rating_text = completion.split("Rating:")[-1].strip()
        rating_text = rating_text.split('\n')[0].strip()
        rating_text = rating_text.replace('[', '').replace(']', '')
        rating_text = rating_text.rstrip('.').strip('"').strip("'").strip("*").strip()
        return float(rating_text)

    def _get_ratings_from_prompts(self, prompts, api_name, min_rating=0.0, max_rating=2.0):
        ratings, _ = self._get_judge_ratings(
            prompts,
            f"{api_name}_{self.winrate_baseline}_WinRateEvaluator",
            self._get_rating_from_completion,
            min_rating=min_rating,
            max_rating=max_rating,
            default_rating=self.DEFAULT_RATING,
        )
        return ratings

    def _get_rating_groups(self, prompt_groups):
        lengths = [len(prompts) for _, prompts in prompt_groups]
        prompts = [prompt for _, group in prompt_groups for prompt in group]
        api_names = [
            f"{api_name}_{self.winrate_baseline}_WinRateEvaluator"
            for api_name, group in prompt_groups
            for _ in group
        ]

        ratings, _ = self._get_judge_ratings(
            prompts,
            api_names,
            self._get_rating_from_completion,
            default_rating=self.DEFAULT_RATING,
        )
        groups = []
        offset = 0
        for length in lengths:
            groups.append(ratings[offset:offset + length])
            offset += length
        return groups

    def _get_all_ratings_from_data(self, data, column_name):
        model_relevance_concept_prompts = []
        model_relevance_instruction_prompts = []
        model_fluency_prompts = []
        # This is a generation dataset.
        for idx, row in data.iterrows():
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
        (
            model_relevance_concept_ratings,
            model_relevance_instruction_ratings,
            model_fluency_ratings,
        ) = self._get_rating_groups([
            (f"{column_name}_concept", model_relevance_concept_prompts),
            (f"{column_name}_instruction", model_relevance_instruction_prompts),
            (f"{column_name}_fluency", model_fluency_prompts),
        ])
        return list(zip(model_relevance_concept_prompts, model_relevance_concept_ratings)), \
               list(zip(model_relevance_instruction_prompts, model_relevance_instruction_ratings)), \
               list(zip(model_fluency_prompts, model_fluency_ratings))

    def compute_metrics(self, data):
        """Compare two responses by concept relevance, instruction relevance, and fluency; responses failing either relevance check cannot win."""
        data_copy = data.copy()
        data_copy = data_copy.reset_index(drop=True)

        baseline_relevance_concept_ratings, baseline_relevance_instruction_ratings, baseline_fluency_ratings = \
            self._get_all_ratings_from_data(data_copy, self.winrate_baseline)
        model_relevance_concept_ratings, model_relevance_instruction_ratings, model_fluency_ratings = \
            self._get_all_ratings_from_data(data_copy, self.model_name)
        
        # calculate win rate.
        winning_results = []
        for i in range(len(baseline_relevance_concept_ratings)):
            def harmonic_mean(scores):
                # Return 0 if any score is 0 to maintain strict evaluation
                if 0 in scores:
                    return 0
                return len(scores) / sum(1/s for s in scores)
            
            baseline_scores = [
                baseline_relevance_concept_ratings[i][-1],
                baseline_relevance_instruction_ratings[i][-1],
                baseline_fluency_ratings[i][-1]
            ]
            model_scores = [
                model_relevance_concept_ratings[i][-1],
                model_relevance_instruction_ratings[i][-1],
                model_fluency_ratings[i][-1]
            ]
            
            baseline_score = harmonic_mean(baseline_scores)
            model_score = harmonic_mean(model_scores)
            
            # Compare scores to determine winner
            if abs(baseline_score - model_score) < 1e-6:  # Float comparison with epsilon
                winning_results.append("tie")
            elif baseline_score > model_score:
                winning_results.append("baseline")
            else:
                winning_results.append("model")

        data[f"{self.model_name}_win_result"] = winning_results
        
        counter = Counter(winning_results)
        win_count = counter["model"]
        loss_count = counter["baseline"]
        tie_count = counter["tie"]
        total_samples = len(winning_results)

        metrics = {
            "win_rate": float(win_count / total_samples),
            "loss_rate": float(loss_count / total_samples),
            "tie_rate": float(tie_count / total_samples),
            "baseline_model": self.winrate_baseline,
        }

        return metrics
            
