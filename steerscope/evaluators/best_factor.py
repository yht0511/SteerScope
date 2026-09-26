from .evaluator import Evaluator

from steerscope.evaluation.dataset import split_by_input_id
from steerscope.evaluation.result import EvaluationResult
from steerscope.evaluation.version import file_signature

import math
import pandas as pd
from pathlib import Path


class BestFactorEvaluator(Evaluator):
    """Select the best factor from an upstream evaluator's metric table."""

    requires_inference = False

    @classmethod
    def execution_context(cls, node, args):
        context = super().execution_context(node, args)
        external_path = node.input.get("path")
        if external_path is not None:
            context["external_input"] = file_signature(Path(external_path))
        return context

    def evaluate(self, models, concepts):
        """Read the explicitly declared upstream table and select factors."""
        self.prepare_evaluation(models, concepts)
        input_config = dict(self.node.input)
        external_path = input_config.get("path")
        if external_path is not None:
            if input_config.get("from") is not None or self.node.depends_on:
                raise ValueError(
                    f"Evaluator '{self.node_id}' input.path cannot be combined "
                    "with input.from or depends_on."
                )
            path = Path(external_path)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Evaluator '{self.node_id}' external input is missing: {path}"
                )
            examples = pd.read_parquet(path)
            filters = input_config.get("filters") or {}
            for column, value in filters.items():
                if column not in examples:
                    raise KeyError(
                        f"External input filter column '{column}' is missing."
                    )
                allowed = value if isinstance(value, list) else [value]
                examples = examples[examples[column].isin(allowed)]
        else:
            if self.results is None:
                raise ValueError(
                    f"Evaluator '{self.node_id}' requires an upstream result view."
                )
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
                kind=input_config.get("kind", "metrics"),
                filters=input_config.get("filters"),
            )
        examples = self.filter_concept_rows(examples, concepts)
        metrics = pd.DataFrame(self.compute_metrics(examples))
        metrics.insert(0, "evaluator_type", self.__class__.__name__)
        metrics.insert(0, "evaluator_id", self.node_id)
        return EvaluationResult(
            samples=examples,
            metrics=metrics,
            metadata=self.evaluation_metadata(),
        )

    def render_report(self, result, output_dir=None):
        if (
            result.metrics is None
            or result.metrics.empty
            or "factor" not in result.metrics
        ):
            raise ValueError("BestFactorEvaluator has no selected factors to report.")
        metrics = result.metrics.copy()
        metrics["factor"] = pd.to_numeric(metrics["factor"], errors="coerce")
        metrics = metrics.dropna(subset=["factor"])
        metrics = metrics[metrics["factor"].map(math.isfinite)]
        if metrics.empty:
            raise ValueError(
                "BestFactorEvaluator selected factors contain no finite values."
            )
        if "selected_score" in metrics:
            metrics["selected_score"] = pd.to_numeric(
                metrics["selected_score"], errors="coerce"
            )
        if "selected_improvement" in metrics:
            metrics["selected_improvement"] = pd.to_numeric(
                metrics["selected_improvement"], errors="coerce"
            )
        if "evaluation_score" in metrics:
            return self._render_split_report(metrics, output_dir)
        group_columns = ["method"] if "method" in metrics else []
        grouped = (
            metrics.groupby(group_columns, dropna=False, sort=True)
            if group_columns
            else [("all", metrics)]
        )
        rows = []
        for key, group in grouped:
            method = key[0] if isinstance(key, tuple) else key
            std = float(group["factor"].std()) if len(group) > 1 else 0.0
            margin = 1.96 * std / (len(group) ** 0.5)
            row = {
                "method": str(method),
                "mean_factor": float(group["factor"].mean()),
                "std_factor": std,
                "count": int(len(group)),
                "ci_lower": float(group["factor"].mean() - margin),
                "ci_upper": float(group["factor"].mean() + margin),
            }
            if "selected_score" in group:
                selected_scores = group["selected_score"].dropna()
                if not selected_scores.empty:
                    row["mean_selected_score"] = float(selected_scores.mean())
            if "selected_improvement" in group:
                improvements = group["selected_improvement"].dropna()
                if not improvements.empty:
                    row["mean_selected_improvement"] = float(
                        improvements.mean()
                    )
            rows.append(row)
        summary = pd.DataFrame(rows)
        summary_path = self._save_report_summary(summary, output_dir)

        plt = self._pyplot()
        figure, axis = plt.subplots(figsize=(max(6.0, 1.2 * len(summary)), 4.5))
        errors = (summary["ci_upper"] - summary["mean_factor"]).to_numpy()
        styles = self._method_plot_styles(summary["method"].tolist(), plt)
        colors = [
            styles[self._method_plot_key(method)]["color"]
            for method in summary["method"]
        ]
        axis.bar(
            summary["method"],
            summary["mean_factor"],
            yerr=errors,
            capsize=4,
            color=colors,
        )
        axis.set_title("Selected steering factor")
        axis.set_xlabel("Method")
        axis.set_ylabel("Mean selected factor (95% CI)")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
        return [summary_path, *self._save_report_figure(figure, output_dir)]

    def _render_split_report(self, metrics, output_dir=None):
        """Report held-out scores after per-group factor selection."""
        params = self.node_config.get("params", {})
        configured_metrics = params.get("report_metrics", {}) or {}
        if not isinstance(configured_metrics, dict):
            raise TypeError("BestFactorEvaluator report_metrics must be a mapping.")
        value_columns = ["evaluation_score"]
        value_columns.extend(
            alias
            for alias in configured_metrics
            if alias in metrics and alias not in value_columns
        )
        for column in value_columns:
            metrics[column] = pd.to_numeric(metrics[column], errors="coerce")

        group_columns = ["method"] if "method" in metrics else []
        grouped = (
            metrics.groupby(group_columns, dropna=False, sort=True)
            if group_columns
            else [("all", metrics)]
        )
        rows = []
        for key, group in grouped:
            method = key[0] if isinstance(key, tuple) else key
            factor_std = float(group["factor"].std()) if len(group) > 1 else 0.0
            factor_margin = 1.96 * factor_std / (len(group) ** 0.5)
            row = {
                "method": str(method),
                "mean_factor": float(group["factor"].mean()),
                "std_factor": factor_std,
                "count": int(len(group)),
                "ci_lower": float(group["factor"].mean() - factor_margin),
                "ci_upper": float(group["factor"].mean() + factor_margin),
            }
            selected_scores = group["selected_score"].dropna()
            if not selected_scores.empty:
                row["mean_selected_score"] = float(selected_scores.mean())
            improvements = group["selected_improvement"].dropna()
            if not improvements.empty:
                row["mean_selected_improvement"] = float(improvements.mean())
            for column in value_columns:
                values = group[column].dropna()
                if values.empty:
                    continue
                std = float(values.std()) if len(values) > 1 else 0.0
                margin = 1.96 * std / (len(values) ** 0.5)
                row[f"mean_{column}"] = float(values.mean())
                row[f"std_{column}"] = std
                row[f"{column}_ci_lower"] = float(values.mean() - margin)
                row[f"{column}_ci_upper"] = float(values.mean() + margin)
            rows.append(row)
        summary = pd.DataFrame(rows)
        summary_path = self._save_report_summary(summary, output_dir)

        plot_metric = (
            "lm_judge_rating"
            if "lm_judge_rating" in configured_metrics
            else "evaluation_score"
        )
        mean_column = f"mean_{plot_metric}"
        lower_column = f"{plot_metric}_ci_lower"
        upper_column = f"{plot_metric}_ci_upper"
        valid = summary.dropna(subset=[mean_column, lower_column, upper_column])
        if valid.empty:
            raise ValueError(
                "BestFactorEvaluator split report contains no held-out scores."
            )
        plt = self._pyplot()
        figure, axis = plt.subplots(figsize=(max(6.0, 1.2 * len(valid)), 4.5))
        errors = (valid[upper_column] - valid[mean_column]).to_numpy()
        styles = self._method_plot_styles(valid["method"].tolist(), plt)
        colors = [
            styles[self._method_plot_key(method)]["color"]
            for method in valid["method"]
        ]
        axis.bar(
            valid["method"],
            valid[mean_column],
            yerr=errors,
            capsize=4,
            color=colors,
        )
        axis.set_title("Held-out selection score")
        axis.set_xlabel("Method")
        axis.set_ylabel(f"Mean {plot_metric} (95% CI)")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
        return [summary_path, *self._save_report_figure(figure, output_dir)]

    def compute_metrics(self, examples):
        params = self.node_config.get("params", {})
        metric = params.get("metric", "lm_judge_rating")
        strategy = str(params.get("strategy", "argmax"))
        aggregation = str(params.get("aggregation", "mean"))
        configured_group_by = params.get("group_by")
        baseline_factor = params.get("baseline_factor")
        fallback_baseline_method = params.get("fallback_baseline_method")
        if strategy not in {"argmax", "argmin"}:
            raise ValueError(
                f"Unsupported BestFactorEvaluator strategy '{strategy}'."
            )
        if aggregation not in {"mean", "median"}:
            raise ValueError(
                "Unsupported BestFactorEvaluator aggregation "
                f"'{aggregation}'. Choose from ['mean', 'median']."
            )
        if baseline_factor is not None:
            baseline_factor = float(baseline_factor)
        if fallback_baseline_method is not None:
            fallback_baseline_method = str(fallback_baseline_method)
            if baseline_factor is None:
                raise ValueError(
                    "BestFactorEvaluator fallback_baseline_method requires "
                    "baseline_factor."
                )
            if "method" not in examples.columns:
                raise KeyError(
                    "BestFactorEvaluator fallback_baseline_method requires a "
                    "'method' column."
                )
        if metric not in examples.columns:
            raise KeyError(f"BestFactorEvaluator cannot find metric '{metric}'.")
        if "factor" not in examples.columns:
            raise KeyError("BestFactorEvaluator requires a 'factor' column.")

        if configured_group_by is None:
            group_columns = [
                column
                for column in ("target_id", "method", "concept_id")
                if column in examples.columns
            ]
        else:
            if isinstance(configured_group_by, str):
                configured_group_by = [configured_group_by]
            if not isinstance(configured_group_by, (list, tuple)):
                raise TypeError(
                    "BestFactorEvaluator group_by must be a column name or list."
                )
            group_columns = list(dict.fromkeys(configured_group_by))
            missing_groups = sorted(
                set(group_columns).difference(examples.columns)
            )
            if missing_groups:
                raise KeyError(
                    "BestFactorEvaluator cannot group by missing columns: "
                    f"{missing_groups}"
                )

        numeric = examples.copy()
        numeric["_selection_factor"] = pd.to_numeric(
            numeric["factor"], errors="coerce"
        )
        numeric["_selection_metric"] = pd.to_numeric(
            numeric[metric], errors="coerce"
        )
        numeric = numeric.dropna(
            subset=["_selection_factor", "_selection_metric"]
        )
        numeric = numeric[
            numeric["_selection_factor"].map(math.isfinite)
            & numeric["_selection_metric"].map(math.isfinite)
        ]
        split_config = params.get("split")
        if split_config is not None:
            return self._compute_split_metrics(
                numeric,
                group_columns=group_columns,
                metric=metric,
                strategy=strategy,
                aggregation=aggregation,
                baseline_factor=baseline_factor,
                fallback_baseline_method=fallback_baseline_method,
                split_config=split_config,
                report_metrics=params.get("report_metrics", {}),
            )
        shared_baseline_score = None
        if fallback_baseline_method is not None:
            baseline_rows = numeric[
                (numeric["method"].astype(str) == fallback_baseline_method)
                & (numeric["_selection_factor"] == baseline_factor)
            ]
            if baseline_rows.empty:
                raise ValueError(
                    "BestFactorEvaluator cannot find fallback baseline "
                    f"method={fallback_baseline_method!r}, "
                    f"factor={baseline_factor}."
                )
            shared_baseline_score = float(
                baseline_rows["_selection_metric"].agg(aggregation)
            )
        grouped = (
            numeric.groupby(group_columns, dropna=False, sort=True)
            if group_columns
            else [((), numeric)]
        )
        rows = []
        for group_key, group in grouped:
            scores = group.groupby("_selection_factor", sort=True)[
                "_selection_metric"
            ].agg(aggregation).dropna()
            if scores.empty:
                continue
            baseline_score = None
            candidate_scores = scores
            if baseline_factor is not None:
                if baseline_factor in scores.index:
                    baseline_score = float(scores.loc[baseline_factor])
                elif shared_baseline_score is not None:
                    baseline_score = shared_baseline_score
                else:
                    identity = self._group_identity(group_columns, group_key)
                    raise ValueError(
                        "BestFactorEvaluator baseline factor "
                        f"{baseline_factor} is missing for {identity}."
                    )
                candidate_scores = scores.drop(
                    index=baseline_factor,
                    errors="ignore",
                )
                if candidate_scores.empty:
                    identity = self._group_identity(group_columns, group_key)
                    raise ValueError(
                        "BestFactorEvaluator has no candidate factors after "
                        f"excluding baseline factor {baseline_factor} for "
                        f"{identity}."
                    )
            if strategy == "argmax":
                factor = candidate_scores.idxmax()
            else:
                factor = candidate_scores.idxmin()
            selected_score = float(candidate_scores.loc[factor])
            if baseline_score is None:
                improvement = None
            else:
                improvement = (
                    selected_score - baseline_score
                    if strategy == "argmax"
                    else baseline_score - selected_score
                )
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            row = dict(zip(group_columns, group_key))
            row.update({
                "factor": float(factor),
                "selected_metric": str(metric),
                "selection_strategy": strategy,
                "selection_aggregation": aggregation,
                "selected_score": selected_score,
                "baseline_factor": baseline_factor,
                "fallback_baseline_method": fallback_baseline_method,
                "baseline_score": baseline_score,
                "selected_improvement": improvement,
            })
            rows.append(row)
        if not rows:
            raise ValueError("BestFactorEvaluator received no numeric factor scores.")
        return rows

    def _compute_split_metrics(
        self,
        numeric,
        *,
        group_columns,
        metric,
        strategy,
        aggregation,
        baseline_factor,
        fallback_baseline_method,
        split_config,
        report_metrics,
    ):
        """Select on one input-ID partition and score on another."""
        if not isinstance(split_config, dict):
            raise TypeError("BestFactorEvaluator split must be a mapping.")
        selection_split = str(split_config.get("selection", "validation"))
        evaluation_split = str(split_config.get("evaluation", "test"))
        if selection_split == evaluation_split:
            raise ValueError(
                "BestFactorEvaluator selection and evaluation splits must differ."
            )
        split_ratio = float(
            split_config.get(
                "ratio", getattr(self.args, "winrate_split_ratio", 0.5)
            )
        )
        selection_data = split_by_input_id(
            numeric, selection_split, split_ratio
        )
        evaluation_data = split_by_input_id(
            numeric, evaluation_split, split_ratio
        )
        if selection_data.empty or evaluation_data.empty:
            raise ValueError(
                "BestFactorEvaluator split produced an empty selection or "
                "evaluation partition."
            )

        report_metrics = report_metrics or {}
        if not isinstance(report_metrics, dict):
            raise TypeError("BestFactorEvaluator report_metrics must be a mapping.")
        reserved = {
            "factor",
            "selected_score",
            "baseline_score",
            "selected_improvement",
            "evaluation_score",
            "evaluation_baseline_score",
            "evaluation_improvement",
        }
        collisions = sorted(set(report_metrics).intersection(reserved))
        if collisions:
            raise ValueError(
                "BestFactorEvaluator report_metrics aliases use reserved "
                f"columns: {collisions}"
            )
        missing_metrics = sorted(
            set(report_metrics.values()).difference(numeric.columns)
        )
        if missing_metrics:
            raise KeyError(
                "BestFactorEvaluator cannot find report metric columns: "
                f"{missing_metrics}"
            )

        grouped = (
            selection_data.groupby(group_columns, dropna=False, sort=True)
            if group_columns
            else [((), selection_data)]
        )
        rows = []
        for group_key, selection_group in grouped:
            scores = selection_group.groupby("_selection_factor", sort=True)[
                "_selection_metric"
            ].agg(aggregation).dropna()
            if scores.empty:
                continue
            baseline_score, candidate_scores = self._partition_scores(
                scores,
                selection_data,
                selection_group,
                group_columns=group_columns,
                group_key=group_key,
                metric_column="_selection_metric",
                aggregation=aggregation,
                baseline_factor=baseline_factor,
                fallback_baseline_method=fallback_baseline_method,
            )
            factor = (
                candidate_scores.idxmax()
                if strategy == "argmax"
                else candidate_scores.idxmin()
            )
            selected_score = float(candidate_scores.loc[factor])
            selected_improvement = self._improvement(
                selected_score, baseline_score, strategy
            )

            evaluation_group = self._matching_group(
                evaluation_data, group_columns, group_key
            )
            selected_evaluation = evaluation_group[
                evaluation_group["_selection_factor"] == factor
            ]
            evaluation_score = self._aggregate_numeric(
                selected_evaluation["_selection_metric"],
                aggregation,
                description=(
                    f"selected factor {float(factor)} on the evaluation split"
                ),
            )
            evaluation_baseline_score = self._partition_baseline_score(
                evaluation_data,
                evaluation_group,
                group_columns=group_columns,
                group_key=group_key,
                metric_column="_selection_metric",
                aggregation=aggregation,
                baseline_factor=baseline_factor,
                fallback_baseline_method=fallback_baseline_method,
            )
            evaluation_improvement = self._improvement(
                evaluation_score, evaluation_baseline_score, strategy
            )

            normalized_key = (
                group_key if isinstance(group_key, tuple) else (group_key,)
            )
            row = dict(zip(group_columns, normalized_key))
            row.update({
                "factor": float(factor),
                "selected_metric": str(metric),
                "selection_strategy": strategy,
                "selection_aggregation": aggregation,
                "selected_score": selected_score,
                "baseline_factor": baseline_factor,
                "fallback_baseline_method": fallback_baseline_method,
                "baseline_score": baseline_score,
                "selected_improvement": selected_improvement,
                "selection_split": selection_split,
                "evaluation_split": evaluation_split,
                "split_ratio": split_ratio,
                "evaluation_score": evaluation_score,
                "evaluation_baseline_score": evaluation_baseline_score,
                "evaluation_improvement": evaluation_improvement,
            })
            for alias, source_column in report_metrics.items():
                row[str(alias)] = self._aggregate_numeric(
                    selected_evaluation[source_column],
                    aggregation,
                    description=(
                        f"report metric {source_column!r} for selected factor "
                        f"{float(factor)}"
                    ),
                )
            rows.append(row)
        if not rows:
            raise ValueError("BestFactorEvaluator received no numeric factor scores.")
        return rows

    def _partition_scores(
        self,
        scores,
        partition,
        group,
        *,
        group_columns,
        group_key,
        metric_column,
        aggregation,
        baseline_factor,
        fallback_baseline_method,
    ):
        baseline_score = self._partition_baseline_score(
            partition,
            group,
            group_columns=group_columns,
            group_key=group_key,
            metric_column=metric_column,
            aggregation=aggregation,
            baseline_factor=baseline_factor,
            fallback_baseline_method=fallback_baseline_method,
        )
        candidate_scores = scores
        if baseline_factor is not None:
            candidate_scores = scores.drop(index=baseline_factor, errors="ignore")
            if candidate_scores.empty:
                identity = self._group_identity(group_columns, group_key)
                raise ValueError(
                    "BestFactorEvaluator has no candidate factors after "
                    f"excluding baseline factor {baseline_factor} for {identity}."
                )
        return baseline_score, candidate_scores

    def _partition_baseline_score(
        self,
        partition,
        group,
        *,
        group_columns,
        group_key,
        metric_column,
        aggregation,
        baseline_factor,
        fallback_baseline_method,
    ):
        if baseline_factor is None:
            return None
        baseline_rows = group[group["_selection_factor"] == baseline_factor]
        if baseline_rows.empty and fallback_baseline_method is not None:
            baseline_rows = partition[
                (partition["method"].astype(str) == fallback_baseline_method)
                & (partition["_selection_factor"] == baseline_factor)
            ]
            normalized_key = (
                group_key if isinstance(group_key, tuple) else (group_key,)
            )
            for column, value in zip(group_columns, normalized_key):
                if column in {"method", "target_id"}:
                    continue
                baseline_rows = self._matching_value(
                    baseline_rows, column, value
                )
        if baseline_rows.empty:
            identity = self._group_identity(group_columns, group_key)
            raise ValueError(
                "BestFactorEvaluator baseline factor "
                f"{baseline_factor} is missing for {identity}."
            )
        return self._aggregate_numeric(
            baseline_rows[metric_column],
            aggregation,
            description=f"baseline factor {baseline_factor}",
        )

    @classmethod
    def _matching_group(cls, data, group_columns, group_key):
        if not group_columns:
            return data
        normalized_key = group_key if isinstance(group_key, tuple) else (group_key,)
        selected = data
        for column, value in zip(group_columns, normalized_key):
            selected = cls._matching_value(selected, column, value)
        return selected

    @staticmethod
    def _matching_value(data, column, value):
        if pd.isna(value):
            return data[data[column].isna()]
        return data[data[column] == value]

    @staticmethod
    def _aggregate_numeric(values, aggregation, *, description):
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        numeric = numeric[numeric.map(math.isfinite)]
        if numeric.empty:
            raise ValueError(
                f"BestFactorEvaluator has no numeric values for {description}."
            )
        return float(numeric.agg(aggregation))

    @staticmethod
    def _improvement(selected_score, baseline_score, strategy):
        if baseline_score is None:
            return None
        return (
            selected_score - baseline_score
            if strategy == "argmax"
            else baseline_score - selected_score
        )

    @staticmethod
    def _group_identity(group_columns, group_key) -> str:
        if not group_columns:
            return "the global group"
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        values = ", ".join(
            f"{column}={value!r}"
            for column, value in zip(group_columns, group_key)
        )
        return f"group ({values})"

    def __str__(self):
        return "BestFactorEvaluator"
