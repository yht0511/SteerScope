"""Small, readable evaluator used to verify the evaluation pipeline."""

import pandas as pd

from steerscope.evaluation.dataset import expand_factors

from .evaluator import Evaluator


class DemoEvaluator(Evaluator):
    """Trace evaluator lifecycle hooks and score simple generated text."""

    def prepare_evaluation(self, models, concepts):
        self._log("prepare", f"{len(models)} model wrapper(s), {len(concepts)} concept(s)")
        self._inspect_dependencies()

    def _inspect_dependencies(self):
        dependencies = tuple(self.node.depends_on)
        if not dependencies:
            self._log("dependencies", "none configured")
            return
        if self.results is None:
            raise ValueError(
                f"DemoEvaluator '{self.node_id}' has dependencies but no result view."
            )

        preview_rows = int(self.params.get("dependency_preview_rows", 5))
        if preview_rows < 1:
            raise ValueError("DemoEvaluator dependency_preview_rows must be positive.")
        for dependency in dependencies:
            manifest = self.results.manifest(dependency) or {}
            result_kinds = tuple(
                (manifest.get("metadata") or {}).get(
                    "result_kinds", ("inference", "samples", "metrics")
                )
            )
            self._log(
                "dependency",
                f"querying '{dependency}' ({', '.join(result_kinds) or 'no tables'})",
            )
            for kind in result_kinds:
                table = self.results.query(dependency, kind=kind)
                columns = ", ".join(map(str, table.columns)) or "none"
                self._log(
                    f"dependency {dependency}.{kind}",
                    f"{len(table)} row(s); columns: {columns}",
                )
                if table.empty:
                    print("  <empty>")
                else:
                    print(table.head(preview_rows).to_string(index=False))
                    remaining = len(table) - preview_rows
                    if remaining > 0:
                        print(f"  ... {remaining} more row(s)")

    def open_resources(self, models):
        self._log("open resources", "none needed")

    def begin_target(self, target_id):
        self._log("begin target", target_id)

    def build_dataset(self, model, factors):
        config = dict(self.node.dataset)
        prompts = config.get("prompts", ["Hello from the demo evaluator."])
        if isinstance(prompts, str):
            prompts = [prompts]
        num_examples = int(config.get("num_examples", len(prompts)))
        if num_examples < 1:
            raise ValueError("DemoEvaluator dataset.num_examples must be positive.")
        if not prompts:
            raise ValueError("DemoEvaluator dataset.prompts must not be empty.")
        prompts = [str(prompts[index % len(prompts)]) for index in range(num_examples)]
        rows = pd.DataFrame({
            "dataset_name": "demo",
            "concept_id": model.concept.concept_id,
            "input_concept": model.concept.text,
            "input_id": range(num_examples),
            "raw_input": prompts,
            "input": prompts,
        })
        result = expand_factors(rows, factors)
        self._log(
            "build dataset",
            f"{num_examples} prompt(s) x {len(factors)} factor(s) = {len(result)} row(s)",
        )
        return result

    def examples_for_model(self, examples, model):
        selected = super().examples_for_model(examples, model)
        self._log(
            "select model input",
            f"{model.method}, concept={model.concept.concept_id}, "
            f"factor={model.factor:g}: {len(selected)} row(s)",
        )
        return selected

    def prepare_examples(self, examples, results=None, config=None):
        self._log("prepare generated rows", f"{len(examples)} row(s)")
        return examples

    def compute_metrics(self, examples):
        column = f"{self.model_name}_steered_generation"
        if column not in examples:
            raise KeyError(f"DemoEvaluator expected generated-text column '{column}'.")
        outputs = examples[column].fillna("").astype(str)
        factors = sorted(float(value) for value in examples["factor"].unique())
        nonempty_rates = []
        mean_lengths = []
        for factor in factors:
            selected = outputs[examples["factor"].astype(float) == factor]
            nonempty_rates.append(float(selected.str.strip().ne("").mean()))
            mean_lengths.append(float(selected.str.len().mean()))
        self._log("score", f"computed metrics for {len(factors)} factor(s)")
        return {
            "factor": factors,
            "demo_nonempty_rate": nonempty_rates,
            "demo_mean_output_length": mean_lengths,
            "raw_demo_output_length": outputs.str.len().tolist(),
        }

    def checkpoint_resources(self):
        self._log("checkpoint", "target result is ready to save")

    def close_resources(self):
        self._log("close resources", "done")

    @staticmethod
    def _log(step, detail):
        print(f"[DemoEvaluator] {step}: {detail}")
