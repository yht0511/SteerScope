"""Evaluator for the official MATH test split."""

from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from steerscope.evaluation.dataset import (
    concept_seed,
    expand_factors,
    require_dataset_type,
    require_num_examples,
    split_by_input_id,
)
from steerscope.evaluation.version import file_signature
from steerscope.utils.constants import HAS_SYSTEM_PROMPT_MODELS

from .evaluator import Evaluator
from .math_official import is_equiv, last_boxed_only_string, remove_boxed


MATH_DATASET = "MATH"
MATH_FILE = "MATH_test.parquet"


class MATHEvaluator(Evaluator):
    """Generate worked solutions and score their final boxed answers."""

    source_dependencies = (
        "evaluators/math_official/math_equivalence.py",
        "evaluators/math_official/util.py",
        "evaluators/math_official/UPSTREAM.md",
        "evaluators/math_official/LICENSE",
    )

    def __init__(self, node, context, **params):
        super().__init__(node, context, **params)
        self._tokenizer = None

    @classmethod
    def execution_context(cls, node, args):
        require_dataset_type(node.dataset, MATH_DATASET)
        require_num_examples(node.dataset)
        root = getattr(args, "master_data_dir", None)
        path = Path(root) / "math" / MATH_FILE if root else None
        return {**super().execution_context(node, args), "dataset": file_signature(path) if path else None}

    def open_resources(self, models):
        super().open_resources(models)
        name = getattr(self.args, "steering_model_name", None) or getattr(self.args, "model_name", None)
        if not name:
            raise ValueError(f"Evaluator '{self.node_id}' requires a base model name.")
        self._tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False, model_max_length=8192)
        self._tokenizer.padding_side = "right"

    def close_resources(self):
        self._tokenizer = None
        super().close_resources()

    def _load_data(self):
        root = getattr(self.args, "master_data_dir", None)
        if not root:
            raise ValueError("MATH requires evaluate.master_data_dir.")
        path = Path(root) / "math" / MATH_FILE
        if not path.exists():
            raise FileNotFoundError(f"MATH data not found at {path}. Run steerscope/data/download-math.py first.")
        data = pd.read_parquet(path)
        required = {"input_id", "problem", "solution", "gold_answer", "level", "subject"}
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(f"MATH data at {path} is missing columns: {missing}")
        return data

    def build_dataset(self, model, factors):
        config = dict(self.node.dataset)
        num_examples = require_num_examples(config)
        data = self._load_data()
        levels = config.get("levels")
        if levels:
            levels = [int(level) for level in ([levels] if isinstance(levels, (int, str)) else levels)]
            unknown = sorted(set(levels).difference(range(1, 6)))
            if unknown:
                raise ValueError(f"Unknown MATH levels: {unknown}")
            data = data[data["level"].isin(levels)]
        subjects = config.get("subjects")
        if subjects:
            subjects = [subjects] if isinstance(subjects, str) else list(subjects)
            unknown = sorted(set(subjects).difference(data["subject"].unique()))
            if unknown:
                raise ValueError(f"Unknown MATH subjects: {unknown}")
            data = data[data["subject"].isin(subjects)]
        if num_examples > len(data):
            raise ValueError(f"MATH requested {num_examples} examples, but the selected data contains only {len(data)}.")
        seed = concept_seed(
            config.get("seed", getattr(self.args, "seed", 42)),
            model.concept.concept_id,
            MATH_DATASET,
        )
        sampled = data.sample(n=num_examples, random_state=seed).sort_values("input_id", kind="stable")
        rows = []
        for _, row in sampled.iterrows():
            prompt = format_math_prompt(row["problem"])
            rows.append({
                "dataset_name": MATH_DATASET,
                "concept_id": model.concept.concept_id,
                "input_concept": model.concept.text,
                "input_id": int(row["input_id"]),
                "original_prompt": prompt,
                "raw_input": prompt,
                "input": self._format_chat(model.target.base_model, prompt),
                "math_problem": str(row["problem"]),
                "math_gold_answer": str(row["gold_answer"]),
                "math_level": int(row["level"]),
                "math_subject": str(row["subject"]),
            })
        examples = expand_factors(pd.DataFrame(rows), factors)
        return split_by_input_id(examples, config.get("split", "all"), config.get("split_ratio", getattr(self.args, "winrate_split_ratio", 0.5)))

    def _format_chat(self, model_name, prompt):
        messages = []
        if model_name in HAS_SYSTEM_PROMPT_MODELS:
            messages.append({"role": "system", "content": "You are a helpful assistant."})
        messages.append({"role": "user", "content": prompt})
        tokens = self._tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if tokens and self._tokenizer.bos_token_id is not None and tokens[0] == self._tokenizer.bos_token_id:
            tokens = tokens[1:]
        return self._tokenizer.decode(tokens)

    def compute_metrics(self, data):
        column = f"{self.model_name}_steered_generation"
        required = {column, "math_gold_answer", "math_level", "factor"}
        missing = sorted(required.difference(data.columns))
        if missing:
            raise KeyError(f"MATH inference is missing columns: {missing}")
        predictions = []
        correct = []
        for generation, gold in zip(data[column], data["math_gold_answer"]):
            prediction = remove_boxed(last_boxed_only_string(str(generation)))
            predictions.append(prediction)
            correct.append(bool(is_equiv(prediction, str(gold))))
        factors = [float(value) for value in data["factor"]]
        levels = [int(value) for value in data["math_level"]]
        result = {
            "factor": [], "math_accuracy": [], "math_format_compliance": [],
            "math_correct_count": [], "math_num_examples": [],
            "raw_math_predicted_answer": predictions,
            "raw_math_gold_answer": data["math_gold_answer"].astype(str).tolist(),
            "raw_math_is_correct": correct,
        }
        for level in range(1, 6):
            result[f"math_level_{level}_accuracy"] = []
        for factor in sorted(set(factors)):
            indices = [i for i, value in enumerate(factors) if value == factor]
            values = [correct[i] for i in indices]
            result["factor"].append(factor)
            result["math_accuracy"].append(sum(values) / len(values))
            result["math_format_compliance"].append(sum(predictions[i] is not None for i in indices) / len(indices))
            result["math_correct_count"].append(sum(values))
            result["math_num_examples"].append(len(indices))
            for level in range(1, 6):
                selected = [correct[i] for i in indices if levels[i] == level]
                result[f"math_level_{level}_accuracy"].append(sum(selected) / len(selected) if selected else None)
        return result

    def render_report(self, result, output_dir=None):
        metrics = {"math_accuracy": "Overall accuracy", "math_format_compliance": "Boxed-answer compliance"}
        metrics.update({f"math_level_{level}_accuracy": f"Level {level} accuracy" for level in range(1, 6)})
        return self._render_curve_report(result, metrics, output_dir, y_limits=(0.0, 1.0), y_axis_label="Score")

    def __str__(self):
        return "MATHEvaluator"


def format_math_prompt(problem):
    return "Solve the following mathematics problem. Show your reasoning, and put only the final answer inside \\boxed{...}.\n\n" + str(problem).strip()
