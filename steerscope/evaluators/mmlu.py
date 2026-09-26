"""MMLU evaluator with an evaluator-owned data and scoring pipeline."""

from pathlib import Path

import pandas as pd
import torch
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
from steerscope.utils.constants import CHAT_MODELS

from .evaluator import Evaluator


MMLU_CHOICES = ("A", "B", "C", "D")
MMLU_DATASET_FILES = {
    "MMLU_train": "MMLU_train.parquet",
    "MMLU_val": "MMLU_val.parquet",
    "MMLU_test": "MMLU_test.parquet",
}


class MMLUEvaluator(Evaluator):
    """Build and score four-choice MMLU next-token inference."""

    def __init__(self, node, context, **params):
        super().__init__(node, context, **params)
        self._tokenizer = None
        self._dev_examples = None

    def render_report(self, result, output_dir=None):
        """Plot MMLU quality as a function of steering strength."""
        return self._render_curve_report(
            result,
            {
                "mmlu_accuracy": "Accuracy",
                "mmlu_mean_gold_probability": "Mean gold-answer probability",
            },
            output_dir,
            y_limits=(0.0, 1.0),
            y_axis_label="Score",
        )

    @classmethod
    def execution_context(cls, node, args):
        dataset_type = require_dataset_type(node.dataset, MMLU_DATASET_FILES)
        require_num_examples(node.dataset)
        data_root = getattr(args, "master_data_dir", None)
        if not data_root:
            return {
                **super().execution_context(node, args),
                "dataset": None,
                "dev": None,
            }
        root = Path(data_root) / "mmlu"
        return {
            **super().execution_context(node, args),
            "dataset": file_signature(root / MMLU_DATASET_FILES[dataset_type]),
            "dev": file_signature(root / "MMLU_dev.parquet"),
            "default_split_ratio": getattr(args, "winrate_split_ratio", 0.5),
        }

    def open_resources(self, models) -> None:
        super().open_resources(models)
        model_name = getattr(self.args, "steering_model_name", None) or getattr(
            self.args,
            "model_name",
            None,
        )
        if not model_name:
            raise ValueError(
                f"Evaluator '{self.node_id}' requires a base model name."
            )
        max_length = 128000 if "google/gemma-3" in model_name else 1024
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            model_max_length=max_length,
        )
        self._tokenizer.padding_side = "right"

    def close_resources(self) -> None:
        self._tokenizer = None
        self._dev_examples = None
        super().close_resources()

    def build_dataset(self, model, factors) -> pd.DataFrame:
        config = dict(self.node.dataset)
        dataset_type = require_dataset_type(config, MMLU_DATASET_FILES)
        num_examples = require_num_examples(config)
        data = self._load_split(dataset_type)
        if num_examples > len(data):
            raise ValueError(
                f"MMLU requested {num_examples} examples, but {dataset_type} "
                f"contains only {len(data)}."
            )
        sampled = data.sample(
            n=num_examples,
            random_state=concept_seed(
                config.get("seed", getattr(self.args, "seed", 42)),
                model.concept.concept_id,
                dataset_type,
            ),
        ).sort_index()
        dev_by_subject = self._load_dev_examples()
        choice_token_ids = self._choice_token_ids(model.target.base_model)
        rows = []
        for source_id, row in sampled.iterrows():
            subject = str(row["subject"])
            if subject and subject not in dev_by_subject:
                raise KeyError(f"No MMLU dev examples for subject '{subject}'.")
            choices = [str(choice) for choice in row["choices"]]
            prompt, formatted, n_shot = self._format_chat_prompt(
                model,
                subject,
                str(row["question"]),
                choices,
                dev_by_subject.get(subject, ()),
            )
            answer_index = int(row["answer"])
            if answer_index not in range(len(MMLU_CHOICES)):
                raise ValueError(f"Invalid MMLU answer index: {answer_index}")
            rows.append({
                "dataset_name": dataset_type,
                "concept_id": model.concept.concept_id,
                "input_concept": model.concept.text,
                "input_id": int(source_id),
                "original_prompt": prompt,
                "raw_input": prompt,
                "input": formatted,
                "mmlu_question": str(row["question"]),
                "mmlu_subject": subject,
                "mmlu_choices": choices,
                "mmlu_answer_index": answer_index,
                "mmlu_answer_label": MMLU_CHOICES[answer_index],
                "mmlu_n_shot": n_shot,
                "choice_token_ids": choice_token_ids,
                "inference_mode": "choice_logits",
            })
        examples = expand_factors(pd.DataFrame(rows), factors)
        return split_by_input_id(
            examples,
            config.get("split", "all"),
            config.get(
                "split_ratio",
                getattr(self.args, "winrate_split_ratio", 0.5),
            ),
        )

    def _format_chat_prompt_text(self, model_name, prompt):
        messages = []
        if model_name in HAS_SYSTEM_PROMPT_MODELS:
            messages.append({"role": "system", "content": "You are a helpful assistant."})
        tokens = self._tokenizer.apply_chat_template(
            messages + [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            tokenizer_kwargs={"verbose": False},
        )
        bos_token_id = self._tokenizer.bos_token_id
        if tokens and bos_token_id is not None and tokens[0] == bos_token_id:
            tokens = tokens[1:]
        return self._tokenizer.decode(tokens)

    def _load_split(self, dataset_type: str) -> pd.DataFrame:
        path = self._mmlu_root() / MMLU_DATASET_FILES[dataset_type]
        if not path.exists():
            raise FileNotFoundError(
                f"MMLU data not found at {path}. Run "
                "steerscope/data/download-mmlu.py first."
            )
        data = pd.read_parquet(path)
        self._validate_columns(data, path)
        return data

    def _load_dev_examples(self):
        if self._dev_examples is not None:
            return self._dev_examples
        path = self._mmlu_root() / "MMLU_dev.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"MMLU dev data not found at {path}. Run "
                "steerscope/data/download-mmlu.py first."
            )
        data = pd.read_parquet(path)
        self._validate_columns(data, path)
        subject_counts = data.groupby("subject").size()
        if len(subject_counts) != 57 or not subject_counts.eq(5).all():
            raise ValueError(
                "MMLU dev data must contain five examples for each of 57 subjects."
            )
        self._dev_examples = {
            str(subject): [
                {
                    "question": str(row["question"]),
                    "choices": [str(choice) for choice in row["choices"]],
                    "answer": int(row["answer"]),
                }
                for _, row in group.iterrows()
            ]
            for subject, group in data.groupby("subject", sort=False)
        }
        return self._dev_examples

    def _mmlu_root(self) -> Path:
        if not self.args.master_data_dir:
            raise ValueError("MMLU requires evaluate.master_data_dir.")
        return Path(self.args.master_data_dir) / "mmlu"

    @staticmethod
    def _validate_columns(data, path) -> None:
        required = {"question", "subject", "choices", "answer"}
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(f"MMLU data at {path} is missing columns: {missing}")

    def _choice_token_ids(self, model_name):
        token_ids = []
        prefix = "" if model_name in CHAT_MODELS else " "
        for choice in MMLU_CHOICES:
            encoded = self._tokenizer.encode(
                f"{prefix}{choice}",
                add_special_tokens=False,
            )
            if len(encoded) != 1:
                raise ValueError(
                    f"MMLU answer {prefix + choice!r} must encode to one token, "
                    f"got {encoded}."
                )
            token_ids.append(int(encoded[0]))
        return token_ids

    def _format_chat_prompt(
        self,
        target_model,
        subject,
        question,
        choices,
        dev_examples,
    ):
        available_shots = min(5, len(dev_examples))
        for n_shot in range(available_shots, -1, -1):
            prompt = format_mmlu_prompt(
                subject, question, choices, dev_examples, n_shot=n_shot
            )
            formatted = self._format_chat_prompt_text(
                target_model.target.base_model, prompt
            )
            if formatted is None:
                continue
            return prompt, formatted, n_shot
        raise ValueError(
            "MMLU question and its input views do not fit within the tokenizer's "
            f"{self._tokenizer.model_max_length}-token context window."
        )

    def compute_metrics(self, data):
        logits_column = f"{self.model_name}_choice_logits"
        required_columns = {logits_column, "mmlu_answer_index", "factor"}
        missing_columns = sorted(required_columns.difference(data.columns))
        if missing_columns:
            raise KeyError(f"MMLU inference is missing columns: {missing_columns}")

        choice_logits = torch.tensor(
            [list(values) for values in data[logits_column]], dtype=torch.float64
        )
        if choice_logits.ndim != 2 or choice_logits.shape[1] != 4:
            raise ValueError(
                "MMLU inference must provide exactly four choice logits per example."
            )
        if not torch.isfinite(choice_logits).all():
            raise ValueError("MMLU inference contains non-finite choice logits.")

        answer_indices = torch.tensor(
            data["mmlu_answer_index"].astype(int).tolist(), dtype=torch.long
        )
        if not ((0 <= answer_indices) & (answer_indices < 4)).all():
            raise ValueError("MMLU answer indices must be between 0 and 3.")

        temperature = float(getattr(self.args, "temperature", 1.0))
        if temperature <= 0:
            raise ValueError("MMLU temperature must be greater than zero.")
        choice_probabilities = torch.softmax(choice_logits / temperature, dim=1)
        predicted_indices = choice_logits.argmax(dim=1)
        correct = predicted_indices.eq(answer_indices)
        gold_probabilities = choice_probabilities.gather(
            1, answer_indices.unsqueeze(1)
        ).squeeze(1)

        metrics = {
            "factor": [],
            "mmlu_accuracy": [],
            "mmlu_correct_count": [],
            "mmlu_num_examples": [],
            "mmlu_mean_gold_probability": [],
            "temperature": temperature,
            "raw_mmlu_choice_probabilities": choice_probabilities.tolist(),
            "raw_mmlu_predicted_index": predicted_indices.tolist(),
            "raw_mmlu_predicted_label": [
                MMLU_CHOICES[index] for index in predicted_indices.tolist()
            ],
            "raw_mmlu_is_correct": correct.tolist(),
            "raw_mmlu_gold_probability": gold_probabilities.tolist(),
        }
        factors = [float(value) for value in data["factor"]]
        for factor in sorted(set(factors)):
            indices = [
                index for index, value in enumerate(factors) if value == factor
            ]
            factor_correct = correct[indices]
            metrics["factor"].append(factor)
            metrics["mmlu_accuracy"].append(
                float(factor_correct.double().mean().item())
            )
            metrics["mmlu_correct_count"].append(int(factor_correct.sum().item()))
            metrics["mmlu_num_examples"].append(len(indices))
            metrics["mmlu_mean_gold_probability"].append(
                float(gold_probabilities[indices].mean().item())
            )
        return metrics

    def __str__(self):
        return "MMLUEvaluator"


def format_mmlu_example(question, choices, answer=None):
    if len(choices) != len(MMLU_CHOICES):
        raise ValueError("MMLU examples must have exactly four choices.")
    lines = [str(question).strip()]
    lines.extend(
        f"{label}. {choice}" for label, choice in zip(MMLU_CHOICES, choices)
    )
    suffix = "Answer:"
    if answer is not None:
        answer = int(answer)
        if answer not in range(len(MMLU_CHOICES)):
            raise ValueError(f"Invalid MMLU answer index: {answer}")
        suffix += f" {MMLU_CHOICES[answer]}"
    lines.append(suffix)
    return "\n".join(lines)


def format_mmlu_prompt(subject, question, choices, dev_examples, n_shot=5):
    subject_text = str(subject).replace("_", " ").strip()
    heading = "The following are multiple choice questions (with answers)"
    if subject_text:
        heading += f" about {subject_text}"
    heading += ".\n\n"
    demonstrations = [
        format_mmlu_example(
            example["question"], example["choices"], example["answer"]
        )
        for example in dev_examples[:n_shot]
    ]
    query = format_mmlu_example(question, choices)
    return heading + "\n\n".join([*demonstrations, query])
