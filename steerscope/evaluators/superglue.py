"""SuperGLUE evaluator using full candidate-continuation likelihoods."""

from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
import json
import re

import evaluate
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from steerscope.evaluation.dataset import (
    concept_seed,
    expand_factors,
    require_dataset_type,
    split_by_input_id,
)
from steerscope.evaluation.version import file_signature
from steerscope.utils.constants import HAS_SYSTEM_PROMPT_MODELS

from .evaluator import Evaluator


SUPERGLUE_DATASET = "SuperGLUE"
SUPERGLUE_TASKS = ("boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc")
SUPERGLUE_METRIC_REVISION = "247e25682266968ee6fc0f4c7d0b1897a4ea740d"
SUPERGLUE_PROMPT_REVISION = "f4d4b3de3ee6741a7151a9fe74945ee515262f4c"
METRIC_CONFIG = {**{task: task for task in SUPERGLUE_TASKS}, "wsc": "wsc.fixed"}


def evaluate_package_version():
    """Read the installed distribution version without relying on module globals."""
    try:
        return version("evaluate")
    except PackageNotFoundError:
        return "unknown"


class SuperGLUEEvaluator(Evaluator):
    """Evaluate all eight main SuperGLUE tasks on labeled validation data."""

    def __init__(self, node, context, **params):
        super().__init__(node, context, **params)
        self._tokenizer = None
        self._metrics = {}

    @classmethod
    def execution_context(cls, node, args):
        require_dataset_type(node.dataset, SUPERGLUE_DATASET)
        root = getattr(args, "master_data_dir", None)
        tasks = normalize_tasks(node.dataset.get("tasks", SUPERGLUE_TASKS))
        signatures = None
        if root:
            directory = Path(root) / "superglue"
            signatures = {
                task: file_signature(directory / f"{task}_validation.parquet")
                for task in tasks
            }
        return {
            **super().execution_context(node, args),
            "datasets": signatures,
            "evaluate_version": evaluate_package_version(),
            "metric_revision": SUPERGLUE_METRIC_REVISION,
            "prompt_revision": SUPERGLUE_PROMPT_REVISION,
            "prompt_spec": "lm-eval-v1-with-corrected-multirc-label-order-v1",
        }

    def open_resources(self, models):
        super().open_resources(models)
        if not callable(getattr(evaluate, "load", None)):
            raise ImportError(
                "SuperGLUE requires Hugging Face's 'evaluate' package, but "
                f"Python imported {getattr(evaluate, '__file__', evaluate)!r}. "
                "Install the package with `pip install evaluate`."
            )
        model_name = getattr(self.args, "steering_model_name", None) or getattr(
            self.args, "model_name", None
        )
        if not model_name:
            raise ValueError(f"Evaluator '{self.node_id}' requires a base model name.")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, model_max_length=8192
        )
        self._tokenizer.padding_side = "right"
        tasks = normalize_tasks(self.node.dataset.get("tasks", SUPERGLUE_TASKS))
        self._metrics = {
            task: evaluate.load(
                "super_glue",
                METRIC_CONFIG[task],
                revision=SUPERGLUE_METRIC_REVISION,
            )
            for task in tasks
        }

    def close_resources(self):
        self._metrics = {}
        self._tokenizer = None
        super().close_resources()

    def _root(self):
        root = getattr(self.args, "master_data_dir", None)
        if not root:
            raise ValueError("SuperGLUE requires evaluate.master_data_dir.")
        return Path(root) / "superglue"

    def _load_task(self, task):
        path = self._root() / f"{task}_validation.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"SuperGLUE data not found at {path}. Run "
                "steerscope/data/download-superglue.py first."
            )
        return pd.read_parquet(path)

    def build_dataset(self, model, factors):
        config = dict(self.node.dataset)
        require_dataset_type(config, SUPERGLUE_DATASET)
        tasks = normalize_tasks(config.get("tasks", SUPERGLUE_TASKS))
        counts = config.get("num_examples_per_task", config.get("num_examples"))
        if counts is None:
            raise ValueError(
                "SuperGLUE dataset requires num_examples_per_task."
            )
        seed = int(config.get("seed", getattr(self.args, "seed", 42)))
        rows = []
        for task in tasks:
            configured_count = counts.get(task, counts.get("default")) if isinstance(counts, dict) else counts
            if configured_count is None:
                raise ValueError(f"SuperGLUE has no example count configured for {task}.")
            count = None if str(configured_count).lower() == "all" else int(configured_count)
            if count is not None and count < 1:
                raise ValueError("SuperGLUE example counts must be positive or 'all'.")
            data = self._sample_task(
                task,
                self._load_task(task),
                count,
                concept_seed(
                    seed,
                    model.concept.concept_id,
                    f"{SUPERGLUE_DATASET}:{task}",
                ),
            )
            rows.extend(self._task_rows(task, data, model))
        examples = expand_factors(pd.DataFrame(rows), factors)
        return split_by_input_id(
            examples,
            config.get("split", "all"),
            config.get("split_ratio", getattr(self.args, "winrate_split_ratio", 0.5)),
        )

    @staticmethod
    def _sample_task(task, data, count, seed):
        if task == "multirc":
            keys = data["idx"].map(
                lambda value: (int(value["paragraph"]), int(value["question"]))
            )
            groups = pd.Series(keys.unique())
            if count is None:
                return data.reset_index(drop=True)
            if count > len(groups):
                raise ValueError(
                    f"SuperGLUE multirc requested {count} questions, but validation "
                    f"contains only {len(groups)}."
                )
            selected = set(groups.sample(n=count, random_state=seed).tolist())
            return data[keys.isin(selected)].reset_index(drop=True)
        if count is None:
            return data.reset_index(drop=True)
        if count > len(data):
            raise ValueError(
                f"SuperGLUE {task} requested {count} examples, but validation "
                f"contains only {len(data)}."
            )
        return data.sample(n=count, random_state=seed).reset_index(drop=True)

    def _task_rows(self, task, data, model):
        rows = []
        for source_position, row in data.iterrows():
            prompt, choices, answer, identity, metadata = format_superglue_row(task, row)
            formatted = self._format_chat(model.target.base_model, prompt)
            rows.append({
                "dataset_name": SUPERGLUE_DATASET,
                "concept_id": model.concept.concept_id,
                "input_concept": model.concept.text,
                "input_id": f"{task}:{identity}",
                "original_prompt": prompt,
                "raw_input": prompt,
                "input": formatted,
                "superglue_task": task,
                "superglue_source_position": int(source_position),
                "superglue_answer_index": int(answer),
                # JSON keeps heterogeneous task metadata Parquet-safe: ordinary
                # tasks use scalar IDs, while MultiRC/ReCoRD use nested IDs.
                "superglue_metadata": json.dumps(
                    metadata, sort_keys=True, ensure_ascii=False
                ),
                "choice_texts": choices,
                "inference_mode": "choice_loglikelihood",
            })
        return rows

    def _format_chat(self, model_name, prompt):
        messages = []
        if model_name in HAS_SYSTEM_PROMPT_MODELS:
            messages.append({"role": "system", "content": "You are a helpful assistant."})
        messages.append({"role": "user", "content": prompt})
        tokens = self._tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if tokens and self._tokenizer.bos_token_id is not None:
            if tokens[0] == self._tokenizer.bos_token_id:
                tokens = tokens[1:]
        return self._tokenizer.decode(tokens)

    def compute_metrics(self, data):
        column = f"{self.model_name}_choice_loglikelihoods"
        required = {
            column, "superglue_task", "superglue_answer_index",
            "superglue_metadata", "factor",
        }
        missing = sorted(required.difference(data.columns))
        if missing:
            raise KeyError(f"SuperGLUE inference is missing columns: {missing}")
        predictions = []
        for scores in data[column]:
            values = np.asarray(list(scores), dtype=np.float64)
            if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
                raise ValueError("SuperGLUE candidate scores must be finite vectors.")
            predictions.append(int(values.argmax()))
        result = {
            "factor": [],
            "superglue_score": [],
            "raw_superglue_predicted_index": predictions,
            "raw_superglue_gold_index": data["superglue_answer_index"].astype(int).tolist(),
            "raw_superglue_task": data["superglue_task"].astype(str).tolist(),
        }
        metric_names = metric_output_names(set(data["superglue_task"]))
        for name in metric_names:
            result[name] = []
        factors = [float(value) for value in data["factor"]]
        for factor in sorted(set(factors)):
            factor_indices = [i for i, value in enumerate(factors) if value == factor]
            task_scores = []
            values_by_name = {}
            for task in normalize_tasks(data.iloc[factor_indices]["superglue_task"].unique()):
                indices = [i for i in factor_indices if data.iloc[i]["superglue_task"] == task]
                task_predictions = [predictions[i] for i in indices]
                references = data.iloc[indices]["superglue_answer_index"].astype(int).tolist()
                metric_predictions, metric_references = self._metric_payload(
                    task, data.iloc[indices], task_predictions, references
                )
                metrics = self._metrics[task].compute(
                    predictions=metric_predictions, references=metric_references
                )
                for key, value in metrics.items():
                    values_by_name[f"superglue_{task}_{key}"] = float(value)
                score = superglue_task_score(task, metrics)
                values_by_name[f"superglue_{task}_score"] = score
                task_scores.append(score)
            result["factor"].append(factor)
            result["superglue_score"].append(float(np.mean(task_scores)))
            for name in metric_names:
                result[name].append(values_by_name.get(name))
        return result

    @staticmethod
    def _metric_payload(task, frame, predictions, references):
        metadata = [json.loads(value) for value in frame["superglue_metadata"]]
        if task == "multirc":
            return [
                {"idx": item["idx"], "prediction": prediction}
                for item, prediction in zip(metadata, predictions)
            ], references
        if task == "record":
            return [
                {"idx": item["idx"], "prediction_text": item["entities"][prediction]}
                for item, prediction in zip(metadata, predictions)
            ], [
                {"idx": item["idx"], "answers": item["answers"]}
                for item in metadata
            ]
        return predictions, references

    def render_report(self, result, output_dir=None):
        metrics = {"superglue_score": "SuperGLUE score"}
        for task in SUPERGLUE_TASKS:
            key = f"superglue_{task}_score"
            if result.metrics is not None and key in result.metrics.columns:
                metrics[key] = f"{task.upper()} score"
        return self._render_curve_report(
            result, metrics, output_dir, y_limits=(0.0, 1.0), y_axis_label="Score"
        )

    def __str__(self):
        return "SuperGLUEEvaluator"


def normalize_tasks(tasks):
    if isinstance(tasks, str):
        tasks = [tasks]
    tasks = [str(task).lower() for task in tasks]
    unknown = sorted(set(tasks).difference(SUPERGLUE_TASKS))
    if unknown:
        raise ValueError(f"Unknown SuperGLUE tasks: {unknown}")
    if len(tasks) != len(set(tasks)):
        raise ValueError("SuperGLUE tasks must not contain duplicates.")
    return tasks


def _idx(value):
    if isinstance(value, dict):
        return {key: int(item) for key, item in value.items()}
    return int(value)


def format_superglue_row(task, row):
    idx = _idx(row["idx"])
    metadata = {"idx": idx}
    if task == "boolq":
        prompt = f"{str(row['passage']).strip()}\nQuestion: {str(row['question']).strip()}?\nAnswer:"
        return prompt, [" no", " yes"], int(row["label"]), idx, metadata
    if task == "cb":
        prompt = f"{str(row['premise']).strip()}\nQuestion: {str(row['hypothesis']).strip()}. True, False, or Neither?\nAnswer:"
        return prompt, [" True", " False", " Neither"], int(row["label"]), idx, metadata
    if task == "copa":
        connector = {"cause": "because", "effect": "therefore"}[str(row["question"])]
        premise = str(row["premise"]).strip()
        if premise.endswith("."):
            premise = premise[:-1]
        choices = [" " + lower_first(str(row[name]).strip()) for name in ("choice1", "choice2")]
        return f"{premise} {connector}", choices, int(row["label"]), idx, metadata
    if task == "rte":
        prompt = f"{str(row['premise']).strip()}\nQuestion: {str(row['hypothesis']).strip()} True or False?\nAnswer:"
        return prompt, [" True", " False"], int(row["label"]), idx, metadata
    if task == "wic":
        sentence1 = str(row["sentence1"])
        word = sentence1[int(row["start1"]):int(row["end1"])]
        prompt = (
            f"Sentence 1: {sentence1}\nSentence 2: {row['sentence2']}\n"
            f"Question: Is the word '{word}' used in the same way in the two "
            "sentences above?\nAnswer:"
        )
        return prompt, [" no", " yes"], int(row["label"]), idx, metadata
    if task == "wsc":
        raw_passage = str(row["text"])
        span2_index = int(row["span2_index"])
        pre = " ".join(raw_passage.split()[:span2_index])
        span2 = str(row["span2_text"])
        post = raw_passage[len(pre) + len(span2) + 1:]
        passage = general_detokenize(pre + f" *{span2}*" + post)
        prompt = (
            f"Passage: {passage}\nQuestion: In the passage above, does the "
            f"pronoun \"*{row['span2_text']}*\" refer to \"*{row['span1_text']}*\"?\n"
            "Answer:"
        )
        return prompt, [" no", " yes"], int(row["label"]), idx, metadata
    if task == "multirc":
        prompt = (
            f"{str(row['paragraph']).strip()}\nQuestion: {str(row['question']).strip()}\n"
            f"Candidate answer: {str(row['answer']).strip()}\n"
            "Is this candidate answer correct?\nAnswer:"
        )
        # All answers for one question share an input ID so evaluator-level
        # validation/test partitioning can never split an official metric group.
        identity = f"{idx['paragraph']}:{idx['question']}"
        return prompt, [" no", " yes"], int(row["label"]), identity, metadata
    if task == "record":
        initial, *highlights = str(row["passage"]).strip().split("\n@highlight\n")
        prompt = initial + "\n\n" + "".join(f"  - {value}.\n" for value in highlights)
        entities = sorted(set(map(str, row["entities"])))
        answers = sorted(set(map(str, row["answers"])))
        query = str(row["query"])
        choices = ["  - " + query.replace("@placeholder", entity) for entity in entities]
        gold_indices = [entities.index(answer) for answer in answers if answer in entities]
        if not gold_indices:
            raise ValueError(f"ReCoRD example {idx} has no gold entity among candidates.")
        metadata.update({"entities": entities, "answers": answers})
        identity = f"{idx['passage']}:{idx['query']}"
        return prompt, choices, gold_indices[0], identity, metadata
    raise ValueError(f"Unsupported SuperGLUE task: {task}")


def lower_first(text):
    return text[:1].lower() + text[1:]


def general_detokenize(text):
    """Match lm-eval's pinned WSC detokenization helper."""
    text = text.replace(" n't", "n't")
    text = text.replace(" )", ")")
    text = text.replace("( ", "(")
    text = text.replace('" ', '"')
    text = text.replace(' "', '"')
    text = re.sub(r" (['.,])", r"\1", text)
    return text


def metric_output_names(tasks):
    names = []
    metrics = {
        "boolq": ("accuracy",), "cb": ("accuracy", "f1"),
        "copa": ("accuracy",), "multirc": ("exact_match", "f1_m", "f1_a"),
        "record": ("exact_match", "f1"), "rte": ("accuracy",),
        "wic": ("accuracy",), "wsc": ("accuracy",),
    }
    for task in SUPERGLUE_TASKS:
        if task in tasks:
            names.extend(f"superglue_{task}_{name}" for name in metrics[task])
            names.append(f"superglue_{task}_score")
    return names


def superglue_task_score(task, metrics):
    if task == "cb":
        return float(np.mean([metrics["accuracy"], metrics["f1"]]))
    if task == "multirc":
        return float(np.mean([metrics["exact_match"], metrics["f1_a"]]))
    if task == "record":
        return float(np.mean([metrics["exact_match"], metrics["f1"]]))
    return float(metrics["accuracy"])
