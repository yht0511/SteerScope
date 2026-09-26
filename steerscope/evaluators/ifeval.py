"""Evaluator for Google's Instruction-Following Evaluation benchmark."""

import json
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


IFEVAL_DATASET = "IFEval"
IFEVAL_DATASET_FILE = "IFEval.parquet"


class IFEvalEvaluator(Evaluator):
    """Generate IFEval responses and apply the pinned official checkers."""

    source_dependencies = (
        "evaluators/ifeval_official/instructions.py",
        "evaluators/ifeval_official/instructions_registry.py",
        "evaluators/ifeval_official/instructions_util.py",
        "evaluators/ifeval_official/UPSTREAM.md",
    )

    def __init__(self, node, context, **params):
        super().__init__(node, context, **params)
        self._tokenizer = None

    @classmethod
    def execution_context(cls, node, args):
        require_dataset_type(node.dataset, IFEVAL_DATASET)
        require_num_examples(node.dataset)
        data_root = getattr(args, "master_data_dir", None)
        path = Path(data_root) / "ifeval" / IFEVAL_DATASET_FILE if data_root else None
        return {
            **super().execution_context(node, args),
            "dataset": file_signature(path) if path else None,
            "default_split_ratio": getattr(args, "winrate_split_ratio", 0.5),
        }

    def open_resources(self, models) -> None:
        super().open_resources(models)
        self._validate_checker_runtime()
        model_name = getattr(self.args, "steering_model_name", None) or getattr(
            self.args, "model_name", None
        )
        if not model_name:
            raise ValueError(f"Evaluator '{self.node_id}' requires a base model name.")
        max_length = 128000 if "google/gemma-3" in model_name else 8192
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, model_max_length=max_length
        )
        self._tokenizer.padding_side = "right"

    @staticmethod
    def _validate_checker_runtime():
        try:
            from .ifeval_official import instructions_registry  # noqa: F401
            import nltk
        except ImportError as error:
            raise RuntimeError(
                "IFEval checker dependencies are missing. Install the project dependencies."
            ) from error
        missing = []
        for package, resource in (
            ("punkt", "tokenizers/punkt"),
            ("punkt_tab", "tokenizers/punkt_tab"),
        ):
            try:
                nltk.data.find(resource)
            except LookupError:
                missing.append(package)
        if missing:
            raise RuntimeError(
                "IFEval requires NLTK resources before generation. Run: "
                f"python -m nltk.downloader {' '.join(missing)}"
            )

    def close_resources(self) -> None:
        self._tokenizer = None
        super().close_resources()

    def build_dataset(self, model, factors) -> pd.DataFrame:
        config = dict(self.node.dataset)
        require_dataset_type(config, IFEVAL_DATASET)
        num_examples = require_num_examples(config)
        source = self._load_source()
        if num_examples > len(source):
            raise ValueError(
                f"IFEval requested {num_examples} examples, but only {len(source)} are available."
            )
        sampled = source.sample(
            n=num_examples,
            random_state=concept_seed(
                config.get("seed", getattr(self.args, "seed", 42)),
                model.concept.concept_id,
                IFEVAL_DATASET,
            ),
        ).sort_values("key", kind="stable")
        rows = []
        for _, row in sampled.iterrows():
            prompt = str(row["prompt"])
            rows.append({
                "dataset_name": IFEVAL_DATASET,
                "concept_id": model.concept.concept_id,
                "input_concept": model.concept.text,
                "input_id": int(row["key"]),
                "original_prompt": prompt,
                "raw_input": prompt,
                "input": self._format_chat(model.target.base_model, prompt),
                "ifeval_instruction_ids_json": str(row["instruction_id_list_json"]),
                "ifeval_kwargs_json": str(row["kwargs_json"]),
            })
        examples = expand_factors(pd.DataFrame(rows), factors)
        return split_by_input_id(
            examples,
            config.get("split", "all"),
            config.get("split_ratio", getattr(self.args, "winrate_split_ratio", 0.5)),
        )

    def _load_source(self):
        if not getattr(self.args, "master_data_dir", None):
            raise ValueError("IFEval requires evaluate.master_data_dir.")
        path = Path(self.args.master_data_dir) / "ifeval" / IFEVAL_DATASET_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"IFEval data not found at {path}. Run steerscope/data/download-ifeval.py first."
            )
        data = pd.read_parquet(path)
        required = {"key", "prompt", "instruction_id_list_json", "kwargs_json"}
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(f"IFEval data at {path} is missing columns: {missing}")
        return data

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
        generation_column = f"{self.model_name}_steered_generation"
        required = {
            generation_column, "original_prompt", "ifeval_instruction_ids_json",
            "ifeval_kwargs_json", "factor",
        }
        missing = sorted(required.difference(data.columns))
        if missing:
            raise KeyError(f"IFEval inference is missing columns: {missing}")

        strict_lists = []
        loose_lists = []
        instruction_ids_per_row = []
        for _, row in data.iterrows():
            instruction_ids = self._decode_list(
                row["ifeval_instruction_ids_json"], "instruction_id_list"
            )
            kwargs = self._decode_list(row["ifeval_kwargs_json"], "kwargs")
            if len(instruction_ids) != len(kwargs):
                raise ValueError("IFEval instruction_id_list and kwargs lengths differ.")
            response = str(row[generation_column])
            prompt = str(row["original_prompt"])
            strict_lists.append(
                self._check_instructions(instruction_ids, kwargs, prompt, [response])
            )
            loose_lists.append(
                self._check_instructions(
                    instruction_ids, kwargs, prompt, self._loose_responses(response)
                )
            )
            instruction_ids_per_row.append(instruction_ids)

        factors = [float(value) for value in data["factor"]]
        result = {
            "factor": [],
            "ifeval_prompt_strict_accuracy": [],
            "ifeval_instruction_strict_accuracy": [],
            "ifeval_prompt_loose_accuracy": [],
            "ifeval_instruction_loose_accuracy": [],
            "ifeval_num_prompts": [],
            "ifeval_num_instructions": [],
            "raw_ifeval_instruction_ids": instruction_ids_per_row,
            "raw_ifeval_strict_follow_instruction_list": strict_lists,
            "raw_ifeval_loose_follow_instruction_list": loose_lists,
            "raw_ifeval_strict_follow_all": [all(values) for values in strict_lists],
            "raw_ifeval_loose_follow_all": [all(values) for values in loose_lists],
        }
        for factor in sorted(set(factors)):
            indices = [index for index, value in enumerate(factors) if value == factor]
            strict = [strict_lists[index] for index in indices]
            loose = [loose_lists[index] for index in indices]
            result["factor"].append(factor)
            result["ifeval_prompt_strict_accuracy"].append(
                sum(all(values) for values in strict) / len(strict)
            )
            result["ifeval_instruction_strict_accuracy"].append(
                self._instruction_accuracy(strict)
            )
            result["ifeval_prompt_loose_accuracy"].append(
                sum(all(values) for values in loose) / len(loose)
            )
            result["ifeval_instruction_loose_accuracy"].append(
                self._instruction_accuracy(loose)
            )
            result["ifeval_num_prompts"].append(len(indices))
            result["ifeval_num_instructions"].append(sum(map(len, strict)))
        return result

    @staticmethod
    def _decode_list(value, name):
        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid IFEval {name} JSON.") from error
        if not isinstance(decoded, list):
            raise ValueError(f"IFEval {name} must decode to a list.")
        return decoded

    @staticmethod
    def _check_instructions(instruction_ids, kwargs, prompt, responses):
        try:
            from .ifeval_official.instructions_registry import INSTRUCTION_DICT
        except ImportError as error:
            raise RuntimeError(
                "IFEval checker dependencies are missing. Install the project dependencies."
            ) from error
        followed = []
        for index, instruction_id in enumerate(instruction_ids):
            try:
                instruction_class = INSTRUCTION_DICT[instruction_id]
            except KeyError as error:
                raise ValueError(f"Unknown IFEval instruction id '{instruction_id}'.") from error
            instruction = instruction_class(instruction_id)
            instruction.build_description(**kwargs[index])
            arguments = instruction.get_instruction_args()
            if arguments and "prompt" in arguments:
                instruction.build_description(prompt=prompt)
            try:
                followed.append(any(
                    response.strip() and instruction.check_following(response)
                    for response in responses
                ))
            except LookupError as error:
                raise RuntimeError(
                    "IFEval requires the NLTK punkt resources. Run: "
                    "python -m nltk.downloader punkt punkt_tab"
                ) from error
        return followed

    @staticmethod
    def _loose_responses(response):
        lines = response.split("\n")
        remove_first = "\n".join(lines[1:]).strip()
        remove_last = "\n".join(lines[:-1]).strip()
        remove_both = "\n".join(lines[1:-1]).strip()
        variants = [response, remove_first, remove_last, remove_both]
        return [*variants, *(variant.replace("*", "") for variant in variants)]

    @staticmethod
    def _instruction_accuracy(values):
        total = sum(map(len, values))
        if total == 0:
            raise ValueError("IFEval examples must contain at least one instruction.")
        return sum(sum(row) for row in values) / total

    def render_report(self, result, output_dir=None):
        return self._render_curve_report(
            result,
            {
                "ifeval_prompt_strict_accuracy": "Prompt accuracy (strict)",
                "ifeval_instruction_strict_accuracy": "Instruction accuracy (strict)",
                "ifeval_prompt_loose_accuracy": "Prompt accuracy (loose)",
                "ifeval_instruction_loose_accuracy": "Instruction accuracy (loose)",
            },
            output_dir,
            columns=2,
            y_limits=(0.0, 1.0),
            y_axis_label="Accuracy",
        )

    def __str__(self):
        return "IFEvalEvaluator"
