"""Shared Alpaca mechanics for evaluators that explicitly choose Alpaca data."""

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


class AlpacaEvaluator(Evaluator):
    """Evaluator base whose subclasses deliberately use AlpacaEval prompts."""

    dataset_type = "AlpacaEval"

    def __init__(self, node, context, **params):
        super().__init__(node, context, **params)
        self._dataset_tokenizer = None

    @classmethod
    def execution_context(cls, node, args):
        require_dataset_type(node.dataset, cls.dataset_type)
        require_num_examples(node.dataset)
        data_root = getattr(args, "master_data_dir", None)
        path = Path(data_root) / "alpaca_eval.json" if data_root else Path()
        return {
            **super().execution_context(node, args),
            "alpaca_eval": file_signature(path) if data_root else None,
            "data_type": node.dataset.get(
                "data_type", getattr(args, "steer_data_type", "concept")
            ),
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
        self._dataset_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            model_max_length=max_length,
        )
        self._dataset_tokenizer.padding_side = "right"

    def close_resources(self) -> None:
        self._dataset_tokenizer = None
        super().close_resources()

    def build_dataset(self, model, factors) -> pd.DataFrame:
        config = dict(self.node.dataset)
        require_dataset_type(config, self.dataset_type)
        num_examples = require_num_examples(config)
        source = self._load_source(
            num_examples,
            model.concept.concept_id,
            config.get("seed", getattr(self.args, "seed", 42)),
        )
        concept = model.concept.text
        rows = []
        for input_id, row in source.reset_index(drop=True).iterrows():
            instruction = str(row["instruction"])
            rows.append({
                "dataset_name": self.dataset_type,
                "concept_id": model.concept.concept_id,
                "input_concept": concept,
                "input_id": input_id,
                "source_input_id": int(row["_source_input_id"]),
                "original_prompt": instruction,
                "raw_input": instruction,
                "input": self._format_chat(model.target.base_model, instruction),
                "suppress_original": "",
                "suppress_rewrite": "",
                "steered_prompt": "",
                "defense": [],
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

    def _load_source(
        self,
        num_examples: int,
        concept_id: int,
        seed: int,
    ) -> pd.DataFrame:
        if not self.args.master_data_dir:
            raise ValueError("AlpacaEval requires evaluate.master_data_dir.")
        path = Path(self.args.master_data_dir) / "alpaca_eval.json"
        if not path.exists():
            raise FileNotFoundError(f"AlpacaEval data not found at {path}.")
        data = pd.read_json(path)
        if "instruction" not in data:
            raise ValueError("AlpacaEval data must contain an instruction column.")
        if num_examples > len(data):
            raise ValueError(
                f"AlpacaEval requested {num_examples} examples, but only "
                f"{len(data)} are available."
            )
        sampled = data.sample(
            n=num_examples,
            random_state=concept_seed(seed, concept_id, self.dataset_type),
        )
        sampled = sampled.copy()
        sampled["_source_input_id"] = sampled.index.astype(int)
        return sampled

    def _format_chat(self, model_name: str, prompt: str) -> str:
        messages = []
        if model_name in HAS_SYSTEM_PROMPT_MODELS:
            messages.append({"role": "system", "content": "You are a helpful assistant."})
        messages.append({"role": "user", "content": prompt})
        tokens = self._dataset_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        bos_token_id = self._dataset_tokenizer.bos_token_id
        if tokens and bos_token_id is not None and tokens[0] == bos_token_id:
            tokens = tokens[1:]
        return self._dataset_tokenizer.decode(tokens)
