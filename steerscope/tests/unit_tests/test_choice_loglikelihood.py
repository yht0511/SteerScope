from types import SimpleNamespace

import pandas as pd
import torch
from transformers import BatchEncoding

from steerscope.models.model import Model


class CharacterTokenizer:
    pad_token_id = 0
    padding_side = "right"
    truncation_side = "right"
    model_max_length = 128

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, values, **kwargs):
        return "".join(chr(int(value)) for value in values)

    def __call__(self, texts, return_tensors, padding, truncation):
        rows = [self.encode(text) for text in texts]
        if truncation:
            rows = [row[-self.model_max_length:] for row in rows]
        width = max(map(len, rows))
        padded, masks = [], []
        for row in rows:
            amount = width - len(row)
            if self.padding_side == "left":
                padded.append([0] * amount + row)
                masks.append([0] * amount + [1] * len(row))
            else:
                padded.append(row + [0] * amount)
                masks.append([1] * len(row) + [0] * amount)
        return BatchEncoding({
            "input_ids": torch.tensor(padded),
            "attention_mask": torch.tensor(masks),
        })


class UniformChoiceModel(Model):
    def __init__(self):
        self.tokenizer = CharacterTokenizer()
        self.device = torch.device("cpu")
        self.vocabulary_size = 128

    def prepare_choice_logits(self, examples, **kwargs):
        pass

    def finish_choice_logits(self, **kwargs):
        pass

    def choice_forward(self, inputs, batch_examples, **kwargs):
        shape = (*inputs["input_ids"].shape, self.vocabulary_size)
        logits = torch.zeros(shape)
        keep = kwargs.get("choice_logits_to_keep")
        if keep is not None:
            logits = logits[:, -int(keep):]
        return SimpleNamespace(logits=logits), batch_examples["factor"].tolist()


def test_choice_loglikelihood_scores_all_candidate_tokens():
    model = UniformChoiceModel()
    examples = pd.DataFrame({
        "input": ["Prompt:"],
        "choice_texts": [[" a", " bb"]],
        "factor": [2.0],
    })
    result = model.predict_choice_loglikelihoods(
        examples, batch_size=2, show_progress=False
    )
    unit = -torch.log(torch.tensor(128.0)).item()
    assert result["choice_loglikelihoods"][0] == pytest.approx([2 * unit, 3 * unit])
    assert result["choice_mean_loglikelihoods"][0] == pytest.approx([unit, unit])
    assert result["strength"] == [2.0]


def test_choice_loglikelihood_rejects_ambiguous_boundary():
    model = UniformChoiceModel()
    examples = pd.DataFrame({
        "input": ["Prompt:"], "choice_texts": [["yes", " no"]], "factor": [1.0]
    })
    with pytest.raises(ValueError, match="start with whitespace"):
        model.predict_choice_loglikelihoods(examples, show_progress=False)


import pytest
