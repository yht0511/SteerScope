from types import SimpleNamespace

import pytest
import torch

from steerscope.models.preference_model import PreferenceModel


def _model(*pairs):
    model = PreferenceModel.__new__(PreferenceModel)
    model.preference_pairs = list(pairs)
    model.training_args = SimpleNamespace(batch_size=6)
    return model


def test_preference_batch_size_uses_final_partial_batch():
    model = _model("orig_add")
    batch = {
        "orig_add_winning_input_ids": torch.zeros(4, 3, dtype=torch.long),
        "orig_add_losing_input_ids": torch.zeros(4, 3, dtype=torch.long),
    }

    assert model._actual_preference_batch_size(batch) == 4
    assert (
        model._actual_preference_batch_size(batch) * len(model.preference_pairs)
        == 4
    )
    assert PreferenceModel.training_fingerprint_context()[
        "partial_batch_semantics"
    ] == "actual_collated_batch_size"


def test_preference_batch_size_rejects_inconsistent_pair_fields():
    model = _model("orig_add")
    batch = {
        "orig_add_winning_input_ids": torch.zeros(4, 3, dtype=torch.long),
        "orig_add_losing_input_ids": torch.zeros(3, 3, dtype=torch.long),
    }

    with pytest.raises(ValueError, match="inconsistent sizes"):
        model._actual_preference_batch_size(batch)
