import json
import pickle

import pandas as pd

from steerscope.scripts.generate import METADATA_FILE, STATE_FILE, save


def test_concept_save_is_idempotent_and_commits_state_last(tmp_path):
    first = pd.DataFrame({
        "concept_id": [0, 0],
        "value": ["stale-a", "stale-b"],
    })
    replacement = pd.DataFrame({
        "concept_id": [0, 0],
        "value": ["fresh-a", "fresh-b"],
    })

    save(
        tmp_path, {"concept_id": 1}, 0, "concept", {"concept": ["text"]},
        "ref", "train", first,
    )
    save(
        tmp_path, {"concept_id": 1}, 0, "concept", {"concept": ["text"]},
        "ref", "train", replacement,
    )

    metadata = [
        json.loads(line)
        for line in (tmp_path / METADATA_FILE).read_text().splitlines()
    ]
    assert [entry["concept_id"] for entry in metadata] == [0]
    frame = pd.read_parquet(tmp_path / "train_data.parquet")
    assert frame.to_dict("records") == replacement.to_dict("records")
    with (tmp_path / STATE_FILE).open("rb") as file:
        assert pickle.load(file) == {"concept_id": 1}


def test_concept_selection_survives_source_changes(tmp_path):
    from steerscope.scripts.generate import frozen_training_concepts
    source = tmp_path/'concepts.csv'
    source.write_text('first,ref1\nsecond,ref2\nthird,ref3\n')
    output = tmp_path/'generate'
    output.mkdir()
    before = frozen_training_concepts(source, output, 42, 2)
    source.write_text('changed,ref4\n')
    assert frozen_training_concepts(source, output, 42, 2) == before
    import pytest
    with pytest.raises(ValueError, match='settings changed'):
        frozen_training_concepts(source, output, 43, 2)


def test_prompt_sampling_independent_of_concept_order_and_global_rng():
    import random
    from datasets import Dataset
    from steerscope.utils.prompt_utils import get_random_content
    data = {'text_train': Dataset.from_dict({'input': [f'prompt {i}' for i in range(1000)]})}
    def sample(concept):
        return get_random_content(data, None, 72, ['text'], [concept], None, 'train',
                                  seed=42, return_indices=True)
    expected = sample('second')
    sample('first')
    for _ in range(100):
        random.uniform(0, 1)
    assert sample('second') == expected
    assert sample('first')[1] != expected[1]
