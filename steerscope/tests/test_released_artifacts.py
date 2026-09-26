"""Checks for generated training data in the four paper profiles."""
import json
import os
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(os.environ.get('STEERSCOPE_RELEASE_ROOT', 'outputs/paper'))
PROFILES = ('2b/l10', '2b/l20', '9b/l20', '9b/l31')


@pytest.mark.parametrize('profile', PROFILES)
def test_paper_training_pool(profile):
    root = ROOT/profile/'generate'
    required = [root/'metadata.jsonl', root/'train_data.parquet']
    if not all(path.is_file() for path in required):
        pytest.skip(f'Generated training data not found: {root}')
    metadata = [json.loads(line) for line in required[0].read_text().splitlines()]
    assert [row['concept_id'] for row in metadata] == list(range(500))
    train = pd.read_parquet(required[1])
    assert set(train.concept_id) == set(range(500))
    counts = train.groupby(['concept_id', 'category']).size()
    assert counts.eq(72).all()
    assert set(train.category) == {'positive', 'negative'}
    assert len(train) == 500*144
    names = {row['concept_id']: row['concept'] for row in metadata}
    for row in train[train.category.eq('positive')].itertuples():
        assert row.output_concept == names[row.concept_id]
    for _, group in train.groupby('concept_id'):
        positive = group[group.category.eq('positive')].input.sort_values().tolist()
        negative = group[group.category.eq('negative')].input.sort_values().tolist()
        assert positive == negative
