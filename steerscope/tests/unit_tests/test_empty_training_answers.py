import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from steerscope.utils import prompt_utils


@pytest.mark.parametrize('name', ['response_with_concept', 'response_without_concept',
                                 'continue_with_concept', 'continue_without_concept'])
def test_empty_training_answer_retries_only_failed_pair(name):
    continuation = name.startswith('continue')
    prefix = 'question ' if continuation else ''
    client = SimpleNamespace(chat_completions=AsyncMock(side_effect=[
        [prefix + 'valid', '  ""  '], [prefix + 'repaired'],
    ]))
    tokenizer = SimpleNamespace(tokenize=lambda text: text.split(),
                                convert_tokens_to_string=lambda tokens: ' '.join(tokens))
    result = asyncio.run(getattr(prompt_utils, name)(
        client, tokenizer, ['concept', 'concept'], ['question', 'question'], 10,
    ))
    assert result == ['valid', 'repaired']
    calls = client.chat_completions.await_args_list
    assert len(calls) == 2
    assert calls[1].args[1] == [calls[0].args[1][1]]
    assert calls[1].kwargs == {'refresh_cache': True}


def test_permanently_empty_training_answer_fails_after_three_attempts():
    client = SimpleNamespace(chat_completions=AsyncMock(return_value=[' \n ']))
    with pytest.raises(ValueError, match='after 3 attempts'):
        asyncio.run(prompt_utils.response_with_concept(
            client, SimpleNamespace(tokenize=str.split), ['concept'], ['question'], length=None,
        ))
    assert client.chat_completions.await_count == 3


def test_null_api_content_is_treated_as_empty():
    from steerscope.models.language_models import LanguageModel
    assert LanguageModel.normalize(None, None) == ''
