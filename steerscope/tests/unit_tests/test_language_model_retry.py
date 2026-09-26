import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from openai import APITimeoutError

from steerscope.models.language_models import LanguageModel


def completion(content="ok"):
    response = MagicMock()
    response.to_dict.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {"completion_tokens": 1, "prompt_tokens": 2},
    }
    return response


def test_language_model_retries_timeout_then_succeeds(tmp_path):
    request = httpx.Request("POST", "https://example.test/chat")
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=[
        APITimeoutError(request=request),
        completion(),
    ])
    model = LanguageModel(
        "gpt-4o-mini", client, use_cache=False,
        api_max_attempts=3, api_retry_initial_delay=0,
    )
    with patch("steerscope.models.language_models.asyncio.sleep", new=AsyncMock()) as sleep:
        result = asyncio.run(model.chat_completions("test", ["prompt"]))
    assert result == ["ok"]
    assert client.chat.completions.create.await_count == 2
    sleep.assert_awaited_once()


def test_language_model_stops_after_max_attempts():
    request = httpx.Request("POST", "https://example.test/chat")
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=APITimeoutError(request=request)
    )
    model = LanguageModel(
        "gpt-4o-mini", client, use_cache=False,
        api_max_attempts=2, api_retry_initial_delay=0,
    )
    with patch("steerscope.models.language_models.asyncio.sleep", new=AsyncMock()):
        try:
            asyncio.run(model.chat_completions("test", ["prompt"]))
        except APITimeoutError:
            pass
        else:
            raise AssertionError("Expected APITimeoutError")
    assert client.chat.completions.create.await_count == 2


def test_language_model_accepts_completion_without_usage():
    response = MagicMock()
    response.to_dict.return_value = {
        "choices": [{"message": {"content": "Rating: [[2]]"}}],
    }
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    model = LanguageModel("gpt-4o-mini", client, use_cache=False)

    result = asyncio.run(model.chat_completions("judge", ["prompt"]))

    assert result == ["Rating: [[2]]"]
    assert model.stats.total_call == 1
    assert model.stats.total_cache_hit == 0
    assert model.stats.get_total_tokens() == (0, 0)


def test_language_model_accepts_responses_api_usage_names():
    response = MagicMock()
    response.to_dict.return_value = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    model = LanguageModel("gpt-4o-mini", client, use_cache=False)

    assert asyncio.run(model.chat_completions("test", ["prompt"])) == ["ok"]
    assert model.stats.get_total_tokens() == (3, 2)


def test_language_model_reports_usage_after_each_completed_batch():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[completion(), completion(), completion()]
    )
    model = LanguageModel("gpt-4o-mini", client, use_cache=False)
    progress = MagicMock()

    result = asyncio.run(model.chat_completions(
        "judge",
        ["one", "two", "three"],
        batch_size=2,
        progress_callback=progress,
    ))

    assert result == ["ok", "ok", "ok"]
    assert progress.call_count == 2
    first, second = [call.args[0] for call in progress.call_args_list]
    assert first == {
        "total_calls": 2,
        "network_calls": 2,
        "total_cache_hits": 0,
        "input_tokens": 4,
        "output_tokens": 2,
        "total_tokens": 6,
        "total_price": first["total_price"],
    }
    assert second == {
        "total_calls": 3,
        "network_calls": 3,
        "total_cache_hits": 0,
        "input_tokens": 6,
        "output_tokens": 3,
        "total_tokens": 9,
        "total_price": second["total_price"],
    }
    assert first["total_price"] > 0
    assert second["total_price"] > first["total_price"]


def test_cache_identity_stable_on_resume_and_separates_settings():
    client = MagicMock(base_url=httpx.URL('https://one.example/v1'))
    model = LanguageModel('gpt-4o-mini', client, use_cache=False, temperature=0.)
    initial = model._get_cache_key('prompt', 5, 'generate')
    assert model._get_cache_key('prompt', 0, 'generate') == initial
    model.temperature = 1.
    assert model._get_cache_key('prompt', 5, 'generate') != initial
    other = LanguageModel('gpt-4o-mini', MagicMock(base_url=httpx.URL('https://two.example/v1')),
                          use_cache=False, temperature=0.)
    assert other._get_cache_key('prompt', 5, 'generate') != initial


def test_retry_jitter_does_not_advance_data_sampling_rng():
    import random
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=[
        APITimeoutError(request=httpx.Request('POST', 'https://example.test')),
        completion(),
    ])
    model = LanguageModel('gpt-4o-mini', client, use_cache=False, api_retry_initial_delay=0)
    before = random.getstate()
    asyncio.run(model.chat_completions('generate', ['prompt']))
    assert random.getstate() == before
