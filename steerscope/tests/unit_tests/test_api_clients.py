import pytest

from steerscope.utils.api_clients import openai_client_credentials


@pytest.fixture(autouse=True)
def clear_api_environment(monkeypatch):
    for name in (
        "STEERSCOPE_GENERATION_API_KEY",
        "STEERSCOPE_GENERATION_BASE_URL",
        "STEERSCOPE_JUDGE_API_KEY",
        "STEERSCOPE_JUDGE_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_role_specific_credentials_are_isolated(monkeypatch):
    monkeypatch.setenv("STEERSCOPE_GENERATION_API_KEY", "generation-key")
    monkeypatch.setenv("STEERSCOPE_GENERATION_BASE_URL", "https://generation.example/v1")
    monkeypatch.setenv("STEERSCOPE_JUDGE_API_KEY", "judge-key")
    monkeypatch.setenv("STEERSCOPE_JUDGE_BASE_URL", "https://judge.example/v1")

    assert openai_client_credentials("generation") == {
        "api_key": "generation-key",
        "base_url": "https://generation.example/v1",
    }
    assert openai_client_credentials("judge") == {
        "api_key": "judge-key",
        "base_url": "https://judge.example/v1",
    }


def test_standard_openai_variables_are_fallbacks(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://fallback.example/v1")

    assert openai_client_credentials("generation") == {
        "api_key": "fallback-key",
        "base_url": "https://fallback.example/v1",
    }
    assert openai_client_credentials("judge") == {
        "api_key": "fallback-key",
        "base_url": "https://fallback.example/v1",
    }


def test_role_specific_values_take_precedence(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://fallback.example/v1")
    monkeypatch.setenv("STEERSCOPE_JUDGE_API_KEY", "judge-key")
    monkeypatch.setenv("STEERSCOPE_JUDGE_BASE_URL", "https://judge.example/v1")

    assert openai_client_credentials("judge") == {
        "api_key": "judge-key",
        "base_url": "https://judge.example/v1",
    }


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError, match="Unknown API role"):
        openai_client_credentials("training")
