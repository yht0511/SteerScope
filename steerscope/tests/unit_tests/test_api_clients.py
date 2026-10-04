import pytest

from steerscope.utils.api_clients import openai_client_credentials


@pytest.fixture(autouse=True)
def clear_api_environment(monkeypatch):
    for name in (
        "STEERSCOPE_GENERATION_MODEL",
        "STEERSCOPE_JUDGE_MODEL",
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


def test_api_model_overrides_preserve_roles_and_backbone(monkeypatch):
    from steerscope.utils.api_clients import apply_api_model_overrides

    source = {
        'generate': {'lm_model': 'original-generator', 'model_name': 'gemma'},
        'train': {'models': {
            name: {'lm_model': 'original-generator'}
            for name in ('PromptSteering', 'APSR', 'SPSR')
        }},
        'evaluate': {'lm_model': 'original-judge', 'model_name': 'gemma'},
    }
    monkeypatch.setenv('STEERSCOPE_GENERATION_MODEL', ' provider/generator ')
    monkeypatch.setenv('STEERSCOPE_JUDGE_MODEL', 'provider/judge')
    result = apply_api_model_overrides(source)
    assert result['generate']['lm_model'] == 'provider/generator'
    for recipe in result['train']['models'].values():
        assert recipe['lm_model'] == 'provider/generator'
    assert result['evaluate']['lm_model'] == 'provider/judge'
    assert result['generate']['model_name'] == result['evaluate']['model_name'] == 'gemma'
    assert source['generate']['lm_model'] == 'original-generator'
    monkeypatch.setenv('STEERSCOPE_GENERATION_MODEL', ' ')
    monkeypatch.setenv('STEERSCOPE_JUDGE_MODEL', '')
    assert apply_api_model_overrides(source) == source


def test_api_model_overrides_reach_scheduler_and_script_args(monkeypatch, tmp_path):
    import sys
    import yaml
    from steerscope.sweep.paper.scheduler import load_yaml
    from steerscope.scripts.args.dataset_args import DatasetArgs
    from steerscope.scripts.args.eval_args import EvalArgs
    from steerscope.scripts.args.training_args import TrainingArgs

    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump({
        'generate': {'lm_model': 'old-generator'},
        'train': {'models': {'PromptSteering': {'lm_model': 'old-generator'}}},
        'evaluate': {'lm_model': 'old-judge'},
    }))
    monkeypatch.setenv('STEERSCOPE_GENERATION_MODEL', 'new-generator')
    monkeypatch.setenv('STEERSCOPE_JUDGE_MODEL', 'new-judge')
    monkeypatch.setattr(sys, 'argv', ['test', '--config', str(path)])
    resolved = load_yaml(path)
    assert resolved['generate']['lm_model'] == 'new-generator'
    assert resolved['evaluate']['lm_model'] == 'new-judge'
    assert DatasetArgs(section='generate').lm_model == 'new-generator'
    assert EvalArgs(section='evaluate').lm_model == 'new-judge'
    args = TrainingArgs(section='train')
    assert args.models['PromptSteering'].lm_model == 'new-generator'


def test_unknown_model_price_does_not_break_reporting():
    from steerscope.models.language_models import LanguageModelStats
    from steerscope.evaluators.judge import JudgeEvaluatorMixin

    stats = LanguageModelStats('provider/custom-model')
    report = stats.get_report()
    assert report['total_price'] is None
    stats.print_report()
    assert JudgeEvaluatorMixin._report_delta(report, report)['total_price'] is None
