from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

from steerscope.models.odesteer import (
    ODEKernelClassifier,
    ODESteer,
    ODESteerConfig,
    NormedPolyCountSketch,
    StepODESteer,
    apply_ode_steering,
    ode_transport,
    step_ode_transport,
)


def small_config(**overrides):
    values = {
        "degree": 2,
        "n_components": 64,
        "gamma": 0.1,
        "coef0": 1.0,
        "linear_classifier": "lr",
        "sketch_seed": 42,
        "solver": "euler",
        "steps": 10,
    }
    values.update(overrides)
    return ODESteerConfig(**values)


def fitted_classifier(config=None, hidden_size=6):
    generator = torch.Generator().manual_seed(7)
    positive = torch.randn(24, hidden_size, generator=generator) + 0.4
    negative = torch.randn(24, hidden_size, generator=generator) - 0.4
    return ODEKernelClassifier(config or small_config()).fit(positive, negative)


def training_args(config=None):
    config = config or small_config()
    return SimpleNamespace(
        batch_size=4,
        ode_degree=config.degree,
        ode_n_components=config.n_components,
        ode_gamma=config.gamma,
        ode_coef0=config.coef0,
        ode_linear_classifier=config.linear_classifier,
        ode_sketch_seed=config.sketch_seed,
        ode_solver=config.solver,
        ode_steps=config.steps,
    )


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity()])


class DummyCausalLM(nn.Module):
    def __init__(self, hidden_size=6):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.model = DummyBackbone()


def test_normalized_tensor_sketch_vjp_matches_autograd():
    config = small_config(n_components=31)
    sketch = NormedPolyCountSketch(config)
    generator = torch.Generator().manual_seed(11)
    inputs = torch.randn(3, 5, generator=generator)
    sketch.fit(inputs)
    vector = torch.randn(config.n_components, generator=generator)

    differentiable = inputs.clone().requires_grad_(True)
    objective = (sketch.transform(differentiable) * vector).sum()
    expected = torch.autograd.grad(objective, differentiable)[0]
    actual = sketch.vjp(inputs, vector)

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)


def test_single_euler_step_matches_step_odesteer():
    classifier = fitted_classifier(small_config(steps=1))
    states = torch.randn(5, 6, generator=torch.Generator().manual_seed(13))
    strengths = torch.full((5,), 2.5)

    full = ode_transport(
        states,
        strengths,
        classifier,
        solver="euler",
        steps=1,
    )
    step = step_ode_transport(states, strengths, classifier)

    torch.testing.assert_close(full, step, rtol=1e-5, atol=1e-6)


def test_mask_and_zero_factor_leave_unselected_states_exactly_unchanged():
    classifier = fitted_classifier()
    hidden = torch.randn(2, 3, 6, generator=torch.Generator().manual_seed(17))
    mask = torch.tensor([[False, False, True], [False, True, False]])
    strengths = torch.tensor([0.0, 2.0])

    output, active = apply_ode_steering(
        hidden,
        strengths,
        mask,
        classifier,
        variant="step",
        solver="euler",
        steps=10,
    )

    assert active.tolist() == [[False, False, False], [False, True, False]]
    assert torch.equal(output[0], hidden[0])
    assert torch.equal(output[1, 0], hidden[1, 0])
    assert torch.equal(output[1, 2], hidden[1, 2])
    assert not torch.equal(output[1, 1], hidden[1, 1])


def test_forward_hook_steers_last_prefill_and_cached_generation_tokens():
    config = small_config()
    subject = DummyCausalLM()
    benchmark = StepODESteer(
        subject,
        tokenizer=None,
        layer=0,
        training_args=training_args(config),
        device="cpu",
    )
    benchmark.make_model()
    benchmark.classifier = fitted_classifier(config)
    hidden = torch.randn(1, 3, 6, generator=torch.Generator().manual_seed(19))
    prompt_mask = torch.tensor([[False, False, True]])

    with benchmark._intervention(torch.tensor([1.0]), prompt_mask):
        prefill = subject.model.layers[0](hidden)
        cached = subject.model.layers[0](hidden[:, :1])

    assert torch.equal(prefill[:, :2], hidden[:, :2])
    assert not torch.equal(prefill[:, 2], hidden[:, 2])
    assert not torch.equal(cached, hidden[:, :1])


@pytest.mark.parametrize("model_class", [ODESteer, StepODESteer])
def test_per_concept_checkpoint_round_trip(tmp_path, model_class):
    config = small_config()
    model = DummyCausalLM()
    original = model_class(
        model,
        tokenizer=None,
        layer=0,
        training_args=training_args(config),
        device="cpu",
    )
    original.make_model()
    original.classifier = fitted_classifier(config)
    original.save(tmp_path, concept_id=3)

    restored = model_class(
        DummyCausalLM(),
        tokenizer=None,
        layer=0,
        training_args=training_args(config),
        device="cpu",
    )
    restored.load(tmp_path, concept_id=3)

    states = torch.randn(4, 6, generator=torch.Generator().manual_seed(23))
    torch.testing.assert_close(
        restored.classifier.vector_field(states),
        original.classifier.vector_field(states),
        rtol=0,
        atol=0,
    )
    checkpoint = tmp_path / model_class.artifact_directory / "3" / "classifier.pt"
    assert checkpoint.is_file()
    assert torch.load(checkpoint, weights_only=True)["variant"] == model_class.variant


def test_strength_validation_rejects_non_finite_values():
    examples = pd.DataFrame({"factor": [0.0, float("nan")]})
    with pytest.raises(ValueError, match="finite"):
        ODESteer._strengths(examples, "cpu")
