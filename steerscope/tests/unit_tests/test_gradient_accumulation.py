import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch

from steerscope.models.flas import FLAS
from steerscope.models.hypersteer import HyperSteer
from steerscope.models.lora import LoRA
from steerscope.models.lsreft import LsReFT
from steerscope.models.preference_model import PreferenceModel
from steerscope.models.probe import LinearProbe
from steerscope.models.psr import APSR, SPSR
from steerscope.models.reft import LoReFT
from steerscope.models.sft import SFT, _fsdp_gradient_accumulation_steps
from steerscope.utils.training import (
    GRADIENT_ACCUMULATION_SEMANTICS,
    accumulation_window_size,
    is_optimizer_step,
    normalize_loss_for_accumulation,
    optimizer_steps_per_epoch,
)


@pytest.mark.parametrize(
    ("num_microbatches", "accumulation_steps", "expected_updates"),
    [(0, 8, 0), (1, 8, 1), (8, 8, 1), (9, 8, 2), (25, 12, 3)],
)
def test_optimizer_steps_per_epoch_includes_tail(
    num_microbatches, accumulation_steps, expected_updates
):
    assert (
        optimizer_steps_per_epoch(num_microbatches, accumulation_steps)
        == expected_updates
    )


@pytest.mark.parametrize(
    ("dataset_size", "batch_size", "world_size", "configured", "expected"),
    [
        (3, 1, 2, 36, 2),
        (6, 1, 2, 36, 3),
        (12, 1, 2, 36, 6),
        (36, 1, 2, 36, 18),
        (72, 1, 2, 36, 36),
        (73, 2, 2, 36, 19),
        (144, 2, 2, 12, 12),
    ],
)
def test_fsdp_sft_caps_accumulation_at_per_rank_epoch_length(
    dataset_size, batch_size, world_size, configured, expected
):
    assert _fsdp_gradient_accumulation_steps(
        dataset_size,
        batch_size,
        world_size,
        configured,
    ) == expected


def test_epoch_local_window_sizes_and_update_boundaries():
    sizes = [accumulation_window_size(i, 25, 12) for i in range(25)]
    boundaries = [is_optimizer_step(i, 25, 12) for i in range(25)]

    assert sizes == [12] * 24 + [1]
    assert [i for i, boundary in enumerate(boundaries) if boundary] == [11, 23, 24]
    assert [accumulation_window_size(i, 5, 8) for i in range(5)] == [5] * 5


def test_every_tail_microbatch_uses_the_actual_tail_divisor():
    values = [
        normalize_loss_for_accumulation(torch.tensor(12.0), i, 10, 4).item()
        for i in range(10)
    ]
    assert values == [3.0] * 8 + [6.0] * 2


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ((1, 0), ValueError),
        ((-1, 1), ValueError),
        ((1.5, 1), TypeError),
    ],
)
def test_optimizer_step_validation(args, error):
    with pytest.raises(error):
        optimizer_steps_per_epoch(*args)


def _run_accumulated_updates(targets, accumulation_steps):
    parameter = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=0.07)
    gradients = []
    for step, target in enumerate(targets):
        loss = (parameter - target).square()
        normalize_loss_for_accumulation(
            loss, step, len(targets), accumulation_steps
        ).backward()
        if is_optimizer_step(step, len(targets), accumulation_steps):
            gradients.append(parameter.grad.detach().clone())
            optimizer.step()
            optimizer.zero_grad()
    return parameter.detach(), gradients


def _run_explicit_window_means(targets, accumulation_steps):
    parameter = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=0.07)
    gradients = []
    for start in range(0, len(targets), accumulation_steps):
        window = targets[start:start + accumulation_steps]
        torch.stack([(parameter - target).square() for target in window]).mean().backward()
        gradients.append(parameter.grad.detach().clone())
        optimizer.step()
        optimizer.zero_grad()
    return parameter.detach(), gradients


def test_tail_gradients_match_explicit_window_means_numerically():
    targets = [
        torch.tensor(value, dtype=torch.float64)
        for value in (1.0, -0.5, 2.0, 0.75, -1.25)
    ]
    accumulated_parameter, accumulated_gradients = _run_accumulated_updates(
        targets, accumulation_steps=3
    )
    reference_parameter, reference_gradients = _run_explicit_window_means(
        targets, accumulation_steps=3
    )

    assert torch.equal(accumulated_parameter, reference_parameter)
    assert len(accumulated_gradients) == len(reference_gradients) == 2
    for actual, expected in zip(accumulated_gradients, reference_gradients):
        assert torch.equal(actual, expected)


def test_loreft_normalizes_before_backward():
    tree = ast.parse(textwrap.dedent(inspect.getsource(LoReFT.train)))
    normalize_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "normalize_loss_for_accumulation"
    ]
    backward_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "backward"
    ]
    assert len(normalize_lines) == len(backward_lines) == 1
    assert normalize_lines[0] < backward_lines[0]


@pytest.mark.parametrize(
    "model_cls",
    [SFT, LoRA, LoReFT, LsReFT, HyperSteer, PreferenceModel, SPSR, APSR, LinearProbe],
)
def test_affected_method_fingerprints_reject_old_accumulation_artifacts(model_cls):
    assert model_cls.training_fingerprint_context()[
        "gradient_accumulation_semantics"
    ] == GRADIENT_ACCUMULATION_SEMANTICS


@pytest.mark.parametrize(
    ("training_loop", "normalization_call"),
    [
        (SFT._train, "normalize_loss_for_accumulation"),
        (LoRA.train, "normalize_loss_for_accumulation"),
        (LoReFT.train, "normalize_loss_for_accumulation"),
        (LsReFT.train, "normalize_loss_for_accumulation"),
        (HyperSteer.train, "normalize_loss_for_accumulation"),
        (PreferenceModel.train, "_normalize_preference_minibatch_loss"),
        (SPSR.train, "normalize_loss_for_accumulation"),
        (LinearProbe.train, "normalize_loss_for_accumulation"),
    ],
)
def test_handwritten_loops_use_the_shared_tail_window_contract(
    training_loop, normalization_call
):
    source = inspect.getsource(training_loop)
    assert "optimizer_steps_per_epoch" in source
    assert normalization_call in source
    assert "is_optimizer_step" in source


def test_flas_keeps_its_released_accumulation_semantics():
    assert "gradient_accumulation_semantics" not in FLAS.training_fingerprint_context()


def test_hypersteer_sets_distributed_sampler_epoch_once_per_epoch(monkeypatch):
    class SamplerSpy:
        def __init__(self):
            self.epochs = []

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

    sampler = SamplerSpy()
    model = HyperSteer.__new__(HyperSteer)
    model.concept_embedding = torch.nn.Linear(1, 1)
    model.hypernet_tokenizer = object()
    model.device = "cpu"
    model.training_args = SimpleNamespace(
        lr=1e-3,
        weight_decay=0.0,
        n_epochs=3,
        gradient_accumulation_steps=2,
    )
    model.make_dataloader = lambda *args, **kwargs: ([], sampler)

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    model.train([], world_size=1)

    assert sampler.epochs == [0, 1, 2]


def test_preference_inner_and_outer_normalization_is_exact():
    parameter = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
    per_example = torch.stack(
        [(parameter - target).square() for target in (1.0, -0.5, 2.0)]
    )

    # Simulate two internal minibatches (2 + 1 examples) in the final outer
    # accumulation window.  That final outer window contains one microbatch.
    PreferenceModel._normalize_preference_minibatch_loss(
        per_example[:2],
        expanded_batch_size=3,
        outer_step=2,
        num_outer_microbatches=3,
        accumulation_steps=2,
    ).backward(retain_graph=True)
    PreferenceModel._normalize_preference_minibatch_loss(
        per_example[2:],
        expanded_batch_size=3,
        outer_step=2,
        num_outer_microbatches=3,
        accumulation_steps=2,
    ).backward()
    actual_gradient = parameter.grad.detach().clone()

    reference = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
    torch.stack(
        [(reference - target).square() for target in (1.0, -0.5, 2.0)]
    ).mean().backward()
    assert torch.equal(actual_gradient, reference.grad)


def test_linear_probe_honors_configured_accumulation(monkeypatch):
    import steerscope.models.probe as probe_module

    class DummyIntervenable:
        def __init__(self, parameter):
            self.parameter = parameter
            self.full_intervention_outputs = []

        def __call__(self, *args, **kwargs):
            latent = self.parameter.reshape(1, 1).expand(1, 2)
            self.full_intervention_outputs = [
                SimpleNamespace(latent=(latent, None))
            ]
            return None, None

    class SchedulerSpy:
        def __init__(self):
            self.steps = 0

        def step(self):
            self.steps += 1

    optimizers = []
    schedulers = []

    class OptimizerSpy(torch.optim.SGD):
        def __init__(self, params, **kwargs):
            super().__init__(params, lr=kwargs["lr"], weight_decay=kwargs["weight_decay"])
            self.steps = 0
            optimizers.append(self)

        def step(self, closure=None):
            self.steps += 1
            return super().step(closure)

    def scheduler_factory(*args, **kwargs):
        assert kwargs["num_training_steps"] == 2
        scheduler = SchedulerSpy()
        schedulers.append(scheduler)
        return scheduler

    parameter_module = torch.nn.Linear(1, 1, bias=False)
    model = LinearProbe.__new__(LinearProbe)
    model.ax = parameter_module
    model.ax_model = DummyIntervenable(parameter_module.weight[0, 0])
    model.device = "cpu"
    model.training_args = SimpleNamespace(
        lr=1e-2,
        weight_decay=0.0,
        n_epochs=1,
        gradient_accumulation_steps=8,
        topk=1,
        coeff_l1_loss=0.0,
    )
    batch = {
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "intervention_locations": torch.zeros(1, 1, 1, dtype=torch.long),
        "intervention_masks": torch.ones(1, 2, dtype=torch.long),
        "labels": torch.ones(1, dtype=torch.long),
    }
    model.make_dataloader = lambda *args, **kwargs: [batch for _ in range(10)]

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.optim, "AdamW", OptimizerSpy)
    monkeypatch.setattr(probe_module, "get_scheduler", scheduler_factory)
    monkeypatch.setattr(probe_module, "set_decoder_norm_to_unit_norm", lambda *a: None)
    monkeypatch.setattr(
        probe_module,
        "remove_gradient_parallel_to_decoder_directions",
        lambda *a: None,
    )

    model.train([])

    assert len(optimizers) == len(schedulers) == 1
    assert optimizers[0].steps == 2
    assert schedulers[0].steps == 2
