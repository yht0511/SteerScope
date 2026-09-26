import copy
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from transformers import Gemma2Config
from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM

from steerscope.models.psr import APSR, SPSR, _PSRTeacherCacheEntry


def _tiny_gemma():
    config = Gemma2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        sliding_window=16,
        max_position_embeddings=32,
        attention_dropout=0.0,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
    )
    return Gemma2ForCausalLM(config).eval()


def _training_args(*, batch_size=1, n_epochs=2):
    return SimpleNamespace(
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        n_epochs=n_epochs,
        lr=1e-3,
        weight_decay=1e-6,
        lm_model="unused",
        prompt_temperature=0.0,
    )


def _psr(model_cls, subject, *, layer=1, batch_size=1, n_epochs=2):
    instance = model_cls(
        subject,
        tokenizer=object(),
        layer=layer,
        training_args=_training_args(
            batch_size=batch_size,
            n_epochs=n_epochs,
        ),
        lm_model_name="google/gemma-2-2b-it",
        device="cpu",
        seed=42,
        dump_dir="unused",
    )
    instance.make_model()
    return instance


def _batch():
    # The two views have different prefixes and the same three response tokens.
    base_ids = torch.tensor([[1, 10, 11, 20, 21, 22]])
    prompt_ids = torch.tensor([[1, 30, 31, 32, 33, 20, 21, 22]])
    return {
        "record_ids": torch.tensor([0]),
        "base": {
            "input_ids": base_ids,
            "attention_mask": torch.ones_like(base_ids),
            "response_mask": torch.tensor(
                [[False, False, False, True, True, True]]
            ),
        },
        "prompt": {
            "input_ids": prompt_ids,
            "attention_mask": torch.ones_like(prompt_ids),
            "response_mask": torch.tensor(
                [[False, False, False, False, False, True, True, True]]
            ),
        },
    }


def _assert_module_tensors_equal(left, right):
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def _assert_nested_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right):
            _assert_nested_equal(left_value, right_value)
    else:
        assert left == right


@pytest.mark.parametrize("model_cls", [SPSR, APSR])
def test_teacher_cache_is_step_and_optimizer_exact(model_cls):
    torch.manual_seed(7)
    subject = _tiny_gemma()

    torch.manual_seed(11)
    uncached = _psr(model_cls, copy.deepcopy(subject))
    torch.manual_seed(11)
    cached = _psr(model_cls, copy.deepcopy(subject))
    batch = _batch()

    old_initial, old_per_example = uncached._imitation_loss(
        batch, requires_grad=False
    )
    teacher_cache = {}
    new_initial, new_per_example = cached._imitation_loss(
        batch,
        requires_grad=False,
        teacher_cache=teacher_cache,
        populate_teacher_cache=True,
    )
    assert torch.equal(old_initial, new_initial)
    assert torch.equal(old_per_example, new_per_example)
    assert len(teacher_cache) == 1
    entry = teacher_cache[0]
    assert entry.residuals.device.type == "cpu"
    assert entry.residuals.dtype == next(subject.parameters()).dtype
    assert not entry.residuals.requires_grad
    assert entry.residuals.grad_fn is None

    expected_cached_layers = 2 if model_cls is SPSR else 3
    assert entry.residuals.shape[:2] == (expected_cached_layers, 3)
    if model_cls is SPSR:
        assert entry.prefix_loss.item() > 0
    else:
        assert entry.prefix_loss.item() == 0

    uncached.model.train()
    cached.model.train()
    old_optimizer = torch.optim.AdamW(
        uncached.psr_modules.parameters(), lr=1e-3, weight_decay=1e-6
    )
    new_optimizer = torch.optim.AdamW(
        cached.psr_modules.parameters(), lr=1e-3, weight_decay=1e-6
    )

    for _ in range(2):
        old_optimizer.zero_grad()
        new_optimizer.zero_grad()
        old_loss, _ = uncached._imitation_loss(batch, requires_grad=True)
        new_loss, _ = cached._imitation_loss(
            {"record_ids": batch["record_ids"], "base": batch["base"]},
            requires_grad=True,
            teacher_cache=teacher_cache,
        )
        assert torch.equal(old_loss, new_loss)
        old_loss.backward()
        new_loss.backward()
        _assert_module_tensors_equal(
            {
                name: parameter.grad
                for name, parameter in uncached.psr_modules.named_parameters()
            },
            {
                name: parameter.grad
                for name, parameter in cached.psr_modules.named_parameters()
            },
        )
        old_optimizer.step()
        new_optimizer.step()
        _assert_module_tensors_equal(
            uncached.psr_modules.state_dict(),
            cached.psr_modules.state_dict(),
        )
        _assert_nested_equal(
            old_optimizer.state_dict(), new_optimizer.state_dict()
        )


def test_teacher_forward_runs_once_when_cache_is_reused(monkeypatch):
    torch.manual_seed(17)
    psr = _psr(SPSR, _tiny_gemma())
    counts = {"teacher": 0, "student": 0}
    original = psr._capture_branch

    def counted(branch, *, intervene, requires_grad, capture_layers=None):
        counts["student" if intervene else "teacher"] += 1
        return original(
            branch,
            intervene=intervene,
            requires_grad=requires_grad,
            capture_layers=capture_layers,
        )

    monkeypatch.setattr(psr, "_capture_branch", counted)
    teacher_cache = {}
    psr._imitation_loss(
        _batch(),
        requires_grad=False,
        teacher_cache=teacher_cache,
        populate_teacher_cache=True,
    )
    cached_batch = {
        "record_ids": _batch()["record_ids"],
        "base": _batch()["base"],
    }
    psr._imitation_loss(
        cached_batch, requires_grad=True, teacher_cache=teacher_cache
    )
    psr._imitation_loss(
        cached_batch, requires_grad=True, teacher_cache=teacher_cache
    )

    assert counts == {"teacher": 1, "student": 3}


def test_num_logits_to_keep_preserves_hooked_residuals_and_gradients():
    torch.manual_seed(23)
    subject = _tiny_gemma()
    torch.manual_seed(29)
    full_logits = _psr(APSR, copy.deepcopy(subject))
    torch.manual_seed(29)
    last_logits = _psr(APSR, copy.deepcopy(subject))
    full_logits._supports_num_logits_to_keep = False
    last_logits._supports_num_logits_to_keep = True

    full_loss, _ = full_logits._imitation_loss(_batch(), requires_grad=True)
    last_loss, _ = last_logits._imitation_loss(_batch(), requires_grad=True)
    assert torch.equal(full_loss, last_loss)
    full_loss.backward()
    last_loss.backward()
    _assert_module_tensors_equal(
        {
            name: parameter.grad
            for name, parameter in full_logits.psr_modules.named_parameters()
        },
        {
            name: parameter.grad
            for name, parameter in last_logits.psr_modules.named_parameters()
        },
    )


def test_teacher_cache_safely_falls_back_for_batching_and_stochastic_model():
    batched = _psr(SPSR, _tiny_gemma(), batch_size=2)
    assert not batched._teacher_cache_is_safe()

    stochastic_subject = _tiny_gemma()
    stochastic_subject.extra_dropout = torch.nn.Dropout(p=0.1)
    stochastic = _psr(SPSR, stochastic_subject)
    assert not stochastic._teacher_cache_is_safe()


def test_train_clears_partial_teacher_cache_after_exception(monkeypatch):
    psr = _psr(SPSR, _tiny_gemma())
    captured = {}
    records = [{"record_id": 0}]
    dummy_entry = _PSRTeacherCacheEntry(
        residuals=torch.ones(1, 1, 1),
        prefix_loss=torch.ones(()),
    )

    monkeypatch.setattr(psr, "_generate_prompt_instruction", lambda *a, **k: "x")
    monkeypatch.setattr(psr, "_training_records", lambda examples: records)
    monkeypatch.setattr(
        psr,
        "_make_psr_dataloader",
        lambda *a, **k: object(),
    )

    def fail(_loader, *, teacher_cache=None):
        captured["cache"] = teacher_cache
        teacher_cache[0] = dummy_entry
        raise RuntimeError("injected")

    monkeypatch.setattr(psr, "_initial_psi_scale", fail)
    examples = pd.DataFrame(
        [{"category": "positive", "raw_input": "q", "raw_output": "a"}]
    )
    with pytest.raises(RuntimeError, match="injected"):
        psr.train(examples)
    assert captured["cache"] == {}


def test_training_records_are_built_once(monkeypatch):
    psr = _psr(SPSR, _tiny_gemma(), batch_size=2, n_epochs=0)
    calls = {"records": 0}

    monkeypatch.setattr(psr, "_generate_prompt_instruction", lambda *a, **k: "x")

    def records(_examples):
        calls["records"] += 1
        return [{"record_id": 0}]

    monkeypatch.setattr(psr, "_training_records", records)
    monkeypatch.setattr(
        psr,
        "_make_psr_dataloader",
        lambda *a, **k: [object()],
    )
    monkeypatch.setattr(psr, "_initial_psi_scale", lambda *a, **k: 1.0)
    examples = pd.DataFrame(
        [{"category": "positive", "raw_input": "q", "raw_output": "a"}]
    )
    psr.train(examples)
    assert calls["records"] == 1
