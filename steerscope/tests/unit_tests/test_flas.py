import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from safetensors.torch import save_file
from transformers import BatchEncoding, Gemma2Config
from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM

from steerscope.models.flas import FLAS
from steerscope.models.flas_core import (
    FLASConfig,
    FLASGenerator,
    build_flow_model_from_base,
    integrate_euler,
)


class TinyTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    model_max_length = 64
    padding_side = "right"
    truncation_side = "right"

    @staticmethod
    def _encode(text, add_special_tokens):
        values = [3 + ord(character) % 50 for character in str(text)]
        return ([1] + values) if add_special_tokens else values

    def __call__(
        self,
        texts,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=None,
        add_special_tokens=True,
        **kwargs,
    ):
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        rows = [self._encode(text, add_special_tokens) for text in texts]
        if truncation and max_length is not None:
            rows = [row[: int(max_length)] for row in rows]
        if return_tensors == "pt":
            width = max(len(row) for row in rows)
            padded, masks = [], []
            for row in rows:
                missing = width - len(row)
                if self.padding_side == "left":
                    padded.append([0] * missing + row)
                    masks.append([0] * missing + [1] * len(row))
                else:
                    padded.append(row + [0] * missing)
                    masks.append([1] * len(row) + [0] * missing)
            return BatchEncoding(
                {
                    "input_ids": torch.tensor(padded),
                    "attention_mask": torch.tensor(masks),
                }
            )
        return {"input_ids": rows[0] if single else rows}

    def pad(self, encoded, return_tensors=None, padding=True, **kwargs):
        rows = [list(row) for row in encoded["input_ids"]]
        width = max(len(row) for row in rows)
        padded = [row + [self.pad_token_id] * (width - len(row)) for row in rows]
        masks = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        return BatchEncoding(
            {
                "input_ids": torch.tensor(padded),
                "attention_mask": torch.tensor(masks),
            }
        )

    @staticmethod
    def batch_decode(rows, skip_special_tokens=True):
        return [" ".join(str(int(value)) for value in row) for row in rows]


def tiny_model():
    config = Gemma2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=16,
        max_position_embeddings=64,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
    )
    return Gemma2ForCausalLM(config)


def training_args(**overrides):
    values = {
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "lr": 5e-5,
        "weight_decay": 0.01,
        "flas_num_blocks": 1,
        "flas_n_steps": 3,
        "flas_t_min": 0.5,
        "flas_t_max": 2.0,
        "flas_div_weight": 0.1,
        "flas_total_steps": 2,
        "flas_warmup_steps": 0,
        "flas_max_length": 32,
        "flas_concept_max_length": 8,
        "flas_n_val_samples": 0,
        "flas_val_n_concepts": 0,
        "flas_val_every": 1,
        "flas_val_batches": 1,
        "flas_patience": 2,
        "flas_num_workers": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_flas_uses_joint_training_and_shared_inference():
    assert FLAS.training_granularity == "all_concepts"
    assert FLAS.inference_instance_scope == "shared"
    assert FLAS.artifact_directory == "flas"
    assert not FLAS.uses_intervention_positions


def test_gemma2_445_core_runs_prefill_and_cached_decode():
    model = tiny_model()
    flow, encoder = build_flow_model_from_base(model, layer=2, num_blocks=1)
    input_ids = torch.randint(0, 128, (2, 5))
    attention_mask = torch.ones_like(input_ids)
    concept = encoder(input_ids[:, :3], attention_mask[:, :3]).float()
    hidden = torch.randn(2, 5, model.config.hidden_size)
    positions = torch.arange(5).unsqueeze(0).expand(2, -1)

    _, _, caches = integrate_euler(
        flow,
        hidden,
        concept,
        attention_mask[:, :3].float(),
        torch.ones(2),
        3,
        padding_mask=attention_mask.float(),
        use_cache=True,
        position_ids=positions,
    )
    decoded, _, caches = integrate_euler(
        flow,
        torch.randn(2, 1, model.config.hidden_size),
        concept,
        attention_mask[:, :3].float(),
        torch.ones(2),
        3,
        self_attn_caches=caches,
        padding_mask=torch.ones(2, 6),
        use_cache=True,
        past_len=5,
        position_ids=torch.full((2, 1), 5),
    )

    assert decoded.shape == (2, 1, model.config.hidden_size)
    assert [cache[0].get_seq_length() for cache in caches] == [6, 6, 6]


def test_zero_flow_time_is_exact_identity():
    model = tiny_model()
    flow, encoder = build_flow_model_from_base(model, layer=2, num_blocks=1)
    input_ids = torch.randint(0, 128, (2, 4))
    attention_mask = torch.ones_like(input_ids)
    concept = encoder(input_ids[:, :2], attention_mask[:, :2]).float()
    hidden = torch.randn(2, 4, model.config.hidden_size)

    output, _, _ = integrate_euler(
        flow,
        hidden,
        concept,
        attention_mask[:, :2].float(),
        torch.zeros(2),
        3,
        padding_mask=attention_mask.float(),
    )

    assert torch.equal(output, hidden)


def test_manual_generation_runs_nonzero_flow_with_hybrid_cache():
    model = tiny_model().eval()
    flow, encoder = build_flow_model_from_base(model, layer=2, num_blocks=1)
    generator = FLASGenerator(
        model,
        TinyTokenizer(),
        flow.eval(),
        encoder.eval(),
        layer=2,
        n_steps=3,
        input_max_length=32,
        device="cpu",
    )

    generations = generator.generate_batch(
        ["abc", "xy"],
        ["pirate", "formal"],
        [1.0, 1.5],
        max_new_tokens=2,
        temperature=1.0,
        do_sample=False,
    )

    assert len(generations) == 2
    assert all(isinstance(value, str) for value in generations)
    assert generator._hook_handle is None


def test_native_predict_and_choice_paths_use_dynamic_concepts():
    tokenizer = TinyTokenizer()
    method = FLAS(
        tiny_model(),
        tokenizer=tokenizer,
        layer=2,
        training_args=training_args(),
        device="cpu",
    )
    method.make_model()
    examples = pd.DataFrame(
        {
            "input": ["abc", "xy"],
            "input_concept": ["pirate", "formal"],
            "factor": [0.0, 1.0],
            "choice_token_ids": [[3, 4], [5, 6]],
        }
    )

    generated = method.predict_steer(
        examples,
        batch_size=2,
        eval_output_length=1,
        do_sample=False,
        show_progress=False,
    )
    choices = method.predict_choice_logits(
        examples,
        batch_size=2,
        show_progress=False,
    )

    assert len(generated["steered_generation"]) == 2
    assert generated["strength"] == [0.0, 1.0]
    assert len(choices["choice_logits"]) == 2
    assert choices["strength"] == [0.0, 1.0]


def test_one_step_native_training_updates_flow_weights():
    tokenizer = TinyTokenizer()
    method = FLAS(
        tiny_model(),
        tokenizer=tokenizer,
        layer=2,
        training_args=training_args(flas_total_steps=1),
        device="cpu",
    )
    method.make_model()
    before = {
        key: value.detach().clone() for key, value in method.flow_fn.state_dict().items()
    }
    examples = pd.DataFrame(
        {
            "input": ["prompt a", "prompt b"],
            "output": ["answer a", "answer b"],
            "output_concept": ["pirate", "formal"],
            "concept_id": [1, 2],
        }
    )

    method.train(examples)

    assert any(
        not torch.equal(before[key], value)
        for key, value in method.flow_fn.state_dict().items()
    )


def test_flas_validates_on_microbatches_during_gradient_accumulation(monkeypatch):
    method = FLAS(
        tiny_model(),
        tokenizer=TinyTokenizer(),
        layer=2,
        training_args=training_args(
            gradient_accumulation_steps=2,
            flas_total_steps=2,
            flas_n_val_samples=2,
            flas_val_every=3,
        ),
        device="cpu",
    )
    method.make_model()
    training_forwards = 0
    validation_at = []
    parameter = next(method.flow_fn.parameters())

    def cheap_training_forward(_batch, _flow_integrator, _terminal_time):
        nonlocal training_forwards
        training_forwards += 1
        return parameter.reshape(-1)[0] * 1e-3, None

    def record_validation(_dataloader, _flow_integrator):
        validation_at.append(training_forwards)
        return float(len(validation_at))

    monkeypatch.setattr(method, "_forward_with_flow", cheap_training_forward)
    monkeypatch.setattr(method, "_validation_loss", record_validation)
    monkeypatch.setattr(
        "steerscope.models.flas.compute_diversity_loss",
        lambda *_args, **_kwargs: torch.zeros((), requires_grad=False),
    )
    examples = pd.DataFrame(
        {
            "input": [f"prompt {index}" for index in range(20)],
            "output": [f"answer {index}" for index in range(20)],
            "output_concept": [f"concept {index % 2}" for index in range(20)],
            "concept_id": [index % 2 for index in range(20)],
        }
    )

    method.train(examples)

    # Validation runs after microbatch 3, between optimizer updates 1 and 2.
    assert validation_at == [3]


def test_flas_lr_schedule_matches_released_microbatch_pacing():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = FLAS._scheduler(
        optimizer,
        warmup_steps=2,
        total_steps=6,
        accumulation_steps=2,
    )
    scale = scheduler.lr_lambdas[0]

    assert scale(0) == pytest.approx(0.0)
    assert scale(1) == pytest.approx(1.0)
    assert scale(2) == pytest.approx(0.5)
    assert scale(3) == pytest.approx(0.0)


def test_flas_validation_loss_weights_partial_batches(monkeypatch):
    method = FLAS(
        tiny_model(),
        tokenizer=TinyTokenizer(),
        layer=2,
        training_args=training_args(flas_val_batches=10),
        device="cpu",
    )
    losses = iter([torch.tensor(1.0), torch.tensor(5.0)])
    monkeypatch.setattr(
        method,
        "_forward_with_flow",
        lambda *_args, **_kwargs: (next(losses), None),
    )
    batches = [
        {"input_ids": torch.zeros(3, 1, dtype=torch.long)},
        {"input_ids": torch.zeros(1, 1, dtype=torch.long)},
    ]
    flow_integrator = torch.nn.Linear(1, 1)

    # Lightning aggregates validation loss by the actual batch size:
    # (3 * 1 + 1 * 5) / 4 = 2.
    assert method._validation_loss(batches, flow_integrator) == pytest.approx(2.0)


def test_flas_uses_bf16_mixed_autocast_on_cuda(monkeypatch):
    method = FLAS(
        tiny_model(),
        tokenizer=TinyTokenizer(),
        layer=2,
        training_args=training_args(),
        device="cpu",
    )
    method.device = torch.device("cuda:0")
    calls = []

    class RecordingContext:
        def __enter__(self):
            calls.append("enter")

        def __exit__(self, exc_type, exc_value, traceback):
            calls.append("exit")

    def record_autocast(*, device_type, dtype):
        calls.append((device_type, dtype))
        return RecordingContext()

    monkeypatch.setattr(torch, "autocast", record_autocast)

    with method._training_precision_context():
        calls.append("body")

    assert calls == [
        ("cuda", torch.bfloat16),
        "enter",
        "body",
        "exit",
    ]


def test_joint_checkpoint_round_trip(tmp_path):
    original = FLAS(
        tiny_model(),
        tokenizer=None,
        layer=2,
        training_args=training_args(),
        device="cpu",
    )
    original.make_model()
    original.trained_concept_ids = [1, 2]
    original.heldout_concept_ids = [3]
    original.save(tmp_path)

    restored = FLAS(
        tiny_model(),
        tokenizer=None,
        layer=2,
        training_args=training_args(),
        device="cpu",
    )
    restored.load(tmp_path)

    for expected, actual in zip(
        original.flow_fn.state_dict().values(),
        restored.flow_fn.state_dict().values(),
    ):
        torch.testing.assert_close(expected.to(torch.bfloat16), actual, rtol=0, atol=0)
    assert restored.trained_concept_ids == [1, 2]
    assert restored.heldout_concept_ids == [3]
    assert (tmp_path / "flas" / "flow.pt").is_file()


def test_released_checkpoint_directory_layout_loads(tmp_path):
    original = FLAS(
        tiny_model(),
        tokenizer=None,
        layer=2,
        training_args=training_args(),
        device="cpu",
    )
    original.make_model()
    with open(tmp_path / "config.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "model_id": "google/gemma-2-2b-it",
                "layer": 2,
                "num_blocks": 1,
                "n_steps": 3,
            },
            file,
        )
    save_file(
        {
            key: value.detach().to(torch.bfloat16).contiguous()
            for key, value in original.flow_fn.state_dict().items()
        },
        tmp_path / "flas-gemma-2-2b-it.safetensors",
    )

    restored = FLAS(
        tiny_model(),
        tokenizer=None,
        layer=2,
        training_args=training_args(),
        device="cpu",
    )
    restored.load(tmp_path)

    assert restored._config().num_blocks == 1
    assert restored._config().n_steps == 3
    assert next(restored.flow_fn.parameters()).dtype == torch.bfloat16


def test_flas_config_rejects_invalid_time_range():
    with pytest.raises(ValueError, match="t_min"):
        FLASConfig(t_min=2.0, t_max=1.0).validate()


def test_flas_config_uses_defaults_for_none_training_values():
    config = FLASConfig.from_training_args(
        SimpleNamespace(flas_num_blocks=None, flas_n_steps=None)
    )

    assert config.num_blocks == 1
    assert config.n_steps == 3
