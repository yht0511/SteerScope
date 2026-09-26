import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch

from steerscope.evaluation import (
    Artifact,
    Concept,
    EvaluationTarget,
)
from steerscope.inference.easysteer import (
    EasySteerRuntime,
    EasySteerUnsupported,
    EasySteerVectorResolver,
    EasySteerVectorSpec,
)
from steerscope.inference.steering import _SteeringTargetRuntime


def _target(
    checkpoint_dir: Path,
    *,
    method: str = "DiffMean",
    concept_id: int = 1,
    ref: str | None = None,
) -> EvaluationTarget:
    return EvaluationTarget(
        target_id=f"{method}/concept-{concept_id}",
        method=method,
        concept=Concept(
            concept_id=concept_id,
            text="test concept",
            ref=ref,
        ),
        base_model="google/gemma-2-2b-it",
        artifact=Artifact("checkpoint", checkpoint_dir),
    )


def test_vector_resolver_exports_concept_row_and_applies_scale(tmp_path):
    checkpoint_dir = tmp_path / "train"
    checkpoint_dir.mkdir()
    weights = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    torch.save(weights, checkpoint_dir / "DiffMean_weight.pt")
    torch.save(
        torch.tensor([10.0, 20.0, 30.0]),
        checkpoint_dir / "DiffMean_scale.pt",
    )
    resolver = EasySteerVectorResolver(
        cache_dir=tmp_path / "vectors",
        default_checkpoint_dir=checkpoint_dir,
        master_data_dir=tmp_path / "data",
    )

    spec = resolver.resolve(
        _target(checkpoint_dir),
        factor=0.5,
        target_layers=[20],
        intervention_type="addition",
        disable_neuronpedia_max_act=False,
    )

    assert spec.scale == 10.0
    assert spec.target_layers == (20,)
    assert spec.path.is_file()
    assert torch.equal(
        torch.load(spec.path, weights_only=True),
        weights[1],
    )
    # The content-addressed export is reused across factor scans.
    second = resolver.resolve(
        _target(checkpoint_dir),
        factor=2.0,
        target_layers=[20],
        intervention_type="addition",
        disable_neuronpedia_max_act=False,
    )
    assert second.path == spec.path
    assert second.scale == 40.0


def test_vector_resolver_reads_sae_feature_scale(tmp_path):
    checkpoint_dir = tmp_path / "train"
    checkpoint_dir.mkdir()
    torch.save(
        {"W_dec": torch.eye(3, 4)},
        checkpoint_dir / "GemmaScopeSAE.pt",
    )
    torch.save(
        torch.tensor([12.5, 20.0, 30.0]),
        checkpoint_dir / "GemmaScopeSAE_scale.pt",
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    resolver = EasySteerVectorResolver(
        cache_dir=tmp_path / "vectors",
        default_checkpoint_dir=checkpoint_dir,
        master_data_dir=data_dir,
    )

    spec = resolver.resolve(
        _target(
            checkpoint_dir,
            method="GemmaScopeSAE",
            concept_id=0,
            ref="https://www.neuronpedia.org/gemma-2-2b/20-res/42",
        ),
        factor=0.2,
        target_layers=[20],
        intervention_type="addition",
        disable_neuronpedia_max_act=False,
    )

    assert spec.scale == 2.5
    assert torch.equal(
        torch.load(spec.path, weights_only=True),
        torch.eye(3, 4)[0],
    )


def test_vector_resolver_maps_max_auc_selected_feature_scale(tmp_path):
    checkpoint_dir = tmp_path / "train"
    checkpoint_dir.mkdir()
    weights = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    torch.save(
        {"W_dec": weights},
        checkpoint_dir / "GemmaScopeSAEMaxAUC.pt",
    )
    with open(
        checkpoint_dir / "GemmaScopeSAEMaxAUC_top_features.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump([100, 200, 300], file)
    torch.save(
        torch.tensor([10.0, 15.0, 20.0]),
        checkpoint_dir / "GemmaScopeSAEMaxAUC_scale.pt",
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    resolver = EasySteerVectorResolver(
        cache_dir=tmp_path / "vectors",
        default_checkpoint_dir=checkpoint_dir,
        master_data_dir=data_dir,
    )

    spec = resolver.resolve(
        _target(
            checkpoint_dir,
            method="GemmaScopeSAEMaxAUC",
            concept_id=1,
            ref="https://www.neuronpedia.org/gemma-2-2b/20-res/42",
        ),
        factor=0.4,
        target_layers=[20],
        intervention_type="addition",
        disable_neuronpedia_max_act=False,
    )

    assert spec.scale == 6.0
    assert torch.equal(
        torch.load(spec.path, weights_only=True),
        weights[1],
    )


def test_vector_resolver_rejects_non_addition(tmp_path):
    resolver = EasySteerVectorResolver(
        cache_dir=tmp_path / "vectors",
        default_checkpoint_dir=tmp_path,
        master_data_dir=tmp_path,
    )

    with pytest.raises(EasySteerUnsupported, match="addition"):
        resolver.resolve(
            _target(tmp_path),
            factor=1.0,
            target_layers=[20],
            intervention_type="clamping",
            disable_neuronpedia_max_act=False,
        )


def test_easysteer_runtime_batches_completion_prompts(tmp_path, monkeypatch):
    vector_path = tmp_path / "vector.pt"
    torch.save(torch.ones(4), vector_path)
    resolver = MagicMock()
    resolver.resolve.return_value = EasySteerVectorSpec(
        path=vector_path,
        scale=7.5,
        target_layers=(20,),
    )
    runtime = EasySteerRuntime(
        base_url="http://127.0.0.1:8017",
        model_name="google/gemma-2-2b-it",
        resolver=resolver,
    )
    completion_payloads = []

    def fake_request(method, path, payload=None):
        if path == "/v1/models":
            return {"data": [{"id": "google/gemma-2-2b-it"}]}
        completion_payloads.append(payload)
        return {
            "choices": [
                {"index": index, "text": f"output:{prompt}"}
                for index, prompt in enumerate(payload["prompt"])
            ]
        }

    monkeypatch.setattr(runtime, "_json_request", fake_request)
    examples = pd.DataFrame({
        "input": ["one", "two", "three"],
        "factor": [0.5, 0.5, 0.5],
    })

    result = runtime.predict(
        _target(tmp_path),
        examples,
        target_layers=[20],
        batch_size=2,
        max_new_tokens=32,
        temperature=1.0,
        seed=42,
        intervention_type="addition",
        intervene_on_prompt=True,
        disable_neuronpedia_max_act=False,
        compute_perplexity=False,
    )

    assert result == {
        "steered_generation": [
            "output:one",
            "output:two",
            "output:three",
        ],
        "strength": [7.5, 7.5, 7.5],
    }
    assert [payload["prompt"] for payload in completion_payloads] == [
        ["one", "two"],
        ["three"],
    ]
    assert all(payload["top_k"] == 50 for payload in completion_payloads)
    assert all(payload["top_p"] == 1.0 for payload in completion_payloads)
    assert all(
        payload["repetition_penalty"] == 1.0
        for payload in completion_payloads
    )
    assert all(
        payload["truncate_prompt_tokens"] == 1024
        for payload in completion_payloads
    )
    steer_request = completion_payloads[0]["steer_vector_request"]
    assert steer_request["target_layers"] == [20]
    assert steer_request["prefill_trigger_tokens"] == [-1]
    assert steer_request["generate_trigger_tokens"] == [-1]
    assert steer_request["normalize"] is False


def test_target_runtime_bypasses_legacy_model_loading(tmp_path):
    checkpoint_dir = tmp_path / "train"
    checkpoint_dir.mkdir()
    target = _target(checkpoint_dir, concept_id=0)
    args = SimpleNamespace(
        runtime_backend="easysteer",
        easysteer_url="http://127.0.0.1:8017",
        easysteer_timeout=30.0,
        seed=42,
        steering_layer=None,
        steering_layers=[20],
        steering_batch_size=2,
        steering_output_length=16,
        steering_intervention_type="addition",
        temperature=1.0,
        intervene_on_prompt=True,
        disable_neuronpedia_max_act=False,
        compute_perplexity=False,
        master_data_dir=str(tmp_path / "data"),
    )
    runtime = _SteeringTargetRuntime(
        args=args,
        training_args=SimpleNamespace(
            overwrite_metadata_dir=None,
            models={},
        ),
        root_dump_dir=tmp_path,
        output_dir=tmp_path / "output",
        cache_dir=tmp_path / "cache",
        targets=[target],
        long_format=True,
        device=torch.device("cpu"),
    )
    easy_runtime = MagicMock()
    easy_runtime.model_name = target.base_model
    easy_runtime.predict.return_value = {
        "steered_generation": ["generated"],
        "strength": [2.0],
    }
    runtime._easysteer_runtime = easy_runtime
    runtime._ensure_runtime_resources = MagicMock(
        side_effect=AssertionError("legacy model loading must not run")
    )

    result = runtime.generate_target(
        target,
        pd.DataFrame({"input": ["prompt"], "factor": [1.0]}),
        batch_size=7,
    )

    runtime._ensure_runtime_resources.assert_not_called()
    easy_runtime.predict.assert_called_once()
    assert result["DiffMean_steered_generation"].tolist() == ["generated"]
    assert easy_runtime.predict.call_args.kwargs["batch_size"] == 7
    assert result["DiffMean_strength"].tolist() == [2.0]
