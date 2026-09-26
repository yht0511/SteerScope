import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import set_seed

from steerscope.utils.training_seed import (
    TRAINING_RECIPE_VERSION,
    derive_training_seed,
)
from steerscope.models.sft import SFT

from steerscope.scripts.train import (
    TRAIN_STATE_VERSION,
    _completed_joint_training_methods,
    _completed_training_methods,
    _final_method_artifact_is_complete,
    _gemmascope_baseline_requires_refresh,
    _index_checkpoint_values_by_concept,
    _save_method_checkpoint,
    _save_collective_sft_checkpoint,
    _save_joint_method_checkpoint,
    _training_method_fingerprints,
    _any_rank_trained,
    _new_rank_zero_artifact,
    _partition_training_methods,
    _rank_training_assignment,
    materialize_rank_artifacts,
    materialize_joint_artifact,
    save_state,
    select_training_concepts,
    write_artifact_manifest,
    write_rank_metadata,
)


class VectorCheckpointModel:
    def __init__(self, value):
        self.value = float(value)

    def save(self, dump_dir, model_name, **kwargs):
        tensor = torch.tensor([[self.value, self.value + 1]])
        torch.save(tensor, dump_dir / f"{model_name}_weight.pt")
        torch.save(torch.tensor([self.value]), dump_dir / f"{model_name}_bias.pt")


class DirectoryCheckpointModel:
    def __init__(self, concept_id):
        self.concept_id = int(concept_id)

    def save(self, dump_dir, model_name, **kwargs):
        artifact = dump_dir / "lora" / str(self.concept_id)
        artifact.mkdir(parents=True)
        (artifact / "adapter_config.json").write_text(
            json.dumps({"concept_id": self.concept_id}), encoding="utf-8"
        )


class PromptCheckpointModel:
    def __init__(self, concept_id):
        self.concept_id = int(concept_id)

    def save(self, dump_dir, **kwargs):
        (dump_dir / "PromptSteering_prompts.json").write_text(
            json.dumps({
                "concept_id": self.concept_id,
                "concept": f"concept {self.concept_id}",
                "instruction": f"instruction {self.concept_id}",
            }),
            encoding="utf-8",
        )


class JointCheckpointModel:
    def save(self, dump_dir, **kwargs):
        artifact = dump_dir / "hyperreft"
        artifact.mkdir(parents=True)
        (artifact / "config.json").write_text("{}", encoding="utf-8")


class CollectiveSFTCheckpointModel:
    def __init__(self):
        self.calls = 0

    def save(self, dump_dir, **kwargs):
        self.calls += 1
        artifact = dump_dir / "sft" / str(kwargs["concept_id"])
        artifact.mkdir(parents=True)
        (artifact / "config.json").write_text("{}", encoding="utf-8")


class FakeFSDPAccelerator:
    def __init__(self, state_dict):
        self.state_dict = state_dict
        self.calls = []

    def get_state_dict(self, model):
        self.calls.append(model)
        return self.state_dict


class FakeFSDPTrainer:
    def __init__(self, state_dict, should_save=True):
        self.model = object()
        self.accelerator = FakeFSDPAccelerator(state_dict)
        self.args = SimpleNamespace(should_save=should_save)
        self.optimizer = object()
        self.lr_scheduler = object()
        self.saved = []

    def _save(self, output_dir, state_dict):
        self.saved.append((Path(output_dir), state_dict.copy()))


def test_training_seed_is_stable_per_method_and_concept():
    first = derive_training_seed(42, "APSR", 7)
    assert first == derive_training_seed(42, "APSR", 7)
    assert first != derive_training_seed(42, "SPSR", 7)
    assert first != derive_training_seed(42, "APSR", 8)
    assert first != derive_training_seed(43, "APSR", 7)
    assert derive_training_seed(42, "FLAS") != first
    assert 0 <= first < 2**31 - 1
    assert TRAINING_RECIPE_VERSION >= 3


def test_training_seed_makes_resume_order_irrelevant():
    target_seed = derive_training_seed(42, "LoRA", 3)
    set_seed(target_seed)
    uninterrupted = torch.rand(5)

    set_seed(derive_training_seed(42, "DiffMean", 0))
    _ = torch.rand(100)
    set_seed(target_seed)
    resumed = torch.rand(5)

    assert torch.equal(uninterrupted, resumed)


def test_cached_training_does_not_request_artifact_merge():
    assert _any_rank_trained(False, torch.device("cpu")) is False
    assert _any_rank_trained(True, torch.device("cpu")) is True


def test_training_methods_are_partitioned_by_model_capability():
    per_concept, all_concepts = _partition_training_methods([
        "HyperSteer",
        "DiffMean",
        "PromptSteering",
    ])
    assert per_concept == ["DiffMean", "PromptSteering"]
    assert all_concepts == ["HyperSteer"]


def test_only_multi_gpu_sft_uses_collective_concept_assignment():
    frames = [(index, object()) for index in range(5)]

    rank_zero, checkpoint_rank, owner, collective = _rank_training_assignment(
        frames,
        world_size=2,
        rank=0,
        model_names=["SFT"],
        lm_model_name="google/gemma-2-9b-it",
    )
    rank_one, other_checkpoint_rank, other_owner, other_collective = (
        _rank_training_assignment(
            frames,
            world_size=2,
            rank=1,
            model_names=["SFT"],
            lm_model_name="google/gemma-2-9b-it",
        )
    )
    assert rank_zero == frames
    assert rank_one == frames
    assert (checkpoint_rank, owner, collective) == (0, True, True)
    assert (other_checkpoint_rank, other_owner, other_collective) == (
        0, False, True,
    )

    ordinary_rank_zero = _rank_training_assignment(
        frames,
        world_size=2,
        rank=0,
        model_names=["DiffMean"],
        lm_model_name="google/gemma-2-9b-it",
    )
    ordinary_rank_one = _rank_training_assignment(
        frames,
        world_size=2,
        rank=1,
        model_names=["DiffMean"],
        lm_model_name="google/gemma-2-9b-it",
    )
    assert ordinary_rank_zero[0] == frames[:3]
    assert ordinary_rank_one[0] == frames[3:]
    assert ordinary_rank_zero[1:] == (0, True, False)
    assert ordinary_rank_one[1:] == (1, True, False)

    single_gpu_sft = _rank_training_assignment(
        frames,
        world_size=1,
        rank=0,
        model_names=["SFT"],
        lm_model_name="google/gemma-2-2b-it",
    )
    assert single_gpu_sft == (frames, 0, True, False)


def test_multi_gpu_sft_rejects_mixed_method_yaml():
    with pytest.raises(ValueError, match="SFT-only"):
        _rank_training_assignment(
            [(0, object())],
            world_size=2,
            rank=0,
            model_names=["SFT", "DiffMean"],
            lm_model_name="google/gemma-2-9b-it",
        )


def test_multi_gpu_sft_rejects_non_fsdp_model():
    with pytest.raises(ValueError, match="only for Gemma-2-9B"):
        _rank_training_assignment(
            [(0, object())],
            world_size=2,
            rank=0,
            model_names=["SFT"],
            lm_model_name="google/gemma-2-2b-it",
        )


def test_collective_sft_checkpoint_is_published_by_rank_zero(
    tmp_path, monkeypatch
):
    barriers = []
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch.distributed, "broadcast_object_list", lambda values, src: None
    )
    monkeypatch.setattr(
        torch.distributed, "barrier", lambda: barriers.append(True)
    )
    model = CollectiveSFTCheckpointModel()

    checkpoint = _save_collective_sft_checkpoint(
        model,
        tmp_path,
        concept_id=7,
        fingerprint="same",
    )

    assert checkpoint == (
        tmp_path / "checkpoints/rank_0/concept_7/SFT"
    )
    assert (checkpoint / "sft/7/config.json").is_file()
    manifest = json.loads((checkpoint / ".complete.json").read_text())
    assert manifest["fingerprint"] == "same"
    assert manifest["distributed_training"] == "collective_fsdp_v1"
    assert model.calls == 1
    assert len(barriers) == 2

    assert _save_collective_sft_checkpoint(
        model,
        tmp_path,
        concept_id=7,
        fingerprint="same",
    ) == checkpoint
    assert model.calls == 1


def test_fsdp_sft_save_casts_only_floating_checkpoint_tensors_to_bf16(
    tmp_path,
):
    state_dict = {
        "weight": torch.tensor([1.25], dtype=torch.float32),
        "already_bf16": torch.tensor([2.5], dtype=torch.bfloat16),
        "position_ids": torch.tensor([0, 1], dtype=torch.int64),
    }
    trainer = FakeFSDPTrainer(state_dict)
    model = SFT.__new__(SFT)
    model.concept_id = 3
    model._fsdp_trainer = trainer

    model.save(tmp_path)

    assert trainer.accelerator.calls == [trainer.model]
    assert len(trainer.saved) == 1
    output_dir, saved = trainer.saved[0]
    assert output_dir == tmp_path / "sft/3"
    assert saved["weight"].dtype == torch.bfloat16
    assert saved["already_bf16"].dtype == torch.bfloat16
    assert saved["position_ids"].dtype == torch.int64
    assert trainer.optimizer is None
    assert trainer.lr_scheduler is None


def test_fsdp_sft_save_is_collective_but_only_main_rank_writes(tmp_path):
    trainer = FakeFSDPTrainer(
        {"weight": torch.tensor([1.25], dtype=torch.float32)},
        should_save=False,
    )
    model = SFT.__new__(SFT)
    model.concept_id = 3
    model._fsdp_trainer = trainer

    model.save(tmp_path)

    assert trainer.accelerator.calls == [trainer.model]
    assert trainer.saved == []


def test_explicit_training_concepts_preserve_requested_ids_and_validate_scope():
    frames = [(index, object()) for index in range(6)]

    selected = select_training_concepts(
        frames,
        concept_ids=[4, 1],
        model_names=["SFT"],
    )

    assert [concept_id for concept_id, _ in selected] == [4, 1]
    with pytest.raises(ValueError, match="mutually exclusive"):
        select_training_concepts(
            frames,
            concept_ids=[1],
            max_concepts=2,
            model_names=["SFT"],
        )
    with pytest.raises(ValueError, match="unsupported methods"):
        select_training_concepts(
            frames,
            concept_ids=[1],
            model_names=["DiffMean"],
        )
    with pytest.raises(ValueError, match="absent from generated data"):
        select_training_concepts(
            frames,
            concept_ids=[99],
            model_names=["SFT"],
        )


def test_sampled_training_concepts_are_seeded_and_order_independent():
    frames = [(index, object()) for index in range(20)]
    reversed_frames = list(reversed(frames))

    first = select_training_concepts(
        frames,
        num_concepts=5,
        seed=123,
        model_names=["SFT"],
    )
    reordered = select_training_concepts(
        reversed_frames,
        num_concepts=5,
        seed=123,
        model_names=["SFT"],
    )
    changed = select_training_concepts(
        frames,
        num_concepts=5,
        seed=124,
        model_names=["SFT"],
    )

    assert {concept_id for concept_id, _ in first} == {
        concept_id for concept_id, _ in reordered
    }
    assert {concept_id for concept_id, _ in first} != {
        concept_id for concept_id, _ in changed
    }
    with pytest.raises(ValueError, match="mutually exclusive"):
        select_training_concepts(
            frames,
            concept_ids=[1],
            num_concepts=5,
            model_names=["SFT"],
        )


def test_prompt_checkpoints_materialize_in_concept_order(tmp_path):
    for concept_id in (1, 0):
        _save_method_checkpoint(
            PromptCheckpointModel(concept_id),
            tmp_path,
            rank=0,
            concept_id=concept_id,
            model_name="PromptSteering",
            fingerprint="same",
        )

    materialize_rank_artifacts(
        tmp_path,
        rank=0,
        concept_ids=[0, 1],
        model_names=["PromptSteering"],
    )

    entries = json.loads(
        (tmp_path / "rank_0_PromptSteering_prompts.json").read_text()
    )
    assert [entry["concept_id"] for entry in entries] == [0, 1]

    (tmp_path / "PromptSteering_prompts.json").write_text(
        json.dumps(entries), encoding="utf-8"
    )
    assert _final_method_artifact_is_complete(
        tmp_path, "PromptSteering", [0, 1]
    )


def test_existing_gemmascope_artifact_is_not_overwritten(tmp_path):
    filename = "GemmaScopeSAE.pt"

    assert _new_rank_zero_artifact(tmp_path, 0, filename) == filename
    (tmp_path / filename).write_bytes(b"existing artifact")

    assert _new_rank_zero_artifact(tmp_path, 0, filename) is None
    assert _new_rank_zero_artifact(tmp_path, 1, filename) is None
    assert _new_rank_zero_artifact(tmp_path, 0, filename, force=True) == filename


def test_joint_checkpoint_is_resumable_and_materialized(tmp_path):
    _save_joint_method_checkpoint(
        JointCheckpointModel(),
        tmp_path,
        model_name="HyperSteer",
        concept_ids=[0, 1],
        fingerprint="same",
    )
    completed = _completed_joint_training_methods(
        tmp_path,
        ["HyperSteer"],
        [0, 1],
        method_fingerprints={"HyperSteer": "same"},
    )
    assert not _final_method_artifact_is_complete(
        tmp_path, "HyperSteer", [0, 1]
    )
    materialize_joint_artifact(tmp_path, "HyperSteer")

    assert completed == {"HyperSteer"}
    assert _final_method_artifact_is_complete(
        tmp_path, "HyperSteer", [0, 1]
    )
    assert (tmp_path / "hyperreft").is_symlink()
    assert (tmp_path / "hyperreft" / "config.json").read_text() == "{}"
    manifest = json.loads((
        tmp_path
        / "checkpoints"
        / "all_concepts"
        / "HyperSteer"
        / ".complete.json"
    ).read_text())
    assert manifest["concept_ids"] == [0, 1]


def test_joint_resume_rejects_changed_scope_or_fingerprint(tmp_path):
    _save_joint_method_checkpoint(
        JointCheckpointModel(),
        tmp_path,
        model_name="HyperSteer",
        concept_ids=[0, 1],
        fingerprint="old",
    )
    with pytest.raises(RuntimeError, match="concept set changed"):
        _completed_joint_training_methods(
            tmp_path,
            ["HyperSteer"],
            [0],
            method_fingerprints={"HyperSteer": "old"},
        )
    with pytest.raises(RuntimeError, match="configuration changed"):
        _completed_joint_training_methods(
            tmp_path,
            ["HyperSteer"],
            [0, 1],
            method_fingerprints={"HyperSteer": "new"},
        )


def test_method_checkpoints_are_atomic_and_materialize_in_concept_order(tmp_path):
    for concept_id, value in ((1, 20), (0, 10)):
        _save_method_checkpoint(
            VectorCheckpointModel(value),
            tmp_path,
            rank=0,
            concept_id=concept_id,
            model_name="DiffMean",
            fingerprint="same",
        )

    completed = _completed_training_methods(
        tmp_path,
        rank=0,
        model_names=["DiffMean"],
        method_fingerprints={"DiffMean": "same"},
    )
    materialize_rank_artifacts(
        tmp_path, rank=0, concept_ids=[0, 1], model_names=["DiffMean"]
    )

    assert completed == {(0, "DiffMean"), (1, "DiffMean")}
    weights = torch.load(
        tmp_path / "rank_0_DiffMean_weight.pt", weights_only=True
    )
    assert weights[:, 0].tolist() == [10.0, 20.0]
    assert not list((tmp_path / ".staging").iterdir())


def test_sparse_checkpoint_rows_keep_their_global_concept_ids():
    indexed = _index_checkpoint_values_by_concept(
        [
            torch.tensor([[10.0, 11.0], [60.0, 61.0]]),
            torch.tensor([[110.0, 111.0]]),
        ],
        [[1, 6], [11]],
    )

    assert indexed.shape == (12, 2)
    assert indexed[1].tolist() == [10.0, 11.0]
    assert indexed[6].tolist() == [60.0, 61.0]
    assert indexed[11].tolist() == [110.0, 111.0]
    assert indexed[0].tolist() == [0.0, 0.0]


def test_sparse_dictionary_sae_columns_and_scales_keep_concept_ids():
    weights = _index_checkpoint_values_by_concept(
        [
            {"proj": torch.tensor([[[10.0]]])},
            {"proj": torch.tensor([[[60.0]], [[110.0]]])},
        ],
        [[1], [6, 11]],
    )
    encoder = _index_checkpoint_values_by_concept(
        [torch.tensor([[10.0]]), torch.tensor([[60.0, 110.0]])],
        [[1], [6, 11]],
        dimension=1,
    )
    scales = _index_checkpoint_values_by_concept(
        [torch.tensor([10.0]), torch.tensor([60.0, 110.0])],
        [[1], [6, 11]],
        fill_value=1,
    )

    assert weights["proj"].shape == (12, 1, 1)
    assert weights["proj"][[1, 6, 11], 0, 0].tolist() == [10.0, 60.0, 110.0]
    assert encoder.shape == (1, 12)
    assert encoder[0, [1, 6, 11]].tolist() == [10.0, 60.0, 110.0]
    assert scales[[1, 6, 11]].tolist() == [10.0, 60.0, 110.0]
    assert scales[0].item() == 1.0


def test_directory_checkpoint_replaces_existing_artifact_without_deleting_it(
    tmp_path,
):
    existing = tmp_path / "lora" / "0"
    existing.mkdir(parents=True)
    (existing / "old.txt").write_text("legacy", encoding="utf-8")
    _save_method_checkpoint(
        DirectoryCheckpointModel(0),
        tmp_path,
        rank=0,
        concept_id=0,
        model_name="LoRA",
        fingerprint="same",
    )

    materialize_rank_artifacts(
        tmp_path, rank=0, concept_ids=[0], model_names=["LoRA"]
    )

    assert (tmp_path / "lora" / "0").is_symlink()
    assert json.loads(
        (tmp_path / "lora" / "0" / "adapter_config.json").read_text()
    ) == {"concept_id": 0}
    assert (tmp_path / "lora" / ".pre_transactional" / "0" / "old.txt").exists()


def test_state_and_rank_metadata_are_overwritten_idempotently(tmp_path):
    state = {
        "version": TRAIN_STATE_VERSION,
        "completed": [(0, "DiffMean")],
    }
    save_state(tmp_path, state, rank=0)
    with open(tmp_path / "train_state.pkl_rank_0", "rb") as file:
        assert pickle.load(file) == state

    metadata = [
        {"concept_id": 0, "concept": "zero"},
        {"concept_id": 1, "concept": "one"},
    ]
    write_rank_metadata(tmp_path, rank=0, concept_ids=[0, 1], metadata=metadata)
    write_rank_metadata(tmp_path, rank=0, concept_ids=[0, 1], metadata=metadata)

    rows = [
        json.loads(line)
        for line in (tmp_path / "rank_0_metadata.jsonl").read_text().splitlines()
    ]
    assert [row["concept_id"] for row in rows] == [0, 1]


def test_final_artifact_is_complete_for_every_configured_concept(tmp_path):
    torch.save(torch.ones(2, 3), tmp_path / "DiffMean_weight.pt")
    torch.save(torch.ones(2), tmp_path / "DiffMean_bias.pt")
    torch.save(torch.ones(2), tmp_path / "DiffMean_scale.pt")
    assert _final_method_artifact_is_complete(
        tmp_path, "DiffMean", [0, 1]
    )
    assert not _final_method_artifact_is_complete(
        tmp_path, "DiffMean", [0, 1, 2]
    )


def test_resume_rejects_changed_checkpoint_fingerprint(tmp_path):
    _save_method_checkpoint(
        VectorCheckpointModel(1),
        tmp_path,
        rank=0,
        concept_id=0,
        model_name="DiffMean",
        fingerprint="old",
    )

    with pytest.raises(RuntimeError, match="configuration changed"):
        _completed_training_methods(
            tmp_path,
            rank=0,
            model_names=["DiffMean"],
            method_fingerprints={"DiffMean": "new"},
        )


def test_training_fingerprint_uses_data_content_not_mtime(tmp_path):
    data = tmp_path / "train_data.parquet"
    metadata = tmp_path / "metadata.jsonl"
    data.write_bytes(b"same parquet bytes")
    metadata.write_text('{"concept_id": 0}\n', encoding="utf-8")
    args = SimpleNamespace(
        model_name="base/model",
        layer=3,
        component="res",
        output_length=32,
        seed=42,
        use_bf16=True,
        max_concepts=1,
        max_num_of_examples=None,
        use_dpo_loss=False,
        models={"DiffMean": SimpleNamespace(n_epochs=1)},
    )
    generate_args = SimpleNamespace(
        output_length=32,
        keep_legacy_output_format=True,
    )

    first = _training_method_fingerprints(
        args, generate_args, tmp_path, 1, ["DiffMean"]
    )
    stat = data.stat()
    os.utime(data, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    same = _training_method_fingerprints(
        args, generate_args, tmp_path, 1, ["DiffMean"]
    )
    data.write_bytes(b"changed parquet bytes")
    changed = _training_method_fingerprints(
        args, generate_args, tmp_path, 1, ["DiffMean"]
    )

    assert same == first
    assert changed != first


def test_artifact_manifest_records_fingerprints_without_hashing_weights(tmp_path):
    args = SimpleNamespace(
        model_name="base/model",
        layer=20,
        component="res",
    )

    payload = write_artifact_manifest(
        tmp_path,
        args=args,
        method_fingerprints={"SFT": "training-fingerprint"},
        concept_ids=[9, 3],
        world_size=1,
    )

    restored = json.loads((tmp_path / "artifact_manifest.json").read_text())
    assert restored == payload
    assert restored["concept_ids"] == [3, 9]
    assert restored["methods"]["SFT"] == {
        "artifact_directory": "sft",
        "fingerprint": "training-fingerprint",
        "inference_instance_scope": "per_concept",
        "training_granularity": "per_concept",
    }


def test_standalone_gemmascope_has_a_fingerprint_and_pool_bound_manifest(tmp_path):
    (tmp_path / "train_data.parquet").write_bytes(b"training rows")
    (tmp_path / "metadata.jsonl").write_text(
        '{"concept_id": 0}\n', encoding="utf-8"
    )
    args = SimpleNamespace(
        model_name="base/model",
        layer=20,
        component="res",
        output_length=32,
        seed=42,
        use_bf16=True,
        max_concepts=1,
        concept_ids=None,
        max_num_of_examples=None,
        subset_seed=None,
        use_dpo_loss=False,
        models={},
    )
    generate_args = SimpleNamespace(
        output_length=32,
        keep_legacy_output_format=True,
    )

    fingerprints = _training_method_fingerprints(
        args,
        generate_args,
        tmp_path,
        1,
        ["GemmaScopeSAE"],
    )
    generated_data = {
        "version": 1,
        "metadata": {"sha256": "pool"},
        "data_files": [],
    }
    write_artifact_manifest(
        tmp_path,
        args=args,
        method_fingerprints=fingerprints,
        concept_ids=[0],
        world_size=1,
        generated_data=generated_data,
    )

    assert fingerprints["GemmaScopeSAE"]
    assert not _gemmascope_baseline_requires_refresh(
        tmp_path,
        generated_data,
        fingerprint=fingerprints["GemmaScopeSAE"],
    )
    assert _gemmascope_baseline_requires_refresh(
        tmp_path,
        {**generated_data, "metadata": {"sha256": "other"}},
        fingerprint=fingerprints["GemmaScopeSAE"],
    )
