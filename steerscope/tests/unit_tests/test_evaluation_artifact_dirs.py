import json
from types import SimpleNamespace

import pandas as pd
import pytest

from steerscope.evaluation.version import generated_data_identity
from steerscope.scripts.evaluate import _implicit_targets, _load_targets


def _args(config_file, artifact_dirs):
    return SimpleNamespace(
        models=["APSR", "FLAS"],
        model_name="test/model",
        steering_model_name=None,
        config_file=str(config_file),
        artifact_dirs_by_model=artifact_dirs,
    )


def test_implicit_targets_use_per_method_checkpoint_roots(tmp_path):
    config = tmp_path / "runtime" / "generalization.yaml"
    config.parent.mkdir()
    config.write_text("evaluate: {}\n", encoding="utf-8")
    apsr = config.parent / "artifacts" / "apsr"
    flas = tmp_path / "flas"
    apsr.mkdir(parents=True)
    flas.mkdir()
    args = _args(config, {"APSR": "artifacts/apsr", "FLAS": str(flas)})
    nodes = [SimpleNamespace(models=("APSR", "FLAS"))]

    targets = _implicit_targets(
        args,
        nodes,
        [{"concept_id": 3, "concept": "test"}],
        tmp_path / "unused-train",
    )

    assert {target.method: target.artifact.path for target in targets} == {
        "APSR": apsr.resolve(),
        "FLAS": flas.resolve(),
    }


def test_implicit_targets_reject_unknown_method_mapping(tmp_path):
    config = tmp_path / "generalization.yaml"
    config.write_text("evaluate: {}\n", encoding="utf-8")
    args = _args(config, {"Unknown": str(tmp_path)})
    nodes = [SimpleNamespace(models=("APSR", "FLAS"))]

    with pytest.raises(ValueError, match="not evaluated"):
        _implicit_targets(
            args,
            nodes,
            [{"concept_id": 0, "concept": "test"}],
            tmp_path / "train",
        )


def test_implicit_targets_require_configured_checkpoint_directory(tmp_path):
    config = tmp_path / "generalization.yaml"
    config.write_text("evaluate: {}\n", encoding="utf-8")
    args = _args(config, {"APSR": "missing"})
    nodes = [SimpleNamespace(models=("APSR", "FLAS"))]

    with pytest.raises(FileNotFoundError, match="APSR"):
        _implicit_targets(
            args,
            nodes,
            [{"concept_id": 0, "concept": "test"}],
            tmp_path / "train",
        )


def _strict_fixture(tmp_path, method="APSR", concept_ids=(0, 1)):
    root = tmp_path / "run"
    generated = root / "generate"
    artifact = tmp_path / "artifact"
    generated.mkdir(parents=True)
    artifact.mkdir()
    metadata = [
        {
            "concept_id": concept_id,
            "concept": f"concept-{concept_id}",
            "ref": f"ref-{concept_id}",
            "concept_genres_map": {f"concept-{concept_id}": ["text"]},
        }
        for concept_id in concept_ids
    ]
    (generated / "metadata.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in metadata),
        encoding="utf-8",
    )
    pd.DataFrame(
        {"concept_id": list(concept_ids), "input": ["x"] * len(concept_ids)}
    ).to_parquet(generated / "train_data.parquet", index=False)
    manifest = {
        "version": 2,
        "base_model": "test/model",
        "layer": 20,
        "component": "res",
        "concept_ids": list(concept_ids),
        "methods": {method: {"fingerprint": "trained"}},
        "generated_data": generated_data_identity(generated),
    }
    (artifact / "artifact_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    config = tmp_path / "runtime.yaml"
    config.write_text("evaluate: {}\n", encoding="utf-8")
    args = SimpleNamespace(
        models=[method],
        model_name="test/model",
        steering_model_name=None,
        steering_layers=[20],
        steering_layer=None,
        config_file=str(config),
        artifact_dirs_by_model={method: str(artifact)},
        targets=None,
    )
    training_args = SimpleNamespace(
        overwrite_data_dir=str(generated),
        overwrite_metadata_dir=str(generated),
        use_dpo_loss=False,
        component="res",
    )
    nodes = [
        SimpleNamespace(
            node_id="score",
            models=(method,),
            requires_inference=True,
            concepts={},
        )
    ]
    return root, artifact, args, training_args, nodes, manifest


def test_explicit_artifact_manifest_accepts_matching_training_identity(tmp_path):
    root, artifact, args, training_args, nodes, _ = _strict_fixture(tmp_path)

    targets = _load_targets(args, nodes, root, training_args)

    assert len(targets) == 2
    assert {target.artifact.path for target in targets} == {artifact.resolve()}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(base_model="wrong/model"), "base-model"),
        (lambda payload: payload.update(layer=10), "layer mismatch"),
        (lambda payload: payload.update(component="mlp"), "component mismatch"),
        (lambda payload: payload.update(methods={}), "fingerprinted entry"),
        (lambda payload: payload.update(concept_ids=[0]), "concept mismatch"),
        (lambda payload: payload.update(generated_data=None), "generated-data"),
    ],
)
def test_explicit_artifact_manifest_rejects_wrong_routing(
    tmp_path, mutation, message
):
    root, artifact, args, training_args, nodes, manifest = _strict_fixture(tmp_path)
    mutation(manifest)
    (artifact / "artifact_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=message):
        _load_targets(args, nodes, root, training_args)


def test_explicit_artifact_manifest_allows_scoped_per_concept_method(tmp_path):
    root, artifact, args, training_args, nodes, manifest = _strict_fixture(
        tmp_path, method="SFT", concept_ids=(0, 1)
    )
    manifest["concept_ids"] = [1]
    (artifact / "artifact_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    nodes[0].concepts = {"ids": [1]}

    targets = _load_targets(args, nodes, root, training_args)

    # Targets retain the full metadata pool; evaluator scope controls which
    # target is instantiated, and the manifest only needs to cover that scope.
    assert {target.concept.concept_id for target in targets} == {0, 1}


def test_stateless_simple_prompt_does_not_require_manifest(tmp_path):
    root, artifact, args, training_args, nodes, _ = _strict_fixture(
        tmp_path, method="SimplePromptSteering"
    )
    (artifact / "artifact_manifest.json").unlink()
    artifact.rmdir()

    targets = _load_targets(args, nodes, root, training_args)

    assert len(targets) == 2
