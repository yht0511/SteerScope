"""Controlled checks for concept/artifact routing during evaluation.

These tests intentionally use synthetic tensors and model doubles.  They are
not quality tests: a very large decoy concept row makes an accidental concept
mix-up immediately observable without downloading a language model.
"""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

import steerscope
from steerscope.evaluation import Artifact, Concept, EvaluationTarget
from steerscope.inference.steering import (
    SteeringModel,
    SteeringModelConfig,
    _SteeringTargetRuntime,
)
from steerscope.models.austeer import AUSteer
from steerscope.models.hidra import HiDRA, _hidra_geometric_batch
from steerscope.models.interventions import (
    AdditionIntervention,
    SimpleAdditionIntervention,
)
from steerscope.models.psr import APSR, FocusedPSRIntervention
from steerscope.models.reft import LoReFT
from steerscope.models.spherical_steering import (
    SphericalSteering,
    _spherical_geometric_batch,
)


# This is the complete ordinary 10/500-concept benchmark method set.  The
# taxonomy documents the artifact contract relied on by train.py and the
# inference instance cache.
PER_CONCEPT_FLAT_BANK = {"APSR", "SPSR"}
PER_CONCEPT_DIRECTORY = {"LoRA", "SFT", "ODESteer", "StepODESteer"}
SHARED_JOINT = {"FLAS", "HyperSteer"}
SHARED_MERGED_BANK = {
    "AUSteer",
    "DiffMean",
    "GemmaScopeSAE",
    "GemmaScopeSAEMaxAUC",
    "HiDRA",
    "LAT",
    "LinearProbe",
    "LoReFT",
    "LsReFT",
    "PCA",
    "PreferenceVector",
    "PromptSteering",
    "Random",
    "SphericalSteering",
    "SteeringVector",
}
SHARED_STATELESS = {"SimplePromptSteering"}
ALL_BENCHMARK_METHODS = (
    PER_CONCEPT_FLAT_BANK
    | PER_CONCEPT_DIRECTORY
    | SHARED_JOINT
    | SHARED_MERGED_BANK
    | SHARED_STATELESS
)


def _target(method, concept_id, artifact_dir, fingerprint="same-artifact"):
    return EvaluationTarget(
        target_id=f"{method}/concept-{concept_id}",
        method=method,
        concept=Concept(
            concept_id=concept_id,
            text=f"concept-{concept_id}",
            ref=None,
            metadata={},
        ),
        base_model="synthetic/base",
        artifact=Artifact(
            kind="checkpoint",
            path=Path(artifact_dir),
            fingerprint=fingerprint,
        ),
    )


def _runtime_args(**overrides):
    values = {
        "steering_batch_size": 2,
        "seed": 42,
        "steering_layer": 1,
        "steering_layers": [1],
        "runtime_backend": "legacy",
        "use_bf16": False,
        "master_data_dir": "unused",
        "disable_neuronpedia_max_act": False,
        "steering_intervention_type": "addition",
        "intervene_on_prompt": True,
        "compute_perplexity": False,
        "steering_output_length": 1,
        "temperature": 0.0,
        "do_sample": False,
        "steering_model_name": "synthetic/base",
        "model_name": "synthetic/base",
        "overwrite_cache": True,
        "evaluation_cache_context": {},
        "demo_interactive": False,
        "demo_output": None,
        "easysteer_url": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _SyntheticBenchmark:
    requires_mean_activations = False
    requires_training_args = False
    uses_intervention_positions = False
    concept_id_map = None

    def __init__(self, values, allowed_concepts=None):
        self.values = dict(values)
        self.allowed_concepts = (
            None if allowed_concepts is None else set(allowed_concepts)
        )

    def prepare_inference_examples(self, examples, **_kwargs):
        return examples

    def predict_steer(self, examples, *, concept_id, **_kwargs):
        if self.allowed_concepts is not None:
            assert concept_id in self.allowed_concepts
        assert set(examples["concept_id"]) == {concept_id}
        value = self.values[concept_id]
        return {
            "steered_generation": [str(value)] * len(examples),
            "strength": examples["factor"].tolist(),
        }


def _synthetic_runtime(tmp_path, method, scope, load_calls):
    class SyntheticMethod:
        inference_instance_scope = scope
        lightweight_runtime = True

    artifact_dir = tmp_path / "train"
    artifact_dir.mkdir()
    targets = [
        _target(method, concept_id, artifact_dir)
        for concept_id in (2, 7)
    ]
    runtime = _SteeringTargetRuntime(
        args=_runtime_args(master_data_dir=str(tmp_path)),
        training_args=SimpleNamespace(overwrite_metadata_dir=None, models={}),
        root_dump_dir=tmp_path,
        output_dir=tmp_path / "output",
        cache_dir=tmp_path / "cache",
        device=torch.device("cpu"),
        targets=targets,
        long_format=True,
    )
    runtime._model_class = lambda _method: SyntheticMethod
    runtime._metadata = lambda: [
        {
            "concept": target.concept.text,
            "ref": None,
            "concept_genres_map": {target.concept.text: ["text"]},
            "concept_id": target.concept.concept_id,
        }
        for target in targets
    ]

    def load_model(
        _method,
        _model,
        _tokenizer,
        _metadata,
        _train_dir,
        _layer,
        _steering_layers,
        concept_id,
    ):
        load_calls.append(concept_id)
        allowed = {concept_id} if scope == "per_concept" else {2, 7}
        return _SyntheticBenchmark(
            {2: 2.0, 7: 7.0}, allowed_concepts=allowed
        )

    runtime._load_concept_model = load_model
    return runtime, targets


def test_all_24_methods_declare_the_expected_instance_and_artifact_scope():
    assert len(ALL_BENCHMARK_METHODS) == 24
    assert not (
        PER_CONCEPT_FLAT_BANK & PER_CONCEPT_DIRECTORY
        or SHARED_JOINT & SHARED_MERGED_BANK
    )

    for method in sorted(PER_CONCEPT_FLAT_BANK | PER_CONCEPT_DIRECTORY):
        model_class = getattr(steerscope, method)
        assert model_class.training_granularity == "per_concept"
        assert model_class.inference_instance_scope == "per_concept"

    for method in sorted(SHARED_MERGED_BANK | SHARED_STATELESS):
        model_class = getattr(steerscope, method)
        assert model_class.training_granularity == "per_concept"
        assert model_class.inference_instance_scope == "shared"

    for method in sorted(SHARED_JOINT):
        model_class = getattr(steerscope, method)
        assert model_class.training_granularity == "all_concepts"
        assert model_class.inference_instance_scope == "shared"

    assert APSR.artifact_directory is None
    assert steerscope.SPSR.artifact_directory is None
    assert steerscope.LoRA.artifact_directory == "lora"
    assert steerscope.SFT.artifact_directory == "sft"
    assert steerscope.ODESteer.artifact_directory == "odesteer"
    assert steerscope.StepODESteer.artifact_directory == "step_odesteer"


def test_per_concept_runtime_never_reuses_another_concepts_instance(tmp_path):
    loads = []
    runtime, targets = _synthetic_runtime(
        tmp_path, "SyntheticPerConcept", "per_concept", loads
    )
    examples = pd.DataFrame({"input": ["same"], "factor": [1.0]})
    try:
        first = runtime.generate_target(targets[0], examples)
        second = runtime.generate_target(targets[1], examples)
    finally:
        runtime.close()

    assert loads == [2, 7]
    assert first["SyntheticPerConcept_steered_generation"].tolist() == ["2.0"]
    assert second["SyntheticPerConcept_steered_generation"].tolist() == ["7.0"]


def test_shared_runtime_constructs_once_and_routes_each_concept_row(tmp_path):
    loads = []
    runtime, targets = _synthetic_runtime(
        tmp_path, "SyntheticShared", "shared", loads
    )
    examples = pd.DataFrame({"input": ["same"], "factor": [1.0]})
    try:
        first = runtime.generate_target(targets[0], examples)
        second = runtime.generate_target(targets[1], examples)
    finally:
        runtime.close()

    # The one shared instance was constructed while processing concept 2, yet
    # its bank correctly serves concept 7 on the next call.
    assert loads == [2]
    assert first["SyntheticShared_steered_generation"].tolist() == ["2.0"]
    assert second["SyntheticShared_steered_generation"].tolist() == ["7.0"]


def test_merged_vector_bank_ignores_huge_decoy_row_and_zero_is_identity():
    intervention = AdditionIntervention(embed_dim=3, low_rank_dimension=3)
    with torch.no_grad():
        intervention.proj.weight.copy_(torch.tensor([
            [1.0, 2.0, 3.0],
            [1.0e20, -1.0e20, 1.0e20],
            [-4.0, -5.0, -6.0],
        ]))
        intervention.proj.bias.zero_()
    hidden = torch.tensor([[[0.5, 1.5, -2.0], [2.0, 3.0, 4.0]]])

    selected = intervention(
        hidden,
        subspaces={
            "idx": torch.tensor([0]),
            "mag": torch.tensor([1.0]),
            "max_act": torch.tensor([1.0]),
        },
    )
    zero = intervention(
        hidden,
        subspaces={
            "idx": torch.tensor([0]),
            "mag": torch.tensor([0.0]),
            "max_act": torch.tensor([1.0]),
        },
    )

    torch.testing.assert_close(
        selected, hidden + torch.tensor([1.0, 2.0, 3.0]), rtol=0, atol=0
    )
    assert torch.equal(zero, hidden)
    assert torch.isfinite(selected).all()


def test_specialized_factor_zero_kernels_are_exact_identities_with_huge_state():
    hidden = torch.tensor([[[0.5, 1.5, -2.0], [2.0, 3.0, 4.0]]])

    # HyperSteer has a dynamically generated vector rather than a stored bank.
    hyper = SimpleAdditionIntervention(embed_dim=3, low_rank_dimension=1)
    hyper._update_v(torch.full((1, 3), 1.0e20))
    hyper_zero = hyper(
        hidden,
        subspaces={"mag": torch.tensor([0.0])},
    ).output
    assert torch.equal(hyper_zero, hidden)

    # PSR's activation-dependent location fit is still multiplied by strength
    # before the steering direction is applied.
    psr = FocusedPSRIntervention(hidden_size=3)
    with torch.no_grad():
        psr.steering_proj.weight.fill_(1.0e20)
        psr.location_fit_proj.weight.fill_(1.0e10)
        psr.location_fit_proj.bias.fill_(1.0e10)
    psr_zero = psr(
        hidden,
        torch.tensor([0.0]),
        torch.ones(hidden.shape[:2], dtype=torch.bool),
    )
    assert torch.equal(psr_zero, hidden)

    # Spherical Steering explicitly excludes alpha==0 from its active mask.
    spherical_zero, spherical_active = _spherical_geometric_batch(
        hidden,
        mu_t=torch.tensor([[1.0, 0.0, 0.0]]),
        mu_h=torch.tensor([[-1.0, 0.0, 0.0]]),
        kappa=20.0,
        alpha=torch.tensor([0.0]),
        beta=0.1,
        token_mask=torch.ones(hidden.shape[:2], dtype=torch.bool),
    )
    assert torch.equal(spherical_zero, hidden)
    assert not spherical_active.any()

    # HiDRA returns before even invoking the costly lifted-space projector.
    class ProjectorMustNotRun:
        def steer(self, *_args, **_kwargs):
            raise AssertionError("zero-factor HiDRA invoked its projector")

    hidra_zero, hidra_active = _hidra_geometric_batch(
        hidden,
        directions=torch.full((1, 5), 1.0e20),
        strengths=torch.tensor([0.0]),
        token_mask=torch.ones(hidden.shape[:2], dtype=torch.bool),
        projector=ProjectorMustNotRun(),
    )
    assert hidra_zero is hidden
    assert not hidra_active.any()


def test_austeer_factor_zero_never_applies_a_huge_decoy_mask():
    class Layer:
        def __init__(self):
            self.mlp = torch.nn.Identity()

    subject = AUSteer.__new__(AUSteer)
    subject.model = SimpleNamespace(
        model=SimpleNamespace(layers=[Layer()])
    )
    subject.au_masks = torch.tensor([
        [[0.25, 0.5, -0.5]],
        [[1.0e20, -1.0e20, 1.0e20]],
    ])
    hidden = torch.tensor([[[1.0, 2.0, 3.0]]])

    with subject._intervention(
        mask_indices=torch.tensor([1]),
        strengths=torch.tensor([0.0]),
        token_mask=torch.ones((1, 1), dtype=torch.bool),
    ):
        output = subject.model.model.layers[0].mlp(hidden)

    assert torch.equal(output, hidden)


def test_nonstandard_shared_banks_select_global_concept_ids():
    # These three new methods do not use the generic AdditionIntervention, so
    # exercise their independent concept-index helpers with decoy rows too.
    spherical = SphericalSteering.__new__(SphericalSteering)
    spherical.concept_id_map = None
    spherical.device = torch.device("cpu")
    spherical.ax = SimpleNamespace(
        proj=SimpleNamespace(weight=torch.zeros(8, 4))
    )
    assert spherical._prototype_indices([2, 7]).tolist() == [2, 7]

    hidra = HiDRA.__new__(HiDRA)
    hidra.concept_id_map = None
    hidra.device = torch.device("cpu")
    hidra.directions = torch.zeros(8, 4)
    hidra.directions[6].fill_(1.0e20)
    assert hidra._direction_indices([2, 7]).tolist() == [2, 7]

    austeer = AUSteer.__new__(AUSteer)
    austeer.concept_id_map = None
    austeer.device = torch.device("cpu")
    austeer.au_masks = torch.zeros(8, 2, 4)
    austeer.au_masks[6].fill_(1.0e20)
    assert austeer._mask_indices([2, 7]).tolist() == [2, 7]

    loreft = LoReFT.__new__(LoReFT)
    loreft.concept_id_map = None
    loreft.device = torch.device("cpu")
    loreft.number_of_interventions = 2
    subspaces, strengths = loreft.choice_subspaces(pd.DataFrame({
        "concept_id": [2, 7],
        "factor": [0.0, 1.0],
    }))
    assert len(subspaces) == 2
    assert subspaces[0]["idx"].tolist() == [2, 7]
    assert strengths == [0.0, 1.0]


def test_psr_per_concept_loader_selects_only_requested_flat_bank_row(tmp_path):
    class FakePSRLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.steering_proj = torch.nn.Linear(3, 1, bias=False)
            self.location_fit_proj = torch.nn.Linear(3, 1, bias=True)

    weights = {
        "layer_4.steering": torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [1.0e20, -1.0e20, 1.0e20],
        ]),
        "layer_4.location": torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [1.0e20, -1.0e20, 1.0e20],
        ]),
    }
    biases = {
        "layer_4.location": torch.tensor([[0.1], [0.2], [1.0e20]])
    }
    torch.save(weights, tmp_path / "APSR_weight.pt")
    torch.save(biases, tmp_path / "APSR_bias.pt")

    psr = APSR.__new__(APSR)
    psr.psr_layers = [4]
    psr.concept_id_map = None
    layer = FakePSRLayer()
    psr.make_model = lambda **_kwargs: setattr(psr, "psr_modules", {"4": layer})

    psr.load(tmp_path, model_name="APSR", concept_id=1)

    torch.testing.assert_close(
        layer.steering_proj.weight,
        torch.tensor([[4.0, 5.0, 6.0]]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        layer.location_fit_proj.weight,
        torch.tensor([[0.4, 0.5, 0.6]]),
        rtol=0,
        atol=0,
    )
    assert psr.concept_id_map == {1: 0}


def test_request_cache_identity_separates_method_concept_factor_and_scope(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()

    def cache_path(
        method="DiffMean", concept_id=2, factor=1.0, scope="identity",
        fingerprint="checkpoint-a", temperature=0.0, do_sample=False,
    ):
        model = SteeringModel(
            target=_target(method, concept_id, artifact_dir, fingerprint),
            args=_runtime_args(overwrite_cache=False),
            training_args=SimpleNamespace(models={}),
            root_dump_dir=tmp_path,
            cache_dir=tmp_path / "cache",
            config=SteeringModelConfig(
                factor=factor,
                layer=1,
                layers=(1,),
                temperature=temperature,
                do_sample=do_sample,
                max_new_tokens=1,
                batch_size=1,
            ),
            device=torch.device("cpu"),
        )
        examples = pd.DataFrame({
            "input": ["identical prompt"],
            "factor": [factor],
            "model_factor": [factor],
            "scope": [scope],
        })
        return model._cache_file(examples)

    reference = cache_path()
    variants = {
        cache_path(method="PCA"),
        cache_path(concept_id=7),
        cache_path(factor=2.0),
        cache_path(scope="base64"),
        cache_path(fingerprint="checkpoint-b"),
        cache_path(temperature=1.0, do_sample=True),
    }
    assert reference not in variants
    assert len(variants) == 6


def test_request_cache_rejects_legacy_file_without_checkpoint_identity(tmp_path):
    artifact_dir = tmp_path / "train"
    artifact_dir.mkdir()
    cache_dir = tmp_path / "cache"
    model = SteeringModel(
        target=_target("APSR", 7, artifact_dir),
        args=_runtime_args(overwrite_cache=False),
        training_args=SimpleNamespace(models={}),
        root_dump_dir=tmp_path,
        cache_dir=cache_dir,
        config=SteeringModelConfig(
            factor=0.8,
            layer=1,
            layers=(1,),
            temperature=0.0,
            do_sample=False,
            max_new_tokens=1,
            batch_size=1,
        ),
        runtime=SimpleNamespace(
            generate_target=lambda _target, rows, **_kwargs: rows.assign(
                method="APSR",
                target_id="APSR/concept-7",
                APSR_steered_generation="fresh generation",
            )
        ),
    )
    examples = pd.DataFrame({
        "concept_id": [7],
        "input": ["cached prompt"],
        "factor": [0.8],
        "model_factor": [0.8],
    })
    cached = examples.assign(
        method="APSR",
        target_id="APSR/concept-7",
        APSR_steered_generation="cached generation",
    )
    request_dir = cache_dir / "requests"
    request_dir.mkdir(parents=True)
    legacy_path = request_dir / "APSR_legacy-execution-hash.parquet"
    cached.to_parquet(legacy_path, index=False)
    result = model.generate(examples)

    assert result["APSR_steered_generation"].tolist() == ["fresh generation"]
    assert pd.read_parquet(legacy_path).equals(cached)
    assert model._cache_file(examples).is_file()


def test_baseline_cache_rejects_legacy_file_without_sampling_identity(tmp_path):
    artifact_dir = tmp_path / "train"
    artifact_dir.mkdir()
    cache_dir = tmp_path / "cache"
    model = SteeringModel(
        target=_target("APSR", 7, artifact_dir),
        args=_runtime_args(overwrite_cache=False),
        training_args=SimpleNamespace(models={}),
        root_dump_dir=tmp_path,
        cache_dir=cache_dir,
        config=SteeringModelConfig(
            factor=0.0, temperature=0.0, do_sample=False,
        ),
        runtime=SimpleNamespace(
            generate_baseline=lambda rows: rows.assign(
                baseline_generation="fresh generation"
            )
        ),
    )
    examples = pd.DataFrame({"input": ["cached prompt"], "factor": [0.0]})
    request_dir = cache_dir / "requests"
    request_dir.mkdir(parents=True)
    legacy_path = request_dir / "Baseline_legacy-execution-hash.parquet"
    examples.assign(baseline_generation="stale generation").to_parquet(
        legacy_path, index=False
    )

    result = model.generate_baseline(examples)

    assert result["baseline_generation"].tolist() == ["fresh generation"]
    assert model._baseline_cache_file(examples).is_file()


@pytest.mark.parametrize("factor", [0.0, 1.0])
def test_bound_model_rejects_cross_factor_requests(factor, tmp_path):
    model = SteeringModel(
        target=_target("DiffMean", 2, tmp_path),
        args=_runtime_args(),
        training_args=SimpleNamespace(models={}),
        root_dump_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        config=SteeringModelConfig(factor=factor),
    )
    with pytest.raises(ValueError, match="bound to factor"):
        model.generate(pd.DataFrame({
            "input": ["x"],
            "factor": [1.0 - factor],
        }))
