from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import asyncio
import json
import os
from pathlib import Path
import threading

import steerscope
import pandas as pd
import pytest
import torch

from steerscope.evaluation import (
    Artifact,
    Concept,
    EvaluationContext,
    EvaluationEngine,
    EvaluationResult,
    EvaluationTarget,
    EvaluatorNode,
    ResultStore,
    apply_node_overrides,
    concept_seed,
    context_for_node,
    expand_factors,
    parse_evaluation_targets,
    parse_evaluator_nodes,
    require_dataset_type,
    require_num_examples,
    split_by_input_id,
)
from steerscope.evaluation.version import (
    _source_fingerprint,
    evaluation_source_fingerprint,
)
from steerscope.evaluators.evaluator import Evaluator
from steerscope.evaluators.demo import DemoEvaluator
from steerscope.evaluators.best_factor import BestFactorEvaluator
from steerscope.evaluators.judge import JudgeEvaluatorMixin
from steerscope.evaluators.jailbreakbench import JailBreakBenchEvaluator
from steerscope.evaluators.lm_judge import LMJudgeEvaluator, _InferenceSpool
from steerscope.evaluators.mmlu import MMLU_DATASET_FILES, MMLUEvaluator
from steerscope.evaluators.output_length import OutputLengthEvaluator
from steerscope.evaluators.prompt_generalization import PromptGeneralizationEvaluator
from steerscope.evaluators.ifeval import IFEvalEvaluator
from steerscope.evaluators.truthfulqa import TruthfulQAEvaluator
from steerscope.evaluators.ppl import PerplexityEvaluator
from steerscope.evaluators.rule_judge import RuleEvaluator
from steerscope.evaluators.winrate import WinRateEvaluator
from steerscope.inference.steering import (
    SteeringModel,
    SteeringModelConfig,
    SteeringInferenceWrapper,
    SteeringTargetRunner,
    _SteeringTargetRuntime,
)
from steerscope.models.language_models import LanguageModel
from steerscope.models.demo import DemoModel
from steerscope.models.model import BaseModel, Model
from steerscope.models.lora import LoRA
from steerscope.models.prompt import PromptSteering
from steerscope.models.random import Random
from steerscope.models.reft import LoReFT
from steerscope.models.sft import SFT
from steerscope.scripts.evaluate import (
    _configured_batch_size,
    _configured_factors,
    _engine_context,
    _file_signature,
    _method_artifact_directory,
    _progress_store_root,
    _request_cache_dir,
    ensure_single_process_environment,
)


class RecordingEvaluator(Evaluator):
    def build_dataset(self, model, factors):
        return self.examples.copy()

    def compute_metrics(self, examples):
        if "factor" not in examples:
            return {"score": float(examples["score"].mean())}
        return {
            "factor": sorted(examples["factor"].unique()),
            "score": [1.0] * examples["factor"].nunique(),
            "raw_score": [1.0] * len(examples),
        }


def evaluator_args(**overrides):
    values = {
        "steering_layers": [3],
        "steering_layer": None,
        "temperature": 1.0,
        "steering_output_length": 32,
        "steering_batch_size": 4,
        "seed": 42,
        "steering_intervention_type": "addition",
        "intervene_on_prompt": True,
        "disable_neuronpedia_max_act": False,
        "winrate_split_ratio": 0.5,
        "steer_data_type": "concept",
        "lm_model": "judge",
        "master_data_dir": "data",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def target(method="DiffMean", concept_id=0):
    return EvaluationTarget(
        target_id=f"{method}/concept-{concept_id}",
        method=method,
        concept=Concept(concept_id, "formal language"),
        base_model="base/model",
        artifact=Artifact("checkpoint"),
    )


def evaluator_context(tmp_path, results=None, progress=None, **arg_overrides):
    return EvaluationContext(
        args=evaluator_args(**arg_overrides),
        root_dump_dir=tmp_path,
        output_dir=tmp_path / "output",
        results=results,
        progress=progress,
    )


def test_parse_nodes_and_apply_inference_overrides():
    node = parse_evaluator_nodes({
        "judge": {
            "type": "LMJudgeEvaluator",
            "models": ["DiffMean"],
            "dataset": {"type": "AlpacaEval", "num_examples": 4},
            "inference": {"strengths": [0.5, 1.0], "output_length": 64},
        }
    })[0]
    args = SimpleNamespace(models=["PromptSteering"])

    overridden = apply_node_overrides(args, node)

    assert overridden.models == ["DiffMean"]
    assert not hasattr(overridden, "steering_datasets")
    assert not hasattr(overridden, "steering_num_of_examples")
    assert not hasattr(overridden, "steering_factors")
    assert overridden.steering_output_length == 64


def test_configured_factors_supports_per_model_overrides():
    node = EvaluatorNode(
        "judge",
        "LMJudgeEvaluator",
        inference={
            "strengths": [0.5, 1.0, 2.0],
            "strengths_by_model": {"LoRA": [1.0]},
        },
    )

    assert _configured_factors(node) == [0.5, 1.0, 2.0]
    assert _configured_factors(node, "DiffMean") == [0.5, 1.0, 2.0]
    assert _configured_factors(node, "LoRA") == [1.0]


def test_configured_batch_size_supports_per_model_overrides():
    args = SimpleNamespace(
        steering_batch_size=100,
        batch_size_by_model={"HyperSteer": 20},
    )

    assert _configured_batch_size(args, "DiffMean") == 100
    assert _configured_batch_size(args, "HyperSteer") == 20


def test_linear_probe_is_resolvable_from_package_namespace():
    assert getattr(steerscope, "LinearProbe").__name__ == "LinearProbe"


def test_random_supports_training_entrypoint_mode():
    model = Random.__new__(Random)
    model.model = SimpleNamespace(config=SimpleNamespace(hidden_size=4))
    model.device = "cpu"

    model.make_model(mode="train", low_rank_dimension=1)

    assert model.ax.proj.weight.shape == (1, 4)


def test_demo_model_prints_prompts_and_formats_automatic_output(capsys):
    model = DemoModel(None, None, layer=0, device="cpu")
    examples = pd.DataFrame({
        "input": ["First prompt", "Second prompt"],
        "input_concept": ["clarity", "clarity"],
        "factor": [0.5, 0.5],
        "input_id": [0, 1],
    })

    result = model.predict_steer(
        examples,
        demo_output="answer {input_id}: {concept} at {factor}",
    )

    assert result == {
        "steered_generation": [
            "answer 0: clarity at 0.5",
            "answer 1: clarity at 0.5",
        ]
    }
    output = capsys.readouterr().out
    assert "First prompt" in output
    assert "Second prompt" in output
    assert "inference complete" in output


def test_demo_evaluator_builds_and_scores_readable_fixture(tmp_path, capsys):
    node = EvaluatorNode(
        "demo",
        "DemoEvaluator",
        dataset={"prompts": ["one", "two"], "num_examples": 3},
    )
    evaluator = DemoEvaluator(node, evaluator_context(tmp_path))
    model = SimpleNamespace(
        method="DemoModel",
        factor=2.0,
        concept=SimpleNamespace(concept_id=7, text="testing"),
    )

    examples = evaluator.build_dataset(model, [1.0, 2.0])
    evaluator.model_name = "DemoModel"
    examples["DemoModel_steered_generation"] = ["x"] * len(examples)
    metrics = evaluator.compute_metrics(examples)

    assert examples["input"].tolist() == ["one", "two", "one"] * 2
    assert metrics["factor"] == [1.0, 2.0]
    assert metrics["demo_nonempty_rate"] == [1.0, 1.0]
    output = capsys.readouterr().out
    assert "[DemoEvaluator] build dataset:" in output
    assert "[DemoEvaluator] score:" in output


def test_demo_evaluator_queries_declared_dependencies(tmp_path, capsys):
    store = ResultStore(tmp_path / "runs", run_id="demo")
    store.mark_running("upstream", "hash", {"id": "upstream"})
    store.save_metrics("upstream", pd.DataFrame({"score": [0.5, 0.75]}))
    store.mark_complete(
        "upstream",
        "hash",
        {"id": "upstream"},
        metadata={"result_kinds": ["metrics"]},
    )
    node = EvaluatorNode(
        "demo",
        "DemoEvaluator",
        depends_on=("upstream",),
    )
    context = evaluator_context(tmp_path, results=store.view(("upstream",)))
    evaluator = DemoEvaluator(node, context, dependency_preview_rows=1)

    evaluator.prepare_evaluation([MagicMock()], [MagicMock()])

    output = capsys.readouterr().out
    assert "querying 'upstream' (metrics)" in output
    assert "upstream.metrics: 2 row(s); columns: score" in output
    assert "0.5" in output
    assert "... 1 more row(s)" in output


def test_output_length_evaluator_queries_and_groups_upstream_inference(
    tmp_path, monkeypatch
):
    store = ResultStore(tmp_path / "runs", run_id="length")
    inference = pd.DataFrame({
        "target_id": ["DiffMean/concept-0"] * 2 + ["DiffMean/concept-1"] * 2,
        "method": ["DiffMean"] * 4,
        "concept_id": [0, 0, 1, 1],
        "input_concept": ["happy", "happy", "formal", "formal"],
        "factor": [1.0] * 4,
        "model_factor": [1.0] * 4,
        "DiffMean_steered_generation": ["a bb", "a bb ccc d", "one", "one two three"],
    })
    store.mark_running("math", "hash", {"id": "math"})
    store.save_inference("math", inference)
    store.mark_complete(
        "math", "hash", {"id": "math"},
        metadata={"result_kinds": ["inference"]},
    )

    tokenizer = MagicMock()
    tokenizer.side_effect = lambda texts, **kwargs: {
        "input_ids": [text.split() for text in texts]
    }
    monkeypatch.setattr(
        "steerscope.evaluators.output_length.AutoTokenizer.from_pretrained",
        MagicMock(return_value=tokenizer),
    )
    node = EvaluatorNode(
        "math_output_length",
        "OutputLengthEvaluator",
        depends_on=("math",),
        input={"from": "math", "kind": "inference"},
        report={"formats": ["png"], "dpi": 72},
    )
    evaluator = OutputLengthEvaluator(
        node,
        evaluator_context(
            tmp_path,
            results=store.view(("math",)),
            steering_model_name="google/gemma-2-2b-it",
        ),
    )

    result = evaluator.evaluate([], [])

    assert result.samples["output_tokens"].tolist() == [2, 4, 1, 3]
    assert result.metrics["base_model"].unique().tolist() == [
        "google/gemma-2-2b-it"
    ]
    assert result.metrics["mean_output_tokens"].tolist() == [3.0, 2.0]
    assert result.metrics["output_length_num_examples"].tolist() == [2, 2]
    tokenizer.assert_called_once_with(
        ["a bb", "a bb ccc d", "one", "one two three"],
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    paths = evaluator.render_report(result, tmp_path / "length_report")
    assert {path.name for path in paths} == {"summary.parquet", "metrics.png"}
    summary = pd.read_parquet(
        tmp_path / "length_report" / "reports" / "summary.parquet"
    )
    assert summary["mean"].tolist() == [2.5]
    assert summary["count"].tolist() == [2]


def test_output_length_evaluator_applies_configured_concept_scope(
    tmp_path, monkeypatch
):
    store = ResultStore(tmp_path / "runs", run_id="scoped-length")
    inference = pd.DataFrame({
        "target_id": ["DiffMean/concept-0", "DiffMean/concept-1"],
        "method": ["DiffMean", "DiffMean"],
        "concept_id": [0, 1],
        "factor": [1.0, 1.0],
        "DiffMean_steered_generation": ["short", "two tokens"],
    })
    store.mark_running("math", "hash", {"id": "math"})
    store.save_inference("math", inference)
    store.mark_complete(
        "math", "hash", {"id": "math"},
        metadata={
            "result_kinds": ["inference"],
            "concept_scope": {"genres": None, "concept_ids": [0, 1]},
        },
    )
    tokenizer = MagicMock()
    tokenizer.side_effect = lambda texts, **kwargs: {
        "input_ids": [text.split() for text in texts]
    }
    monkeypatch.setattr(
        "steerscope.evaluators.output_length.AutoTokenizer.from_pretrained",
        MagicMock(return_value=tokenizer),
    )
    evaluator = OutputLengthEvaluator(
        EvaluatorNode(
            "math_output_length",
            "OutputLengthEvaluator",
            depends_on=("math",),
            concepts={"ids": [1]},
            input={"from": "math", "kind": "inference"},
        ),
        evaluator_context(
            tmp_path,
            results=store.view(("math",)),
            steering_model_name="google/gemma-2-2b-it",
        ),
    )

    result = evaluator.evaluate(
        [],
        [Concept(0, "happy"), Concept(1, "formal")],
    )

    assert result.samples["concept_id"].tolist() == [1]
    assert result.samples["output_tokens"].tolist() == [2]
    assert result.metadata["concept_scope"]["concept_ids"] == [1]


def test_demo_runtime_skips_base_model_resources(tmp_path, monkeypatch):
    runtime = _SteeringTargetRuntime(
        args=evaluator_args(
            models=["DemoModel"],
            steering_model_name="unused/model",
            overwrite_cache=False,
            use_bf16=False,
            demo_interactive=False,
            demo_output="ok",
        ),
        training_args=SimpleNamespace(models={}, overwrite_metadata_dir=None),
        root_dump_dir=tmp_path,
        output_dir=tmp_path / "cache",
        targets=[target("DemoModel")],
        long_format=True,
    )
    load_resources = MagicMock(side_effect=AssertionError("must not load an LLM"))
    predict = MagicMock(return_value={"steered_generation": ["ok"]})
    monkeypatch.setattr(runtime, "_ensure_runtime_resources", load_resources)
    monkeypatch.setattr(DemoModel, "predict_steer", predict)
    examples = pd.DataFrame({
        "input": ["visible prompt"],
        "input_concept": ["formal language"],
        "input_id": [0],
        "factor": [1.0],
    })

    result = runtime.generate_target(
        target("DemoModel"), examples, batch_size=3
    )

    assert result["DemoModel_steered_generation"].tolist() == ["ok"]
    assert predict.call_args.kwargs["batch_size"] == 3
    load_resources.assert_not_called()


def test_model_process_rank_defaults_to_zero_without_process_group(monkeypatch):
    get_rank = MagicMock(return_value=7)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "get_rank", get_rank)

    assert BaseModel().process_rank == 0
    get_rank.assert_not_called()


def test_evaluation_single_process_guard_does_not_initialize_distributed(
    monkeypatch,
):
    init_process_group = MagicMock()
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.distributed, "init_process_group", init_process_group
    )

    ensure_single_process_environment()

    init_process_group.assert_not_called()


def test_evaluation_single_process_guard_rejects_torchrun(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")

    with pytest.raises(RuntimeError, match="must run as one process"):
        ensure_single_process_environment()


def test_evaluation_single_process_guard_rejects_initialized_group(
    monkeypatch,
):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    with pytest.raises(RuntimeError, match="plain python, not torchrun"):
        ensure_single_process_environment()


def test_sft_reenables_gradients_after_lora_unload():
    base_model = torch.nn.Linear(4, 4)
    base_model.requires_grad_(False)
    model = SFT(
        base_model,
        tokenizer=MagicMock(),
        layer=0,
        lm_model_name="google/gemma-2-2b-it",
    )

    model.make_model(mode="train", concept_id=3)
    loss = model.ax_model(torch.ones(1, 4)).sum()

    assert all(parameter.requires_grad for parameter in base_model.parameters())
    assert loss.requires_grad
    assert model.concept_id == 3


@pytest.mark.parametrize("legacy_container", [False, True])
def test_loreft_save_accepts_current_and_legacy_intervention_containers(
    tmp_path, legacy_container
):
    intervention = MagicMock()
    intervention.state_dict.return_value = {
        "rotate_layer": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        "weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "bias": torch.tensor([0.25, -0.25]),
    }
    stored_intervention = [intervention] if legacy_container else intervention
    model = LoReFT.__new__(LoReFT)
    model.ax_model = SimpleNamespace(
        interventions={"layer_20": stored_intervention}
    )

    model.save(tmp_path, model_name="rank_0_LoReFT")

    weights = torch.load(
        tmp_path / "rank_0_LoReFT_weight.pt", weights_only=True
    )
    biases = torch.load(
        tmp_path / "rank_0_LoReFT_bias.pt", weights_only=True
    )
    assert weights["layer_20.proj_weight"].shape == (1, 4, 2)
    assert weights["layer_20.source_weight"].shape == (1, 4, 2)
    assert biases["layer_20.bias"].shape == (1, 2)


@pytest.mark.parametrize("legacy_container", [False, True])
def test_loreft_load_accepts_current_and_legacy_intervention_containers(
    tmp_path, legacy_container
):
    weights = {
        "layer_20.proj_weight": torch.arange(8, dtype=torch.float32).reshape(
            1, 4, 2
        ),
        "layer_20.source_weight": torch.arange(8, dtype=torch.float32).reshape(
            1, 4, 2
        ),
    }
    biases = {"layer_20.bias": torch.tensor([[0.25, -0.25]])}
    torch.save(weights, tmp_path / "LoReFT_weight.pt")
    torch.save(biases, tmp_path / "LoReFT_bias.pt")

    intervention = SimpleNamespace(
        W_proj=torch.nn.Parameter(torch.empty(1, 4, 2)),
        W_source=torch.nn.Parameter(torch.empty(1, 4, 2)),
        b_source=torch.nn.Parameter(torch.empty(1, 2)),
        eval=MagicMock(),
    )
    stored_intervention = [intervention] if legacy_container else intervention
    ax_model = SimpleNamespace(
        interventions={"layer_20": stored_intervention},
        set_device=MagicMock(),
    )
    model = LoReFT.__new__(LoReFT)
    model.device = "cpu"
    model.model = MagicMock()
    model.make_model = MagicMock(side_effect=lambda **_: setattr(model, "ax_model", ax_model))

    model.load(tmp_path)

    assert torch.equal(intervention.W_proj.data, weights["layer_20.proj_weight"])
    assert torch.equal(
        intervention.W_source.data, weights["layer_20.source_weight"]
    )
    assert torch.equal(intervention.b_source.data, biases["layer_20.bias"])
    intervention.eval.assert_called_once_with()


def test_lm_judge_builds_its_own_alpaca_dataset(tmp_path):
    (tmp_path / "alpaca_eval.json").write_text(
        '[{"instruction": "Explain photosynthesis."}]',
        encoding="utf-8",
    )
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.side_effect = lambda messages, **_: [0, 1]
    tokenizer.decode.side_effect = lambda _: "formatted prompt"
    tokenizer.bos_token_id = 0
    evaluator = LMJudgeEvaluator(
        node=EvaluatorNode(
            "judge",
            "LMJudgeEvaluator",
            dataset={"type": "AlpacaEval", "num_examples": 1},
        ),
        context=evaluator_context(tmp_path, master_data_dir=str(tmp_path)),
    )
    evaluator._dataset_tokenizer = tokenizer
    wrapped_model = SimpleNamespace(
        method="DiffMean",
        target=target(),
        concept=target().concept,
    )

    data = evaluator.build_dataset(wrapped_model, [1.0])

    assert data["raw_input"].tolist() == ["Explain photosynthesis."]
    assert "steered_input" not in data.columns
    assert "simple_steered_input" not in data.columns
    assert data["model_factor"].tolist() == [1.0]


def test_alpaca_sampling_is_paired_within_concept_and_varies_by_concept(
    tmp_path,
):
    pd.DataFrame({
        "instruction": [f"prompt {index}" for index in range(40)],
    }).to_json(tmp_path / "alpaca_eval.json", orient="records")
    evaluator = LMJudgeEvaluator(
        node=EvaluatorNode(
            "judge",
            "LMJudgeEvaluator",
            dataset={"type": "AlpacaEval", "num_examples": 10, "seed": 42},
        ),
        context=evaluator_context(tmp_path, master_data_dir=str(tmp_path)),
    )
    evaluator._dataset_tokenizer = MagicMock(
        bos_token_id=None,
        apply_chat_template=MagicMock(return_value=[1]),
        decode=MagicMock(return_value="formatted"),
    )

    def wrapped(method, concept_id):
        current_target = target(method, concept_id)
        return SimpleNamespace(
            method=method,
            target=current_target,
            concept=current_target.concept,
        )

    first = evaluator.build_dataset(wrapped("DiffMean", 3), [0.0, 1.0])
    same_concept = evaluator.build_dataset(wrapped("LoRA", 3), [2.0])
    other_concept = evaluator.build_dataset(wrapped("DiffMean", 9), [1.0])

    first_ids = first[first["factor"] == 0.0]["source_input_id"].tolist()
    assert first_ids == same_concept["source_input_id"].tolist()
    assert first_ids != other_concept["source_input_id"].tolist()
    assert first[first["factor"] == 0.0]["raw_input"].tolist() == first[
        first["factor"] == 1.0
    ]["raw_input"].tolist()


def test_concept_seed_is_stable_and_namespaced():
    assert concept_seed(42, 3, "dataset") == concept_seed(42, 3, "dataset")
    assert concept_seed(42, 3, "dataset") != concept_seed(42, 9, "dataset")
    assert concept_seed(42, 3, "dataset") != concept_seed(42, 3, "other")


def test_report_config_does_not_invalidate_evaluation_hash():
    first = EvaluatorNode(
        "judge",
        "LMJudgeEvaluator",
        report={"formats": ["png"], "dpi": 100},
    )
    second = EvaluatorNode(
        "judge",
        "LMJudgeEvaluator",
        report={"formats": ["pdf"], "dpi": 300},
    )

    assert first.config_hash(context={"model": "same"}) == second.config_hash(
        context={"model": "same"}
    )


def test_lm_judge_renders_evaluator_owned_report(tmp_path):
    evaluator = LMJudgeEvaluator(
        node=EvaluatorNode(
            "judge",
            "LMJudgeEvaluator",
            report={"formats": ["png", "pdf"], "dpi": 72},
        ),
        context=evaluator_context(tmp_path),
    )
    metrics = pd.DataFrame({
        "method": ["DiffMean", "DiffMean", "LsReFT", "LsReFT"] * 2,
        "concept_id": [0] * 4 + [1] * 4,
        "factor": [0.5, 1.0, 0.5, 1.0] * 2,
        "lm_judge_rating": [0.8, 1.1, 1.0, 1.3, 0.9, 1.2, 1.1, 1.4],
        "relevance_concept_ratings": [0.7, 1.2, 1.0, 1.5] * 2,
        "relevance_instruction_ratings": [1.8, 1.7, 1.8, 1.7] * 2,
        "fluency_ratings": [1.9, 1.8, 1.9, 1.8] * 2,
    })

    paths = evaluator.render_report(
        EvaluationResult(metrics=metrics), tmp_path / "judge"
    )

    assert {path.name for path in paths} == {
        "summary.parquet",
        "metrics.png",
        "metrics.pdf",
    }
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    summary = pd.read_parquet(tmp_path / "judge" / "reports" / "summary.parquet")
    assert set(summary["metric"]) == {
        "lm_judge_rating",
        "relevance_concept_ratings",
        "relevance_instruction_ratings",
        "fluency_ratings",
    }


def test_lm_judge_pipeline_overlaps_targets_and_preserves_output_order(tmp_path):
    second_target_generated = threading.Event()
    first_score_started = threading.Event()
    factors = [2.0, 0.5]
    models = []
    for concept_id in (0, 1):
        current_target = target(concept_id=concept_id)
        for factor in factors:
            wrapped = MagicMock(
                target=current_target,
                method="DiffMean",
                concept=current_target.concept,
                factor=factor,
            )

            def generate(examples, current=factor, concept=concept_id):
                if concept == 1:
                    second_target_generated.set()
                return examples.assign(
                    target_id=f"DiffMean/concept-{concept}",
                    method="DiffMean",
                    concept_id=concept,
                    DiffMean_steered_generation=(
                        f"concept {concept} factor {current}"
                    ),
                )

            wrapped.generate.side_effect = generate
            models.append(wrapped)

    evaluator = LMJudgeEvaluator(
        EvaluatorNode(
            "judge",
            "LMJudgeEvaluator",
            params={
                "judge_pipeline_enabled": True,
            },
        ),
        evaluator_context(tmp_path),
    )
    evaluator.open_resources = MagicMock()
    evaluator.close_resources = MagicMock()
    evaluator._open_model_progress = MagicMock()
    evaluator._close_model_progress = MagicMock()
    evaluator.build_dataset = MagicMock(side_effect=lambda model, current: pd.DataFrame({
        "factor": current,
        "model_factor": current,
        "input": [f"concept {model.concept.concept_id}"] * len(current),
    }))

    def score(model, inference):
        if model.concept.concept_id == 0:
            first_score_started.set()
            assert second_target_generated.wait(timeout=2.0)
        scored_factors = sorted(inference["factor"].unique())
        return (
            pd.DataFrame({
                "target_id": model.target.target_id,
                "factor": scored_factors,
                "score": scored_factors,
            }),
            inference.assign(scored=True),
        )

    evaluator.score = score

    result = evaluator.evaluate(
        models,
        [target(concept_id=0).concept, target(concept_id=1).concept],
    )

    assert first_score_started.is_set()
    assert second_target_generated.is_set()
    assert result.inference["concept_id"].tolist() == [0, 0, 1, 1]
    assert result.samples["concept_id"].tolist() == [0, 0, 1, 1]
    assert result.metrics["target_id"].tolist() == [
        "DiffMean/concept-0",
        "DiffMean/concept-0",
        "DiffMean/concept-1",
        "DiffMean/concept-1",
    ]


def test_lm_judge_can_score_base_model_as_fixed_method_factor_zero(tmp_path):
    wrapped = MagicMock(
        target=target("SFT"),
        method="SFT",
        concept=target("SFT").concept,
        factor=1.0,
    )
    wrapped.generate_baseline.side_effect = lambda examples: examples.assign(
        baseline_generation="base"
    )
    wrapped.generate.side_effect = lambda examples: examples.assign(
        target_id="SFT/concept-0",
        method="SFT",
        SFT_steered_generation="fine-tuned",
    )
    evaluator = LMJudgeEvaluator(
        EvaluatorNode(
            "judge",
            "LMJudgeEvaluator",
            params={
                "include_baseline": True,
                "baseline_factor": 0.0,
                "judge_pipeline_enabled": False,
            },
        ),
        evaluator_context(tmp_path),
    )
    evaluator.build_dataset = MagicMock(return_value=pd.DataFrame({
        "factor": [0.0, 1.0],
        "model_factor": [0.0, 1.0],
        "input": ["same prompt", "same prompt"],
    }))
    evaluator.prepare_examples = lambda examples, **_: examples

    def compute_metrics(inference):
        factors = inference["factor"].drop_duplicates().tolist()
        return {"factor": factors, "score": factors}

    evaluator.compute_metrics = compute_metrics

    result = evaluator.evaluate_target([wrapped])

    wrapped.generate_baseline.assert_called_once()
    wrapped.generate.assert_called_once()
    assert result.inference["model_factor"].tolist() == [0.0, 1.0]
    assert result.inference["SFT_steered_generation"].tolist() == [
        "base",
        "fine-tuned",
    ]
    assert result.samples["factor"].tolist() == [0.0, 1.0]
    assert result.metrics["method"].tolist() == ["SFT", "SFT"]
    assert result.metrics["factor"].tolist() == [0.0, 1.0]


def test_lm_judge_baseline_only_skips_candidate_generation(tmp_path):
    wrapped = MagicMock(
        target=target("SimplePromptSteering"),
        method="SimplePromptSteering",
        concept=target("SimplePromptSteering").concept,
        factor=1.0,
    )
    wrapped.generate_baseline.side_effect = lambda examples: examples.assign(
        baseline_generation="base"
    )
    evaluator = LMJudgeEvaluator(
        EvaluatorNode(
            "id_lm_judge",
            "LMJudgeEvaluator",
            params={"baseline_only": True, "baseline_factor": 0.0},
        ),
        evaluator_context(tmp_path),
    )
    evaluator.build_dataset = MagicMock(return_value=pd.DataFrame({
        "factor": [0.0],
        "model_factor": [0.0],
        "input": ["same prompt"],
    }))
    evaluator.prepare_examples = lambda examples, **_: examples
    evaluator.compute_metrics = lambda inference: {
        "factor": [float(inference["factor"].iloc[0])],
        "score": [1.0],
    }

    result = evaluator.evaluate_target([wrapped])

    wrapped.generate_baseline.assert_called_once()
    wrapped.generate.assert_not_called()
    assert result.inference["model_factor"].tolist() == [0.0]
    assert result.inference[
        "SimplePromptSteering_steered_generation"
    ].tolist() == ["base"]
    assert result.metrics["factor"].tolist() == [0.0]


def test_request_cache_dir_can_be_shared_across_result_stores(tmp_path):
    first = ResultStore(tmp_path / "first", run_id="one")
    second = ResultStore(tmp_path / "second", run_id="two")
    shared = tmp_path / "shared" / "requests"
    args = SimpleNamespace(
        shared_request_cache_dir=str(shared),
        config_file=str(tmp_path / "config.yaml"),
    )

    assert _request_cache_dir(args, first) == shared.resolve()
    assert _request_cache_dir(args, second) == shared.resolve()
    assert shared.is_dir()


def test_progress_root_mirrors_evaluate_output_below_source_root(tmp_path):
    source = tmp_path / "output"
    evaluate_dump = source / "methods/apsr/evaluate"
    args = SimpleNamespace(
        progress_root=str(tmp_path / "local-progress"),
        progress_source_root=str(source),
        config_file=str(tmp_path / "config.yaml"),
    )

    assert _progress_store_root(args, evaluate_dump) == (
        tmp_path / "local-progress/methods/apsr/evaluate/runs"
    ).resolve()


def test_mirrored_progress_root_keeps_target_writes_out_of_result_tree(tmp_path):
    result_root = tmp_path / "nfs/runs"
    progress_root = tmp_path / "local/runs"
    store = ResultStore(
        result_root, run_id="demo", progress_root=progress_root
    )
    node = EvaluatorNode("judge", "RecordingEvaluator")
    store.mark_running(node.node_id, "same", node.as_dict())
    progress = store.progress(node.node_id)
    progress.begin(["first"])
    progress.save_target(
        "first", EvaluationResult(metrics=pd.DataFrame({"score": [1.0]}))
    )

    assert progress.root.is_relative_to(progress_root)
    assert not (store.node_dir(node.node_id) / "progress").exists()
    raw_manifest = json.loads(
        (store.node_dir(node.node_id) / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "progress" not in (raw_manifest.get("metadata") or {})
    assert store.manifest(node.node_id)["metadata"]["progress"][
        "completed_targets"
    ] == 1

    resumed = ResultStore(
        result_root, run_id="demo", progress_root=progress_root
    )
    restored = resumed.progress(node.node_id).load_target("first")
    assert restored.metrics["score"].tolist() == [1.0]


def test_mirrored_progress_root_seeds_legacy_checkpoint_once(tmp_path):
    result_root = tmp_path / "nfs/runs"
    legacy = ResultStore(result_root, run_id="demo")
    node = EvaluatorNode("judge", "RecordingEvaluator")
    legacy.mark_running(node.node_id, "same", node.as_dict())
    legacy_progress = legacy.progress(node.node_id)
    legacy_progress.begin(["first"])
    legacy_progress.save_target(
        "first", EvaluationResult(metrics=pd.DataFrame({"score": [1.0]}))
    )

    local_root = tmp_path / "local/runs"
    migrated = ResultStore(
        result_root, run_id="demo", progress_root=local_root
    )
    restored = migrated.progress(node.node_id).load_target("first")

    assert restored.metrics["score"].tolist() == [1.0]
    assert legacy_progress.root.is_dir()
    assert (
        migrated.progress_execution_dir(node.node_id, "same")
        / ".legacy_seed_complete.json"
    ).is_file()


def test_lm_judge_pipeline_recovers_pending_inference_without_regeneration(
    tmp_path,
):
    node = EvaluatorNode(
        "judge",
        "LMJudgeEvaluator",
        dataset={"type": "AlpacaEval", "num_examples": 1},
        inference={"strengths": [1.0]},
    )
    store = ResultStore(tmp_path / "runs")
    execution_hash = "same-execution"
    store.mark_running(node.node_id, execution_hash, node.as_dict())
    targets = [target(concept_id=index) for index in (0, 1)]

    def make_models(fail_on_generate=False):
        wrapped_models = []
        for current_target in targets:
            wrapped = MagicMock(
                target=current_target,
                method="DiffMean",
                concept=current_target.concept,
                factor=1.0,
            )
            if fail_on_generate:
                wrapped.generate.side_effect = AssertionError(
                    "recovery must not repeat inference"
                )
            else:
                wrapped.generate.side_effect = (
                    lambda examples, current=current_target: examples.assign(
                        target_id=current.target_id,
                        method="DiffMean",
                        concept_id=current.concept.concept_id,
                        DiffMean_steered_generation="generated",
                    )
                )
            wrapped_models.append(wrapped)
        return wrapped_models

    def configure(evaluator):
        evaluator.open_resources = MagicMock()
        evaluator.close_resources = MagicMock()
        evaluator.checkpoint_resources = MagicMock()
        evaluator._open_model_progress = MagicMock()
        evaluator._close_model_progress = MagicMock()
        evaluator.build_dataset = MagicMock(side_effect=lambda model, _: pd.DataFrame({
            "factor": [1.0],
            "model_factor": [1.0],
            "input": [f"concept {model.concept.concept_id}"],
        }))

    first_models = make_models()
    first = LMJudgeEvaluator(
        node,
        evaluator_context(
            tmp_path,
            progress=store.progress(node.node_id),
        ),
    )
    configure(first)

    def fail_first(model, inference):
        if model.concept.concept_id == 0:
            raise RuntimeError("judge unavailable")
        return (
            pd.DataFrame({"concept_id": [model.concept.concept_id]}),
            inference.copy(),
        )

    first.score = fail_first
    with pytest.raises(RuntimeError, match="judge unavailable"):
        first.evaluate(first_models, [item.concept for item in targets])

    assert _InferenceSpool(
        store.progress(node.node_id)
    ).load(targets[0].target_id) is not None
    assert store.progress(node.node_id).load_target(
        targets[1].target_id
    ) is not None
    assert all(model.generate.call_count == 1 for model in first_models)

    store.mark_failed(
        node.node_id, execution_hash, node.as_dict(), "judge unavailable"
    )
    store.mark_running(node.node_id, execution_hash, node.as_dict())
    resumed_models = make_models(fail_on_generate=True)
    resumed = LMJudgeEvaluator(
        node,
        evaluator_context(
            tmp_path,
            progress=store.progress(node.node_id),
        ),
    )
    configure(resumed)
    resumed.score = lambda model, inference: (
        pd.DataFrame({"concept_id": [model.concept.concept_id]}),
        inference.copy(),
    )

    result = resumed.evaluate(
        resumed_models, [item.concept for item in targets]
    )

    assert result.metrics["concept_id"].tolist() == [0, 1]
    assert all(model.generate.call_count == 0 for model in resumed_models)
    assert _InferenceSpool(
        store.progress(node.node_id)
    ).load(targets[0].target_id) is None



@pytest.mark.parametrize(
    ("evaluator_class", "node_id", "metric", "values"),
    [
        (PerplexityEvaluator, "perplexity", "perplexity", [0.1, 10.0, 0.2, 12.0]),
        (RuleEvaluator, "rule", "rule_following", [0.0, 1.0, 0.5, 1.0]),
    ],
)
def test_curve_evaluators_render_reports(
    tmp_path, evaluator_class, node_id, metric, values
):
    evaluator = evaluator_class(
        node=EvaluatorNode(
            node_id,
            evaluator_class.__name__,
            report={"formats": ["png"], "dpi": 72},
        ),
        context=evaluator_context(tmp_path),
    )
    metrics = pd.DataFrame({
        "method": ["DiffMean"] * 4,
        "concept_id": [0, 1, 0, 1],
        "factor": [0.5, 0.5, 1.0, 1.0],
        metric: values,
    })

    paths = evaluator.render_report(
        EvaluationResult(metrics=metrics), tmp_path / node_id
    )

    assert {path.name for path in paths} == {"summary.parquet", "metrics.png"}
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)


def test_curve_reports_assign_distinct_colors_to_many_methods():
    from matplotlib import pyplot as plt

    methods = [f"method-{index:02d}" for index in range(25)]
    styles = Evaluator._method_plot_styles(methods, plt)
    colors = [tuple(styles[method]["color"]) for method in methods]

    assert len(colors) == len(set(colors))
    assert styles == Evaluator._method_plot_styles(list(reversed(methods)), plt)


def test_rule_report_uses_full_rating_scale(tmp_path):
    evaluator = RuleEvaluator(
        node=EvaluatorNode("rule", "RuleEvaluator"),
        context=evaluator_context(tmp_path),
    )
    evaluator._render_curve_report = MagicMock(return_value=[])
    result = EvaluationResult(metrics=pd.DataFrame())

    evaluator.render_report(result, tmp_path / "rule")

    assert evaluator._render_curve_report.call_args.kwargs["y_limits"] == (
        0.0,
        2.0,
    )


def test_rule_evaluator_rejects_unknown_rule(tmp_path):
    evaluator = RuleEvaluator(
        node=EvaluatorNode("rule", "RuleEvaluator"),
        context=evaluator_context(tmp_path),
    )

    with pytest.raises(ValueError, match="Unknown rule type"):
        evaluator.compute_metrics(pd.DataFrame(), rule_type="not-a-rule")


def test_jailbreakbench_renders_rate_report(tmp_path):
    evaluator = JailBreakBenchEvaluator(
        node=EvaluatorNode(
            "jbb",
            "JailBreakBenchEvaluator",
            params={
                "harmful_judge_model_name": "harmful/judge",
                "benign_judge_model_name": "benign/judge",
            },
            report={"formats": ["png"], "dpi": 72},
        ),
        context=evaluator_context(tmp_path),
    )
    metrics = pd.DataFrame({
        "method": ["DiffMean"] * 4,
        "concept_id": [0, 1, 0, 1],
        "factor": [0.0, 0.0, 1.0, 1.0],
        "attack_success_rate": [0.4, 0.6, None, None],
        "false_refusal_rate": [None, None, 0.2, 0.4],
    })

    paths = evaluator.render_report(
        EvaluationResult(metrics=metrics), tmp_path / "jbb"
    )

    assert {path.name for path in paths} == {"summary.parquet", "metrics.png"}
    summary = pd.read_parquet(tmp_path / "jbb" / "reports" / "summary.parquet")
    assert set(summary["metric"]) == {
        "attack_success_rate",
        "false_refusal_rate",
    }


def test_winrate_report_accepts_model_factor_without_factor(tmp_path):
    evaluator = WinRateEvaluator(
        node=EvaluatorNode(
            "winrate",
            "WinRateEvaluator",
            report={"formats": ["png"], "dpi": 72},
        ),
        context=evaluator_context(tmp_path),
    )
    metrics = pd.DataFrame({
        "model_factor": [0.5, 1.0],
        "win_rate": [0.4, 0.6],
        "loss_rate": [0.4, 0.2],
        "tie_rate": [0.2, 0.2],
    })

    paths = evaluator.render_report(
        EvaluationResult(metrics=metrics), tmp_path / "winrate"
    )

    assert {path.name for path in paths} == {"summary.parquet", "metrics.png"}
    summary = pd.read_parquet(
        tmp_path / "winrate" / "reports" / "summary.parquet"
    )
    assert summary["model_factor"].unique().tolist() == [0.5, 1.0]


def test_best_factor_single_method_renders_report(tmp_path):
    evaluator = BestFactorEvaluator(
        node=EvaluatorNode(
            "best_factor",
            "BestFactorEvaluator",
            requires_inference=False,
            report={"formats": ["png"], "dpi": 72},
        ),
        context=evaluator_context(tmp_path),
    )
    metrics = pd.DataFrame({
        "method": ["DiffMean", "DiffMean"],
        "concept_id": [0, 1],
        "factor": [0.5, 1.0],
        "selected_score": [0.8, 0.9],
    })

    paths = evaluator.render_report(
        EvaluationResult(metrics=metrics), tmp_path / "best_factor"
    )

    assert {path.name for path in paths} == {"summary.parquet", "metrics.png"}
    summary = pd.read_parquet(
        tmp_path / "best_factor" / "reports" / "summary.parquet"
    )
    assert summary.loc[0, "method"] == "DiffMean"
    assert summary.loc[0, "mean_factor"] == pytest.approx(0.75)


def test_report_requires_metric_rows(tmp_path):
    evaluator = PerplexityEvaluator(
        node=EvaluatorNode("perplexity", "PerplexityEvaluator"),
        context=evaluator_context(tmp_path),
    )

    with pytest.raises(ValueError, match="without metric rows"):
        evaluator.render_report(EvaluationResult(metrics=None), tmp_path)


def test_engine_persists_evaluator_result_and_invalidates_dependents(tmp_path):
    nodes = [
        EvaluatorNode("second", "Metric", depends_on=("first",), requires_inference=False),
        EvaluatorNode("first", "Metric"),
    ]
    store = ResultStore(tmp_path)
    calls = []

    def execute(node, results):
        calls.append(node.node_id)
        if node.node_id == "second":
            assert results.metrics("first")["score"].tolist() == [1.0]
        return EvaluationResult(metrics=pd.DataFrame({"score": [1.0]}))

    engine = EvaluationEngine(nodes, store, execute)
    assert engine.run() == ["first", "second"]
    assert calls == ["first", "second"]
    assert store.load("first", "metrics")["score"].tolist() == [1.0]

    calls.clear()
    engine.run()
    assert calls == []

    store._frame_path("first", "metrics").unlink()
    engine.run()
    assert calls == ["first"]


def test_report_failure_does_not_fail_completed_evaluation(tmp_path):
    node = EvaluatorNode("judge", "RecordingEvaluator")
    store = ResultStore(tmp_path / "runs")

    def execute(node, results):
        return EvaluationResult(metrics=pd.DataFrame({"score": [1.0]}))

    def fail_report(node, result):
        raise RuntimeError("plotting failed")

    engine = EvaluationEngine(
        [node],
        store,
        execute,
        reporter=fail_report,
    )

    assert engine.run() == ["judge"]
    manifest = store.manifest("judge")
    assert manifest["status"] == "complete"
    assert manifest["metadata"]["report"]["status"] == "failed"
    assert "plotting failed" in manifest["metadata"]["report"]["error"]


def test_report_only_loads_persisted_result_without_execution(tmp_path):
    node = EvaluatorNode("judge", "RecordingEvaluator")
    store = ResultStore(tmp_path / "runs")
    execution_hash = node.config_hash()
    store.mark_running(node.node_id, execution_hash, node.as_dict())
    store.save_metrics(node.node_id, pd.DataFrame({"score": [1.0]}))
    store.mark_complete(
        node.node_id,
        execution_hash,
        node.as_dict(),
        metadata={"result_kinds": ["metrics"], "metrics_rows": 1},
    )
    executor = MagicMock(side_effect=AssertionError("must not execute"))
    reporter = MagicMock(return_value=[])
    engine = EvaluationEngine(
        [node],
        store,
        executor,
        reporter=reporter,
        report_only=True,
    )

    assert engine.run() == ["judge"]
    executor.assert_not_called()
    reported_result = reporter.call_args.args[1]
    assert reported_result.metrics["score"].tolist() == [1.0]
    assert store.manifest("judge")["metadata"]["report"]["status"] == "complete"


def test_disabled_report_does_not_load_cached_result(tmp_path):
    node = EvaluatorNode(
        "judge",
        "RecordingEvaluator",
        report={"enabled": False},
    )
    store = ResultStore(tmp_path / "runs")
    execution_hash = node.config_hash()
    store.mark_running(node.node_id, execution_hash, node.as_dict())
    store.save_metrics(node.node_id, pd.DataFrame({"score": [1.0]}))
    store.mark_complete(
        node.node_id,
        execution_hash,
        node.as_dict(),
        metadata={"result_kinds": ["metrics"], "metrics_rows": 1},
    )
    reporter = MagicMock()
    store.load_result = MagicMock(
        side_effect=AssertionError("disabled report must not load results")
    )
    engine = EvaluationEngine(
        [node],
        store,
        MagicMock(side_effect=AssertionError("must not execute")),
        reporter=reporter,
    )

    assert engine.run() == ["judge"]
    reporter.assert_not_called()
    store.load_result.assert_not_called()


def test_evaluator_checkpoints_each_target_and_resumes_after_failure(tmp_path):
    node = EvaluatorNode(
        "judge",
        "RecordingEvaluator",
        dataset={"type": "AlpacaEval", "num_examples": 1},
        inference={"strengths": [1.0]},
    )
    store = ResultStore(tmp_path / "runs")
    execution_hash = "same-execution"
    store.mark_running(node.node_id, execution_hash, node.as_dict())

    first_target = target(concept_id=0)
    second_target = target(concept_id=1)
    first = MagicMock(
        target=first_target,
        method="DiffMean",
        concept=first_target.concept,
        factor=1.0,
    )
    second = MagicMock(
        target=second_target,
        method="DiffMean",
        concept=second_target.concept,
        factor=1.0,
    )
    first.generate.side_effect = lambda examples: examples.assign(
        method="DiffMean",
        target_id=first_target.target_id,
        DiffMean_steered_generation="first",
    )
    second.generate.side_effect = RuntimeError("interrupted")
    examples = pd.DataFrame({
        "factor": [1.0],
        "model_factor": [1.0],
        "input": ["prompt"],
    })
    evaluator = RecordingEvaluator(
        node=node,
        context=evaluator_context(
            tmp_path,
            progress=store.progress(node.node_id),
        ),
    )
    evaluator.examples = examples

    with pytest.raises(RuntimeError, match="interrupted"):
        evaluator.evaluate(
            [first, second], [first_target.concept, second_target.concept]
        )

    checkpointed = store.progress(node.node_id).load_target(first_target.target_id)
    assert checkpointed.metrics["target_id"].tolist() == [
        first_target.target_id
    ]
    with pytest.raises(FileNotFoundError):
        store.load("judge", "metrics")
    manifest = store.manifest("judge")
    assert manifest["metadata"]["progress"]["completed_targets"] == 1

    store.mark_failed(node.node_id, execution_hash, node.as_dict(), "interrupted")
    store.mark_running(node.node_id, execution_hash, node.as_dict())
    resumed_first = MagicMock(
        target=first_target,
        method="DiffMean",
        concept=first_target.concept,
        factor=1.0,
    )
    resumed_second = MagicMock(
        target=second_target,
        method="DiffMean",
        concept=second_target.concept,
        factor=1.0,
    )
    resumed_second.generate.side_effect = lambda examples: examples.assign(
        method="DiffMean",
        target_id=second_target.target_id,
        DiffMean_steered_generation="second",
    )
    resumed = RecordingEvaluator(
        node=node,
        context=evaluator_context(
            tmp_path,
            progress=store.progress(node.node_id),
        ),
    )
    resumed.examples = examples

    result = resumed.evaluate(
        [resumed_first, resumed_second],
        [first_target.concept, second_target.concept],
    )

    resumed_first.generate.assert_not_called()
    resumed_second.generate.assert_called_once()
    assert sorted(result.metrics["target_id"].tolist()) == [
        first_target.target_id,
        second_target.target_id,
    ]
    assert len(result.inference) == 2
    with pytest.raises(FileNotFoundError):
        store.load("judge", "inference")
    assert store.manifest("judge")["metadata"]["progress"][
        "completed_targets"
    ] == 2


def test_progress_save_target_does_not_reload_or_materialize_prior_targets(tmp_path):
    store = ResultStore(tmp_path / "runs")
    node = EvaluatorNode("judge", "RecordingEvaluator")
    execution_hash = "same-execution"
    store.mark_running(node.node_id, execution_hash, node.as_dict())
    progress = store.progress(node.node_id)
    progress.begin(["first", "second"])
    progress.completed_target_ids = MagicMock(
        side_effect=AssertionError("save_target must not rescan target frames")
    )
    progress.materialize = MagicMock(
        side_effect=AssertionError("save_target must not materialize aggregate frames")
    )

    for target_id, score in (("first", 1.0), ("second", 2.0)):
        progress.save_target(
            target_id,
            EvaluationResult(metrics=pd.DataFrame({"score": [score]})),
        )

    progress.completed_target_ids.assert_not_called()
    progress.materialize.assert_not_called()
    manifest = store.manifest(node.node_id)
    assert manifest["metadata"]["progress"]["completed_target_ids"] == [
        "first",
        "second",
    ]
    with pytest.raises(FileNotFoundError):
        store.load(node.node_id, "metrics")


def test_progress_begin_recovers_target_committed_before_progress_manifest(tmp_path):
    store = ResultStore(tmp_path / "runs")
    node = EvaluatorNode("judge", "RecordingEvaluator")
    execution_hash = "same-execution"
    store.mark_running(node.node_id, execution_hash, node.as_dict())
    progress = store.progress(node.node_id)
    progress.begin(["first"])
    progress._update_node_manifest = MagicMock(
        side_effect=RuntimeError("interrupted after target commit")
    )

    with pytest.raises(RuntimeError, match="interrupted after target commit"):
        progress.save_target(
            "first",
            EvaluationResult(metrics=pd.DataFrame({"score": [1.0]})),
        )

    assert progress.load_target("first").metrics["score"].tolist() == [1.0]
    assert store.manifest(node.node_id)["metadata"]["progress"][
        "completed_targets"
    ] == 0

    recovered = store.progress(node.node_id)
    recovered.begin(["first"])

    assert store.manifest(node.node_id)["metadata"]["progress"][
        "completed_target_ids"
    ] == ["first"]


def test_engine_materializes_combined_target_results_once(tmp_path):
    store = ResultStore(tmp_path / "runs")
    node = EvaluatorNode("judge", "RecordingEvaluator")
    original_save_frame = store._save_frame
    store._save_frame = MagicMock(wraps=original_save_frame)

    def execute(current_node, _results):
        progress = store.progress(current_node.node_id)
        progress.begin(["first", "second"])
        for target_id, score in (("first", 1.0), ("second", 2.0)):
            progress.save_target(
                target_id,
                EvaluationResult(metrics=pd.DataFrame({"score": [score]})),
            )
        return progress.combine(["first", "second"])

    engine = EvaluationEngine(
        [node],
        store,
        execute,
        generate_reports=False,
    )

    assert engine.run() == [node.node_id]
    assert store._save_frame.call_count == 1
    assert store.load(node.node_id, "metrics")["score"].tolist() == [1.0, 2.0]


def test_evaluator_delegates_checkpoint_metadata_aggregation(tmp_path):
    class MetadataEvaluator(RecordingEvaluator):
        def combine_metadata(self, target_metadata):
            return {
                "custom_total": sum(
                    metadata.get("custom_value", 0)
                    for metadata in target_metadata
                )
            }

    store = ResultStore(tmp_path / "runs")
    node = EvaluatorNode("custom", "MetadataEvaluator")
    execution_hash = "same-execution"
    store.mark_running(node.node_id, execution_hash, node.as_dict())
    progress = store.progress(node.node_id)
    progress.begin(["first", "second"])
    progress.save_target(
        "first",
        EvaluationResult(
            metrics=pd.DataFrame({"score": [1.0]}),
            metadata={"custom_value": 2},
        ),
    )
    progress.save_target(
        "second",
        EvaluationResult(
            metrics=pd.DataFrame({"score": [2.0]}),
            metadata={"custom_value": 3},
        ),
    )
    evaluator = MetadataEvaluator(node=node, context=evaluator_context(tmp_path))

    result = progress.combine(
        ["first", "second"],
        metadata_combiner=evaluator.combine_metadata,
    )

    assert result.metadata == {"custom_total": 5}


def test_engine_evaluator_adapter_only_calls_public_evaluate(tmp_path):
    node = EvaluatorNode("judge", "RecordingEvaluator")
    evaluator = MagicMock()
    evaluator.evaluate.return_value = EvaluationResult(
        metrics=pd.DataFrame({"score": [1.0]})
    )
    evaluator_factory = MagicMock(return_value=evaluator)
    models = [MagicMock()]
    concepts = [Concept(0, "formal language")]
    engine = EvaluationEngine.for_evaluators(
        [node],
        ResultStore(tmp_path),
        evaluator_factory=evaluator_factory,
        models_factory=MagicMock(return_value=(models, concepts)),
    )

    engine.run()

    evaluator.evaluate.assert_called_once_with(models, concepts)


def test_engine_evaluator_adapter_prefers_progress_resume(tmp_path):
    node = EvaluatorNode("judge", "RecordingEvaluator")
    restored = EvaluationResult(metrics=pd.DataFrame({"score": [2.0]}))
    evaluator = MagicMock()
    evaluator.resume_from_progress.return_value = restored
    evaluator_factory = MagicMock(return_value=evaluator)
    models_factory = MagicMock(side_effect=AssertionError(
        "progress resume must not construct model wrappers"
    ))
    targets_factory = MagicMock(return_value=(
        [MagicMock()], [Concept(0, "formal language")]
    ))
    engine = EvaluationEngine.for_evaluators(
        [node],
        ResultStore(tmp_path),
        evaluator_factory=evaluator_factory,
        models_factory=models_factory,
        targets_factory=targets_factory,
    )

    engine.run()

    evaluator.resume_from_progress.assert_called_once()
    evaluator.evaluate.assert_not_called()
    models_factory.assert_not_called()


def test_result_only_engine_adapter_requests_no_models(tmp_path):
    source = EvaluatorNode("search", "RecordingEvaluator")
    node = EvaluatorNode(
        "best",
        "BestFactorEvaluator",
        depends_on=("search",),
        requires_inference=False,
    )
    evaluator = MagicMock()
    evaluator.evaluate.return_value = EvaluationResult(
        metrics=pd.DataFrame({"factor": [1.0]})
    )
    source_evaluator = MagicMock()
    source_evaluator.evaluate.return_value = EvaluationResult(
        metrics=pd.DataFrame({"score": [1.0]})
    )
    evaluator_factory = MagicMock(
        side_effect=lambda configured, _: (
            evaluator if configured.node_id == "best" else source_evaluator
        )
    )
    models_factory = MagicMock(
        side_effect=lambda configured: (
            ([], []) if configured.node_id == "best" else ([MagicMock()], [])
        )
    )
    engine = EvaluationEngine.for_evaluators(
        [source, node],
        ResultStore(tmp_path),
        evaluator_factory=evaluator_factory,
        models_factory=models_factory,
    )

    engine.run()

    assert models_factory.call_args_list[-1].args == (node,)
    evaluator.evaluate.assert_called_once_with([], [])


def test_evaluator_owns_dataset_construction_and_model_inference(
    tmp_path, monkeypatch
):
    progress_bar = MagicMock()
    progress_factory = MagicMock(return_value=progress_bar)
    monkeypatch.setattr("steerscope.evaluators.evaluator.tqdm", progress_factory)
    examples = pd.DataFrame({
        "concept_id": [0, 0],
        "factor": [0.5, 1.0],
        "model_factor": [0.5, 1.0],
        "input": ["a", "b"],
    })
    wrapped_models = []
    for factor, generation in ((0.5, "x"), (1.0, "y")):
        wrapped_model = MagicMock()
        wrapped_model.target = target()
        wrapped_model.method = "DiffMean"
        wrapped_model.concept = wrapped_model.target.concept
        wrapped_model.factor = factor
        wrapped_model.generate.side_effect = lambda examples, value=generation: examples.assign(
            method="DiffMean",
            target_id="DiffMean/concept-0",
            DiffMean_steered_generation=value,
        )
        wrapped_models.append(wrapped_model)
    evaluator = RecordingEvaluator(
        node=EvaluatorNode(
            "judge",
            "RecordingEvaluator",
            dataset={"type": "AlpacaEval", "num_examples": 2},
            inference={"strengths": [0.5, 1.0]},
        ),
        context=evaluator_context(tmp_path),
    )
    evaluator.examples = examples
    evaluator.open_resources = MagicMock()
    evaluator.close_resources = MagicMock()

    result = evaluator.evaluate(wrapped_models, [wrapped_models[0].concept])

    evaluator.open_resources.assert_called_once_with(wrapped_models)
    evaluator.close_resources.assert_called_once_with()
    for wrapped_model in wrapped_models:
        wrapped_model.generate.assert_called_once()
        assert len(wrapped_model.generate.call_args.args) == 1
        assert wrapped_model.generate.call_args.args[0]["model_factor"].unique().tolist() == [
            wrapped_model.factor
        ]
    assert result.inference["DiffMean_steered_generation"].tolist() == ["x", "y"]
    assert result.metrics["score"].tolist() == [1.0, 1.0]
    assert progress_factory.call_args.kwargs["total"] == 2
    assert [item.args for item in progress_bar.update.call_args_list] == [
        (1,),
        (1,),
    ]
    progress_bar.close.assert_called_once()


def test_evaluator_closes_partial_resources_when_open_fails(tmp_path):
    evaluator = RecordingEvaluator(
        node=EvaluatorNode("judge", "RecordingEvaluator"),
        context=evaluator_context(tmp_path),
    )
    model = MagicMock(
        target=target(),
        method="DiffMean",
        concept=target().concept,
        factor=1.0,
    )
    evaluator.open_resources = MagicMock(
        side_effect=RuntimeError("resource setup failed")
    )
    evaluator.close_resources = MagicMock()

    with pytest.raises(RuntimeError, match="resource setup failed"):
        evaluator.evaluate([model], [model.concept])

    evaluator.close_resources.assert_called_once_with()


def test_evaluator_keeps_dataset_factor_separate_from_model_factor(tmp_path):
    examples = pd.DataFrame({
        "factor": [-0.5],
        "model_factor": [0.5],
        "input": ["a"],
    })
    model = MagicMock(
        target=target(),
        method="DiffMean",
        concept=target().concept,
        factor=0.5,
    )
    model.generate.side_effect = lambda examples: examples.assign(
        DiffMean_steered_generation="x"
    )
    evaluator = RecordingEvaluator(
        node=EvaluatorNode("judge", "RecordingEvaluator"),
        context=evaluator_context(tmp_path),
    )
    evaluator.examples = examples

    result = evaluator.evaluate([model], [model.concept])

    generated_input = model.generate.call_args.args[0]
    assert generated_input["factor"].tolist() == [-0.5]
    assert generated_input["model_factor"].tolist() == [0.5]
    assert result.metrics["factor"].tolist() == [-0.5]


def test_engine_rejects_missing_dependencies_and_cycles(tmp_path):
    with pytest.raises(ValueError, match="unknown dependencies"):
        EvaluationEngine(
            [EvaluatorNode("node", "Metric", depends_on=("missing",))],
            ResultStore(tmp_path / "missing"),
            lambda *_: EvaluationResult(),
        )
    engine = EvaluationEngine(
        [
            EvaluatorNode("a", "Metric", depends_on=("b",)),
            EvaluatorNode("b", "Metric", depends_on=("a",)),
        ],
        ResultStore(tmp_path / "cycle"),
        lambda *_: EvaluationResult(),
    )
    with pytest.raises(ValueError, match="cycle"):
        engine.execution_order()


def test_result_view_only_reads_declared_dependencies(tmp_path):
    store = ResultStore(tmp_path)
    store.save_metrics("allowed", pd.DataFrame({"method": ["A", "B"], "score": [1, 2]}))
    view = store.view(["allowed"])
    assert view.metrics("allowed", filters={"method": "B"})["score"].tolist() == [2]
    with pytest.raises(ValueError, match="not a declared dependency"):
        view.metrics("other")


def test_evaluator_selects_factor_for_each_target(tmp_path):
    store = ResultStore(tmp_path)
    store.save_metrics("search", pd.DataFrame({
        "target_id": ["A/concept-0", "A/concept-0"],
        "factor": [0.5, 1.0],
        "score": [0.2, 0.8],
    }))
    models = [
        MagicMock(
            target=target("A"),
            method="A",
            concept=target("A").concept,
            factor=factor,
        )
        for factor in (0.5, 1.0)
    ]
    evaluator = RecordingEvaluator(
        node=EvaluatorNode(
            "test",
            "RecordingEvaluator",
            depends_on=("search",),
            inference={"select": {"from": "search", "metric": "score"}},
        ),
        context=evaluator_context(tmp_path, results=store.view(["search"])),
    )
    selected = evaluator.select_models(models, store.view(["search"]))
    assert [model.factor for model in selected] == [1.0]


def test_evaluator_consumes_method_selection_and_per_method_control_factor(tmp_path):
    store = ResultStore(tmp_path)
    store.save_metrics("best", pd.DataFrame({
        "method": ["A", "B"],
        "factor": [1.0, 2.0],
        "selected_score": [0.8, 0.9],
    }))
    models = []
    for method, concept_id in (("A", 0), ("A", 1), ("B", 0)):
        current_target = target(method, concept_id)
        models.extend([
            SimpleNamespace(
                target=current_target,
                method=method,
                concept=current_target.concept,
                factor=factor,
            )
            for factor in (0.0, 1.0, 2.0)
        ])
    evaluator = RecordingEvaluator(
        node=EvaluatorNode(
            "test",
            "RecordingEvaluator",
            depends_on=("best",),
            inference={"select": {
                "from": "best",
                "metric": "selected_score",
                "include_factors": [0.0],
                "include_factors_by_model": {"B": []},
            }},
        ),
        context=evaluator_context(tmp_path, results=store.view(["best"])),
    )

    scheduled = evaluator.evaluation_models(models)
    assert scheduled == models
    expected = {"A": [0.0, 1.0], "B": [2.0]}
    for target_models in evaluator._models_by_target(scheduled):
        selected = evaluator.select_models(
            target_models, store.view(["best"])
        )
        assert [model.factor for model in selected] == expected[
            target_models[0].method
        ]


def test_result_only_evaluator_owns_its_dependency_query(tmp_path):
    store = ResultStore(tmp_path / "store")
    store.save_metrics(
        "search",
        pd.DataFrame({
            "target_id": ["A/concept-0", "A/concept-0"],
            "factor": [0.5, 1.0],
            "score": [0.2, 0.8],
        }),
    )
    evaluator = BestFactorEvaluator(
        node=EvaluatorNode(
            "best",
            "BestFactorEvaluator",
            depends_on=("search",),
            requires_inference=False,
            input={"from": "search"},
            params={"metric": "score"},
        ),
        context=evaluator_context(tmp_path, results=store.view(["search"])),
    )

    result = evaluator.evaluate([], [])

    assert result.inference is None
    assert result.metrics["factor"].tolist() == [1.0]


def test_best_factor_supports_method_level_selection_and_validity(tmp_path):
    rows = []
    scores = {
        "A": {
            0: (0.1, 0.3),
            1: (0.5, 0.7),
            2: (0.6, 0.8),
        },
        "B": {
            0: (0.4, 0.4),
            1: (0.45, 0.45),
            2: (0.44, 0.44),
        },
    }
    for method, factor_scores in scores.items():
        for factor, concept_scores in factor_scores.items():
            for concept_id, score in enumerate(concept_scores):
                rows.append({
                    "target_id": f"{method}/concept-{concept_id}",
                    "method": method,
                    "concept_id": concept_id,
                    "factor": float(factor),
                    "score": score,
                })
    evaluator = BestFactorEvaluator(
        node=EvaluatorNode(
            "best",
            "BestFactorEvaluator",
            requires_inference=False,
            params={
                "metric": "score",
                "group_by": ["method"],
                "aggregation": "mean",
                "strategy": "argmax",
                "baseline_factor": 0.0,
            },
        ),
        context=evaluator_context(tmp_path),
    )

    selected = pd.DataFrame(
        evaluator.compute_metrics(pd.DataFrame(rows))
    ).set_index("method")

    assert selected.loc["A", "factor"] == pytest.approx(2.0)
    assert selected.loc["A", "selected_score"] == pytest.approx(0.7)
    assert selected.loc["A", "baseline_score"] == pytest.approx(0.2)
    assert selected.loc["A", "selected_improvement"] == pytest.approx(0.5)
    assert selected.loc["B", "factor"] == pytest.approx(1.0)
    assert selected.loc["B", "selected_improvement"] == pytest.approx(0.05)
    assert "selection_valid" not in selected


def test_best_factor_uses_one_method_as_shared_baseline(tmp_path):
    examples = pd.DataFrame({
        "method": [
            "Random", "Random", "Random", "Random",
            "A", "A", "A", "A",
            "SFT", "SFT",
        ],
        "concept_id": [0, 1] * 5,
        "factor": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        "score": [0.2, 0.4, 0.5, 0.7, 0.3, 0.5, 0.8, 1.0, 0.9, 1.1],
    })
    evaluator = BestFactorEvaluator(
        node=EvaluatorNode(
            "best",
            "BestFactorEvaluator",
            requires_inference=False,
            params={
                "metric": "score",
                "group_by": ["method"],
                "baseline_factor": 0.0,
                "fallback_baseline_method": "Random",
            },
        ),
        context=evaluator_context(tmp_path),
    )

    selected = pd.DataFrame(
        evaluator.compute_metrics(examples)
    ).set_index("method")

    assert selected.loc["Random", "factor"] == pytest.approx(1.0)
    assert selected.loc["A", "baseline_score"] == pytest.approx(0.4)
    assert selected.loc["A", "selected_improvement"] == pytest.approx(0.5)
    assert selected.loc["SFT", "factor"] == pytest.approx(1.0)
    assert selected.loc["SFT", "baseline_score"] == pytest.approx(0.3)
    assert selected.loc["SFT", "selected_improvement"] == pytest.approx(0.7)
    assert selected.loc["SFT", "fallback_baseline_method"] == "Random"


def test_best_factor_supports_global_median_argmin(tmp_path):
    examples = pd.DataFrame({
        "factor": [0.0] * 3 + [1.0] * 3 + [2.0] * 3,
        "loss": [10.0, 10.0, 10.0, 1.0, 100.0, 100.0, 2.0, 2.0, 200.0],
    })
    evaluator = BestFactorEvaluator(
        node=EvaluatorNode(
            "best",
            "BestFactorEvaluator",
            requires_inference=False,
            params={
                "metric": "loss",
                "group_by": [],
                "aggregation": "median",
                "strategy": "argmin",
                "baseline_factor": 0.0,
            },
        ),
        context=evaluator_context(tmp_path),
    )

    selected = evaluator.compute_metrics(examples)

    assert selected[0]["factor"] == pytest.approx(2.0)
    assert selected[0]["selected_improvement"] == pytest.approx(8.0)


def test_best_factor_reads_and_filters_external_parquet(tmp_path):
    path = tmp_path / "phase1-id-metrics.parquet"
    pd.DataFrame({
        "method": ["APSR", "APSR", "Other", "Other"],
        "concept_id": [0, 0, 0, 0],
        "factor": [0.0, 0.4, 0.0, 0.8],
        "lm_judge_rating": [0.2, 0.9, 0.1, 1.0],
    }).to_parquet(path, index=False)
    evaluator = BestFactorEvaluator(
        node=EvaluatorNode(
            "best_external",
            "BestFactorEvaluator",
            depends_on=(),
            requires_inference=False,
            input={"path": str(path), "filters": {"method": "APSR"}},
            params={
                "metric": "lm_judge_rating",
                "group_by": ["method"],
                "baseline_factor": 0.0,
            },
        ),
        context=evaluator_context(tmp_path),
    )

    result = evaluator.evaluate([], [Concept(0, "formal language")])

    assert set(result.samples["method"]) == {"APSR"}
    assert result.metrics.loc[0, "factor"] == pytest.approx(0.4)
    assert result.metrics.loc[0, "selected_improvement"] == pytest.approx(0.7)


def test_best_factor_selects_per_concept_and_scores_held_out_split(tmp_path):
    rows = []
    validation_scores = {
        0: {0.0: 0.2, 1.0: 0.9, 2.0: 0.1},
        1: {0.0: 0.2, 1.0: 0.2, 2.0: 0.8},
    }
    test_scores = {
        0: {0.0: 0.2, 1.0: 0.7, 2.0: 1.0},
        1: {0.0: 0.2, 1.0: 1.0, 2.0: 0.6},
    }
    for concept_id in (0, 1):
        for factor in (0.0, 1.0, 2.0):
            for input_id in range(10):
                score = (
                    validation_scores[concept_id][factor]
                    if input_id < 5
                    else test_scores[concept_id][factor]
                )
                rows.append({
                    "method": "A",
                    "concept_id": concept_id,
                    "input_id": input_id,
                    "factor": factor,
                    "raw_aggregated_ratings": score,
                    "raw_relevance_concept_ratings": score + 0.1,
                })
    evaluator = BestFactorEvaluator(
        node=EvaluatorNode(
            "best_steerscope",
            "BestFactorEvaluator",
            requires_inference=False,
            report={"formats": ["png"], "dpi": 72},
            params={
                "metric": "raw_aggregated_ratings",
                "group_by": ["method", "concept_id"],
                "baseline_factor": 0.0,
                "split": {
                    "selection": "validation",
                    "evaluation": "test",
                    "ratio": 0.5,
                },
                "report_metrics": {
                    "lm_judge_rating": "raw_aggregated_ratings",
                    "relevance_concept_ratings": (
                        "raw_relevance_concept_ratings"
                    ),
                },
            },
        ),
        context=evaluator_context(tmp_path),
    )

    metrics = pd.DataFrame(evaluator.compute_metrics(pd.DataFrame(rows)))
    selected = metrics.set_index("concept_id")

    assert selected.loc[0, "factor"] == pytest.approx(1.0)
    assert selected.loc[1, "factor"] == pytest.approx(2.0)
    assert selected.loc[0, "selected_score"] == pytest.approx(0.9)
    assert selected.loc[1, "selected_score"] == pytest.approx(0.8)
    assert selected.loc[0, "evaluation_score"] == pytest.approx(0.7)
    assert selected.loc[1, "evaluation_score"] == pytest.approx(0.6)
    assert selected.loc[0, "evaluation_improvement"] == pytest.approx(0.5)
    assert selected.loc[1, "evaluation_improvement"] == pytest.approx(0.4)
    assert selected.loc[0, "relevance_concept_ratings"] == pytest.approx(0.8)
    assert selected.loc[1, "relevance_concept_ratings"] == pytest.approx(0.7)
    assert set(selected["selection_split"]) == {"validation"}
    assert set(selected["evaluation_split"]) == {"test"}

    paths = evaluator.render_report(
        EvaluationResult(metrics=metrics), tmp_path / "best_steerscope"
    )
    assert {path.name for path in paths} == {"summary.parquet", "metrics.png"}
    summary = pd.read_parquet(
        tmp_path / "best_steerscope" / "reports" / "summary.parquet"
    )
    assert summary.loc[0, "mean_evaluation_score"] == pytest.approx(0.65)
    assert summary.loc[0, "mean_lm_judge_rating"] == pytest.approx(0.65)


def test_best_factor_split_matches_fallback_baseline_by_concept(tmp_path):
    rows = []
    for method, factors in {"Random": (0.0, 1.0), "SFT": (1.0,)}.items():
        for concept_id, baseline in ((0, 0.1), (1, 0.4)):
            for factor in factors:
                for input_id in range(10):
                    score = baseline if method == "Random" else 0.8
                    rows.append({
                        "method": method,
                        "concept_id": concept_id,
                        "input_id": input_id,
                        "factor": factor,
                        "score": score,
                    })
    evaluator = BestFactorEvaluator(
        node=EvaluatorNode(
            "best_steerscope",
            "BestFactorEvaluator",
            requires_inference=False,
            params={
                "metric": "score",
                "group_by": ["method", "concept_id"],
                "baseline_factor": 0.0,
                "fallback_baseline_method": "Random",
                "split": {"ratio": 0.5},
            },
        ),
        context=evaluator_context(tmp_path),
    )

    selected = pd.DataFrame(
        evaluator.compute_metrics(pd.DataFrame(rows))
    ).set_index(["method", "concept_id"])

    assert selected.loc[("SFT", 0), "baseline_score"] == pytest.approx(0.1)
    assert selected.loc[("SFT", 1), "baseline_score"] == pytest.approx(0.4)
    assert selected.loc[("SFT", 0), "evaluation_baseline_score"] == pytest.approx(0.1)
    assert selected.loc[("SFT", 1), "evaluation_baseline_score"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    "evaluator_class",
    [
        LMJudgeEvaluator,
        IFEvalEvaluator,
        TruthfulQAEvaluator,
        MMLUEvaluator,
        PerplexityEvaluator,
        RuleEvaluator,
        WinRateEvaluator,
    ],
)
def test_current_inference_evaluators_use_owned_lifecycle(
    evaluator_class, tmp_path
):
    if evaluator_class is LMJudgeEvaluator:
        assert evaluator_class.evaluate is not Evaluator.evaluate
    else:
        assert evaluator_class.evaluate is Evaluator.evaluate
    evaluator = evaluator_class(
        node=EvaluatorNode("node", evaluator_class.__name__),
        context=evaluator_context(tmp_path),
    )
    assert evaluator.context.root_dump_dir == tmp_path
    assert evaluator_class.build_dataset is not Evaluator.build_dataset


def test_only_judge_evaluators_compose_judge_capability():
    assert issubclass(LMJudgeEvaluator, JudgeEvaluatorMixin)
    assert issubclass(WinRateEvaluator, JudgeEvaluatorMixin)
    for evaluator_class in (
        IFEvalEvaluator, TruthfulQAEvaluator, MMLUEvaluator, RuleEvaluator,
        PerplexityEvaluator,
    ):
        assert not issubclass(evaluator_class, JudgeEvaluatorMixin)
        assert not hasattr(evaluator_class, "_get_judge_ratings")


def test_alpaca_prompt_steering_does_not_open_a_prompt_client(tmp_path, monkeypatch):
    judge_client = MagicMock(close=AsyncMock())
    judge_model = MagicMock()
    judge_model.close = AsyncMock()
    judge_model.stats.get_report.return_value = {
        "total_calls": 0,
        "total_cache_hits": 0,
        "total_price": 0.0,
    }
    judge_client_factory = MagicMock(return_value=judge_client)
    judge_model_factory = MagicMock(return_value=judge_model)
    monkeypatch.setattr(
        "steerscope.evaluators.alpaca.AutoTokenizer.from_pretrained",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        "steerscope.evaluators.judge.AsyncOpenAI", judge_client_factory
    )
    monkeypatch.setattr(
        "steerscope.evaluators.judge.LanguageModel", judge_model_factory
    )
    evaluator = LMJudgeEvaluator(
        node=EvaluatorNode("judge", "LMJudgeEvaluator"),
        context=evaluator_context(
            tmp_path,
            steering_model_name="base/model",
            master_data_dir=str(tmp_path),
        ),
    )

    evaluator.open_resources([SimpleNamespace(method="PromptSteering")])

    assert evaluator._judge_client is judge_client
    assert judge_model_factory.call_args.args[1] is judge_client
    assert not hasattr(evaluator, "_prompt_client")
    assert not hasattr(evaluator, "_prompt_model")

    evaluator.close_resources()

    judge_model.close.assert_awaited_once()
    judge_client.close.assert_not_awaited()


def test_winrate_owns_dataset_and_calls_candidate_and_baseline(tmp_path):
    examples = pd.DataFrame({
        "factor": [1.0],
        "model_factor": [1.0],
        "input": ["a"],
    })
    baseline = MagicMock(
        target=target("PromptSteering"),
        method="PromptSteering",
        concept=target("PromptSteering").concept,
        factor=1.0,
    )
    candidate = MagicMock(
        target=target("DiffMean"),
        method="DiffMean",
        concept=target("DiffMean").concept,
        factor=1.0,
    )
    baseline.generate.side_effect = lambda examples: examples.assign(
        PromptSteering_steered_generation="baseline"
    )
    candidate.generate.side_effect = lambda examples: examples.assign(
        DiffMean_steered_generation="candidate"
    )
    evaluator = WinRateEvaluator(
        node=EvaluatorNode(
            "winrate",
            "WinRateEvaluator",
            params={"baseline": "PromptSteering"},
        ),
        context=evaluator_context(tmp_path),
    )
    evaluator.build_dataset = MagicMock(return_value=examples)
    evaluator.open_resources = MagicMock()
    evaluator.close_resources = MagicMock()
    evaluator.compute_metrics = MagicMock(return_value={
        "win_rate": 1.0,
        "loss_rate": 0.0,
        "tie_rate": 0.0,
        "baseline_model": "PromptSteering",
    })

    result = evaluator.evaluate([candidate, baseline], [candidate.concept])

    evaluator.build_dataset.assert_called_once()
    candidate.generate.assert_called_once()
    baseline.generate.assert_called_once()
    evaluator.close_resources.assert_called_once()
    assert result.metrics["win_rate"].tolist() == [1.0]


def test_winrate_reuses_one_fixed_baseline_for_candidate_sweep(tmp_path):
    examples = pd.DataFrame({
        "factor": [0.5],
        "model_factor": [0.5],
        "input": ["a"],
    })
    baseline = MagicMock(
        target=target("PromptSteering"),
        method="PromptSteering",
        concept=target("PromptSteering").concept,
        factor=1.0,
    )
    candidate = MagicMock(
        target=target("DiffMean"),
        method="DiffMean",
        concept=target("DiffMean").concept,
        factor=0.5,
    )
    candidate.generate.side_effect = lambda examples: examples.assign(
        DiffMean_steered_generation="candidate"
    )
    baseline.generate.side_effect = lambda examples: examples.assign(
        PromptSteering_steered_generation="baseline"
    )
    evaluator = WinRateEvaluator(
        node=EvaluatorNode(
            "winrate",
            "WinRateEvaluator",
            params={"baseline": "PromptSteering"},
        ),
        context=evaluator_context(tmp_path),
    )
    evaluator.build_dataset = MagicMock(return_value=examples)
    evaluator.open_resources = MagicMock()
    evaluator.close_resources = MagicMock()
    evaluator.compute_metrics = MagicMock(return_value={
        "win_rate": 1.0,
        "loss_rate": 0.0,
        "tie_rate": 0.0,
        "baseline_model": "PromptSteering",
    })

    evaluator.evaluate([candidate, baseline], [candidate.concept])

    baseline_input = baseline.generate.call_args.args[0]
    assert baseline_input["factor"].tolist() == [0.5]
    assert baseline_input["model_factor"].tolist() == [1.0]


def test_model_config_binds_one_factor():
    config = SteeringModelConfig(
        factor=1.5,
        temperature=0.2,
        do_sample=False,
        layers=(10, 20),
    )

    assert config.factor == 1.5
    assert config.temperature == 0.2
    assert config.do_sample is False
    assert config.layers == (10, 20)


def test_generation_kwargs_support_sampling_and_greedy_decoding():
    assert BaseModel.generation_kwargs(32, 0.7, True) == {
        "max_new_tokens": 32,
        "do_sample": True,
        "temperature": 0.7,
    }
    assert BaseModel.generation_kwargs(32, 0.0, False) == {
        "max_new_tokens": 32,
        "do_sample": False,
    }
    with pytest.raises(ValueError, match="positive temperature"):
        BaseModel.generation_kwargs(32, 0.0, True)


def test_model_wrapper_selects_by_model_factor_but_infers_with_dataset_factor(
    tmp_path, monkeypatch
):
    model = SteeringModel(
        target=target(),
        args=evaluator_args(overwrite_cache=False, evaluation_cache_context={}),
        training_args=SimpleNamespace(models={}),
        root_dump_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        config=SteeringModelConfig(factor=0.5),
    )
    seen = {}

    def generate(_, examples):
        seen["factor"] = examples["factor"].tolist()
        return examples.assign(DiffMean_steered_generation="x")

    monkeypatch.setattr(SteeringTargetRunner, "generate", generate)
    examples = pd.DataFrame({
        "factor": [-0.5],
        "model_factor": [0.5],
        "input": ["a"],
    })

    generated = model.generate(examples)

    assert seen["factor"] == [-0.5]
    assert generated["factor"].tolist() == [-0.5]
    assert generated["model_factor"].tolist() == [0.5]


def test_request_cache_is_reused_by_a_new_model_wrapper(tmp_path, monkeypatch):
    calls = []

    def generate(_, examples):
        calls.append(1)
        return examples.assign(
            method="DiffMean",
            target_id="DiffMean/concept-0",
            DiffMean_steered_generation="cached",
        )

    monkeypatch.setattr(SteeringTargetRunner, "generate", generate)
    kwargs = {
        "target": target(),
        "args": evaluator_args(overwrite_cache=False, evaluation_cache_context={}),
        "training_args": SimpleNamespace(models={}),
        "root_dump_dir": tmp_path,
        "cache_dir": tmp_path / "cache",
        "config": SteeringModelConfig(factor=0.5),
    }
    examples = pd.DataFrame({
        "factor": [0.5],
        "model_factor": [0.5],
        "input": ["same prompt"],
    })

    first = SteeringModel(**kwargs).generate(examples)
    second = SteeringModel(**kwargs).generate(examples)

    assert calls == [1]
    assert first.equals(second)


def test_models_cache_is_reused_without_predicting_again(tmp_path):
    benchmark = MagicMock(
        requires_training_args=False,
        uses_intervention_positions=False,
    )
    benchmark.predict_steer.return_value = {
        "steered_generation": ["cached"],
        "strength": [1.0],
    }
    benchmark.prepare_inference_examples.side_effect = lambda examples, **_: examples
    inference_args = evaluator_args(
        overwrite_cache=False,
        steering_model_name="base/model",
        model_name="base/model",
        steering_layers=[3],
        steering_layer=None,
        temperature=0.0,
        do_sample=False,
    )
    examples = pd.DataFrame({"factor": [1.0], "input": ["same prompt"]})

    def wrapper():
        return SteeringInferenceWrapper(
            "DiffMean",
            benchmark,
            SimpleNamespace(models={}),
            inference_args,
            prefix_length=1,
            cache_dir=tmp_path / "models",
            cache_context={"artifact": "same"},
        )

    assert wrapper().predict(examples, concept_id=0, metadata={})[
        "steered_generation"
    ] == ["cached"]
    assert wrapper().predict(examples, concept_id=0, metadata={})[
        "steered_generation"
    ] == ["cached"]
    benchmark.predict_steer.assert_called_once()
    assert benchmark.predict_steer.call_args.kwargs["do_sample"] is False
    assert benchmark.predict_steer.call_args.kwargs["temperature"] == 0.0


def test_perplexity_always_overwrites_method_supplied_values(tmp_path):
    benchmark = MagicMock(
        requires_training_args=False,
        uses_intervention_positions=False,
    )
    benchmark.predict_steer.return_value = {
        "steered_generation": ["generated"],
        "perplexity": [999.0],
    }
    benchmark.prepare_inference_examples.side_effect = lambda examples, **_: examples
    wrapper = SteeringInferenceWrapper(
        "LoRA",
        benchmark,
        SimpleNamespace(models={}),
        evaluator_args(
            overwrite_cache=True,
            compute_perplexity=True,
            steering_model_name="base/model",
            model_name="base/model",
            steering_layers=[3],
            steering_layer=None,
        ),
        prefix_length=1,
        cache_dir=tmp_path / "models",
    )
    wrapper._perplexities = MagicMock(return_value=[3.0])

    result = wrapper.predict(
        pd.DataFrame({"factor": [1.0], "input": ["a"]}),
        concept_id=0,
        metadata={},
    )

    assert result["perplexity"] == [3.0]
    wrapper._perplexities.assert_called_once_with(["generated"])


def test_non_perplexity_inference_drops_method_supplied_values(tmp_path):
    benchmark = MagicMock(
        requires_training_args=False,
        uses_intervention_positions=False,
    )
    benchmark.predict_steer.return_value = {
        "steered_generation": ["generated"],
        "perplexity": [999.0],
    }
    benchmark.prepare_inference_examples.side_effect = lambda examples, **_: examples
    wrapper = SteeringInferenceWrapper(
        "LoRA",
        benchmark,
        SimpleNamespace(models={}),
        evaluator_args(
            overwrite_cache=True,
            compute_perplexity=False,
            steering_model_name="base/model",
            model_name="base/model",
            steering_layers=[3],
            steering_layer=None,
        ),
        prefix_length=1,
        cache_dir=tmp_path / "models",
    )

    result = wrapper.predict(
        pd.DataFrame({"factor": [1.0], "input": ["a"]}),
        concept_id=0,
        metadata={},
    )

    assert "perplexity" not in result


def test_source_fingerprint_changes_when_evaluation_source_changes(
    tmp_path,
):
    package_dir = tmp_path / "steerscope"
    source = package_dir / "utils" / "constants.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    first = _source_fingerprint(package_dir)

    source.write_text("VALUE = 2\n", encoding="utf-8")

    assert _source_fingerprint(package_dir) != first


def test_source_fingerprint_ignores_training_dataset_implementation(tmp_path):
    package_dir = tmp_path / "steerscope"
    evaluation_source = package_dir / "evaluation" / "engine.py"
    evaluation_source.parent.mkdir(parents=True)
    evaluation_source.write_text("VERSION = 1\n", encoding="utf-8")
    training_dataset = package_dir / "utils" / "dataset.py"
    training_dataset.parent.mkdir(parents=True)
    training_dataset.write_text("VERSION = 1\n", encoding="utf-8")
    first = _source_fingerprint(package_dir)

    training_dataset.write_text("VERSION = 2\n", encoding="utf-8")

    assert _source_fingerprint(package_dir) == first


def test_model_returns_steered_candidate_logits_in_configured_order():
    class Inputs(dict):
        def to(self, device):
            return self

    class Tokenizer:
        padding_side = "right"

        def __call__(self, strings, **kwargs):
            return Inputs({
                "input_ids": torch.tensor([[1, 2], [3, 4]]),
                "attention_mask": torch.ones(2, 2, dtype=torch.long),
            })

    class ChoiceModel(Model):
        def __str__(self):
            return "LinearProbe"

    model = ChoiceModel(
        model=MagicMock(),
        tokenizer=Tokenizer(),
        layer=3,
        device="cpu",
    )
    model.ax = MagicMock()
    logits = torch.zeros(2, 2, 6)
    logits[0, -1] = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    logits[1, -1] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    model.ax_model = MagicMock(
        return_value=(None, SimpleNamespace(logits=logits))
    )
    examples = pd.DataFrame({
        "input": ["first", "second"],
        "factor": [0.0, 1.0],
        "concept_id": [0, 0],
        "choice_token_ids": [[1, 3, 4, 5], [1, 3, 4, 5]],
    })

    result = model.predict_choice_logits(
        examples,
        batch_size=2,
        prefix_length=1,
        disable_neuronpedia_max_act=False,
    )

    assert result["choice_logits"] == [
        [1.0, 3.0, 4.0, 5.0],
        [4.0, 2.0, 1.0, 0.0],
    ]
    assert result["strength"] == [0.0, 1.0]
    assert model.ax_model.call_args.kwargs["use_cache"] is False


def test_choice_forward_requests_only_final_logits_when_base_model_supports_it():
    class FinalLogitsModel:
        def forward(self, input_ids=None, num_logits_to_keep=0):
            raise AssertionError("Forward is mocked by the intervention wrapper.")

    model = Model(
        model=FinalLogitsModel(),
        tokenizer=MagicMock(),
        layer=3,
        device="cpu",
    )
    inputs = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }

    model_inputs = model.choice_model_inputs(inputs)

    assert model_inputs["num_logits_to_keep"] == 1


def test_prompt_model_scores_candidate_logits_with_evaluator_owned_input():
    class Inputs(dict):
        def to(self, device):
            return self

    tokenizer = MagicMock(padding_side="right")
    tokenizer.return_value = Inputs({
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    })
    base_model = MagicMock()
    base_model.return_value = SimpleNamespace(
        logits=torch.tensor([[[0.0, 0.0, 0.0], [1.0, 3.0, 2.0]]])
    )
    model = PromptSteering(
        model=base_model,
        tokenizer=tokenizer,
        layer=3,
        device="cpu",
    )
    examples = pd.DataFrame({
        "input": ["MMLU prompt"],
        "factor": [1.0],
        "concept_id": [0],
        "choice_token_ids": [[1, 2]],
    })

    result = model.predict_choice_logits(examples, batch_size=1)

    assert result == {"choice_logits": [[3.0, 2.0]], "strength": [1.0]}
    tokenizer.assert_called_once_with(
        ["MMLU prompt"], return_tensors="pt", padding=True, truncation=True
    )


def test_adapter_model_scores_candidate_logits_through_its_forward_wrapper():
    class Inputs(dict):
        def to(self, device):
            return self

    tokenizer = MagicMock(padding_side="right")
    tokenizer.return_value = Inputs({
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    })
    model = LoRA(
        model=MagicMock(),
        tokenizer=tokenizer,
        layer=3,
        training_args=MagicMock(),
        device="cpu",
    )
    model.ax_model = MagicMock(return_value=SimpleNamespace(
        logits=torch.tensor([[[0.0, 0.0, 0.0], [4.0, 2.0, 3.0]]])
    ))
    examples = pd.DataFrame({
        "input": ["MMLU prompt"],
        "factor": [1.0],
        "concept_id": [0],
        "choice_token_ids": [[0, 2]],
    })

    result = model.predict_choice_logits(examples, batch_size=1)

    assert result == {"choice_logits": [[4.0, 3.0]], "strength": [1.0]}
    model.ax_model.assert_called_once()


def test_wrapper_dispatches_choice_logit_inference_and_reuses_cache(tmp_path):
    benchmark = MagicMock(
        requires_training_args=False,
        uses_intervention_positions=False,
    )
    benchmark.predict_choice_logits.return_value = {
        "choice_logits": [[1.0, 2.0, 3.0, 4.0]],
        "strength": [1.0],
    }
    benchmark.prepare_inference_examples.side_effect = lambda examples, **_: examples
    inference_args = evaluator_args(
        overwrite_cache=False,
        steering_model_name="base/model",
        model_name="base/model",
        steering_layers=[3],
        steering_layer=None,
    )
    examples = pd.DataFrame({
        "factor": [1.0],
        "input": ["same prompt"],
        "choice_token_ids": [[10, 11, 12, 13]],
        "inference_mode": ["choice_logits"],
    })

    def wrapper():
        return SteeringInferenceWrapper(
            "LinearProbe",
            benchmark,
            SimpleNamespace(models={}),
            inference_args,
            prefix_length=1,
            cache_dir=tmp_path / "models",
        )

    first = wrapper().predict(examples, concept_id=0, metadata={})
    second = wrapper().predict(examples, concept_id=0, metadata={})

    assert first["choice_logits"] == [[1.0, 2.0, 3.0, 4.0]]
    assert [list(values) for values in second["choice_logits"]] == [
        [1.0, 2.0, 3.0, 4.0]
    ]
    assert second["strength"] == [1.0]
    benchmark.predict_choice_logits.assert_called_once()
    assert benchmark.predict_choice_logits.call_args.kwargs["show_progress"] is False
    benchmark.predict_steer.assert_not_called()


def test_mmlu_evaluator_scores_each_factor_and_preserves_sample_details(tmp_path):
    evaluator = MMLUEvaluator(
        node=EvaluatorNode("mmlu", "MMLUEvaluator"),
        context=evaluator_context(tmp_path),
    )
    inference = pd.DataFrame({
        "factor": [0.0, 0.0, 1.0, 1.0],
        "mmlu_answer_index": [0, 1, 0, 1],
        "LinearProbe_choice_logits": [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
        ],
    })
    wrapped_model = SimpleNamespace(
        method="LinearProbe",
        target=target(method="LinearProbe"),
        concept=Concept(0, "formal language"),
    )

    metrics, samples = evaluator.score(wrapped_model, inference)

    assert metrics["factor"].tolist() == [0.0, 1.0]
    assert metrics["mmlu_accuracy"].tolist() == [1.0, 0.5]
    assert metrics["mmlu_correct_count"].tolist() == [2, 1]
    assert metrics["mmlu_num_examples"].tolist() == [2, 2]
    assert samples["raw_mmlu_predicted_label"].tolist() == ["A", "B", "B", "B"]
    assert samples["raw_mmlu_is_correct"].tolist() == [True, True, False, True]


def test_truthfulqa_binary_scores_each_factor(tmp_path):
    evaluator = TruthfulQAEvaluator(
        node=EvaluatorNode("truthfulqa", "TruthfulQAEvaluator"),
        context=evaluator_context(tmp_path),
    )
    evaluator.model_name = "DiffMean"
    data = pd.DataFrame({
        "factor": [0.0, 0.0, 1.0, 1.0],
        "truthfulqa_answer_index": [0, 1, 0, 1],
        "DiffMean_choice_logits": [
            [4.0, 0.0], [0.0, 4.0], [0.0, 4.0], [0.0, 4.0]
        ],
    })

    result = evaluator.compute_metrics(data)

    assert result["factor"] == [0.0, 1.0]
    assert result["truthfulqa_binary_accuracy"] == [1.0, 0.5]
    assert result["truthfulqa_binary_correct_count"] == [2, 1]
    assert result["truthfulqa_binary_num_examples"] == [2, 2]
    assert result["raw_truthfulqa_predicted_label"] == ["A", "B", "B", "B"]
    assert result["raw_truthfulqa_is_correct"] == [True, True, False, True]


def test_truthfulqa_binary_order_is_stable_and_seeded(tmp_path):
    data_dir = tmp_path / "data" / "truthfulqa"
    data_dir.mkdir(parents=True)
    pd.DataFrame({
        "input_id": [0, 1, 2],
        "type": ["Adversarial"] * 3,
        "category": ["Misconceptions"] * 3,
        "question": ["Question zero?", "Question one?", "Question two?"],
        "best_answer": ["True zero", "True one", "True two"],
        "best_incorrect_answer": ["False zero", "False one", "False two"],
        "source": ["source"] * 3,
    }).to_parquet(data_dir / "TruthfulQA_binary.parquet", index=False)
    node = EvaluatorNode(
        "truthfulqa", "TruthfulQAEvaluator",
        dataset={"type": "TruthfulQA_binary", "num_examples": 3, "seed": 7},
    )
    context = evaluator_context(
        tmp_path,
        master_data_dir=str(tmp_path / "data"),
        model_name="base/model",
        steering_model_name="base/model",
    )
    model = SimpleNamespace(
        concept=SimpleNamespace(concept_id=3, text="formal language"),
        target=SimpleNamespace(base_model="base/model"),
    )

    def build():
        evaluator = TruthfulQAEvaluator(node=node, context=context)
        tokenizer = MagicMock(bos_token_id=1)
        tokenizer.encode.side_effect = [[10], [11]]
        tokenizer.apply_chat_template.return_value = [1, 20]
        tokenizer.decode.return_value = "formatted prompt"
        evaluator._tokenizer = tokenizer
        return evaluator.build_dataset(model, [0.0, 1.0])

    first = build()
    second = build()

    columns = [
        "input_id", "model_factor", "truthfulqa_choices",
        "truthfulqa_answer_index", "raw_input",
    ]
    assert first[columns].equals(second[columns])
    assert len(first) == 6
    for _, row in first.drop_duplicates("input_id").iterrows():
        answer_index = int(row["truthfulqa_answer_index"])
        assert row["truthfulqa_choices"][answer_index].startswith("True")


def test_truthfulqa_binary_rejects_non_binary_logits(tmp_path):
    evaluator = TruthfulQAEvaluator(
        node=EvaluatorNode("truthfulqa", "TruthfulQAEvaluator"),
        context=evaluator_context(tmp_path),
    )
    evaluator.model_name = "DiffMean"
    data = pd.DataFrame({
        "factor": [1.0],
        "truthfulqa_answer_index": [0],
        "DiffMean_choice_logits": [[1.0, 2.0, 3.0]],
    })

    with pytest.raises(ValueError, match="exactly two choice logits"):
        evaluator.compute_metrics(data)


def test_truthfulqa_binary_renders_report(tmp_path):
    evaluator = TruthfulQAEvaluator(
        node=EvaluatorNode(
            "truthfulqa", "TruthfulQAEvaluator",
            report={"formats": ["png"], "dpi": 72},
        ),
        context=evaluator_context(tmp_path),
    )
    metrics = pd.DataFrame({
        "method": ["DiffMean", "DiffMean"],
        "factor": [0.0, 1.0],
        "truthfulqa_binary_accuracy": [0.5, 0.6],
        "truthfulqa_binary_mean_gold_probability": [0.55, 0.65],
    })

    paths = evaluator.render_report(EvaluationResult(metrics=metrics), tmp_path)

    assert {path.name for path in paths} == {"summary.parquet", "metrics.png"}


def test_ifeval_scores_official_strict_and_loose_metrics(tmp_path):
    evaluator = IFEvalEvaluator(
        node=EvaluatorNode("ifeval", "IFEvalEvaluator"),
        context=evaluator_context(tmp_path),
    )
    evaluator.model_name = "DiffMean"
    data = pd.DataFrame({
        "factor": [0.0, 0.0, 1.0, 1.0],
        "original_prompt": ["prompt"] * 4,
        "ifeval_instruction_ids_json": [
            '["punctuation:no_comma"]',
            '["change_case:english_capital"]',
            '["punctuation:no_comma"]',
            '["change_case:english_capital"]',
        ],
        "ifeval_kwargs_json": ["[{}]"] * 4,
        "DiffMean_steered_generation": [
            "No commas here", "lowercase", "This, has a comma",
            "THIS RESPONSE IS WRITTEN ENTIRELY IN ENGLISH",
        ],
    })

    result = evaluator.compute_metrics(data)

    assert result["ifeval_prompt_strict_accuracy"] == [0.5, 0.5]
    assert result["ifeval_instruction_strict_accuracy"] == [0.5, 0.5]
    assert result["ifeval_prompt_loose_accuracy"] == [0.5, 0.5]
    assert result["ifeval_instruction_loose_accuracy"] == [0.5, 0.5]
    assert result["raw_ifeval_strict_follow_instruction_list"] == [
        [True], [False], [False], [True]
    ]


def test_ifeval_loose_scoring_accepts_removing_first_line(tmp_path):
    evaluator = IFEvalEvaluator(
        node=EvaluatorNode("ifeval", "IFEvalEvaluator"),
        context=evaluator_context(tmp_path),
    )
    evaluator.model_name = "DiffMean"
    data = pd.DataFrame({
        "factor": [1.0],
        "original_prompt": ["prompt"],
        "ifeval_instruction_ids_json": ['["change_case:english_lowercase"]'],
        "ifeval_kwargs_json": ["[{}]"],
        "DiffMean_steered_generation": ["Sure!\nthis is lowercase."],
    })

    result = evaluator.compute_metrics(data)

    assert result["ifeval_prompt_strict_accuracy"] == [0.0]
    assert result["ifeval_prompt_loose_accuracy"] == [1.0]


def test_ifeval_rejects_unknown_instruction_id(tmp_path):
    evaluator = IFEvalEvaluator(
        node=EvaluatorNode("ifeval", "IFEvalEvaluator"),
        context=evaluator_context(tmp_path),
    )
    evaluator.model_name = "DiffMean"
    data = pd.DataFrame({
        "factor": [1.0],
        "original_prompt": ["prompt"],
        "ifeval_instruction_ids_json": ['["unknown:rule"]'],
        "ifeval_kwargs_json": ["[{}]"],
        "DiffMean_steered_generation": ["response"],
    })

    with pytest.raises(ValueError, match="Unknown IFEval instruction id"):
        evaluator.compute_metrics(data)


def test_ifeval_builds_dataset_with_original_constraint_metadata(tmp_path):
    data_dir = tmp_path / "data" / "ifeval"
    data_dir.mkdir(parents=True)
    pd.DataFrame({
        "key": [1000, 1001],
        "prompt": ["First prompt", "Second prompt"],
        "instruction_id_list_json": [
            '["punctuation:no_comma"]', '["change_case:english_capital"]'
        ],
        "kwargs_json": ["[{}]", "[{}]"],
    }).to_parquet(data_dir / "IFEval.parquet", index=False)
    evaluator = IFEvalEvaluator(
        node=EvaluatorNode(
            "ifeval", "IFEvalEvaluator",
            dataset={"type": "IFEval", "num_examples": 2},
        ),
        context=evaluator_context(
            tmp_path,
            master_data_dir=str(tmp_path / "data"),
            model_name="base/model",
            steering_model_name="base/model",
        ),
    )
    tokenizer = MagicMock(bos_token_id=1)
    tokenizer.apply_chat_template.side_effect = [[1, 10], [1, 11]]
    tokenizer.decode.side_effect = ["formatted first", "formatted second"]
    evaluator._tokenizer = tokenizer
    model = SimpleNamespace(
        concept=SimpleNamespace(concept_id=3, text="formal language"),
        target=SimpleNamespace(base_model="base/model"),
    )

    examples = evaluator.build_dataset(model, [0.0, 1.0])

    assert len(examples) == 4
    assert set(examples["raw_input"]) == {"First prompt", "Second prompt"}
    assert set(examples["input_id"]) == {1000, 1001}
    assert set(examples["model_factor"]) == {0.0, 1.0}
    assert "ifeval_instruction_ids_json" in examples


def test_ifeval_renders_four_accuracy_curves(tmp_path):
    evaluator = IFEvalEvaluator(
        node=EvaluatorNode(
            "ifeval", "IFEvalEvaluator", report={"formats": ["png"], "dpi": 72}
        ),
        context=evaluator_context(tmp_path),
    )
    metrics = pd.DataFrame({
        "method": ["DiffMean", "DiffMean"],
        "factor": [0.0, 1.0],
        "ifeval_prompt_strict_accuracy": [0.2, 0.3],
        "ifeval_instruction_strict_accuracy": [0.3, 0.4],
        "ifeval_prompt_loose_accuracy": [0.4, 0.5],
        "ifeval_instruction_loose_accuracy": [0.5, 0.6],
    })

    paths = evaluator.render_report(EvaluationResult(metrics=metrics), tmp_path)

    assert {path.name for path in paths} == {"summary.parquet", "metrics.png"}


def test_mmlu_evaluator_renders_accuracy_report(tmp_path):
    evaluator = MMLUEvaluator(
        node=EvaluatorNode(
            "mmlu",
            "MMLUEvaluator",
            report={"formats": ["png"], "dpi": 72},
        ),
        context=evaluator_context(tmp_path),
    )
    metrics = pd.DataFrame({
        "method": ["LinearProbe"] * 4,
        "concept_id": [0, 1, 0, 1],
        "factor": [0.0, 0.0, 1.0, 1.0],
        "mmlu_accuracy": [0.7, 0.8, 0.6, 0.7],
        "mmlu_mean_gold_probability": [0.5, 0.6, 0.4, 0.5],
    })

    paths = evaluator.render_report(
        EvaluationResult(metrics=metrics), tmp_path / "mmlu"
    )

    assert {path.name for path in paths} == {"summary.parquet", "metrics.png"}
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    summary = pd.read_parquet(tmp_path / "mmlu" / "reports" / "summary.parquet")
    assert set(summary["metric"]) == {
        "mmlu_accuracy",
        "mmlu_mean_gold_probability",
    }
    assert summary["mean"].between(0.0, 1.0).all()


def test_mmlu_opens_only_its_tokenizer_resource(tmp_path, monkeypatch):
    tokenizer = MagicMock()
    tokenizer_loader = MagicMock(return_value=tokenizer)
    monkeypatch.setattr(
        "steerscope.evaluators.mmlu.AutoTokenizer.from_pretrained",
        tokenizer_loader,
    )
    evaluator = MMLUEvaluator(
        node=EvaluatorNode(
            "mmlu",
            "MMLUEvaluator",
            dataset={"type": "MMLU_test", "num_examples": 1},
        ),
        context=evaluator_context(
            tmp_path,
            steering_model_name="base/model",
            model_name="base/model",
            master_data_dir=str(tmp_path),
        ),
    )

    evaluator.open_resources([])

    tokenizer_loader.assert_called_once()
    assert evaluator._tokenizer is tokenizer
    assert not hasattr(evaluator, "_judge_client")
    assert not hasattr(evaluator, "_get_judge_ratings")


def test_mmlu_builds_its_dataset_without_alpaca_or_openai(tmp_path):
    mmlu_dir = tmp_path / "mmlu"
    mmlu_dir.mkdir()
    test_data = pd.DataFrame({
        "question": ["2 + 2?"],
        "subject": ["subject_0"],
        "choices": [["3", "4", "5", "6"]],
        "answer": [1],
    })
    dev_rows = []
    for subject_index in range(57):
        for example_index in range(5):
            dev_rows.append({
                "question": f"Question {example_index}",
                "subject": f"subject_{subject_index}",
                "choices": ["A0", "B0", "C0", "D0"],
                "answer": example_index % 4,
            })
    test_data.to_parquet(mmlu_dir / "MMLU_test.parquet", index=False)
    pd.DataFrame(dev_rows).to_parquet(
        mmlu_dir / "MMLU_dev.parquet", index=False
    )
    tokenizer = MagicMock(
        model_max_length=1024,
        bos_token_id=0,
    )
    tokenizer.encode.side_effect = lambda value, **_: ["ABCD".index(value.strip()) + 10]
    tokenizer.apply_chat_template.return_value = [0, 20, 21]
    tokenizer.decode.return_value = "formatted MMLU prompt"
    evaluator = MMLUEvaluator(
        node=EvaluatorNode(
            "mmlu",
            "MMLUEvaluator",
            dataset={"type": "MMLU_test", "num_examples": 1},
        ),
        context=evaluator_context(tmp_path, master_data_dir=str(tmp_path)),
    )
    evaluator._tokenizer = tokenizer
    wrapped_model = SimpleNamespace(
        method="LinearProbe",
        target=target("LinearProbe"),
        concept=target("LinearProbe").concept,
    )

    data = evaluator.build_dataset(wrapped_model, [0.0, 1.0])

    assert data["dataset_name"].unique().tolist() == ["MMLU_test"]
    assert data["model_factor"].tolist() == [0.0, 1.0]
    assert data["choice_token_ids"].tolist() == [[10, 11, 12, 13]] * 2
    assert data["inference_mode"].unique().tolist() == ["choice_logits"]
    assert data["raw_input"].tolist() == data["original_prompt"].tolist()
    assert "steered_input" not in data.columns
    assert "simple_steered_input" not in data.columns
    assert [call.args[0] for call in tokenizer.encode.call_args_list] == [
        " A",
        " B",
        " C",
        " D",
    ]
    assert not hasattr(evaluator, "_prompt_model")


def test_prompt_models_apply_the_same_model_owned_composition_to_mmlu_and_alpaca():
    tokenizer = MagicMock(bos_token_id=0)
    tokenizer.apply_chat_template.side_effect = lambda messages, **_: [0, messages[-1]["content"]]
    tokenizer.decode.side_effect = lambda tokens: tokens[-1]
    model = PromptSteering(
        model=MagicMock(),
        tokenizer=tokenizer,
        layer=3,
        device="cpu",
        lm_model_name="base/model",
    )
    model.prompt_by_concept = {0: "shared instruction"}
    examples = pd.DataFrame({
        "raw_input": ["Alpaca task", "MMLU task\nA. one\nB. two\nAnswer:"],
        "input": ["old alpaca", "old mmlu"],
        "input_concept": ["formal language", "formal language"],
        "concept_id": [0, 0],
    })

    prepared = model.prepare_inference_examples(examples)

    assert prepared["input"].tolist() == [
        "shared instruction\n\nQuestion: Alpaca task",
        "shared instruction\n\nQuestion: MMLU task\nA. one\nB. two\nAnswer:",
    ]


def test_mmlu_uses_bare_choice_tokens_after_gemma_chat_template(tmp_path):
    evaluator = MMLUEvaluator(
        node=EvaluatorNode("mmlu", "MMLUEvaluator"),
        context=evaluator_context(tmp_path),
    )
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda value, **_: [{
        "A": 1, "B": 2, "C": 3, "D": 4
    }[value]]
    evaluator._tokenizer = tokenizer

    assert evaluator._choice_token_ids("google/gemma-2-2b-it") == [1, 2, 3, 4]
    assert [call.args[0] for call in tokenizer.encode.call_args_list] == list("ABCD")


def test_artifact_signature_ignores_mtime_changes(tmp_path):
    artifact = tmp_path / "weight.pt"
    artifact.write_bytes(b"same weights")
    first = _file_signature(artifact)
    os.utime(artifact, (artifact.stat().st_atime + 5, artifact.stat().st_mtime + 5))

    second = _file_signature(artifact)

    assert first == second


def test_reft_adapter_directories_participate_in_artifact_context():
    assert _method_artifact_directory("ConceptLoReFT") == "concept_loreft"
    assert _method_artifact_directory("PreferenceLoReFT") == "preference_loreft"
    assert _method_artifact_directory("HyperSteer") == "hyperreft"


def test_completion_hash_uses_result_content_not_mtime(tmp_path):
    store = ResultStore(tmp_path / "runs", run_id="test")
    node_id = "source"
    execution_hash = "execution"
    config = {"id": node_id}
    store.mark_running(node_id, execution_hash, config)
    path = store.save_metrics(node_id, pd.DataFrame({"score": [1.0]}))
    store.mark_complete(
        node_id,
        execution_hash,
        config,
        metadata={"result_kinds": ["metrics"]},
    )
    first = store.completion_hash(node_id, execution_hash)

    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    same = store.completion_hash(node_id, execution_hash)
    path.write_bytes(path.read_bytes() + b"changed")
    changed = store.completion_hash(node_id, execution_hash)

    assert same == first
    assert changed != first


def test_engine_context_changes_only_when_artifact_content_changes(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    weight = train_dir / "DiffMean_weight.pt"
    weight.write_bytes(b"version one")
    args = evaluator_args(
        steering_model_name="base/model",
        model_name="base/model",
        use_bf16=False,
        lm_model="gpt-4o-mini",
        multishot_factors_parquet=None,
        suppress_eval_dir=None,
    )
    training_args = SimpleNamespace(models={})
    wrapped_target = EvaluationTarget(
        target_id="DiffMean/concept-0",
        method="DiffMean",
        concept=Concept(0, "formal language"),
        base_model="base/model",
        artifact=Artifact("checkpoint", train_dir),
    )
    node = EvaluatorNode(
        "judge",
        "LMJudgeEvaluator",
        models=("DiffMean",),
        dataset={"type": "AlpacaEval", "num_examples": 1},
    )

    first = _engine_context(
        args, [wrapped_target], tmp_path, training_args, nodes=[node]
    )
    os.utime(weight, (weight.stat().st_atime + 5, weight.stat().st_mtime + 5))
    same = _engine_context(
        args, [wrapped_target], tmp_path, training_args, nodes=[node]
    )
    weight.write_bytes(b"version two")
    changed = _engine_context(
        args, [wrapped_target], tmp_path, training_args, nodes=[node]
    )

    assert first == same
    assert first != changed


def test_engine_context_prefers_small_training_artifact_manifest(
    tmp_path, monkeypatch
):
    # A normal per-concept method routes its explicit artifact through
    # root/train, whose merged artifact directory can contain large weights.
    external_train = tmp_path / "train"
    external_train.mkdir(parents=True)
    large_weight = external_train / "sft" / "0" / "large.safetensors"
    large_weight.parent.mkdir(parents=True)
    large_weight.write_bytes(b"weight-v1")
    artifact_manifest = external_train / "artifact_manifest.json"
    artifact_manifest.write_text('{"fingerprint":"one"}', encoding="utf-8")
    args = evaluator_args(
        steering_model_name="base/model",
        model_name="base/model",
        use_bf16=False,
        lm_model="gpt-4o-mini",
        multishot_factors_parquet=None,
        suppress_eval_dir=None,
    )
    wrapped_target = EvaluationTarget(
        target_id="SFT/concept-0",
        method="SFT",
        concept=Concept(0, "formal language"),
        base_model="base/model",
        artifact=Artifact("checkpoint", external_train),
    )
    node = EvaluatorNode(
        "judge",
        "LMJudgeEvaluator",
        models=("SFT",),
        dataset={"type": "AlpacaEval", "num_examples": 1},
    )
    training_args = SimpleNamespace(models={})

    def reject_recursive_signature(path):
        raise AssertionError(f"recursively hashed manifested artifact: {path}")

    monkeypatch.setattr(
        "steerscope.scripts.evaluate._path_signature",
        reject_recursive_signature,
    )

    first = _engine_context(
        args, [wrapped_target], tmp_path, training_args, nodes=[node]
    )
    large_weight.write_bytes(b"weight-v2")
    same = _engine_context(
        args, [wrapped_target], tmp_path, training_args, nodes=[node]
    )
    artifact_manifest.write_text('{"fingerprint":"two"}', encoding="utf-8")
    changed = _engine_context(
        args, [wrapped_target], tmp_path, training_args, nodes=[node]
    )

    assert same == first
    assert changed != first


def test_engine_adopts_only_exact_allowlisted_completion(tmp_path, monkeypatch):
    store = ResultStore(tmp_path / "runs", run_id="apsr")
    node = EvaluatorNode(
        "mmlu", "MMLUEvaluator", models=("APSR",),
        dataset={"type": "MMLU_test", "num_examples": 20},
    )
    old_hash = "a" * 64
    new_context = {"artifact": "manifest-v2"}
    new_hash = node.config_hash(context=new_context)
    store.save_metrics("mmlu", pd.DataFrame({"score": [1.0]}))
    store.save_samples("mmlu", pd.DataFrame({"score": [1.0]}))
    store.mark_complete(
        "mmlu", old_hash, node.as_dict(),
        metadata={"result_kinds": ("metrics", "samples")},
    )
    allowlist = {
        "version": 1,
        "entries": {"apsr": {"mmlu": {
            "from_execution_hash": old_hash,
            "to_execution_hash": new_hash,
            "config_sha256": store._config_sha256(node.as_dict()),
            "result_signatures": store.result_signatures("mmlu"),
        }}},
    }
    path = tmp_path / "migration.json"
    path.write_text(json.dumps(allowlist), encoding="utf-8")
    monkeypatch.setenv(store.IDENTITY_MIGRATION_ENV, str(path))
    executor = MagicMock(side_effect=AssertionError("must reuse completion"))

    EvaluationEngine(
        [node], store, executor, context=new_context, generate_reports=False,
    ).run()

    executor.assert_not_called()
    manifest = store.manifest("mmlu")
    assert manifest["execution_hash"] == new_hash
    assert manifest["metadata"]["cache_identity_migration"]["reason"] == (
        "exact_allowlist"
    )


def test_allowlisted_completion_rejects_changed_result(tmp_path, monkeypatch):
    store = ResultStore(tmp_path / "runs", run_id="apsr")
    node = EvaluatorNode("mmlu", "MMLUEvaluator", models=("APSR",))
    store.save_metrics("mmlu", pd.DataFrame({"score": [1.0]}))
    store.mark_complete(
        "mmlu", "a" * 64, node.as_dict(),
        metadata={"result_kinds": ("metrics",)},
    )
    entry = {
        "from_execution_hash": "a" * 64,
        "to_execution_hash": "b" * 64,
        "config_sha256": store._config_sha256(node.as_dict()),
        "result_signatures": store.result_signatures("mmlu"),
    }
    path = tmp_path / "migration.json"
    path.write_text(json.dumps({
        "version": 1, "entries": {"apsr": {"mmlu": entry}}
    }), encoding="utf-8")
    store.save_metrics("mmlu", pd.DataFrame({"score": [0.0]}))
    monkeypatch.setenv(store.IDENTITY_MIGRATION_ENV, str(path))

    assert not store.adopt_allowlisted_completion(
        "mmlu", "b" * 64, node.as_dict()
    )
    assert store.manifest("mmlu")["execution_hash"] == "a" * 64


def test_completed_evaluator_requires_matching_execution_hash(tmp_path):
    store = ResultStore(tmp_path / "runs")
    node = EvaluatorNode(
        "mmlu",
        "MMLUEvaluator",
        models=("APSR",),
        concepts={"ids": [7]},
        inference={"strengths": [0.8]},
    )
    old_context = {
        "source_fingerprint": "old-source",
        "pipeline_version": "3",
        "base_model": "synthetic/base",
    }
    identity = store.semantic_cache_identity(node.as_dict(), old_context)
    store.save_metrics("mmlu", pd.DataFrame({"score": [1.0]}))
    store.mark_complete(
        "mmlu",
        "old-execution",
        node.as_dict(),
        metadata={"result_kinds": ("metrics",)},
        cache_identity=identity,
    )

    changed_source = dict(
        old_context,
        source_fingerprint="new-source",
        pipeline_version="4",
    )
    assert not store.is_complete(
        "mmlu",
        "new-execution",
        config=node.as_dict(),
        context=changed_source,
    )
    assert store.is_complete(
        "mmlu",
        "old-execution",
        config=node.as_dict(),
        context=changed_source,
    )
    assert not store.is_complete(
        "mmlu",
        "new-execution",
        config=node.as_dict(),
        context=dict(changed_source, base_model="synthetic/other"),
    )


def test_new_execution_does_not_delete_last_complete_results(tmp_path):
    store = ResultStore(tmp_path / "runs")
    config = {"id": "mmlu"}
    expected = pd.DataFrame({"score": [0.75]})
    store.save_metrics("mmlu", expected)
    store.mark_complete(
        "mmlu",
        "old-execution",
        config,
        metadata={"result_kinds": ("metrics",)},
    )

    store.mark_running("mmlu", "new-execution", config)

    pd.testing.assert_frame_equal(store.load("mmlu", "metrics"), expected)


def test_evaluator_data_signatures_are_isolated_between_nodes(tmp_path):
    (tmp_path / "alpaca_eval.json").write_text(
        '[{"instruction": "Explain gravity."}]', encoding="utf-8"
    )
    mmlu_dir = tmp_path / "mmlu"
    mmlu_dir.mkdir()
    mmlu_test = mmlu_dir / "MMLU_test.parquet"
    mmlu_dev = mmlu_dir / "MMLU_dev.parquet"
    mmlu_test.write_bytes(b"mmlu test v1")
    mmlu_dev.write_bytes(b"mmlu dev")
    args = evaluator_args(
        master_data_dir=str(tmp_path),
        use_bf16=False,
        intervene_on_prompt=True,
        disable_neuronpedia_max_act=False,
    )
    training_args = SimpleNamespace(models={})
    wrapped_target = EvaluationTarget(
        target_id="DiffMean/concept-0",
        method="DiffMean",
        concept=Concept(0, "formal language"),
        base_model="base/model",
        artifact=Artifact("checkpoint", tmp_path / "train"),
    )
    nodes = [
        EvaluatorNode(
            "judge",
            "LMJudgeEvaluator",
            models=("DiffMean",),
            dataset={"type": "AlpacaEval", "num_examples": 1},
        ),
        EvaluatorNode(
            "mmlu",
            "MMLUEvaluator",
            models=("DiffMean",),
            dataset={"type": "MMLU_test", "num_examples": 1},
        ),
    ]
    first = _engine_context(
        args, [wrapped_target], tmp_path, training_args, nodes=nodes
    )

    mmlu_test.write_bytes(b"mmlu test v2")
    changed = _engine_context(
        args, [wrapped_target], tmp_path, training_args, nodes=nodes
    )

    assert context_for_node(first, "judge") == context_for_node(changed, "judge")
    assert context_for_node(first, "mmlu") != context_for_node(changed, "mmlu")


def test_lm_judge_uses_configured_api_concurrency(tmp_path):
    evaluator = LMJudgeEvaluator(
        node=EvaluatorNode(
            "judge", "LMJudgeEvaluator", params={"judge_concurrency": 7}
        ),
        context=evaluator_context(tmp_path),
        judge_concurrency=7,
    )
    evaluator.model_name = "DiffMean"
    evaluator._judge_model = MagicMock()
    evaluator._judge_model.chat_completions = AsyncMock(
        return_value=["Rating: [[2]]"]
    )

    ratings, _ = evaluator._get_ratings_from_prompts(["prompt"], "concept")

    assert ratings == [2.0]
    assert evaluator._judge_model.chat_completions.await_args.kwargs[
        "batch_size"
    ] == 7


def test_lm_judge_retries_only_invalid_responses_without_cache(tmp_path):
    evaluator = LMJudgeEvaluator(
        node=EvaluatorNode(
            "judge",
            "LMJudgeEvaluator",
            params={"judge_parse_retries": 2},
        ),
        context=evaluator_context(tmp_path),
        judge_parse_retries=2,
    )
    evaluator.model_name = "DiffMean"
    evaluator._judge_model = MagicMock()
    evaluator._judge_model.chat_completions = AsyncMock(side_effect=[
        ["Rating: [[2]]", "missing rating"],
        ["Rating: [[1]]"],
    ])

    ratings, completions = evaluator._get_ratings_from_prompts(
        ["valid prompt", "invalid prompt"], "concept"
    )

    assert ratings == [2.0, 1.0]
    assert completions == ["Rating: [[2]]", "Rating: [[1]]"]
    assert evaluator._judge_model.chat_completions.await_count == 2
    retry = evaluator._judge_model.chat_completions.await_args_list[1]
    assert retry.args[1] == ["invalid prompt"]
    assert retry.kwargs["refresh_cache"] is True


def test_lm_judge_uses_default_after_parse_retries_are_exhausted(tmp_path):
    evaluator = LMJudgeEvaluator(
        node=EvaluatorNode(
            "judge",
            "LMJudgeEvaluator",
            params={"judge_parse_retries": 2},
        ),
        context=evaluator_context(tmp_path),
        judge_parse_retries=2,
    )
    evaluator.model_name = "DiffMean"
    evaluator._judge_model = MagicMock()
    evaluator._judge_model.chat_completions = AsyncMock(side_effect=[
        ["bad initial response"],
        ["bad retry one"],
        ["bad retry two"],
    ])

    ratings, completions = evaluator._get_ratings_from_prompts(
        ["prompt"], "concept"
    )

    assert ratings == [LMJudgeEvaluator.DEFAULT_RATING]
    assert completions == ["bad retry two"]
    assert evaluator._judge_fallback_count == 1
    failure = json.loads((Path(evaluator.output_dir)/"judge_parse_failures.jsonl").read_text().splitlines()[0])
    assert failure["fallback_rating"] == 0
    assert failure["completion"] == "bad retry two"
    assert evaluator._judge_model.chat_completions.await_count == 3


def test_lm_judge_combines_rating_dimensions_into_one_queue(tmp_path):
    evaluator = LMJudgeEvaluator(
        node=EvaluatorNode("judge", "LMJudgeEvaluator"),
        context=evaluator_context(tmp_path),
        judge_concurrency=5,
    )
    evaluator.model_name = "DiffMean"
    evaluator._judge_model = MagicMock()
    evaluator._judge_model.chat_completions = AsyncMock(
        return_value=["Rating: [[2]]", "Rating: [[1]]", "Rating: [[2]]"]
    )
    data = pd.DataFrame({
        "dataset_name": ["AlpacaEval"],
        "input_concept": ["formal language"],
        "original_prompt": ["answer this"],
        "DiffMean_steered_generation": ["a generated answer"],
    })

    evaluator._get_all_ratings_from_data(data, "DiffMean")

    evaluator._judge_model.chat_completions.assert_awaited_once()
    call = evaluator._judge_model.chat_completions.await_args
    assert len(call.args[0]) == 3
    assert len(call.args[1]) == 3
    assert call.kwargs["batch_size"] == 5
    assert call.kwargs["progress_callback"].__self__ is evaluator


def test_judge_usage_metadata_is_scoped_to_current_invocation(tmp_path):
    evaluator = LMJudgeEvaluator(
        node=EvaluatorNode("judge", "LMJudgeEvaluator"),
        context=evaluator_context(tmp_path),
    )
    stats = MagicMock()
    stats.get_report.side_effect = [
        {
            "total_calls": 5,
            "network_calls": 3,
            "total_cache_hits": 2,
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "total_price": 1.0,
        },
        {
            "total_calls": 8,
            "network_calls": 4,
            "total_cache_hits": 4,
            "input_tokens": 160,
            "output_tokens": 35,
            "total_tokens": 195,
            "total_price": 1.75,
        },
    ]
    evaluator._judge_model = MagicMock(stats=stats)
    evaluator._evaluation_judge_report = evaluator._judge_report()

    metadata = evaluator.evaluation_metadata()

    assert metadata["language_model"] == {
        "total_calls": 3,
        "network_calls": 1,
        "total_cache_hits": 2,
        "input_tokens": 60,
        "output_tokens": 15,
        "total_tokens": 75,
        "total_price": pytest.approx(0.75),
    }


def test_judge_progress_displays_live_usage(tmp_path):
    evaluator = LMJudgeEvaluator(
        node=EvaluatorNode(
            "judge",
            "LMJudgeEvaluator",
            params={"judge_progress_interval": 0},
        ),
        context=evaluator_context(tmp_path),
    )
    evaluator._model_progress_bar = MagicMock()

    evaluator._update_judge_progress({
        "total_calls": 140,
        "network_calls": 128,
        "total_cache_hits": 12,
        "input_tokens": 1_250_000,
        "output_tokens": 86_400,
        "total_tokens": 1_336_400,
        "total_price": 0.40956,
    })

    evaluator._model_progress_bar.set_postfix_str.assert_called_once_with(
        "$0.4096 | judge=128 | tok=1.2M+86.4K | hit=12",
        refresh=True,
    )


def test_language_model_cache_is_incremental_and_persistent(tmp_path):
    kwargs = {
        "model": "gpt-4o-mini",
        "client": MagicMock(),
        "dump_dir": tmp_path / "output",
        "use_cache": True,
        "cache_level": "prompt",
        "cache_tag": "judge",
        "master_data_dir": tmp_path,
        "lm_cache_dir": tmp_path / "sqlite-cache",
    }
    model = LanguageModel(**kwargs)
    model.cache.upsert_many({"prompt": "completion"})
    model.save_cache()

    restored = LanguageModel(**kwargs)
    assert restored.cache.get_many(["prompt"]) == {"prompt": "completion"}
    assert restored.cache_db_file.is_file()
    model.cache.close()
    restored.cache.close()


def test_language_model_cache_supports_provider_prefixed_model_id(tmp_path):
    model = LanguageModel(
        model="DeepSeek-V3.2-Instruct",
        client=MagicMock(),
        dump_dir=tmp_path / "output",
        use_cache=True,
        cache_level="prompt",
        master_data_dir=tmp_path,
        lm_cache_dir=tmp_path / "sqlite-cache",
    )
    model.cache.upsert_many({"prompt": "completion"})
    model.save_cache()

    expected = (
        tmp_path / "sqlite-cache/DeepSeek-V3.2-Instruct_cache.sqlite3"
    )
    assert model.cache_db_file == expected
    assert expected.is_file()
    model.cache.close()


def test_language_model_refreshes_and_overwrites_prompt_cache(tmp_path):
    response = MagicMock()
    response.to_dict.return_value = {
        "choices": [{"message": {"content": "Rating: [[2]]"}}],
        "usage": {"completion_tokens": 1, "prompt_tokens": 2},
    }
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    model = LanguageModel(
        model="gpt-4o-mini",
        client=client,
        dump_dir=tmp_path / "output",
        use_cache=True,
        cache_level="prompt",
        cache_tag="judge",
        master_data_dir=tmp_path,
        lm_cache_dir=tmp_path / "sqlite-cache",
    )
    prompt = "rate this"
    cache_key = model._get_cache_key(prompt, 0, "judge")
    model.cache.upsert_many({cache_key: "bad cached response"})

    cached = asyncio.run(model.chat_completions("judge", [prompt]))
    refreshed = asyncio.run(model.chat_completions(
        "judge", [prompt], refresh_cache=True
    ))

    assert cached == ["bad cached response"]
    assert refreshed == ["Rating: [[2]]"]
    client.chat.completions.create.assert_awaited_once()
    assert model.cache.get_many([cache_key]) == {
        cache_key: "Rating: [[2]]"
    }
    model.cache.close()


def test_parse_standalone_steering_vector_target(tmp_path):
    vector = tmp_path / "vector.pt"
    vector.touch()
    parsed = parse_evaluation_targets({
        "custom-sv": {
            "method": "SteeringVector",
            "concept": {"id": 3, "text": "formal language"},
            "artifact": {"kind": "steering_vector", "path": vector},
        }
    }, default_base_model="base/model")[0]
    assert parsed.target_id == "custom-sv"
    assert parsed.concept.concept_id == 3
    assert parsed.artifact.path == vector.resolve()


def test_explicit_checkpoint_target_requires_directory(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    with pytest.raises(ValueError, match="checkpoint artifact must be a directory"):
        parse_evaluation_targets({
            "target": {
                "method": "DiffMean",
                "concept": {"text": "formal language"},
                "artifact": {"kind": "checkpoint", "path": checkpoint},
            }
        }, default_base_model="base/model")


def test_unknown_evaluator_has_clear_error():
    with pytest.raises(ValueError, match="Unknown evaluator type"):
        parse_evaluator_nodes(
            {"bad": {"type": "MissingEvaluator"}},
            evaluator_resolver=lambda name: (_ for _ in ()).throw(AttributeError(name)),
        )


def test_evaluator_cannot_override_its_inference_capability():
    with pytest.raises(ValueError, match="cannot override requires_inference"):
        parse_evaluator_nodes(
            {
                "judge": {
                    "type": "LMJudgeEvaluator",
                    "requires_inference": False,
                }
            },
            evaluator_resolver=lambda _: LMJudgeEvaluator,
        )


def test_standalone_vector_maps_arbitrary_concept_id_to_local_row(tmp_path):
    vector = torch.arange(4, dtype=torch.float32)
    vector_path = tmp_path / "vector.pt"
    torch.save(vector, vector_path)
    runtime = _SteeringTargetRuntime.__new__(_SteeringTargetRuntime)
    benchmark_model = MagicMock()
    benchmark_model.model.config.hidden_size = 4
    projection = MagicMock()
    projection.weight.data = torch.zeros(1, 4)
    projection.bias.data = torch.ones(1)
    benchmark_model.ax.proj = projection

    def make_model(**kwargs):
        benchmark_model.ax.proj.weight.data = torch.zeros(kwargs["low_rank_dimension"], 4)
        benchmark_model.ax.proj.bias.data = torch.ones(kwargs["low_rank_dimension"])

    benchmark_model.make_model.side_effect = make_model
    runtime._load_steering_vector(
        benchmark_model,
        vector_path,
        "SteeringVector",
        concept_id=17,
        intervention_type="addition",
    )
    assert torch.equal(benchmark_model.ax.proj.weight.data[0], vector)
    assert benchmark_model.concept_id_map == {17: 0}


def test_runtime_cache_key_uses_declared_inference_scope():
    runtime = _SteeringTargetRuntime.__new__(_SteeringTargetRuntime)
    runtime.args = evaluator_args(steering_intervention_type="addition")

    shared = target(method="DiffMean", concept_id=3)
    joint = target(method="HyperSteer", concept_id=3)
    per_concept = target(method="SFT", concept_id=3)
    psr = target(method="APSR", concept_id=3)
    shared_key = runtime._benchmark_cache_key(shared, 5, [5])
    joint_key = runtime._benchmark_cache_key(joint, 5, [5])
    per_concept_key = runtime._benchmark_cache_key(per_concept, 5, [5])
    psr_key = runtime._benchmark_cache_key(psr, 5, [5])

    assert shared_key[1] is None
    assert joint_key[1] is None
    assert per_concept_key[1] == 3
    assert psr_key[1] == 3


def test_runtime_keeps_checkpoint_global_concept_index(tmp_path):
    runtime = _SteeringTargetRuntime.__new__(_SteeringTargetRuntime)
    runtime.args = evaluator_args(
        steering_layer=3,
        steering_layers=[3],
        steering_model_name="base/model",
        model_name="base/model",
        overwrite_cache=False,
        evaluation_cache_context={},
        compute_perplexity=False,
    )
    runtime.training_args = SimpleNamespace(overwrite_metadata_dir=None, models={})
    runtime.root_dump_dir = tmp_path
    runtime.cache_dir = tmp_path / "cache"
    runtime.model_cache_dir = tmp_path / "model-cache"
    runtime.device = torch.device("cpu")
    runtime.targets = {}
    benchmark_model = MagicMock(
        requires_mean_activations=False,
        requires_training_args=False,
        uses_intervention_positions=False,
        concept_id_map=None,
    )
    benchmark_model.predict_steer.return_value = {
        "steered_generation": ["x"],
        "strength": [1.0],
    }
    runtime._metadata = MagicMock(return_value=[{
        "concept": "formal language",
        "ref": None,
        "concept_genres_map": {"formal language": ["text"]},
        "concept_id": 17,
    }])
    runtime._load_tokenizer = MagicMock(return_value=MagicMock())
    runtime._load_base_model = MagicMock(return_value=MagicMock())
    runtime._ensure_padding = MagicMock()
    runtime._load_concept_model = MagicMock(return_value=benchmark_model)
    runtime._target_cache_context = MagicMock(return_value={})
    wrapped_target = target(concept_id=17)

    runtime.generate_target(
        wrapped_target,
        pd.DataFrame({"factor": [1.0], "input": ["a"]}),
    )

    assert benchmark_model.concept_id_map is None


def test_dataset_agnostic_helper_splits_validation_and_test():
    data = pd.DataFrame({
        "concept_id": [0] * 8,
        "input_id": [0, 0, 1, 1, 2, 2, 3, 3],
    })
    validation = split_by_input_id(data, "validation", None)
    test = split_by_input_id(data, "test", None)
    assert validation["input_id"].unique().tolist() == [0, 1]
    assert test["input_id"].unique().tolist() == [2, 3]


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda: require_dataset_type({"num_examples": 2}, "AlpacaEval"), "define 'type'"),
        (lambda: require_num_examples({"type": "AlpacaEval"}), "define num_examples"),
    ],
)
def test_dataset_agnostic_helpers_validate_structure(operation, message):
    with pytest.raises(ValueError, match=message):
        operation()


def test_factor_expansion_preserves_model_and_dataset_factors():
    data = expand_factors(
        pd.DataFrame({"input_id": [0]}),
        [0.5, 1.0],
        dataset_factor=lambda factor: -factor,
    )

    assert data["factor"].tolist() == [-0.5, -1.0]
    assert data["model_factor"].tolist() == [0.5, 1.0]


def test_evaluation_framework_contains_no_dataset_dispatcher():
    framework_files = [
        Path(steerscope.__file__).parent / "evaluation" / "dataset.py",
        Path(steerscope.__file__).parent / "evaluation" / "engine.py",
        Path(steerscope.__file__).parent / "evaluators" / "evaluator.py",
    ]
    framework_source = "\n".join(
        path.read_text(encoding="utf-8") for path in framework_files
    )

    assert "AlpacaEval" not in framework_source
    assert "MMLU_" not in framework_source
    assert "EvaluationDatasetBuilder" not in framework_source


def _scoped_concepts(count=10):
    return [
        Concept(
            concept_id,
            f"concept-{concept_id}",
            metadata={
                "concept_genres_map": {
                    f"concept-{concept_id}": [
                        "text" if concept_id % 2 == 0 else "code"
                    ]
                }
            },
        )
        for concept_id in range(count)
    ]


def test_concept_scope_count_is_stable_and_order_independent(tmp_path):
    node = EvaluatorNode(
        "limited",
        "RecordingEvaluator",
        concepts={"count": 4, "seed": 17},
    )
    first = RecordingEvaluator(node, evaluator_context(tmp_path))
    second = RecordingEvaluator(node, evaluator_context(tmp_path))
    concepts = _scoped_concepts()

    first.prepare_evaluation([], concepts)
    second.prepare_evaluation([], reversed(concepts))

    first_ids = first.evaluation_metadata()["concept_scope"]["concept_ids"]
    second_ids = second.evaluation_metadata()["concept_scope"]["concept_ids"]
    assert first_ids == second_ids
    assert len(first_ids) == 4


def test_concept_scope_combines_genres_and_explicit_ids(tmp_path):
    node = EvaluatorNode(
        "selected",
        "RecordingEvaluator",
        concepts={"genres": ["text"], "ids": [6, 2]},
    )
    evaluator = RecordingEvaluator(node, evaluator_context(tmp_path))

    evaluator.prepare_evaluation([], _scoped_concepts())

    assert evaluator.evaluation_metadata()["concept_scope"] == {
        "genres": ["text"],
        "concept_ids": [2, 6],
    }


@pytest.mark.parametrize(
    ("scope", "message"),
    [
        ({"count": 0}, "at least 1"),
        ({"count": 11}, "only 10 are available"),
        ({"seed": 42}, "requires concepts.count"),
        ({"ids": [99]}, "unavailable concept IDs"),
    ],
)
def test_concept_scope_rejects_invalid_sampling_config(tmp_path, scope, message):
    evaluator = RecordingEvaluator(
        EvaluatorNode("invalid", "RecordingEvaluator", concepts=scope),
        evaluator_context(tmp_path),
    )

    with pytest.raises((TypeError, ValueError), match=message):
        evaluator.prepare_evaluation([], _scoped_concepts())


def test_generic_evaluator_contains_no_judge_service_knowledge():
    source = (Path(steerscope.__file__).parent / "evaluators" / "evaluator.py").read_text(
        encoding="utf-8"
    )

    assert "AsyncOpenAI" not in source
    assert "LanguageModel" not in source
    assert "judge_" not in source
