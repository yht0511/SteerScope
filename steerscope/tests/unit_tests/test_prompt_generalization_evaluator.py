import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import threading

import pandas as pd
import pytest

import steerscope
from steerscope.evaluation import (
    Artifact,
    Concept,
    EvaluationContext,
    EvaluationResult,
    EvaluationTarget,
    EvaluatorNode,
    ResultStore,
)
from steerscope.evaluators.prompt_generalization import (
    PromptGeneralizationEvaluator,
)


def _context(tmp_path, results=None):
    return EvaluationContext(
        args=SimpleNamespace(
            steering_model_name="google/gemma-2-2b-it",
            model_name="google/gemma-2-2b-it",
            master_data_dir=str(tmp_path),
            winrate_split_ratio=0.5,
            steer_data_type="concept",
            seed=42,
            lm_model="judge",
            runtime_backend="legacy",
            overwrite_cache=False,
            report_formats=["png"],
            report_dpi=72,
        ),
        root_dump_dir=tmp_path,
        output_dir=tmp_path / "output",
        results=results,
    )


def _node(num_examples=2):
    return EvaluatorNode(
        "prompt_generalization",
        "PromptGeneralizationEvaluator",
        dataset={
            "type": "XAlpacaEval",
            "path": "x_alpaca_eval/XAlpacaEval.parquet",
            "num_examples": num_examples,
            "seed": 42,
            "reference_column": "instruction_en",
            "augmenters": [
                {
                    "id": "chinese",
                    "name": "LanguageAugmenter",
                    "kwargs": {"column": "instruction_cn"},
                },
                {"id": "base64", "name": "Base64Augmenter"},
            ],
        },
        params={"judge_concurrency": 2, "min_id_effect": 0.1},
        report={"enabled": True},
    )


def _evaluator(tmp_path, num_examples=2):
    return PromptGeneralizationEvaluator(
        node=_node(num_examples),
        context=_context(tmp_path),
    )


def _write_dataset(tmp_path):
    directory = tmp_path / "x_alpaca_eval"
    directory.mkdir(parents=True)
    data = pd.DataFrame({
        "id": [1, 2],
        "dataset": ["first", "second"],
        "instruction_en": ["English one", "English two"],
        "instruction_cn": ["Chinese one", "Chinese two"],
    })
    data.to_parquet(directory / "XAlpacaEval.parquet", index=False)
    return data


def _target(method="DiffMean", concept_id=3):
    return EvaluationTarget(
        target_id=f"{method}/concept-{concept_id}",
        method=method,
        concept=Concept(concept_id, f"concept {concept_id}"),
        base_model="google/gemma-2-2b-it",
        artifact=Artifact("checkpoint"),
    )


def _model(method="DiffMean", concept_id=3):
    target = _target(method, concept_id)
    return SimpleNamespace(
        method=target.method,
        target=target,
        concept=target.concept,
    )


def test_prompt_bank_builds_aligned_language_and_base64_variants(tmp_path):
    _write_dataset(tmp_path)
    evaluator = _evaluator(tmp_path)

    bank = evaluator._build_prompt_bank(3)

    assert len(bank) == 6
    assert bank.groupby("source_input_id").size().to_dict() == {1: 3, 2: 3}
    assert bank.groupby("augmenter").size().to_dict() == {
        "base64": 2,
        "chinese": 2,
        "identity": 2,
    }
    assert bank["system_prompt"].isna().all()
    sources = bank.set_index(["source_input_id", "augmenter"])
    for source_id in (1, 2):
        identity = sources.loc[(source_id, "identity"), "raw_input"]
        chinese = sources.loc[(source_id, "chinese"), "raw_input"]
        encoded_prompt = sources.loc[(source_id, "base64"), "raw_input"]
        payload = encoded_prompt.split("Encoded instruction:\n", 1)[1]
        assert base64.b64decode(payload).decode("utf-8") == identity
        assert chinese == f"Chinese {'one' if source_id == 1 else 'two'}"


def test_generalization_source_and_cache_are_keyed_by_concept(tmp_path):
    directory = tmp_path / "x_alpaca_eval"
    directory.mkdir(parents=True)
    pd.DataFrame({
        "id": list(range(40)),
        "dataset": ["fixture"] * 40,
        "instruction_en": [f"English {index}" for index in range(40)],
        "instruction_cn": [f"Chinese {index}" for index in range(40)],
    }).to_parquet(directory / "XAlpacaEval.parquet", index=False)
    evaluator = _evaluator(tmp_path, num_examples=10)

    first = evaluator._load_fixed_source(3)["id"].tolist()
    repeated = evaluator._load_fixed_source(3)["id"].tolist()
    other = evaluator._load_fixed_source(9)["id"].tolist()

    assert first == repeated
    assert first != other
    assert evaluator._prompt_bank_cache_path(3) != (
        evaluator._prompt_bank_cache_path(9)
    )


def test_methods_and_factors_share_only_their_concept_prompt_bank(tmp_path):
    _write_dataset(tmp_path)
    evaluator = _evaluator(tmp_path)
    first_bank = evaluator._build_prompt_bank(3)
    second_bank = first_bank.copy()
    second_bank["raw_input"] = "concept-9:" + second_bank["raw_input"]
    evaluator._prompt_banks = {3: first_bank, 9: second_bank}
    evaluator._baseline_inference = {
        3: pd.DataFrame(),
        9: pd.DataFrame(),
    }
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.side_effect = (
        lambda messages, **_: [messages[-1]["content"]]
    )
    tokenizer.decode.side_effect = lambda tokens: f"chat:{tokens[0]}"
    tokenizer.bos_token_id = None
    evaluator._dataset_tokenizer = tokenizer

    first = evaluator.build_dataset(_model("DiffMean", 3), [0.0, 1.0])
    second = evaluator.build_dataset(_model("LoRA", 9), [1.0])

    expected_prompt_ids = first_bank["prompt_id"].tolist()
    assert first.groupby("factor")["prompt_id"].apply(list).tolist() == [
        expected_prompt_ids,
        expected_prompt_ids,
    ]
    assert second["prompt_id"].tolist() == expected_prompt_ids
    assert first["raw_input"].str.startswith("concept-9:").sum() == 0
    assert second["raw_input"].str.startswith("concept-9:").all()
    assert set(first["concept_id"]) == {3}
    assert set(second["concept_id"]) == {9}


def test_generalization_pipeline_overlaps_targets_with_embedded_baseline(
    tmp_path,
):
    evaluator = _evaluator(tmp_path, num_examples=1)
    evaluator.open_resources = MagicMock()
    evaluator.close_resources = MagicMock()
    evaluator._open_model_progress = MagicMock()
    evaluator._close_model_progress = MagicMock()
    second_target_generated = threading.Event()
    first_score_started = threading.Event()
    models = []
    for concept_id in (3, 9):
        current_target = _target(concept_id=concept_id)
        wrapped = MagicMock(
            method="DiffMean",
            target=current_target,
            concept=current_target.concept,
            factor=1.0,
        )

        def generate(examples, concept=concept_id):
            if concept == 9:
                second_target_generated.set()
            return examples.assign(
                method="DiffMean",
                target_id=f"DiffMean/concept-{concept}",
                DiffMean_steered_generation=f"steered {concept}",
            )

        wrapped.generate.side_effect = generate
        models.append(wrapped)

    def build_dataset(model, factors):
        concept_id = int(model.concept.concept_id)
        evaluator._baseline_inference[concept_id] = pd.DataFrame({
            "prompt_id": [0],
            "source_prompt": ["source"],
            "baseline_generation": [f"baseline {concept_id}"],
        })
        return pd.DataFrame({
            "factor": factors,
            "model_factor": factors,
            "prompt_id": [0],
            "source_prompt": ["source"],
            "concept_id": [concept_id],
            "input": ["formatted"],
        })

    evaluator.build_dataset = build_dataset

    def score(model, inference):
        if model.concept.concept_id == 3:
            first_score_started.set()
            assert second_target_generated.wait(timeout=2.0)
        assert inference["baseline_generation"].tolist() == [
            f"baseline {model.concept.concept_id}"
        ]
        return (
            pd.DataFrame({"concept_id": [model.concept.concept_id]}),
            inference.copy(),
        )

    evaluator.score = score

    result = evaluator.evaluate(
        models, [model.concept for model in models]
    )

    assert first_score_started.is_set()
    assert second_target_generated.is_set()
    assert result.metrics["concept_id"].tolist() == [3, 9]
    assert result.inference["baseline_generation"].tolist() == [
        "baseline 3",
        "baseline 9",
    ]
    assert evaluator._baseline_inference == {}


def test_generalization_filters_methods_before_target_scheduling(tmp_path):
    store = ResultStore(tmp_path / "results")
    store.save_metrics("best_factor", pd.DataFrame({
        "method": ["Strong", "Weak"],
        "factor": [1.0, 2.0],
        "selected_score": [0.8, 0.7],
        "selected_improvement": [0.2, 0.05],
    }))
    base_node = _node()
    evaluator = PromptGeneralizationEvaluator(
        node=EvaluatorNode(
            base_node.node_id,
            base_node.evaluator_type,
            depends_on=("best_factor",),
            dataset=base_node.dataset,
            inference={"select": {
                "from": "best_factor",
                "metric": "selected_score",
                "include_factors": [0.0],
            }},
            params={
                **dict(base_node.params),
                "min_selection_improvement": 0.1,
            },
        ),
        context=_context(
            tmp_path,
            results=store.view(["best_factor"]),
        ),
    )
    models = []
    for method, concept_id in (
        ("Strong", 0),
        ("Strong", 1),
        ("Weak", 0),
    ):
        current_target = _target(method, concept_id)
        models.extend([
            SimpleNamespace(
                method=method,
                target=current_target,
                concept=current_target.concept,
                factor=factor,
            )
            for factor in (0.0, 1.0, 2.0)
        ])

    scheduled = evaluator.evaluation_models(models)

    assert {model.method for model in scheduled} == {"Strong"}
    assert {model.concept.concept_id for model in scheduled} == {0, 1}


def test_gemma_base64_wrapper_is_formatted_as_one_user_message(tmp_path):
    evaluator = _evaluator(tmp_path)
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = [17]
    tokenizer.decode.return_value = "formatted"
    tokenizer.bos_token_id = None
    evaluator._dataset_tokenizer = tokenizer

    result = evaluator._format_variant(
        "google/gemma-2-2b-it",
        "wrapper and payload",
        None,
    )

    assert result == "formatted"
    messages = tokenizer.apply_chat_template.call_args.args[0]
    assert messages == [{"role": "user", "content": "wrapper and payload"}]
    with pytest.raises(ValueError, match="does not support system prompts"):
        evaluator._format_variant(
            "google/gemma-2-2b-it",
            "payload",
            "system wrapper",
        )


def test_metrics_compute_paired_id_and_ood_effects(tmp_path):
    evaluator = _evaluator(tmp_path)
    evaluator.model_name = "DiffMean"
    evaluator.target = _target()
    evaluator._baseline_inference = {
        3: pd.DataFrame({
            "prompt_id": list(range(6)),
            "source_prompt": [f"source {i}" for i in range(6)],
            "baseline_generation": [f"b{i}" for i in range(6)],
        })
    }
    scopes = ["identity", "chinese", "base64"] * 2
    source_ids = [1, 1, 1, 2, 2, 2]
    rows = []
    for factor in (0.0, 1.0):
        for prompt_id, (source_id, scope) in enumerate(zip(source_ids, scopes)):
            rows.append({
                "factor": factor,
                "prompt_id": prompt_id,
                "source_input_id": source_id,
                "source_prompt": f"source {source_id}",
                "is_reference": scope == "identity",
                "augmenter": scope,
                "DiffMean_steered_generation": f"s-{factor}-{prompt_id}",
            })
    data = pd.DataFrame(rows)

    def ratings(frame, generation_column, api_name):
        if generation_column == "baseline_generation":
            scores = [0.0] * 6
        else:
            factor_zero = [0.0] * 6
            factor_one = [1.0, 0.5, 0.0, 2.0, 1.5, 1.0]
            scores = factor_zero + factor_one
        return {
            "scores": {
                component: list(scores)
                for component in ("concept", "instruction", "fluency", "overall")
            },
            "completions": {
                component: ["judge"] * len(scores)
                for component in ("concept", "instruction", "fluency")
            },
        }

    evaluator._joint_ratings = MagicMock(side_effect=ratings)

    result = evaluator.compute_metrics(data)
    metrics = pd.DataFrame({
        key: value
        for key, value in result.items()
        if not key.startswith("raw_")
        and "completions" not in key
        and isinstance(value, list)
        and len(value) == 6
    })
    factor_one = metrics[metrics["factor"] == 1.0].set_index("scope")

    assert factor_one.loc["chinese", "id_effect"] == pytest.approx(1.5)
    assert factor_one.loc["chinese", "ood_effect"] == pytest.approx(1.0)
    assert factor_one.loc["base64", "ood_effect"] == pytest.approx(0.5)
    assert factor_one.loc["overall", "ood_effect"] == pytest.approx(0.75)
    assert "retention" not in metrics
    assert "retention_valid" not in metrics
    assert len(result["raw_net_steering_effect"]) == 12
    assert len(result["raw_steered_overall_score"]) == 12
    assert len(result["raw_baseline_instruction_score"]) == 12


def test_joint_rating_uses_source_prompt_and_harmonic_mean(tmp_path):
    evaluator = _evaluator(tmp_path)
    evaluator.model_name = "DiffMean"
    evaluator.target = _target()
    evaluator._get_rating_groups = MagicMock(return_value=[
        ([2.0], ["concept completion"]),
        ([1.0], ["instruction completion"]),
        ([2.0], ["fluency completion"]),
    ])
    data = pd.DataFrame({
        "source_prompt": ["Write a short poem about rain."],
        "original_prompt": ["Encoded instruction: V3JpdGU="],
        "generation": ["Rain falls softly."],
    })

    ratings = evaluator._joint_ratings(data, "generation", "test")

    prompt_groups = evaluator._get_rating_groups.call_args.args[0]
    instruction_prompt = prompt_groups[1][1][0]
    assert "Write a short poem about rain." in instruction_prompt
    assert "Encoded instruction: V3JpdGU=" not in instruction_prompt
    assert ratings["scores"]["overall"] == [pytest.approx(1.5)]
    assert evaluator._harmonic_mean([2.0, 0.0, 2.0]) == 0.0


def test_report_contains_every_augmenter_and_overall(tmp_path, monkeypatch):
    evaluator = _evaluator(tmp_path)
    metrics = pd.DataFrame([
        {
            "method": "DiffMean",
            "concept_id": 0,
            "factor": factor,
            "scope": scope,
            "id_effect": factor,
            "ood_effect": factor * 0.5,
            "num_prompts": 2,
        }
        for scope in ("overall", "chinese", "base64")
        for factor in (0.0, 1.0)
    ])
    rendered = []
    monkeypatch.setattr(
        evaluator,
        "_render_scope_figure",
        lambda summary, scoped, scope, labels, output_dir: (
            rendered.append(scope) or [Path(f"{scope}.png")]
        ),
    )

    paths = evaluator.render_report(
        EvaluationResult(metrics=metrics),
        tmp_path,
    )

    assert rendered == ["overall", "chinese", "base64"]
    assert len(paths) == 4
    assert paths[0].name == "summary.parquet"


def test_report_renders_factor_and_id_effect_curves(tmp_path):
    evaluator = _evaluator(tmp_path)
    metrics = pd.DataFrame([
        {
            "method": method,
            "concept_id": 0,
            "factor": factor,
            "scope": scope,
            "id_effect": factor * method_scale,
            "ood_effect": factor * method_scale * scope_scale,
            "num_prompts": 2,
        }
        for method, method_scale in (("DiffMean", 1.0), ("LoRA", 0.8))
        for scope, scope_scale in (
            ("overall", 0.5),
            ("chinese", 0.6),
            ("base64", 0.4),
        )
        for factor in (0.0, 1.0)
    ])

    paths = evaluator.render_report(
        EvaluationResult(metrics=metrics),
        tmp_path,
    )

    assert [path.name for path in paths] == [
        "summary.parquet",
        "overall.png",
        "chinese.png",
        "base64.png",
    ]
    assert all(path.is_file() for path in paths)


def test_method_metrics_use_all_concepts_and_ratio_of_means(tmp_path):
    evaluator = _evaluator(tmp_path)
    metrics = pd.DataFrame([
        {
            "method": method,
            "concept_id": concept_id,
            "factor": factor,
            "scope": "overall",
            "id_effect": id_effect,
            "ood_effect": ood_effect,
            "num_prompts": 3,
        }
        for method, factor, effects in (
            ("Strong", 1.0, ((0.2, 0.1), (0.8, 0.3))),
            ("Boundary", 1.0, ((0.1, 0.05), (0.1, 0.05))),
            ("Weak", 1.0, ((0.02, 0.1), (0.08, 0.2))),
        )
        for concept_id, (id_effect, ood_effect) in enumerate(effects)
    ])

    aggregated = evaluator._aggregate_method_metrics(metrics).set_index(
        "method"
    )

    assert aggregated.loc["Strong", "num_concepts"] == 2
    assert aggregated.loc["Strong", "id_effect"] == pytest.approx(0.5)
    assert aggregated.loc["Strong", "ood_effect"] == pytest.approx(0.2)
    assert aggregated.loc["Strong", "retention"] == pytest.approx(0.4)
    assert bool(aggregated.loc["Strong", "retention_valid"])
    assert aggregated.loc["Boundary", "retention"] == pytest.approx(0.5)
    assert bool(aggregated.loc["Boundary", "retention_valid"])
    assert pd.isna(aggregated.loc["Weak", "retention"])
    assert not bool(aggregated.loc["Weak", "retention_valid"])


def test_method_metrics_require_same_complete_concept_set(tmp_path):
    evaluator = _evaluator(tmp_path)
    metrics = pd.DataFrame([
        {
            "method": method,
            "concept_id": concept_id,
            "factor": 1.0,
            "scope": "overall",
            "id_effect": 0.5,
            "ood_effect": 0.25,
        }
        for method, concept_ids in (("Complete", (0, 1)), ("Missing", (0,)))
        for concept_id in concept_ids
    ])

    with pytest.raises(ValueError, match="complete concept set"):
        evaluator._aggregate_method_metrics(metrics)


def test_unknown_augmenter_is_rejected(tmp_path):
    node = _node()
    dataset = dict(node.dataset)
    dataset["augmenters"] = [{"id": "bad", "name": "MissingAugmenter"}]
    invalid = EvaluatorNode(
        node.node_id,
        node.evaluator_type,
        dataset=dataset,
        params=node.params,
    )

    with pytest.raises(ValueError, match="Unknown generalization augmenter"):
        PromptGeneralizationEvaluator(invalid, _context(tmp_path))


def test_prompt_generalization_evaluator_is_registered():
    assert (
        steerscope.PromptGeneralizationEvaluator
        is PromptGeneralizationEvaluator
    )
