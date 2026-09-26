from types import SimpleNamespace

import pandas as pd
import pytest

from steerscope.evaluators.alpaca import AlpacaEvaluator
from steerscope.scripts.training_data_study import (
    _prepare_shared_generate_dir,
    _reuse_reference_training_artifacts,
)
from steerscope.scripts.train import (
    prepare_all_concepts_training_data,
    prepare_concept_training_data,
    prepare_df,
    prepare_training_data,
)
from steerscope.studies.training_data import (
    StudyVariant,
    aggregate_study_metrics,
    apply_reference_factors,
    build_study_variants,
    derive_variant_config,
    extract_reference_factors,
)
from steerscope.utils.training_subset import select_balanced_subset


def _paired_frames(count=12):
    positive = pd.DataFrame({
        "pair_id": range(count),
        "input": [f"prompt-{index}" for index in range(count)],
        "output": [f"positive-{index}" for index in range(count)],
    })
    negative = pd.DataFrame({
        "pair_id": range(count),
        "input": [f"prompt-{index}" for index in range(count)],
        "output": [f"negative-{index}" for index in range(count)],
    })
    return positive, negative


def _train_rows(include_negative=True, negative_prompt="prompt-0"):
    rows = [{
        "pair_id": 0,
        "input": "prompt-0",
        "output": "positive-0",
        "output_concept": "target concept",
        "category": "positive",
    }]
    if include_negative:
        rows.append({
            "pair_id": 0,
            "input": negative_prompt,
            "output": "negative-0",
            "output_concept": "EEEEE",
            "category": "negative",
        })
    return pd.DataFrame(rows)


class MinimalTokenizer:
    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        return [1, len(messages[0]["content"]), 2]

    def decode(self, tokens):
        return "|".join(str(token) for token in tokens)


def test_prepare_df_rejects_missing_paired_negative_without_fallback():
    with pytest.raises(ValueError, match="requires paired negative"):
        prepare_df(
            _train_rows(include_negative=False),
            "target concept",
            tokenizer=None,
            binarize=False,
            train_on_negative=True,
            is_chat_model=False,
            output_length=32,
            model_name="test-model",
        )


def test_prepare_df_rejects_mismatched_pair_prompt():
    with pytest.raises(ValueError, match="identical pair_id and prompt"):
        prepare_df(
            _train_rows(negative_prompt="different-prompt"),
            "target concept",
            tokenizer=None,
            binarize=False,
            train_on_negative=True,
            is_chat_model=False,
            output_length=32,
            model_name="test-model",
        )


def test_all_concepts_preparation_reuses_single_concept_rules():
    frames = []
    selected = [(0, "concept zero"), (1, "concept one")]
    for concept_id, concept in selected:
        frame = _train_rows().copy()
        frame["concept_id"] = concept_id
        frame.loc[frame["category"] == "positive", "output_concept"] = concept
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    common = {
        "binarize": False,
        "train_on_negative": False,
        "is_chat_model": False,
        "output_length": 32,
        "model_name": "test-model",
        "max_num_of_examples": None,
        "subset_seed": 7,
    }

    expected = pd.concat([
        prepare_concept_training_data(
            raw[raw["concept_id"] == concept_id].copy(),
            concept,
            MinimalTokenizer(),
            concept_id=concept_id,
            show_sample=False,
            **common,
        )
        for concept_id, concept in selected
    ], ignore_index=True)
    actual = prepare_all_concepts_training_data(
        raw,
        selected,
        MinimalTokenizer(),
        **common,
    )
    dispatched = prepare_training_data(
        raw,
        MinimalTokenizer(),
        training_granularity="all_concepts",
        selected_concepts=selected,
        **common,
    )

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(dispatched, expected)
    assert set(actual["category"]) == {"positive"}
    assert set(actual["concept_id"]) == {0, 1}


def test_paired_subsets_are_aligned_nested_and_seeded():
    positive, negative = _paired_frames()
    small_pos, small_neg, small_meta = select_balanced_subset(
        positive, negative, max_num_of_examples=6, subset_seed=11, group_key=3
    )
    large_pos, large_neg, _ = select_balanced_subset(
        positive, negative, max_num_of_examples=12, subset_seed=11, group_key=3
    )
    other_pos, _, _ = select_balanced_subset(
        positive, negative, max_num_of_examples=6, subset_seed=12, group_key=3
    )

    small_ids = set(small_pos["pair_id"])
    assert small_ids == set(small_neg["pair_id"])
    assert small_ids < set(large_pos["pair_id"])
    assert small_ids != set(other_pos["pair_id"])
    assert small_meta["selected_pairs"] == 3


def test_study_variants_deduplicate_shared_run():
    config = {
        "generate": {"num_of_examples": 24},
        "study": {
            "sample_efficiency": {"sizes": [6, 12], "subset_seed": 42},
            "sample_sensitivity": {
                "size": 12,
                "subset_seeds": [42, 43, 44],
            },
        },
    }
    variants = build_study_variants(config)
    assert [(item.train_examples, item.subset_seed) for item in variants] == [
        (6, 42),
        (12, 42),
        (12, 43),
        (12, 44),
    ]
    shared = variants[1]
    assert shared.efficiency is True
    assert shared.sensitivity is True


def test_factor_reference_is_merged_and_scheduled_first():
    config = {
        "generate": {"num_of_examples": 24},
        "study": {
            "factor_selection": {
                "train_examples": 24,
                "subset_seed": 42,
            },
            "sample_efficiency": {"sizes": [6, 24], "subset_seed": 42},
            "sample_sensitivity": {
                "size": 12,
                "subset_seeds": [42, 43],
            },
        },
    }
    variants = build_study_variants(config)
    assert variants[0] == StudyVariant(
        24,
        42,
        efficiency=True,
        factor_reference=True,
    )
    assert sum(variant.factor_reference for variant in variants) == 1


def test_reference_factors_restrict_non_reference_scan_to_one_factor():
    config = {
        "study": {
            "result": {
                "evaluator": "best_factor",
                "metric": "selected_improvement",
            }
        },
        "evaluate": {
            "models": ["DiffMean", "PCA", "LoRA"],
            "evaluators": {
                "id_lm_judge": {
                    "type": "LMJudgeEvaluator",
                    "inference": {"strengths": [0.0, 1.0, 2.0]},
                },
                "best_factor": {
                    "type": "BestFactorEvaluator",
                    "depends_on": ["id_lm_judge"],
                    "input": {"from": "id_lm_judge"},
                    "params": {
                        "group_by": ["method"],
                        "baseline_factor": 0.0,
                        "fallback_baseline_method": "DiffMean",
                    },
                },
            },
        },
    }
    metrics = pd.DataFrame({
        "source_evaluator": ["best_factor"] * 3,
        "method": ["DiffMean", "PCA", "LoRA"],
        "factor": [2.0, 1.0, 1.0],
    })
    factors = extract_reference_factors(metrics, config)
    fixed = apply_reference_factors(config, factors)
    strengths = fixed["evaluate"]["evaluators"]["id_lm_judge"][
        "inference"
    ]["strengths_by_model"]
    assert strengths == {
        "DiffMean": [0.0, 2.0],
        "PCA": [1.0],
        "LoRA": [1.0],
    }
    assert "strengths" not in fixed["evaluate"]["evaluators"][
        "id_lm_judge"
    ]["inference"]
    # The original full-scan config remains unchanged.
    assert "strengths_by_model" not in config["evaluate"]["evaluators"][
        "id_lm_judge"
    ]["inference"]


def test_reference_factor_validation_rejects_non_method_or_infinite_results():
    config = {
        "study": {"result": {"evaluator": "best_factor"}},
        "evaluate": {
            "models": ["DiffMean"],
            "evaluators": {
                "scan": {"type": "LMJudgeEvaluator", "inference": {}},
                "best_factor": {
                    "type": "BestFactorEvaluator",
                    "depends_on": ["scan"],
                    "params": {"group_by": ["method", "concept_id"]},
                },
            },
        },
    }
    metrics = pd.DataFrame({
        "source_evaluator": ["best_factor"],
        "method": ["DiffMean"],
        "factor": [float("inf")],
    })
    with pytest.raises(ValueError, match="group_by"):
        extract_reference_factors(metrics, config)
    config["evaluate"]["evaluators"]["best_factor"]["params"][
        "group_by"
    ] = ["method"]
    with pytest.raises(ValueError, match="non-finite"):
        extract_reference_factors(metrics, config)


def test_fixed_factors_follow_explicit_scan_models():
    config = {
        "study": {"result": {"evaluator": "best_factor"}},
        "evaluate": {
            "models": ["DiffMean", "PCA", "SFT"],
            "evaluators": {
                "scan": {
                    "type": "LMJudgeEvaluator",
                    "models": ["DiffMean", "PCA"],
                    "inference": {"strengths": [0.0, 1.0]},
                },
                "best_factor": {
                    "type": "BestFactorEvaluator",
                    "depends_on": ["scan"],
                    "params": {
                        "group_by": ["method"],
                        "baseline_factor": 0.0,
                    },
                },
            },
        },
    }
    metrics = pd.DataFrame({
        "source_evaluator": ["best_factor", "best_factor"],
        "method": ["DiffMean", "PCA"],
        "factor": [1.0, 1.0],
    })
    fixed = apply_reference_factors(
        config, extract_reference_factors(metrics, config)
    )
    assert fixed["evaluate"]["evaluators"]["scan"]["inference"][
        "strengths_by_model"
    ] == {"DiffMean": [0.0, 1.0], "PCA": [0.0, 1.0]}


def test_derived_config_reuses_generated_pool_without_changing_train_seed(tmp_path):
    config = {
        "generate": {"num_of_examples": 24},
        "train": {"seed": 7},
        "study": {"sample_efficiency": {"sizes": [6]}},
    }
    derived = derive_variant_config(
        config,
        StudyVariant(6, 13, efficiency=True),
        tmp_path / "generate",
    )
    assert "study" not in derived
    assert derived["train"]["seed"] == 7
    assert derived["train"]["subset_seed"] == 13
    assert derived["train"]["max_num_of_examples"] == 6
    assert derived["train"]["overwrite_data_dir"] == str(
        (tmp_path / "generate").resolve()
    )


def test_sensitivity_sampling_controls_do_not_change_efficiency_runs(tmp_path):
    config = {
        "generate": {"num_of_examples": 24},
        "train": {"seed": 7},
        "evaluate": {
            "temperature": 1.0,
            "evaluators": {
                "judge": {
                    "inference": {"temperature": 1.0, "do_sample": True}
                },
                "result": {"depends_on": ["judge"]},
            },
        },
        "study": {
            "sample_sensitivity": {
                "size": 24,
                "subset_seeds": [42, 43],
                "temperature": 0.0,
                "do_sample": False,
            }
        },
    }
    sensitivity = derive_variant_config(
        config,
        StudyVariant(24, 43, sensitivity=True),
        tmp_path / "generate",
    )
    assert sensitivity["evaluate"]["temperature"] == 0.0
    assert sensitivity["evaluate"]["evaluators"]["judge"]["inference"] == {
        "temperature": 0.0,
        "do_sample": False,
    }

    efficiency = derive_variant_config(
        config,
        StudyVariant(24, 42, efficiency=True),
        tmp_path / "generate",
    )
    assert efficiency["evaluate"]["temperature"] == 1.0
    assert efficiency["evaluate"]["evaluators"]["judge"]["inference"] == {
        "temperature": 1.0,
        "do_sample": True,
    }


def test_study_external_generate_dir_is_reused_without_generation(
    tmp_path, monkeypatch
):
    generated = tmp_path / "canonical" / "generate"
    generated.mkdir(parents=True)
    (generated / "metadata.jsonl").write_text(
        '{"concept_id": 0, "concept": "test"}\n', encoding="utf-8"
    )

    def fail_run(*_args, **_kwargs):
        raise AssertionError("generate.py must not run for an external pool")

    monkeypatch.setattr(
        "steerscope.scripts.training_data_study._run", fail_run
    )
    actual = _prepare_shared_generate_dir(
        str(generated),
        tmp_path / "study",
        tmp_path / "study.yaml",
        tmp_path,
    )
    assert actual == generated.resolve()


def test_study_external_generate_dir_requires_metadata(tmp_path):
    generated = tmp_path / "empty-generate"
    generated.mkdir()
    with pytest.raises(FileNotFoundError, match="metadata.jsonl"):
        _prepare_shared_generate_dir(
            str(generated),
            tmp_path / "study",
            tmp_path / "study.yaml",
            tmp_path,
        )


def test_study_reuses_only_run_invariant_reference_artifacts(tmp_path):
    reference = tmp_path / "runs" / "reference" / "train"
    reference.mkdir(parents=True)
    (reference / "GemmaScopeSAE.pt").write_bytes(b"sae")
    (reference / "GemmaScopeSAE_scale.pt").write_bytes(b"scale")
    (reference / "APSR.pt").write_bytes(b"must-not-be-shared")

    destination = tmp_path / "runs" / "child"
    reused = _reuse_reference_training_artifacts(
        reference.parent,
        destination,
    )

    assert {path.name for path in reused} == {
        "GemmaScopeSAE.pt",
        "GemmaScopeSAE_scale.pt",
    }
    assert (destination / "train/GemmaScopeSAE.pt").read_bytes() == b"sae"
    assert (destination / "train/GemmaScopeSAE_scale.pt").read_bytes() == b"scale"
    assert not (destination / "train/APSR.pt").exists()
    assert (destination / "train/GemmaScopeSAE.pt").samefile(
        reference / "GemmaScopeSAE.pt"
    )


def test_study_aggregation_keeps_efficiency_and_subset_variance_separate():
    rows = []
    for run_id, size, seed, efficiency, sensitivity, score in [
        ("n6s1", 6, 1, True, False, 0.2),
        ("n12s1", 12, 1, True, True, 0.6),
        ("n12s2", 12, 2, False, True, 0.8),
    ]:
        rows.append({
            "study_run_id": run_id,
            "train_examples": size,
            "subset_seed": seed,
            "efficiency": efficiency,
            "sensitivity": sensitivity,
            "source_evaluator": "best_factor",
            "method": "DiffMean",
            "selected_improvement": score,
            "concept_id": 0,
        })
    config = {
        "study": {"result": {"evaluator": "best_factor"},
                  "sample_sensitivity": {"subset_seeds": [1, 2]}},
        "evaluate": {
            "evaluators": {
                "best_factor": {"type": "BestFactorEvaluator"},
            }
        },
    }
    efficiency, sensitivity = aggregate_study_metrics(
        pd.DataFrame(rows), config
    )
    assert efficiency["train_examples"].tolist() == [6, 12]
    summary = sensitivity.iloc[0]
    assert summary["score_mean"] == 0.7
    assert summary["seeds"] == 2
    assert summary["score_std"] > 0


def test_relative_efficiency_uses_reference_improvement_and_filters_small_denominators():
    rows = []
    for method, small_score, reference_score in [
        ("keep", 0.12, 0.2),
        ("drop", 0.03, 0.05),
    ]:
        for run_id, size, score, factor_reference in [
            ("small", 6, small_score, False),
            ("reference", 72, reference_score, True),
        ]:
            rows.append({
                "study_run_id": run_id,
                "train_examples": size,
                "subset_seed": 42,
                "factor_reference": factor_reference,
                "efficiency": True,
                "sensitivity": False,
                "source_evaluator": "best_factor",
                "method": method,
                "selected_improvement": score,
            "concept_id": 0,
            })
    config = {
        "study": {
            "sample_efficiency": {
                "relative_report": {"min_reference_improvement": 0.1}
            },
            "result": {
                "evaluator": "best_factor",
                "metric": "selected_improvement",
            },
        },
        "evaluate": {
            "evaluators": {
                "best_factor": {"type": "BestFactorEvaluator"},
            }
        },
    }
    efficiency, _ = aggregate_study_metrics(pd.DataFrame(rows), config)
    keep = efficiency[efficiency["method"] == "keep"].set_index(
        "train_examples"
    )
    assert keep.loc[6, "relative_improvement_pct"] == pytest.approx(60.0)
    assert keep.loc[72, "relative_improvement_pct"] == pytest.approx(100.0)
    assert keep["relative_eligible"].all()
    assert not efficiency.loc[
        efficiency["method"] == "drop", "relative_eligible"
    ].any()


def test_alpaca_sampling_uses_dataset_seed_and_concept_id(tmp_path):
    pd.DataFrame({"instruction": [f"prompt-{index}" for index in range(30)]}).to_json(
        tmp_path / "alpaca_eval.json"
    )
    class ConcreteAlpacaEvaluator(AlpacaEvaluator):
        def compute_metrics(self, data, write_to_dir=None):
            return {}

    evaluator = ConcreteAlpacaEvaluator.__new__(ConcreteAlpacaEvaluator)
    evaluator.args = SimpleNamespace(master_data_dir=str(tmp_path))
    first = evaluator._load_source(8, concept_id=2, seed=42)
    repeated = evaluator._load_source(8, concept_id=2, seed=42)
    new_seed = evaluator._load_source(8, concept_id=2, seed=43)
    new_concept = evaluator._load_source(8, concept_id=3, seed=42)
    assert first.index.tolist() == repeated.index.tolist()
    assert first.index.tolist() != new_seed.index.tolist()
    assert first.index.tolist() != new_concept.index.tolist()
