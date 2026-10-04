import copy
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from steerscope.evaluation import (
    EvaluationContext,
    EvaluationEngine,
    ResultStore,
    parse_evaluator_nodes,
)
from steerscope.evaluators.best_factor import BestFactorEvaluator
from steerscope.sweep.paper.generate_configs import (
    MODEL_LAYERS,
    _ten_concept_profile,
    factors_for,
    generalization_config,
    method_config,
    sensitivity_study_config,
    study_config,
    training_models,
)
from steerscope.sweep.paper.scheduler import (
    GPUAllocator,
    GPUInfo,
    MethodTask,
    PipelineTask,
    Scheduler,
    _configure_fixed_factor_self_baseline,
    _configured_training_concept_count,
    _run_checked,
    _scope_evaluators_to_training_panel,
    _training_task_config,
    run_study_worker,
    wandb_launch_config,
)
from steerscope.utils.concept_scope import select_concept_ids


def _cli(**overrides):
    values = {
        "methods": None,
        "dry_run": False,
        "skip_study": False,
        "skip_generalization": False,
        "no_wandb": True,
        "no_resume": False,
        "retry_failed": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _source_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "sweep/paper/2b/l20"


def _scheduler_config(tmp_path, *, study=True, generalization=True):
    executable = Path(sys.executable)
    return {
        "experiment": {
            "model_key": "2b",
            "layer": 20,
            "config_dir": str(_source_dir()),
            "output_dir": str(tmp_path / "output"),
        },
        "runtime": {
            "python": str(executable),
            "torchrun": str(executable.with_name("torchrun")),
            "hot_data_root": str(tmp_path / "hot-data"),
            "lm_cache_min_free_gb": 0,
        },
        "easysteer": {"enabled": False, "gpus": []},
        "study": {"enabled": study, "gpus": [], "max_retries": 2},
        "workers": {"gpus": [0, 1], "safety_margin_gb": 6},
        "generalization": {"enabled": generalization, "gpus": []},
                "methods": {
            "defaults": {
                "gpu_count": 1,
                "nproc_per_node": 1,
                "vram_reservation_gb": 12,
                "max_retries": 1,
            }
        },
    }


def _make_scheduler(tmp_path, **config_overrides) -> Scheduler:
    config = _scheduler_config(tmp_path)
    config.update(config_overrides)
    path = tmp_path / "scheduler.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return Scheduler(path, _cli())


def _write_generated_pool(
    scheduler: Scheduler,
    count: int = 500,
    generated: Path | None = None,
) -> Path:
    generated = generated or scheduler.output_dir / "generate"
    generated.mkdir(parents=True, exist_ok=True)
    rows = []
    metadata = []
    for concept_id in range(count):
        concept = f"concept-{concept_id}"
        metadata.append({
            "concept_id": concept_id,
            "concept": concept,
            "concept_genres_map": {
                concept: ["text" if concept_id % 2 == 0 else "code"]
            },
        })
        rows.extend({"concept_id": concept_id} for _ in range(72))
    (generated / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metadata),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_parquet(generated / "train_data.parquet", index=False)
    return generated


def test_prepare_hot_storage_serializes_relative_namespace(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    scheduler._prepare_hot_storage()
    manifest = json.loads(
        (scheduler.output_dir / "runtime_configs/hot_data.json").read_text()
    )
    assert manifest["namespace"] == str(scheduler.hot_data_namespace)
    assert isinstance(manifest["namespace"], str)


def test_method_status_reporting_before_task_graph_keeps_methods_queued(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    assert scheduler.pipeline_tasks == []
    scheduler._sync_method_status()
    assert all(method.status == "queued" for method in scheduler.methods)
    assert all(method.phase == "queued" for method in scheduler.methods)
    assert all(method.progress == 0.0 for method in scheduler.methods)
    assert all(method.allocated_gpus == () for method in scheduler.methods)
    progress = scheduler._progress_values()
    assert progress["methods"] == 0.0
    assert progress["study"] == 0.0
    assert progress["generalization"] == 0.0
    updates = []
    scheduler.reporter = SimpleNamespace(
        update=lambda *args, **kwargs: updates.append((args, kwargs))
    )
    scheduler.current_stage = scheduler.stage = "generate"
    scheduler.generate_status = "running"
    scheduler._report()
    assert len(updates) == 1


def test_source_yaml_roles_are_explicit_and_disjoint():
    method_paths = sorted(
        path for path in _source_dir().glob("*.yaml")
        if path.name not in {"generalization.yaml", "study.yaml"}
    )
    assert len(method_paths) == 24
    for path in method_paths:
        config = yaml.safe_load(path.read_text())
        assert config["evaluate"]["generate_reports"] is False
        nodes = config["evaluate"]["evaluators"]
        assert "id_lm_judge" in nodes
        assert "prompt_generalization" not in nodes

    generalization = yaml.safe_load(
        (_source_dir() / "generalization.yaml").read_text()
    )
    assert generalization["evaluate"]["generate_reports"] is False
    assert set(generalization["evaluate"]["evaluators"]) == {
        "best_factor", "prompt_generalization"
    }
    study = yaml.safe_load((_source_dir() / "study.yaml").read_text())
    assert study["evaluate"]["generate_reports"] is False
    assert set(study["evaluate"]["evaluators"]) == {
        "study_lm_judge", "study_result"
    }
    assert study["study"]["result"]["evaluator"] == "study_result"


def test_generated_configs_use_configured_side_effect_sample_counts():
    spec = MODEL_LAYERS["2b/l20"]
    config = method_config("APSR", spec, training_models(spec))
    nodes = config["evaluate"]["evaluators"]
    for node_id in ("mmlu", "bbq", "truthfulqa", "math", "ifeval", "jailbreakbench"):
        assert nodes[node_id]["dataset"]["num_examples"] == 20
    assert nodes["superglue"]["dataset"]["num_examples_per_task"] == 20
    id_inference = nodes["id_lm_judge"]["inference"]
    assert id_inference["temperature"] == 1.0
    assert id_inference["do_sample"] is True

    gen = generalization_config(spec, training_models(spec))
    prompt_generalization = gen["evaluate"]["evaluators"][
        "prompt_generalization"
    ]
    assert prompt_generalization["concepts"] == {
        "genres": ["text"], "count": 50, "seed": 42
    }
    assert prompt_generalization["dataset"]["num_examples"] == 20
    inference = prompt_generalization["inference"]
    assert inference["temperature"] == 1.0
    assert inference["do_sample"] is True

    study = study_config(spec, training_models(spec))
    assert "num_concepts" not in study["train"]
    assert study["generate"]["max_concepts"] == 500
    assert study["train"]["seed"] == 42
    assert study["evaluate"]["evaluators"]["study_lm_judge"]["dataset"][
        "num_examples"
    ] == 20


@pytest.mark.parametrize(
    "ten_concepts,expected_methods", [(False, 24), (True, 21)]
)
def test_sensitivity_profiles_are_fixed_factor_deterministic_studies(
    ten_concepts, expected_methods,
):
    spec = MODEL_LAYERS["2b/l20"]
    config = sensitivity_study_config(
        spec, training_models(spec), ten_concepts=ten_concepts
    )
    assert len(config["evaluate"]["models"]) == expected_methods
    assert config["evaluate"]["temperature"] == 0.0
    judge = config["evaluate"]["evaluators"]["study_lm_judge"]
    assert judge["inference"]["temperature"] == 0.0
    assert judge["inference"]["do_sample"] is False
    assert judge["dataset"]["num_examples"] == 20
    assert "sample_efficiency" not in config["study"]
    assert config["study"]["sample_sensitivity"] == {
        "num_examples": 20,
        "size": 24,
        "subset_seeds": [42, 43, 44, 45, 46],
        "temperature": 0.0,
        "do_sample": False,
    }
    assert config["study"]["factor_selection"]["external"] is True
    if ten_concepts:
        assert {"HyperSteer", "FLAS", "SFT"}.isdisjoint(
            config["evaluate"]["models"]
        )
        assert config["train"]["max_concepts"] == 10
        assert "concept_scopes" not in config["study"]
        expected_ids = list(range(10))
        assert judge["concepts"] == {"ids": expected_ids}
        assert config["evaluate"]["evaluators"]["study_result"][
            "concepts"
        ] == {"ids": expected_ids}
    else:
        scopes = config["study"]["concept_scopes"]
        assert scopes["default"]["count"] == 50
        assert scopes["by_method"]["HyperSteer"]["count"] == 500
        assert scopes["by_method"]["FLAS"]["count"] == 500
        assert scopes["by_method"]["SFT"]["count"] == 20


def test_sensitivity_only_task_graph_reuses_pool_and_frozen_factors(tmp_path):
    spec = MODEL_LAYERS["2b/l20"]
    study = sensitivity_study_config(
        spec, training_models(spec), ten_concepts=False
    )
    study_path = tmp_path / "sensitivity-study.yaml"
    study_path.write_text(yaml.safe_dump(study), encoding="utf-8")
    reference_path = tmp_path / "main-best-factor.parquet"
    pd.DataFrame([
        {"method": method, "factor": 1.0}
        for method in study["evaluate"]["models"]
    ]).to_parquet(reference_path, index=False)

    config = _scheduler_config(tmp_path, study=True, generalization=False)
    shared_generate = tmp_path / "shared-main-generate"
    config["study"].update({
        "sensitivity_only": True,
        "config": str(study_path),
        "shared_generate_dir": str(shared_generate),
        "reference_metrics": str(reference_path),
        "judge_concurrency": 256,
    })
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    scheduler = Scheduler(config_path, _cli())
    _write_generated_pool(scheduler, generated=shared_generate)
    scheduler._prepare_tasks()

    assert not any(
        task.kind.startswith("main_") or task.kind in {
            "factor_select", "generalization"
        }
        for task in scheduler.pipeline_tasks
    )
    assert len([
        task for task in scheduler.pipeline_tasks
        if task.kind == "study_evaluate"
    ]) == 5 * 24
    train_tasks = [
        task for task in scheduler.pipeline_tasks if task.kind == "study_train"
    ]
    assert len(train_tasks) == 5 * sum(
        method.requires_training for method in scheduler.methods
    )
    concepts = {
        method: {
            task.training_concepts
            for task in train_tasks
            if task.method == method
        }
        for method in ("HyperSteer", "FLAS", "SFT", "DiffMean")
    }
    assert concepts == {
        "HyperSteer": {500},
        "FLAS": {500},
        "SFT": {20},
        "DiffMean": {50},
    }
    snapshot = scheduler.output_dir / "studies/reference_best_factors.parquet"
    assert snapshot.is_file()
    assert set(pd.read_parquet(snapshot)["method"]) == set(
        study["evaluate"]["models"]
    )
    for task in scheduler.pipeline_tasks:
        if task.kind != "study_evaluate":
            continue
        job = json.loads(task.config_path.read_text())
        assert Path(job["shared_generate"]) == shared_generate.resolve()
        assert Path(job["reference_metrics"]) == snapshot.resolve()
        assert job["judge_concurrency"] == 256


def test_sft_is_one_normal_seeded_twenty_concept_method():
    spec = MODEL_LAYERS["2b/l20"]
    config = method_config("SFT", spec, training_models(spec))
    assert config["train"]["models"] == {"SFT": training_models(spec)["SFT"]}
    assert config["train"]["num_concepts"] == 20
    assert config["train"]["seed"] == 42
    assert all(
        node["concepts"] == {"count": 20, "seed": 42}
        for node in config["evaluate"]["evaluators"].values()
    )


@pytest.mark.parametrize("model_layer", sorted(MODEL_LAYERS))
def test_study_result_uses_judge_concept_panel(model_layer):
    spec = MODEL_LAYERS[model_layer]
    config = study_config(spec, training_models(spec))
    nodes = config["evaluate"]["evaluators"]
    assert nodes["study_result"].get("concepts") == {"count": 50, "seed": 42}
    assert nodes["study_result"]["concepts"] == nodes["study_lm_judge"]["concepts"]
    assert (
        nodes["study_result"]["concepts"]
        is not nodes["study_lm_judge"]["concepts"]
    )
    legacy = copy.deepcopy(config)
    legacy["evaluate"]["evaluators"]["study_result"].pop("concepts")
    assert _training_task_config(config) == _training_task_config(legacy)
    smoke = _ten_concept_profile(config)
    assert all(
        "concepts" not in node
        for node in smoke["evaluate"]["evaluators"].values()
    )


def test_checked_in_study_result_scopes_match_upstream():
    paths = sorted(_source_dir().parents[1].glob("*/*/study.yaml"))
    assert paths
    for path in paths:
        config = yaml.safe_load(path.read_text())
        nodes = config["evaluate"]["evaluators"]
        assert (
            nodes["study_result"].get("concepts")
            == nodes["study_lm_judge"].get("concepts")
        ), str(path)


@pytest.mark.parametrize(
    "pool_size,training_count", [(500, None), (500, 20), (10, None), (10, 5)],
)
def test_study_result_reuses_cached_judge_for_its_panel(
    tmp_path, pool_size, training_count,
):
    scheduler = _make_scheduler(tmp_path)
    shared = _write_generated_pool(scheduler, count=pool_size)
    spec = MODEL_LAYERS["2b/l20"]
    config = study_config(spec, training_models(spec))
    if pool_size == 10:
        config = _ten_concept_profile(config)
    if training_count is not None:
        _scope_evaluators_to_training_panel(
            config, generate_dir=shared, count=training_count, seed=42,
        )
    method = "SFT" if training_count is not None else "FLAS"
    _configure_fixed_factor_self_baseline(
        config, scan_id="study_lm_judge", method=method, factor=2.5,
    )
    nodes = parse_evaluator_nodes(
        config["evaluate"]["evaluators"],
        evaluator_resolver=lambda name: (
            BestFactorEvaluator if name == "BestFactorEvaluator" else None
        ),
    )
    scan, selector = nodes
    scope = scan.concepts
    ids = list(scope["ids"]) if "ids" in scope else select_concept_ids(
        range(pool_size), count=int(scope.get("count", pool_size)), seed=42,
    )
    metrics = pd.DataFrame([
        {
            "method": method, "concept_id": concept_id,
            "factor": factor, "lm_judge_rating": score,
        }
        for concept_id in ids for factor, score in ((0.0, 0.2), (2.5, 0.7))
    ])
    store = ResultStore(tmp_path / "results")
    store.save_metrics(scan.node_id, metrics)
    store.mark_complete(scan.node_id, scan.config_hash(), scan.as_dict(), metadata={
        "result_kinds": ["metrics"],
        "concept_scope": {"genres": None, "concept_ids": ids},
    })
    cached_manifest = store.manifest(scan.node_id)
    executed = []

    def execute(node, results):
        executed.append(node.node_id)
        assert node.node_id == selector.node_id, "Cached judge must not run again"
        evaluator = BestFactorEvaluator(node, EvaluationContext(
            args=SimpleNamespace(), root_dump_dir=tmp_path,
            output_dir=tmp_path / "report", results=results,
        ))
        return evaluator.evaluate(
            [], [SimpleNamespace(concept_id=i) for i in range(pool_size)],
        )

    engine = EvaluationEngine(nodes, store, execute, generate_reports=False)
    assert engine.run() == [scan.node_id, selector.node_id]
    assert executed == [selector.node_id]
    assert store.manifest(scan.node_id) == cached_manifest
    selected = store.load(selector.node_id, "metrics").iloc[0]
    assert selected["selected_score"] == pytest.approx(0.7)
    assert selected["selected_improvement"] == pytest.approx(0.5)
    assert set(store.load(selector.node_id, "samples")["concept_id"]) == set(ids)


def test_ten_concept_profile_preserves_dataset_semantics():
    source = {
        "generate": {"max_concepts": 500},
        "evaluate": {"evaluators": {
            "slow": {"concepts": {"count": 50, "seed": 42}},
            "text": {
                "concepts": {"genres": ["text"], "count": 50, "seed": 42}
            },
        }},
        "study": {"sample_efficiency": {"sizes": [6, 12, 36, 72]}},
    }
    result = _ten_concept_profile(source)
    assert result["generate"]["max_concepts"] == 10
    assert "concepts" not in result["evaluate"]["evaluators"]["slow"]
    assert result["evaluate"]["evaluators"]["text"]["concepts"] == {
        "genres": ["text"]
    }
    assert result["study"]["sample_efficiency"]["sizes"] == [8, 12, 36, 72]
    assert source["generate"]["max_concepts"] == 500


def test_ten_concept_profile_can_limit_sft_to_five_concepts():
    source = method_config(
        "SFT", MODEL_LAYERS["2b/l20"], training_models(MODEL_LAYERS["2b/l20"])
    )
    result = _ten_concept_profile(source, sft_concepts=5)
    assert result["generate"]["max_concepts"] == 10
    assert result["train"]["num_concepts"] == 5
    assert all(
        node["concepts"] == {"count": 5, "seed": 42}
        for node in result["evaluate"]["evaluators"].values()
    )


def test_run_checked_handles_standard_subprocess_exit_codes(tmp_path):
    _run_checked([sys.executable, "-c", "pass"], cwd=tmp_path)
    with pytest.raises(subprocess.CalledProcessError) as error:
        _run_checked(
            [sys.executable, "-c", "raise SystemExit(7)"], cwd=tmp_path
        )
    assert error.value.returncode == 7


def test_factor_zero_and_fixed_method_baselines_are_distinct():
    assert factors_for("APSR", include_baseline=True)[0] == 0.0
    assert factors_for("SFT", include_baseline=True) == [1.0]
    config = {
        "evaluate": {
            "models": ["SFT", "DiffMean"],
            "evaluators": {
                "scan": {
                    "type": "LMJudgeEvaluator",
                    "models": ["SFT", "DiffMean"],
                    "params": {},
                    "inference": {"strengths_by_model": {
                        "SFT": [1.0], "DiffMean": [0.0]
                    }},
                },
                "selector": {
                    "type": "BestFactorEvaluator",
                    "depends_on": ["scan"],
                    "params": {"baseline_factor": 0.0},
                },
            },
        }
    }
    _configure_fixed_factor_self_baseline(
        config, scan_id="scan", method="SFT", factor=1.0
    )
    evaluate = config["evaluate"]
    assert evaluate["models"] == ["SFT"]
    assert evaluate["evaluators"]["scan"]["params"]["include_baseline"] is True
    assert evaluate["evaluators"]["scan"]["inference"]["strengths_by_model"] == {
        "SFT": [1.0]
    }


def test_model_specific_factor_grids_and_spsr_recipe():
    assert factors_for("ODESteer", model_key="2b") == [
        5.0, 10.0, 15.0, 20.0, 40.0, 60.0, 80.0,
        100.0, 120.0, 140.0, 180.0, 250.0, 350.0, 500.0,
    ]
    assert factors_for("ODESteer", model_key="9b") == [
        10.0, 15.0, 18.0, 20.0, 36.0, 54.0, 72.0,
        90.0, 108.0, 144.0, 180.0, 225.0, 360.0, 450.0,
    ]
    assert factors_for("SPSR", model_key="2b") == [
        0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4,
        1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0,
    ]
    recipe = training_models(MODEL_LAYERS["2b/l20"])["SPSR"]
    assert recipe["batch_size"] == 1
    assert recipe["n_epochs"] == 15
    assert recipe["lr"] == pytest.approx(1e-3)
    assert recipe["weight_decay"] == pytest.approx(1e-6)


def test_training_panel_is_order_independent_and_shared_by_nodes(tmp_path):
    generated = tmp_path / "generate"
    generated.mkdir()
    metadata = [
        {
            "concept_id": value,
            "concept": f"c{value}",
            "concept_genres_map": {
                f"c{value}": ["text" if value % 2 == 0 else "code"]
            },
        }
        for value in range(10)
    ]
    (generated / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metadata),
        encoding="utf-8",
    )
    config = {"train": {"num_concepts": 4, "seed": 17}, "evaluate": {
        "evaluators": {
            "all": {},
            "text": {"concepts": {"genres": ["text"]}},
            "derived": {"depends_on": ["text"]},
        }
    }}
    expected = select_concept_ids(range(10), count=4, seed=17)
    assert expected == select_concept_ids(reversed(range(10)), count=4, seed=17)
    selected = _scope_evaluators_to_training_panel(
        config, generate_dir=generated, count=4, seed=17
    )
    assert selected == tuple(expected)
    assert _configured_training_concept_count(config) == 4
    nodes = config["evaluate"]["evaluators"]
    assert nodes["all"]["concepts"]["ids"] == sorted(expected)
    assert set(nodes["text"]["concepts"]["ids"]) <= {0, 2, 4, 6, 8}


def test_scheduler_builds_exact_two_phase_task_graph(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    _write_generated_pool(scheduler)
    scheduler._prepare_tasks()

    phase1 = scheduler._phase_tasks(1)
    phase2 = scheduler._phase_tasks(2)
    assert len(scheduler.methods) == 24
    assert len(phase1) == 254
    assert len(phase2) == 240
    assert {task.kind for task in phase1} == {
        "main_train", "main_evaluate", "study_train"
    }
    assert {task.kind for task in phase2} == {
        "generalization", "study_evaluate"
    }
    assert len([task for task in phase1 if task.kind == "main_evaluate"]) == 24
    assert len([task for task in phase2 if task.kind == "generalization"]) == 24
    assert all(not task.uses_api for task in phase1 if task.kind.endswith("train"))
    assert all(task.uses_api for task in phase1 if task.kind == "main_evaluate")
    assert all(task.uses_api for task in phase2)
    for task in scheduler.pipeline_tasks:
        if task.kind not in {"study_train", "study_evaluate"}:
            continue
        expected = (
            20 if task.method == "SFT"
            else 500 if task.method in {"HyperSteer", "FLAS"}
            else 50
        )
        assert task.training_concepts == expected
        job = json.loads(task.config_path.read_text())
        assert job["training_concept_scope"] == {
            "count": expected,
            "seed": 42,
        }


def test_main_evaluator_workers_split_dependency_components(tmp_path):
    config = _scheduler_config(
        tmp_path, study=False, generalization=False
    )
    config["methods"].update({
        "include": ["spsr"],
        "providers": ["diff_mean"],
        "overrides": {"SPSR": {"evaluator_workers": 4}},
    })
    path = tmp_path / "scheduler.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    scheduler = Scheduler(path, _cli())
    _write_generated_pool(scheduler)
    scheduler._prepare_tasks()

    shards = [
        task for task in scheduler.pipeline_tasks
        if task.method == "SPSR" and task.kind == "main_evaluate"
    ]
    assert len(shards) == 8
    assert {task.evaluator_ids for task in shards} == {
        ("mmlu",),
        ("bbq",),
        ("truthfulqa",),
        ("superglue",),
        ("math", "math_output_length"),
        ("ifeval",),
        ("jailbreakbench",),
        ("id_lm_judge",),
    }
    assert {task.concurrency_limit for task in shards} == {4}
    assert len({task.concurrency_key for task in shards}) == 1
    for task in shards:
        runtime = yaml.safe_load(task.config_path.read_text())
        assert tuple(runtime["evaluate"]["evaluators"]) == task.evaluator_ids
        expected_models = (
            ["SPSR"]
            if "id_lm_judge" in task.evaluator_ids
            else ["SPSR", "DiffMean"]
        )
        assert runtime["evaluate"]["models"] == expected_models
        assert list(runtime["evaluate"]["artifact_dirs_by_model"]) == (
            expected_models
        )
        assert list(runtime["train"]["models"]) == expected_models

    barrier = scheduler.pipeline_task_map["phase1:evaluate:spsr"]
    assert barrier.kind == "main_evaluate_barrier"
    assert barrier.gpu_count == 0
    assert set(barrier.dependencies) == {task.stem for task in shards}
    assert set(barrier.evaluator_ids) == {
        evaluator_id for task in shards for evaluator_id in task.evaluator_ids
    }

    for task in shards[:4]:
        task.status = "running"
    assert scheduler._concurrency_slot_available(shards[4]) is False
    shards[0].status = "complete"
    assert scheduler._concurrency_slot_available(shards[4]) is True

    for task in scheduler._phase_tasks(1):
        task.status = "complete"
    judge = next(
        task for task in shards if "id_lm_judge" in task.evaluator_ids
    )
    judge.status = "running"
    barrier.status = "queued"
    assert scheduler._phase_complete(1) is False


def test_factor_expansion_preserves_only_semantically_compatible_study_receipts(
    tmp_path,
):
    scheduler = _make_scheduler(tmp_path)
    _write_generated_pool(scheduler)
    scheduler._prepare_tasks()

    apsr_train = scheduler.pipeline_task_map[
        "phase1:study-train:n-0036_subset-42:apsr"
    ]
    spherical_train = scheduler.pipeline_task_map[
        "phase1:study-train:n-0036_subset-42:spherical_steering"
    ]
    apsr_eval = scheduler.pipeline_task_map[
        "phase2:study-evaluate:n-0036_subset-42:apsr"
    ]
    spherical_eval = scheduler.pipeline_task_map[
        "phase2:study-evaluate:n-0036_subset-42:spherical_steering"
    ]

    # Training never depends on an inference factor, so both old artifacts
    # remain valid.  Evaluation must be redone only where the selected factor
    # can change; APSR's pre-SPSR receipt remains compatible.
    assert len(apsr_train.compatible_signatures) == 1
    assert len(spherical_train.compatible_signatures) == 1
    assert len(apsr_eval.compatible_signatures) == 1
    assert apsr_eval.accept_legacy_signature is True
    assert spherical_eval.compatible_signatures == ()
    assert spherical_eval.accept_legacy_signature is False


def test_legacy_study_receipt_is_migrated_once_not_accepted_forever(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    _write_generated_pool(scheduler)
    scheduler._prepare_tasks()
    task = scheduler.pipeline_task_map[
        "phase2:study-evaluate:n-0036_subset-42:apsr"
    ]
    for evaluator_id in task.evaluator_ids:
        root = (
            task.output_dir / "evaluate/runs" / task.run_id
            / "evaluators" / evaluator_id
        )
        root.mkdir(parents=True, exist_ok=True)
        (root / "manifest.json").write_text(json.dumps({
            "status": "complete", "metadata": {"result_kinds": []}
        }), encoding="utf-8")
    task.receipt_path.parent.mkdir(parents=True, exist_ok=True)
    task.receipt_path.write_text(json.dumps({
        "version": 1,
        "status": "complete",
        "signature": "historical-shared-study-signature",
    }), encoding="utf-8")

    assert scheduler._receipt_complete(task) is True
    migrated = json.loads(task.receipt_path.read_text())
    assert migrated["signature"] == task.signature
    assert migrated["signature_schema"] == 2

    task.signature = "a-later-real-config-change"
    task.compatible_signatures = ()
    assert scheduler._receipt_complete(task) is False



def test_scheduler_can_disable_only_sample_sensitivity(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    _write_generated_pool(scheduler)
    scheduler._prepare_tasks()
    retained = {
        task.stem: task.signature
        for task in scheduler.pipeline_tasks
        if task.kind in {"study_train", "study_evaluate"}
        and any(
            run_id in task.stem
            for run_id in (
                "n-0006_subset-42",
                "n-0012_subset-42",
                "n-0036_subset-42",
                "n-0072_subset-42",
            )
        )
    }

    scheduler.sample_sensitivity_enabled = False
    scheduler._prepare_tasks()
    phase1 = scheduler._phase_tasks(1)
    phase2 = scheduler._phase_tasks(2)
    assert len(phase1) == 139
    assert len(phase2) == 120
    assert len([task for task in phase1 if task.kind == "study_train"]) == 92
    assert len([task for task in phase2 if task.kind == "study_evaluate"]) == 96
    assert not any(
        any(f"n-0036_subset-{seed}" in task.stem for seed in range(43, 47))
        for task in scheduler.pipeline_tasks
    )
    assert {
        task.stem: task.signature
        for task in scheduler.pipeline_tasks
        if task.stem in retained
    } == retained


def test_phase_two_uses_phase_one_factors_without_id_rescan(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    _write_generated_pool(scheduler)
    scheduler._prepare_tasks()

    for method in scheduler.methods:
        main = scheduler.pipeline_task_map[f"phase1:evaluate:{method.stem}"]
        generalization = scheduler.pipeline_task_map[
            f"phase2:generalization:{method.stem}"
        ]
        runtime = yaml.safe_load(generalization.config_path.read_text())
        nodes = runtime["evaluate"]["evaluators"]
        assert set(nodes) == {"best_factor", "prompt_generalization"}
        assert nodes["best_factor"]["depends_on"] == []
        input_path = Path(nodes["best_factor"]["input"]["path"])
        assert input_path == (
            main.output_dir / "evaluate/runs" / main.run_id
            / "evaluators/id_lm_judge/metrics.parquet"
        ).resolve()
        assert generalization.dependencies == (main.stem,)

    for task in scheduler._phase_tasks(2):
        if task.kind != "study_evaluate":
            continue
        job = json.loads(task.config_path.read_text())
        factor_task = scheduler.pipeline_task_map[
            f"phase2:generalization:{next(
                method.stem for method in scheduler.methods
                if method.method == task.method
            )}"
        ]
        expected = (
            factor_task.output_dir / "evaluate/runs" / factor_task.run_id
            / "evaluators/best_factor/metrics.parquet"
        ).resolve()
        assert Path(job["reference_metrics"]) == expected
        study_nodes = yaml.safe_load(
            (_source_dir() / "study.yaml").read_text()
        )["evaluate"]["evaluators"]
        assert "id_lm_judge" not in study_nodes


def test_every_main_inference_node_has_factors_for_every_routed_model(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    _write_generated_pool(scheduler)
    scheduler._prepare_tasks()
    for method in scheduler.methods:
        runtime = yaml.safe_load(method.runtime_config.read_text())
        for node_id, node in runtime["evaluate"]["evaluators"].items():
            inference = node.get("inference")
            if not isinstance(inference, dict):
                continue
            by_model = inference.get("strengths_by_model") or {}
            generic = inference.get("strengths", inference.get("factors"))
            for routed_method in node["models"]:
                assert routed_method in by_model or generic, (
                    method.method, node_id, routed_method
                )


def test_setting_factor_map_atomically_binds_node_models():
    node = {
        "models": ["stale-model"],
        "inference": {"strengths": [1.0]},
    }
    Scheduler._set_factors(node, {"SFT": [1.0], "DiffMean": [0.0]})
    assert node["models"] == ["SFT", "DiffMean"]
    assert node["inference"] == {
        "strengths_by_model": {"SFT": [1.0], "DiffMean": [0.0]}
    }


def test_sft_has_one_main_task_and_no_shard_protocol(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    _write_generated_pool(scheduler)
    scheduler._prepare_tasks()
    sft_method = next(method for method in scheduler.methods if method.method == "SFT")
    main_train = [
        task for task in scheduler.pipeline_tasks
        if task.kind == "main_train" and task.method == "SFT"
    ]
    assert len(main_train) == 1
    assert "shard" not in " ".join(main_train[0].command).lower()
    config = yaml.safe_load(main_train[0].config_path.read_text())
    assert config["train"]["num_concepts"] == 20
    assert config["train"]["seed"] == 42
    evaluation = yaml.safe_load(sft_method.runtime_config.read_text())
    assert evaluation["train"]["num_concepts"] == 20
    nodes = evaluation["evaluate"]["evaluators"]
    assert nodes["id_lm_judge"]["models"] == ["SFT"]
    assert nodes["mmlu"]["models"] == ["SFT", "DiffMean"]
    assert all(
        len(node["concepts"]["ids"]) <= 20
        for node in nodes.values()
    )


def test_main_side_effects_use_one_shared_diffmean_baseline(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    _write_generated_pool(scheduler)
    scheduler._prepare_tasks()
    apsr = next(method for method in scheduler.methods if method.method == "APSR")
    evaluation = yaml.safe_load(apsr.runtime_config.read_text())["evaluate"]
    mmlu = evaluation["evaluators"]["mmlu"]
    assert mmlu["models"] == ["APSR", "DiffMean"]
    assert mmlu["inference"]["strengths_by_model"] == {
        "APSR": factors_for("APSR"),
        "DiffMean": [0.0],
    }
    assert evaluation["evaluators"]["id_lm_judge"]["models"] == ["APSR"]
    # APSR sorts before DiffMean, so this also guards against making the
    # cross-method dependency conditional on discovery/priority order.
    apsr_evaluate = scheduler.pipeline_task_map["phase1:evaluate:apsr"]
    assert "phase1:train:diff_mean" in apsr_evaluate.dependencies
    assert len(apsr_evaluate.compatible_signatures) == 1



def test_sensitivity_phase1_only_does_not_require_factor_reference(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    scheduler.sensitivity_only = True
    scheduler.cli.phase1_only = True
    scheduler.config["generalization"] = {"enabled": False}
    scheduler.reference_output_dir = tmp_path / "missing-main-results"
    _write_generated_pool(scheduler)

    scheduler._prepare_tasks()

    assert scheduler.pipeline_tasks
    assert all(task.stem.startswith("phase1:") for task in scheduler.pipeline_tasks)
    assert all(task.kind == "study_train" for task in scheduler.pipeline_tasks)
    assert scheduler.study_reference_metrics_path is None


def test_main_completion_boundary_ignores_study_training(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    _write_generated_pool(scheduler)
    scheduler._prepare_tasks()
    for task in scheduler.pipeline_tasks:
        if task.kind == "main_train":
            task.status = "complete"
        elif task.kind == "main_evaluate":
            task.status = "complete"
        elif task.kind == "study_train":
            task.status = "running"

    assert scheduler._main_inference_ready() is True
    assert scheduler._phase_complete(1) is False


def test_easysteer_is_needed_only_for_unfinished_supported_methods(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    _write_generated_pool(scheduler)
    scheduler._prepare_tasks()
    evaluations = [
        task for task in scheduler.pipeline_tasks
        if task.kind == "main_evaluate"
    ]
    for task in evaluations:
        task.status = "complete"

    spsr = next(task for task in evaluations if task.method == "SPSR")
    spsr.status = "queued"
    assert scheduler._needs_easysteer() is False

    diff_mean = next(task for task in evaluations if task.method == "DiffMean")
    diff_mean.status = "queued"
    assert scheduler._needs_easysteer() is True

    diff_mean.status = "complete"
    assert scheduler._needs_easysteer() is False



def test_receipt_never_hides_deleted_training_or_evaluation_outputs(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    _write_generated_pool(scheduler)
    scheduler._prepare_tasks()
    train = scheduler.pipeline_task_map["phase1:train:apsr"]
    train.receipt_path.parent.mkdir(parents=True, exist_ok=True)
    train.receipt_path.write_text(json.dumps({
        "status": "complete", "signature": train.signature
    }), encoding="utf-8")
    assert scheduler._receipt_complete(train) is False
    artifact = train.output_dir / "train/artifact_manifest.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("{}", encoding="utf-8")
    assert scheduler._receipt_complete(train) is True

    evaluate = scheduler.pipeline_task_map["phase1:evaluate:apsr"]
    evaluate.receipt_path.parent.mkdir(parents=True, exist_ok=True)
    evaluate.receipt_path.write_text(json.dumps({
        "status": "complete", "signature": evaluate.signature
    }), encoding="utf-8")
    assert scheduler._receipt_complete(evaluate) is False



def test_study_worker_trains_then_evaluates_at_external_factor(tmp_path, monkeypatch):
    shared = tmp_path / "generate"
    shared.mkdir()
    (shared / "metadata.jsonl").write_text(
        '{"concept_id":0,"concept":"c0","concept_genres_map":{"c0":["text"]}}\n',
        encoding="utf-8",
    )
    source = study_config(MODEL_LAYERS["2b/l20"], training_models(MODEL_LAYERS["2b/l20"]))
    source["generate"]["max_concepts"] = 1
    for node in source["evaluate"]["evaluators"].values():
        node["concepts"]["count"] = 1
    source_path = tmp_path / "study.yaml"
    source_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    reference = tmp_path / "factor.parquet"
    pd.DataFrame({"method": ["APSR"], "factor": [0.4]}).to_parquet(reference)
    output = tmp_path / "run"
    job = {
        "stage": "evaluate",
        "python": sys.executable,
        "torchrun": str(Path(sys.executable).with_name("torchrun")),
        "nproc_per_node": 1,
        "method": "APSR",
        "variant": {
            "train_examples": 36,
            "subset_seed": 43,
            "efficiency": False,
            "sensitivity": True,
            "factor_reference": False,
        },
        "task_output": str(output),
        "shared_generate": str(shared),
        "shared_request_cache_dir": str(tmp_path / "cache"),
        "progress_root": str(tmp_path / "progress"),
        "progress_source_root": str(tmp_path),
        "study_config": str(source_path),
        "reference_metrics": str(reference),
        "easysteer_url": "http://127.0.0.1:8017",
        "judge_concurrency": 8,
        "training_concept_scope": {"count": 1, "seed": 42},
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        "steerscope.sweep.paper.scheduler._run_checked",
        lambda command, *, cwd: calls.append(command),
    )
    assert run_study_worker(job_path) == 0
    runtime = yaml.safe_load((output / "runtime/evaluate.yaml").read_text())
    nodes = runtime["evaluate"]["evaluators"]
    assert set(nodes) == {"study_lm_judge", "study_result"}
    assert nodes["study_lm_judge"]["inference"]["strengths_by_model"] == {
        "APSR": [0.4]
    }
    assert nodes["study_lm_judge"]["params"]["include_baseline"] is True
    assert len(calls) == 1 and "evaluate.py" in " ".join(calls[0])


def test_generate_pool_accepts_steerscope_positive_half_layout(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    _write_generated_pool(scheduler, count=10)
    for source in scheduler.source_configs.values():
        source["generate"]["max_concepts"] = 10
    assert scheduler._validate_generate_pool() == (True, "ok")


def test_generate_pool_counts_only_positive_rows_in_paired_layout(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    generated = scheduler.output_dir / "generate"
    generated.mkdir(parents=True)
    metadata = []
    rows = []
    for concept_id in range(10):
        concept = f"concept-{concept_id}"
        metadata.append({"concept_id": concept_id, "concept": concept})
        rows.extend(
            {"concept_id": concept_id, "category": category}
            for category in ("positive", "negative")
            for _ in range(72)
        )
    (generated / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metadata),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_parquet(generated / "train_data.parquet", index=False)
    for source in scheduler.source_configs.values():
        source["generate"]["max_concepts"] = 10
    assert scheduler._validate_generate_pool() == (True, "ok")


def test_gpu_allocator_packs_tasks_and_preserves_service_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(
        GPUAllocator,
        "query",
        staticmethod(lambda: {
            0: GPUInfo(0, total_gb=80, free_gb=60),
            1: GPUInfo(1, total_gb=80, free_gb=80),
        }),
    )
    allocator = GPUAllocator(
        [0, 1], safety_margin_gb=6, reserve_initial_usage=True
    )
    allocator.validate()

    def task(name, vram):
        return MethodTask(
            stem=name, method=name, source_config=tmp_path / f"{name}.yaml",
            runtime_config=tmp_path / f"runtime-{name}.yaml",
            output_dir=tmp_path / name, train_models=(name,), evaluator_ids=(),
            gpu_count=1, nproc_per_node=1, vram_gb=vram, exclusive=False,
            priority=0, max_retries=0,
        )

    large = task("large", 56)
    assert allocator.allocate(large) == (1,)
    first = task("first", 28)
    second = task("second", 28)
    assert allocator.allocate(first) == (0,)
    assert allocator.allocate(second) is None

    allocator.release_persistent_reservations([0])
    assert allocator.allocate(second) == (0,)


def test_scheduler_rejects_easysteer_parallelism_mismatch(tmp_path):
    config = _scheduler_config(tmp_path)
    config["easysteer"] = {
        "enabled": True,
        "executable": "/opt/easysteer/bin/vllm",
        "gpus": [0],
        "tensor_parallel_size": 2,
        "data_parallel_size": 1,
    }
    path = tmp_path / "scheduler.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        Scheduler(path, _cli(dry_run=True))


def test_wandb_launch_config_creates_new_timestamped_run():
    launched_at = dt.datetime(2026, 8, 13, 4, 21, 9, 123456, tzinfo=dt.UTC)
    config = wandb_launch_config(
        {"name": "steerscope-2b-l20", "run_id": "steerscope-2b-l20"},
        "unused-default",
        launched_at,
    )
    assert config["name"] == "steerscope-2b-l20-20260813-042109-123Z"
    assert config["run_id"] == config["name"]
    assert config["resume"] == "never"
