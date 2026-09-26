import json

import numpy as np
import pandas as pd
import pytest

from steerscope.studies.statistics import (
    load_study_concept_scores, summarize_sensitivity,
)
from steerscope.studies.training_data import aggregate_study_metrics


def sensitivity_rows(seeds=range(42, 47)):
    return pd.DataFrame([
        dict(method="Demo", train_examples=24, concept_id=c, subset_seed=seed,
             score=x if c == 0 else 1-x)
        for seed, x in zip(seeds, np.linspace(0, 1, len(seeds))) for c in [0, 1]
    ])


def test_sensitivity_does_not_cancel_opposite_concept_shifts():
    _, result = summarize_sensitivity(sensitivity_rows())
    assert result.iloc[0].score_std == pytest.approx(np.std([0, .25, .5, .75, 1], ddof=1))
    assert result.iloc[0].score_mean == .5
    assert result.iloc[0].score_sem == 0
    assert result.iloc[0].seeds == 5


@pytest.mark.parametrize("seeds", [[42], [42, 43, 44, 45], [42, 43, 44, 45, 99]])
def test_sensitivity_rejects_missing_or_replaced_seed(seeds):
    with pytest.raises(ValueError, match="Incomplete sensitivity seeds"):
        summarize_sensitivity(sensitivity_rows(seeds))


def test_sensitivity_rejects_missing_concept_and_duplicate():
    rows = sensitivity_rows()
    with pytest.raises(ValueError, match="different concept panels"):
        summarize_sensitivity(rows.iloc[1:])
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_sensitivity(pd.concat([rows, rows.iloc[:1]]))


def write_samples(root, evaluator, values, count):
    folder = root / 'study' / 'evaluators' / evaluator
    folder.mkdir(parents=True)
    (folder/'manifest.json').write_text(json.dumps({'status': 'complete', 'config': {'models': ['Demo']}}))
    rows = []
    for c, value in values.items():
        for f in [0., 1.]:
            for i in range(count):
                score = value(i) if callable(value) else value
                rows.append(dict(method='Demo', concept_id=c, factor=f, input_id=i,
                                 source_input_id=f'c{c}-{i}', raw_aggregated_ratings=score*f))
    pd.DataFrame(rows).to_parquet(folder/'samples.parquet')


def test_file_aggregation_aligns_prompts_concepts_and_keeps_sensitivity_20(tmp_path):
    root = tmp_path/'studies'
    main = tmp_path/'methods/demo/evaluate/runs'
    write_samples(main, 'id_lm_judge', {0:1., 1:.5, 2:2.}, 10)
    selections = []
    def selection(run, n, seed, efficiency, sensitivity, reference=False):
        selections.append(dict(source_evaluator='study_result', method='Demo', study_run_id=run,
            train_examples=n, subset_seed=seed, factor=1., baseline_factor=0.,
            efficiency=efficiency, sensitivity=sensitivity, factor_reference=reference))
    selection('full',144,42,True,False,True)
    selection('n-0006_subset-42',6,42,True,False)
    write_samples(root/'runs/n-0006_subset-42/demo/evaluate/runs','study_lm_judge',
                  {0:lambda i:.2 if i<10 else 1.8,1:lambda i:.4 if i<10 else 1.6},20)
    for seed,x in zip(range(42,47),np.linspace(0,1,5)):
        run=f'n-0024_subset-{seed}'
        selection(run,24,seed,False,True)
        write_samples(root/f'runs/{run}/demo/evaluate/runs','study_lm_judge',{0:x,1:1-x},20)
    metrics=pd.DataFrame(selections)
    config={'study': {'result':{'evaluator':'study_result','metric':'selected_improvement'},
            'sample_efficiency':{'relative_report':{'min_reference_improvement':.1}},
            'sample_sensitivity':{'subset_seeds':list(range(42,47))}}}
    scores=load_study_concept_scores(metrics,config,root,{'Demo':main})
    eff,sens=aggregate_study_metrics(metrics,config,concept_scores=scores)
    assert eff.set_index('train_examples').loc[6,'score']==pytest.approx(.3)
    assert eff.set_index('train_examples').loc[144,'score']==pytest.approx(.75)
    assert eff.set_index('train_examples').loc[6,'relative_improvement_pct']==pytest.approx(40)
    assert sens.iloc[0].score_std==pytest.approx(np.std(np.linspace(0,1,5),ddof=1))
    assert eff.n_concepts.eq(2).all()
    path=root/'runs/n-0006_subset-42/demo/evaluate/runs/study/evaluators/study_lm_judge/samples.parquet'
    data=pd.read_parquet(path)
    data['source_input_id']=data.source_input_id.str.replace('c0-0','wrong',regex=False)
    data.to_parquet(path)
    with pytest.raises(ValueError,match='prompt IDs differ'):
        load_study_concept_scores(metrics,config,root,{'Demo':main})


@pytest.mark.parametrize("sensitivity_only", [False, True])
def test_scheduler_finalizes_using_concept_scores(tmp_path, sensitivity_only):
    from types import SimpleNamespace
    import yaml
    from steerscope.studies.training_data import StudyVariant
    from steerscope.sweep.paper.scheduler import Scheduler

    variants = [StudyVariant(24, seed, sensitivity=True) for seed in range(42, 47)]
    if not sensitivity_only:
        variants.append(StudyVariant(6, 42, efficiency=True))
    study_root = tmp_path / "studies"
    main = tmp_path / "methods/demo/evaluate/runs"
    if not sensitivity_only:
        write_samples(main, "id_lm_judge", {0: 1., 1: .5, 2: 2.}, 10)
    selection = pd.DataFrame([dict(method="Demo", factor=1., baseline_factor=0.,
                                   selected_improvement=.5)])
    for variant in variants:
        root = study_root / "runs" / variant.run_id / "demo/evaluate/runs"
        x = (variant.subset_seed - 42) / 4
        write_samples(root, "study_lm_judge", {0: x, 1: 1-x},
                      20 if variant.sensitivity else 10)
        selector = root / "study/evaluators/study_result"
        selector.mkdir(parents=True)
        selection.to_parquet(selector / "metrics.parquet")
    reference = (study_root / "reference_best_factors.parquet" if sensitivity_only else
                 tmp_path / "generalization/methods/demo/evaluate/runs/generalization/evaluators/best_factor/metrics.parquet")
    reference.parent.mkdir(parents=True, exist_ok=True)
    selection.to_parquet(reference)
    config = {"evaluate": {"evaluation_run_id": "study"}, "study": {
        "result": {"evaluator": "study_result", "metric": "selected_improvement"},
        "factor_selection": {"train_examples": 144, "subset_seed": 42},
        "sample_efficiency": {"relative_report": {"min_reference_improvement": .1}},
        "sample_sensitivity": {"subset_seeds": list(range(42, 47))},
    }}
    config_path = tmp_path / "study.yaml"
    config_path.write_text(yaml.safe_dump(config))
    scheduler = SimpleNamespace(
        study_status="running", study_config_path=config_path,
        _study_variants=lambda source: variants,
        methods=[SimpleNamespace(method="Demo", stem="demo")], output_dir=tmp_path,
        sensitivity_only=sensitivity_only, sample_sensitivity_enabled=True,
        study_reference_metrics_path=reference if sensitivity_only else None,
        _atomic_frame=Scheduler._atomic_frame,
    )
    Scheduler._aggregate_study(scheduler)
    result = pd.read_parquet(study_root / "sample_sensitivity.parquet")
    assert result.iloc[0].score_std == pytest.approx(np.std(np.linspace(0, 1, 5), ddof=1))
    efficiency = pd.read_parquet(study_root / "sample_efficiency.parquet")
    assert efficiency.empty == sensitivity_only
    assert json.loads((study_root / "manifest.json").read_text())["status"] == "complete"


def test_standalone_reference_uses_its_own_samples(tmp_path):
    root = tmp_path / "studies"
    records = []
    for run, n, reference, score in [("full", 144, True, 1.), ("reduced", 6, False, .5)]:
        write_samples(root / "runs" / run / "evaluate/runs", "study_lm_judge", {0: score}, 10)
        records.append(dict(source_evaluator="study_result", method="Demo", study_run_id=run,
                            train_examples=n, subset_seed=42, factor=1., baseline_factor=0.,
                            efficiency=True, sensitivity=False, factor_reference=reference))
    config = {"study": {"result": {"evaluator": "study_result", "metric": "selected_improvement"},
                       "sample_efficiency": {"relative_report": {"min_reference_improvement": .1}}}}
    scores = load_study_concept_scores(pd.DataFrame(records), config, root,
                                      {"Demo": tmp_path / "methods/demo/evaluate/runs"})
    efficiency, _ = aggregate_study_metrics(pd.DataFrame(records), config, concept_scores=scores)
    assert efficiency.set_index("train_examples").loc[6, "relative_improvement_pct"] == 50.
