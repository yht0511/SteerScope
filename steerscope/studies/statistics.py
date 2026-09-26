"""Shared concept-level statistics for study summaries and notebook tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

SENSITIVITY_SEEDS = (42, 43, 44, 45, 46)
EFFICIENCY_PROMPTS = 10
SENSITIVITY_PROMPTS = 20


def paired_concept_effects(scores, factor, baseline_factor, metric):
    values = scores.groupby(["concept_id", "factor"])[metric].mean().unstack("factor")
    selected = [x for x in values if np.isclose(x, float(factor))]
    baseline = [x for x in values if np.isclose(x, float(baseline_factor))]
    if len(selected) != 1 or len(baseline) != 1:
        raise ValueError("Missing selected or baseline factor in study scores.")
    panel = values[[selected[0], baseline[0]]]
    if panel.empty or not np.isfinite(panel.to_numpy()).all():
        raise ValueError("Incomplete selected/baseline concept panel.")
    return (panel.iloc[:, 0] - panel.iloc[:, 1]).rename("score").reset_index()


def summarize_sensitivity(rows, expected_seeds=SENSITIVITY_SEEDS):
    """Average concept-wise seed SDs; SEM describes their macro-average."""
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    expected = {int(s) for s in expected_seeds}
    if len(expected) < 2:
        raise ValueError("Sensitivity requires at least two distinct subset seeds.")
    keys = ["method", "train_examples", "concept_id", "subset_seed"]
    if rows.duplicated(keys).any():
        raise ValueError("Duplicate concept/seed study scores.")
    if not np.isfinite(rows["score"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite sensitivity scores.")
    for key, group in rows.groupby(["method", "train_examples"]):
        observed = set(group.subset_seed.astype(int))
        if observed != expected:
            raise ValueError(f"Incomplete sensitivity seeds for {key}: expected {sorted(expected)}, got {sorted(observed)}")
        panels = [frozenset(part.concept_id) for _, part in group.groupby("subset_seed")]
        if len(set(panels)) != 1:
            raise ValueError(f"Sensitivity seeds use different concept panels: {key}")
    by_concept = rows.groupby(["method", "train_examples", "concept_id"], as_index=False).agg(
        score_mean=("score", "mean"), concept_seed_sd=("score", "std"),
        seeds=("subset_seed", "nunique"),
    )
    summary = by_concept.groupby(["train_examples", "method"], as_index=False).agg(
        score_mean=("score_mean", "mean"), score_std=("concept_seed_sd", "mean"),
        concept_sd_min=("concept_seed_sd", "min"), concept_sd_max=("concept_seed_sd", "max"),
        sd_of_sds=("concept_seed_sd", "std"), n_concepts=("concept_id", "nunique"),
        seeds=("seeds", "min"),
    )
    summary["score_sem"] = summary["sd_of_sds"] / np.sqrt(summary["n_concepts"])
    summary["ci95"] = summary["score_sem"] * t.ppf(.975, summary["n_concepts"] - 1)
    summary["ci95_lower"] = (summary["score_std"] - summary["ci95"]).clip(lower=0)
    summary["ci95_upper"] = summary["score_std"] + summary["ci95"]
    return by_concept, summary.drop(columns="sd_of_sds")


def align_efficiency_concepts(rows):
    """Restrict the full-data reference to exactly the study concepts."""
    aligned = []
    for method, group in rows.groupby("method", sort=False):
        variants = group[~group.factor_reference.fillna(False)]
        references = group[group.factor_reference.fillna(False)]
        if variants.empty:
            continue
        panels = [frozenset(part.concept_id) for _, part in variants.groupby("study_run_id")]
        if len(set(panels)) != 1:
            raise ValueError(f"Efficiency variants use different concept panels: {method}")
        if references.empty:
            aligned.append(variants)
            continue
        if references.study_run_id.nunique() != 1 or not panels[0].issubset(set(references.concept_id)):
            raise ValueError(f"Missing or ambiguous full-data concept reference: {method}")
        aligned.extend([variants, references[references.concept_id.isin(panels[0])]])
    return pd.concat(aligned, ignore_index=True) if aligned else rows.iloc[:0].copy()


def complete_sample_file(root, evaluator):
    paths = list(Path(root).glob(f"*/evaluators/{evaluator}/samples.parquet"))
    completed = [p for p in paths if p.with_name("manifest.json").is_file()
                 and json.loads(p.with_name("manifest.json").read_text()).get("status") == "complete"]
    if len(completed) != 1:
        raise ValueError(f"Need exactly one completed {evaluator} sample file below {root}; found {len(completed)}")
    return completed[0]


def _sample_scores(path, method, factor, baseline_factor, count):
    data = pd.read_parquet(path)
    data["factor"] = pd.to_numeric(data["factor"], errors="raise")
    own = data[data.method.eq(method)].copy()
    if not np.isclose(own.factor, baseline_factor).any():
        shared = data[data.method.eq("DiffMean") & np.isclose(data.factor, baseline_factor)].copy()
        shared["method"] = method
        own = pd.concat([own, shared], ignore_index=True)
    data = own
    data["input_id"] = pd.to_numeric(data["input_id"], errors="raise")
    data = data[data.input_id.between(0, count - 1)].copy()
    data = data[np.isclose(data.factor, factor) | np.isclose(data.factor, baseline_factor)]
    keys = ["concept_id", "factor", "source_input_id"]
    if data.empty or data.duplicated(keys).any():
        raise ValueError(f"Missing or duplicate study prompt scores: {path}")
    for concept, panel in data.groupby("concept_id"):
        ids = [frozenset(part.source_input_id) for _, part in panel.groupby("factor")]
        if len(ids) != 2 or len(set(ids)) != 1 or len(ids[0]) != count:
            raise ValueError(f"Unpaired {count}-prompt panel for {method}/{concept}: {path}")
    column = "raw_aggregated_ratings"
    if not np.isfinite(data[column].to_numpy(dtype=float)).all():
        raise ValueError(f"Invalid judge ratings in {path}")
    scores = data.groupby(["concept_id", "factor"], as_index=False)[column].mean()
    panel = set(map(tuple, data[["concept_id", "source_input_id"]].drop_duplicates().to_numpy()))
    return paired_concept_effects(scores, factor, baseline_factor, column), panel


def load_study_concept_scores(metrics, config, study_root, main_roots):
    """Read raw judge samples so legacy 20-prompt efficiency runs also align."""
    result_id = config["study"]["result"]["evaluator"]
    selections = metrics[metrics.source_evaluator.eq(result_id)].copy()
    records, main_panels = [], {}
    for method, group in selections.groupby("method", sort=False):
        reference = group[group.factor_reference.fillna(False)]
        if len(reference) != 1:
            raise ValueError(f"Expected one full-data factor per method: {method}")
        factor = float(reference.iloc[0].factor)
        if not np.allclose(group.factor.astype(float), factor):
            raise ValueError(f"Study factors differ from full-data factor: {method}")
        for row in group.itertuples(index=False):
            is_reference = bool(row.factor_reference)
            root = Path(study_root) / "runs" / row.study_run_id
            stem = Path(main_roots[method]).parents[1].name
            method_root = root / stem / "evaluate/runs"
            if not method_root.is_dir():
                method_root = root / "evaluate/runs"
            if is_reference:
                if not (config["study"].get("sample_efficiency") or {}):
                    continue
                if method_root.is_dir():
                    path = complete_sample_file(method_root, "study_lm_judge")
                else:
                    path = complete_sample_file(main_roots[method], "id_lm_judge")
                count = EFFICIENCY_PROMPTS
            else:
                path = complete_sample_file(method_root, "study_lm_judge")
                count = SENSITIVITY_PROMPTS if bool(row.sensitivity) else EFFICIENCY_PROMPTS
            scores, panel = _sample_scores(path, method, factor, float(row.baseline_factor), count)
            if is_reference:
                main_panels[method] = panel
            for key in ("study_run_id", "train_examples", "subset_seed", "method", "efficiency", "sensitivity", "factor_reference"):
                scores[key] = getattr(row, key)
            records.append(scores)
            if bool(row.efficiency) and not is_reference:
                scores.attrs["prompt_panel"] = panel
    for scores in records:
        panel = scores.attrs.get("prompt_panel")
        if panel is not None:
            method = scores.method.iloc[0]
            concepts = set(scores.concept_id)
            expected = {pair for pair in main_panels[method] if pair[0] in concepts}
            if expected != panel:
                raise ValueError(f"Efficiency prompt IDs differ from full-data reference: {method}")
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()
