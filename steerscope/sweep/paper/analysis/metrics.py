"""Scheduler result analysis used by scheduler_metrics_analysis.ipynb.

Call configure() before analysis. Each public analysis runs independently for
every configured output directory. Publication exports intentionally write PDFs.
"""


from __future__ import annotations


import hashlib


from functools import wraps


import json


import math


import re


import warnings


from pathlib import Path


import matplotlib.pyplot as plt


from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


import numpy as np


import pandas as pd


from IPython.display import display
from .display_names import METHOD_DISPLAY_NAMES, method_display_name, display_method_names
from steerscope.studies.statistics import paired_concept_effects, summarize_sensitivity


SAMPLE_EFFICIENCY_PROMPTS_PER_CONCEPT = 10
SAMPLE_SENSITIVITY_PROMPTS_PER_CONCEPT = 20


ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]


def _display_table(value, *, all_columns=False):
    formatted = display_method_names(value)
    if all_columns:
        with pd.option_context(
            "display.max_columns", None,
            "display.width", None,
            "display.expand_frame_repr", False,
        ):
            display(formatted)
        return
    display(formatted)


OUTPUT_DIR_VALUES = [
    REPO_ROOT / "outputs/paper/2b/l20",
    REPO_ROOT / "outputs/paper/9b/l20",
]


METHODS = None


CONCEPT_SCORE_EVALUATOR = "id_lm_judge"


CONCEPT_SCORE_COLUMN = "relevance_concept_ratings"


PAPER_RELEVANCE_THRESHOLD = 0.2


GENERALIZATION_MIN_ID_EFFECT = 0.1


MIN_REFERENCE_IMPROVEMENT = 0.1


PAPER_EXCLUDED_METHODS = set()


FULL_TRAIN_EXAMPLES = 144


TRADEOFF_BOOTSTRAP_RESAMPLES = 1000


TRADEOFF_BOOTSTRAP_SEED = 42


FACTOR_GRID_BOOTSTRAP_RESAMPLES = 5000
FACTOR_GRID_BOOTSTRAP_SEED = 42


METRIC_WEIGHTING_DRAWS = 1_000_000


METRIC_WEIGHTING_BATCH_SIZE = 100_000


METRIC_WEIGHTING_SEED = 20260913


METRIC_WEIGHTING_DIRICHLET_ALPHAS = (1.0, 5.0, 0.2)


COMPOSITE_SIDE_EFFECT_SPECS = (
    {"name": "Instruction relevance", "evaluator": "id_lm_judge",
     "metric": "relevance_instruction_ratings", "higher_is_better": True, "scale": 2.0},
    {"name": "Fluency", "evaluator": "id_lm_judge",
     "metric": "fluency_ratings", "higher_is_better": True, "scale": 2.0},
    {"name": "MMLU", "evaluator": "mmlu",
     "metric": "mmlu_accuracy", "higher_is_better": True, "scale": 1.0},
    {"name": "TruthfulQA", "evaluator": "truthfulqa",
     "metric": "truthfulqa_binary_accuracy", "higher_is_better": True, "scale": 1.0},
    {"name": "SuperGLUE", "evaluator": "superglue",
     "metric": "superglue_score", "higher_is_better": True, "scale": 1.0},
    {"name": "MATH", "evaluator": "math",
     "metric": "math_accuracy", "higher_is_better": True, "scale": 1.0},
    {"name": "IFEval strict", "evaluator": "ifeval",
     "metric": "ifeval_prompt_strict_accuracy", "higher_is_better": True, "scale": 1.0},
    {"name": "BBQ ambiguous bias", "evaluator": "bbq",
     "metric": "bbq_ambiguous_abs_bias", "higher_is_better": False, "scale": 1.0},
    {"name": "BBQ disambiguated bias", "evaluator": "bbq",
     "metric": "bbq_disambiguated_abs_bias", "higher_is_better": False, "scale": 1.0},
    {"name": "JBB attack success", "evaluator": "jailbreakbench",
     "metric": "attack_success_rate", "higher_is_better": False, "scale": 1.0,
     "filters": {"jbb_split": "harmful"}},
    {"name": "JBB false refusal", "evaluator": "jailbreakbench",
     "metric": "false_refusal_rate", "higher_is_better": False, "scale": 1.0,
     "filters": {"jbb_split": "benign"}},
)


PAPER_FIGURE_BASE = ANALYSIS_DIR.parent / "paper_figures"


PAPER_FIGURE_ROOT = PAPER_FIGURE_BASE


PAPER_MAIN_DIR = PAPER_FIGURE_ROOT / "main"


PAPER_APPENDIX_DIR = PAPER_FIGURE_ROOT / "appendix"


PAPER_MAIN_SIZE = (6.5, 3.65)


PAPER_GRID_COLUMNS = 6


PAPER_GRID_ROW_HEIGHT = 1.08


DISPLAY_FIGURES_IN_NOTEBOOK = False




PLOT_DPI = 160


def resolve_output_dir(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd = Path.cwd().resolve()
    candidates = [
        (REPO_ROOT / path).resolve(),
        *((base / path).resolve() for base in (cwd, *cwd.parents)),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _output_key(path: Path) -> str:
    """Return a stable, filesystem-safe label such as `2b/l20`."""
    parts = path.parts
    if "paper" in parts:
        index = len(parts) - 1 - tuple(reversed(parts)).index("paper")
        relative = parts[index + 1:]
    else:
        relative = parts[-2:]
    return "/".join(relative)


def _activate_output_dir(value: str | Path) -> str:
    """Switch every data/cache/figure global to one scheduler output."""
    global OUTPUT_DIR, OUTPUT_KEY, REPORT_DIR
    global PAPER_FIGURE_ROOT, PAPER_MAIN_DIR, PAPER_APPENDIX_DIR
    global _PAPER_BEST_FACTOR_CACHE, _PAPER_EFFECTIVENESS_CACHE
    global _PAPER_COMPOSITE_RESULT
    OUTPUT_DIR = resolve_output_dir(value)
    OUTPUT_KEY = _output_key(OUTPUT_DIR)
    REPORT_DIR = OUTPUT_DIR / "notebook_reports"
    PAPER_FIGURE_ROOT = PAPER_FIGURE_BASE / Path(OUTPUT_KEY)
    PAPER_MAIN_DIR = PAPER_FIGURE_ROOT / "main"
    PAPER_APPENDIX_DIR = PAPER_FIGURE_ROOT / "appendix"
    _PAPER_BEST_FACTOR_CACHE = None
    _PAPER_EFFECTIVENESS_CACHE = None
    _PAPER_COMPOSITE_RESULT = None
    return OUTPUT_KEY


_METHOD_ORDER = (
    "APSR", "AUSteer", "DiffMean", "FLAS",
    "GemmaScopeSAE", "GemmaScopeSAEMaxAUC", "HiDRA", "HyperSteer",
    "LAT", "LinearProbe", "LoRA", "LoReFT", "LsReFT",
    "ODESteer", "PCA", "PreferenceVector", "PromptSteering",
    "Random", "SFT", "SPSR", "SimplePromptSteering",
    "SphericalSteering", "SteeringVector", "StepODESteer",
)


_METHOD_INDEX = {method: index for index, method in enumerate(_METHOD_ORDER)}


_METHOD_COLORS = (
    "#4477AA", "#EE6677", "#228833", "#CCBB44",
    "#66CCEE", "#AA3377", "#BBBBBB", "#332288",
    "#44AA99", "#117733", "#999933", "#CC6677",
    "#882255", "#6699CC", "#DDCC77", "#AA4499",
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#6F4E7C", "#5F6B6D",
)


_METHOD_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p")


_METHOD_LINESTYLES = ("-", "--", "-.", ":")


def _method_style_index(method: str) -> int:
    method = str(method)
    if method in _METHOD_INDEX:
        return _METHOD_INDEX[method]
    # Keep unknown future methods deterministic and outside the reserved block.
    digest = hashlib.sha256(method.encode("utf-8")).digest()
    return len(_METHOD_ORDER) + int.from_bytes(digest[:4], "big")


def _method_style(method: str) -> dict:
    index = _method_style_index(method)
    return {
        "color": _METHOD_COLORS[index % len(_METHOD_COLORS)],
        "marker": _METHOD_MARKERS[index % len(_METHOD_MARKERS)],
        "linestyle": _METHOD_LINESTYLES[(index // 6) % len(_METHOD_LINESTYLES)],
    }


def _method_color(method: str):
    return _method_style(method)["color"]


def _method_marker_size(method: str) -> float:
    """Compensate markers whose visible ink is small at equal point size."""
    marker = _method_style(method)["marker"]
    return {
        "*": 11.0,
        "X": 7.8,
        "P": 7.6,
        "h": 7.0,
        "p": 6.8,
        "<": 6.7,
        ">": 6.7,
    }.get(marker, 6.3)


def _method_scatter_size(method: str) -> float:
    marker = _method_style(method)["marker"]
    return {
        "*": 220.0,
        "X": 125.0,
        "P": 120.0,
        "h": 110.0,
        "p": 105.0,
        "<": 105.0,
        ">": 105.0,
    }.get(marker, 96.0)


def _method_line_kwargs(method: str, *, label: bool = True) -> dict:
    style = _method_style(method)
    return {
        **style,
        "linewidth": 1.35,
        "markersize": 0.85 * _method_marker_size(method),
        "markeredgecolor": "white",
        "markeredgewidth": 0.55,
        **({"label": method_display_name(method)} if label else {}),
    }


def _method_legend_handle(method: str) -> Line2D:
    return Line2D([], [], **_method_line_kwargs(method, label=False), label=method_display_name(method))


def _add_method_legend(figure, methods) -> None:
    methods = sorted({str(method) for method in methods}, key=_method_style_index)
    if not methods:
        return
    figure.legend(
        handles=[_method_legend_handle(method) for method in methods],
        loc="center left", bbox_to_anchor=(0.80, 0.5),
        ncol=1, fontsize=7.5,
        title="Method", title_fontsize=8,
        frameon=True, borderaxespad=0.0, handlelength=2.8,
    )


def _selected_methods(frame: pd.DataFrame) -> pd.DataFrame:
    if METHODS is None or "method" not in frame:
        return frame
    return frame[frame["method"].astype(str).isin(set(METHODS))].copy()


def _final_metric_paths(evaluator_id: str) -> list[Path]:
    """Choose the newest finalized evaluator file in each method directory."""
    selected: dict[str, Path] = {}
    if evaluator_id == "prompt_generalization":
        current = sorted(OUTPUT_DIR.glob("generalization/methods/*/evaluate/runs/*/evaluators/prompt_generalization/metrics.parquet"))
        if current:
            return [p for p in current if json.loads(p.with_name("manifest.json").read_text()).get("status") == "complete"]
    pattern = f"methods/*/evaluate/runs/*/evaluators/{evaluator_id}/metrics.parquet"
    for path in OUTPUT_DIR.glob(pattern):
        try:
            stem = path.relative_to(OUTPUT_DIR).parts[1]
        except (ValueError, IndexError):
            continue
        prior = selected.get(stem)
        if prior is None or path.stat().st_mtime_ns > prior.stat().st_mtime_ns:
            selected[stem] = path
    return [selected[key] for key in sorted(selected)]


def _normalize_shared_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach a shared DiffMean factor-0 row to its owning method curve."""
    if not {"method", "factor"}.issubset(frame.columns):
        return frame
    data = frame.copy()
    factors = pd.to_numeric(data["factor"], errors="coerce")
    non_baseline = sorted(
        set(data.loc[data["method"].astype(str) != "DiffMean", "method"].astype(str))
    )
    if len(non_baseline) != 1:
        return data
    primary = non_baseline[0]
    own = data[data["method"].astype(str) == primary].copy()
    if np.isclose(pd.to_numeric(own["factor"], errors="coerce"), 0.0).any():
        return own

    baseline = data[
        (data["method"].astype(str) == "DiffMean") & np.isclose(factors, 0.0)
    ].copy()
    if baseline.empty:
        return own
    baseline["method"] = primary
    if "target_id" in baseline:
        baseline["target_id"] = baseline["concept_id"].map(
            lambda concept_id: f"{primary}/concept-{int(concept_id)}"
        )
    return pd.concat([baseline, own], ignore_index=True, sort=False)


def _derive_bbq_bias_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Add absolute ambiguous and disambiguated BBQ bias scores."""
    required = {
        "bbq_ambiguous_bias_score", "bbq_disambiguated_bias_score",
    }
    if not required.issubset(data.columns):
        return data
    result = data.copy()
    result["bbq_ambiguous_abs_bias"] = pd.to_numeric(
        result["bbq_ambiguous_bias_score"], errors="coerce"
    ).abs()
    result["bbq_disambiguated_abs_bias"] = pd.to_numeric(
        result["bbq_disambiguated_bias_score"], errors="coerce"
    ).abs()
    return result


def load_metric(evaluator_id: str) -> pd.DataFrame:
    """Load finalized main-method metrics; progress and SFT shard files are excluded."""
    frames = []
    for path in _final_metric_paths(evaluator_id):
        try:
            frame = pd.read_parquet(path)
        except Exception as error:
            warnings.warn(f"Skipping unreadable parquet {path}: {error}")
            continue
        frame = _normalize_shared_baseline(frame)
        frame["_source_path"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True, sort=False)
    if evaluator_id == "bbq":
        data = _derive_bbq_bias_metrics(data)
    data = _selected_methods(data)
    for column in ("concept_id", "factor", "model_factor"):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    identity = [
        column
        for column in ("method", "concept_id", "factor", "scope", "jbb_split")
        if column in data
    ]
    if identity:
        data = data.drop_duplicates(identity, keep="last")
    return data.reset_index(drop=True)


def _with_normalized_factor(data: pd.DataFrame) -> pd.DataFrame:
    """Add a per-method normalized factor axis while retaining raw factors as join keys."""
    result = data.copy()
    result["factor"] = pd.to_numeric(result["factor"], errors="coerce")

    fallback = (
        result.groupby("method", observed=True)["factor"]
        .apply(lambda values: values.abs().max())
        .to_dict()
    )
    scale_map = {}
    try:
        reference = load_metric(CONCEPT_SCORE_EVALUATOR)
        if not reference.empty:
            scale_map = (
                reference.groupby("method", observed=True)["factor"]
                .apply(
                    lambda values: pd.to_numeric(values, errors="coerce").abs().max()
                )
                .to_dict()
            )
    except (FileNotFoundError, KeyError, ValueError):
        pass

    scales = pd.to_numeric(result["method"].map(scale_map), errors="coerce")
    fallback_scales = pd.to_numeric(
        result["method"].map(fallback), errors="coerce"
    )
    result["factor_scale"] = scales.fillna(fallback_scales)
    valid_scale = result["factor_scale"].notna() & result["factor_scale"].gt(0)
    result["normalized_factor"] = np.where(
        valid_scale,
        result["factor"] / result["factor_scale"],
        0.0,
    )
    return result


def _apply_filters(data: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    result = data
    for column, expected in (filters or {}).items():
        if column not in result:
            return result.iloc[0:0].copy()
        if callable(expected):
            result = result[expected(result[column])]
        elif isinstance(expected, (set, list, tuple)):
            result = result[result[column].isin(expected)]
        else:
            result = result[result[column] == expected]
    return result.copy()


def _aggregate_curve(data: pd.DataFrame, x: str, metric: str) -> pd.DataFrame:
    selected = data[["method", x, metric, "concept_id"]].copy()
    selected[x] = pd.to_numeric(selected[x], errors="coerce")
    selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
    selected = selected.dropna(subset=[x, metric])
    if selected.empty:
        return pd.DataFrame()

    # Give every concept equal weight at each x value, even if an evaluator
    # emitted duplicate rows for that (method, concept, x) identity.
    concept_values = (
        selected.groupby(
            ["method", x, "concept_id"],
            as_index=False, dropna=False, observed=True,
        )[metric]
        .mean()
    )

    summary = (
        concept_values.groupby(
            ["method", x], as_index=False, dropna=False, observed=True
        )
        .agg(
            metric_mean=(metric, "mean"),
            metric_std=(metric, "std"),
            n_rows=(metric, "size"),
            n_concepts=("concept_id", "nunique"),
        )
        .sort_values(["method", x])
    )
    summary["metric_std"] = summary["metric_std"].fillna(0.0)
    summary["metric_sem"] = summary["metric_std"] / np.sqrt(
        summary["n_concepts"].clip(lower=1)
    )
    summary["ci95"] = 1.96 * summary["metric_sem"]
    return summary


def _concept_relation(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    keys = ["method", "concept_id", "factor"]
    if not set(keys).issubset(data.columns):
        return pd.DataFrame()
    if CONCEPT_SCORE_COLUMN in data.columns:
        relation = data.copy()
        relation["concept_score"] = pd.to_numeric(
            relation[CONCEPT_SCORE_COLUMN], errors="coerce"
        )
        return relation.dropna(subset=["concept_score", metric])

    scores = load_metric(CONCEPT_SCORE_EVALUATOR)
    if scores.empty or CONCEPT_SCORE_COLUMN not in scores:
        return pd.DataFrame()
    score_rows = scores[keys + [CONCEPT_SCORE_COLUMN]].copy()
    score_rows[CONCEPT_SCORE_COLUMN] = pd.to_numeric(
        score_rows[CONCEPT_SCORE_COLUMN], errors="coerce"
    )
    score_rows = (
        score_rows.dropna(subset=[CONCEPT_SCORE_COLUMN])
        .groupby(keys, as_index=False)[CONCEPT_SCORE_COLUMN]
        .mean()
        .rename(columns={CONCEPT_SCORE_COLUMN: "concept_score"})
    )
    relation = data.merge(score_rows, on=keys, how="inner", validate="many_to_one")
    relation[metric] = pd.to_numeric(relation[metric], errors="coerce")
    return relation.dropna(subset=["concept_score", metric])


def _effect_side_effect_rows(
    data: pd.DataFrame, metric: str, *, higher_is_better: bool
) -> pd.DataFrame:
    """Build factor-zero paired deltas on each method's complete concept panel."""
    relation = _concept_relation(data, metric)
    required = {"method", "concept_id", "factor", "concept_score", metric}
    if relation.empty or not required.issubset(relation.columns):
        return pd.DataFrame()

    paired = relation[list(required)].copy()
    for column in ("concept_id", "factor", "concept_score", metric):
        paired[column] = pd.to_numeric(paired[column], errors="coerce")
    paired = paired.dropna(subset=list(required))
    paired = (
        paired.groupby(
            ["method", "concept_id", "factor"],
            as_index=False, observed=True,
        )
        .agg(concept_score=("concept_score", "mean"), metric_value=(metric, "mean"))
    )
    if paired.empty:
        return paired

    method_factor_counts = paired.groupby("method", observed=True)["factor"].transform("nunique")
    concept_factor_counts = paired.groupby(
        ["method", "concept_id"], observed=True
    )["factor"].transform("nunique")
    paired = paired[concept_factor_counts.eq(method_factor_counts)].copy()
    if paired.empty:
        return paired

    baseline = paired[np.isclose(paired["factor"], 0.0)].copy()
    baseline = baseline[["method", "concept_id", "concept_score", "metric_value"]]
    baseline = baseline.rename(columns={
        "concept_score": "baseline_concept_score",
        "metric_value": "baseline_metric_value",
    })
    if baseline.duplicated(["method", "concept_id"]).any():
        raise ValueError("Trade-off analysis found duplicate factor-0 baselines.")
    paired = paired.merge(
        baseline, on=["method", "concept_id"], how="inner", validate="many_to_one"
    )
    paired["effect_delta"] = (
        paired["concept_score"] - paired["baseline_concept_score"]
    )
    if higher_is_better:
        paired["side_effect_delta"] = (
            paired["baseline_metric_value"] - paired["metric_value"]
        )
    else:
        paired["side_effect_delta"] = (
            paired["metric_value"] - paired["baseline_metric_value"]
        )
    paired[metric] = paired["metric_value"]
    paired["common_concept_count"] = paired.groupby("method", observed=True)["concept_id"].transform("nunique")
    return _with_normalized_factor(paired)


def _aggregate_effect_side_effect(paired: pd.DataFrame) -> pd.DataFrame:
    """Aggregate method-factor operating points with paired concept bootstrap CIs."""
    if paired.empty:
        return pd.DataFrame()
    rows = []
    bootstrap_cache: dict[tuple[int, ...], np.ndarray] = {}
    for (method, factor), group in paired.groupby(
        ["method", "factor"], sort=True, observed=True
    ):
        group = group.sort_values("concept_id")
        concept_ids = tuple(group["concept_id"].astype(int))
        if concept_ids not in bootstrap_cache:
            rng = np.random.default_rng(TRADEOFF_BOOTSTRAP_SEED)
            bootstrap_cache[concept_ids] = rng.integers(
                0, len(concept_ids),
                size=(TRADEOFF_BOOTSTRAP_RESAMPLES, len(concept_ids)),
            )
        indices = bootstrap_cache[concept_ids]
        effect = group["effect_delta"].to_numpy(dtype=float)
        side = group["side_effect_delta"].to_numpy(dtype=float)
        effect_bootstrap = effect[indices].mean(axis=1)
        side_bootstrap = side[indices].mean(axis=1)
        effect_lower, effect_upper = np.quantile(effect_bootstrap, [0.025, 0.975])
        side_lower, side_upper = np.quantile(side_bootstrap, [0.025, 0.975])
        rows.append({
            "method": method,
            "factor": float(factor),
            "normalized_factor": float(group["normalized_factor"].iloc[0]),
            "effect_mean": float(effect.mean()),
            "effect_ci_lower": float(effect_lower),
            "effect_ci_upper": float(effect_upper),
            "side_effect_mean": float(side.mean()),
            "side_effect_ci_lower": float(side_lower),
            "side_effect_ci_upper": float(side_upper),
            "n_concepts": len(concept_ids),
        })
    return pd.DataFrame(rows).sort_values(["method", "factor"]).reset_index(drop=True)


def _plot_factor_tradeoff_metric(
    factor_summary: pd.DataFrame,
    tradeoff_summary: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
    slug: str,
    show_tradeoff: bool,
):
    if show_tradeoff:
        figure, axes = plt.subplots(1, 2, figsize=(17, 6), dpi=PLOT_DPI)
        factor_axis, tradeoff_axis = axes
    else:
        figure, factor_axis = plt.subplots(1, 1, figsize=(9, 6), dpi=PLOT_DPI)
        tradeoff_axis = None

    if factor_summary.empty:
        factor_axis.text(0.5, 0.5, "No completed factor data", ha="center", va="center")
    else:
        for method, group in factor_summary.groupby("method", sort=True):
            group = group.sort_values("normalized_factor")
            style = _method_style(method)
            factor_axis.plot(
                group["normalized_factor"], group["metric_mean"],
                **_method_line_kwargs(method),
            )
            factor_axis.fill_between(
                group["normalized_factor"],
                group["metric_mean"] - group["ci95"],
                group["metric_mean"] + group["ci95"],
                color=style["color"], alpha=0.13,
            )
    factor_axis.set_title(
        "Metric vs normalized steering factor\n(shading: 95% CI across concepts)"
    )
    factor_axis.set_xlabel("normalized factor = raw / method max |factor|")
    factor_axis.set_ylabel(ylabel)
    factor_axis.grid(alpha=0.25)

    if tradeoff_axis is not None:
        if tradeoff_summary.empty:
            tradeoff_axis.text(
                0.5, 0.5,
                "No complete paired method-factor data",
                ha="center", va="center",
            )
        else:
            for method, group in tradeoff_summary.groupby("method", sort=True):
                group = group.sort_values("factor")
                style = _method_style(method)
                x = group["effect_mean"].to_numpy()
                y = group["side_effect_mean"].to_numpy()
                xerr = np.vstack([
                    x - group["effect_ci_lower"].to_numpy(),
                    group["effect_ci_upper"].to_numpy() - x,
                ])
                yerr = np.vstack([
                    y - group["side_effect_ci_lower"].to_numpy(),
                    group["side_effect_ci_upper"].to_numpy() - y,
                ])
                tradeoff_axis.errorbar(
                    x, y, xerr=xerr, yerr=yerr, fmt="none",
                    ecolor=style["color"], elinewidth=0.8, alpha=0.22, capsize=2,
                )
                tradeoff_axis.plot(
                    x, y, **_method_line_kwargs(method),
                )
        tradeoff_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
        tradeoff_axis.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
        tradeoff_axis.set_title(
            "Effect–side-effect operating points\n"
            f"one point per (method, factor); {TRADEOFF_BOOTSTRAP_RESAMPLES}x paired concept bootstrap CI"
        )
        tradeoff_axis.set_xlabel(
            "Mean Concept Expression Delta vs. Factor 0 (Higher Is Better)"
        )
        tradeoff_axis.set_ylabel(
            f"{metric} side-effect delta vs factor 0 (positive = worse)"
        )
        tradeoff_axis.grid(alpha=0.25)

    legend_methods = set(factor_summary.get("method", []))
    legend_methods.update(tradeoff_summary.get("method", []))
    _add_method_legend(figure, legend_methods)
    figure.suptitle(title, fontsize=14, y=1.02)
    _save_or_show(figure, slug)


def analyze_global_best_factors():
    """Report one main-sweep Overall-best factor for every method."""
    selected = _paper_best_factors()
    if selected.empty:
        print(f"[waiting] No finalized {CONCEPT_SCORE_EVALUATOR} factor sweep.")
        return None
    table = selected[[
        "method", "factor", "overall_score", "baseline_score",
        "overall_improvement", "n_concepts", "n_complete_factors",
    ]].copy()
    table["_method_order"] = table["method"].astype(str).map(
        lambda method: _METHOD_INDEX.get(method, len(_METHOD_INDEX))
    )
    table = table.sort_values(["_method_order", "method"]).drop(
        columns="_method_order"
    )
    print(
        "Global Overall-best factors — one fixed factor per method; selected "
        "by mean main-sweep Overall score across the complete concept panel."
    )
    _display_table(table.style.format({
        "factor": "{:.4g}",
        "overall_score": "{:.4f}",
        "baseline_score": "{:.4f}",
        "overall_improvement": "{:+.4f}",
    }).hide(axis="index"))
    return {"selected_factors": selected, "table": table}

def _composite_side_effect_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build normalized, factor-0-paired rows for every top-level indicator."""
    frames = []
    diagnostics = []
    for spec in COMPOSITE_SIDE_EFFECT_SPECS:
        data = load_metric(spec["evaluator"])
        data = _apply_filters(data, spec.get("filters"))
        metric = spec["metric"]
        if data.empty or metric not in data:
            raise ValueError(
                f"Composite side effect is missing {spec['evaluator']}/{metric}."
            )
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
        data = data.dropna(subset=["method", "concept_id", "factor", metric])
        paired = _effect_side_effect_rows(
            data, metric, higher_is_better=spec["higher_is_better"]
        )
        if paired.empty:
            raise ValueError(
                f"Composite side effect could not pair {spec['evaluator']}/{metric}."
            )
        selected = paired[[
            "method", "concept_id", "factor", "side_effect_delta"
        ]].copy()
        selected["side_effect_name"] = spec["name"]
        selected["normalized_side_effect_delta"] = (
            selected["side_effect_delta"] / float(spec["scale"])
        )
        frames.append(selected)
        diagnostics.append({
            "side_effect": spec["name"],
            "evaluator": spec["evaluator"],
            "metric": metric,
            "scale": float(spec["scale"]),
            "n_rows": len(selected),
            "n_methods": selected["method"].nunique(),
            "n_concepts": selected["concept_id"].nunique(),
        })

    rows = pd.concat(frames, ignore_index=True, sort=False)
    diagnostics = pd.DataFrame(diagnostics)
    return rows, diagnostics


def _composite_side_effect_factor_means(rows: pd.DataFrame) -> pd.DataFrame:
    """Give every side-effect indicator equal weight at each method/factor."""
    indicator_means = (
        rows.groupby(
            ["method", "factor", "side_effect_name"],
            as_index=False, observed=True,
        )
        .agg(
            indicator_delta=("normalized_side_effect_delta", "mean"),
            indicator_concepts=("concept_id", "nunique"),
        )
    )
    composite = (
        indicator_means.groupby(
            ["method", "factor"], as_index=False, observed=True
        )
        .agg(
            mean_side_effect=("indicator_delta", "mean"),
            n_side_effects=("side_effect_name", "nunique"),
            min_concepts_per_side_effect=("indicator_concepts", "min"),
            max_concepts_per_side_effect=("indicator_concepts", "max"),
        )
    )
    expected = len(COMPOSITE_SIDE_EFFECT_SPECS)
    incomplete = composite[composite["n_side_effects"] != expected]
    if not incomplete.empty:
        examples = incomplete[["method", "factor", "n_side_effects"]].head()
        raise ValueError(
            f"Composite requires all {expected} side effects at every point; examples:\n"
            f"{examples.to_string(index=False)}"
        )
    return composite


def load_study_runs() -> pd.DataFrame:
    """Load canonical aggregate study data, or aggregate completed worker runs."""
    canonical = OUTPUT_DIR / "studies/metrics.parquet"
    if canonical.is_file():
        metrics = pd.read_parquet(canonical)
        if "source_evaluator" in metrics:
            metrics = metrics[metrics["source_evaluator"].isin(["best_factor", "study_result"])].copy()
        if "selected_improvement" not in metrics:
            return pd.DataFrame()
        return _selected_methods(metrics)

    frames = []
    jobs = sorted(
        OUTPUT_DIR.glob(
            "runtime_configs/integrated/workers/study/*/*.json"
        )
    )
    stem_to_method = {}
    for job_path in jobs:
        try:
            payload = json.loads(job_path.read_text(encoding="utf-8"))
        except Exception as error:
            warnings.warn(f"Skipping invalid study job {job_path}: {error}")
            continue
        variant = payload.get("variant") or {}
        method = str(payload.get("method") or job_path.stem)
        stem = job_path.stem
        stem_to_method[stem] = method
        run_id = (
            f"n-{int(variant.get('train_examples', 0)):04d}_"
            f"subset-{int(variant.get('subset_seed', 0))}"
        )
        root = OUTPUT_DIR / "studies/runs" / run_id / stem / "evaluate/runs"
        paths = sorted(
            root.glob("*/evaluators/best_factor/metrics.parquet"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if not paths:
            continue
        frame = pd.read_parquet(paths[-1])
        frame = frame[frame["method"].astype(str) == method].copy()
        if frame.empty or "selected_improvement" not in frame:
            continue
        frame["study_run_id"] = run_id
        frame["train_examples"] = int(variant["train_examples"])
        frame["subset_seed"] = int(variant["subset_seed"])
        frame["efficiency"] = bool(variant.get("efficiency", False))
        frame["sensitivity"] = bool(variant.get("sensitivity", False))
        frame["factor_reference"] = bool(variant.get("factor_reference", False))
        frames.append(frame)

    # Add each method's full-data reference produced by the main run.
    for stem, method in sorted(stem_to_method.items()):
        paths = sorted(
            (OUTPUT_DIR / "methods" / stem / "evaluate/runs").glob(
                "*/evaluators/study_reference_best_factor/metrics.parquet"
            ),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if not paths:
            continue
        frame = pd.read_parquet(paths[-1])
        frame = frame[frame["method"].astype(str) == method].copy()
        if frame.empty or "selected_improvement" not in frame:
            continue
        frame["study_run_id"] = f"n-{FULL_TRAIN_EXAMPLES:04d}_subset-42"
        frame["train_examples"] = FULL_TRAIN_EXAMPLES
        frame["subset_seed"] = 42
        frame["efficiency"] = True
        frame["sensitivity"] = False
        frame["factor_reference"] = True
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True, sort=False)
    data["selected_improvement"] = pd.to_numeric(
        data["selected_improvement"], errors="coerce"
    )
    data = data.dropna(subset=["selected_improvement"])
    data = _selected_methods(data)
    return data.drop_duplicates(
        ["study_run_id", "method"], keep="last"
    ).reset_index(drop=True)


_ALIGNED_STUDY_JUDGE_CACHE = {}


def _latest_complete_samples(paths, *, label):
    """Select the newest completed evaluator sample file."""
    completed = []
    for path in paths:
        manifest = path.with_name("manifest.json")
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if payload.get("status") == "complete":
            completed.append(path)
    if not completed:
        raise ValueError(f"Missing completed evaluator samples: {label}")
    if len(completed) > 1:
        warnings.warn(
            f"Multiple completed evaluator sample files for {label}; using "
            "the newest one.",
            stacklevel=2,
        )
    return max(completed, key=lambda path: path.stat().st_mtime_ns)


def _aligned_study_judge_metrics(runs, main_sources):
    """Reaggregate efficiency scores on the main run's 10-prompt panel."""
    non_reference = runs[~runs["factor_reference"].fillna(False)].copy()
    requested = tuple(sorted(
        (str(row.method), str(row.study_run_id))
        for row in non_reference.drop_duplicates(
            ["method", "study_run_id"]
        ).itertuples(index=False)
    ))
    cache_key = (str(OUTPUT_DIR), requested)
    cached = _ALIGNED_STUDY_JUDGE_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    raw_columns = {
        "raw_relevance_concept_ratings": "relevance_concept_ratings",
        "raw_relevance_instruction_ratings": "relevance_instruction_ratings",
        "raw_fluency_ratings": "fluency_ratings",
        "raw_aggregated_ratings": "lm_judge_rating",
    }
    main_configs = {}
    main_panels = {}
    for method in sorted({method for method, _ in requested}):
        source = Path(main_sources[method])
        payload = json.loads(
            source.with_name("manifest.json").read_text(encoding="utf-8")
        )
        config = payload.get("config", {}).get("dataset", {})
        if int(config.get("num_examples", -1)) != SAMPLE_EFFICIENCY_PROMPTS_PER_CONCEPT:
            raise ValueError(
                f"Main evaluator is not configured for 10 prompts: {method}"
            )
        main_configs[method] = (
            str(config.get("type")), int(config.get("seed", 42))
        )
        samples = source.with_name("samples.parquet")
        if samples.is_file():
            panel = pd.read_parquet(
                samples, columns=["concept_id", "source_input_id"]
            ).drop_duplicates()
            counts = panel.groupby("concept_id")["source_input_id"].nunique()
            if counts.empty or not counts.eq(
                SAMPLE_EFFICIENCY_PROMPTS_PER_CONCEPT
            ).all():
                observed = sorted(counts.unique().tolist())
                raise ValueError(
                    f"Main prompt panel is not uniformly 10 examples/concept: "
                    f"{method}, observed={observed}"
                )
            main_panels[method] = panel

    frames = []
    for row in non_reference.drop_duplicates(
        ["method", "study_run_id"]
    ).itertuples(index=False):
        method, run_id = str(row.method), str(row.study_run_id)
        source = Path(main_sources[method])
        stem = source.relative_to(OUTPUT_DIR).parts[1]
        root = (
            OUTPUT_DIR / "studies/runs" / run_id / stem / "evaluate/runs"
        )
        samples = _latest_complete_samples(
            root.glob("*/evaluators/study_lm_judge/samples.parquet"),
            label=f"{method}/{run_id}",
        )
        study_payload = json.loads(
            samples.with_name("manifest.json").read_text(encoding="utf-8")
        )
        study_config = study_payload.get("config", {}).get("dataset", {})
        study_identity = (
            str(study_config.get("type")), int(study_config.get("seed", 42))
        )
        if study_identity != main_configs[method]:
            raise ValueError(
                f"Main/study prompt sampling differs: {method}/{run_id}; "
                f"main={main_configs[method]}, study={study_identity}"
            )
        if int(study_config.get("num_examples", -1)) < SAMPLE_EFFICIENCY_PROMPTS_PER_CONCEPT:
            raise ValueError(f"Study panel has fewer than 10 prompts: {method}/{run_id}")

        columns = [
            "method", "concept_id", "factor", "input_id", "source_input_id",
            *raw_columns,
        ]
        data = _normalize_shared_baseline(
            pd.read_parquet(samples, columns=columns)
        )
        data = data[data["method"].astype(str).eq(method)].copy()
        data["input_id"] = pd.to_numeric(data["input_id"], errors="raise")
        data = data[
            data["input_id"].ge(0)
            & data["input_id"].lt(SAMPLE_EFFICIENCY_PROMPTS_PER_CONCEPT)
        ].copy()

        actual_panel = data[["concept_id", "source_input_id"]].drop_duplicates()
        if method in main_panels:
            concepts = set(actual_panel["concept_id"])
            expected_panel = main_panels[method][
                main_panels[method]["concept_id"].isin(concepts)
            ].drop_duplicates()
            actual_pairs = set(map(tuple, actual_panel.to_numpy()))
            expected_pairs = set(map(tuple, expected_panel.to_numpy()))
            if actual_pairs != expected_pairs:
                raise ValueError(
                    f"Study prefix does not match main prompt IDs: {method}/{run_id}"
                )

        identity = ["concept_id", "factor", "source_input_id"]
        if data.duplicated(identity).any():
            raise ValueError(f"Duplicate aligned judge samples: {method}/{run_id}")
        counts = data.groupby(["concept_id", "factor"])[
            "source_input_id"
        ].nunique()
        if counts.empty or not counts.eq(
            SAMPLE_EFFICIENCY_PROMPTS_PER_CONCEPT
        ).all():
            observed = sorted(counts.unique().tolist())
            raise ValueError(
                f"Incomplete 10-prompt study panel: {method}/{run_id}, "
                f"observed={observed}"
            )
        for column in raw_columns:
            data[column] = pd.to_numeric(data[column], errors="raise")
        grouped = (
            data.groupby(["method", "concept_id", "factor"], as_index=False)
            [list(raw_columns)].mean()
            .rename(columns=raw_columns)
        )
        grouped["study_run_id"] = run_id
        frames.append(grouped)

    result = pd.concat(frames, ignore_index=True, sort=False)
    _ALIGNED_STUDY_JUDGE_CACHE[cache_key] = result.copy()
    return result


def _current_study_concept_improvements(
    runs, metrics, *, align_main_prompt_panel=False,
):
    """Pair C with C0; optionally align variants to the main 10 prompts."""
    main = load_metric(CONCEPT_SCORE_EVALUATOR)
    main_sources, source_issues = resolve_unique_main_sources(
        runs["method"], main
    )
    for issue in source_issues:
        method = issue["method"]
        if issue["status"] == "missing":
            detail = (
                f"missing main results for {method}: expected exactly one "
                f"completed {CONCEPT_SCORE_EVALUATOR} source, found 0"
            )
        else:
            detail = (
                f"ambiguous main results for {method}: expected exactly one "
                f"completed {CONCEPT_SCORE_EVALUATOR} source, found "
                f"{issue['source_count']}: {issue['sources']}"
            )
        warnings.warn(
            f"Skipping study method; {detail}.", stacklevel=2
        )
    runs = runs[runs["method"].astype(str).isin(main_sources)].copy()
    if runs.empty:
        warnings.warn(
            "No study method has a unique completed main result; study plots "
            "will be empty.",
            stacklevel=2,
        )
        return pd.DataFrame()

    if align_main_prompt_panel:
        judge = _aligned_study_judge_metrics(runs, main_sources)
        prompt_count = SAMPLE_EFFICIENCY_PROMPTS_PER_CONCEPT
    else:
        judge = metrics[
            metrics["source_evaluator"] == "study_lm_judge"
        ].copy()
        prompt_count = SAMPLE_SENSITIVITY_PROMPTS_PER_CONCEPT
    paired_by_key, panels, sizes = {}, {}, {}
    for row in runs.itertuples(index=False):
        method, run_id = str(row.method), str(row.study_run_id)
        data = (main[main["method"].astype(str) == method] if bool(row.factor_reference)
                else judge[(judge["method"].astype(str) == method)
                           & (judge["study_run_id"].astype(str) == run_id)]).copy()
        if data.empty:
            raise ValueError(f"Missing completed concept scores: {method}/{run_id}")
        data["factor"] = pd.to_numeric(data["factor"], errors="raise")
        for column in (CONCEPT_SCORE_COLUMN, "lm_judge_rating"):
            data[column] = pd.to_numeric(data[column], errors="raise")
        def values(factor):
            return (data[np.isclose(data["factor"], float(factor))]
                    .groupby("concept_id")[[CONCEPT_SCORE_COLUMN, "lm_judge_rating"]].mean())
        selected, baseline = values(row.factor), values(row.baseline_factor)
        if selected.empty or baseline.empty or set(selected.index) != set(baseline.index):
            raise ValueError(f"Unpaired selected/baseline concepts: {method}/{run_id}")
        paired = pd.DataFrame({
            "selected": selected[CONCEPT_SCORE_COLUMN],
            "baseline": baseline[CONCEPT_SCORE_COLUMN],
            "selected_overall": selected["lm_judge_rating"],
            "baseline_overall": baseline["lm_judge_rating"],
        })
        if paired.isna().any().any():
            raise ValueError(f"Missing concept ratings: {method}/{run_id}")
        paired_by_key[(method, run_id)] = paired
        if not bool(row.factor_reference):
            panels.setdefault(method, []).append(set(paired.index))
        sizes[(method, run_id)] = (
            SAMPLE_EFFICIENCY_PROMPTS_PER_CONCEPT
            if bool(row.factor_reference) else prompt_count
        )
    records = []
    for method, group in runs.groupby("method", sort=False):
        reference = group[group["factor_reference"].fillna(False)]
        if len(reference) != 1 or not panels.get(str(method)):
            raise ValueError(f"Need one full-data reference and study runs: {method}")
        if not np.isclose(group["factor"].astype(float), float(reference.iloc[0]["factor"])).all():
            raise ValueError(f"Study factors differ from full-data Overall factor: {method}")
        method_panels = panels[str(method)]
        panel = method_panels[0]
        if any(other != panel for other in method_panels[1:]):
            raise ValueError(f"Study variants use different concept panels: {method}")
        for row in group.itertuples(index=False):
            key = (str(method), str(row.study_run_id))
            paired = paired_by_key[key]
            if not panel.issubset(set(paired.index)):
                raise ValueError(f"Reference does not cover study panel: {key}")
            aligned = paired.loc[sorted(panel)]
            record = row._asdict()
            record.update(
                selected_concept_score=float(aligned["selected"].mean()),
                baseline_concept_score=float(aligned["baseline"].mean()),
                selected_concept_improvement=float((aligned["selected"] - aligned["baseline"]).mean()),
                selected_overall_improvement=float((aligned["selected_overall"] - aligned["baseline_overall"]).mean()),
                n_concepts=len(panel), n_concepts_available=len(paired),
                num_examples=sizes[key],
            )
            records.append(record)
    result = pd.DataFrame(records)
    panel_description = (
        "the same 10-prompt main-evaluation panel"
        if align_main_prompt_panel
        else "the original 20-prompt sensitivity panel"
    )
    print(
        "Study coverage: 144-reference and variants share each method's "
        f"concept panel; variants use {panel_description}."
    )
    coverage = result.groupby(["train_examples", "num_examples", "n_concepts"], as_index=False).agg(methods=("method", "nunique"), runs=("study_run_id", "size"))
    _display_table(coverage)
    return result


def _study_concept_improvements(
    *, align_main_prompt_panel=False,
) -> pd.DataFrame:
    """Evaluate concept expression at each run's global factor."""
    runs = load_study_runs()
    if runs.empty:
        return runs
    if align_main_prompt_panel:
        runs = runs[
            runs["efficiency"].fillna(False)
            | runs["factor_reference"].fillna(False)
        ].copy()
    canonical = OUTPUT_DIR / "studies/metrics.parquet"
    if canonical.is_file():
        metrics = pd.read_parquet(canonical)
        if "source_evaluator" in metrics and metrics["source_evaluator"].eq("study_lm_judge").any():
            return _current_study_concept_improvements(
                runs, metrics,
                align_main_prompt_panel=align_main_prompt_panel,
            )

    method_to_stem = {}
    for job_path in sorted(OUTPUT_DIR.glob(
        "runtime_configs/integrated/workers/study/*/*.json"
    )):
        try:
            payload = json.loads(job_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        method_to_stem[str(payload.get("method") or job_path.stem)] = job_path.stem

    judge_by_key = {}
    baseline_frames = {}
    for row in runs.itertuples(index=False):
        method = str(row.method)
        run_id = str(row.study_run_id)
        key = (run_id, method)
        if key in judge_by_key:
            continue
        stem = method_to_stem.get(method)
        if stem is None:
            continue
        if bool(row.factor_reference):
            root = OUTPUT_DIR / "methods" / stem / "evaluate/runs"
        else:
            root = OUTPUT_DIR / "studies/runs" / run_id / stem / "evaluate/runs"
        paths = sorted(
            root.glob("*/evaluators/id_lm_judge/metrics.parquet"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if not paths:
            continue
        judge = pd.read_parquet(paths[-1]).copy()
        required = {"method", "factor", "relevance_concept_ratings"}
        if not required.issubset(judge.columns):
            continue
        judge["factor"] = pd.to_numeric(judge["factor"], errors="coerce")
        judge["relevance_concept_ratings"] = pd.to_numeric(
            judge["relevance_concept_ratings"], errors="coerce"
        )
        judge = judge.dropna(subset=["factor", "relevance_concept_ratings"])
        judge_by_key[key] = judge[judge["method"].astype(str) == method].copy()
        zero = judge[np.isclose(judge["factor"], 0.0)].copy()
        if not zero.empty:
            baseline_frames.setdefault(run_id, []).append(zero)

    selected_scores = []
    baseline_scores = []
    improvements = []
    for row in runs.itertuples(index=False):
        method = str(row.method)
        run_id = str(row.study_run_id)
        judge = judge_by_key.get((run_id, method), pd.DataFrame())
        factor = float(row.factor)
        selected = (
            judge.loc[
                np.isclose(judge["factor"], factor),
                "relevance_concept_ratings",
            ] if not judge.empty else pd.Series(dtype=float)
        )
        baseline = (
            judge.loc[
                np.isclose(judge["factor"], 0.0),
                "relevance_concept_ratings",
            ] if not judge.empty else pd.Series(dtype=float)
        )
        if baseline.empty and baseline_frames.get(run_id):
            pool = pd.concat(baseline_frames[run_id], ignore_index=True)
            fallback = getattr(row, "fallback_baseline_method", None)
            if fallback is not None and not pd.isna(fallback):
                candidate = pool[pool["method"].astype(str) == str(fallback)]
            else:
                candidate = pool.iloc[0:0]
            if candidate.empty and not pool.empty:
                baseline_method = sorted(pool["method"].astype(str).unique())[0]
                candidate = pool[pool["method"].astype(str) == baseline_method]
            baseline = candidate["relevance_concept_ratings"]
        selected_score = float(selected.mean()) if not selected.empty else np.nan
        baseline_score = float(baseline.mean()) if not baseline.empty else np.nan
        selected_scores.append(selected_score)
        baseline_scores.append(baseline_score)
        improvements.append(selected_score - baseline_score)

    result = runs.copy()
    result["selected_concept_score"] = selected_scores
    result["baseline_concept_score"] = baseline_scores
    result["selected_concept_improvement"] = improvements
    missing = result["selected_concept_improvement"].isna()
    if missing.any():
        warnings.warn(
            f"Concept-expression study join missed {int(missing.sum())}/{len(result)} runs."
        )
    result = result[~missing].reset_index(drop=True)
    # Older outputs lack the canonical study metrics. Align their Overall
    # reference to the concepts present in the corresponding study variants.
    overall_improvements = []
    for row in result.itertuples(index=False):
        method, run_id = str(row.method), str(row.study_run_id)
        study_panels = [
            set(frame["concept_id"])
            for (other_run, other_method), frame in judge_by_key.items()
            if other_method == method and other_run != run_id
            and "concept_id" in frame and not frame.empty
        ] if bool(row.factor_reference) else []
        judge = judge_by_key.get((run_id, method), pd.DataFrame())
        if study_panels:
            panel = set.intersection(*study_panels)
        elif "concept_id" in judge:
            panel = set(judge["concept_id"])
        else:
            panel = set()
        if not panel or "lm_judge_rating" not in judge:
            overall_improvements.append(np.nan)
            continue
        judge = judge[judge["concept_id"].isin(panel)].copy()
        judge["lm_judge_rating"] = pd.to_numeric(judge["lm_judge_rating"], errors="coerce")
        selected = judge.loc[np.isclose(judge["factor"], float(row.factor))].groupby("concept_id")["lm_judge_rating"].mean()
        baseline = judge.loc[np.isclose(judge["factor"], float(row.baseline_factor))].groupby("concept_id")["lm_judge_rating"].mean()
        if set(selected.index) != panel or set(baseline.index) != panel:
            overall_improvements.append(np.nan)
        else:
            overall_improvements.append(float((selected - baseline).mean()))
    result["selected_overall_improvement"] = overall_improvements
    if result["selected_overall_improvement"].isna().any():
        warnings.warn("Legacy study Overall scores could not be paired for every run; incomplete methods will be omitted.")
    return result


def _eligible_study_methods(
    runs: pd.DataFrame,
    value_column: str = "selected_concept_improvement",
    reference_column: str = "reference_concept_improvement",
):
    references = (
        runs[runs["factor_reference"].fillna(False)]
        .groupby("method", as_index=False)[value_column].mean()
        .rename(columns={value_column: reference_column})
        .sort_values(reference_column, ascending=False)
    )
    references["eligible"] = (
        references[reference_column] >= MIN_REFERENCE_IMPROVEMENT
    ) & (
        ~references["method"].astype(str).isin(PAPER_EXCLUDED_METHODS)
    )
    eligible = set(references.loc[references["eligible"], "method"].astype(str))
    return eligible, references


def _study_line_kwargs(method: str) -> dict:
    kwargs = _method_line_kwargs(method)
    kwargs.update(
        linewidth=0.9, markersize=0.58 * kwargs["markersize"],
        markeredgewidth=0.3, alpha=0.92,
    )
    return kwargs


def _plot_sample_efficiency_panels(
    absolute_summary: pd.DataFrame,
    relative_summary: pd.DataFrame,
    *,
    absolute_ylabel: str,
    relative_ylabel: str,
    section: str,
    slug: str,
):
    """Plot absolute curves and thresholded relative-retention curves."""
    figure, axes = plt.subplots(1, 2, figsize=(6.5, 3.05), dpi=PLOT_DPI)
    for method, group in absolute_summary.groupby("method", sort=True):
        group = group.sort_values("train_examples")
        axes[0].plot(
            group["train_examples"], group["score"],
            **_study_line_kwargs(method),
        )
    for method, group in relative_summary.groupby("method", sort=True):
        group = group.sort_values("train_examples")
        axes[1].plot(
            group["train_examples"], group["relative_improvement_pct"],
            **_study_line_kwargs(method),
        )
    axes[0].set_xlabel("Examples per Concept")
    axes[0].set_ylabel(absolute_ylabel)
    axes[1].set_xlabel("Examples per Concept")
    axes[1].set_ylabel(relative_ylabel)
    axes[1].axhline(100, color="black", ls="--", lw=0.7, alpha=0.35)
    for axis in axes:
        axis.grid(alpha=0.60, linewidth=0.45, linestyle="--")
    legend_methods = pd.concat(
        [absolute_summary["method"], relative_summary["method"]],
        ignore_index=True,
    )
    _top_method_legend(
        figure, _paper_methods(legend_methods), y=0.96, fontsize=5.9,
        expand=True, expand_width=0.74 if section == "main" else 0.905,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.85), pad=0.35, w_pad=0.55)
    return _paper_save(figure, section, slug)


from matplotlib.ticker import MaxNLocator


_PAPER_SERIF_FONTS = [
    "Times New Roman", "Times", "Nimbus Roman",
    "Liberation Serif", "STIXGeneral", "DejaVu Serif",
]


plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": _PAPER_SERIF_FONTS,
    "mathtext.fontset": "stix",
    "font.size": 8.2,
    "axes.labelsize": 8.5,
    "axes.labelpad": 3.0,
    "axes.titlesize": 9.0,
    "axes.titlepad": 4.0,
    "axes.facecolor": "#FCFCFD",
    "axes.edgecolor": "#4B5563",
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "grid.color": "#D7DCE3",
    "grid.linestyle": "--",
    "grid.linewidth": 0.45,
    "grid.alpha": 0.65,
    "legend.fontsize": 6.6,
    "legend.frameon": False,
    "legend.labelspacing": 0.25,
    "lines.linewidth": 1.25,
    "lines.markersize": 4.2,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.025,
})


_PAPER_METRIC_REGISTRY = {}


_PAPER_BEST_FACTOR_CACHE = None


_PAPER_EFFECTIVENESS_CACHE = None


_PAPER_BASE_LABELS = {
    "relevance_concept_ratings": "Concept Expression",
    "fluency_ratings": "Fluency",
    "relevance_instruction_ratings": "Instruction Relevance",
    "superglue_score": "SuperGLUE",
    "mmlu_accuracy": "MMLU",
    "math_accuracy": "MATH",
    "ifeval_prompt_strict_accuracy": "IFEval Prompt",
    "ifeval_instruction_strict_accuracy": "IFEval Instruction",
    "attack_success_rate": "Attack Success",
    "false_refusal_rate": "False Refusal",
    "bbq_accuracy": "BBQ Accuracy",
    "bbq_ambiguous_abs_bias": "BBQ Ambiguous Bias",
    "bbq_disambiguated_abs_bias": "BBQ Disambiguated Bias",
    "truthfulqa_binary_accuracy": "TruthfulQA",
    "retention": "Generalization Retention",
}


def _paper_base_label(metric: str, title: str) -> str:
    if metric in _PAPER_BASE_LABELS:
        return _PAPER_BASE_LABELS[metric]
    match = re.fullmatch(r"superglue_(.+)_score", metric)
    if match:
        task = {
            "boolq": "BoolQ", "cb": "CB", "copa": "COPA",
            "multirc": "MultiRC", "record": "ReCoRD", "rte": "RTE",
            "wic": "WiC", "wsc": "WSC",
        }.get(match.group(1), match.group(1).upper())
        return f"SuperGLUE {task}"
    return re.sub(r"^(Understanding|Knowledge|Reasoning|Bias)\s*[—-]\s*", "", title)


def _paper_method_label(method: str) -> str:
    return method_display_name(method)


def _paper_absolute_label(metric: str, title: str, higher: bool) -> str:
    return f"{_paper_base_label(metric, title)} {'↑' if higher else '↓'}"


def _paper_side_effect_label(metric: str, title: str, higher: bool) -> str:
    kind = "Degradation" if higher else "Increase"
    return f"{_paper_base_label(metric, title)} {kind} ↓"


def _paper_save(figure, section: str, slug: str) -> Path:
    directory = PAPER_MAIN_DIR if section == "main" else PAPER_APPENDIX_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{slug}.pdf"
    figure.savefig(
        path, format="pdf", dpi=300, bbox_inches="tight",
        pad_inches=0.025, facecolor="white",
    )
    print(f"saved: {path}")
    if DISPLAY_FIGURES_IN_NOTEBOOK:
        plt.show()
    plt.close(figure)
    return path


def _save_or_show(figure, slug: str):
    figure.tight_layout(pad=0.35)
    return _paper_save(figure, "main", slug)


def _paper_best_factors() -> pd.DataFrame:
    """Select one global factor per method by mean main-sweep Overall, breaking ties toward the smaller factor."""
    global _PAPER_BEST_FACTOR_CACHE
    if _PAPER_BEST_FACTOR_CACHE is not None:
        return _PAPER_BEST_FACTOR_CACHE.copy()

    data = load_metric(CONCEPT_SCORE_EVALUATOR)
    required = {"method", "concept_id", "factor", "lm_judge_rating"}
    columns = [
        "method", "factor", "_factor_key", "overall_score",
        "baseline_score", "overall_improvement", "overall_std",
        "n_concepts", "n_complete_factors",
    ]
    if data.empty or not required.issubset(data.columns):
        result = pd.DataFrame(columns=columns)
        _PAPER_BEST_FACTOR_CACHE = result
        return result.copy()

    source_columns = ["method", "concept_id", "factor", "lm_judge_rating"]
    rows = data[source_columns].copy()
    for column in ("concept_id", "factor", "lm_judge_rating"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows.dropna(subset=source_columns)
    rows = (
        rows.groupby(
            ["method", "concept_id", "factor"],
            as_index=False, observed=True,
        )
        .agg(overall_score=("lm_judge_rating", "mean"))
    )
    by_factor = (
        rows.groupby(["method", "factor"], as_index=False, observed=True)
        .agg(
            overall_score=("overall_score", "mean"),
            overall_std=("overall_score", "std"),
            n_concepts=("concept_id", "nunique"),
        )
    )
    expected_concepts = (
        rows.groupby("method", observed=True)["concept_id"].nunique()
        .rename("expected_concepts")
    )
    by_factor = by_factor.merge(
        expected_concepts, on="method", how="left", validate="many_to_one"
    )
    incomplete = by_factor[
        by_factor["n_concepts"] != by_factor["expected_concepts"]
    ]
    if not incomplete.empty:
        preview = incomplete[[
            "method", "factor", "n_concepts", "expected_concepts"
        ]].head(12)
        warnings.warn(
            "Global factor selection ignored incomplete method/factor concept "
            f"panels:\n{preview.to_string(index=False)}",
            stacklevel=2,
        )
    complete = by_factor[
        by_factor["n_concepts"] == by_factor["expected_concepts"]
    ].copy()
    if complete.empty:
        result = pd.DataFrame(columns=columns)
        _PAPER_BEST_FACTOR_CACHE = result
        return result.copy()

    baselines = complete[np.isclose(complete["factor"], 0.0)][
        ["method", "overall_score"]
    ].rename(columns={"overall_score": "baseline_score"})
    baseline_counts = baselines.groupby("method", observed=True).size()
    invalid_baselines = baseline_counts[baseline_counts != 1]
    if not invalid_baselines.empty:
        raise ValueError(
            "Global factor selection requires exactly one factor-0 baseline "
            f"per method: {invalid_baselines.to_dict()}."
        )

    candidates = complete[~np.isclose(complete["factor"], 0.0)].copy()
    candidate_counts = (
        candidates.groupby("method", observed=True)["factor"].nunique()
        .rename("n_complete_factors")
    )
    missing_candidates = sorted(
        set(complete["method"].astype(str))
        - set(candidate_counts.index.astype(str))
    )
    if missing_candidates:
        raise ValueError(
            "Global factor selection requires at least one complete nonzero "
            f"candidate factor per method: {missing_candidates}."
        )

    candidates = candidates.sort_values(
        ["method", "overall_score", "factor"],
        ascending=[True, False, True],
    )
    selected = candidates.drop_duplicates("method", keep="first").copy()
    factor_counts = candidate_counts
    selected = selected.merge(
        baselines, on="method", how="left", validate="one_to_one"
    ).merge(
        factor_counts, on="method", how="left", validate="one_to_one"
    )
    if selected["baseline_score"].isna().any():
        missing = selected.loc[selected["baseline_score"].isna(), "method"].tolist()
        raise ValueError(f"Missing factor-0 Overall baselines: {missing}.")
    selected["overall_improvement"] = (
        selected["overall_score"] - selected["baseline_score"]
    )
    selected["_factor_key"] = selected["factor"].round(8)
    selected = selected[columns].sort_values("method").reset_index(drop=True)
    _PAPER_BEST_FACTOR_CACHE = selected
    return selected.copy()


def _paper_id_effects() -> pd.DataFrame:
    columns = [
        "method", "concept_id", "factor", CONCEPT_SCORE_COLUMN,
        "baseline_relevance", "effect_delta", "factor_scale",
        "normalized_factor", "_factor_key",
    ]
    data = load_metric(CONCEPT_SCORE_EVALUATOR)
    required = {"method", "concept_id", "factor", CONCEPT_SCORE_COLUMN}
    if data.empty or not required.issubset(data.columns):
        return pd.DataFrame(columns=columns)
    rows = data[["method", "concept_id", "factor", CONCEPT_SCORE_COLUMN]].copy()
    for column in ("concept_id", "factor", CONCEPT_SCORE_COLUMN):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = (
        rows.dropna()
        .groupby(["method", "concept_id", "factor"], as_index=False, observed=True)
        [CONCEPT_SCORE_COLUMN].mean()
    )
    baseline = rows[np.isclose(rows["factor"], 0.0)][
        ["method", "concept_id", CONCEPT_SCORE_COLUMN]
    ].rename(columns={CONCEPT_SCORE_COLUMN: "baseline_relevance"})
    rows = rows.merge(
        baseline, on=["method", "concept_id"], how="inner", validate="many_to_one"
    )
    rows["effect_delta"] = rows[CONCEPT_SCORE_COLUMN] - rows["baseline_relevance"]
    rows = _with_normalized_factor(rows)
    rows["_factor_key"] = rows["factor"].round(8)
    return rows


def _paper_effectiveness_table() -> pd.DataFrame:
    global _PAPER_EFFECTIVENESS_CACHE
    if _PAPER_EFFECTIVENESS_CACHE is not None:
        return _PAPER_EFFECTIVENESS_CACHE.copy()
    rows = _paper_id_effects()
    if rows.empty:
        table = pd.DataFrame(columns=[
            "method", "max_relevance_improvement",
            "relevance_concepts", "included_in_main",
        ])
        _PAPER_EFFECTIVENESS_CACHE = table
        return table.copy()
    by_factor = (
        rows.groupby(["method", "factor"], as_index=False, observed=True)
        .agg(
            relevance_improvement=("effect_delta", "mean"),
            n_concepts=("concept_id", "nunique"),
        )
    )
    table = (
        by_factor.groupby("method", as_index=False, observed=True)
        .agg(
            max_relevance_improvement=("relevance_improvement", "max"),
            relevance_concepts=("n_concepts", "max"),
        )
        .sort_values("max_relevance_improvement", ascending=False)
    )
    table["included_in_main"] = (
        table["max_relevance_improvement"] >= PAPER_RELEVANCE_THRESHOLD
    )
    _PAPER_EFFECTIVENESS_CACHE = table
    return table.copy()


def _paper_main_methods() -> set[str]:
    table = _paper_effectiveness_table()
    if table.empty:
        # The relevance judge is often imported after side-effect inference.
        # Until then, keep every configured method instead of filtering all out.
        methods = set(_METHOD_ORDER)
    else:
        methods = set(
            table.loc[table["included_in_main"], "method"].astype(str)
        )
    methods.difference_update(PAPER_EXCLUDED_METHODS)
    if METHODS is not None:
        methods.intersection_update(map(str, METHODS))
    return methods


def _paper_tradeoff_summary(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    return (
        paired.groupby(["method", "factor"], as_index=False, observed=True)
        .agg(
            normalized_factor=("normalized_factor", "first"),
            effect_mean=("effect_delta", "mean"),
            side_effect_mean=("side_effect_delta", "mean"),
            n_concepts=("concept_id", "nunique"),
        )
        .sort_values(["method", "factor"])
    )


def _paper_best_tradeoff(paired: pd.DataFrame) -> pd.DataFrame:
    selected = _paper_best_factors()[["method", "_factor_key"]]
    if paired.empty or selected.empty:
        return pd.DataFrame()
    rows = paired.copy()
    rows["_factor_key"] = pd.to_numeric(rows["factor"], errors="coerce").round(8)
    rows = rows.merge(
        selected, on=["method", "_factor_key"],
        how="inner", validate="many_to_one",
    )
    return (
        rows.groupby("method", as_index=False, observed=True)
        .agg(
            effect_mean=("effect_delta", "mean"),
            side_effect_mean=("side_effect_delta", "mean"),
            normalized_factor=("normalized_factor", "mean"),
            n_concepts=("concept_id", "nunique"),
        )
    )


def _paper_best_metric(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    selected = _paper_best_factors()[["method", "_factor_key"]]
    if data.empty or selected.empty:
        return pd.DataFrame()
    rows = _with_normalized_factor(data.copy())
    rows[metric] = pd.to_numeric(rows[metric], errors="coerce")
    rows["_factor_key"] = pd.to_numeric(rows["factor"], errors="coerce").round(8)
    rows = (
        rows.dropna(subset=["method", "concept_id", "factor", metric])
        .groupby(
            ["method", "concept_id", "factor", "_factor_key"],
            as_index=False, observed=True,
        )
        .agg(
            metric_mean=(metric, "mean"),
            normalized_factor=("normalized_factor", "first"),
        )
    )
    rows = rows.merge(
        selected, on=["method", "_factor_key"],
        how="inner", validate="many_to_one",
    )
    return (
        rows.groupby("method", as_index=False, observed=True)
        .agg(
            normalized_factor=("normalized_factor", "mean"),
            metric_mean=("metric_mean", "mean"),
            n_concepts=("concept_id", "nunique"),
        )
    )


def _paper_methods(values) -> list[str]:
    methods = sorted(
        {str(value) for value in values}.difference(PAPER_EXCLUDED_METHODS),
        key=_method_style_index,
    )
    if METHODS is not None:
        allowed = set(map(str, METHODS))
        methods = [method for method in methods if method in allowed]
    return methods


def _bottom_method_legend(
    figure, methods, *, y=0.132, fontsize=6.4,
):
    """Use two rows for multiple methods and fit the legend to the figure."""
    methods = list(methods)
    if not methods:
        return None

    handles = [_method_legend_handle(method) for method in methods]
    labels = [_paper_method_label(method) for method in methods]
    columns = max(1, math.ceil(len(methods) / 2))
    available_width = figure.bbox.width * 0.97
    font_size = float(fontsize)
    for attempt in range(6):
        legend = figure.legend(
            handles=handles, labels=labels,
            loc="upper center", bbox_to_anchor=(0.5, y),
            ncol=columns, fontsize=font_size,
            frameon=True, framealpha=0.92, facecolor="white",
            edgecolor="#D1D5DB", fancybox=False,
            columnspacing=0.28, handlelength=0.9, handletextpad=0.2,
            borderaxespad=0.0, borderpad=0.28, labelspacing=0.22,
        )
        figure.canvas.draw()
        legend_width = legend.get_window_extent(
            figure.canvas.get_renderer()
        ).width
        if legend_width <= available_width or attempt == 5:
            return legend
        legend.remove()
        font_size *= 0.98 * available_width / legend_width


def _top_method_legend(
    figure, methods, *, y=0.995, fontsize=5.9, expand=False,
    expand_width=0.905,
):
    """Use a two-row method legend above a multi-panel figure."""
    methods = list(methods)
    if not methods:
        return None

    handles = [_method_legend_handle(method) for method in methods]
    labels = [_paper_method_label(method) for method in methods]
    columns = max(1, math.ceil(len(methods) / 2))
    available_width = figure.bbox.width * 0.97
    font_size = float(fontsize)
    for attempt in range(6):
        legend_kwargs = {}
        if expand:
            expand_width = float(expand_width)
            legend_kwargs.update(
                loc="upper left",
                bbox_to_anchor=(
                    (1.0 - expand_width) / 2.0, y, expand_width, 0.0,
                ),
                mode="expand",
            )
        else:
            legend_kwargs.update(
                loc="upper center", bbox_to_anchor=(0.5, y),
            )
        legend = figure.legend(
            handles=handles, labels=labels,
            ncol=columns, fontsize=font_size, frameon=False,
            columnspacing=0.28, handlelength=0.9, handletextpad=0.2,
            borderaxespad=0.0, borderpad=0.28, labelspacing=0.22,
            **legend_kwargs,
        )
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        legend_bbox = legend.get_window_extent(renderer)
        legend_width = legend_bbox.width
        if expand or legend_width <= available_width or attempt == 5:
            legend_bbox_figure = legend_bbox.transformed(
                figure.transFigure.inverted()
            )
            frame = Rectangle(
                (0.045, legend_bbox_figure.y0 - 0.004),
                0.945, legend_bbox_figure.height + 0.008,
                transform=figure.transFigure, clip_on=False,
                facecolor="none", edgecolor="#9CA3AF",
                linewidth=0.55, zorder=legend.get_zorder() - 1,
            )
            figure.add_artist(frame)
            return legend
        legend.remove()
        font_size *= 0.98 * available_width / legend_width


def _inside_upper_left_method_legend(
    axis, methods, *, fontsize=5.8, rows=2, anchor_x=0.01,
    width_fraction=None, frameon=True,
):
    """Place a compact shared legend inside the plot at its upper left."""
    methods = list(methods)
    if not methods:
        return None

    figure = axis.figure
    handles = [_method_legend_handle(method) for method in methods]
    labels = [_paper_method_label(method) for method in methods]
    columns = max(1, math.ceil(len(methods) / max(1, int(rows))))
    if width_fraction is None:
        legend_bbox = (anchor_x, 0.99)
        legend_mode = None
        available_width = axis.bbox.width * max(0.1, 0.99 - anchor_x)
    else:
        legend_bbox = (anchor_x, 0.99, float(width_fraction), 0.0)
        legend_mode = "expand"
        available_width = axis.bbox.width * float(width_fraction)
    font_size = float(fontsize)
    for attempt in range(6):
        legend = axis.legend(
            handles=handles, labels=labels,
            loc="upper left", bbox_to_anchor=legend_bbox, mode=legend_mode,
            ncol=columns, fontsize=font_size,
            frameon=frameon, framealpha=0.86, facecolor="white",
            edgecolor="#D1D5DB", fancybox=False,
            columnspacing=0.24, handlelength=0.85, handletextpad=0.18,
            borderaxespad=0.0, borderpad=0.25, labelspacing=0.18,
        )
        figure.canvas.draw()
        legend_width = legend.get_window_extent(
            figure.canvas.get_renderer()
        ).width
        if legend_width <= available_width or attempt == 5:
            return legend
        legend.remove()
        font_size *= 0.98 * available_width / legend_width


def _paper_legend(
    figure, methods, *, best: bool, appendix: bool, fontsize=6.4,
):
    return _bottom_method_legend(figure, methods, fontsize=fontsize)


def _paper_better(axis):
    axis.text(
        0.985, 0.035, "Better ↘", transform=axis.transAxes,
        ha="right", va="bottom", fontsize=6.2, color="#18794E",
    )


def _paper_best_factor_cutoffs() -> dict[str, float]:
    """Return each method's single global Overall-best factor."""
    best = _paper_best_factors()
    if best.empty:
        return {}
    factors = best[["method", "factor"]].copy()
    factors["factor"] = pd.to_numeric(factors["factor"], errors="coerce")
    factors = factors.dropna(subset=["method", "factor"])
    return (
        factors.groupby("method", observed=True)["factor"]
        .max()
        .astype(float)
        .to_dict()
    )


def _truncate_at_first_left_turn(
    group: pd.DataFrame, *, best_factor: float | None = None,
) -> pd.DataFrame:
    """Truncate at the first efficacy decrease strictly after the global best factor."""
    group = group.sort_values("factor").reset_index(drop=True)
    left_steps = group["effect_mean"].diff().lt(0)
    if best_factor is not None and np.isfinite(best_factor):
        factors = pd.to_numeric(group["factor"], errors="coerce")
        left_steps &= factors.gt(float(best_factor))
    if left_steps.any():
        first_left_position = int(np.flatnonzero(left_steps.to_numpy())[0])
        return group.iloc[:first_left_position].copy()
    return group


def _paper_tradeoff_plot(
    summary, best_points, *, methods, ylabel, section, slug,
    truncate_at_left_turn=False, legend_inside=False,
):
    rows = summary.copy()
    if methods is not None:
        rows = rows[rows["method"].astype(str).isin(methods)]
    names = _paper_methods(rows.get("method", []))
    best_factor_cutoffs = (
        _paper_best_factor_cutoffs() if truncate_at_left_turn else {}
    )
    figure, axis = plt.subplots(figsize=PAPER_MAIN_SIZE, dpi=PLOT_DPI)
    for method in names:
        group = rows[rows["method"].astype(str) == method].sort_values("factor")
        if truncate_at_left_turn:
            group = _truncate_at_first_left_turn(
                group, best_factor=best_factor_cutoffs.get(method)
            )
        nonzero = group[~np.isclose(group["factor"], 0.0)]
        style = _method_style(method)
        if len(nonzero) <= 1:
            point = nonzero.iloc[-1] if not nonzero.empty else None
            if point is not None:
                axis.scatter(
                    point["effect_mean"], point["side_effect_mean"],
                    color=style["color"],
                    marker=style["marker"], s=34, edgecolor="white",
                    linewidth=0.55, zorder=5,
                )
            continue
        axis.plot(
            group["effect_mean"], group["side_effect_mean"],
            **_method_line_kwargs(method, label=False),
        )
    axis.axhline(0, color="#374151", lw=0.65, alpha=0.32)
    axis.axvline(0, color="#374151", lw=0.65, alpha=0.32)
    axis.set_xlabel("Concept Expression ↑")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.60, linewidth=0.45, linestyle="--")
    axis.margins(x=0.035, y=0.055)
    _paper_better(axis)
    if legend_inside:
        _inside_upper_left_method_legend(
            axis, names, fontsize=5.8 if truncate_at_left_turn else 5.6,
        )
        figure.tight_layout(pad=0.35)
    else:
        _paper_legend(
            figure, names, best=False, appendix=section == "appendix",
            fontsize=6.8 if truncate_at_left_turn else 6.4,
        )
        figure.tight_layout(rect=(0, 0.145, 1, 1), pad=0.35)
    return _paper_save(figure, section, slug)


def _paper_factor_grid(entry):
    summary = entry["factor_summary"]
    if summary.empty:
        return None
    ci = concept_bootstrap_factor_ci(
        entry["factor_ci_rows"], resamples=FACTOR_GRID_BOOTSTRAP_RESAMPLES,
        seed=FACTOR_GRID_BOOTSTRAP_SEED,
    )
    ci = _with_normalized_factor(ci)
    summary = summary.merge(
        ci[["method", "normalized_factor", "ci95_lower", "ci95_upper"]],
        on=["method", "normalized_factor"], how="left", validate="one_to_one",
    )
    entry["factor_summary_with_ci"] = summary
    methods = _paper_methods(summary["method"])
    columns = PAPER_GRID_COLUMNS
    rows = max(1, math.ceil(len(methods) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(6.5, PAPER_GRID_ROW_HEIGHT * rows),
        dpi=PLOT_DPI, sharex=True, sharey=True, squeeze=False,
    )
    for axis, method in zip(axes.flat, methods):
        group = summary[summary["method"].astype(str) == method].sort_values(
            "normalized_factor"
        )
        nonzero = group[~np.isclose(group["normalized_factor"], 0.0)]
        style = _method_style(method)
        if len(nonzero) <= 1:
            point = nonzero.iloc[-1] if not nonzero.empty else None
            if point is not None and np.isfinite(point["ci95_lower"]) and np.isfinite(point["ci95_upper"]):
                axis.vlines(point["normalized_factor"], point["ci95_lower"],
                            point["ci95_upper"], color=style["color"], lw=1.0)
                axis.plot([point["normalized_factor"]] * 2,
                          [point["ci95_lower"], point["ci95_upper"]],
                          linestyle="none", marker="_", markersize=4,
                          color=style["color"])

            if point is not None:
                axis.scatter(
                    point["normalized_factor"], point["metric_mean"],
                    color=style["color"],
                    marker=style["marker"], s=24, edgecolor="white",
                    linewidth=0.5, zorder=5,
                )
        else:
            axis.fill_between(
                group["normalized_factor"].to_numpy(dtype=float),
                group["ci95_lower"].to_numpy(dtype=float),
                group["ci95_upper"].to_numpy(dtype=float),
                color=style["color"], alpha=0.20, linewidth=0, zorder=1,
            )
            axis.plot(
                group["normalized_factor"], group["metric_mean"],
                color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], lw=0.95, markersize=2.5,
                markeredgecolor="white", markeredgewidth=0.3,
            )
        axis.text(
            0.01, 1.015, _paper_method_label(method),
            transform=axis.transAxes, ha="left", va="bottom",
            fontsize=5.6, clip_on=False,
        )
        axis.grid(alpha=0.55, linewidth=0.4, linestyle="--")
        axis.tick_params(labelsize=5.8, length=1.8)
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.yaxis.set_major_locator(MaxNLocator(4))
    for axis in axes.flat[len(methods):]:
        axis.set_visible(False)
    figure.supxlabel("Normalized Strength", fontsize=8.0)
    figure.supylabel(entry["absolute_label"], fontsize=8.0)
    figure.tight_layout(
        rect=(0.032, 0.032, 1, 0.995), pad=0.25, h_pad=0.45, w_pad=0.35
    )
    return _paper_save(figure, "appendix", f"{entry['slug']}__factor_grid")


def analyze_metric(
    evaluator_id: str,
    metric: str,
    title: str,
    *,
    filters: dict | None = None,
    derive=None,
    ylabel: str | None = None,
    higher_is_better: bool = True,
    show_tradeoff: bool = True,
    paper_main: bool = True,
):
    data = load_metric(evaluator_id)
    if data.empty:
        print(f"[waiting] No finalized {evaluator_id} metrics.")
        return None
    data = _apply_filters(data, filters)
    if derive is not None:
        data = derive(data.copy())
    if metric not in data:
        print(f"[waiting] Missing {evaluator_id}/{metric}.")
        return None
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.dropna(subset=["method", "concept_id", "factor", metric])
    paired = (
        _effect_side_effect_rows(data, metric, higher_is_better=higher_is_better)
        if show_tradeoff else pd.DataFrame()
    )
    factor_summary = _aggregate_curve(
        _with_normalized_factor(data), "normalized_factor", metric
    )
    tradeoff = _paper_tradeoff_summary(paired)
    best_tradeoff = _paper_best_tradeoff(paired)
    best_metric = _paper_best_metric(data, metric)
    slug = f"{evaluator_id}__{metric}"
    entry = {
        "slug": slug,
        "metric": metric,
        "absolute_label": _paper_absolute_label(metric, title, higher_is_better),
        "side_effect_label": _paper_side_effect_label(metric, title, higher_is_better),
        "factor_summary": factor_summary,
        "factor_ci_rows": data[["method", "concept_id", "factor", metric]].rename(columns={metric: "value"}),
        "tradeoff": tradeoff,
        "best_tradeoff": best_tradeoff,
        "best_metric": best_metric,
    }
    _PAPER_METRIC_REGISTRY[(OUTPUT_KEY, slug)] = entry
    if paper_main and show_tradeoff and not tradeoff.empty:
        _paper_tradeoff_plot(
            tradeoff, best_tradeoff, methods=_paper_main_methods(),
            ylabel=entry["side_effect_label"], section="main",
            slug=f"{slug}__tradeoff_filtered",
        )
    print(
        f"{title}: {data['method'].nunique()} methods; "
        f"{len(set(data['method']).intersection(_paper_main_methods()))} in main."
    )
    return {
        "rows": data, "by_factor": factor_summary,
        "paired_deltas": paired, "effect_side_effect": tradeoff,
        "best_tradeoff_points": best_tradeoff,
    }


def render_registered_metric_appendix():
    paths = []
    for (output_key, _slug), entry in _PAPER_METRIC_REGISTRY.items():
        if output_key != OUTPUT_KEY:
            continue
        if not entry["tradeoff"].empty:
            paths.append(_paper_tradeoff_plot(
                entry["tradeoff"], entry["best_tradeoff"], methods=None,
                ylabel=entry["side_effect_label"], section="appendix",
                slug=f"{entry['slug']}__tradeoff_all",
                legend_inside=entry["slug"] == "composite_side_effect",
            ))
        path = _paper_factor_grid(entry)
        if path is not None:
            paths.append(path)
    print(f"Appendix export complete: {len(paths)} PDF figures.")
    return paths


_PAPER_COMPOSITE_RESULT = None


def _paper_build_composite():
    rows, diagnostics = _composite_side_effect_rows()
    rows, incomplete = retain_complete_method_factors(
        rows, [spec["name"] for spec in COMPOSITE_SIDE_EFFECT_SPECS]
    )
    if not incomplete.empty:
        preview = incomplete.head(12).to_string(index=False)
        suffix = (
            ""
            if len(incomplete) <= 12
            else f"\n... and {len(incomplete) - 12} more incomplete points"
        )
        warnings.warn(
            "Composite side-effect plots skipped method/factor points that are "
            "not present in all configured indicators. The intersection is "
            f"computed separately for each method:\n{preview}{suffix}",
            stacklevel=2,
        )
    if rows.empty:
        raise ValueError(
            "Composite has no method/factor point shared by all configured "
            "side-effect indicators."
        )
    composite = _composite_side_effect_factor_means(rows)
    effects = _paper_id_effects()
    effect_means = (
        effects.groupby(["method", "factor"], as_index=False, observed=True)
        .agg(
            effect_mean=("effect_delta", "mean"),
            relevance_concepts=("concept_id", "nunique"),
        )
    )
    tradeoff = composite.merge(
        effect_means, on=["method", "factor"], how="inner", validate="one_to_one"
    ).rename(columns={
        "mean_side_effect": "side_effect_mean",
        "relevance_concepts": "n_concepts",
    })

    factor_summary = _with_normalized_factor(
        composite[["method", "factor", "mean_side_effect"]]
        .rename(columns={"mean_side_effect": "metric_mean"})
    )
    factor_summary = factor_summary.merge(
        composite[[
            "method", "factor", "min_concepts_per_side_effect",
        ]].rename(columns={"min_concepts_per_side_effect": "n_concepts"}),
        on=["method", "factor"], how="left", validate="one_to_one",
    )

    selected = _paper_best_factors()[["method", "_factor_key"]]
    selected_side = rows.copy()
    selected_side["_factor_key"] = pd.to_numeric(
        selected_side["factor"], errors="coerce"
    ).round(8)
    selected_side = selected_side.merge(
        selected, on=["method", "_factor_key"],
        how="inner", validate="many_to_one",
    )
    indicator = (
        selected_side.groupby(
            ["method", "side_effect_name"], as_index=False, observed=True
        )
        .agg(value=("normalized_side_effect_delta", "mean"))
    )
    best_y = (
        indicator.groupby("method", as_index=False, observed=True)
        .agg(side_effect_mean=("value", "mean"))
    )
    selected_effect = effects.merge(
        selected, on=["method", "_factor_key"],
        how="inner", validate="many_to_one",
    )
    best_x = (
        selected_effect.groupby("method", as_index=False, observed=True)
        .agg(
            effect_mean=("effect_delta", "mean"),
            normalized_factor=("normalized_factor", "mean"),
            n_concepts=("concept_id", "nunique"),
        )
    )
    best_tradeoff = best_x.merge(best_y, on="method", how="inner")
    best_metric = (
        best_x[["method", "normalized_factor", "n_concepts"]]
        .merge(
            best_y.rename(columns={"side_effect_mean": "metric_mean"}),
            on="method", how="inner",
        )
    )
    entry = {
        "slug": "composite_side_effect",
        "metric": "mean_side_effect",
        "absolute_label": "Mean Side Effect ↓",
        "side_effect_label": "Mean Side Effect ↓",
        "factor_summary": factor_summary,
        "factor_ci_rows": rows[["method", "concept_id", "factor", "side_effect_name", "normalized_side_effect_delta"]].rename(
            columns={"side_effect_name": "component", "normalized_side_effect_delta": "value"}
        ),
        "tradeoff": tradeoff[[
            "method", "factor", "effect_mean", "side_effect_mean", "n_concepts",
        ]],
        "best_tradeoff": best_tradeoff,
        "best_metric": best_metric,
    }
    return rows, diagnostics, entry


def analyze_composite_side_effect_curves():
    global _PAPER_COMPOSITE_RESULT
    try:
        rows, diagnostics, entry = _paper_build_composite()
    except ValueError as error:
        print(f"[waiting] {error}")
        empty_points = pd.DataFrame(columns=[
            "method", "effect_mean", "side_effect_mean",
            "normalized_factor", "n_concepts",
        ])
        _PAPER_COMPOSITE_RESULT = {
            "rows": pd.DataFrame(),
            "diagnostics": [],
            "curves": pd.DataFrame(columns=[
                "method", "factor", "effect_mean",
                "side_effect_mean", "n_concepts",
            ]),
            "best_points": empty_points,
        }
        return _PAPER_COMPOSITE_RESULT
    _PAPER_METRIC_REGISTRY[(OUTPUT_KEY, entry["slug"])] = entry
    _PAPER_COMPOSITE_RESULT = {
        "rows": rows,
        "diagnostics": diagnostics,
        "curves": entry["tradeoff"],
        "best_points": entry["best_tradeoff"],
    }
    _paper_tradeoff_plot(
        entry["tradeoff"], entry["best_tradeoff"],
        methods=_paper_main_methods(),
        ylabel=entry["side_effect_label"],
        section="main",
        slug="composite_side_effect__tradeoff_filtered",
        truncate_at_left_turn=True,
        legend_inside=True,
    )
    return _PAPER_COMPOSITE_RESULT


def analyze_composite_side_effect_best_factor():
    global _PAPER_COMPOSITE_RESULT
    if _PAPER_COMPOSITE_RESULT is None:
        analyze_composite_side_effect_curves()
    points = _PAPER_COMPOSITE_RESULT["best_points"].copy()
    if points.empty:
        print("[waiting] Composite best-factor points require finalized judge metrics.")
        return {"points": points}
    print("Global Overall-best-factor aggregate positions.")
    _display_table(points.sort_values("effect_mean", ascending=False).round(4))
    return {"points": points}


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson interval for a Monte Carlo proportion."""
    if total <= 0:
        raise ValueError("Monte Carlo sample count must be positive.")
    z = 1.959963984540054
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - radius, center + radius


def _metric_weighting_panels(composite_result=None):
    """Build per-indicator vectors for factor-wise and selected-factor points."""
    if composite_result is None or composite_result.get("rows", pd.DataFrame()).empty:
        rows, _, entry = _paper_build_composite()
        curve_effects = entry["tradeoff"][["method", "factor", "effect_mean"]]
        best_effects = entry["best_tradeoff"][["method", "effect_mean"]]
    else:
        rows = composite_result["rows"].copy()
        curve_effects = composite_result["curves"][
            ["method", "factor", "effect_mean"]
        ]
        best_effects = composite_result["best_points"][
            ["method", "effect_mean"]
        ]

    indicator_names = [spec["name"] for spec in COMPOSITE_SIDE_EFFECT_SPECS]
    indicator_curve = (
        rows.groupby(
            ["method", "factor", "side_effect_name"], observed=True
        )["normalized_side_effect_delta"]
        .mean()
        .unstack("side_effect_name")
        .reindex(columns=indicator_names)
        .reset_index()
    )
    curve = curve_effects.merge(
        indicator_curve,
        on=["method", "factor"],
        how="inner",
        validate="one_to_one",
    )
    curve = curve[~np.isclose(curve["factor"], 0.0)].reset_index(drop=True)

    selected = _paper_best_factors()[["method", "_factor_key"]]
    selected_side = rows.copy()
    selected_side["_factor_key"] = pd.to_numeric(
        selected_side["factor"], errors="coerce"
    ).round(8)
    selected_side = selected_side.merge(
        selected,
        on=["method", "_factor_key"],
        how="inner",
        validate="many_to_one",
    )
    indicator_best = (
        selected_side.groupby(
            ["method", "side_effect_name"], observed=True
        )["normalized_side_effect_delta"]
        .mean()
        .unstack("side_effect_name")
        .reindex(columns=indicator_names)
        .reset_index()
    )
    best = best_effects.merge(
        indicator_best, on="method", how="inner", validate="one_to_one"
    )

    for label, panel in (("factor-wise", curve), ("Overall-selected", best)):
        missing = panel[indicator_names].isna().any(axis=1)
        if missing.any():
            examples = panel.loc[missing, "method"].astype(str).head().tolist()
            raise ValueError(
                f"Metric-weighting robustness has incomplete {label} points: "
                f"{examples}."
            )
    return indicator_names, curve, best


def _metric_weighting_comparison(
    panel: pd.DataFrame,
    indicator_names: list[str],
    *,
    prompt_method: str,
    comparison: str,
) -> tuple[pd.Series, pd.DataFrame, np.ndarray]:
    """Return Prompt Steering and points that can dominate it in efficacy."""
    prompt_rows = panel[panel["method"].astype(str).eq(prompt_method)]
    if len(prompt_rows) != 1:
        raise ValueError(
            f"{comparison}: expected exactly one {prompt_method} point, "
            f"found {len(prompt_rows)}."
        )
    prompt = prompt_rows.iloc[0]
    epsilon = 1e-12
    competitors = panel[
        ~panel["method"].astype(str).eq(prompt_method)
        & panel["effect_mean"].gt(float(prompt["effect_mean"]) + epsilon)
    ].copy()
    prompt_vector = prompt[indicator_names].to_numpy(dtype=float)
    competitor_vectors = competitors[indicator_names].to_numpy(dtype=float)
    differences = competitor_vectors - prompt_vector[None, :]
    return prompt, competitors, differences


def _simulate_metric_weighting(
    differences: np.ndarray,
    *,
    alpha: float,
    draws: int,
    batch_size: int,
    seed: int,
) -> tuple[int, np.ndarray]:
    """Count draws for which no efficacy-superior point has lower side effect."""
    if alpha <= 0:
        raise ValueError(f"Dirichlet concentration must be positive, got {alpha}.")
    if draws <= 0 or batch_size <= 0:
        raise ValueError("Monte Carlo draws and batch size must be positive.")
    rng = np.random.default_rng(seed)
    nondominated = 0
    dominated_by = np.zeros(len(differences), dtype=np.int64)
    remaining = int(draws)
    epsilon = 1e-12
    while remaining:
        count = min(int(batch_size), remaining)
        weights = rng.dirichlet(
            np.full(differences.shape[1], float(alpha)), size=count
        )
        weighted_differences = weights @ differences.T
        individual_dominators = weighted_differences < -epsilon
        nondominated += int((~individual_dominators.any(axis=1)).sum())
        dominated_by += individual_dominators.sum(axis=0)
        remaining -= count
    return nondominated, dominated_by


def _metric_weighting_display_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary.copy()
    table["Weight distribution"] = table["dirichlet_alpha"].map(
        lambda value: f"Dirichlet({value:g})"
    )
    table["Undominated draws"] = table["undominated_probability"].map(
        lambda value: f"{100.0 * value:.4f}%"
    )
    table["95% Monte Carlo CI"] = table.apply(
        lambda row: (
            f"[{100.0 * row['ci95_lower']:.4f}%, "
            f"{100.0 * row['ci95_upper']:.4f}%]"
        ),
        axis=1,
    )
    return table[[
        "model", "Weight distribution", "Undominated draws",
        "95% Monte Carlo CI",
    ]].rename(columns={"model": "Model"})


def analyze_metric_weighting_robustness(
    composite_results=None,
    *,
    draws: int | None = None,
    alphas: tuple[float, ...] | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    prompt_method: str = "PromptSteering",
):
    """Estimate Prompt Steering's Pareto robustness under random side-effect weights."""
    draws = METRIC_WEIGHTING_DRAWS if draws is None else int(draws)
    batch_size = (
        METRIC_WEIGHTING_BATCH_SIZE if batch_size is None else int(batch_size)
    )
    seed = METRIC_WEIGHTING_SEED if seed is None else int(seed)
    alphas = (
        METRIC_WEIGHTING_DIRICHLET_ALPHAS
        if alphas is None
        else tuple(float(alpha) for alpha in alphas)
    )
    composite_result = (
        composite_results.get(OUTPUT_KEY)
        if isinstance(composite_results, dict)
        else composite_results
    )
    indicator_names, curve, best = _metric_weighting_panels(composite_result)
    model_index = OUTPUT_KEYS.index(OUTPUT_KEY)
    model = {
        "2b": "Gemma-2-2B-it",
        "9b": "Gemma-2-9B-it",
    }.get(OUTPUT_KEY.split("/", 1)[0].lower(), OUTPUT_KEY)

    summary_rows = []
    competitor_rows = []
    comparisons = (
        ("all_nonzero_factor_points", curve),
        ("overall_selected_factor", best),
    )
    for comparison_index, (comparison, panel) in enumerate(comparisons):
        prompt, competitors, differences = _metric_weighting_comparison(
            panel,
            indicator_names,
            prompt_method=prompt_method,
            comparison=comparison,
        )
        equal_weight_differences = differences.mean(axis=1)
        equal_weight_nondominated = not np.any(
            equal_weight_differences < -1e-12
        )
        for alpha_index, alpha in enumerate(alphas):
            run_seed = (
                seed + model_index * 100
                + comparison_index * 10 + alpha_index
            )
            nondominated, dominated_by = _simulate_metric_weighting(
                differences,
                alpha=alpha,
                draws=draws,
                batch_size=batch_size,
                seed=run_seed,
            )
            lower, upper = _wilson_interval(nondominated, draws)
            summary_rows.append({
                "model": model,
                "comparison": comparison,
                "dirichlet_alpha": alpha,
                "draws": draws,
                "undominated_count": nondominated,
                "undominated_probability": nondominated / draws,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "prompt_effectiveness": float(prompt["effect_mean"]),
                "eligible_points": len(panel),
                "efficacy_superior_competitors": len(competitors),
                "equal_weight_nondominated": equal_weight_nondominated,
                "seed": run_seed,
                "n_indicators": len(indicator_names),
            })
            for (_, competitor), count, equal_difference in zip(
                competitors.iterrows(), dominated_by, equal_weight_differences
            ):
                competitor_rows.append({
                    "model": model,
                    "comparison": comparison,
                    "dirichlet_alpha": alpha,
                    "method": competitor["method"],
                    "factor": competitor.get("factor", np.nan),
                    "effectiveness": competitor["effect_mean"],
                    "dominates_prompt_probability": count / draws,
                    "equal_weight_side_effect_minus_prompt": equal_difference,
                })

    summary = pd.DataFrame(summary_rows)
    competitors = pd.DataFrame(competitor_rows)
    if not competitors.empty:
        competitors = competitors.sort_values(
            ["comparison", "dirichlet_alpha", "dominates_prompt_probability"],
            ascending=[True, True, False],
        ).reset_index(drop=True)

    print(
        f"Metric-weighting robustness: {len(indicator_names)} normalized "
        f"side-effect indicators; {draws:,} draws per condition; "
        f"base seed {seed}."
    )
    all_points = summary[
        summary["comparison"].eq("all_nonzero_factor_points")
    ]
    print("All complete nonzero-factor method points")
    _display_table(_metric_weighting_display_table(all_points))
    selected_points = summary[
        summary["comparison"].eq("overall_selected_factor")
    ]
    print("Overall-selected factor aggregates")
    _display_table(_metric_weighting_display_table(selected_points))
    return {
        "summary": summary,
        "competitors": competitors,
        "indicator_names": indicator_names,
        "curve_points": curve,
        "selected_points": best,
    }


def _generalization_coverage(samples):
    """Check aligned prompts/languages, and report actual evaluation sizes."""
    records = []
    for method, data in samples.groupby("method"):
        language_sets, counts = [], []
        for concept, group in data.groupby("concept_id"):
            by_language = {str(language): set(part["source_input_id"])
                           for language, part in group.groupby("augmenter")}
            if "identity" not in by_language or len(by_language) < 2:
                raise ValueError(f"Missing ID/OOD prompts: {method}/{concept}")
            reference = by_language["identity"]
            if any(ids != reference for ids in by_language.values()):
                raise ValueError(f"Unaligned ID/OOD prompt panels: {method}/{concept}")
            language_sets.append(set(by_language))
            counts.append(len(reference))
        if any(languages != language_sets[0] for languages in language_sets):
            raise ValueError(f"Different OOD languages across concepts: {method}")
        records.append(dict(
            method=method, prompts_per_concept_min=min(counts),
            prompts_per_concept_max=max(counts),
            ood_languages=len(language_sets[0]) - 1,
        ))
    return pd.DataFrame(records)


def _load_generalization_concept_samples() -> pd.DataFrame:
    frames = []
    required = {
        "method", "concept_id", "factor", "augmenter",
        "source_input_id", "raw_steered_concept_score",
        "raw_baseline_concept_score", "raw_steered_instruction_score",
        "raw_baseline_instruction_score", "raw_steered_fluency_score",
        "raw_baseline_fluency_score", "raw_steered_overall_score",
        "raw_baseline_overall_score",
    }
    for metrics_path in _final_metric_paths("prompt_generalization"):
        samples_path = metrics_path.parent / "samples.parquet"
        if not samples_path.is_file():
            warnings.warn(f"Missing prompt-generalization samples: {samples_path}")
            continue
        frame = pd.read_parquet(samples_path)
        missing = sorted(required.difference(frame.columns))
        if missing:
            warnings.warn(
                f"Skipping prompt-generalization samples without {missing}: {samples_path}"
            )
            continue
        frame = frame.copy()
        frame["_source_path"] = str(samples_path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    data = _selected_methods(pd.concat(frames, ignore_index=True, sort=False))
    score_columns = [
        f"raw_{variant}_{part}_score"
        for part in ("concept", "instruction", "fluency", "overall")
        for variant in ("baseline", "steered")
    ]
    for column in ("concept_id", "factor", *score_columns):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["concept_id", "factor", *score_columns])
    for part in ("concept", "instruction", "fluency", "overall"):
        data[f"{part}_effect"] = (
            data[f"raw_steered_{part}_score"]
            - data[f"raw_baseline_{part}_score"]
        )
    identity = [
        "method", "concept_id", "factor", "augmenter", "source_input_id"
    ]
    return data.drop_duplicates(identity, keep="last").reset_index(drop=True)


def analyze_generalization_retention(scope: str = "overall"):
    samples = _load_generalization_concept_samples()
    if samples.empty:
        print("[waiting] No finalized prompt-generalization concept samples.")
        return None

    coverage = _generalization_coverage(samples)
    factor_counts = samples.groupby("method", observed=True)["factor"].nunique()
    invalid = factor_counts[factor_counts != 1]
    if not invalid.empty:
        raise ValueError(
            "Prompt generalization must contain exactly one selected factor per method; "
            f"found {invalid.to_dict()}."
        )

    keys = ["method", "concept_id", "factor"]
    identity = (
        samples[samples["augmenter"].astype(str) == "identity"]
        .groupby(keys, as_index=False, observed=True)
        .agg(
            id_concept_effect=("concept_effect", "mean"),
            id_overall_effect=("overall_effect", "mean"),
        )
    )
    if scope == "overall":
        ood = samples[samples["augmenter"].astype(str) != "identity"].copy()
    else:
        ood = samples[samples["augmenter"].astype(str) == str(scope)].copy()
    if ood.empty:
        print(f"[waiting] No prompt-generalization scope {scope!r}.")
        return None
    # First average prompts within a language, then languages, so every
    # configured OOD transformation has equal weight.
    ood = (
        ood.groupby([*keys, "augmenter"], as_index=False, observed=True)
        .agg(
            language_concept_effect=("concept_effect", "mean"),
            language_overall_effect=("overall_effect", "mean"),
        )
        .groupby(keys, as_index=False, observed=True)
        .agg(
            ood_concept_effect=("language_concept_effect", "mean"),
            ood_overall_effect=("language_overall_effect", "mean"),
        )
    )
    per_concept = identity.merge(ood, on=keys, how="inner", validate="one_to_one")
    summary = (
        per_concept.groupby("method", as_index=False, observed=True)
        .agg(
            factor=("factor", "first"),
            id_concept_effect=("id_concept_effect", "mean"),
            ood_concept_effect=("ood_concept_effect", "mean"),
            id_overall_effect=("id_overall_effect", "mean"),
            ood_overall_effect=("ood_overall_effect", "mean"),
            n_concepts=("concept_id", "nunique"),
        )
        .sort_values("ood_concept_effect", ascending=False)
    )
    summary = summary.merge(coverage, on="method", how="left", validate="one_to_one")
    summary["concept_retention_valid"] = (
        summary["id_concept_effect"] >= GENERALIZATION_MIN_ID_EFFECT
    )
    summary["overall_retention_valid"] = (
        summary["id_overall_effect"] >= GENERALIZATION_MIN_ID_EFFECT
    )
    summary["concept_retention"] = np.where(
        summary["concept_retention_valid"],
        summary["ood_concept_effect"] / summary["id_concept_effect"],
        np.nan,
    )
    summary["overall_harmonic_retention"] = np.where(
        summary["overall_retention_valid"],
        summary["ood_overall_effect"] / summary["id_overall_effect"],
        np.nan,
    )
    print(
        f"Prompt generalization ({scope}) — one Overall-selected global factor per "
        f"method; concept and Overall-harmonic retention use their corresponding "
        f"ID-effect threshold >= {GENERALIZATION_MIN_ID_EFFECT}."
    )
    components = generalization_components(samples, scope)
    summary = summary.merge(
        components, on="method", how="left", validate="one_to_one",
        suffixes=("", "_component"),
    )
    for prefix in ("id", "ood"):
        for part in ("concept", "overall"):
            column = f"{prefix}_{part}_effect"
            component_column = f"{column}_component"
            if not np.allclose(
                summary[column], summary[component_column], equal_nan=True,
            ):
                raise ValueError(
                    f"Inconsistent generalization aggregation for {column}."
                )
            summary = summary.drop(columns=component_column)
    summary = summary.sort_values("ood_overall_effect", ascending=False)
    transfer_summary = generalization_transfer_summary(summary)
    print("Across-method ID-to-OOD transfer summary (all values are effects)")
    _display_table(transfer_summary.round(4))
    print(
        "Complete per-method generalization results: absolute baseline and "
        "steered scores, baseline-relative effects, and transfer values"
    )
    table = generalization_display_table(summary)
    _display_table(table.round(4), all_columns=True)

    plotted = summary[
        summary["concept_retention_valid"]
        & ~summary["method"].astype(str).isin(PAPER_EXCLUDED_METHODS)
    ].copy()
    methods = _paper_methods(plotted.get("method", []))
    figure, axis = plt.subplots(figsize=PAPER_MAIN_SIZE, dpi=PLOT_DPI)
    for row in plotted.itertuples(index=False):
        style = _method_style(row.method)
        axis.scatter(
            row.id_concept_effect, row.concept_retention,
            color=style["color"], marker=style["marker"],
            s=0.36 * _method_scatter_size(row.method),
            edgecolor="white", linewidth=0.5, zorder=4,
        )
    axis.set_xlabel("ID Concept Expression Improvement ↑")
    axis.set_ylabel("OOD / ID Concept Retention ↑")
    axis.grid(alpha=0.60, linewidth=0.45, linestyle="--")
    axis.margins(x=0.08, y=0.10)
    _paper_legend(figure, methods, best=False, appendix=False)
    figure.tight_layout(rect=(0, 0.145, 1, 1), pad=0.35)
    _paper_save(
        figure, "main",
        f"prompt_generalization__concept_retention__{scope}__filtered",
    )

    plotted_overall = summary[
        summary["overall_retention_valid"]
        & ~summary["method"].astype(str).isin(PAPER_EXCLUDED_METHODS)
    ].copy()
    overall_methods = _paper_methods(plotted_overall.get("method", []))
    figure, axis = plt.subplots(figsize=PAPER_MAIN_SIZE, dpi=PLOT_DPI)
    for row in plotted_overall.itertuples(index=False):
        style = _method_style(row.method)
        axis.scatter(
            row.id_overall_effect, row.overall_harmonic_retention,
            color=style["color"], marker=style["marker"],
            s=0.36 * _method_scatter_size(row.method),
            edgecolor="white", linewidth=0.5, zorder=4,
        )
    axis.set_xlabel("ID Overall Harmonic Effect ↑")
    axis.set_ylabel("OOD / ID Overall Retention ↑")
    axis.grid(alpha=0.60, linewidth=0.45, linestyle="--")
    axis.margins(x=0.08, y=0.10)
    _paper_legend(figure, overall_methods, best=False, appendix=False)
    figure.tight_layout(rect=(0, 0.145, 1, 1), pad=0.35)
    _paper_save(
        figure, "main",
        f"prompt_generalization__overall_harmonic_retention__{scope}__filtered",
    )
    return {
        "samples": samples,
        "per_concept": per_concept,
        "summary": summary,
        "transfer_summary": transfer_summary,
        "table": table,
    }


def _paper_best_relation(relation: pd.DataFrame, metric: str) -> pd.DataFrame:
    selected = _paper_best_factors()[["method", "_factor_key"]]
    if relation.empty or selected.empty:
        return pd.DataFrame()
    rows = relation.copy()
    rows["_factor_key"] = pd.to_numeric(rows["factor"], errors="coerce").round(8)
    rows = rows.merge(
        selected, on=["method", "_factor_key"],
        how="inner", validate="many_to_one",
    )
    return (
        rows.groupby("method", as_index=False, observed=True)
        .agg(
            concept_score=("concept_score", "mean"),
            metric_mean=(metric, "mean"),
            n_concepts=("concept_id", "nunique"),
        )
    )


def _plot_metric(
    data: pd.DataFrame,
    factor_summary: pd.DataFrame,
    relation: pd.DataFrame,
    relation_summary: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
    slug: str,
):
    """Generalization main figure without its factor panel or confidence band."""
    best_metric = _paper_best_metric(data, metric)
    _PAPER_METRIC_REGISTRY[(OUTPUT_KEY, slug)] = {
        "slug": slug,
        "metric": metric,
        "absolute_label": _paper_absolute_label(metric, title, True),
        "side_effect_label": "",
        "factor_summary": factor_summary,
        "factor_ci_rows": data[["method", "concept_id", "factor", metric]].rename(columns={metric: "value"}),
        "tradeoff": pd.DataFrame(),
        "best_tradeoff": pd.DataFrame(),
        "best_metric": best_metric,
    }
    selected = relation_summary[
        relation_summary["method"].astype(str).isin(_paper_main_methods())
    ].copy()
    methods = _paper_methods(selected.get("method", []))
    figure, axis = plt.subplots(figsize=PAPER_MAIN_SIZE, dpi=PLOT_DPI)
    for method in methods:
        group = selected[selected["method"].astype(str) == method].sort_values(
            "concept_score"
        )
        axis.plot(
            group["concept_score"], group["metric_mean"],
            **_method_line_kwargs(method, label=False),
        )
    axis.set_xlabel("Concept Expression ↑")
    axis.set_ylabel("Generalization Retention ↑")
    axis.grid(alpha=0.60, linewidth=0.45, linestyle="--")
    _paper_legend(figure, methods, best=False, appendix=False)
    figure.tight_layout(rect=(0, 0.145, 1, 1), pad=0.35)
    _paper_save(figure, "main", f"{slug}__filtered")


def analyze_sample_efficiency():
    runs = _study_concept_improvements(align_main_prompt_panel=True)
    if runs.empty:
        print("[waiting] No completed study best_factor results.")
        return None
    candidates = runs[
        runs["efficiency"].fillna(False) | runs["factor_reference"].fillna(False)
    ].copy()
    if candidates.empty:
        print("[waiting] No completed sample-efficiency variants.")
        return None
    summary = (
        candidates.groupby(["method", "train_examples"], as_index=False)
        .agg(
            score=("selected_concept_improvement", "mean"),
            score_std=("selected_concept_improvement", "std"),
            runs=("selected_concept_improvement", "size"),
        )
        .sort_values(["method", "train_examples"])
    )
    summary["score_std"] = summary["score_std"].fillna(0.0)
    absolute_summary = summary.copy()
    eligible, references = _eligible_study_methods(runs)
    excluded = references[~references["eligible"]].copy()
    if not excluded.empty:
        print("Excluded by full-data concept improvement threshold:")
        _display_table(excluded.round(4))
    selected = candidates[candidates["method"].astype(str).isin(eligible)].copy()
    summary = summary[summary["method"].astype(str).isin(eligible)].copy()
    summary = summary.merge(
        references[["method", "reference_concept_improvement"]],
        on="method", how="left",
    )
    summary["relative_improvement_pct"] = (
        100.0 * summary["score"] / summary["reference_concept_improvement"]
    )
    _display_table(summary.round(4))
    _plot_sample_efficiency_panels(
        summary,
        summary,
        absolute_ylabel="Concept Expression Improvement ↑",
        relative_ylabel="Relative Improvement (%) ↑",
        section="main",
        slug="study__sample_efficiency",
    )
    _plot_sample_efficiency_panels(
        absolute_summary,
        summary,
        absolute_ylabel="Concept Expression Improvement ↑",
        relative_ylabel="Relative Improvement (%) ↑",
        section="appendix",
        slug="study__sample_efficiency__absolute_all_methods",
    )

    # Diagnostic companion: retain the Overall-selected factor and apply
    # the same threshold to its own full-data Overall-effect denominator.
    overall_eligible, overall_references = _eligible_study_methods(
        runs,
        value_column="selected_overall_improvement",
        reference_column="reference_overall_improvement",
    )
    excluded_overall = overall_references[~overall_references["eligible"]].copy()
    if not excluded_overall.empty:
        print("Excluded by full-data Overall improvement threshold:")
        _display_table(excluded_overall.round(4))
    overall_absolute_summary = (
        candidates.groupby(["method", "train_examples"], as_index=False)
        .agg(
            score=("selected_overall_improvement", "mean"),
            score_std=("selected_overall_improvement", "std"),
            runs=("selected_overall_improvement", "size"),
        )
        .sort_values(["method", "train_examples"])
    )
    overall_absolute_summary = overall_absolute_summary.dropna(
        subset=["score"]
    ).reset_index(drop=True)
    overall_absolute_summary["score_std"] = (
        overall_absolute_summary["score_std"].fillna(0.0)
    )
    overall_summary = overall_absolute_summary[
        overall_absolute_summary["method"].astype(str).isin(overall_eligible)
    ].copy()
    overall_summary["score_std"] = overall_summary["score_std"].fillna(0.0)
    overall_summary = overall_summary.merge(
        overall_references[["method", "reference_overall_improvement"]],
        on="method", how="left", validate="many_to_one"
    )
    overall_summary["relative_improvement_pct"] = (
        100.0 * overall_summary["score"]
        / overall_summary["reference_overall_improvement"]
    )
    print("Sample efficiency — Overall harmonic-effect improvement")
    _display_table(overall_summary.round(4))
    _plot_sample_efficiency_panels(
        overall_summary,
        overall_summary,
        absolute_ylabel="Overall Harmonic Effect ↑",
        relative_ylabel="Relative Overall Effect (%) ↑",
        section="main",
        slug="study__sample_efficiency__overall",
    )
    _plot_sample_efficiency_panels(
        overall_absolute_summary,
        overall_summary,
        absolute_ylabel="Overall Harmonic Effect ↑",
        relative_ylabel="Relative Overall Effect (%) ↑",
        section="appendix",
        slug="study__sample_efficiency__overall__absolute_all_methods",
    )
    return {
        "runs": selected, "summary": summary,
        "absolute_summary": absolute_summary,
        "overall_summary": overall_summary,
        "overall_absolute_summary": overall_absolute_summary,
        "references": references,
        "overall_references": overall_references,
    }


def _sample_sensitivity_concept_summary(candidates, value_column):
    """Compute seed SD per concept first, then average concept SDs."""
    canonical = OUTPUT_DIR / "studies/metrics.parquet"
    if not canonical.is_file():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    metrics = pd.read_parquet(canonical)
    required = {
        "source_evaluator", "study_run_id", "method", "concept_id",
        "factor", value_column,
    }
    if not required.issubset(metrics.columns):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    judge = metrics[metrics["source_evaluator"].eq("study_lm_judge")].copy()
    judge["factor"] = pd.to_numeric(judge["factor"], errors="coerce")
    judge[value_column] = pd.to_numeric(judge[value_column], errors="coerce")
    judge = judge.dropna(subset=["concept_id", "factor", value_column])
    records = []
    for row in candidates.drop_duplicates(["study_run_id", "method"]).itertuples(index=False):
        panel = judge[
            judge["study_run_id"].astype(str).eq(str(row.study_run_id))
            & judge["method"].astype(str).eq(str(row.method))
        ].copy()
        paired = paired_concept_effects(panel, row.factor, row.baseline_factor, value_column)
        for score in paired.itertuples(index=False):
            records.append({
                "method": str(row.method), "train_examples": int(row.train_examples),
                "subset_seed": int(row.subset_seed), "study_run_id": str(row.study_run_id),
                "concept_id": int(score.concept_id), "score": float(score.score),
            })
    rows = pd.DataFrame(records)
    if rows.empty:
        return rows, pd.DataFrame(), pd.DataFrame()
    by_concept, summary = summarize_sensitivity(rows)
    return rows, by_concept, summary


def _plot_sample_sensitivity(summary, train_sizes, xlabel, slug):
    figure, axes = plt.subplots(
        1, len(train_sizes),
        figsize=(6.5, 3.85), dpi=PLOT_DPI, squeeze=False,
    )
    for axis, train_examples in zip(axes.flat, train_sizes):
        group = summary[summary["train_examples"] == train_examples].sort_values("score_mean")
        for row in group.itertuples(index=False):
            style = _method_style(row.method)
            axis.errorbar(
                [row.score_mean], [method_display_name(row.method)], xerr=[row.score_std],
                fmt=style["marker"], color=style["color"],
                ecolor=style["color"], capsize=2.0, markersize=4.7,
                markeredgecolor="white", markeredgewidth=0.5,
            )
        axis.text(
            0.03, 0.97, f"{int(train_examples)} examples/concept",
            transform=axis.transAxes, ha="left", va="top", fontsize=8,
        )
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", alpha=0.60, linewidth=0.45, linestyle="--")
    figure.tight_layout(pad=0.35)
    _paper_save(figure, "main", slug)


def analyze_sample_sensitivity():
    runs = _study_concept_improvements()
    if runs.empty:
        print("[waiting] No completed study best_factor results.")
        return None
    candidates = runs[runs["sensitivity"].fillna(False)].copy()
    if candidates.empty:
        print("[waiting] No completed sample-sensitivity variants.")
        return None
    eligible, references = _eligible_study_methods(runs)
    selected = candidates[candidates["method"].astype(str).isin(eligible)].copy()
    concept_rows, concept_by_concept, summary = _sample_sensitivity_concept_summary(
        candidates, "relevance_concept_ratings"
    )
    if summary.empty:
        print("[waiting] No per-concept sample-sensitivity judge metrics.")
        return None
    summary = summary.merge(
        references[["method", "reference_concept_improvement", "eligible"]]
        .rename(columns={"eligible": "plot_eligible"}),
        on="method", how="left", validate="many_to_one",
    )
    plot_summary = summary[summary["plot_eligible"].fillna(False)].copy()
    print("Sample sensitivity — mean per-concept subset-seed SD (Concept)")
    _display_table(summary.round(4))
    train_sizes = sorted(summary["train_examples"].unique())
    _plot_sample_sensitivity(
        plot_summary, train_sizes,
        "Concept Improvement (mean ± mean concept seed SD)",
        "study__sample_sensitivity",
    )

    overall_eligible, overall_references = _eligible_study_methods(
        runs, value_column="selected_overall_improvement",
        reference_column="reference_overall_improvement",
    )
    overall_rows, overall_by_concept, overall_summary = (
        _sample_sensitivity_concept_summary(candidates, "lm_judge_rating")
    )
    overall_summary = overall_summary.merge(
        overall_references[["method", "reference_overall_improvement", "eligible"]]
        .rename(columns={"eligible": "plot_eligible"}),
        on="method", how="left", validate="many_to_one",
    )
    overall_plot_summary = overall_summary[
        overall_summary["plot_eligible"].fillna(False)
    ].copy()
    print("Sample sensitivity — mean per-concept subset-seed SD (Overall)")
    _display_table(overall_summary.round(4))
    _plot_sample_sensitivity(
        overall_plot_summary, train_sizes,
        "Overall Improvement (mean ± mean concept seed SD)",
        "study__sample_sensitivity__overall",
    )
    return {
        "runs": selected, "summary": summary,
        "concept_rows": concept_rows, "concept_by_concept": concept_by_concept,
        "plot_summary": plot_summary,
        "overall_rows": overall_rows, "overall_by_concept": overall_by_concept,
        "overall_summary": overall_summary,
        "overall_plot_summary": overall_plot_summary,
        "references": references,
        "overall_references": overall_references,
    }


def _comparison_panel_title(output_key: str) -> str:
    """Turn an output key such as 2b/l20 into a paper-facing panel title."""
    model, _, layer = str(output_key).partition("/")
    model_label = {"2b": "Gemma-2-2B", "9b": "Gemma-2-9B"}.get(
        model.lower(), model
    )
    return f"{model_label} · {layer.upper()}" if layer else model_label


def _save_comparison_figure(figure, slug: str) -> Path:
    """Save a cross-output figure outside either model's output subtree."""
    directory = Path(PAPER_FIGURE_BASE) / "comparison" / "main"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{slug}.pdf"
    figure.savefig(
        path, format="pdf", dpi=300, bbox_inches="tight",
        pad_inches=0.025, facecolor="white",
    )
    print(f"saved: {path}")
    if DISPLAY_FIGURES_IN_NOTEBOOK:
        plt.show()
    plt.close(figure)
    return path


def analyze_composite_side_effect_comparison(composite_results=None):
    """Render configured model outputs side by side with shared scales/legend."""
    global _MULTI_OUTPUT_ACTIVE
    if len(OUTPUT_DIRS) < 2:
        warnings.warn(
            "Composite comparison needs at least two configured output directories.",
            stacklevel=2,
        )
        return None

    previous_output = OUTPUT_DIR
    panels = {}
    _MULTI_OUTPUT_ACTIVE = True
    try:
        for output_dir in OUTPUT_DIRS:
            output_key = _activate_output_dir(output_dir)
            result = (
                composite_results.get(output_key)
                if isinstance(composite_results, dict)
                else None
            )
            curves = result.get("curves", pd.DataFrame()) if result else pd.DataFrame()
            if curves.empty:
                try:
                    _, _, entry = _paper_build_composite()
                    curves = entry["tradeoff"]
                except ValueError as error:
                    warnings.warn(
                        f"Skipping composite panel {output_key}: {error}",
                        stacklevel=2,
                    )
                    continue
            methods = _paper_main_methods()
            curves = curves[curves["method"].astype(str).isin(methods)].copy()
            if curves.empty:
                warnings.warn(
                    f"Skipping empty composite panel {output_key}.", stacklevel=2
                )
                continue
            panels[output_key] = {
                "curves": curves,
                "best_factor_cutoffs": _paper_best_factor_cutoffs(),
            }
    finally:
        _MULTI_OUTPUT_ACTIVE = False
        _activate_output_dir(previous_output)

    if len(panels) < 2:
        warnings.warn(
            "Fewer than two complete composite panels are available; no "
            "comparison figure was generated.",
            stacklevel=2,
        )
        return None

    panel_items = list(panels.items())
    figure, axes = plt.subplots(
        1, len(panel_items), figsize=(9.5, 3.65), dpi=PLOT_DPI,
        sharex=True, sharey=True, squeeze=False,
    )
    all_methods = set()
    for axis, (output_key, panel) in zip(axes.flat, panel_items):
        rows = panel["curves"]
        best_factor_cutoffs = panel["best_factor_cutoffs"]
        methods = _paper_methods(rows["method"])
        all_methods.update(methods)
        for method in methods:
            group = rows[rows["method"].astype(str) == method].sort_values("factor")
            group = _truncate_at_first_left_turn(
                group, best_factor=best_factor_cutoffs.get(method)
            )
            nonzero = group[~np.isclose(group["factor"], 0.0)]
            style = _method_style(method)
            if len(nonzero) <= 1:
                if not nonzero.empty:
                    point = nonzero.iloc[-1]
                    axis.scatter(
                        point["effect_mean"], point["side_effect_mean"],
                        color=style["color"], marker=style["marker"], s=30,
                        edgecolor="white", linewidth=0.5, zorder=5,
                    )
                continue
            axis.plot(
                group["effect_mean"], group["side_effect_mean"],
                **_method_line_kwargs(method, label=False),
            )
        axis.axhline(0, color="#374151", lw=0.65, alpha=0.32)
        axis.axvline(0, color="#374151", lw=0.65, alpha=0.32)
        axis.set_title(_comparison_panel_title(output_key))
        axis.set_xlabel("Concept Expression ↑")
        axis.grid(alpha=0.60, linewidth=0.45, linestyle="--")
        axis.margins(x=0.035, y=0.055)
        _paper_better(axis)
    axes.flat[0].set_ylabel("Mean Side Effect ↓")

    methods = _paper_methods(all_methods)
    _inside_upper_left_method_legend(
        axes.flat[0], methods, fontsize=9.2, rows=5, anchor_x=0.075,
        width_fraction=0.87, frameon=False,
    )
    figure.tight_layout(pad=0.35, w_pad=0.15)
    figure.subplots_adjust(wspace=0.06)
    path = _save_comparison_figure(
        figure, "composite_side_effect__2b_9b_shared_legend"
    )
    print(
        f"Composite comparison: {len(panel_items)} panels, "
        f"{len(methods)} methods in the shared legend; shared x/y scales."
    )
    return {"panels": panels, "methods": methods, "path": path}


SIDE_EFFECT_CAPABILITY_PANELS = (
    {
        "title": "MMLU",
        "panel_title": "Knowledge (MMLU)",
        "evaluator": "mmlu",
        "metric": "mmlu_accuracy",
        "higher_is_better": True,
        "scale": 1.0,
    },
    {
        "title": "MATH",
        "panel_title": "Math Reasoning (MATH)",
        "evaluator": "math",
        "metric": "math_accuracy",
        "higher_is_better": True,
        "scale": 1.0,
    },
    {
        "title": "IFEval Prompt Strict",
        "panel_title": "Instruction Following (IFEval)",
        "evaluator": "ifeval",
        "metric": "ifeval_prompt_strict_accuracy",
        "higher_is_better": True,
        "scale": 1.0,
    },
    {
        "title": "JBB Attack Success",
        "panel_title": "Jailbreak Safety (JBB)",
        "evaluator": "jailbreakbench",
        "metric": "attack_success_rate",
        "higher_is_better": False,
        "scale": 1.0,
        "filters": {"jbb_split": "harmful"},
    },
    {
        "title": "SuperGLUE",
        "panel_title": "Language Understanding (SuperGLUE)",
        "evaluator": "superglue",
        "metric": "superglue_score",
        "higher_is_better": True,
        "scale": 1.0,
    },
    {
        "title": "TruthfulQA",
        "panel_title": "Truthfulness (TruthfulQA)",
        "evaluator": "truthfulqa",
        "metric": "truthfulqa_binary_accuracy",
        "higher_is_better": True,
        "scale": 1.0,
    },
)


def _side_effect_capability_best_points(spec: dict) -> pd.DataFrame:
    """Return normalized best-factor operating points for one capability."""
    data = _apply_filters(load_metric(spec["evaluator"]), spec.get("filters"))
    metric = spec["metric"]
    if data.empty or metric not in data:
        return pd.DataFrame()
    data = data.copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.dropna(subset=["method", "concept_id", "factor", metric])
    paired = _effect_side_effect_rows(
        data, metric, higher_is_better=spec["higher_is_better"]
    )
    points = _paper_best_tradeoff(paired)
    if points.empty:
        return points
    points = points.copy()
    # Both axes use percentage points of their natural metric ranges. Concept
    # relevance ratings span [0, 2]; the side-effect scales are declared above.
    points["effect_percent"] = 100.0 * points["effect_mean"] / 2.0
    points["side_effect_percent"] = (
        100.0 * points["side_effect_mean"] / float(spec["scale"])
    )
    points["capability"] = spec["title"]
    return points


def analyze_side_effect_capability_panels(*, main_output_keys=("9b/l20",)):
    """Plot six metric-wise trade-offs at each method's global factor."""
    panels = []
    for spec in SIDE_EFFECT_CAPABILITY_PANELS:
        points = _side_effect_capability_best_points(spec)
        if points.empty:
            warnings.warn(
                f"Skipping unavailable metric panel {spec['title']} for "
                f"{OUTPUT_KEY}.",
                stacklevel=2,
            )
            continue
        expected_methods = _paper_main_methods()
        points = points[
            points["method"].astype(str).isin(expected_methods)
        ].copy()
        if points.empty:
            warnings.warn(
                f"Skipping empty metric panel {spec['title']} "
                f"for {OUTPUT_KEY}.",
                stacklevel=2,
            )
            continue
        missing_methods = sorted(
            expected_methods.difference(points["method"].astype(str)),
            key=_method_style_index,
        )
        if missing_methods:
            warnings.warn(
                f"Metric panel {spec['title']} for {OUTPUT_KEY} skipped "
                f"methods without complete best-factor results: "
                f"{[_paper_method_label(method) for method in missing_methods]}",
                stacklevel=2,
            )
        panels.append((spec, points))

    if not panels:
        warnings.warn(
            f"No best-factor metric panels are available for {OUTPUT_KEY}.",
            stacklevel=2,
        )
        return None

    figure, axes = plt.subplots(
        2, 3, figsize=(7.15, 4.05), dpi=PLOT_DPI,
        sharex=True, squeeze=False,
    )
    all_methods = set()
    for axis, (spec, points) in zip(axes.flat, panels):
        methods = _paper_methods(points["method"])
        all_methods.update(methods)
        for method in methods:
            point = points[points["method"].astype(str).eq(method)].iloc[-1]
            style = _method_style(method)
            axis.scatter(
                point["effect_percent"], point["side_effect_percent"],
                color=style["color"], marker=style["marker"],
                s=0.40 * _method_scatter_size(method), edgecolor="white",
                linewidth=0.5, zorder=4,
            )
        axis.axhline(0, color="#374151", lw=0.55, alpha=0.34)
        axis.axvline(0, color="#374151", lw=0.55, alpha=0.34)
        axis.set_title(
            spec.get("panel_title", spec["title"]), fontsize=7.6, pad=3.2,
        )
        axis.grid(alpha=0.55, linewidth=0.4, linestyle="--")
        axis.margins(x=0.06, y=0.09)
        axis.tick_params(labelsize=6.6)
        _paper_better(axis)
    for axis in axes.flat[len(panels):]:
        axis.set_visible(False)

    methods = _paper_methods(all_methods)
    _top_method_legend(figure, methods, y=0.96, fontsize=8.2)
    figure.supxlabel("Concept Expression Gain (pp)", fontsize=7.4, y=0.035)
    figure.supylabel("Metric Degradation (pp)", fontsize=7.4, x=0.025)
    figure.tight_layout(
        rect=(0.025, 0.035, 1, 0.855), pad=0.4, w_pad=0.65, h_pad=0.65,
    )

    main_output_keys = {str(value).lower() for value in main_output_keys}
    section = "main" if OUTPUT_KEY.lower() in main_output_keys else "appendix"
    path = _paper_save(
        figure, section, "side_effect_capabilities__best_factor"
    )
    print(
        f"Best-factor metric panels: {len(panels)} metrics, "
        f"{len(methods)} methods; exported to {section}."
    )
    return {
        "panels": {spec["title"]: points for spec, points in panels},
        "methods": methods,
        "section": section,
        "path": path,
    }


_MULTI_OUTPUT_ACTIVE = False


def _for_each_output(function):
    """Run one public analysis entry point independently for every output."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        global _MULTI_OUTPUT_ACTIVE
        if _MULTI_OUTPUT_ACTIVE:
            return function(*args, **kwargs)
        results = {}
        _MULTI_OUTPUT_ACTIVE = True
        try:
            for output_dir in OUTPUT_DIRS:
                output_key = _activate_output_dir(output_dir)
                print(f"\n=== {output_key}: {function.__name__} ===")
                results[output_key] = function(*args, **kwargs)
        finally:
            _MULTI_OUTPUT_ACTIVE = False
        return results
    return wrapped


def configure(**settings):
    """Apply notebook settings, reset caches, and inspect selected outputs."""
    global OUTPUT_DIR, OUTPUT_DIRS, OUTPUT_KEY, OUTPUT_KEYS, _paper_effectiveness
    unknown = set(settings).difference(_CONFIG_KEYS)
    if unknown:
        raise TypeError(f"Unknown analysis settings: {sorted(unknown)}")
    globals().update(settings)
    _PAPER_METRIC_REGISTRY.clear()
    OUTPUT_DIRS = [resolve_output_dir(value) for value in OUTPUT_DIR_VALUES]
    missing_output_dirs = [path for path in OUTPUT_DIRS if not path.is_dir()]
    if missing_output_dirs:
        raise FileNotFoundError(f"Scheduler output directories not found: {missing_output_dirs}")
    if len(set(OUTPUT_DIRS)) != len(OUTPUT_DIRS):
        raise ValueError(f"Duplicate scheduler output directories: {OUTPUT_DIRS}")
    OUTPUT_KEYS = [_output_key(path) for path in OUTPUT_DIRS]
    if len(set(OUTPUT_KEYS)) != len(OUTPUT_KEYS):
        raise ValueError(f"Output-directory labels collide: {OUTPUT_KEYS}")
    OUTPUT_DIR = OUTPUT_DIRS[0]
    OUTPUT_KEY = _activate_output_dir(OUTPUT_DIR)
    print(f"OUTPUT_DIRS = {OUTPUT_DIRS}")
    print(f"METHODS = {METHODS or 'all completed methods'}")
    _paper_effectiveness = {}
    for output_dir in OUTPUT_DIRS:
        output_key = _activate_output_dir(output_dir)
        print(f"[{output_key}] PAPER_FIGURE_ROOT = {PAPER_FIGURE_ROOT}")
        table = _paper_effectiveness_table()
        _paper_effectiveness[output_key] = table
        if table.empty:
            print(
                f"[waiting] No finalized {CONCEPT_SCORE_EVALUATOR}/"
                f"{CONCEPT_SCORE_COLUMN}; main-figure method filtering is disabled."
            )
        else:
            _display_table(table.round(4))


def _best_factor_side_effect_values(*, report_missing: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect method means at each method's global Overall-best factor."""
    selected = _paper_best_factors()[["method", "_factor_key"]]
    if selected.empty:
        return pd.DataFrame(), selected
    frames = []
    for spec in COMPOSITE_SIDE_EFFECT_SPECS:
        data = _apply_filters(load_metric(spec["evaluator"]), spec.get("filters"))
        if data.empty or spec["metric"] not in data:
            if report_missing:
                print(f"[waiting] Missing {spec['evaluator']}/{spec['metric']}.")
            return pd.DataFrame(), selected
        data = data.copy()
        data["_factor_key"] = pd.to_numeric(data["factor"], errors="coerce").round(8)
        data["score"] = pd.to_numeric(data[spec["metric"]], errors="coerce")
        # Preserve the metric's own concept panel; do not silently average a
        # subset when a selected factor is absent from its evaluated grid.
        panel = data[["method", "concept_id"]].drop_duplicates().merge(
            selected, on="method", how="left", validate="many_to_one"
        )
        scores = data[["method", "concept_id", "_factor_key", "score"]]
        matched = panel.merge(scores, on=["method", "concept_id", "_factor_key"],
                              how="left", validate="one_to_one")
        missing = matched.loc[matched["score"].isna(), "method"].unique()
        if report_missing and len(missing):
            print(f"Excluded incomplete {spec['name']} panels: {sorted(missing)}")
        matched = matched[~matched["method"].isin(missing)]
        means = matched.groupby("method", as_index=False).agg(
            score=("score", "mean"), n_concepts=("concept_id", "nunique")
        )
        means["side_effect"] = spec["name"]
        frames.append(means)
    return pd.concat(frames, ignore_index=True), selected


def _complete_rank_methods(values_long: pd.DataFrame) -> set[str]:
    """Return methods with one finite score for every rank indicator."""
    if values_long.empty:
        return set()
    required = {spec["name"] for spec in COMPOSITE_SIDE_EFFECT_SPECS}
    available = (
        values_long.dropna(subset=["score"])
        .groupby("method", observed=True)["side_effect"]
        .agg(lambda values: set(map(str, values)))
    )
    return {str(method) for method, indicators in available.items()
            if indicators == required}


def analyze_best_factor_side_effect_rankings():
    """Rank raw indicator scores on a shared complete cross-model method set."""
    active_output = OUTPUT_DIR
    values_long, selected = _best_factor_side_effect_values(report_missing=True)
    if selected.empty:
        print("[waiting] No Overall-selected best factors.")
        return None
    if values_long.empty:
        return None

    complete_by_output = {OUTPUT_KEY: _complete_rank_methods(values_long)}
    try:
        for output_dir in OUTPUT_DIRS:
            if resolve_output_dir(output_dir) == active_output:
                continue
            output_key = _activate_output_dir(output_dir)
            other_values, _ = _best_factor_side_effect_values(report_missing=False)
            complete_by_output[output_key] = _complete_rank_methods(other_values)
    finally:
        _activate_output_dir(active_output)

    shared_methods = set.intersection(*complete_by_output.values())
    if not shared_methods:
        print("[waiting] No methods have all ten indicators in every configured output.")
        return None
    current_complete = complete_by_output[OUTPUT_KEY]
    output_only = sorted(current_complete.difference(shared_methods))
    if output_only:
        warnings.warn(
            f"Excluded methods unavailable in another configured output: {output_only}",
            stacklevel=2,
        )
    print(
        f"Shared complete method set across {len(complete_by_output)} outputs "
        f"({len(shared_methods)} methods): {sorted(shared_methods)}"
    )
    values_long = values_long[values_long["method"].astype(str).isin(shared_methods)]
    values, ranks, correlation = rank_best_factor_scores(values_long, COMPOSITE_SIDE_EFFECT_SPECS)
    if len(values) < 2:
        print("[waiting] Need at least two methods with all ten indicators.")
        return None
    figure, axis = plt.subplots(figsize=(7.0, 6.1), dpi=PLOT_DPI)
    names = list(values.columns)
    image = axis.imshow(correlation, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axis.set_xticks(np.arange(len(names)), labels=names, rotation=36, ha="right")
    axis.set_yticks(np.arange(len(names)), labels=names)
    for row in range(len(names)):
        for col in range(len(names)):
            value = correlation.iloc[row, col]
            axis.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=6.6,
                      color="white" if abs(value) >= 0.55 else "black")
    figure.colorbar(image, ax=axis, fraction=0.040, pad=0.025).set_label("Spearman rank correlation")
    figure.tight_layout(pad=0.35)
    _paper_save(figure, "main", "side_effect_rank_spearman__best_factor")
    _display_table(values.round(4))
    _display_table(ranks.round(2))
    return {"values_long": values_long, "values": values, "ranks": ranks,
            "spearman": correlation, "selected_factors": selected}


def analyze_best_factor_side_effect_rank_comparison(rank_results=None):
    """Render configured outputs' Spearman matrices with one shared colorbar."""
    panels = []
    for output_dir in OUTPUT_DIRS:
        output_key = _output_key(resolve_output_dir(output_dir))
        result = (
            rank_results.get(output_key)
            if isinstance(rank_results, dict)
            else None
        )
        correlation = result.get("spearman") if result else None
        if correlation is None or correlation.empty:
            warnings.warn(
                f"Skipping unavailable Spearman panel {output_key}.",
                stacklevel=2,
            )
            continue
        panels.append((output_key, correlation))

    if len(panels) < 2:
        warnings.warn(
            "Fewer than two Spearman panels are available; no comparison "
            "figure was generated.",
            stacklevel=2,
        )
        return None

    figure = plt.figure(figsize=(12.3, 5.35), dpi=PLOT_DPI)
    grid = figure.add_gridspec(
        1, len(panels) + 1,
        width_ratios=[1.0] * len(panels) + [0.035],
        wspace=0.10,
    )
    axes = []
    for index in range(len(panels)):
        shared_axis = axes[0] if axes else None
        axes.append(figure.add_subplot(grid[0, index], sharex=shared_axis,
                                       sharey=shared_axis))
    colorbar_axis = figure.add_subplot(grid[0, -1])
    image = None
    for panel_index, (axis, (output_key, correlation)) in enumerate(
        zip(axes, panels)
    ):
        names = list(correlation.columns)
        image = axis.imshow(correlation, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        axis.set_xticks(
            np.arange(len(names)), labels=names, rotation=42, ha="right"
        )
        axis.set_yticks(np.arange(len(names)), labels=names)
        axis.tick_params(axis="both", labelsize=6.5)
        if panel_index:
            axis.tick_params(axis="y", labelleft=False)
        axis.set_title(_comparison_panel_title(output_key), fontsize=9.2, pad=5)
        for row in range(len(names)):
            for col in range(len(names)):
                value = correlation.iloc[row, col]
                axis.text(
                    col, row, f"{value:.2f}", ha="center", va="center",
                    fontsize=5.7,
                    color="white" if abs(value) >= 0.55 else "black",
                )

    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Spearman rank correlation", fontsize=8.0)
    colorbar.ax.tick_params(labelsize=6.8)
    figure.subplots_adjust(left=0.13, right=0.94, bottom=0.22, top=0.92)
    path = _save_comparison_figure(
        figure, "side_effect_rank_spearman__2b_9b_shared_colorbar"
    )
    print(
        f"Spearman comparison: {len(panels)} panels with one shared colorbar."
    )
    return {"panels": dict(panels), "path": path}


from .metrics_tables import (
    concept_bootstrap_factor_ci,
    generalization_components,
    generalization_display_table,
    generalization_transfer_summary,
    rank_best_factor_scores,
    resolve_unique_main_sources,
    retain_complete_method_factors,
)


_CONFIG_KEYS = {
    "OUTPUT_DIR_VALUES",
    "METHODS",
    "CONCEPT_SCORE_EVALUATOR",
    "CONCEPT_SCORE_COLUMN",
    "PAPER_RELEVANCE_THRESHOLD",
    "GENERALIZATION_MIN_ID_EFFECT",
    "MIN_REFERENCE_IMPROVEMENT",
    "PAPER_EXCLUDED_METHODS",
    "FULL_TRAIN_EXAMPLES",
    "TRADEOFF_BOOTSTRAP_RESAMPLES",
    "TRADEOFF_BOOTSTRAP_SEED",
    "FACTOR_GRID_BOOTSTRAP_RESAMPLES",
    "FACTOR_GRID_BOOTSTRAP_SEED",
    "METRIC_WEIGHTING_DRAWS",
    "METRIC_WEIGHTING_BATCH_SIZE",
    "METRIC_WEIGHTING_SEED",
    "METRIC_WEIGHTING_DIRICHLET_ALPHAS",
    "COMPOSITE_SIDE_EFFECT_SPECS",
    "PAPER_FIGURE_BASE",
    "PAPER_MAIN_SIZE",
    "PAPER_GRID_COLUMNS",
    "PAPER_GRID_ROW_HEIGHT",
    "DISPLAY_FIGURES_IN_NOTEBOOK",
    "PLOT_DPI",
}


for _analysis_name in ('analyze_metric', 'analyze_global_best_factors', 'analyze_composite_side_effect_curves', 'analyze_composite_side_effect_best_factor', 'analyze_metric_weighting_robustness', 'analyze_best_factor_side_effect_rankings', 'analyze_side_effect_capability_panels', 'analyze_generalization_retention', 'analyze_sample_efficiency', 'analyze_sample_sensitivity', 'render_registered_metric_appendix'):
    globals()[_analysis_name] = _for_each_output(globals()[_analysis_name])
