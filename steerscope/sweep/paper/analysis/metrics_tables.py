"""Pure table calculations for scheduler metric analysis."""

import numpy as np
import pandas as pd


def resolve_unique_main_sources(methods, main):
    """Partition methods into unique main-result sources and source issues."""
    sources_by_method = {}
    issues = []
    for method in dict.fromkeys(map(str, methods)):
        sources = sorted(map(str, main.loc[
            main["method"].astype(str).eq(method), "_source_path"
        ].dropna().unique()))
        if len(sources) == 1:
            sources_by_method[method] = sources[0]
            continue
        issues.append({
            "method": method,
            "status": "missing" if not sources else "ambiguous",
            "source_count": len(sources),
            "sources": sources,
        })
    return sources_by_method, issues


def retain_complete_method_factors(rows, required_indicators):
    """Keep the factor intersection separately within each method."""
    required = frozenset(map(str, required_indicators))
    if not required:
        raise ValueError("At least one required indicator is needed.")
    if rows.empty:
        return rows.copy(), pd.DataFrame(columns=[
            "method", "factor", "n_side_effects", "missing_side_effects",
        ])

    available = (
        rows.groupby(["method", "factor"], as_index=False, observed=True)
        .agg(available_side_effects=(
            "side_effect_name", lambda values: frozenset(map(str, values))
        ))
    )
    available["n_side_effects"] = available["available_side_effects"].map(len)
    available["missing_side_effects"] = available["available_side_effects"].map(
        lambda values: ", ".join(sorted(required.difference(values)))
    )
    is_complete = available["available_side_effects"].map(
        lambda values: values == required
    )
    complete = available.loc[is_complete, ["method", "factor"]]
    incomplete = available.loc[
        ~is_complete,
        ["method", "factor", "n_side_effects", "missing_side_effects"],
    ]
    filtered = rows.merge(
        complete, on=["method", "factor"], how="inner", validate="many_to_one"
    )
    return filtered, incomplete.reset_index(drop=True)


def rank_best_factor_scores(values_long, specs):
    """Rank complete methods by raw scores; rank 1 is best for every metric."""
    names = [spec["name"] for spec in specs]
    values = values_long.pivot(index="method", columns="side_effect", values="score")
    values = values.reindex(columns=names).dropna(how="any")
    ranks = pd.DataFrame(index=values.index)
    for spec in specs:
        ranks[spec["name"]] = values[spec["name"]].rank(
            method="average", ascending=not spec["higher_is_better"]
        )
    correlation = ranks.corr(method="spearman")
    ranks["mean_rank"] = ranks.mean(axis=1)
    ranks = ranks.sort_values("mean_rank", kind="stable")
    return values.loc[ranks.index], ranks, correlation


def generalization_components(samples, scope="overall"):
    """Mean absolute scores and effects: prompts -> languages -> concepts."""
    parts = ("concept", "instruction", "fluency", "overall")
    data = samples.copy()
    columns = []
    for part in parts:
        baseline = f"raw_baseline_{part}_score"
        steered = f"raw_steered_{part}_score"
        effect = f"{part}_effect"
        data[effect] = data[steered] - data[baseline]
        columns.extend([baseline, steered, effect])
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if data[columns].isna().any().any():
        raise ValueError("Missing generalization submetric scores.")
    language = data.groupby(["method", "concept_id", "augmenter"])[columns].mean()
    language = language.reset_index()
    panels = []
    for prefix, mask in (
        ("id", language.augmenter.eq("identity")),
        ("ood", language.augmenter.ne("identity") if scope == "overall"
         else language.augmenter.eq(scope)),
    ):
        panel = language[mask].groupby(["method", "concept_id"])[columns].mean()
        panel = panel.groupby("method").mean()
        panel.columns = [
            f"{prefix}_{part}_{value}"
            for part in parts
            for value in ("baseline", "steered", "effect")
        ]
        panels.append(panel)
    result = pd.concat(panels, axis=1).reset_index()
    for scope in ("id", "ood"):
        for part in ("instruction", "fluency"):
            result[f"{scope}_{part}_degradation"] = -result[f"{scope}_{part}_effect"]
    return result


def generalization_display_table(summary):
    """Show every absolute score, baseline-relative effect, and transfer value."""
    summary = summary.sort_values(
        "overall_harmonic_retention",
        ascending=False,
        na_position="last",
        kind="stable",
    )
    fields = [(("Setup", "", "Factor"), "factor"),
              (("Setup", "", "Concepts"), "n_concepts")]
    for part in ("concept", "instruction", "fluency", "overall"):
        label = part.title()
        id_column = f"id_{part}_effect"
        ood_column = f"ood_{part}_effect"
        delta_column = f"{part}_effect_ood_minus_id"
        ratio_column = f"{part}_effect_ood_over_id"
        summary[delta_column] = summary[ood_column] - summary[id_column]
        summary[ratio_column] = np.where(
            ~np.isclose(summary[id_column], 0.0),
            summary[ood_column] / summary[id_column],
            np.nan,
        )
        fields.extend([
            ((label, "ID", "Baseline"), f"id_{part}_baseline"),
            ((label, "ID", "Steered"), f"id_{part}_steered"),
            ((label, "ID", "Improvement"), id_column),
            ((label, "OOD", "Baseline"), f"ood_{part}_baseline"),
            ((label, "OOD", "Steered"), f"ood_{part}_steered"),
            ((label, "OOD", "Improvement"), ood_column),
            ((label, "Transfer", "OOD−ID Improvement"), delta_column),
            ((label, "Transfer", "OOD/ID Improvement"), ratio_column),
        ])
        if part in ("instruction", "fluency"):
            for scope in ("id", "ood"):
                column = f"{scope}_{part}_degradation"
                summary[column] = -summary[f"{scope}_{part}_effect"]
                fields.append(((label, scope.upper(), "Degradation"), column))
    fields.extend([
        (("Validated Retention", "", "Concept"), "concept_retention"),
        (("Validated Retention", "", "Overall"), "overall_harmonic_retention"),
    ])
    table = summary.set_index("method")[[field for _, field in fields]].copy()
    table.columns = pd.MultiIndex.from_tuples([label for label, _ in fields])
    return table


def generalization_transfer_summary(summary):
    """Summarize across-method ID-to-OOD changes for each effect component."""
    records = []
    for part in ("concept", "instruction", "fluency", "overall"):
        id_values = summary[f"id_{part}_effect"]
        ood_values = summary[f"ood_{part}_effect"]
        difference = ood_values - id_values
        records.append({
            "metric": part.title(),
            "mean_id_effect": id_values.mean(),
            "mean_ood_effect": ood_values.mean(),
            "mean_ood_minus_id": difference.mean(),
            "median_ood_minus_id": difference.median(),
            "methods_ood_ge_id_pct": 100.0 * difference.ge(0).mean(),
            "methods": int(difference.notna().sum()),
        })
    return pd.DataFrame(records).set_index("metric")


def concept_bootstrap_factor_ci(rows, *, resamples=5000, seed=42):
    """Compute pointwise concept-bootstrap intervals, paired across factors and stratified by metric coverage."""
    if resamples < 2:
        raise ValueError("At least two bootstrap replicates are required.")
    columns = ["method", "factor", "ci95_lower", "ci95_upper"]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    data = rows.copy()
    if "component" not in data:
        data["component"] = "metric"
    for col in ("factor", "value"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["method", "concept_id", "factor", "component", "value"]
    )
    records = []
    for method, group in data.groupby("method", sort=True, observed=True):
        panel = group.pivot_table(
            index="concept_id", columns=["factor", "component"],
            values="value", aggfunc="mean", observed=True,
        ).sort_index().sort_index(axis=1)
        values = panel.to_numpy(dtype=float)
        present = np.isfinite(values)
        counts = present.sum(axis=0)
        _, strata = np.unique(present, axis=0, return_inverse=True)
        totals = np.zeros((resamples, values.shape[1]))
        rng = np.random.default_rng(seed)
        for stratum in np.unique(strata):
            positions = np.flatnonzero(strata == stratum)
            block = np.nan_to_num(values[positions], nan=0.0)
            n = len(positions)
            # Multinomial multiplicities are exactly resampling n concepts
            # with replacement; one draw is shared across all panel columns.
            for start in range(0, resamples, 250):
                size = min(250, resamples - start)
                weights = rng.multinomial(n, np.full(n, 1.0 / n), size=size)
                totals[start:start + size] += weights @ block
        means = totals / counts[None, :]
        for factor in panel.columns.get_level_values("factor").unique():
            mask = panel.columns.get_level_values("factor") == factor
            if (counts[mask] < 2).any():
                lower = upper = np.nan
            else:
                lower, upper = np.quantile(means[:, mask].mean(axis=1), [.025, .975])
            records.append(dict(method=method, factor=float(factor),
                                ci95_lower=lower, ci95_upper=upper))
    return pd.DataFrame(records, columns=columns)
