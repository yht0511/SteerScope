"""Run with: python -m unittest discover -s steerscope/sweep/paper/analysis -p 'test_metrics_tables.py'."""
import unittest

import numpy as np
import pandas as pd

from .metrics_tables import (
    concept_bootstrap_factor_ci,
    generalization_components,
    rank_best_factor_scores,
    resolve_unique_main_sources,
    retain_complete_method_factors,
)


class MetricsTablesTests(unittest.TestCase):
    def test_main_sources_distinguish_missing_and_ambiguous(self):
        main = pd.DataFrame([
            ("complete", "/one"),
            ("ambiguous", "/first"),
            ("ambiguous", "/second"),
        ], columns=["method", "_source_path"])
        sources, issues = resolve_unique_main_sources(
            ["complete", "missing", "ambiguous"], main
        )
        self.assertEqual(sources, {"complete": "/one"})
        self.assertEqual(
            {issue["method"]: issue["status"] for issue in issues},
            {"missing": "missing", "ambiguous": "ambiguous"},
        )

    def test_composite_uses_per_method_factor_intersection(self):
        rows = pd.DataFrame([
            ("a", 0.0, "x"), ("a", 0.0, "y"),
            ("a", 1.0, "x"),
            ("b", 10.0, "x"), ("b", 10.0, "y"),
            ("b", 20.0, "y"),
        ], columns=["method", "factor", "side_effect_name"])
        kept, dropped = retain_complete_method_factors(rows, ["x", "y"])
        self.assertEqual(
            set(map(tuple, kept[["method", "factor"]].drop_duplicates().to_numpy())),
            {("a", 0.0), ("b", 10.0)},
        )
        self.assertEqual(
            set(map(tuple, dropped[["method", "factor"]].to_numpy())),
            {("a", 1.0), ("b", 20.0)},
        )
        self.assertEqual(set(dropped["missing_side_effects"]), {"x", "y"})

    def test_rank_directions_ties_and_complete_methods(self):
        specs = [dict(name="accuracy", higher_is_better=True),
                 dict(name="harm", higher_is_better=False)]
        rows = [("a", "accuracy", .9), ("b", "accuracy", .5),
                ("c", "accuracy", .5), ("incomplete", "accuracy", 1),
                ("a", "harm", .1), ("b", "harm", .4), ("c", "harm", .4)]
        values, ranks, correlation = rank_best_factor_scores(
            pd.DataFrame(rows, columns=["method", "side_effect", "score"]), specs
        )
        self.assertEqual(list(values.index), ["a", "b", "c"])
        self.assertEqual(ranks.loc["a", "accuracy"], 1)
        self.assertEqual(ranks.loc["b", "accuracy"], 2.5)
        self.assertEqual(ranks.loc["b", "harm"], 2.5)
        self.assertEqual(correlation.loc["accuracy", "harm"], 1)

    def test_generalization_equal_language_and_concept_weights(self):
        rows = []
        for concept in [0, 1]:
            for language, count, value in [("identity", 1, .2), ("chinese", 1, .4),
                                           ("korean", 3, .8)]:
                for _ in range(count):
                    row = dict(method="a", concept_id=concept, augmenter=language)
                    for part in ("concept", "instruction", "fluency", "overall"):
                        row[f"raw_baseline_{part}_score"] = 0.0
                        row[f"raw_steered_{part}_score"] = value + concept * .2
                    rows.append(row)
        samples = pd.DataFrame(rows)
        result = generalization_components(samples).iloc[0]
        self.assertAlmostEqual(result.id_overall_steered, .3)
        self.assertAlmostEqual(result.id_overall_effect, .3)
        self.assertAlmostEqual(result.ood_overall_steered, .7)
        self.assertAlmostEqual(result.ood_overall_effect, .7)
        language = generalization_components(samples, "chinese").iloc[0]
        self.assertAlmostEqual(language.ood_instruction_steered, .5)
        self.assertAlmostEqual(language.ood_instruction_effect, .5)
        self.assertAlmostEqual(language.ood_instruction_degradation, -.5)

    def test_generalization_display_keeps_components_and_compact_overall(self):
        from .metrics_tables import generalization_display_table

        summary = pd.DataFrame([
            {
                "method": method, "factor": 1.0, "n_concepts": 50,
                **{f"{scope}_{part}_{value}": 1.0
                   for scope in ("id", "ood")
                   for part in ("concept", "instruction", "fluency", "overall")
                   for value in ("baseline", "steered", "effect")},
                "concept_retention": retention,
                "overall_harmonic_retention": retention,
            }
            for method, retention in (("low", .5), ("invalid", float("nan")),
                                      ("high", .9))
        ])
        table = generalization_display_table(summary)
        self.assertEqual(list(table.index), ["high", "low", "invalid"])
        self.assertEqual(table.columns.nlevels, 3)
        self.assertIn(("Concept", "ID", "Baseline"), table.columns)
        self.assertIn(("Concept", "ID", "Steered"), table.columns)
        self.assertIn(("Concept", "ID", "Improvement"), table.columns)
        self.assertIn(("Concept", "OOD", "Improvement"), table.columns)
        self.assertIn(("Concept", "Transfer", "OOD−ID Improvement"), table.columns)
        self.assertIn(("Overall", "Transfer", "OOD/ID Improvement"), table.columns)
        self.assertIn(("Validated Retention", "", "Overall"), table.columns)

    def test_missing_submetric_is_not_silently_averaged(self):
        row = dict(method="a", concept_id=0, augmenter="identity")
        row.update({f"raw_baseline_{part}_score": 0 for part in
                    ("concept", "instruction", "fluency", "overall")})
        row.update({f"raw_steered_{part}_score": 1 for part in
                    ("concept", "instruction", "fluency", "overall")})
        row["raw_steered_fluency_score"] = float("nan")
        with self.assertRaises(ValueError):
            generalization_components(pd.DataFrame([row]))


class BestFactorSelectionTests(unittest.TestCase):
    def test_global_selection_excludes_factor_zero(self):
        from unittest.mock import patch

        from . import metrics

        rows = pd.DataFrame([
            (method, concept_id, factor, score)
            for method, scores in {
                "weaker": {0.0: 0.9, 1.0: 0.5, 2.0: 0.6},
                "tie": {0.0: 0.2, 1.0: 0.7, 2.0: 0.7},
            }.items()
            for concept_id in (0, 1)
            for factor, score in scores.items()
        ], columns=["method", "concept_id", "factor", "lm_judge_rating"])

        with (
            patch.object(metrics, "_PAPER_BEST_FACTOR_CACHE", None),
            patch.object(metrics, "load_metric", return_value=rows),
        ):
            selected = metrics._paper_best_factors().set_index("method")

        self.assertEqual(selected.loc["weaker", "factor"], 2.0)
        self.assertAlmostEqual(
            selected.loc["weaker", "overall_improvement"], -0.3
        )
        self.assertEqual(selected.loc["weaker", "n_complete_factors"], 2)
        self.assertEqual(selected.loc["tie", "factor"], 1.0)


class ConceptBootstrapTests(unittest.TestCase):
    def test_constant_and_single_concept(self):
        rows = pd.DataFrame([
            ("constant", c, f, 3.0) for c in range(5) for f in (0., 1.)
        ] + [("single", 0, 1., 2.)],
            columns=["method", "concept_id", "factor", "value"])
        ci = concept_bootstrap_factor_ci(rows, resamples=500)
        stable = ci[ci.method.eq("constant")]
        np.testing.assert_allclose(stable[["ci95_lower", "ci95_upper"]], 3.)
        self.assertTrue(ci[ci.method.eq("single")].ci95_lower.isna().all())

    def test_paired_factors_and_duplicate_rows(self):
        rows = pd.DataFrame([
            ("a", c, f, c / 10 + 2 * f) for c in range(10) for f in (0., 1.)
        ], columns=["method", "concept_id", "factor", "value"])
        ci = concept_bootstrap_factor_ci(rows, resamples=1000).set_index("factor")
        np.testing.assert_allclose(
            ci.loc[1., ["ci95_lower", "ci95_upper"]].astype(float)
            - ci.loc[0., ["ci95_lower", "ci95_upper"]].astype(float), 2.)
        duplicated = pd.concat([rows, rows.iloc[[0, 1, 2]]], ignore_index=True)
        pd.testing.assert_frame_equal(ci, concept_bootstrap_factor_ci(
            duplicated.sample(frac=1, random_state=7), resamples=1000
        ).set_index("factor"))

    def test_composite_covariance_and_unequal_panels(self):
        # Opposite shared-concept effects cancel exactly; an independently
        # resampled component bootstrap would incorrectly produce a wide CI.
        rows = pd.DataFrame([
            ("a", c, 1., k, v) for c in range(6)
            for k, v in [("x", float(c)), ("y", 10. - c)]
        ], columns=["method", "concept_id", "factor", "component", "value"])
        ci = concept_bootstrap_factor_ci(rows, resamples=1000)
        np.testing.assert_allclose(ci[["ci95_lower", "ci95_upper"]], 5.)
        # Different panel sizes must not alter the equal-indicator weighting.
        rows = pd.DataFrame([
            ("a", c, 1., "large", 2.) for c in range(10)
        ] + [("a", c, 1., "small", 8.) for c in range(4)],
            columns=["method", "concept_id", "factor", "component", "value"])
        ci = concept_bootstrap_factor_ci(rows, resamples=1000)
        np.testing.assert_allclose(ci[["ci95_lower", "ci95_upper"]], 5.)


if __name__ == "__main__":
    unittest.main()
