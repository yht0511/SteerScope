from types import SimpleNamespace

import pandas as pd

from steerscope.evaluators.math import MATHEvaluator, format_math_prompt
from steerscope.evaluators.math_official import is_equiv, last_boxed_only_string, remove_boxed


def test_math_answer_extraction_uses_last_balanced_box():
    text = r"First \boxed{0}; finally \boxed{\frac{1}{2}}."
    assert remove_boxed(last_boxed_only_string(text)) == r"\frac{1}{2}"
    assert last_boxed_only_string("no boxed answer") is None


def test_math_official_equivalence_normalizes_common_forms():
    assert is_equiv("0.5", r"\frac{1}{2}")
    assert is_equiv("x = 3", "3")


def test_math_compute_metrics_groups_factors_and_levels():
    evaluator = object.__new__(MATHEvaluator)
    evaluator.model_name = "Demo"
    data = pd.DataFrame({
        "factor": [0.0, 0.0, 1.0, 1.0],
        "math_level": [1, 2, 1, 2],
        "math_gold_answer": ["2", "3", "2", "3"],
        "Demo_steered_generation": [r"\boxed{2}", "3", r"work \boxed{0}", r"work \boxed{3}"],
    })
    result = MATHEvaluator.compute_metrics(evaluator, data)
    assert result["factor"] == [0.0, 1.0]
    assert result["math_accuracy"] == [0.5, 0.5]
    assert result["math_format_compliance"] == [0.5, 1.0]
    assert result["math_level_1_accuracy"] == [1.0, 0.0]
    assert result["math_level_2_accuracy"] == [0.0, 1.0]


def test_math_prompt_requires_boxed_final_answer():
    prompt = format_math_prompt("What is 1 + 1?")
    assert "Show your reasoning" in prompt
    assert r"\boxed{...}" in prompt
