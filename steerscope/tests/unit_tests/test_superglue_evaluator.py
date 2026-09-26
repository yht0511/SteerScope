import pandas as pd
import pytest

from steerscope.evaluators.superglue import (
    SuperGLUEEvaluator,
    evaluate_package_version,
    format_superglue_row,
    normalize_tasks,
    superglue_task_score,
)


def test_superglue_reads_evaluate_distribution_version():
    assert evaluate_package_version()


def test_superglue_boolq_and_copa_templates():
    prompt, choices, answer, identity, _ = format_superglue_row("boolq", pd.Series({
        "idx": 7, "passage": "The sky is blue.", "question": "is the sky blue", "label": 1,
    }))
    assert prompt.endswith("Question: is the sky blue?\nAnswer:")
    assert choices == [" no", " yes"]
    assert answer == 1 and identity == 7

    prompt, choices, answer, _, _ = format_superglue_row("copa", pd.Series({
        "idx": 3, "premise": "The ground was wet.", "question": "cause",
        "choice1": "It rained.", "choice2": "The sun shone.", "label": 0,
    }))
    assert prompt == "The ground was wet because"
    assert choices == [" it rained.", " the sun shone."]
    assert answer == 0


def test_superglue_multirc_sampling_keeps_complete_questions():
    rows = []
    for question in range(3):
        for answer in range(question + 1):
            rows.append({
                "idx": {"paragraph": 0, "question": question, "answer": answer},
                "label": answer % 2,
            })
    sampled = SuperGLUEEvaluator._sample_task("multirc", pd.DataFrame(rows), 2, 42)
    selected = {(value["paragraph"], value["question"]) for value in sampled["idx"]}
    for key in selected:
        expected = sum(
            (value["paragraph"], value["question"]) == key
            for value in pd.DataFrame(rows)["idx"]
        )
        actual = sum((value["paragraph"], value["question"]) == key for value in sampled["idx"])
        assert actual == expected


def test_superglue_record_uses_all_entities_and_gold_answers():
    _, choices, answer, identity, metadata = format_superglue_row("record", pd.Series({
        "idx": {"passage": 2, "query": 4}, "passage": "Story",
        "query": "@placeholder arrived.", "entities": ["Bob", "Alice", "Bob"],
        "answers": ["Bob"],
    }))
    assert choices == ["  - Alice arrived.", "  - Bob arrived."]
    assert answer == 1
    assert identity == "2:4"
    assert metadata["answers"] == ["Bob"]


def test_superglue_task_score_matches_official_composites():
    assert superglue_task_score("boolq", {"accuracy": 0.7}) == 0.7
    assert superglue_task_score("cb", {"accuracy": 0.8, "f1": 0.6}) == pytest.approx(0.7)
    assert superglue_task_score(
        "multirc", {"exact_match": 0.4, "f1_a": 0.8, "f1_m": 0.9}
    ) == pytest.approx(0.6)


def test_superglue_rejects_unknown_or_duplicate_tasks():
    with pytest.raises(ValueError, match="Unknown"):
        normalize_tasks(["boolq", "unknown"])
    with pytest.raises(ValueError, match="duplicates"):
        normalize_tasks(["boolq", "boolq"])
