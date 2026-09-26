"""Viewer reads exact rows; never mixes factors or substitutes aggregate scores."""
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from steerscope.sweep.paper.analysis.browse_samples import (
    Filters, Node, RowReader, Source, build_index, discover, document,
    matches_row, progress_directory, safe_text, score_lines, sources_for, wrap,
)


def make_node(root, method="FLAS", metric="id_lm_judge", status="complete"):
    path = root / f"methods/{method}/evaluate/runs/{method}/evaluators/{metric}"
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({
        "status": status, "execution_hash": "current",
        "config": {"models": [method]},
    }))
    return Node(path, metric, "main", (method,))


def test_exact_rows_across_row_groups_and_pages(tmp_path):
    node = make_node(tmp_path)
    rows = [{"method": "FLAS", "factor": i % 3, "concept_id": i % 5,
             "input_concept": "概念", "input": f"prompt {i}",
             "FLAS_steered_generation": f"response {i}", "raw_aggregated_ratings": i / 1000}
            for i in range(250)]
    pq.write_table(pa.Table.from_pylist(rows), node.path / "samples.parquet", row_group_size=83)
    filters = Filters(method="flas", factor="1", concept="2")
    index = build_index(discover(tmp_path), filters)
    expected = [r for r in rows if r["factor"] == 1 and r["concept_id"] == 2]
    reader = RowReader()
    actual = [reader.read(*index.locate(i)) for i in range(len(index))]
    assert actual == expected
    for i in reversed(range(len(index))):
        assert reader.read(*index.locate(i)) == expected[i]
    assert len(reader.pages) <= 4
    with pytest.raises(IndexError):
        index.locate(len(index))


@pytest.mark.parametrize("filters,expected", [
    (Filters(method="FLAS", factor="0.3", concept="8"), True),
    (Filters(concept="1"), False),
    (Filters(concept="sports"), True),
    (Filters(concept="7,8", method="APSR,FLAS"), True),
    (Filters(method="FLAS", factor="3"), False),
    (Filters(task="chinese"), True),
    (Filters(task="korean"), False),
])
def test_joint_filters(filters, expected):
    assert matches_row({"method": "FLAS", "factor": 0.30000000000000004,
                        "concept_id": 8, "input_concept": "Sports tournaments",
                        "augmenter": "chinese"}, filters) == expected


def test_pending_hot_storage_does_not_use_stale_final_or_other_hash(tmp_path):
    node = make_node(tmp_path, status="running")
    hot = tmp_path / "hot"
    config = tmp_path / "runtime_configs"
    config.mkdir()
    (config / "hot_data.json").write_text(json.dumps({"progress_root": str(hot), "output_root": str(tmp_path)}))
    root = progress_directory(node)
    assert root == hot / node.path.relative_to(tmp_path) / "progress/current"
    pq.write_table(pa.Table.from_pylist([{"method": "WRONG"}]), node.path / "samples.parquet")
    for name, digest, target in [("good", "current", "FLAS/concept-0"),
                                 ("old", "old", "FLAS/concept-1")]:
        path = root / "judge_pending_inference" / name
        path.mkdir(parents=True)
        (path / "manifest.json").write_text(json.dumps({"status": "complete", "execution_hash": digest, "target_id": target}))
        pq.write_table(pa.Table.from_pylist([{"method": "FLAS", "factor": 1, "concept_id": 0}]), path / "inference.parquet")
    sources = sources_for(node)
    assert len(sources) == 1
    assert sources[0].path.parent.name == "good"
    assert sources[0].kind == "pending"
    assert "未评分" in score_lines({"raw_aggregated_ratings": 2}, sources[0])[0]


def test_aggregate_only_node_not_exposed_as_samples(tmp_path):
    node = make_node(tmp_path, metric="best_factor")
    pq.write_table(pa.Table.from_pylist([{"factor": 1, "score": 0.5}]), node.path / "metrics.parquet")
    assert sources_for(node) == []


def test_saved_scores_no_recalculation_or_baseline_mix(tmp_path):
    source = Source(tmp_path / "samples.parquet", Node(tmp_path, "id_lm_judge", "main", ("FLAS",)), "samples")
    row = {"method": "FLAS", "factor": 2, "concept_id": 3,
           "input": "actual chat template", "original_prompt": "original",
           "FLAS_steered_generation": "steered", "baseline_generation": "baseline",
           "raw_aggregated_ratings": 0.12345, "raw_relevance_concept_ratings": 2,
           "fluency_completions": "Rating: [[1]]"}
    text = "\n".join(document(row, source))
    assert "0.12345" in text
    assert "actual chat template" in text and "original" in text
    assert "OUTPUT [FLAS_steered_generation]" in text
    assert "OUTPUT [baseline_generation]" in text
    assert "Rating: [[1]]" in "\n".join(document(row, source, "judge"))


def test_superglue_not_inventing_per_row_f1(tmp_path):
    source = Source(tmp_path / "samples.parquet", Node(tmp_path, "superglue", "main", ("FLAS",)), "samples")
    row = {"raw_superglue_predicted_index": 1, "raw_superglue_gold_index": 0,
           "choice_texts": ["no", "yes"], "FLAS_choice_loglikelihoods": [-3, -1]}
    text = "\n".join(document(row, source))
    assert "No generated text stored" in text
    assert "NOT a per-row score" in text
    assert "False" in text


def test_terminal_controls_and_wide_unicode():
    assert "\x1b" not in safe_text("hello\x1b[31m evil\r\x00")
    assert wrap("中文abc", 4) == ["中文", "abc"]
    assert wrap("a\nb", 2) == ["a", "b"]


def test_scopes_and_cancel(tmp_path):
    make_node(tmp_path)
    other = tmp_path / "studies/runs/n-0012_subset-42/FLAS/evaluate/runs/study/evaluators/study_lm_judge"
    other.mkdir(parents=True)
    (other / "manifest.json").write_text(json.dumps({"status": "complete", "config": {"models": ["FLAS"]}}))
    nodes = discover(tmp_path)
    assert {n.scope for n in nodes} == {"main", "study/n-0012_subset-42"}
    assert build_index(nodes, Filters(), cancelled=lambda: True) is None


def test_invalid_factor():
    with pytest.raises(ValueError):
        Filters(factor="nan").validate()
