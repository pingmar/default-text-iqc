from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ssm_extract_ufe_viz.config import ContentContextSplit
from ssm_extract_ufe_viz.dictionary import FeatureDictionary, FeatureRecord
from ssm_extract_ufe_viz.differentiation import (
    classify_pair,
    differentiation_table,
    equivalence_classes,
    hierarchy_parent_jaccard,
    pairwise_pvalues,
    split_pair_scores,
)


def _record(feature_id: int, vector, top_k=None, hist=None) -> FeatureRecord:
    if top_k is None:
        top_k = list(range(feature_id * 3, feature_id * 3 + 3))
    if hist is None:
        hist = ([float(v) for v in [1.0 + feature_id, 0.5, 0.1]], [0.0, 1.0, 2.0, 3.0])
    return FeatureRecord(
        feature_id=feature_id,
        layer=1,
        vector=list(vector),
        top_k_image_indices=top_k,
        activation_histogram=hist,
        spatial_activation_maps=[],
        visualization_path="",
    )


def test_split_pair_scores_drops_missing_axes():
    scores = {"cos": 0.2, "jsd": 0.4, "topk_overlap": 0.0}
    split = ContentContextSplit()
    result = split_pair_scores(scores, split)
    assert result["axes_used"]["content"] == ["cos"]
    assert set(result["axes_used"]["context"]) == {"jsd", "topk_overlap"}
    assert 0.0 <= result["d_content"] <= 1.0
    assert 0.0 <= result["d_context"] <= 1.0
    # cos=0.2 -> contribution 0.8 with weight 1.0 / 1.0 weights => 0.8
    assert result["d_content"] == pytest.approx(0.8, abs=1e-9)


def test_content_context_split_for_pca_disables_cosine():
    scores = {"cos": 0.0, "jsd": 0.4}
    pca = split_pair_scores(scores, ContentContextSplit.for_decomposition("pca"))
    ica = split_pair_scores(scores, ContentContextSplit.for_decomposition("ica"))
    assert pca["d_content"] == 0.0
    assert pca["axes_used"]["content"] == []
    assert ica["d_content"] == pytest.approx(1.0)


def test_classify_pair_quadrants():
    split = ContentContextSplit(eps_content=0.5, eps_context=0.5)
    assert classify_pair(0.9, 0.9, split) == "distinct"
    assert classify_pair(0.9, 0.1, split) == "siblings"
    assert classify_pair(0.1, 0.1, split) == "redundant"
    assert classify_pair(0.1, 0.9, split) == "echo"


def test_hierarchy_parent_jaccard_string_and_int_keys():
    pi = {0: [1, 2], "1": [2, 3], 2: []}
    assert hierarchy_parent_jaccard(0, 1, pi) == pytest.approx(1.0 / 3.0)
    assert hierarchy_parent_jaccard(0, 2, pi) == pytest.approx(0.0)
    assert hierarchy_parent_jaccard(2, 2, pi) is None


def test_differentiation_table_classifies_siblings_and_distinct():
    rec_a = _record(0, [1.0, 0.0], top_k=[1, 2, 3])
    rec_b = _record(1, [0.0, 1.0], top_k=[1, 2, 3])  # opposite direction, shared inputs
    rec_c = _record(2, [0.0, 1.0], top_k=[100, 200, 300])  # opposite direction, disjoint inputs
    table = differentiation_table([rec_a, rec_b, rec_c], split=ContentContextSplit())
    by_pair = {(row["feature_i"], row["feature_j"]): row for row in table["rows"]}
    assert by_pair[(0, 1)]["label"] == "siblings"
    assert by_pair[(0, 2)]["label"] == "distinct"
    assert by_pair[(0, 1)]["d_content"] > 0.5
    assert by_pair[(0, 2)]["d_context"] > by_pair[(0, 1)]["d_context"]


def test_equivalence_classes_union_find_and_medoid():
    triangle_a = _record(0, [1.0, 0.0, 0.0])
    triangle_b = _record(1, [0.99, 0.01, 0.0])
    triangle_c = _record(2, [0.98, 0.02, 0.0])
    outlier = _record(3, [0.0, 0.0, 1.0])
    classes = equivalence_classes(
        [triangle_a, triangle_b, triangle_c, outlier],
        split=ContentContextSplit(eps_content=0.5),
    )
    assert len(classes) == 2
    sizes = sorted(len(c.members) for c in classes)
    assert sizes == [1, 3]
    triple = next(c for c in classes if len(c.members) == 3)
    # Medoid of three nearly-collinear vectors should be the middle one (id=1).
    assert triple.representative == 1
    assert triple.within_class_max >= 0.0


def test_equivalence_classes_include_threshold_boundary():
    rec_a = _record(0, [1.0, 0.0])
    rec_b = _record(1, [0.5, 0.8660254037844386])
    classes = equivalence_classes(
        [rec_a, rec_b],
        split=ContentContextSplit(eps_content=0.5),
    )
    assert len(classes) == 1


def test_pairwise_pvalues_distinct_vs_identical():
    near_identical = _record(0, [1.0, 0.0], hist=([5.0, 0.0], [0.0, 1.0, 2.0]))
    other_orientation = _record(
        1, [-1.0, 0.0], top_k=[99, 100, 101], hist=([0.0, 5.0], [0.0, 1.0, 2.0])
    )
    twin = _record(
        2,
        [1.0, 0.0],
        top_k=near_identical.top_k_image_indices,
        hist=([5.0, 0.0], [0.0, 1.0, 2.0]),
    )
    result = pairwise_pvalues(
        [near_identical, other_orientation, twin],
        split=ContentContextSplit(),
        noise_std=0.02,
        n_resample=8,
        seed=0,
    )
    p = result["p_matrix"]
    # Pairs with very different feature directions should be significant.
    assert p[0, 1] < 0.1
    # Identical-direction pairs should not be significant.
    assert p[0, 2] > 0.5
    assert result["null_samples"].size > 0
    assert result["n_null"] == 24
    assert result["p_floor"] == pytest.approx(1.0 / 25.0)
    assert "effect_matrix" in result
    assert result["effect_matrix"][0, 1] >= result["effect_matrix"][0, 2]


def test_validate_dictionary_smoke(tmp_path):
    rng = np.random.default_rng(0)
    records = []
    for fid in range(6):
        vec = rng.normal(size=8)
        vec = vec / np.linalg.norm(vec)
        hist = ([float(v) for v in rng.uniform(0.0, 1.0, size=4)], [0.0, 0.25, 0.5, 0.75, 1.0])
        records.append(
            FeatureRecord(
                feature_id=fid,
                layer=12,
                vector=vec.tolist(),
                top_k_image_indices=[fid * 5 + k for k in range(5)],
                activation_histogram=hist,
                spatial_activation_maps=[
                    rng.uniform(0.0, 1.0, size=(3, 3)).tolist(),
                ],
                visualization_path="",
                decomposition="pca",
            )
        )
    # Make one pair a near-duplicate so equivalence classes < n.
    records[1].vector = records[0].vector
    records[1].activation_histogram = records[0].activation_histogram
    # Make one pair share top-k for the "siblings" path.
    records[3].top_k_image_indices = list(records[2].top_k_image_indices)

    dictionary = FeatureDictionary(records=records, metadata={"smoke": True})
    dict_path = tmp_path / "tiny.json"
    dictionary.save(dict_path)
    parent_path = tmp_path / "parents.json"
    parent_path.write_text(
        json.dumps({"parent_index": {"0": [1], "1": [1], "2": [2], "3": [2]}}),
        encoding="utf-8",
    )

    output_dir = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_dictionary.py"
    completed = subprocess.run(
        [sys.executable, str(script),
         "--dict-path", str(dict_path),
         "--output-dir", str(output_dir),
         "--bootstrap", "50",
         "--n-resample", "4",
         "--grid-top-n", "2",
         "--parent-index-path", str(parent_path)],
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr

    expected = [
        "pairwise_metrics.csv",
        "pairwise_classification.csv",
        "equivalence_classes.json",
        "confidence_intervals.json",
        "controls.json",
        "content_heatmap.png",
        "context_heatmap.png",
        "report.md",
    ]
    for name in expected:
        path = output_dir / name
        assert path.exists() and path.stat().st_size > 0, f"missing {name}"

    classes_data = json.loads((output_dir / "equivalence_classes.json").read_text())
    assert classes_data["n_features"] == len(records)
    assert classes_data["n_classes"] <= len(records)
    pairwise_text = (output_dir / "pairwise_classification.csv").read_text()
    assert "effect_size" in pairwise_text
    report = (output_dir / "report.md").read_text()
    assert "Feature differentiation report" in report
    assert "equivalence classes" in report
    assert "p-floor" in report
    assert "Effect size" in report
    assert str(parent_path) in report
