import pytest

from ssm_extract_ufe_viz.analysis import (
    compute_pair_scores,
    cross_layer_overlap,
    differentiation_matrix,
    pair_distance,
)
from ssm_extract_ufe_viz.config import DifferentiationThresholds, DistanceWeights
from ssm_extract_ufe_viz.dictionary import FeatureRecord


def _records() -> list[FeatureRecord]:
    return [
        FeatureRecord(
            feature_id=0,
            layer=1,
            vector=[1.0, 0.0],
            top_k_image_indices=[1, 2, 3],
            activation_histogram=([10.0, 0.0], [0.0, 1.0, 2.0]),
            spatial_activation_maps=[[[1.0, 0.0], [0.0, 0.0]]],
        ),
        FeatureRecord(
            feature_id=1,
            layer=1,
            vector=[0.0, 1.0],
            top_k_image_indices=[10, 20, 30],
            activation_histogram=([0.0, 10.0], [0.0, 1.0, 2.0]),
            spatial_activation_maps=[[[0.0, 0.0], [0.0, 1.0]]],
        ),
    ]


def test_compute_pair_scores_skips_cka_without_activations():
    rec_i, rec_j = _records()
    scores = compute_pair_scores(rec_i, rec_j)
    assert "cos" in scores
    assert "jsd" in scores
    assert "cka" not in scores, "cka must be absent (not masked as cos) when raw activations are missing"


def test_compute_pair_scores_rejects_mismatched_histogram_edges():
    rec_i, rec_j = _records()
    rec_j.activation_histogram = ([0.0, 10.0], [-1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="bin edges must match"):
        compute_pair_scores(rec_i, rec_j)


def test_pair_distance_returns_scalar_and_scores():
    rec_i, rec_j = _records()
    distance, scores = pair_distance(rec_i, rec_j, weights=DistanceWeights())
    assert isinstance(distance, float)
    assert scores["distance"] == distance
    assert distance > 0.0


def test_differentiation_matrix_returns_bool_and_distance():
    records = _records()
    bool_matrix, dist_matrix, scores = differentiation_matrix(
        records, DifferentiationThresholds(), DistanceWeights(),
    )
    assert bool_matrix.shape == (2, 2)
    assert dist_matrix.shape == (2, 2)
    assert dist_matrix[0, 1] == dist_matrix[1, 0]
    assert dist_matrix[0, 0] == 0.0
    assert "cos" in scores[0][1]


def test_cross_layer_overlap_reports_shared_parents():
    shallow = [
        FeatureRecord(feature_id=0, layer=1, vector=[1.0], top_k_image_indices=[1, 2, 3]),
        FeatureRecord(feature_id=1, layer=1, vector=[1.0], top_k_image_indices=[10, 11, 12]),
    ]
    deep = [
        FeatureRecord(feature_id=0, layer=2, vector=[1.0], top_k_image_indices=[1, 2, 99]),
        FeatureRecord(feature_id=1, layer=2, vector=[1.0], top_k_image_indices=[1, 3, 88]),
        FeatureRecord(feature_id=2, layer=2, vector=[1.0], top_k_image_indices=[10, 11, 77]),
    ]
    result = cross_layer_overlap(shallow, deep, threshold=0.5)
    assert result["shared_parent_rate"] > 0.0
    assert result["n_deep"] == 3
    assert result["pair_count"] == 3


def test_cross_layer_overlap_respects_top_k_limit():
    shallow = [
        FeatureRecord(feature_id=0, layer=1, vector=[1.0], top_k_image_indices=[1, 2, 3]),
    ]
    deep = [
        FeatureRecord(feature_id=0, layer=2, vector=[1.0], top_k_image_indices=[1, 2, 9]),
        FeatureRecord(feature_id=1, layer=2, vector=[1.0], top_k_image_indices=[8, 2, 3]),
    ]
    limited = cross_layer_overlap(shallow, deep, threshold=0.5, top_k_limit=1)
    full = cross_layer_overlap(shallow, deep, threshold=0.5)
    assert limited["shared_parent_pairs"] == 0
    assert full["shared_parent_pairs"] == 1
