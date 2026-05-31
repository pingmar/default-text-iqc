import numpy as np
import pytest

from ssm_extract_ufe_viz.features import (
    build_feature_records,
    decompose_pca_spatial,
    decompose_spatial_features,
    feature_activation_maps,
    spatial_to_matrix,
)


def test_spatial_to_matrix_pooling_shapes():
    activations = np.random.default_rng(0).normal(size=(5, 2, 3, 4))
    assert spatial_to_matrix(activations, pool=True).shape == (5, 4)
    assert spatial_to_matrix(activations, pool=False).shape == (30, 4)


def test_pca_returns_feature_directions():
    activations = np.random.default_rng(1).normal(size=(8, 2, 2, 5))
    pca_dirs, pca_scores = decompose_pca_spatial(activations, 3)
    assert pca_dirs.shape == (3, 5)
    assert pca_scores.shape == (8, 3)


def test_decomposition_dispatch_rejects_unknown_method():
    activations = np.random.default_rng(3).normal(size=(8, 2, 2, 5))
    with pytest.raises(ValueError):
        decompose_spatial_features(activations, 2, method="bad")


def test_ica_returns_feature_directions_when_sklearn_is_available():
    pytest.importorskip("sklearn")
    activations = np.random.default_rng(4).normal(size=(10, 2, 2, 5))
    dirs, scores = decompose_spatial_features(activations, 2, method="ica", seed=0)
    assert dirs.shape == (2, 5)
    assert scores.shape == (10, 2)


def test_build_feature_records_serializable():
    activations = np.random.default_rng(2).normal(size=(6, 2, 2, 4))
    directions, scores = decompose_pca_spatial(activations, 2)
    maps = feature_activation_maps(activations, directions)
    records = build_feature_records(activations, directions, scores, layer=1, decomposition="pca")
    assert maps.shape == (2, 6, 2, 2)
    assert len(records) == 2
    assert records[0].top_k_image_indices
    assert records[0].spatial_activation_maps


def test_build_feature_records_uses_shared_histogram_edges():
    activations = np.random.default_rng(5).normal(size=(6, 2, 2, 4))
    directions, scores = decompose_pca_spatial(activations, 3)
    records = build_feature_records(activations, directions, scores, layer=1, decomposition="pca")
    edges = [record.activation_histogram[1] for record in records]
    assert edges[1:] == edges[:-1]
    for counts, hist_edges in (record.activation_histogram for record in records):
        assert len(hist_edges) == len(counts) + 1
