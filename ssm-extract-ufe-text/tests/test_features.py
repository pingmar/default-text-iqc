import tempfile

import numpy as np
import pytest

from ssm_extract_ufe_text.analysis import differentiation_matrix, epsilon_different
from ssm_extract_ufe_text.config import DifferentiationThresholds
from ssm_extract_ufe_text.dictionary import FeatureDictionary, FeatureRecord
from ssm_extract_ufe_text.features import (
    compute_activation_histograms,
    decompose_nmf,
    decompose_pca,
    get_top_k_examples,
    orthogonalize_polysemantic,
)


N, D, K = 80, 16, 4


def _random_matrix(seed=0):
    return np.random.default_rng(seed).standard_normal((N, D)).astype(np.float32)


# decompose_pca

def test_pca_components_shape():
    A = _random_matrix()
    components, projections = decompose_pca(A, n_components=K)
    assert components.shape == (K, D)
    assert projections.shape == (N, K)


def test_pca_components_unit_normed():
    A = _random_matrix()
    components, _ = decompose_pca(A, n_components=K)
    norms = np.linalg.norm(components, axis=1)
    np.testing.assert_allclose(norms, np.ones(K), atol=1e-5)


def test_pca_projections_variance_ordered():
    A = _random_matrix()
    _, projections = decompose_pca(A, n_components=K)
    variances = projections.var(axis=0)
    # PCA guarantees descending explained variance
    assert all(variances[i] >= variances[i + 1] - 1e-5 for i in range(K - 1))


# decompose_nmf

def test_nmf_projections_nonnegative():
    A = _random_matrix()
    _, projections = decompose_nmf(A, n_components=K)
    assert (projections >= -1e-6).all()


def test_nmf_components_unit_normed():
    A = _random_matrix()
    components, _ = decompose_nmf(A, n_components=K)
    norms = np.linalg.norm(components, axis=1)
    np.testing.assert_allclose(norms, np.ones(K), atol=1e-5)


# compute_activation_histograms

def test_activation_histograms_count():
    projections = np.random.default_rng(0).standard_normal((N, K)).astype(np.float32)
    histograms = compute_activation_histograms(projections, n_bins=20)
    assert len(histograms) == K


def test_activation_histograms_sum_to_one():
    projections = np.random.default_rng(1).standard_normal((N, K)).astype(np.float32)
    histograms = compute_activation_histograms(projections, n_bins=20)
    for edges, counts in histograms:
        np.testing.assert_allclose(counts.sum(), 1.0, atol=1e-6)


# get_top_k_examples

def test_top_k_examples_count():
    projections = np.random.default_rng(2).standard_normal((N, K)).astype(np.float32)
    texts = [f"text_{i}" for i in range(N)]
    result = get_top_k_examples(projections, texts, k=5)
    assert len(result) == K
    for feature_texts in result:
        assert len(feature_texts) == 5


def test_top_k_examples_highest_activation():
    projections = np.zeros((N, 1), dtype=np.float32)
    projections[7, 0] = 99.0  # highest
    projections[3, 0] = 50.0  # second
    texts = [f"text_{i}" for i in range(N)]
    result = get_top_k_examples(projections, texts, k=2)
    assert result[0][0] == "text_7"
    assert result[0][1] == "text_3"


# FeatureDictionary roundtrip

def test_feature_dictionary_roundtrip():
    record = FeatureRecord(
        feature_id=0,
        layer=3,
        vector=[0.1, 0.2, 0.3],
        top_k_examples=["hello world", "foo bar"],
        activation_histogram=([0.0, 0.5, 1.0], [0.6, 0.4]),
        semantic_label="test",
        decomposition="pca",
    )
    fd = FeatureDictionary()
    fd.add(record)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    fd.save(path)
    fd2 = FeatureDictionary.load(path)

    r2 = fd2.get(layer=3, feature_id=0)
    assert r2.feature_id == record.feature_id
    assert r2.layer == record.layer
    assert r2.vector == record.vector
    assert r2.top_k_examples == record.top_k_examples
    assert r2.semantic_label == record.semantic_label
    assert r2.decomposition == record.decomposition


# epsilon_different

def _make_record(feat_id, counts):
    n = len(counts)
    edges = list(np.linspace(0.0, 1.0, n + 1))
    total = sum(counts) + 1e-10
    norm_counts = [c / total for c in counts]
    return FeatureRecord(
        feature_id=feat_id, layer=0,
        vector=[1.0, 0.0, 0.0],
        top_k_examples=[],
        activation_histogram=(edges, norm_counts),
    )


def test_epsilon_different_jsd_driven_pass():
    """Features with very different activation histograms should be ε-different."""
    thresholds = DifferentiationThresholds(eps_cos=0.9, eps_jsd=0.05, eps_cka=0.99)
    # Left-heavy vs right-heavy distribution → high JSD; orthogonal direction vectors.
    ra = _make_record(0, [10, 1, 1, 1, 1])
    rb = _make_record(1, [1, 1, 1, 1, 10])
    # Use orthogonal direction vectors so cos_pass is satisfied.
    ra.vector = [1.0, 0.0, 0.0]
    rb.vector = [0.0, 1.0, 0.0]
    is_diff, scores = epsilon_different(ra, rb, thresholds)
    assert scores["jsd"] > 0.3
    assert is_diff


def test_epsilon_different_jsd_driven_fail():
    """Features with nearly identical distributions should NOT be ε-different."""
    thresholds = DifferentiationThresholds(eps_cos=0.9, eps_jsd=0.5, eps_cka=0.99)
    # Both uniform → low JSD
    ra = _make_record(0, [2, 2, 2, 2, 2])
    rb = _make_record(1, [2, 2, 2, 2, 2])
    is_diff, scores = epsilon_different(ra, rb, thresholds)
    assert scores["jsd"] < 0.01
    assert not is_diff


# differentiation_matrix score range

def test_differentiation_matrix_score_range():
    """Composite score must lie in [0, 1] for all off-diagonal pairs."""
    rng = np.random.default_rng(5)
    F = 4
    projections = rng.standard_normal((30, F)).astype(np.float32)
    histograms = compute_activation_histograms(projections, n_bins=10)
    records = [
        FeatureRecord(
            feature_id=i, layer=0,
            vector=rng.standard_normal(8).tolist(),
            top_k_examples=[],
            activation_histogram=(histograms[i][0].tolist(), histograms[i][1].tolist()),
        )
        for i in range(F)
    ]
    thresholds = DifferentiationThresholds()
    score_mat, binary_mat = differentiation_matrix(records, projections, thresholds)
    off = score_mat[~np.eye(F, dtype=bool)]
    assert off.min() >= 0.0 - 1e-6
    assert off.max() <= 1.0 + 1e-6


def test_differentiation_matrix_diagonal_is_one():
    """Diagonal of score_matrix must be 1.0 (self-similarity by convention)."""
    rng = np.random.default_rng(6)
    F = 3
    projections = rng.standard_normal((20, F)).astype(np.float32)
    histograms = compute_activation_histograms(projections, n_bins=10)
    records = [
        FeatureRecord(
            feature_id=i, layer=0,
            vector=rng.standard_normal(4).tolist(),
            top_k_examples=[],
            activation_histogram=(histograms[i][0].tolist(), histograms[i][1].tolist()),
        )
        for i in range(F)
    ]
    thresholds = DifferentiationThresholds()
    score_mat, _ = differentiation_matrix(records, projections, thresholds)
    np.testing.assert_allclose(np.diag(score_mat), np.ones(F), atol=1e-6)


# orthogonalize_polysemantic

def test_orthogonalize_keeps_all_monosemantic():
    """All scores below threshold: K' == K and output shapes unchanged."""
    A = _random_matrix()
    components, projections = decompose_pca(A, n_components=K)
    scores = [0.1] * K
    new_comps, new_projs = orthogonalize_polysemantic(A, components, projections, scores, threshold=0.5)
    assert new_comps.shape[0] == K
    assert new_projs.shape == (N, K)


def test_orthogonalize_expands_polysemantic_count():
    """One feature above threshold with n_sub=2: K' == K + 1."""
    A = _random_matrix()
    components, projections = decompose_pca(A, n_components=K)
    scores = [0.1, 0.9, 0.1, 0.1]   # only index 1 is polysemantic
    new_comps, new_projs = orthogonalize_polysemantic(
        A, components, projections, scores, threshold=0.5, n_sub=2
    )
    assert new_comps.shape[0] == K + 1
    assert new_projs.shape[1] == K + 1


def test_orthogonalize_components_unit_normed():
    """Every row of new_components has L2 norm ≈ 1.0."""
    A = _random_matrix()
    components, projections = decompose_pca(A, n_components=K)
    scores = [0.9] * K
    new_comps, _ = orthogonalize_polysemantic(A, components, projections, scores, threshold=0.5)
    norms = np.linalg.norm(new_comps, axis=1)
    np.testing.assert_allclose(norms, np.ones(len(norms)), atol=1e-4)


def test_orthogonalize_components_orthogonal():
    """Rows of new_components are mutually orthogonal (QR guarantee)."""
    A = _random_matrix()
    components, projections = decompose_pca(A, n_components=K)
    scores = [0.1] * K
    new_comps, _ = orthogonalize_polysemantic(A, components, projections, scores, threshold=0.5)
    gram = new_comps @ new_comps.T
    np.testing.assert_allclose(gram, np.eye(K), atol=1e-4)


def test_orthogonalize_projections_shape():
    """new_projections.shape == (N, K') and values equal activations @ new_components.T."""
    A = _random_matrix()
    components, projections = decompose_pca(A, n_components=K)
    scores = [0.0] * K
    new_comps, new_projs = orthogonalize_polysemantic(A, components, projections, scores, threshold=0.5)
    assert new_projs.shape == (N, new_comps.shape[0])
    expected = (A @ new_comps.T).astype(np.float32)
    np.testing.assert_allclose(new_projs, expected, atol=1e-4)
