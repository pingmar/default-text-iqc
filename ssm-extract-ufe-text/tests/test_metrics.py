import numpy as np
import pytest

from ssm_extract_ufe_text.metrics import cosine_similarity, js_divergence, linear_cka


# cosine_similarity

def test_cosine_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_orthogonal_vectors():
    e0 = np.array([1.0, 0.0])
    e1 = np.array([0.0, 1.0])
    assert abs(cosine_similarity(e0, e1)) < 1e-9


def test_cosine_antiparallel_vectors():
    v = np.array([1.0, 0.0])
    assert abs(cosine_similarity(v, -v) - (-1.0)) < 1e-9


def test_cosine_zero_vector_raises():
    v = np.array([1.0, 0.0])
    zero = np.array([0.0, 0.0])
    with pytest.raises(ValueError):
        cosine_similarity(v, zero)


# js_divergence

def test_jsd_identical_distributions():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert abs(js_divergence(p, p)) < 1e-9


def test_jsd_disjoint_distributions():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    # JSD of disjoint distributions with base-2 = 1.0
    assert abs(js_divergence(p, q) - 1.0) < 1e-6


def test_jsd_symmetry():
    rng = np.random.default_rng(0)
    raw_p = rng.dirichlet(np.ones(8))
    raw_q = rng.dirichlet(np.ones(8))
    assert abs(js_divergence(raw_p, raw_q) - js_divergence(raw_q, raw_p)) < 1e-9


def test_jsd_normalisation_check():
    p = np.array([0.5, 0.3])  # sums to 0.8
    q = np.array([0.5, 0.5])
    with pytest.raises(ValueError):
        js_divergence(p, q)


def test_jsd_bounded():
    rng = np.random.default_rng(1)
    for _ in range(20):
        p = rng.dirichlet(np.ones(10))
        q = rng.dirichlet(np.ones(10))
        val = js_divergence(p, q)
        assert 0.0 <= val <= 1.0 + 1e-9


# linear_cka

def test_cka_identical_representations():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 8))
    assert abs(linear_cka(X, X) - 1.0) < 1e-6


def test_cka_orthogonal_representations():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((100, 4))
    Y = rng.standard_normal((100, 4))
    # Uncorrelated random matrices should have low CKA
    assert linear_cka(X, Y) < 0.5


def test_cka_invariant_to_scaling():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((60, 5))
    assert abs(linear_cka(X, 5.0 * X) - 1.0) < 1e-6


def test_cka_invariant_to_orthogonal_transform():
    rng = np.random.default_rng(9)
    X = rng.standard_normal((60, 4))
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    assert abs(linear_cka(X, X @ Q) - 1.0) < 1e-6


def test_cka_mismatched_rows_raises():
    X = np.ones((10, 3))
    Y = np.ones((11, 3))
    with pytest.raises(ValueError):
        linear_cka(X, Y)
