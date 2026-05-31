import numpy as np
import pytest

from ssm_extract_ufe_viz.metrics import (
    bootstrap_ci,
    cosine_similarity,
    js_divergence,
    linear_cka,
)


def test_cosine_similarity_edge_cases():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert abs(cosine_similarity([1, 0], [0, 1])) < 1e-9
    with pytest.raises(ValueError, match="zero vectors"):
        cosine_similarity([0, 0], [1, 0])


def test_js_divergence_identical_distribution_is_zero():
    assert js_divergence([1, 2, 3], [1, 2, 3]) < 1e-9
    assert js_divergence([1, 0], [0, 1]) > 0.1


def test_linear_cka_identical_is_one():
    x = np.arange(12, dtype=float).reshape(4, 3)
    assert abs(linear_cka(x, x) - 1.0) < 1e-9


def test_bootstrap_ci_is_deterministic_with_seed():
    first = bootstrap_ci([1, 2, 3, 4], n_bootstrap=50, seed=7)
    second = bootstrap_ci([1, 2, 3, 4], n_bootstrap=50, seed=7)
    assert first == second
    assert first.lower <= first.estimate <= first.upper
