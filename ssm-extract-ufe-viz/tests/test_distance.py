from ssm_extract_ufe_viz.config import DistanceWeights
from ssm_extract_ufe_viz.metrics import composite_distance


def test_zero_on_identical_scores():
    scores = {
        "cos": 1.0,
        "jsd": 0.0,
        "cka": 1.0,
        "lpips": 0.0,
        "ssim_distance": 0.0,
        "map_iou": 1.0,
    }
    assert composite_distance(scores, DistanceWeights()) == 0.0


def test_large_on_dissimilar_scores():
    scores = {
        "cos": -1.0,
        "jsd": 0.5,
        "cka": 0.0,
        "lpips": 0.5,
        "ssim_distance": 1.0,
        "map_iou": 0.0,
    }
    assert composite_distance(scores, DistanceWeights()) > 4.0


def test_drops_missing_axes_instead_of_zeroing():
    scores = {"cos": 0.5, "jsd": 0.1}
    expected = 1.0 * (1 - 0.5) + 2.0 * 0.1
    assert composite_distance(scores, DistanceWeights()) == expected


def test_treats_cka_none_as_missing():
    with_cka = {"cos": 0.0, "cka": None}
    without_cka = {"cos": 0.0}
    weights = DistanceWeights()
    assert composite_distance(with_cka, weights) == composite_distance(without_cka, weights)


def test_clamps_tiny_negative_rounding_to_zero():
    scores = {"cos": 1.0 + 1e-16}
    assert composite_distance(scores, DistanceWeights()) == 0.0


def test_distance_weights_for_pca_disable_cosine():
    assert DistanceWeights.for_decomposition("pca").w_cos == 0.0
    assert DistanceWeights.for_decomposition("ica").w_cos == 1.0
