import numpy as np

from ssm_extract_ufe_viz.config import DistanceWeights
from ssm_extract_ufe_viz.controls import (
    identical_pair_control,
    perturbed_pair_control,
    run_controls,
    signal_to_noise,
    true_pair_control,
)
from ssm_extract_ufe_viz.dictionary import FeatureRecord


def _record(feature_id: int, vector: list[float]) -> FeatureRecord:
    return FeatureRecord(
        feature_id=feature_id,
        layer=1,
        vector=vector,
        top_k_image_indices=[feature_id, feature_id + 10],
        activation_histogram=([1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]),
        spatial_activation_maps=[[[1.0, 0.0], [0.0, 0.0]]],
    )


def test_identical_distance_is_zero():
    records = [_record(0, [1.0, 0.0]), _record(1, [0.0, 1.0])]
    summary = identical_pair_control(records, DistanceWeights())
    assert summary.mean == 0.0
    assert summary.n == len(records)


def test_perturbed_smaller_than_real():
    rng = np.random.default_rng(0)
    records = [_record(i, rng.normal(size=8).tolist()) for i in range(8)]
    p = perturbed_pair_control(records, DistanceWeights(), noise_std=0.01, seed=0)
    r = true_pair_control(records, DistanceWeights())
    assert p.mean < r.mean


def test_run_controls_returns_all_three_with_snr():
    records = [_record(0, [1.0, 0.0]), _record(1, [0.0, 1.0]), _record(2, [0.5, 0.5])]
    controls = run_controls(records, DistanceWeights())
    assert set(controls) == {"identical", "perturbed", "real"}
    snr = signal_to_noise(controls)
    assert snr > 0.0
