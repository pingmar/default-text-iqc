import pytest

from ssm_extract_ufe_viz.datasets import balanced_sample_indices, remap_topk_indices
from ssm_extract_ufe_viz.dictionary import FeatureRecord


def test_balanced_sample_indices_are_deterministic():
    targets = [0, 0, 0, 1, 1, 1, 2, 2]
    first = balanced_sample_indices(targets, 3, max_samples=6, seed=7)
    second = balanced_sample_indices(targets, 3, max_samples=6, seed=7)
    assert first == second
    assert len(first) == 6
    assert len([idx for idx in first if targets[idx] == 0]) == 2
    assert len([idx for idx in first if targets[idx] == 1]) == 2
    assert len([idx for idx in first if targets[idx] == 2]) == 2


def test_samples_per_class_respects_max_samples_upper_bound():
    targets = [0, 0, 0, 1, 1, 1]
    indices = balanced_sample_indices(
        targets, 2, max_samples=2, samples_per_class=3, seed=0,
    )
    assert len(indices) == 2


def test_balanced_sample_indices_can_subsample_classes():
    targets = [0, 1, 2, 3, 4]
    indices = balanced_sample_indices(
        targets, 5, max_samples=3, samples_per_class=1, seed=0,
    )
    assert len(indices) == 3
    assert len({targets[idx] for idx in indices}) == 3


def test_balanced_sample_indices_rejects_invalid_limits():
    with pytest.raises(ValueError):
        balanced_sample_indices([0], 1, max_samples=0)
    with pytest.raises(ValueError):
        balanced_sample_indices([0], 1, samples_per_class=0)


def test_remap_topk_indices_uses_source_indices():
    record = FeatureRecord(feature_id=0, layer=1, vector=[1.0], top_k_image_indices=[2, 0])
    remap_topk_indices([record], [100, 101, 102])
    assert record.top_k_image_indices == [102, 100]
