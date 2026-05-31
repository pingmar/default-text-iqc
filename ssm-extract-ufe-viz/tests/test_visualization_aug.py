import pytest

torch = pytest.importorskip("torch")

from ssm_extract_ufe_viz.config import VisualizationConfig  # noqa: E402
from ssm_extract_ufe_viz.visualization import (  # noqa: E402
    _diversity_loss,
    _padded_jitter,
    _random_affine,
    _to_valid_rgb,
)


def test_padded_jitter_preserves_shape():
    image = torch.rand(2, 3, 8, 8)
    assert _padded_jitter(image, jitter=2).shape == image.shape


def test_padded_jitter_noop_for_zero():
    image = torch.rand(1, 3, 4, 4)
    assert torch.equal(_padded_jitter(image, jitter=0), image)


def test_random_affine_preserves_shape():
    image = torch.rand(3, 3, 16, 16)
    assert _random_affine(image, 5.0, (0.95, 1.05)).shape == image.shape


def test_to_valid_rgb_clamped_to_unit_interval():
    param = torch.randn(1, 3, 4, 4) * 5.0
    cfg = VisualizationConfig(color_decorrelation=True, image_size=4)
    out = _to_valid_rgb(param, cfg)
    assert torch.all(out >= 0) and torch.all(out <= 1)
    assert out.shape == (1, 3, 4, 4)


def test_diversity_loss_high_when_facets_identical():
    maps = torch.ones(3, 4, 4)
    assert _diversity_loss(maps).item() > 0.99


def test_diversity_loss_zero_for_single_facet():
    maps = torch.ones(1, 4, 4)
    assert _diversity_loss(maps).item() == 0.0
