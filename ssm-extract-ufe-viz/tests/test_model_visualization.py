import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ssm_extract_ufe_viz.config import VisualizationConfig  # noqa: E402
from ssm_extract_ufe_viz.model import MambaVisionProbe  # noqa: E402
from ssm_extract_ufe_viz.visualization import optimize_feature_visualization  # noqa: E402


class ToyVisionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images):
        x = torch.relu(self.conv(images))
        return self.pool(x).flatten(1)


def test_probe_collects_spatial_activations_from_toy_model():
    probe = MambaVisionProbe(model=ToyVisionModel(), device="cpu")
    probe.register_hooks(["conv"])
    spatial = probe.collect_batch_spatial(torch.zeros(2, 3, 16, 16))
    assert spatial["conv"].shape == (2, 16, 16, 4)


def test_feature_activation_map_returns_per_pixel_tensor():
    probe = MambaVisionProbe(model=ToyVisionModel(), device="cpu")
    probe.register_hooks(["conv"])
    direction = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    images = torch.zeros(2, 3, 16, 16)
    activation_map = probe.feature_activation_map(images, "conv", direction)
    assert activation_map.shape == (2, 16, 16)
    scalar = probe.feature_activation(images, "conv", direction)
    assert scalar.shape == ()


def test_activation_max_smoke_test_on_toy_model():
    probe = MambaVisionProbe(model=ToyVisionModel(), device="cpu")
    probe.register_hooks(["conv"])
    config = VisualizationConfig(
        n_steps=2,
        image_size=16,
        jitter=1,
        fourier_init=False,
        color_decorrelation=False,
        rotate_degrees=0.0,
        scale_range=(1.0, 1.0),
    )
    image = optimize_feature_visualization(
        probe,
        "conv",
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        config,
        device="cpu",
    )
    assert image.shape == (1, 3, 16, 16)
    assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0


def test_activation_max_supports_multiple_facets():
    probe = MambaVisionProbe(model=ToyVisionModel(), device="cpu")
    probe.register_hooks(["conv"])
    config = VisualizationConfig(
        n_steps=2,
        image_size=16,
        jitter=1,
        n_facets=3,
        fourier_init=False,
        color_decorrelation=False,
        rotate_degrees=0.0,
        scale_range=(1.0, 1.0),
    )
    image = optimize_feature_visualization(
        probe,
        "conv",
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        config,
        device="cpu",
    )
    assert image.shape == (3, 3, 16, 16)
