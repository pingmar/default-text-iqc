"""Distill-style activation maximization for feature directions.

Implements the canonical recipe from
https://distill.pub/2017/feature-visualization/:

- decorrelated color space (Lucid-style ImageNet SVD matrix);
- Fourier-filtered initial parametrization;
- transformation robustness via padded jitter and random affine
  (rotation + scale);
- TV and L2 regularizers;
- optional diversity loss across `n_facets` parallel optimizations.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .config import VisualizationConfig

if TYPE_CHECKING:
    import torch

DEFAULT_VISUALIZATION_CONFIG = VisualizationConfig()

# Lucid's ImageNet color-correlation matrix (decorrelated → RGB).
_COLOR_CORR_VALUES = (
    (0.26, 0.09, 0.02),
    (0.27, 0.00, -0.05),
    (0.27, -0.09, 0.03),
)
_COLOR_NORM = 0.5577


def _torch_modules():
    import torch
    import torch.nn.functional as torch_functional

    return torch, torch_functional


class FeatureActivationProbe(Protocol):
    def feature_activation_map(
        self,
        images: torch.Tensor,
        layer: int | str,
        feature_dir: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Return per-pixel activations [B, H, W] for a batch of images."""


def optimize_feature_visualization(
    probe: FeatureActivationProbe,
    layer: int | str,
    feature_dir: np.ndarray,
    config: VisualizationConfig = DEFAULT_VISUALIZATION_CONFIG,
    *,
    device: str = "cpu",
) -> torch.Tensor:
    """Optimize `n_facets` images to maximize a feature direction.

    Returns an image tensor with shape [n_facets, 3, H, W] in [0, 1].
    """

    torch, _ = _torch_modules()
    torch.manual_seed(config.seed)
    param = _initial_param(config, device=device)
    param.requires_grad_(True)
    optimizer = torch.optim.Adam([param], lr=config.lr)

    for _ in range(config.n_steps):
        optimizer.zero_grad(set_to_none=True)
        image = _to_valid_rgb(param, config)
        transformed = _augment(image, config)
        maps = probe.feature_activation_map(transformed, layer, feature_dir)
        activations = maps.mean(dim=(1, 2))
        loss = -activations.mean()
        if config.n_facets > 1 and config.diversity_weight > 0:
            loss = loss + config.diversity_weight * _diversity_loss(maps)
        loss = (
            loss
            + config.tv_weight * total_variation(image)
            + config.l2_weight * torch.mean(image ** 2)
        )
        loss.backward()
        optimizer.step()

    return _to_valid_rgb(param.detach(), config).cpu().clamp(0.0, 1.0)


def total_variation(image: torch.Tensor) -> torch.Tensor:
    """Anisotropic TV for batched [B, C, H, W] images."""

    torch, _ = _torch_modules()
    return (
        torch.mean(torch.abs(image[..., 1:, :] - image[..., :-1, :]))
        + torch.mean(torch.abs(image[..., :, 1:] - image[..., :, :-1]))
    )


def save_image(tensor: torch.Tensor, path: str | Path) -> None:
    """Save [C, H, W] or [B, C, H, W] tensor in [0, 1] as PNG (first facet)."""

    from PIL import Image

    arr = tensor.detach().cpu().clamp(0, 1).numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError("expected [C,H,W] or [B,C,H,W] tensor")
    arr = np.moveaxis(arr, 0, -1)
    arr = (arr * 255.0).round().astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def plot_feature_grid(
    image_paths: list[str | Path],
    labels: list[str],
    output_path: str | Path,
    *,
    n_cols: int = 4,
) -> None:
    """Write a simple grid of generated feature images."""

    import matplotlib.pyplot as plt
    from PIL import Image

    if not image_paths:
        raise ValueError("image_paths must be non-empty")
    n_cols = max(1, n_cols)
    n_rows = int(np.ceil(len(image_paths) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes_arr = np.atleast_1d(axes).reshape(n_rows, n_cols)
    for ax in axes_arr.reshape(-1):
        ax.axis("off")
    for ax, path, label in zip(axes_arr.reshape(-1), image_paths, labels, strict=False):
        ax.imshow(Image.open(path))
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _initial_param(config: VisualizationConfig, *, device: str) -> torch.Tensor:
    torch, _ = _torch_modules()
    shape = (config.n_facets, 3, config.image_size, config.image_size)
    if not config.fourier_init:
        return torch.randn(shape, device=device) * 0.1
    noise = torch.randn(shape, device=device)
    freq = torch.fft.rfft2(noise)
    height = config.image_size
    width = config.image_size
    fy = torch.fft.fftfreq(height, device=device).reshape(-1, 1)
    fx = torch.fft.rfftfreq(width, device=device).reshape(1, -1)
    scale = torch.sqrt(fx ** 2 + fy ** 2).clamp_min(1.0 / max(height, width))
    filtered = torch.fft.irfft2(freq / scale, s=(height, width))
    return filtered * 0.05


def _to_valid_rgb(param: torch.Tensor, config: VisualizationConfig) -> torch.Tensor:
    """Map raw param to [0, 1] RGB, optionally through a color-decorrelation matrix."""

    torch, _ = _torch_modules()
    if not config.color_decorrelation:
        return torch.sigmoid(param)
    matrix = torch.tensor(_COLOR_CORR_VALUES, device=param.device, dtype=param.dtype) / _COLOR_NORM
    flat = param.permute(0, 2, 3, 1) @ matrix.t()
    return torch.sigmoid(flat.permute(0, 3, 1, 2))


def _augment(image: torch.Tensor, config: VisualizationConfig) -> torch.Tensor:
    image = _padded_jitter(image, config.jitter)
    image = _random_affine(image, config.rotate_degrees, config.scale_range)
    return image


def _padded_jitter(image: torch.Tensor, jitter: int) -> torch.Tensor:
    torch, torch_functional = _torch_modules()
    if jitter <= 0:
        return image
    _, _, h, w = image.shape
    padded = torch_functional.pad(image, [jitter] * 4, mode="reflect")
    shift_y = int(torch.randint(0, 2 * jitter + 1, ()).item())
    shift_x = int(torch.randint(0, 2 * jitter + 1, ()).item())
    return padded[..., shift_y : shift_y + h, shift_x : shift_x + w]


def _random_affine(
    image: torch.Tensor,
    rotate_degrees: float,
    scale_range: tuple[float, float],
) -> torch.Tensor:
    torch, torch_functional = _torch_modules()
    lo, hi = scale_range
    if rotate_degrees == 0.0 and lo == 1.0 and hi == 1.0:
        return image
    batch = image.shape[0]
    device = image.device
    dtype = image.dtype
    angle = (torch.rand(batch, device=device, dtype=dtype) * 2 - 1) * math.radians(rotate_degrees)
    scale = lo + torch.rand(batch, device=device, dtype=dtype) * (hi - lo)
    inv_scale = 1.0 / scale
    cos_a = torch.cos(angle) * inv_scale
    sin_a = torch.sin(angle) * inv_scale
    theta = torch.zeros(batch, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    grid = torch_functional.affine_grid(theta, list(image.shape), align_corners=False)
    return torch_functional.grid_sample(image, grid, align_corners=False, padding_mode="reflection")


def _diversity_loss(maps: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal cosine between flattened activation maps.

    Pushes facets toward different spatial activation patterns, surfacing
    multiple modes of a multi-modal feature direction.
    """

    torch, _ = _torch_modules()
    flat = maps.reshape(maps.shape[0], -1)
    flat = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cos_matrix = flat @ flat.t()
    n = cos_matrix.shape[0]
    if n <= 1:
        return torch.zeros((), dtype=maps.dtype, device=maps.device)
    mask = ~torch.eye(n, dtype=torch.bool, device=cos_matrix.device)
    return cos_matrix[mask].mean()
