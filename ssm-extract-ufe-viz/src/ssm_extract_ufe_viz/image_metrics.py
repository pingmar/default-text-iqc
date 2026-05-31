"""Image and spatial-map metrics for visual feature differentiation."""

from __future__ import annotations

import numpy as np


def to_channel_last(image: np.ndarray) -> np.ndarray:
    """Convert [C,H,W] or [H,W,C] image arrays to [H,W,C] float arrays."""

    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError("expected image with 2 or 3 dimensions")
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    return arr


def _normalize01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("metric input must be non-empty")
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v - min_v < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_v) / (max_v - min_v)


def _image01(image: np.ndarray) -> np.ndarray:
    arr = to_channel_last(image).astype(np.float32)
    if arr.max(initial=0.0) > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def ssim_distance(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Return 1 - SSIM, falling back to normalized MSE if scikit-image is absent."""

    a = _image01(image_a)
    b = _image01(image_b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} != {b.shape}")
    try:
        from skimage.metrics import structural_similarity

        if a.ndim == 2:
            score = structural_similarity(a, b, data_range=1.0)
        else:
            score = structural_similarity(a, b, data_range=1.0, channel_axis=-1)
        return float(1.0 - score)
    except Exception:
        return float(np.mean((a - b) ** 2))


def activation_map_iou(
    map_a: np.ndarray,
    map_b: np.ndarray,
    *,
    quantile: float = 0.80,
) -> float:
    """IoU between the high-activation regions of two spatial maps."""

    a = _normalize01(np.asarray(map_a, dtype=np.float32))
    b = _normalize01(np.asarray(map_b, dtype=np.float32))
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} != {b.shape}")
    a_mask = a >= np.quantile(a, quantile)
    b_mask = b >= np.quantile(b, quantile)
    union = np.logical_or(a_mask, b_mask).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a_mask, b_mask).sum() / union)


def topk_overlap(indices_a: list[int], indices_b: list[int]) -> float:
    """Fractional overlap between two top-k example index lists."""

    set_a = set(indices_a)
    set_b = set(indices_b)
    if not set_a and not set_b:
        return 1.0
    denom = max(len(set_a), len(set_b), 1)
    return float(len(set_a & set_b) / denom)


_LPIPS_CACHE: dict[tuple[str, str], object] = {}


def _get_lpips(net: str, device: str):
    key = (net, device)
    cached = _LPIPS_CACHE.get(key)
    if cached is not None:
        return cached
    import lpips

    model = lpips.LPIPS(net=net).to(device).eval()
    _LPIPS_CACHE[key] = model
    return model


def lpips_distance(image_a: np.ndarray, image_b: np.ndarray, *, device: str = "cpu") -> float:
    """Return LPIPS distance if installed, otherwise a normalized L2 proxy.

    The fallback keeps tests and lightweight CPU environments runnable. Real
    experiment runs should install the `lpips` optional extra.
    """

    a = _image01(image_a)
    b = _image01(image_b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} != {b.shape}")
    try:
        import torch

        loss = _get_lpips("alex", device)
        tensor_a = torch.from_numpy(np.moveaxis(a, -1, 0))[None].to(device) * 2.0 - 1.0
        tensor_b = torch.from_numpy(np.moveaxis(b, -1, 0))[None].to(device) * 2.0 - 1.0
        with torch.no_grad():
            return float(loss(tensor_a.float(), tensor_b.float()).item())
    except Exception:
        return float(np.sqrt(np.mean((a - b) ** 2)))
