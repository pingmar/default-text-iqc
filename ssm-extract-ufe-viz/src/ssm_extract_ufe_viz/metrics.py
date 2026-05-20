"""Activation-space metrics and confidence intervals.

The metrics are intentionally NumPy-only so they can be tested without model
downloads or GPU-specific dependencies.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import DistanceWeights

ArrayLike = np.ndarray | list[float] | tuple[float, ...]
DEFAULT_DISTANCE_WEIGHTS = DistanceWeights()


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float


def _as_float_array(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("metric input must be non-empty")
    return arr


def cosine_similarity(a: ArrayLike, b: ArrayLike, eps: float = 1e-12) -> float:
    """Return cosine similarity in [-1, 1]. Zero vectors are invalid."""

    av = _as_float_array(a).reshape(-1)
    bv = _as_float_array(b).reshape(-1)
    if av.shape != bv.shape:
        raise ValueError(f"shape mismatch: {av.shape} != {bv.shape}")
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    if denom < eps:
        raise ValueError("cosine similarity is undefined for zero vectors")
    return float(np.dot(av, bv) / denom)


def js_divergence(p: ArrayLike, q: ArrayLike, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence over non-negative vectors, using natural logs."""

    pv = _as_float_array(p).reshape(-1)
    qv = _as_float_array(q).reshape(-1)
    if pv.shape != qv.shape:
        raise ValueError(f"shape mismatch: {pv.shape} != {qv.shape}")
    pv = np.maximum(pv, 0.0)
    qv = np.maximum(qv, 0.0)
    if pv.sum() <= eps or qv.sum() <= eps:
        return 0.0
    pv = pv / pv.sum()
    qv = qv / qv.sum()
    mv = 0.5 * (pv + qv)
    kl_pm = np.sum(np.where(pv > 0, pv * np.log((pv + eps) / (mv + eps)), 0.0))
    kl_qm = np.sum(np.where(qv > 0, qv * np.log((qv + eps) / (mv + eps)), 0.0))
    return float(0.5 * (kl_pm + kl_qm))


def linear_cka(x: ArrayLike, y: ArrayLike, eps: float = 1e-12) -> float:
    """Linear centered-kernel alignment for two activation matrices.

    Inputs are interpreted as [samples, features]. One-dimensional inputs are
    promoted to column vectors.
    """

    x_arr = _as_float_array(x)
    y_arr = _as_float_array(y)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    if y_arr.ndim == 1:
        y_arr = y_arr[:, None]
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("CKA inputs must have the same sample count")
    x_arr = x_arr - x_arr.mean(axis=0, keepdims=True)
    y_arr = y_arr - y_arr.mean(axis=0, keepdims=True)
    dot_xy = np.linalg.norm(x_arr.T @ y_arr, ord="fro") ** 2
    dot_xx = np.linalg.norm(x_arr.T @ x_arr, ord="fro")
    dot_yy = np.linalg.norm(y_arr.T @ y_arr, ord="fro")
    denom = dot_xx * dot_yy
    if denom < eps:
        return 0.0
    return float(dot_xy / denom)


def composite_distance(
    scores: Mapping[str, Any],
    weights: DistanceWeights = DEFAULT_DISTANCE_WEIGHTS,
) -> float:
    """Linear scalar distance from per-axis pair scores.

    Missing axes are dropped (not zero-imputed) so a removed term cannot
    inflate the composite. `cka` set to None is treated as missing.
    """

    distance = 0.0
    cos = scores.get("cos")
    if cos is not None:
        distance += weights.w_cos * (1.0 - float(cos))
    jsd = scores.get("jsd")
    if jsd is not None:
        distance += weights.w_jsd * float(jsd)
    cka = scores.get("cka")
    if cka is not None:
        distance += weights.w_cka * (1.0 - float(cka))
    lpips = scores.get("lpips")
    if lpips is not None:
        distance += weights.w_lpips * float(lpips)
    ssim_dist = scores.get("ssim_distance")
    if ssim_dist is not None:
        distance += weights.w_ssim * float(ssim_dist)
    map_iou = scores.get("map_iou")
    if map_iou is not None:
        distance += weights.w_iou * (1.0 - float(map_iou))
    return max(0.0, float(distance))


def bootstrap_ci(
    values: ArrayLike,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    """Return a percentile bootstrap CI for a one-sample statistic."""

    arr = _as_float_array(values).reshape(-1)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        sample = rng.choice(arr, size=arr.size, replace=True)
        estimates[idx] = statistic(sample)
    alpha = 1.0 - confidence
    return ConfidenceInterval(
        estimate=float(statistic(arr)),
        lower=float(np.quantile(estimates, alpha / 2.0)),
        upper=float(np.quantile(estimates, 1.0 - alpha / 2.0)),
    )
