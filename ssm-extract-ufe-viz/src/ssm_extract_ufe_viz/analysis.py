"""Pairwise feature differentiation, distances, and hierarchy probes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import BootstrapConfig, DifferentiationThresholds, DistanceWeights
from .dictionary import FeatureRecord
from .image_metrics import activation_map_iou, lpips_distance, ssim_distance, topk_overlap
from .metrics import bootstrap_ci, composite_distance, cosine_similarity, js_divergence, linear_cka

DEFAULT_BOOTSTRAP_CONFIG = BootstrapConfig()
DEFAULT_DIFFERENTIATION_THRESHOLDS = DifferentiationThresholds()
DEFAULT_DISTANCE_WEIGHTS = DistanceWeights()


def compute_pair_scores(
    rec_i: FeatureRecord,
    rec_j: FeatureRecord,
    *,
    activations_i: np.ndarray | None = None,
    activations_j: np.ndarray | None = None,
    viz_i: np.ndarray | None = None,
    viz_j: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return per-axis pair scores. CKA is included only when raw activations
    are provided; never silently masked as cosine."""

    scores: dict[str, Any] = {
        "cos": cosine_similarity(rec_i.vector, rec_j.vector),
        "topk_overlap": topk_overlap(rec_i.top_k_image_indices, rec_j.top_k_image_indices),
    }
    hist_counts = _matching_histogram_counts(rec_i, rec_j)
    if hist_counts is not None:
        scores["jsd"] = js_divergence(*hist_counts)
    if activations_i is not None and activations_j is not None:
        scores["cka"] = linear_cka(activations_i, activations_j)

    map_i = _first_map(rec_i)
    map_j = _first_map(rec_j)
    if map_i is not None and map_j is not None:
        scores["map_iou"] = activation_map_iou(map_i, map_j)

    if viz_i is None:
        viz_i = _load_viz_from_record(rec_i)
    if viz_j is None:
        viz_j = _load_viz_from_record(rec_j)
    if viz_i is not None and viz_j is not None:
        scores["lpips"] = lpips_distance(viz_i, viz_j)
        scores["ssim_distance"] = ssim_distance(viz_i, viz_j)

    return scores


def pair_distance(
    rec_i: FeatureRecord,
    rec_j: FeatureRecord,
    weights: DistanceWeights = DEFAULT_DISTANCE_WEIGHTS,
    **score_kwargs: Any,
) -> tuple[float, dict[str, Any]]:
    """Scalar composite distance for two features, plus the per-axis scores."""

    scores = compute_pair_scores(rec_i, rec_j, **score_kwargs)
    distance = composite_distance(scores, weights)
    scores["distance"] = distance
    return distance, scores


def _matching_histogram_counts(
    rec_i: FeatureRecord,
    rec_j: FeatureRecord,
) -> tuple[list[float], list[float]] | None:
    counts_i, edges_i = rec_i.activation_histogram
    counts_j, edges_j = rec_j.activation_histogram
    if not counts_i and not counts_j:
        return None
    if not counts_i or not counts_j:
        raise ValueError("both activation histograms must be present to compute JSD")
    if len(edges_i) != len(counts_i) + 1 or len(edges_j) != len(counts_j) + 1:
        raise ValueError("activation histogram edges must have len(counts) + 1")
    edge_arr_i = np.asarray(edges_i, dtype=np.float64)
    edge_arr_j = np.asarray(edges_j, dtype=np.float64)
    if edge_arr_i.shape != edge_arr_j.shape or not np.allclose(edge_arr_i, edge_arr_j):
        raise ValueError("activation histogram bin edges must match before computing JSD")
    return counts_i, counts_j


def epsilon_different(
    rec_i: FeatureRecord,
    rec_j: FeatureRecord,
    thresholds: DifferentiationThresholds = DEFAULT_DIFFERENTIATION_THRESHOLDS,
    weights: DistanceWeights = DEFAULT_DISTANCE_WEIGHTS,
    **score_kwargs: Any,
) -> tuple[bool, dict[str, Any]]:
    """Derived boolean view on top of the composite distance.

    A pair passes the activation gate when cosine and JSD agree on
    separation (and CKA agrees when available). It passes the visual gate
    when optimized images and spatial maps disagree. The CKA axis is
    skipped, not faked from cosine, when raw activations are not provided.
    """

    distance, scores = pair_distance(rec_i, rec_j, weights=weights, **score_kwargs)
    cos = scores["cos"]
    jsd = scores.get("jsd", 0.0)
    cka_pass = (scores.get("cka") is None) or (scores["cka"] < thresholds.eps_cka)
    activation_gate = cos < thresholds.eps_cos and jsd > thresholds.eps_jsd and cka_pass

    lpips = scores.get("lpips")
    ssim_dist = scores.get("ssim_distance")
    map_iou = scores.get("map_iou", 1.0)
    if lpips is None or ssim_dist is None:
        visual_gate = False
    else:
        visual_gate = (
            lpips > thresholds.eps_lpips
            and ssim_dist > thresholds.eps_ssim
            and map_iou < thresholds.eps_iou
        )

    scores["activation_gate"] = activation_gate
    scores["visual_gate"] = visual_gate
    return activation_gate and visual_gate, scores


def differentiation_matrix(
    records: list[FeatureRecord],
    thresholds: DifferentiationThresholds = DEFAULT_DIFFERENTIATION_THRESHOLDS,
    weights: DistanceWeights = DEFAULT_DISTANCE_WEIGHTS,
) -> tuple[np.ndarray, np.ndarray, list[list[dict[str, Any]]]]:
    """Return (boolean different-matrix, scalar distance matrix, per-pair scores)."""

    n = len(records)
    bool_matrix = np.zeros((n, n), dtype=bool)
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    grid: list[list[dict[str, Any]]] = [[{} for _ in range(n)] for _ in range(n)]
    viz_cache = [_load_viz_from_record(rec) for rec in records]
    for i in range(n):
        for j in range(i + 1, n):
            different, scores = epsilon_different(
                records[i],
                records[j],
                thresholds,
                weights,
                viz_i=viz_cache[i],
                viz_j=viz_cache[j],
            )
            bool_matrix[i, j] = bool_matrix[j, i] = different
            dist_matrix[i, j] = dist_matrix[j, i] = scores["distance"]
            grid[i][j] = grid[j][i] = scores
    return bool_matrix, dist_matrix, grid


def cross_layer_overlap(
    records_shallow: list[FeatureRecord],
    records_deep: list[FeatureRecord],
    *,
    threshold: float = 0.4,
    top_k_limit: int | None = None,
) -> dict[str, Any]:
    """Quantify how often pairs of deep features share a shallow "parent".

    For every deep feature we list shallow features whose top-k image
    indices overlap by at least `threshold`. If two deep features share at
    least one shallow parent, they are counted as relying on a common
    lower-level dependency (the "wheels for car and bicycle" case).
    """

    shallow_sets = [
        (rec.feature_id, set(_limited_topk(rec.top_k_image_indices, top_k_limit)))
        for rec in records_shallow
    ]
    parent_index: dict[int, list[int]] = {}
    for deep in records_deep:
        deep_top = set(_limited_topk(deep.top_k_image_indices, top_k_limit))
        parents: list[int] = []
        if deep_top:
            for shallow_id, shallow_top in shallow_sets:
                if not shallow_top:
                    continue
                denom = max(len(deep_top), len(shallow_top))
                overlap = len(deep_top & shallow_top) / denom
                if overlap >= threshold:
                    parents.append(shallow_id)
        parent_index[deep.feature_id] = parents

    pair_count = 0
    shared_parent = 0
    for i in range(len(records_deep)):
        for j in range(i + 1, len(records_deep)):
            pair_count += 1
            parents_i = set(parent_index[records_deep[i].feature_id])
            parents_j = set(parent_index[records_deep[j].feature_id])
            if parents_i & parents_j:
                shared_parent += 1

    return {
        "n_deep": len(records_deep),
        "n_shallow": len(records_shallow),
        "threshold": threshold,
        "top_k_limit": top_k_limit,
        "pair_count": pair_count,
        "shared_parent_pairs": shared_parent,
        "shared_parent_rate": shared_parent / pair_count if pair_count else 0.0,
        "parent_index": {str(k): v for k, v in parent_index.items()},
    }


def _limited_topk(indices: list[int], top_k_limit: int | None) -> list[int]:
    if top_k_limit is None:
        return indices
    if top_k_limit <= 0:
        raise ValueError("top_k_limit must be positive")
    return indices[:top_k_limit]


def summarize_metric_confidence(
    pair_scores: list[dict[str, Any]],
    *,
    bootstrap: BootstrapConfig = DEFAULT_BOOTSTRAP_CONFIG,
) -> dict[str, dict[str, float]]:
    """Bootstrap aggregate confidence intervals for numeric pairwise metrics."""

    if not pair_scores:
        return {}
    numeric_keys = sorted({
        key
        for scores in pair_scores
        for key, value in scores.items()
        if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool)
    })
    summary: dict[str, dict[str, float]] = {}
    for key in numeric_keys:
        values = np.array([score[key] for score in pair_scores if key in score], dtype=float)
        if values.size == 0:
            continue
        ci = bootstrap_ci(
            values,
            n_bootstrap=bootstrap.n_bootstrap,
            confidence=bootstrap.confidence,
            seed=bootstrap.seed,
        )
        summary[key] = {"estimate": ci.estimate, "lower": ci.lower, "upper": ci.upper}
    return summary


def plot_heatmap(
    matrix: np.ndarray,
    output_path: str | Path,
    *,
    title: str = "Feature differences",
    cmap: str = "viridis",
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
) -> None:
    """Write a compact heatmap for a square matrix (boolean or scalar)."""

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix.astype(float), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("feature id")
    ax.set_ylabel("feature id")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _first_map(record: FeatureRecord) -> np.ndarray | None:
    if not record.spatial_activation_maps:
        return None
    return np.asarray(record.spatial_activation_maps[0], dtype=np.float32)


def _load_viz_from_record(record: FeatureRecord) -> np.ndarray | None:
    if not record.visualization_path:
        return None
    try:
        from PIL import Image

        return np.asarray(Image.open(record.visualization_path).convert("RGB"), dtype=np.float32) / 255.0
    except Exception:
        return None
