"""Feature-direction extraction from spatial activations."""

from __future__ import annotations

import numpy as np

from .dictionary import FeatureRecord
from .image_metrics import _normalize01


def spatial_to_matrix(activations: np.ndarray, *, pool: bool = True) -> np.ndarray:
    """Convert [N,H,W,D] activations into [N,D] or [N*H*W,D]."""

    arr = np.asarray(activations, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError("expected activations with shape [N,H,W,D]")
    if pool:
        # Default feature discovery is image-level: one activation vector per image.
        return arr.mean(axis=(1, 2))
    # The unpooled path is kept for experiments that want patch-level PCA.
    n, h, w, d_model = arr.shape
    return arr.reshape(n * h * w, d_model)


def decompose_pca_spatial(
    activations: np.ndarray,
    n_components: int,
    *,
    pool: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return PCA directions [K,D] and sample scores [N,K]."""

    matrix = spatial_to_matrix(activations, pool=pool)
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    # Use SVD directly to avoid a scikit-learn dependency in the PR artifact.
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(matrix, full_matrices=False)
    components = vt[:n_components]
    components = _row_normalize(components)
    # Scores are always per image so histograms/top-k examples stay interpretable.
    pooled = spatial_to_matrix(activations, pool=True)
    scores = pooled @ components.T
    return components, scores


def decompose_ica_spatial(
    activations: np.ndarray,
    n_components: int,
    *,
    pool: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ICA directions [K,D] and sample scores [N,K]."""

    if n_components <= 0:
        raise ValueError("n_components must be positive")
    try:
        from sklearn.decomposition import FastICA
    except ImportError as exc:
        raise ImportError(
            "ICA decomposition requires the [study] extra; install with "
            '`uv pip install -e ".[study]"`.'
        ) from exc

    matrix = spatial_to_matrix(activations, pool=pool)
    ica = FastICA(
        n_components=n_components,
        random_state=seed,
        whiten="unit-variance",
        max_iter=1000,
        tol=1e-3,
    )
    ica.fit(matrix)
    components = _row_normalize(np.asarray(ica.components_, dtype=np.float64))
    pooled = spatial_to_matrix(activations, pool=True)
    scores = pooled @ components.T
    return components, scores


def decompose_spatial_features(
    activations: np.ndarray,
    n_components: int,
    *,
    method: str = "pca",
    pool: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch spatial activations to the requested feature decomposition."""

    if method == "pca":
        return decompose_pca_spatial(activations, n_components, pool=pool)
    if method == "ica":
        return decompose_ica_spatial(activations, n_components, pool=pool, seed=seed)
    raise ValueError(f"unknown decomposition method: {method}")


def feature_activation_maps(
    activations: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    """Project [N,H,W,D] activations onto [K,D] directions -> [K,N,H,W]."""

    arr = np.asarray(activations, dtype=np.float64)
    dirs = np.asarray(directions, dtype=np.float64)
    if arr.ndim != 4 or dirs.ndim != 2:
        raise ValueError("expected activations [N,H,W,D] and directions [K,D]")
    if arr.shape[-1] != dirs.shape[-1]:
        raise ValueError("activation and direction feature dimensions differ")
    return np.einsum("nhwd,kd->knhw", arr, dirs)


def build_feature_records(
    activations: np.ndarray,
    directions: np.ndarray,
    scores: np.ndarray,
    *,
    layer: int | str,
    decomposition: str,
    top_k: int = 5,
    histogram_bins: int = 20,
) -> list[FeatureRecord]:
    """Create serializable feature records from directions and activations."""

    score_matrix = np.asarray(scores, dtype=np.float64)
    if score_matrix.ndim != 2:
        raise ValueError("expected scores with shape [N,K]")
    if score_matrix.shape[1] != len(directions):
        raise ValueError("score and direction feature counts differ")
    hist_edges = np.histogram_bin_edges(score_matrix.reshape(-1), bins=histogram_bins)
    maps = feature_activation_maps(activations, directions)
    records: list[FeatureRecord] = []
    for feature_id, direction in enumerate(directions):
        feature_scores = score_matrix[:, feature_id]
        # Top-k image indices provide dataset evidence without synthetic labels.
        top_indices = np.argsort(feature_scores)[::-1][:top_k].astype(int).tolist()
        hist_counts, _ = np.histogram(feature_scores, bins=hist_edges)
        # Store normalized maps only for the top-k examples to keep JSON compact.
        top_maps = [_normalize01(maps[feature_id, idx]).tolist() for idx in top_indices]
        records.append(
            FeatureRecord(
                feature_id=feature_id,
                layer=layer,
                vector=direction.astype(float).tolist(),
                top_k_image_indices=top_indices,
                activation_histogram=(hist_counts.astype(float).tolist(), hist_edges.tolist()),
                spatial_activation_maps=top_maps,
                decomposition=decomposition,
            )
        )
    return records


def _row_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)
