from __future__ import annotations

import numpy as np
from sklearn.decomposition import NMF, PCA
from torch.utils.data import DataLoader
from tqdm import tqdm


def collect_activations(
    probe,
    loader: DataLoader,
) -> dict[int, np.ndarray]:
    """
    Run all batches through probe; concatenate per-layer activations.
    Returns {layer_idx: A} where A ∈ R^(N, d_model).
    """
    per_layer: dict[int, list[np.ndarray]] = {}
    for batch in tqdm(loader, desc="collecting activations", leave=False):
        activations = probe.collect_batch(batch)
        for layer_idx, act in activations.items():
            per_layer.setdefault(layer_idx, []).append(act.numpy())
    return {layer: np.concatenate(chunks, axis=0) for layer, chunks in per_layer.items()}


def decompose_pca(
    A: np.ndarray,
    n_components: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    PCA decomposition of A ∈ R^(N, d).

    Returns:
        components:  R^(k, d) - unit-normed feature direction vectors (L2).
        projections: R^(N, k) - per-sample activation on each feature.

    sklearn PCA.components_ rows are already unit-normed. Projections are
    computed as A_centered @ components.T, equivalent to pca.transform(A).
    """
    pca = PCA(n_components=n_components, random_state=seed)
    projections = pca.fit_transform(A).astype(np.float32)
    components = pca.components_.astype(np.float32)
    return components, projections


def decompose_nmf(
    A: np.ndarray,
    n_components: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    NMF decomposition of ReLU(A) ∈ R^(N, d).

    NMF requires non-negative input. ReLU projection discards negative
    activations; the resulting part-based decomposition encourages
    monosemantic, sparse features (Lee & Seung 1999).

    Returns:
        components:  R^(k, d) - L2-normalised feature directions.
        projections: R^(N, k) - non-negative activation coefficients.
    """
    A_nn = np.maximum(A, 0.0)
    nmf = NMF(n_components=n_components, random_state=seed, max_iter=400)
    projections = nmf.fit_transform(A_nn).astype(np.float32)
    components = nmf.components_.astype(np.float32)
    # L2-normalise rows so each direction is a unit vector.
    norms = np.linalg.norm(components, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    components = components / norms
    return components, projections


def compute_activation_histograms(
    projections: np.ndarray,
    n_bins: int = 50,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Compute a normalised histogram for each feature (column of projections).

    Returns list of (bin_edges [n_bins+1], counts [n_bins]).
    counts sums to 1.0 - represents the discrete distribution P_i used
    in JSD computation.
    """
    n_features = projections.shape[1]
    histograms = []
    for i in range(n_features):
        counts, edges = np.histogram(projections[:, i], bins=n_bins)
        total = counts.sum()
        normalised = counts / total if total > 0 else counts.astype(float)
        histograms.append((edges.astype(np.float32), normalised.astype(np.float32)))
    return histograms


def get_top_k_examples(
    projections: np.ndarray,
    texts: list[str],
    k: int = 10,
) -> list[list[str]]:
    """
    For each feature (column i of projections), return the k texts with
    the highest projections[:, i] value.

    Returns list[list[str]], outer index = feature, inner = top-k texts.
    """
    n_features = projections.shape[1]
    result = []
    for i in range(n_features):
        top_indices = np.argsort(projections[:, i])[-k:][::-1]
        result.append([texts[idx] for idx in top_indices])
    return result


def orthogonalize_polysemantic(
    activations: np.ndarray,
    components: np.ndarray,
    projections: np.ndarray,
    polysemanticity_scores: list[float],
    threshold: float = 0.5,
    n_sub: int = 2,
    top_k_samples: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Replace polysemantic features (score > threshold) with sub-features derived
    from SVD on their top-activating samples, then re-orthogonalize via QR.

    Returns (new_components [K', d_model], new_projections [N, K']).
    K' >= K when any features are split (each polysemantic feature becomes n_sub).
    """
    N = activations.shape[0]
    new_dirs: list[np.ndarray] = []
    for i, score in enumerate(polysemanticity_scores):
        if score != -1.0 and score > threshold:
            k = min(top_k_samples, N)
            top_idx = np.argsort(projections[:, i])[-k:]
            _, _, Vt = np.linalg.svd(activations[top_idx], full_matrices=False)
            actual_n = min(n_sub, Vt.shape[0])
            new_dirs.extend(Vt[:actual_n])
        else:
            new_dirs.append(components[i])

    stacked = np.vstack(new_dirs)
    Q, _ = np.linalg.qr(stacked.T)
    new_components = Q.T.astype(np.float32)
    norms = np.linalg.norm(new_components, axis=1, keepdims=True)
    new_components = new_components / np.where(norms < 1e-10, 1.0, norms)
    new_projections = (activations @ new_components.T).astype(np.float32)
    return new_components, new_projections
