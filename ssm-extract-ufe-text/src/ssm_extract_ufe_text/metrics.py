"""
Pure metric functions - no model loading, no I/O. Fully unit-testable on CPU.

Mathematical definitions

Cosine similarity:
    cos(u, v) = (u · v) / (||u||₂ · ||v||₂)

Jensen-Shannon divergence (base-2, bounded in [0, 1]):
    M = (P + Q) / 2
    JSD(P ∥ Q) = (KL(P ∥ M) + KL(Q ∥ M)) / 2

Linear CKA (Kornblith et al. 2019):
    HSIC(K, L) = (1/(n-1)²) tr(K_c L_c)
    CKA(X, Y)  = HSIC(XX^T, YY^T) / sqrt(HSIC(XX^T, XX^T) · HSIC(YY^T, YY^T))
    where K_c = HKH, H = I - (1/n)11^T (double-centering)
"""
from __future__ import annotations

import numpy as np


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0.0 or norm_v == 0.0:
        raise ValueError("cosine_similarity: zero-norm input vector")
    return float(np.dot(u, v) / (norm_u * norm_v))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon divergence between two discrete distributions.
    p and q must be non-negative and sum to 1.0 (±1e-6).
    Returns a value in [0, 1] using base-2 logarithms.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if abs(p.sum() - 1.0) > 1e-6 or abs(q.sum() - 1.0) > 1e-6:
        raise ValueError("js_divergence: inputs must be normalised distributions (sum=1)")
    eps = 1e-10
    m = (p + q) / 2.0
    kl_pm = np.sum(p * np.log2((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log2((q + eps) / (m + eps)))
    return float(np.clip((kl_pm + kl_qm) / 2.0, 0.0, 1.0))


def linear_cka(X: np.ndarray, Y: np.ndarray, seed: int = 42) -> float:
    """
    Linear CKA between activation matrices X ∈ R^(n, p) and Y ∈ R^(n, q).
    Both are mean-centred column-wise before kernel computation (invariant
    to mean shift; Kornblith 2019 eq. 2).

    Subsamples to 2000 rows when n > 2000 to keep O(n²) cost tractable.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("linear_cka: X and Y must have the same number of rows")
    n = X.shape[0]
    if n > 2000:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, 2000, replace=False)
        X, Y = X[idx], Y[idx]
    # Column-wise mean centering
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    K = X @ X.T
    L = Y @ Y.T
    hsic_kl = _hsic(K, L)
    hsic_kk = _hsic(K, K)
    hsic_ll = _hsic(L, L)
    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-10:
        return 0.0
    return float(np.clip(hsic_kl / denom, 0.0, 1.0))


def _hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Biased HSIC estimator: (1/(n-1)²) tr(K_c L_c)."""
    n = K.shape[0]
    K_c = _double_center(K)
    L_c = _double_center(L)
    return float(np.trace(K_c @ L_c) / (n - 1) ** 2)


def _double_center(K: np.ndarray) -> np.ndarray:
    """Apply double-centering: K_c = HKH, H = I - (1/n)11^T."""
    n = K.shape[0]
    row_mean = K.mean(axis=1, keepdims=True)
    col_mean = K.mean(axis=0, keepdims=True)
    grand_mean = K.mean()
    return K - row_mean - col_mean + grand_mean
