"""Dataset sampling helpers for feature-dictionary extraction."""

from __future__ import annotations

import numpy as np


def balanced_sample_indices(
    targets: list[int] | np.ndarray,
    n_classes: int,
    *,
    max_samples: int = 512,
    samples_per_class: int | None = None,
    seed: int = 0,
) -> list[int]:
    """Return deterministic class-balanced source indices.

    When `samples_per_class` is omitted, the per-class cap is derived from
    `max_samples`. When it is provided, `max_samples` still remains a global
    upper bound.
    """

    if n_classes <= 0:
        raise ValueError("n_classes must be positive")
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if samples_per_class is None:
        per_class = max(1, int(np.ceil(max_samples / n_classes)))
    else:
        if samples_per_class <= 0:
            raise ValueError("samples_per_class must be positive")
        per_class = samples_per_class

    arr = np.asarray(targets)
    rng = np.random.default_rng(seed)
    per_class_indices: list[list[int]] = []
    for cls_idx in rng.permutation(n_classes):
        cls_indices = np.where(arr == cls_idx)[0]
        if cls_indices.size == 0:
            continue
        chosen = rng.choice(cls_indices, min(per_class, cls_indices.size), replace=False)
        per_class_indices.append(chosen.astype(int).tolist())

    indices: list[int] = []
    offset = 0
    while len(indices) < max_samples:
        added = False
        for cls_indices in per_class_indices:
            if offset >= len(cls_indices):
                continue
            indices.append(cls_indices[offset])
            added = True
            if len(indices) >= max_samples:
                break
        if not added:
            break
        offset += 1
    return indices


def remap_topk_indices(records, source_indices: list[int]) -> None:
    """Rewrite subset-relative top-k indices to source dataset indices in-place."""

    for record in records:
        record.top_k_image_indices = [
            source_indices[row_idx] for row_idx in record.top_k_image_indices
        ]
