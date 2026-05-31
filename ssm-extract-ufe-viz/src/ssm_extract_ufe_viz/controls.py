"""Calibration controls for the composite pair distance.

The composite distance is only interpretable if we know what "zero",
"small", and "large" look like for it. These controls produce reference
distributions:

- `identical` – distance of every record against itself; must be ≈ 0.
- `perturbed` – distance to a noise-perturbed copy; small but nonzero.
- `real` – distance over all real (i, j) pairs from the dictionary.

A useful signal-to-noise sanity check is `real.mean / perturbed.mean`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .analysis import pair_distance
from .config import DistanceWeights
from .dictionary import FeatureRecord

DEFAULT_DISTANCE_WEIGHTS = DistanceWeights()


@dataclass(frozen=True)
class ControlSummary:
    name: str
    mean: float
    std: float
    p05: float
    p50: float
    p95: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mean": self.mean,
            "std": self.std,
            "p05": self.p05,
            "p50": self.p50,
            "p95": self.p95,
            "n": self.n,
        }


def identical_pair_control(
    records: list[FeatureRecord],
    weights: DistanceWeights = DEFAULT_DISTANCE_WEIGHTS,
) -> ControlSummary:
    distances: list[float] = []
    for record in records:
        distance, _ = pair_distance(record, record, weights=weights)
        distances.append(distance)
    return _summarize("identical", distances)


def perturbed_pair_control(
    records: list[FeatureRecord],
    weights: DistanceWeights = DEFAULT_DISTANCE_WEIGHTS,
    *,
    noise_std: float = 0.05,
    seed: int = 0,
) -> ControlSummary:
    rng = np.random.default_rng(seed)
    distances: list[float] = []
    for record in records:
        perturbed = _perturb(record, noise_std, rng)
        distance, _ = pair_distance(record, perturbed, weights=weights)
        distances.append(distance)
    return _summarize("perturbed", distances)


def true_pair_control(
    records: list[FeatureRecord],
    weights: DistanceWeights = DEFAULT_DISTANCE_WEIGHTS,
) -> ControlSummary:
    distances: list[float] = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            distance, _ = pair_distance(records[i], records[j], weights=weights)
            distances.append(distance)
    return _summarize("real", distances or [0.0])


def run_controls(
    records: list[FeatureRecord],
    weights: DistanceWeights = DEFAULT_DISTANCE_WEIGHTS,
    *,
    noise_std: float = 0.05,
    seed: int = 0,
) -> dict[str, ControlSummary]:
    return {
        "identical": identical_pair_control(records, weights),
        "perturbed": perturbed_pair_control(records, weights, noise_std=noise_std, seed=seed),
        "real": true_pair_control(records, weights),
    }


def signal_to_noise(controls: dict[str, ControlSummary]) -> float:
    """Real-pair mean over perturbed-pair mean. Higher = better separation."""

    perturbed = controls["perturbed"].mean
    real = controls["real"].mean
    return real / perturbed if perturbed > 1e-9 else float("inf")


def _summarize(name: str, values: list[float]) -> ControlSummary:
    arr = np.asarray(values, dtype=np.float64)
    return ControlSummary(
        name=name,
        mean=float(arr.mean()),
        std=float(arr.std()),
        p05=float(np.quantile(arr, 0.05)),
        p50=float(np.quantile(arr, 0.50)),
        p95=float(np.quantile(arr, 0.95)),
        n=int(arr.size),
    )


def _perturb(record: FeatureRecord, std: float, rng: np.random.Generator) -> FeatureRecord:
    vector = np.asarray(record.vector, dtype=np.float64)
    noise = rng.normal(0.0, std, size=vector.shape)
    new_vector = vector + noise
    norm = np.linalg.norm(new_vector)
    if norm > 1e-12:
        new_vector = new_vector / norm
    hist_counts, hist_edges = record.activation_histogram
    counts = np.asarray(hist_counts, dtype=np.float64)
    if counts.size:
        peak = max(float(counts.max()), 1.0)
        counts = np.maximum(counts + rng.normal(0.0, std * peak, size=counts.shape), 0.0)
    return FeatureRecord(
        feature_id=record.feature_id,
        layer=record.layer,
        vector=new_vector.tolist(),
        top_k_image_indices=list(record.top_k_image_indices),
        activation_histogram=(counts.tolist(), list(hist_edges)),
        spatial_activation_maps=record.spatial_activation_maps,
        visualization_path=record.visualization_path,
        decomposition=record.decomposition,
    )
