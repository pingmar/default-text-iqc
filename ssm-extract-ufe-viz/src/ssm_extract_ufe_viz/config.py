"""Configuration objects used by the visual feature differentiation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DistanceWeights:
    """Linear weights for the composite pair distance.

    The composite distance is
        d = w_cos*(1-cos) + w_jsd*JSD + w_cka*(1-CKA)
          + w_lpips*LPIPS + w_ssim*SSIM_dist + w_iou*(1-IoU)

    Axes whose underlying score is unavailable are skipped rather than zeroed,
    so an absent term cannot inflate the apparent distance. CKA is opt-in:
    the CLI dictionary format does not store raw activation matrices, so CKA
    appears only when callers pass those matrices to `compute_pair_scores`.
    """

    w_cos: float = 1.0
    w_jsd: float = 2.0
    w_cka: float = 1.0
    w_lpips: float = 1.0
    w_ssim: float = 1.0
    w_iou: float = 0.5

    @classmethod
    def for_decomposition(cls, method: str) -> DistanceWeights:
        """Return metric weights matched to the feature decomposition."""

        if method.lower() == "pca":
            return cls(w_cos=0.0)
        return cls()


@dataclass(frozen=True)
class DifferentiationThresholds:
    """Per-axis thresholds for the derived boolean `epsilon_different` gate.

    The defaults match the loose operating point used end-to-end in
    `scripts/run_demo_vim.py`, so the dictionary-only validation script and
    the demo agree without further configuration.
    """

    eps_cos: float = 0.30
    eps_jsd: float = 0.05
    eps_cka: float = 0.50
    eps_lpips: float = 0.05
    eps_ssim: float = 0.10
    eps_iou: float = 0.50


@dataclass(frozen=True)
class VisualizationConfig:
    """Activation-maximization settings, Distill-style."""

    n_steps: int = 256
    lr: float = 0.05
    image_size: int = 224
    n_facets: int = 1
    color_decorrelation: bool = True
    fourier_init: bool = True
    jitter: int = 8
    rotate_degrees: float = 5.0
    scale_range: tuple[float, float] = (0.95, 1.05)
    tv_weight: float = 1e-4
    l2_weight: float = 1e-4
    diversity_weight: float = 1e-2
    seed: int = 0


@dataclass(frozen=True)
class BootstrapConfig:
    """Bootstrap confidence interval settings."""

    n_bootstrap: int = 500
    confidence: float = 0.95
    seed: int = 0


@dataclass(frozen=True)
class ContentContextSplit:
    """Hierarchy-aware split of the pair-distance axes.

    Separates the per-axis pair scores produced by ``compute_pair_scores``
    into a *content* group (direction and optimized-visualization
    disagreement) and a *context* group (whether they fire on different
    inputs / regions). Each group is summed with its own weights, axes whose
    underlying score is missing are dropped on both sides (matching
    ``composite_distance`` in metrics.py).

    The four-way classification used by ``classify_pair`` thresholds
    ``d_content`` against ``eps_content`` and ``d_context`` against
    ``eps_context``. Empirical p-values are reported at level ``alpha``.
    """

    content_axes: tuple[str, ...] = ("cos", "lpips", "ssim_distance")
    context_axes: tuple[str, ...] = (
        "jsd",
        "cka",
        "topk_overlap",
        "map_iou",
        "parent_jaccard",
    )
    axis_weights: dict[str, float] = field(
        default_factory=lambda: {
            "cos": 1.0,
            "lpips": 1.0,
            "ssim_distance": 1.0,
            "jsd": 2.0,
            "cka": 1.0,
            "topk_overlap": 1.0,
            "map_iou": 0.5,
            "parent_jaccard": 1.0,
        }
    )
    eps_content: float = 0.35
    eps_context: float = 0.35
    alpha: float = 0.05

    @classmethod
    def for_decomposition(cls, method: str, **overrides) -> ContentContextSplit:
        """Return content/context weights matched to the feature decomposition."""

        axis_weights = dict(overrides.pop("axis_weights", cls().axis_weights))
        if method.lower() == "pca":
            axis_weights["cos"] = 0.0
        return cls(axis_weights=axis_weights, **overrides)
