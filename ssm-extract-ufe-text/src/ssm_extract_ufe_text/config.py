from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExtractionConfig:
    model_name: str = "state-spaces/mamba-130m-hf"
    # None = hook all layers; explicit list overrides.
    layers: list[int] | None = None
    max_seq_len: int = 128
    batch_size: int = 32
    # pca or nmf
    decomposition: str = "pca"
    n_components: int = 32
    # Top-k highest-activating corpus texts stored per feature.
    top_k: int = 10
    n_hist_bins: int = 50
    seed: int = 42
    device: str = "auto"


@dataclass
class DifferentiationThresholds:
    """
    Two features are ε-different iff ALL three conditions hold:
        cos(proj_i, proj_j) < eps_cos   (activation-profile similarity)
        JSD(P_i || P_j)     > eps_jsd   (distributional separation)
        CKA(A_i, A_j)       < eps_cka   (representational similarity)

    Threshold calibration notes:
        eps_jsd = 0.20 bisects observed within-corpus mean (≈0.15) and
        cross-corpus NER mean (≈0.30), making it the primary discriminator.

        eps_cos and eps_cka become informative only with NMF decomposition;
        PCA enforces orthogonality so both activation cosine and CKA are
        near zero for all within-corpus pairs (trivially passing).
        With NMF these criteria carry real signal.
    """
    eps_cos: float = 0.3
    eps_jsd: float = 0.2
    eps_cka: float = 0.8
