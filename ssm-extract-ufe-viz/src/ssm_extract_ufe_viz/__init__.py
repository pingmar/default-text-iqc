"""Visual feature differentiation tools for vision SSM experiments."""

from .config import (
    BootstrapConfig,
    ContentContextSplit,
    DifferentiationThresholds,
    DistanceWeights,
    VisualizationConfig,
)
from .controls import ControlSummary, run_controls, signal_to_noise
from .dictionary import FeatureDictionary, FeatureRecord
from .differentiation import (
    EquivalenceClass,
    classify_pair,
    differentiation_table,
    equivalence_classes,
    hierarchy_parent_jaccard,
    pairwise_pvalues,
    split_pair_scores,
)
from .metrics import composite_distance

__all__ = [
    "BootstrapConfig",
    "ContentContextSplit",
    "ControlSummary",
    "DifferentiationThresholds",
    "DistanceWeights",
    "EquivalenceClass",
    "FeatureDictionary",
    "FeatureRecord",
    "VisualizationConfig",
    "classify_pair",
    "composite_distance",
    "differentiation_table",
    "equivalence_classes",
    "hierarchy_parent_jaccard",
    "pairwise_pvalues",
    "run_controls",
    "signal_to_noise",
    "split_pair_scores",
]

__version__ = "0.1.0"
