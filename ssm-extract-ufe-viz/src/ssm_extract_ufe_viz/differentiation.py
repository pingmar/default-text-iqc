"""Hierarchy-aware feature differentiation.

This module formalizes "two features are different" by splitting the
per-axis pair scores into a *content* group (direction and optimized
visualization disagreement) and a *context* group (whether they activate
on different inputs / regions). See ``docs/formal_definition.md`` for
the written definition.

The split classifies a pair as one of four relationships ("distinct",
"siblings", "redundant", "echo"), exposes equivalence classes whose
medoids form the visual dictionary, and computes per-pair empirical
p-values against the noise-perturbation null already used by
``controls.perturbed_pair_control``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .analysis import compute_pair_scores
from .config import ContentContextSplit, DistanceWeights
from .controls import _perturb
from .dictionary import FeatureRecord

PairLabel = Literal["distinct", "siblings", "redundant", "echo"]
DEFAULT_CONTENT_CONTEXT_SPLIT = ContentContextSplit()
DEFAULT_DISTANCE_WEIGHTS = DistanceWeights()


_RAW_TO_DISTANCE = {
    "cos": lambda v: 1.0 - float(v),
    "cka": lambda v: 1.0 - float(v),
    "map_iou": lambda v: 1.0 - float(v),
    "topk_overlap": lambda v: 1.0 - float(v),
    "parent_jaccard": lambda v: 1.0 - float(v),
    "jsd": lambda v: float(v),
    "lpips": lambda v: float(v),
    "ssim_distance": lambda v: float(v),
}


@dataclass(frozen=True)
class EquivalenceClass:
    """A connected component of features that are not content-different."""

    class_id: int
    members: tuple[int, ...]
    representative: int
    within_class_max: float
    within_class_p95: float


def _axis_distance(axis: str, scores: Mapping[str, Any]) -> float | None:
    value = scores.get(axis)
    if value is None:
        return None
    converter = _RAW_TO_DISTANCE.get(axis)
    if converter is None:
        return float(value)
    return converter(value)


def _group_distance(
    scores: Mapping[str, Any],
    axes: tuple[str, ...],
    axis_weights: Mapping[str, float],
) -> tuple[float, list[str]]:
    """Weighted-mean distance over the axes that are present in ``scores``."""

    numer = 0.0
    denom = 0.0
    used: list[str] = []
    for axis in axes:
        contribution = _axis_distance(axis, scores)
        if contribution is None:
            continue
        weight = float(axis_weights.get(axis, 1.0))
        if weight <= 0.0:
            continue
        numer += weight * contribution
        denom += weight
        used.append(axis)
    if denom <= 0.0:
        return 0.0, used
    return numer / denom, used


def split_pair_scores(
    scores: Mapping[str, Any],
    split: ContentContextSplit = DEFAULT_CONTENT_CONTEXT_SPLIT,
) -> dict[str, Any]:
    """Return ``{"d_content", "d_context", "axes_used": {"content": [...], "context": [...]}}``.

    Each group is a weight-normalized mean over the axes that the
    underlying score dict actually provides, so absent axes are dropped
    (not zero-imputed) — the same convention as ``composite_distance``.
    """

    d_content, content_used = _group_distance(scores, split.content_axes, split.axis_weights)
    d_context, context_used = _group_distance(scores, split.context_axes, split.axis_weights)
    return {
        "d_content": d_content,
        "d_context": d_context,
        "axes_used": {"content": content_used, "context": context_used},
    }


def classify_pair(
    d_content: float,
    d_context: float,
    split: ContentContextSplit = DEFAULT_CONTENT_CONTEXT_SPLIT,
) -> PairLabel:
    """Classify a pair into one of the four hierarchy-aware quadrants."""

    high_content = d_content > split.eps_content
    high_context = d_context > split.eps_context
    if high_content and high_context:
        return "distinct"
    if high_content and not high_context:
        return "siblings"
    if not high_content and not high_context:
        return "redundant"
    return "echo"


def hierarchy_parent_jaccard(
    deep_id_i: int,
    deep_id_j: int,
    parent_index: Mapping[int | str, list[int]],
) -> float | None:
    """Jaccard of parent sets for two deep features.

    ``parent_index`` is the mapping produced by ``analysis.cross_layer_overlap``
    (deep feature id → list of shallow parent ids). Keys may be int or str
    because that helper stringifies for JSON serialization. If neither
    feature has parent evidence, return ``None`` so callers can drop the axis
    instead of treating "unknown" as shared context.
    """

    def _lookup(key: int) -> set[int]:
        if key in parent_index:
            return set(parent_index[key])  # type: ignore[index]
        skey = str(key)
        if skey in parent_index:
            return set(parent_index[skey])  # type: ignore[index]
        return set()

    parents_i = _lookup(deep_id_i)
    parents_j = _lookup(deep_id_j)
    if not parents_i and not parents_j:
        return None
    union = parents_i | parents_j
    if not union:
        return None
    return float(len(parents_i & parents_j) / len(union))


def _augment_with_parents(
    scores: dict[str, Any],
    rec_i: FeatureRecord,
    rec_j: FeatureRecord,
    parent_index: Mapping[int | str, list[int]] | None,
) -> dict[str, Any]:
    if parent_index is None:
        return scores
    scores = dict(scores)
    parent_jaccard = hierarchy_parent_jaccard(
        rec_i.feature_id, rec_j.feature_id, parent_index
    )
    if parent_jaccard is not None:
        scores["parent_jaccard"] = parent_jaccard
    return scores


def differentiation_table(
    records: list[FeatureRecord],
    *,
    split: ContentContextSplit = DEFAULT_CONTENT_CONTEXT_SPLIT,
    weights: DistanceWeights = DEFAULT_DISTANCE_WEIGHTS,
    parent_index: Mapping[int | str, list[int]] | None = None,
    pair_scores: list[list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Build a per-pair classification table.

    If ``pair_scores`` is provided (the score grid returned by
    ``analysis.differentiation_matrix``), no per-axis computation is
    repeated. Otherwise we call ``compute_pair_scores`` for each (i,j).

    Returns a dict with:
        - ``rows``: list of per-pair records (feature_i, feature_j,
          d_content, d_context, label, axes_used, raw_scores)
        - ``content_matrix``, ``context_matrix``: [n,n] float arrays
        - ``labels``: list of length len(records)*(len(records)-1)/2
    """

    n = len(records)
    content_matrix = np.zeros((n, n), dtype=np.float64)
    context_matrix = np.zeros((n, n), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    labels: list[PairLabel] = []
    for i in range(n):
        for j in range(i + 1, n):
            if pair_scores is not None and pair_scores[i][j]:
                scores = dict(pair_scores[i][j])
            else:
                scores = compute_pair_scores(records[i], records[j])
            scores = _augment_with_parents(scores, records[i], records[j], parent_index)
            split_result = split_pair_scores(scores, split)
            d_content = split_result["d_content"]
            d_context = split_result["d_context"]
            label = classify_pair(d_content, d_context, split)
            content_matrix[i, j] = content_matrix[j, i] = d_content
            context_matrix[i, j] = context_matrix[j, i] = d_context
            labels.append(label)
            rows.append(
                {
                    "feature_i": records[i].feature_id,
                    "feature_j": records[j].feature_id,
                    "index_i": i,
                    "index_j": j,
                    "d_content": d_content,
                    "d_context": d_context,
                    "label": label,
                    "axes_used": split_result["axes_used"],
                    "raw_scores": {
                        key: value
                        for key, value in scores.items()
                        if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool)
                    },
                }
            )
    return {
        "rows": rows,
        "content_matrix": content_matrix,
        "context_matrix": context_matrix,
        "labels": labels,
    }


def _union_find(n: int) -> list[int]:
    return list(range(n))


def _find(parent: list[int], x: int) -> int:
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


def _union(parent: list[int], a: int, b: int) -> None:
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[rb] = ra


def equivalence_classes(
    records: list[FeatureRecord],
    *,
    split: ContentContextSplit = DEFAULT_CONTENT_CONTEXT_SPLIT,
    weights: DistanceWeights = DEFAULT_DISTANCE_WEIGHTS,
    parent_index: Mapping[int | str, list[int]] | None = None,
    pair_scores: list[list[dict[str, Any]]] | None = None,
    table: dict[str, Any] | None = None,
) -> list[EquivalenceClass]:
    """Group records into equivalence classes by the d_content <= eps_content edge.

    Each class's representative is the medoid (minimum total d_content to
    other class members; the unique member for singletons). Reusing the
    ``table`` returned by ``differentiation_table`` avoids recomputing the
    per-pair content distances.
    """

    if table is None:
        table = differentiation_table(
            records,
            split=split,
            weights=weights,
            parent_index=parent_index,
            pair_scores=pair_scores,
        )
    content_matrix = np.asarray(table["content_matrix"], dtype=np.float64)
    n = len(records)
    parent = _union_find(n)
    for i in range(n):
        for j in range(i + 1, n):
            if content_matrix[i, j] <= split.eps_content:
                _union(parent, i, j)

    components: dict[int, list[int]] = {}
    for i in range(n):
        root = _find(parent, i)
        components.setdefault(root, []).append(i)

    classes: list[EquivalenceClass] = []
    for class_id, (_, members) in enumerate(sorted(components.items())):
        members_sorted = sorted(members)
        if len(members_sorted) == 1:
            rep = members_sorted[0]
            within_max = 0.0
            within_p95 = 0.0
        else:
            sub = content_matrix[np.ix_(members_sorted, members_sorted)]
            row_sums = sub.sum(axis=1)
            rep = members_sorted[int(np.argmin(row_sums))]
            triu = sub[np.triu_indices_from(sub, k=1)]
            within_max = float(triu.max()) if triu.size else 0.0
            within_p95 = float(np.quantile(triu, 0.95)) if triu.size else 0.0
        classes.append(
            EquivalenceClass(
                class_id=class_id,
                members=tuple(records[idx].feature_id for idx in members_sorted),
                representative=int(records[rep].feature_id),
                within_class_max=within_max,
                within_class_p95=within_p95,
            )
        )
    return classes


def _activation_axis_distance(
    rec_i: FeatureRecord,
    rec_j: FeatureRecord,
    split: ContentContextSplit,
) -> float | None:
    """Distance restricted to axes computable from (vector, histogram) alone.

    This matches the axes the perturbation null actually varies, so the
    observed and null distance distributions are functions of the same
    inputs. Returns ``None`` only when no eligible axis is available.
    """

    eligible = ("cos", "jsd")
    eligible_present = tuple(a for a in eligible if a in set(split.content_axes) | set(split.context_axes))
    if not eligible_present:
        return None
    scores = compute_pair_scores(rec_i, rec_j)
    numer = 0.0
    denom = 0.0
    for axis in eligible_present:
        contribution = _axis_distance(axis, scores)
        if contribution is None:
            continue
        weight = float(split.axis_weights.get(axis, 1.0))
        numer += weight * contribution
        denom += weight
    if denom <= 0.0:
        return None
    return numer / denom


def pairwise_pvalues(
    records: list[FeatureRecord],
    *,
    split: ContentContextSplit = DEFAULT_CONTENT_CONTEXT_SPLIT,
    noise_std: float = 0.05,
    n_resample: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    """Empirical right-tail p-values against the perturbation null.

    Null: for each record we draw ``n_resample`` noise-perturbed copies
    (using the same perturbation as ``controls.perturbed_pair_control``)
    and collect the activation-axis distance d_act(record, copy). This is
    a distribution of d_act under H0: "i and j are the same feature, up
    to noise". For every real pair we report

        p(i, j) = (1 + #{x in null : x >= d_act(i, j)}) / (1 + |null|)

    Only axes derivable from (vector, histogram) participate, so the
    observed and null distances are matched on which axes are summed.
    Visual-axis disagreement is reported separately by the content/context
    split and is *not* folded into the p-value (the perturbation does not
    re-optimize the visualization). The returned effect size is the observed
    activation-axis distance standardized by the shared null distribution; it
    rescales ``d_act`` in null standard-deviation units and is not an
    independent statistical signal.
    """

    if n_resample < 1:
        raise ValueError("n_resample must be >= 1")
    rng = np.random.default_rng(seed)
    null_samples: list[float] = []
    for record in records:
        for _ in range(n_resample):
            copy = _perturb(record, noise_std, rng)
            d = _activation_axis_distance(record, copy, split)
            if d is not None:
                null_samples.append(d)
    null_array = np.asarray(null_samples, dtype=np.float64)
    n_null = null_array.size
    null_mean = float(null_array.mean()) if n_null else float("nan")
    null_std = float(null_array.std()) if n_null else float("nan")
    p_floor = 1.0 / (1.0 + n_null) if n_null else float("nan")

    n = len(records)
    p_matrix = np.ones((n, n), dtype=np.float64)
    observed = np.zeros((n, n), dtype=np.float64)
    effect_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d_obs = _activation_axis_distance(records[i], records[j], split)
            if d_obs is None:
                p_matrix[i, j] = p_matrix[j, i] = float("nan")
                effect_matrix[i, j] = effect_matrix[j, i] = float("nan")
                continue
            observed[i, j] = observed[j, i] = d_obs
            if n_null == 0:
                p_value = 1.0
            else:
                exceed = int(np.sum(null_array >= d_obs))
                p_value = (1.0 + exceed) / (1.0 + n_null)
            p_matrix[i, j] = p_matrix[j, i] = p_value
            if n_null == 0 or np.isnan(null_std):
                effect = float("nan")
            elif null_std <= 1e-12:
                effect = float("inf") if d_obs > null_mean else 0.0
            else:
                effect = (d_obs - null_mean) / null_std
            effect_matrix[i, j] = effect_matrix[j, i] = effect
    return {
        "p_matrix": p_matrix,
        "observed_d_act": observed,
        "effect_matrix": effect_matrix,
        "null_samples": null_array,
        "null_mean": null_mean,
        "null_std": null_std,
        "n_null": int(n_null),
        "p_floor": p_floor,
        "alpha": split.alpha,
    }
