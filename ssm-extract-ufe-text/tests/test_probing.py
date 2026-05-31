"""
Unit tests for probing.py and the new analysis functions:
  polysemanticity_score, show_feature_pair, find_most_different_pairs.
No GPU, no model download required.
"""
from __future__ import annotations

import numpy as np
import pytest

from ssm_extract_ufe_text.analysis import (
    find_most_different_pairs,
    polysemanticity_score,
    show_feature_pair,
)
from ssm_extract_ufe_text.config import DifferentiationThresholds
from ssm_extract_ufe_text.dictionary import FeatureRecord
from ssm_extract_ufe_text.probing import (
    PROBE_SENTENCES,
    assign_labels_from_activations,
    get_probe_categories,
)


# Helpers
def _make_record(
    fid: int,
    vector: list[float],
    hist_counts: list[float] | None = None,
    label: str = "",
    polysem: float = -1.0,
) -> FeatureRecord:
    if hist_counts is None:
        hist_counts = [1.0] * 10
    edges = list(range(len(hist_counts) + 1))
    return FeatureRecord(
        feature_id=fid,
        layer=6,
        vector=vector,
        top_k_examples=[f"example {i} for f{fid}" for i in range(5)],
        activation_histogram=(edges, hist_counts),
        semantic_label=label,
        polysemanticity=polysem,
    )



# PROBE_SENTENCES structure
def test_probe_sentences_categories():
    cats = get_probe_categories()
    assert len(cats) == 8


def test_probe_sentences_min_length():
    for cat, sents in PROBE_SENTENCES.items():
        assert len(sents) >= 10, f"Category {cat} has fewer than 10 sentences"


def test_probe_sentences_all_non_empty():
    for cat, sents in PROBE_SENTENCES.items():
        for s in sents:
            assert isinstance(s, str) and len(s) > 0


def test_get_probe_categories_matches_probe_sentences():
    assert set(get_probe_categories()) == set(PROBE_SENTENCES.keys())



# assign_labels_from_activations
def test_assign_labels_empty_activations():
    rec = _make_record(0, [1.0, 0.0])
    assign_labels_from_activations({}, [rec])
    assert rec.semantic_label == ""  # unchanged


def test_assign_labels_zero_vector():
    acts = {"cat_a": np.ones((5, 2)), "cat_b": np.zeros((5, 2))}
    rec = _make_record(0, [0.0, 0.0])
    assign_labels_from_activations(acts, [rec])
    assert rec.semantic_label == "unknown"


def test_assign_labels_clear_winner():
    d = 4
    # cat_a mean activation strongly aligns with v = [1,0,0,0]
    acts_a = np.zeros((10, d))
    acts_a[:, 0] = 5.0
    acts_b = np.zeros((10, d))
    acts_b[:, 1] = 0.1  # small activation in different direction

    rec = _make_record(0, [1.0, 0.0, 0.0, 0.0])
    assign_labels_from_activations({"cat_a": acts_a, "cat_b": acts_b}, [rec])
    assert rec.semantic_label == "cat_a"


def test_assign_labels_mixed_when_all_equal():
    d = 2
    # Both categories project equally onto v = [1, 0]
    acts = np.ones((5, d))
    rec = _make_record(0, [1.0, 0.0])
    assign_labels_from_activations({"cat_a": acts, "cat_b": acts}, [rec])
    assert rec.semantic_label == "mixed"


def test_assign_labels_low_z_score_gives_mixed():
    d = 2
    # Near-equal projections → max z-score below threshold
    rng = np.random.default_rng(42)
    acts_a = rng.normal(1.0, 0.01, (20, d))
    acts_b = rng.normal(1.0, 0.01, (20, d))
    rec = _make_record(0, [1.0, 0.0])
    assign_labels_from_activations({"cat_a": acts_a, "cat_b": acts_b}, [rec], min_z_score=0.5)
    assert rec.semantic_label in ("mixed", "cat_a", "cat_b")


def test_assign_labels_multiple_records():
    d = 3
    acts_a = np.zeros((5, d)); acts_a[:, 0] = 10.0
    acts_b = np.zeros((5, d)); acts_b[:, 1] = 10.0
    category_activations = {"cat_a": acts_a, "cat_b": acts_b}

    rec0 = _make_record(0, [1.0, 0.0, 0.0])
    rec1 = _make_record(1, [0.0, 1.0, 0.0])
    assign_labels_from_activations(category_activations, [rec0, rec1])
    assert rec0.semantic_label == "cat_a"
    assert rec1.semantic_label == "cat_b"



# polysemanticity_score


def test_polysemanticity_score_monosemantic():
    # All rows identical → all pairwise cos = 1 → score = 0
    projections = np.tile([1.0, 0.0, 0.0], (20, 1))
    score = polysemanticity_score(projections, feature_col=0, top_k=10)
    assert 0.0 <= score <= 0.05


def test_polysemanticity_score_polysemantic():
    # Diverse context vectors → low mean cosine → score near 1
    rng = np.random.default_rng(0)
    projections = rng.standard_normal((50, 8))
    # Make feature_col 0 activate uniformly so top-k contexts are diverse
    projections[:, 0] = 1.0
    score = polysemanticity_score(projections, feature_col=0, top_k=10)
    assert score > 0.3


def test_polysemanticity_score_returns_minus1_on_too_few_rows():
    projections = np.ones((1, 3))
    score = polysemanticity_score(projections, feature_col=0)
    assert score == -1.0


def test_polysemanticity_score_returns_minus1_on_bad_col():
    projections = np.ones((10, 3))
    score = polysemanticity_score(projections, feature_col=5)
    assert score == -1.0


def test_polysemanticity_score_in_unit_range():
    rng = np.random.default_rng(7)
    projections = rng.standard_normal((30, 6))
    for col in range(6):
        score = polysemanticity_score(projections, feature_col=col, top_k=8)
        assert -1.0 <= score <= 1.0



# show_feature_pair
def test_show_feature_pair_contains_feature_ids():
    thresholds = DifferentiationThresholds()
    ri = _make_record(3, [1.0, 0.0, 0.0])
    rj = _make_record(7, [0.0, 1.0, 0.0])
    output = show_feature_pair(ri, rj, thresholds)
    assert "f3" in output
    assert "f7" in output


def test_show_feature_pair_shows_status():
    thresholds = DifferentiationThresholds(eps_cos=0.9, eps_jsd=0.0, eps_cka=1.0)
    ri = _make_record(0, [1.0, 0.0, 0.0], hist_counts=[1.0] + [0.0] * 9)
    rj = _make_record(1, [0.0, 1.0, 0.0], hist_counts=[0.0] * 5 + [1.0] + [0.0] * 4)
    output = show_feature_pair(ri, rj, thresholds)
    assert "DIFFERENT" in output or "similar" in output


def test_show_feature_pair_shows_labels_when_set():
    thresholds = DifferentiationThresholds()
    ri = _make_record(0, [1.0, 0.0], label="positive_sentiment", polysem=0.12)
    rj = _make_record(1, [0.0, 1.0], label="negation", polysem=0.88)
    output = show_feature_pair(ri, rj, thresholds)
    assert "positive_sentiment" in output
    assert "negation" in output
    assert "0.120" in output or "0.88" in output


def test_show_feature_pair_shows_top_k_examples():
    thresholds = DifferentiationThresholds()
    ri = _make_record(0, [1.0, 0.0])
    rj = _make_record(1, [0.0, 1.0])
    output = show_feature_pair(ri, rj, thresholds)
    assert "example 0 for f0" in output
    assert "example 0 for f1" in output



# find_most_different_pairs
def _make_pair_setup():
    rng = np.random.default_rng(99)
    F, N = 6, 100
    # Build projections with clearly different activation profiles
    projections = np.zeros((N, F))
    for i in range(F):
        start = i * (N // F)
        end = start + (N // F)
        projections[start:end, i] = 5.0

    records = []
    for i in range(F):
        # Alternating histogram shapes for JSD signal
        counts = [float(j % 2) + 0.1 for j in range(10)]
        if i % 2 == 1:
            counts = counts[::-1]
        records.append(_make_record(i, [float(k == i) for k in range(F)], hist_counts=counts))

    return records, projections


def test_find_most_different_pairs_returns_list():
    records, projections = _make_pair_setup()
    thresholds = DifferentiationThresholds(eps_cos=0.9, eps_jsd=0.05, eps_cka=1.0)
    pairs = find_most_different_pairs(records, projections, thresholds, n=3)
    assert isinstance(pairs, list)


def test_find_most_different_pairs_all_epsilon_different():
    records, projections = _make_pair_setup()
    thresholds = DifferentiationThresholds(eps_cos=0.9, eps_jsd=0.05, eps_cka=1.0)
    pairs = find_most_different_pairs(records, projections, thresholds, n=10)
    for ri, rj, sc in pairs:
        assert sc["cos_pass"] and sc["jsd_pass"] and sc["cka_pass"]


def test_find_most_different_pairs_sorted_by_jsd():
    records, projections = _make_pair_setup()
    thresholds = DifferentiationThresholds(eps_cos=0.9, eps_jsd=0.05, eps_cka=1.0)
    pairs = find_most_different_pairs(records, projections, thresholds, n=5)
    jsds = [sc["jsd"] for _, _, sc in pairs]
    assert jsds == sorted(jsds, reverse=True)


def test_find_most_different_pairs_respects_n():
    records, projections = _make_pair_setup()
    thresholds = DifferentiationThresholds(eps_cos=0.9, eps_jsd=0.05, eps_cka=1.0)
    pairs = find_most_different_pairs(records, projections, thresholds, n=2)
    assert len(pairs) <= 2


def test_find_most_different_pairs_empty_when_none_different():
    records = [_make_record(i, [1.0, 0.0]) for i in range(3)]
    projections = np.ones((10, 3))
    thresholds = DifferentiationThresholds(eps_cos=0.01, eps_jsd=0.99, eps_cka=0.0)
    pairs = find_most_different_pairs(records, projections, thresholds, n=5)
    assert pairs == []
