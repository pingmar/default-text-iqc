import numpy as np
import pytest

from ssm_extract_ufe_text.analysis import plot_feature_histograms, plot_token_feature_heatmap
from ssm_extract_ufe_text.dictionary import FeatureRecord


def _make_record(feature_id: int, semantic_label: str = "", polysemanticity: float = -1.0) -> FeatureRecord:
    edges = list(np.linspace(0.0, 1.0, 6))
    counts = [0.2, 0.2, 0.2, 0.2, 0.2]
    return FeatureRecord(
        feature_id=feature_id,
        layer=0,
        vector=[1.0, 0.0],
        top_k_examples=[],
        activation_histogram=(edges, counts),
        semantic_label=semantic_label,
        polysemanticity=polysemanticity,
    )


# plot_feature_histograms

def test_plot_histograms_creates_file(tmp_path):
    records = [_make_record(i) for i in range(3)]
    out = str(tmp_path / "hist.png")
    plot_feature_histograms(records, out)
    assert (tmp_path / "hist.png").exists()


def test_plot_histograms_empty_no_error(tmp_path):
    out = str(tmp_path / "empty.png")
    plot_feature_histograms([], out)
    assert not (tmp_path / "empty.png").exists()


def test_plot_histograms_clips_to_max_features(tmp_path):
    records = [_make_record(i) for i in range(20)]
    out = str(tmp_path / "clipped.png")
    plot_feature_histograms(records, out, max_features=8)
    assert (tmp_path / "clipped.png").exists()


def test_plot_histograms_polysemanticity_shown(tmp_path):
    records = [_make_record(0, polysemanticity=0.75)]
    out = str(tmp_path / "poly.png")
    plot_feature_histograms(records, out)
    assert (tmp_path / "poly.png").exists()


def test_plot_histograms_uses_semantic_label(tmp_path):
    records = [_make_record(0, semantic_label="negation")]
    out = str(tmp_path / "labeled.png")
    plot_feature_histograms(records, out)
    assert (tmp_path / "labeled.png").exists()


# plot_token_feature_heatmap
def test_plot_token_heatmap_creates_file(tmp_path):
    T, D, K = 8, 16, 4
    acts = np.random.default_rng(0).standard_normal((T, D)).astype(np.float32)
    comps = np.random.default_rng(1).standard_normal((K, D)).astype(np.float32)
    tokens = [f"tok{i}" for i in range(T)]
    labels = [f"f{i}" for i in range(K)]
    out = str(tmp_path / "heatmap.png")
    plot_token_feature_heatmap(acts, comps, tokens, labels, out)
    assert (tmp_path / "heatmap.png").exists()


def test_plot_token_heatmap_top_k_clips(tmp_path):
    T, D, K = 6, 16, 10
    acts = np.random.default_rng(2).standard_normal((T, D)).astype(np.float32)
    comps = np.random.default_rng(3).standard_normal((K, D)).astype(np.float32)
    out = str(tmp_path / "clip.png")
    plot_token_feature_heatmap(acts, comps, [f"t{i}" for i in range(T)],
                               [f"f{i}" for i in range(K)], out, top_k_features=4)
    assert (tmp_path / "clip.png").exists()


def test_plot_token_heatmap_top_k_exceeds_k(tmp_path):
    T, D, K = 4, 8, 3
    acts = np.random.default_rng(4).standard_normal((T, D)).astype(np.float32)
    comps = np.random.default_rng(5).standard_normal((K, D)).astype(np.float32)
    out = str(tmp_path / "exceed.png")
    plot_token_feature_heatmap(acts, comps, [f"t{i}" for i in range(T)],
                               [f"f{i}" for i in range(K)], out, top_k_features=100)
    assert (tmp_path / "exceed.png").exists()
