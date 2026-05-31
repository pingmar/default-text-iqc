"""
Cross-corpus validation of feature differentiation.

Loads three pre-built FeatureDictionary JSON files and computes pairwise
ε-differentiation statistics, printing a summary table and saving heatmaps.

Usage:
    python scripts/validate_dictionary.py \\
        --sentiment-dict  results/features/sentiment_l6.json \\
        --syntactic-dict  results/features/syntactic_l6.json \\
        --ner-dict        results/features/ner_l6.json \\
        --layer           6 \\
        --output-dir      results/validation/
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from ssm_extract_ufe_text.analysis import (
    cross_projection_jsd,
    differentiation_matrix,
    find_most_different_pairs,
    plot_feature_histograms,
    plot_heatmap,
    rank_corpus_specific_features,
    show_feature_pair,
    summarise_cross_corpus,
)
from ssm_extract_ufe_text.config import DifferentiationThresholds
from ssm_extract_ufe_text.dictionary import FeatureDictionary


def parse_args() -> argparse.Namespace:
    _d = DifferentiationThresholds()
    p = argparse.ArgumentParser()
    p.add_argument("--sentiment-dict", required=True)
    p.add_argument("--syntactic-dict", required=True)
    p.add_argument("--ner-dict", required=True)
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--output-dir", default="results/validation")
    p.add_argument("--eps-cos", type=float, default=_d.eps_cos)
    p.add_argument("--eps-jsd", type=float, default=_d.eps_jsd)
    p.add_argument("--eps-cka", type=float, default=_d.eps_cka)
    return p.parse_args()


def _load_projections(dict_path: str, layer: int) -> np.ndarray:
    """Load the .npy projections file saved alongside the JSON dict."""
    base = os.path.splitext(dict_path)[0]
    npy_path = base + "_projections.npy"
    if not os.path.exists(npy_path):
        dirn = os.path.dirname(dict_path)
        name = os.path.basename(base)
        # strip layer tag and rebuild
        parts = name.rsplit("_", 1)[0]
        npy_path = os.path.join(dirn, f"{parts}_l{layer}_projections.npy")
    return np.load(npy_path)


def main() -> None:
    args = parse_args()
    thresholds = DifferentiationThresholds(
        eps_cos=args.eps_cos, eps_jsd=args.eps_jsd, eps_cka=args.eps_cka
    )

    dicts = {
        "sentiment": FeatureDictionary.load(args.sentiment_dict),
        "syntactic": FeatureDictionary.load(args.syntactic_dict),
        "ner":       FeatureDictionary.load(args.ner_dict),
    }
    projs = {
        "sentiment": _load_projections(args.sentiment_dict, args.layer),
        "syntactic": _load_projections(args.syntactic_dict, args.layer),
        "ner":       _load_projections(args.ner_dict, args.layer),
    }

    # Raw activation matrices (.npy saved by extract_features if --save-activations).
    # Optional: only used for cross_projection_jsd.
    def _try_load_activations(dict_path: str, layer: int) -> np.ndarray | None:
        base = os.path.splitext(dict_path)[0]
        act_path = base.rsplit("_", 1)[0] + f"_l{layer}_activations.npy"
        return np.load(act_path) if os.path.exists(act_path) else None

    os.makedirs(args.output_dir, exist_ok=True)

    # Cross-corpus summary table
    names = list(dicts.keys())
    print(f"\n{'Corpus pair':<28} {'mean_cos':>8} {'mean_jsd':>8} {'mean_cka':>8} {'pct_diff':>9}")
    print("-" * 65)
    for i, na in enumerate(names):
        for nb in names[i:]:
            stats = summarise_cross_corpus(
                dicts[na], dicts[nb], projs[na], projs[nb], thresholds, layer=args.layer
            )
            label = f"{na} x {nb}"
            print(
                f"{label:<28} {stats['mean_cos']:>8.3f} {stats['mean_jsd']:>8.3f} "
                f"{stats['mean_cka']:>8.3f} {stats['pct_different']:>8.1f}%"
            )

    # Feature Dictionary summary (semantic labels + polysemanticity)
    print(f"\nSentiment feature summary (layer {args.layer})")
    print(f"{'Feature':<10} {'label':<28} {'polysem.':>9}")
    print("-" * 50)
    for rec in sorted(dicts["sentiment"].records_for_layer(args.layer), key=lambda r: r.feature_id):
        poly = f"{rec.polysemanticity:.3f}" if rec.polysemanticity >= 0 else "n/a"
        print(f"f{rec.feature_id:<9} {rec.semantic_label or 'unlabeled':<28} {poly:>9}")

    # Per-corpus differentiation heatmap (sentiment)
    recs = dicts["sentiment"].records_for_layer(args.layer)
    proj_s = projs["sentiment"]
    if recs and proj_s.shape[1] >= len(recs):
        score_mat, _ = differentiation_matrix(recs, proj_s, thresholds)
        labels = [f"f{r.feature_id}" for r in recs]
        heatmap_path = os.path.join(args.output_dir, f"sentiment_l{args.layer}_heatmap.png")
        plot_heatmap(
            score_mat, labels, heatmap_path,
            title=f"Sentiment features (layer {args.layer}) - similarity matrix"
        )
        print(f"\nHeatmap saved -> {heatmap_path}")

        hist_path = os.path.join(args.output_dir, f"sentiment_l{args.layer}_histograms.png")
        plot_feature_histograms(recs, hist_path, title=f"Sentiment features (layer {args.layer})")
        print(f"Histograms saved  -> {hist_path}")

        # Top ε-different feature pairs showcase
        top_pairs = find_most_different_pairs(recs, proj_s, thresholds, n=3)
        if top_pairs:
            print(f"\nTop {len(top_pairs)} most ε-different sentiment feature pairs")
            for ri, rj, sc in top_pairs:
                print()
                print(show_feature_pair(ri, rj, thresholds))
                print()

    # Cross-projection JSD ranked by corpus-specificity
    # Projects syntactic and NER raw activations onto sentiment feature directions.
    # Features are ranked by mean JSD across corpora: high = sentiment-specific,
    # low = corpus-agnostic (universal Mamba-130M feature).
    act_synt = _try_load_activations(args.syntactic_dict, args.layer)
    act_ner  = _try_load_activations(args.ner_dict, args.layer)
    if act_synt is not None and act_ner is not None:
        jsd_synt = cross_projection_jsd(act_synt, dicts["sentiment"], args.layer)
        jsd_ner  = cross_projection_jsd(act_ner,  dicts["sentiment"], args.layer)
        ranked = rank_corpus_specific_features({"syntactic": jsd_synt, "ner": jsd_ner})

        print(f"\n{'Feature':<10} {'->syntactic':>12} {'->NER':>8} {'mean JSD':>10} {'specificity':>13}")
        print("-" * 57)
        for row in ranked:
            fid = row["feature_id"]
            synt_val = row.get("syntactic", float("nan"))
            ner_val  = row.get("ner", float("nan"))
            print(
                f"f{fid:<9} {synt_val:>12.3f} {ner_val:>8.3f} "
                f"{row['mean_jsd']:>10.3f} {row['specificity']:>13}"
            )
    else:
        print("\n(re-run extract_features.py with --save-activations to enable cross-projection table)")


if __name__ == "__main__":
    main()
