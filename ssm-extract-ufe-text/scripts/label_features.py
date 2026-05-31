"""
Assign semantic labels and polysemanticity scores to a FeatureDictionary.

Loads a pre-built FeatureDictionary JSON file, runs probe sentences through the
model to assign semantic labels, computes polysemanticity scores from the saved
projection matrix, and writes an updated JSON in-place (or to --output).

Usage:
    python scripts/label_features.py \\
        --dict-path     results/features/sentiment_l6.json \\
        --layer         6 \\
        --model-name    state-spaces/mamba-130m-hf \\
        --device        auto
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from ssm_extract_ufe_text.analysis import polysemanticity_score
from ssm_extract_ufe_text.config import ExtractionConfig
from ssm_extract_ufe_text.dictionary import FeatureDictionary
from ssm_extract_ufe_text.probing import assign_semantic_labels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dict-path", required=True, help="Path to FeatureDictionary JSON")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--model-name", default="state-spaces/mamba-130m-hf")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--min-z-score", type=float, default=0.5)
    p.add_argument("--top-k-poly", type=int, default=10,
                   help="Top-k examples to use for polysemanticity scoring")
    p.add_argument("--output", default=None,
                   help="Output path (defaults to overwriting --dict-path)")
    return p.parse_args()


def _load_projections(dict_path: str, layer: int) -> np.ndarray | None:
    base = os.path.splitext(dict_path)[0]
    npy_path = base + "_projections.npy"
    if not os.path.exists(npy_path):
        dirn = os.path.dirname(dict_path)
        name = os.path.basename(base)
        parts = name.rsplit("_", 1)[0]
        npy_path = os.path.join(dirn, f"{parts}_l{layer}_projections.npy")
    if os.path.exists(npy_path):
        return np.load(npy_path)
    return None


def main() -> None:
    args = parse_args()

    fd = FeatureDictionary.load(args.dict_path)
    print(f"Loaded {len(fd)} records from {args.dict_path}")

    config = ExtractionConfig(
        model_name=args.model_name,
        layers=[args.layer],
        batch_size=args.batch_size,
        device=args.device,
    )

    print(f"Loading model {args.model_name} for semantic probing...")
    from ssm_extract_ufe_text.model import MambaProbe
    probe = MambaProbe.from_pretrained(config)

    print("Assigning semantic labels via probe sentences...")
    assign_semantic_labels(probe, fd, args.layer, args.batch_size, args.min_z_score)

    projections = _load_projections(args.dict_path, args.layer)
    if projections is not None:
        print(f"Computing polysemanticity scores (top_k={args.top_k_poly})...")
        records = fd.records_for_layer(args.layer)
        for rec in records:
            rec.polysemanticity = polysemanticity_score(
                projections, rec.feature_id, top_k=args.top_k_poly
            )
    else:
        print("No projections .npy found - skipping polysemanticity scoring.")
        print("Re-run extract_features.py to generate projection files.")

    out_path = args.output or args.dict_path
    fd.save(out_path)
    print(f"Saved updated dictionary -> {out_path}")

    records = fd.records_for_layer(args.layer)
    print(f"\n{'Feature':<10} {'label':<28} {'polysem.':>9}")
    print("-" * 50)
    for rec in sorted(records, key=lambda r: r.feature_id):
        poly = f"{rec.polysemanticity:.3f}" if rec.polysemanticity >= 0 else "n/a"
        print(f"f{rec.feature_id:<9} {rec.semantic_label or 'unlabeled':<28} {poly:>9}")


if __name__ == "__main__":
    main()
