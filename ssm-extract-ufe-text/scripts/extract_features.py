"""
Extract features from a Mamba SSM for a single text corpus.

Usage:
    python scripts/extract_features.py \\
        --corpus sentiment \\
        --layers 6 \\
        --n-components 32 \\
        --output-dir results/features/
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from ssm_extract_ufe_text.config import ExtractionConfig
from ssm_extract_ufe_text.corpus import load_ner_corpus, load_sentiment_corpus, load_syntactic_corpus
from ssm_extract_ufe_text.dictionary import FeatureDictionary
from ssm_extract_ufe_text.features import decompose_nmf, decompose_pca

CORPUS_LOADERS = {
    "sentiment": load_sentiment_corpus,
    "ner": load_ner_corpus,
    "syntactic": load_syntactic_corpus,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", choices=list(CORPUS_LOADERS), required=True)
    p.add_argument("--layers", nargs="+", type=int, default=[6])
    p.add_argument("--n-components", type=int, default=32)
    p.add_argument("--decomposition", choices=["pca", "nmf"], default="pca")
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output-dir", default="results/features")
    p.add_argument("--model-name", default="state-spaces/mamba-130m-hf")
    p.add_argument("--device", default="auto")
    p.add_argument("--save-activations", action="store_true",
                   help="Save raw activation matrices as .npy (needed for cross_projection_jsd)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = ExtractionConfig(
        model_name=args.model_name,
        layers=args.layers,
        batch_size=args.batch_size,
        decomposition=args.decomposition,
        n_components=args.n_components,
        device=args.device,
    )

    print(f"Loading corpus: {args.corpus}")
    loader_fn = CORPUS_LOADERS[args.corpus]
    loader, texts = loader_fn(
        args.model_name,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    print(f"Loading model: {args.model_name}")
    from ssm_extract_ufe_text.model import MambaProbe
    probe = MambaProbe.from_pretrained(config)

    print("Collecting activations...")
    raw = probe.collect(loader)

    components_by_layer: dict[int, np.ndarray] = {}
    projections_by_layer: dict[int, np.ndarray] = {}
    decompose = decompose_pca if args.decomposition == "pca" else decompose_nmf

    raw_activations: dict[int, np.ndarray] = {}
    for layer, A in raw.items():
        A_np = A.numpy() if hasattr(A, "numpy") else np.array(A)
        print(f"Layer {layer}: decomposing A {A_np.shape}")
        comps, projs = decompose(A_np, n_components=args.n_components, seed=config.seed)
        components_by_layer[layer] = comps
        projections_by_layer[layer] = projs
        raw_activations[layer] = A_np

    fd = FeatureDictionary.build(components_by_layer, projections_by_layer, texts, config)

    os.makedirs(args.output_dir, exist_ok=True)
    layer_tag = "_".join(f"l{l}" for l in sorted(components_by_layer))
    out_path = os.path.join(args.output_dir, f"{args.corpus}_{layer_tag}.json")
    fd.save(out_path)
    print(f"Saved {len(fd)} feature records -> {out_path}")

    for layer, projs in projections_by_layer.items():
        np_path = os.path.join(args.output_dir, f"{args.corpus}_l{layer}_projections.npy")
        np.save(np_path, projs)
        print(f"Saved projections -> {np_path}")

    if args.save_activations:
        for layer, A_np in raw_activations.items():
            act_path = os.path.join(args.output_dir, f"{args.corpus}_l{layer}_activations.npy")
            np.save(act_path, A_np)
            print(f"Saved activations -> {act_path}")


if __name__ == "__main__":
    main()
