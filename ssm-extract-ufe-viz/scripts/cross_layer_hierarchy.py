"""Quantify shared shallow-feature parents between deep-layer features.

If two deep-layer features both share top-k support with the same
shallow-layer feature, they likely rely on a common lower-level dependency
("wheels for car and bicycle"). `shared_parent_rate` reports how often this
happens across all deep pairs.
"""

from __future__ import annotations

import argparse
import json

from ssm_extract_ufe_viz.analysis import cross_layer_overlap
from ssm_extract_ufe_viz.dictionary import FeatureDictionary, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shallow", required=True, help="Dictionary JSON from the earlier layer.")
    parser.add_argument("--deep", required=True, help="Dictionary JSON from the later layer.")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--top-k-limit", type=int)
    parser.add_argument("--output-path", default="results/hierarchy.json")
    args = parser.parse_args()

    shallow = FeatureDictionary.load(args.shallow)
    deep = FeatureDictionary.load(args.deep)
    result = cross_layer_overlap(
        shallow.records, deep.records,
        threshold=args.threshold,
        top_k_limit=args.top_k_limit,
    )
    write_json(args.output_path, result)
    summary = {k: v for k, v in result.items() if k != "parent_index"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
