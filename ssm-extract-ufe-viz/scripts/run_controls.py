"""Run identical/perturbed/real distance controls on a saved dictionary."""

from __future__ import annotations

import argparse
import json

from ssm_extract_ufe_viz.config import DistanceWeights
from ssm_extract_ufe_viz.controls import run_controls, signal_to_noise
from ssm_extract_ufe_viz.dictionary import FeatureDictionary, write_json


def _decomposition_for_records(records) -> str:
    methods = {str(record.decomposition).lower() for record in records if record.decomposition}
    return next(iter(methods)) if len(methods) == 1 else "mixed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict-path", required=True)
    parser.add_argument("--output-path", default="results/controls.json")
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dictionary = FeatureDictionary.load(args.dict_path)
    weights = DistanceWeights.for_decomposition(_decomposition_for_records(dictionary.records))
    summaries = run_controls(
        dictionary.records,
        weights=weights,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    output = {key: summary.to_dict() for key, summary in summaries.items()}
    output["signal_to_noise"] = signal_to_noise(summaries)

    write_json(args.output_path, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
