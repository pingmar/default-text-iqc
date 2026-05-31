"""Run the medium ImageFolder study grid for Vim feature differentiation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_LAYERS = "layers.12,layers.16,layers.18,layers.20,layers.22"
DEFAULT_SEEDS = "0,1,2"
DEFAULT_COMPONENTS = "16,32"
DEFAULT_METHODS = "pca,ica"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", required=True, help="ImageFolder validation root.")
    parser.add_argument("--output-root", default="results_imagenet_subset")
    parser.add_argument("--dataset-name", default="imagenet_subset")
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--components", default=DEFAULT_COMPONENTS)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--viz-steps", type=int, default=200)
    parser.add_argument("--viz-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device")
    parser.add_argument("--hierarchy-thresholds", default="0.2")
    parser.add_argument("--hierarchy-top-k-limits", default="5,10,20")
    parser.add_argument("--skip-hierarchy", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    layers = _csv_str(args.layers)
    seeds = _csv_int(args.seeds)
    components = _csv_int(args.components)
    methods = _csv_str(args.methods)
    thresholds = _csv_float(args.hierarchy_thresholds)
    top_k_limits = _csv_int(args.hierarchy_top_k_limits)
    output_root = Path(args.output_root)

    scripts_dir = Path(__file__).resolve().parent
    run_demo = scripts_dir / "run_demo_vim.py"
    hierarchy = scripts_dir / "cross_layer_hierarchy.py"

    dict_paths: dict[tuple[str, int, int, str], Path] = {}
    for method in methods:
        for k in components:
            for seed in seeds:
                for layer in layers:
                    layer_dir = layer.replace(".", "_")
                    out = output_root / method / f"k{k}" / f"seed{seed}" / layer_dir
                    summary_path = out / "validation" / "summary.json"
                    if args.skip_existing and summary_path.exists():
                        print(f"[skip] {out}")
                    else:
                        cmd = [
                            sys.executable, "-u", str(run_demo),
                            "--image-root", args.image_root,
                            "--dataset-name", args.dataset_name,
                            "--layer", layer,
                            "--output-dir", str(out),
                            "--max-samples", str(args.max_samples),
                            "--n-components", str(k),
                            "--decomposition", method,
                            "--top-k", str(args.top_k),
                            "--seed", str(seed),
                            "--viz-steps", str(args.viz_steps),
                            "--viz-size", str(args.viz_size),
                            "--batch-size", str(args.batch_size),
                            "--num-workers", str(args.num_workers),
                        ]
                        if args.samples_per_class is not None:
                            cmd += ["--samples-per-class", str(args.samples_per_class)]
                        if args.device:
                            cmd += ["--device", args.device]
                        _run(cmd)

                    dict_path = _find_single(out / "features", "*.json")
                    dict_paths[(method, k, seed, layer)] = dict_path

                if not args.skip_hierarchy:
                    for shallow, deep in zip(layers, layers[1:], strict=False):
                        shallow_path = dict_paths[(method, k, seed, shallow)]
                        deep_path = dict_paths[(method, k, seed, deep)]
                        for threshold in thresholds:
                            for top_k_limit in top_k_limits:
                                out = (
                                    output_root / method / f"k{k}" / f"seed{seed}" /
                                    "hierarchy" / f"{shallow.replace('.', '_')}_to_{deep.replace('.', '_')}" /
                                    f"thr{threshold:g}_top{top_k_limit}.json"
                                )
                                if args.skip_existing and out.exists():
                                    print(f"[skip] {out}")
                                    continue
                                _run([
                                    sys.executable, "-u", str(hierarchy),
                                    "--shallow", str(shallow_path),
                                    "--deep", str(deep_path),
                                    "--threshold", str(threshold),
                                    "--top-k-limit", str(top_k_limit),
                                    "--output-path", str(out),
                                ])


def _csv_str(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_int(value: str) -> list[int]:
    return [int(item) for item in _csv_str(value)]


def _csv_float(value: str) -> list[float]:
    return [float(item) for item in _csv_str(value)]


def _find_single(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {pattern} under {root}, found {len(matches)}")
    return matches[0]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
