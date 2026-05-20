"""Aggregate Vim study runs into CSV, JSON, and Markdown summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

FIELDS = [
    "dataset",
    "method",
    "k",
    "seed",
    "layer",
    "n_samples",
    "different_pairs",
    "total_pairs",
    "distance_mean",
    "snr",
    "topk_overlap_mean",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results_imagenet_subset")
    parser.add_argument("--output-prefix", default="study_summary")
    args = parser.parse_args()

    root = Path(args.results_root)
    rows = collect_rows(root)
    json_path = root / f"{args.output_prefix}.json"
    csv_path = root / f"{args.output_prefix}.csv"
    md_path = root / f"{args.output_prefix}.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"wrote {len(rows)} rows")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {md_path}")


def collect_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.rglob("validation/summary.json")):
        summary = _load_json(summary_path)
        validation_dir = summary_path.parent
        confidence = _load_json(validation_dir / "confidence_intervals.json", default={})
        topk = confidence.get("topk_overlap", {})
        rows.append({
            "dataset": summary.get("dataset", ""),
            "method": summary.get("decomposition", ""),
            "k": summary.get("n_components", ""),
            "seed": summary.get("seed", ""),
            "layer": summary.get("layer", ""),
            "n_samples": summary.get("n_samples", ""),
            "different_pairs": summary.get("different_pairs", ""),
            "total_pairs": summary.get("total_pairs", ""),
            "distance_mean": summary.get("distance_mean", ""),
            "snr": summary.get("snr", ""),
            "topk_overlap_mean": topk.get("estimate", ""),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| Method | K | Seed | Layer | SNR | Gate | Distance |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        gate = f"{row['different_pairs']} / {row['total_pairs']}"
        lines.append(
            f"| {row['method']} | {row['k']} | {row['seed']} | {row['layer']} | "
            f"{_fmt(row['snr'])} | {gate} | "
            f"{_fmt(row['distance_mean'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    if value == "":
        return ""
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
