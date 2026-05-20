"""Validate feature differentiation metrics and run controls for a saved dictionary."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ssm_extract_ufe_viz.analysis import (
    differentiation_matrix,
    plot_heatmap,
    summarize_metric_confidence,
)
from ssm_extract_ufe_viz.config import (
    BootstrapConfig,
    ContentContextSplit,
    DifferentiationThresholds,
    DistanceWeights,
)
from ssm_extract_ufe_viz.controls import run_controls, signal_to_noise
from ssm_extract_ufe_viz.dictionary import FeatureDictionary, write_json
from ssm_extract_ufe_viz.differentiation import (
    differentiation_table,
    equivalence_classes,
    pairwise_pvalues,
)


def _try_plot_feature_grid(paths, labels, output_path, n_cols):
    try:
        from ssm_extract_ufe_viz.visualization import plot_feature_grid
    except ImportError as exc:
        return False, f"plotting unavailable ({exc})"
    plot_feature_grid(paths, labels, output_path, n_cols=n_cols)
    return True, "written"


def _existing_viz_paths(records, indices):
    paths: list[str] = []
    labels: list[str] = []
    missing: list[int] = []
    for idx in indices:
        rec = records[idx]
        if rec.visualization_path and Path(rec.visualization_path).exists():
            paths.append(rec.visualization_path)
            labels.append(f"f{rec.feature_id}")
        else:
            missing.append(rec.feature_id)
    return paths, labels, missing


def _plot_pair_grid(records, pair_rows, output_path, n_pairs):
    if not pair_rows:
        return False, "no pairs"
    selected = pair_rows[:n_pairs]
    paths: list[str] = []
    labels: list[str] = []
    for row in selected:
        i, j = row["index_i"], row["index_j"]
        rec_i, rec_j = records[i], records[j]
        if not (
            rec_i.visualization_path
            and rec_j.visualization_path
            and Path(rec_i.visualization_path).exists()
            and Path(rec_j.visualization_path).exists()
        ):
            continue
        paths.extend([rec_i.visualization_path, rec_j.visualization_path])
        labels.extend(
            [
                f"f{rec_i.feature_id}\nd_content={row['d_content']:.2f}",
                f"f{rec_j.feature_id}\nd_context={row['d_context']:.2f}",
            ]
        )
    if not paths:
        return False, "no visualizations available"
    ok, status = _try_plot_feature_grid(paths, labels, output_path, n_cols=2)
    if not ok:
        return False, status
    return True, f"{len(paths) // 2} pairs"


def _load_parent_index(path: str | None):
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    parent_index = data.get("parent_index", data)
    if not isinstance(parent_index, dict):
        raise ValueError("parent index must be a JSON object or contain parent_index")
    return {
        str(key): [int(parent) for parent in value]
        for key, value in parent_index.items()
    }


def _heatmap_vmax(matrix) -> float:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.size == 0:
        return 1.0
    return max(1.0, float(arr.max()))


def _decomposition_for_records(records) -> str:
    methods = {str(record.decomposition).lower() for record in records if record.decomposition}
    return next(iter(methods)) if len(methods) == 1 else "mixed"


def _write_report(
    output_dir: Path,
    *,
    n_features: int,
    n_pairs: int,
    label_counts: Counter,
    controls,
    snr: float,
    p_values_flat: np.ndarray,
    effect_values_flat: np.ndarray,
    alpha: float,
    p_floor: float,
    n_null: int,
    classes,
    legacy_different_pairs: int,
    parent_index_path: str | None,
    artifacts: dict[str, str],
) -> None:
    p_finite = p_values_flat[np.isfinite(p_values_flat)]
    if p_finite.size:
        mean_p = float(p_finite.mean())
        frac_sig = float((p_finite < alpha).mean())
    else:
        mean_p = float("nan")
        frac_sig = float("nan")
    effect_finite = effect_values_flat[np.isfinite(effect_values_flat)]
    mean_effect = float(effect_finite.mean()) if effect_finite.size else float("nan")
    max_effect = float(effect_finite.max()) if effect_finite.size else float("nan")
    lines = []
    lines.append("# Feature differentiation report\n")
    lines.append(
        f"Features: **{n_features}** · pairs: **{n_pairs}** · "
        f"equivalence classes: **{len(classes)}** "
        f"(compression: {len(classes) / n_features:.2f})\n"
    )
    lines.append("## Hierarchy-aware classification\n")
    if parent_index_path:
        lines.append(f"Parent index: `{parent_index_path}`.\n")
    else:
        lines.append("Parent index: not supplied; `parent_jaccard` axis was dropped.\n")
    lines.append("| label | count | meaning |")
    lines.append("|---|---:|---|")
    meanings = {
        "distinct": "content-different, different inputs",
        "siblings": "content-different, shared inputs/parents",
        "redundant": "content-similar, same context",
        "echo": "content-similar, disjoint inputs",
    }
    for label in ("distinct", "siblings", "redundant", "echo"):
        lines.append(f"| {label} | {label_counts.get(label, 0)} | {meanings[label]} |")
    lines.append("")
    lines.append(
        f"Legacy `epsilon_different` gate: **{legacy_different_pairs}** "
        f"of {n_pairs} pairs flagged.\n"
    )
    lines.append("## Significance vs. perturbation null\n")
    lines.append(
        f"α = {alpha:.3f} · mean p-value = {mean_p:.3f} · "
        f"fraction significant = {frac_sig:.3f} · "
        f"p-floor = {p_floor:.6f} (n_null={n_null})\n"
    )
    lines.append(
        "Effect size standardizes `d_act` by the shared perturbation null; "
        f"mean = {mean_effect:.3f} · max = {max_effect:.3f}. "
        "It is a rescale of the observed activation-axis distance, not an independent test.\n"
    )
    lines.append("## Controls\n")
    lines.append(
        f"identical = {controls['identical'].mean:.3f} · "
        f"perturbed = {controls['perturbed'].mean:.3f} · "
        f"real = {controls['real'].mean:.3f} · "
        f"SNR = {snr:.2f}\n"
    )
    lines.append("## Visual dictionary (equivalence-class representatives)\n")
    if classes:
        if len(classes) == n_features:
            lines.append("No compression observed: every feature is its own equivalence class.\n")
        lines.append("| class | size | representative | within-class max | p95 |")
        lines.append("|---:|---:|---:|---:|---:|")
        for cls in classes:
            lines.append(
                f"| {cls.class_id} | {len(cls.members)} | f{cls.representative} | "
                f"{cls.within_class_max:.3f} | {cls.within_class_p95:.3f} |"
            )
        lines.append("")
    lines.append("## Artifacts\n")
    for name, descr in artifacts.items():
        lines.append(f"- `{name}` — {descr}")
    lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict-path", required=True)
    parser.add_argument("--output-dir", default="results/validation")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="significance level for per-pair p-values")
    parser.add_argument("--eps-content", type=float, default=None,
                        help="override ContentContextSplit.eps_content")
    parser.add_argument("--eps-context", type=float, default=None,
                        help="override ContentContextSplit.eps_context")
    parser.add_argument("--noise-std", type=float, default=0.05,
                        help="perturbation std for the p-value null")
    parser.add_argument("--n-resample", type=int, default=4,
                        help="perturbed copies per record for the null")
    parser.add_argument("--grid-top-n", type=int, default=6,
                        help="number of pairs in extremes_*.png")
    parser.add_argument("--parent-index-path",
                        help="optional cross_layer_hierarchy JSON with parent_index")
    args = parser.parse_args()

    dictionary = FeatureDictionary.load(args.dict_path)
    records = dictionary.records
    decomposition = _decomposition_for_records(records)
    thresholds = DifferentiationThresholds()
    weights = DistanceWeights.for_decomposition(decomposition)
    split_kwargs = {"alpha": args.alpha}
    if args.eps_content is not None:
        split_kwargs["eps_content"] = args.eps_content
    if args.eps_context is not None:
        split_kwargs["eps_context"] = args.eps_context
    split = ContentContextSplit.for_decomposition(decomposition, **split_kwargs)
    parent_index = _load_parent_index(args.parent_index_path)

    bool_matrix, dist_matrix, score_grid = differentiation_matrix(records, thresholds, weights)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_heatmap(bool_matrix, output_dir / "heatmap.png", title="differentiation (1 = different)")
    plot_heatmap(
        dist_matrix, output_dir / "distance_heatmap.png",
        title="composite pair distance",
        vmin=0.0, vmax=float(dist_matrix.max()) if dist_matrix.size else 1.0,
    )

    pair_scores: list[dict] = []
    with (output_dir / "pairwise_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for i, row in enumerate(score_grid):
            for j, scores in enumerate(row):
                if j <= i or not scores:
                    continue
                record = {"feature_i": i, "feature_j": j, **scores, "different": bool(bool_matrix[i, j])}
                pair_scores.append(scores)
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(record), lineterminator="\n")
                    writer.writeheader()
                writer.writerow(record)

    summary = summarize_metric_confidence(
        pair_scores,
        bootstrap=BootstrapConfig(n_bootstrap=args.bootstrap, seed=args.seed),
    )
    write_json(output_dir / "confidence_intervals.json", summary)

    controls = run_controls(records, weights)
    snr = signal_to_noise(controls)
    write_json(
        output_dir / "controls.json",
        {key: value.to_dict() for key, value in controls.items()},
    )

    table = differentiation_table(
        records,
        split=split,
        weights=weights,
        parent_index=parent_index,
        pair_scores=score_grid,
    )
    pval_result = pairwise_pvalues(
        records,
        split=split,
        noise_std=args.noise_std,
        n_resample=args.n_resample,
        seed=args.seed,
    )
    p_matrix = pval_result["p_matrix"]
    effect_matrix = pval_result["effect_matrix"]

    plot_heatmap(
        table["content_matrix"], output_dir / "content_heatmap.png",
        title="d_content (content-like distance)",
        vmin=0.0, vmax=_heatmap_vmax(table["content_matrix"]),
    )
    plot_heatmap(
        table["context_matrix"], output_dir / "context_heatmap.png",
        title="d_context (shared-inputs distance)",
        vmin=0.0, vmax=_heatmap_vmax(table["context_matrix"]),
    )

    with (output_dir / "pairwise_classification.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "feature_i", "feature_j", "d_content", "d_context",
            "label", "p_value", "effect_size", "different_legacy", "different_hierarchy_aware",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in table["rows"]:
            i, j = row["index_i"], row["index_j"]
            p_value = float(p_matrix[i, j])
            effect_size = float(effect_matrix[i, j])
            writer.writerow({
                "feature_i": row["feature_i"],
                "feature_j": row["feature_j"],
                "d_content": f"{row['d_content']:.6f}",
                "d_context": f"{row['d_context']:.6f}",
                "label": row["label"],
                "p_value": f"{p_value:.6f}" if np.isfinite(p_value) else "",
                "effect_size": f"{effect_size:.6f}" if np.isfinite(effect_size) else "",
                "different_legacy": int(bool_matrix[i, j]),
                "different_hierarchy_aware": int(row["d_content"] > split.eps_content),
            })

    classes = equivalence_classes(
        records,
        split=split,
        weights=weights,
        parent_index=parent_index,
        table=table,
    )
    write_json(
        output_dir / "equivalence_classes.json",
        {
            "n_features": len(records),
            "n_classes": len(classes),
            "eps_content": split.eps_content,
            "classes": [
                {
                    "class_id": cls.class_id,
                    "members": list(cls.members),
                    "representative": cls.representative,
                    "within_class_max": cls.within_class_max,
                    "within_class_p95": cls.within_class_p95,
                }
                for cls in classes
            ],
        },
    )

    artifacts = {
        "heatmap.png": "legacy boolean differentiation matrix",
        "distance_heatmap.png": "legacy composite distance",
        "content_heatmap.png": "d_content per pair (content-like distance)",
        "context_heatmap.png": "d_context per pair (shared-inputs distance)",
        "pairwise_metrics.csv": "all per-axis scores per pair (legacy)",
        "pairwise_classification.csv": "hierarchy-aware label and p-value per pair",
        "equivalence_classes.json": "visual dictionary entries",
        "confidence_intervals.json": "bootstrap CIs for every pairwise metric",
        "controls.json": "identical/perturbed/real control summaries",
    }
    if args.parent_index_path:
        artifacts["parent_index"] = f"loaded from {args.parent_index_path}"

    rep_indices = []
    feature_id_to_index = {rec.feature_id: idx for idx, rec in enumerate(records)}
    for cls in classes:
        idx = feature_id_to_index.get(cls.representative)
        if idx is not None:
            rep_indices.append(idx)
    paths, labels, missing = _existing_viz_paths(records, rep_indices)
    if paths:
        ok, status = _try_plot_feature_grid(paths, labels, output_dir / "visual_dictionary.png", n_cols=4)
        artifacts["visual_dictionary.png"] = (
            f"grid of equivalence-class representatives ({len(paths)} of {len(classes)})"
            if ok else f"skipped ({status})"
        )
    elif rep_indices:
        artifacts["visual_dictionary.png"] = "skipped (no visualization_path on records)"

    distinct_rows = sorted(
        [row for row in table["rows"] if row["label"] == "distinct"],
        key=lambda r: r["d_content"],
        reverse=True,
    )
    sibling_rows = sorted(
        [row for row in table["rows"] if row["label"] == "siblings"],
        key=lambda r: r["d_content"],
        reverse=True,
    )
    ok, status = _plot_pair_grid(records, distinct_rows, output_dir / "extremes_distinct.png", args.grid_top_n)
    artifacts["extremes_distinct.png"] = (
        f"top-{args.grid_top_n} most distinct pairs ({status})" if ok else f"skipped ({status})"
    )
    ok, status = _plot_pair_grid(records, sibling_rows, output_dir / "extremes_siblings.png", args.grid_top_n)
    artifacts["extremes_siblings.png"] = (
        f"top-{args.grid_top_n} sibling pairs ({status})" if ok else f"skipped ({status})"
    )

    label_counts = Counter(row["label"] for row in table["rows"])
    n_pairs = len(table["rows"])
    legacy_different_pairs = int(bool_matrix.sum() // 2)

    p_values_flat = p_matrix[np.triu_indices_from(p_matrix, k=1)]
    effect_values_flat = effect_matrix[np.triu_indices_from(effect_matrix, k=1)]

    _write_report(
        output_dir,
        n_features=len(records),
        n_pairs=n_pairs,
        label_counts=label_counts,
        controls=controls,
        snr=snr,
        p_values_flat=p_values_flat,
        effect_values_flat=effect_values_flat,
        alpha=split.alpha,
        p_floor=float(pval_result["p_floor"]),
        n_null=int(pval_result["n_null"]),
        classes=classes,
        legacy_different_pairs=legacy_different_pairs,
        parent_index_path=args.parent_index_path,
        artifacts=artifacts,
    )

    print(f"different pairs (gate): {legacy_different_pairs}")
    print(
        f"controls: identical={controls['identical'].mean:.3f}  "
        f"perturbed={controls['perturbed'].mean:.3f}  "
        f"real={controls['real'].mean:.3f}  "
        f"SNR={snr:.2f}"
    )
    print(
        "classification: "
        + " ".join(f"{label}={label_counts.get(label, 0)}" for label in ("distinct", "siblings", "redundant", "echo"))
    )
    print(f"equivalence classes: {len(classes)} / {len(records)} features")


if __name__ == "__main__":
    main()
