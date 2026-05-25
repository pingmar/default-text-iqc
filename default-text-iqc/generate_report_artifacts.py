import json
from pathlib import Path


ARTIFACTS = Path("report_artifacts")
ARTIFACTS.mkdir(exist_ok=True)


def pct(x):
    return 100 * float(x)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bar_svg(rows, metric, title, out_path, width=920, height=420):
    margin_left, margin_right, margin_top, margin_bottom = 190, 30, 50, 90
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    values = [pct(r[metric]) for r in rows]
    max_v = max(max(values), 1)
    bar_gap = 8
    bar_h = (plot_h - bar_gap * (len(rows) - 1)) / len(rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top+plot_h}" stroke="#222"/>',
        f'<line x1="{margin_left}" y1="{margin_top+plot_h}" x2="{margin_left+plot_w}" y2="{margin_top+plot_h}" stroke="#222"/>',
    ]
    for tick in range(0, int(max_v) + 11, 10):
        x = margin_left + plot_w * tick / max_v
        parts.append(f'<line x1="{x:.1f}" y1="{margin_top+plot_h}" x2="{x:.1f}" y2="{margin_top+plot_h+5}" stroke="#222"/>')
        parts.append(f'<text x="{x:.1f}" y="{margin_top+plot_h+22}" text-anchor="middle" font-family="Arial" font-size="11">{tick}%</text>')
    for i, r in enumerate(rows):
        y = margin_top + i * (bar_h + bar_gap)
        w = plot_w * values[i] / max_v
        color = r.get("color", "#2f6f9f")
        parts.append(f'<text x="{margin_left-8}" y="{y+bar_h*0.65:.1f}" text-anchor="end" font-family="Arial" font-size="12">{r["label"]}</text>')
        parts.append(f'<rect x="{margin_left}" y="{y:.1f}" width="{w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{margin_left+w+6:.1f}" y="{y+bar_h*0.65:.1f}" font-family="Arial" font-size="12">{values[i]:.1f}%</text>')
    parts.append('</svg>')
    Path(out_path).write_text("\n".join(parts), encoding="utf-8")


def line_svg(points, title, out_path, width=720, height=420):
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 50, 60
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    max_y = max(ys) + 5
    def sx(x):
        return margin_left + (x - min_x) / (max_x - min_x) * plot_w
    def sy(y):
        return margin_top + plot_h - y / max_y * plot_h
    poly = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top+plot_h}" stroke="#222"/>',
        f'<line x1="{margin_left}" y1="{margin_top+plot_h}" x2="{margin_left+plot_w}" y2="{margin_top+plot_h}" stroke="#222"/>',
        f'<polyline points="{poly}" fill="none" stroke="#2f6f9f" stroke-width="3"/>',
    ]
    for x, y in points:
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="#d1495b"/>')
        parts.append(f'<text x="{sx(x):.1f}" y="{sy(y)-9:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{y:.1f}%</text>')
        parts.append(f'<text x="{sx(x):.1f}" y="{margin_top+plot_h+22}" text-anchor="middle" font-family="Arial" font-size="11">{x:.2f}</text>')
    parts.append(f'<text x="{width/2}" y="{height-14}" text-anchor="middle" font-family="Arial" font-size="12">Recurrent norm cap</text>')
    parts.append('</svg>')
    Path(out_path).write_text("\n".join(parts), encoding="utf-8")


def main():
    ablation = load_json("ablation_outputs/summary.json")
    improvement = load_json("improvement_outputs/summary.json")
    final_metrics = load_json("improvement_outputs/adversarial_embedding_cap_0_60_local_50k/metrics.json")

    cap_points = []
    for r in ablation:
        name = r["name"]
        if name.startswith("cap_0_"):
            cap = float(name.replace("cap_0_", "0."))
            cap_points.append((cap, pct(r["certified_robust_accuracy"])))
    cap_points.sort()
    line_svg(cap_points, "Certified Robust Accuracy vs Recurrent Norm Cap", ARTIFACTS / "cap_vs_certified.svg")

    selected_names = [
        ("standard_rnn", "Standard RNN"),
        ("cap_0_60", "Cap 0.60"),
        ("adversarial_embedding_cap_0_60_local", "Adv emb + local"),
        ("adversarial_embedding_cap_0_60_local_50k", "Final 50k"),
    ]
    lookup = {r["name"]: r for r in ablation + improvement}
    rows = []
    colors = ["#8d99ae", "#2f6f9f", "#4f772d", "#d1495b"]
    for (name, label), color in zip(selected_names, colors):
        r = lookup[name]
        rows.append({"label": label, "clean_accuracy": r["clean_accuracy"], "certified_robust_accuracy": r["certified_robust_accuracy"], "attacked_accuracy": r["attacked_accuracy"], "color": color})
    bar_svg(rows, "clean_accuracy", "Clean Accuracy by Method", ARTIFACTS / "clean_accuracy.svg")
    bar_svg(rows, "certified_robust_accuracy", "Certified Robust Accuracy by Method", ARTIFACTS / "certified_accuracy.svg")
    bar_svg(rows, "attacked_accuracy", "Attacked Accuracy by Method", ARTIFACTS / "attacked_accuracy.svg")

    examples = final_metrics["examples"][:8]
    lines = [
        "| Original | Attacked | Label | Clean p(pos) | Attacked p(pos) | Certified |",
        "|---|---|---:|---:|---:|---|",
    ]
    for ex in examples:
        original = ex["text"].replace("|", "/")
        attacked = ex["attacked_text"].replace("|", "/")
        lines.append(
            f"| {original} | {attacked} | {ex['label']} | {ex['clean_prob_positive']:.3f} | {ex['attacked_prob_positive']:.3f} | {ex['certified']} |"
        )
    (ARTIFACTS / "example_predictions.md").write_text("\n".join(lines), encoding="utf-8")

    print("Generated report artifacts in", ARTIFACTS)


if __name__ == "__main__":
    main()
