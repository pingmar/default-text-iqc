import csv
import json
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = [
    {
        "name": "standard_rnn",
        "justification": "Reference point: ordinary RNN training without a certifiability constraint.",
        "args": [],
    },
    {
        "name": "cap_0_95",
        "justification": "Mild recurrent norm cap; tests whether a small stability constraint improves certificates with little accuracy loss.",
        "args": ["--recurrent-norm-cap", "0.95"],
    },
    {
        "name": "cap_0_80",
        "justification": "Moderate recurrent norm cap; previously looked promising because it prevents recurrent perturbation growth.",
        "args": ["--recurrent-norm-cap", "0.8"],
    },
    {
        "name": "cap_0_60",
        "justification": "Stronger recurrent norm cap; tests whether certificate gains continue or accuracy collapses.",
        "args": ["--recurrent-norm-cap", "0.6"],
    },
    {
        "name": "cap_0_40",
        "justification": "Very strong recurrent norm cap; stress test for the accuracy/certification trade-off.",
        "args": ["--recurrent-norm-cap", "0.4"],
    },
    {
        "name": "hidden_32_cap_0_80",
        "justification": "Smaller hidden state may reduce sensitivity and overfitting while keeping the same certificate logic.",
        "args": ["--hidden-dim", "32", "--recurrent-norm-cap", "0.8"],
    },
    {
        "name": "hidden_96_cap_0_80",
        "justification": "Larger hidden state may improve clean margins, but could increase sensitivity.",
        "args": ["--hidden-dim", "96", "--recurrent-norm-cap", "0.8"],
    },
    {
        "name": "more_data_cap_0_80",
        "justification": "More training data may improve clean accuracy and margins without changing the certificate.",
        "args": ["--train-limit", "50000", "--recurrent-norm-cap", "0.8"],
    },
    {
        "name": "attack_aug_cap_0_80",
        "justification": "Training on attacked text should improve empirical attacked accuracy and may improve margins near substitutions.",
        "args": ["--recurrent-norm-cap", "0.8", "--augment-attacks"],
    },
    {
        "name": "noise_cap_0_80",
        "justification": "Embedding noise should smooth the model around token embeddings, which may improve robustness.",
        "args": ["--recurrent-norm-cap", "0.8", "--embedding-noise-std", "0.03"],
    },
    {
        "name": "weight_decay_cap_0_80",
        "justification": "Weight decay may reduce operator norms and therefore reduce the certified perturbation bound.",
        "args": ["--recurrent-norm-cap", "0.8", "--weight-decay", "0.0001"],
    },
]


KEYS = [
    "clean_accuracy",
    "attacked_accuracy",
    "attack_success_rate_over_changed",
    "certified_robust_accuracy",
    "certified_fraction_all",
    "mean_clean_margin",
    "mean_logit_perturbation_bound",
]


def main():
    root = Path("ablation_outputs")
    root.mkdir(exist_ok=True)
    rows = []
    for exp in EXPERIMENTS:
        out_dir = root / exp["name"]
        cmd = [
            sys.executable,
            "robustness_rnn_experiment.py",
            "--epochs",
            "3",
            "--train-limit",
            "20000",
            "--lr",
            "0.003",
            "--out-dir",
            str(out_dir),
            *exp["args"],
        ]
        print("Running", exp["name"])
        subprocess.run(cmd, check=True)
        metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        row = {
            "name": exp["name"],
            "justification": exp["justification"],
            "wh_norm": metrics["norm_info"]["wh_norm"],
            "wx_norm": metrics["norm_info"]["wx_norm"],
            "wo_norm": metrics["norm_info"]["wo_norm"],
        }
        for key in KEYS:
            row[key] = metrics[key]
        rows.append(row)

    summary_json = root / "summary.json"
    summary_csv = root / "summary.csv"
    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved", summary_json, "and", summary_csv)


if __name__ == "__main__":
    main()
