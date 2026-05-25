import csv
import json
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = [
    {
        "name": "baseline_manual_cap_0_60",
        "suggestion": "reference",
        "justification": "Best previous robustness-oriented setting, using the original manual attack set.",
        "args": ["--recurrent-norm-cap", "0.60"],
    },
    {
        "name": "wordnet_attack_cap_0_60",
        "suggestion": "1",
        "justification": "Use WordNet synonyms to broaden the attack set beyond the small hand-written dictionary.",
        "args": ["--recurrent-norm-cap", "0.60", "--attack-source", "wordnet"],
    },
    {
        "name": "combined_attack_cap_0_60",
        "suggestion": "1",
        "justification": "Combine manual sentiment/movie synonyms with WordNet synonyms for broader lexical coverage.",
        "args": ["--recurrent-norm-cap", "0.60", "--attack-source", "combined"],
    },
    {
        "name": "combined_training_cap_0_60",
        "suggestion": "2",
        "justification": "Combine the previously strongest ideas: cap 0.60, attack augmentation, and weight decay.",
        "args": [
            "--recurrent-norm-cap",
            "0.60",
            "--attack-source",
            "combined",
            "--augment-attacks",
            "--weight-decay",
            "0.0001",
        ],
    },
    {
        "name": "cap_0_50_combined_attack",
        "suggestion": "3",
        "justification": "Tune the recurrent norm between the previously tested 0.40 and 0.60 settings.",
        "args": ["--recurrent-norm-cap", "0.50", "--attack-source", "combined"],
    },
    {
        "name": "cap_0_70_combined_attack",
        "suggestion": "3",
        "justification": "Tune the recurrent norm between the previously tested 0.60 and 0.80 settings.",
        "args": ["--recurrent-norm-cap", "0.70", "--attack-source", "combined"],
    },
    {
        "name": "adversarial_embedding_cap_0_60",
        "suggestion": "4",
        "justification": "Train with FGSM-style embedding perturbations to match the embedding-space robustness objective.",
        "args": [
            "--recurrent-norm-cap",
            "0.60",
            "--attack-source",
            "combined",
            "--adversarial-embedding-eps",
            "0.03",
        ],
    },
    {
        "name": "local_certificate_cap_0_60",
        "suggestion": "5",
        "justification": "Use local tanh slopes in the certificate to tighten the global Lipschitz bound.",
        "args": ["--recurrent-norm-cap", "0.60", "--attack-source", "combined", "--local-certificate"],
    },
    {
        "name": "combined_all_cap_0_50_local",
        "suggestion": "2-5",
        "justification": "Stress-test the combined recipe: tuned cap, attack augmentation, weight decay, adversarial embedding training, and local certificate.",
        "args": [
            "--recurrent-norm-cap",
            "0.50",
            "--attack-source",
            "combined",
            "--augment-attacks",
            "--weight-decay",
            "0.0001",
            "--adversarial-embedding-eps",
            "0.03",
            "--local-certificate",
        ],
    },
]


KEYS = [
    "clean_accuracy",
    "attacked_accuracy",
    "attack_coverage",
    "attack_success_rate_over_changed",
    "certified_robust_accuracy",
    "certified_fraction_all",
    "mean_clean_margin",
    "mean_logit_perturbation_bound",
]


def main():
    root = Path("improvement_outputs")
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
            "suggestion": exp["suggestion"],
            "justification": exp["justification"],
            "wh_norm": metrics["norm_info"]["wh_norm"],
            "wx_norm": metrics["norm_info"]["wx_norm"],
            "wo_norm": metrics["norm_info"]["wo_norm"],
            "local_certificate": metrics["norm_info"].get("local_certificate", False),
        }
        for key in KEYS:
            row[key] = metrics[key]
        rows.append(row)

    (root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved improvement summary")


if __name__ == "__main__":
    main()
