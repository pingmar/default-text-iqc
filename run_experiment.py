import argparse
import math
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as st
from ssm_model import (
    make_hippo_legs,
    svd_low_rank,
    FrozenSSM,
    FrozenSSMLowRankEfficient,
)
from tasks import associative_recall_task, long_range_regression_task, evaluate_memory

PALETTE = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
]
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.family": "DejaVu Sans",
    }
)

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)


def savefig(fig, name: str):
    fig.savefig(OUT_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {name}")


def relative_error(A, A_approx):
    return np.linalg.norm(A - A_approx, "fro") / np.linalg.norm(A, "fro")


def correlation_ci(r, n, confidence=0.95):
    if n <= 3:
        return r, r
    z = np.arctanh(np.clip(r, -0.999, 0.999))
    se = 1.0 / math.sqrt(n - 3)
    z_crit = st.norm.ppf((1 + confidence) / 2)
    return np.tanh(z - z_crit * se), np.tanh(z + z_crit * se)


# ── Exp 1: Спектр A ──────────────────────────────────────────────────────────
def exp_svd_spectrum(N: int):
    print("\n[1] SVD Spectrum")
    A = make_hippo_legs(N)
    _, S, _ = np.linalg.svd(A, full_matrices=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].semilogy(np.arange(1, N + 1), S, color=PALETTE[0])
    axes[0].set_title("Spectrum")

    cumvar = np.cumsum(S**2) / np.sum(S**2)
    axes[1].plot(np.arange(1, N + 1), cumvar * 100, color=PALETTE[1])
    axes[1].set_title("Variance %")

    ranks = list(range(1, N + 1, max(1, N // 20)))
    errors = [relative_error(A, svd_low_rank(A, r)[0]) for r in ranks]
    axes[2].plot(ranks, errors, color=PALETTE[2], marker="o")
    axes[2].set_title("Error")
    savefig(fig, "1_svd_spectrum.png")

    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
    for ax, r in zip(axes2, [4, N // 4, N // 2]):
        im = ax.imshow(svd_low_rank(A, r)[0], cmap="RdBu_r")
        ax.set_title(f"Rank {r}")
    savefig(fig2, "1b_matrix_vis.png")
    return S


# ── Exp 2: Швидкість ─────────────────────────────────────────────────────────
def exp_inference_speed(
    N: int, seq_lens: list[int], ranks: list[int], n_runs: int = 10
):
    print("\n[2] Speed Benchmark")
    results = {T: {} for T in seq_lens}
    for T in seq_lens:
        u = np.random.default_rng().standard_normal(T).astype(np.float32)
        _, t_full, _ = FrozenSSM(N, rank=None).timed_forward(u, n_runs)
        results[T]["full"] = t_full
        for r in ranks:
            _, t_lr, s_lr = FrozenSSMLowRankEfficient(N, rank=r).timed_forward(
                u, n_runs
            )
            results[T][r] = (t_lr, s_lr)
            print(f"  T={T} r={r} speedup={t_full/t_lr:.2f}x")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    speed_mat = np.array(
        [[results[T]["full"] / results[T][r][0] for r in ranks] for T in seq_lens]
    )
    im = axes[0].imshow(speed_mat, cmap="YlGn")
    axes[0].set_title("Speedup Heatmap")
    plt.colorbar(im, ax=axes[0])

    for i, T in enumerate(seq_lens):
        axes[1].plot(
            ranks, [results[T][r][0] for r in ranks], marker="o", label=f"T={T}"
        )
    axes[1].legend()
    axes[1].set_title("Absolute Time")
    savefig(fig, "2_inference_speed.png")
    return results


# ── Exp 3: Пам'ять ──────────────────────────────────────────────────────────
def exp_temporal_memory(N: int, ranks: list[int], seq_len: int, n_samples: int):
    print("\n[3] Memory Quality")
    tasks = [
        ("AR", associative_recall_task, {"vocab_size": 8}),
        ("Regr", long_range_regression_task, {"signal_window": 16}),
    ]

    baseline = {}
    for name, fn, kw in tasks:
        m = evaluate_memory(FrozenSSM(N, rank=None), fn, n_samples, seq_len, **kw)
        m["ci"] = correlation_ci(m["correlation"], n_samples)
        baseline[name] = m

    lr_res = {name: {} for name, *_ in tasks}
    for r in ranks:
        ssm = FrozenSSM(N, rank=r)
        for name, fn, kw in tasks:
            m = evaluate_memory(ssm, fn, n_samples, seq_len, **kw)
            m["ci"] = correlation_ci(m["correlation"], n_samples)
            lr_res[name][r] = m

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (name, *_) in zip(axes, tasks):
        corrs = [lr_res[name][r]["correlation"] for r in ranks]
        mses = [lr_res[name][r]["mse"] for r in ranks]
        mse_sems = [lr_res[name][r]["mse_sem"] for r in ranks]
        
        ax.plot(ranks, corrs, marker="o", label="LR Corr", color=PALETTE[0])
        ax.axhline(baseline[name]["correlation"], ls=":", color=PALETTE[0], label="Full Corr")
        
        ax2 = ax.twinx()
        ax2.plot(ranks, mses, marker="s", ls="--", label="LR MSE", color=PALETTE[1])
        ax2.errorbar(ranks, mses, yerr=mse_sems, fmt="none", ecolor=PALETTE[1], alpha=0.5)
        ax2.axhline(baseline[name]["mse"], ls=":", color=PALETTE[1], label="Full MSE")
        
        ax.set_title(name)
        ax.set_ylabel("Correlation", color=PALETTE[0])
        ax2.set_ylabel("MSE", color=PALETTE[1])
        ax.tick_params(axis='y', labelcolor=PALETTE[0])
        ax2.tick_params(axis='y', labelcolor=PALETTE[1])
    
    fig.tight_layout()
    savefig(fig, "3_temporal_memory.png")
    return baseline, lr_res


# ── Exp 4: Стійкість ─────────────────────────────────────────────────────────
def exp_robustness(N, ranks, train_len, test_lens, n_samples):
    print("\n[4] Robustness")
    res = {}
    for cfg in ["full"] + ranks:
        res[str(cfg)] = {}
        ssm = FrozenSSM(N, rank=(None if cfg == "full" else cfg))
        for T in test_lens:
            m = evaluate_memory(
                ssm, long_range_regression_task, n_samples, T, signal_window=16
            )
            m["ci"] = correlation_ci(m["correlation"], n_samples)
            res[str(cfg)][T] = m

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, cfg in enumerate(["full"] + ranks):
        vals = [res[str(cfg)][T]["correlation"] for T in test_lens]
        ax.plot(test_lens, vals, label=f"r={cfg}", marker="o")
    ax.axvline(train_len, ls=":")
    ax.set_title("Robustness: Correlation vs T")
    ax.set_xlabel("Sequence Length T")
    ax.set_ylabel("Correlation")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "4_robustness.png")
    return res


# ── Exp 5: FLOP ──────────────────────────────────────────────────────────────
def exp_flop_analysis(N, ranks):
    print("\n[5] FLOP Analysis")
    data = [{"rank": r, "theoretical_speedup": (2 * N**2) / (4 * N * r)} for r in ranks]
    fig, ax = plt.subplots()
    ax.plot(ranks, [d["theoretical_speedup"] for d in data], marker="o")
    savefig(fig, "5_flop_analysis.png")
    return data


# ── Exp 6: Heatmap ───────────────────────────────────────────────────────────
def exp_degradation_heatmap(N, ranks, seq_lens, n_samples):
    print("\n[6] Degradation Heatmap")
    mat = np.zeros((len(seq_lens), len(ranks)))
    for i, T in enumerate(seq_lens):
        for j, r in enumerate(ranks):
            mat[i, j] = evaluate_memory(
                FrozenSSM(N, rank=r), associative_recall_task, n_samples, T
            )["correlation"]
    fig, ax = plt.subplots()
    im = ax.imshow(mat, cmap="YlOrRd")
    plt.colorbar(im)
    savefig(fig, "6_degradation_heatmap.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=64)
    args = parser.parse_args()
    N, T, R = args.N, 256, [2, 4, 8, 16, 32]

    S = exp_svd_spectrum(N)
    speed = exp_inference_speed(N, [64, 256, 1024], R)
    base_m, lr_m = exp_temporal_memory(N, R, T, n_samples=300)
    robust = exp_robustness(N, R[:3], T, [64, 256, 512, 1024], n_samples=100)
    flops = exp_flop_analysis(N, R)
    exp_degradation_heatmap(N, R, [64, 256, 512], n_samples=100)


if __name__ == "__main__":
    main()
