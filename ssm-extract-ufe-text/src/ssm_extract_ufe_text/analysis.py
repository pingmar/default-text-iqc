from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ssm_extract_ufe_text.config import DifferentiationThresholds
from ssm_extract_ufe_text.dictionary import FeatureDictionary, FeatureRecord
from ssm_extract_ufe_text.metrics import cosine_similarity, js_divergence, linear_cka


def epsilon_different(
    rec_i: FeatureRecord,
    rec_j: FeatureRecord,
    thresholds: DifferentiationThresholds,
) -> tuple[bool, dict[str, float]]:
    """
    Formal ε-differentiation test using only FeatureRecord data (no raw projections).

    Cosine is computed between feature direction vectors. For PCA this is
    trivially ≈0 (PCA enforces orthogonality), so cos_pass is always True and
    only the JSD criterion discriminates.  With NMF all three criteria carry signal.
    Use differentiation_matrix() when raw projections are available - it replaces
    direction cosine with activation-profile cosine, which works for both PCA and NMF.

    Returns (is_different, scores) with keys 'cosine', 'jsd', 'cka',
    'cos_pass', 'jsd_pass', 'cka_pass'.
    """
    vi = np.array(rec_i.vector, dtype=float)
    vj = np.array(rec_j.vector, dtype=float)
    cos = cosine_similarity(vi, vj)
    cos_pass = abs(cos) < thresholds.eps_cos

    pi = np.array(rec_i.activation_histogram[1], dtype=float)
    pj = np.array(rec_j.activation_histogram[1], dtype=float)
    pi = pi / (pi.sum() + 1e-10)
    pj = pj / (pj.sum() + 1e-10)
    jsd = js_divergence(pi, pj)
    jsd_pass = jsd > thresholds.eps_jsd

    Xi = pi.reshape(-1, 1)
    Xj = pj.reshape(-1, 1)
    cka = linear_cka(Xi, Xj)
    cka_pass = cka < thresholds.eps_cka

    is_different = cos_pass and jsd_pass and cka_pass
    scores = {
        "cosine": cos, "jsd": jsd, "cka": cka,
        "cos_pass": float(cos_pass), "jsd_pass": float(jsd_pass), "cka_pass": float(cka_pass),
    }
    return is_different, scores


def differentiation_matrix(
    records: list[FeatureRecord],
    projections: np.ndarray,
    thresholds: DifferentiationThresholds,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise differentiation for F features using full projections.

    Args:
        records: F FeatureRecords (histograms used for JSD).
        projections: R^(N, F) activation matrix - columns correspond to records.
        thresholds:  ε-differentiation thresholds.

    Cosine is computed between activation profiles (projections columns) rather
    than direction vectors. Activation-profile cosine asks "do the same samples
    activate both features?" - meaningful for both PCA and NMF. Direction-vector
    cosine is always ≈0 for PCA (orthogonality by construction) and carries no
    information.

    Composite similarity score = 1 - JSD. JSD is the only metric with reliable
    variation for PCA decompositions; multiplying by activation cosine or CKA
    (which are both ≈0 for PCA) would collapse the matrix to all-zero.
    CKA is still used in the binary ε-different decision.

    Returns:
        score_matrix:  R^(F, F) in [0, 1]. High = similar, low = different.
                       Diagonal = 1.0 by convention.
        binary_matrix: bool^(F, F) - True where features are ε-different.
    """
    F = len(records)
    score_matrix = np.ones((F, F), dtype=np.float32)
    binary_matrix = np.zeros((F, F), dtype=bool)

    for i in range(F):
        for j in range(i + 1, F):
            pi = np.array(records[i].activation_histogram[1], dtype=float)
            pj = np.array(records[j].activation_histogram[1], dtype=float)
            pi /= pi.sum() + 1e-10
            pj /= pj.sum() + 1e-10
            jsd = js_divergence(pi, pj)

            # Activation-profile cosine: correlation between which samples activate each feature.
            ai = projections[:, i].astype(float)
            aj = projections[:, j].astype(float)
            norm_i, norm_j = np.linalg.norm(ai), np.linalg.norm(aj)
            if norm_i > 1e-10 and norm_j > 1e-10:
                cos = float(np.dot(ai, aj) / (norm_i * norm_j))
            else:
                cos = 0.0

            Xi = projections[:, i:i+1]
            Xj = projections[:, j:j+1]
            cka = linear_cka(Xi, Xj)

            # Composite score: 1 - JSD (primary discriminating signal).
            score = float(1.0 - jsd)
            is_diff = (abs(cos) < thresholds.eps_cos) and (jsd > thresholds.eps_jsd) and (cka < thresholds.eps_cka)

            score_matrix[i, j] = score_matrix[j, i] = score
            binary_matrix[i, j] = binary_matrix[j, i] = is_diff

    return score_matrix, binary_matrix


def plot_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    output_path: str,
    title: str = "Feature Differentiation Matrix",
) -> None:
    """
    Save a seaborn annotated heatmap of matrix ∈ R^(F, F).
    Palette: low (different) = blue, high (similar) = red.
    vmin/vmax are inferred from data to use the full colour range.
    """
    vmin = float(np.min(matrix))
    vmax = float(np.max(matrix))
    # Ensure diagonal (always 1.0) doesn't compress the lower range.
    off_diag = matrix[~np.eye(len(labels), dtype=bool)]
    if off_diag.size:
        vmin = float(off_diag.min())
        vmax = float(off_diag.max())

    fig, ax = plt.subplots(figsize=(max(6, len(labels) // 2), max(5, len(labels) // 2)))
    sns.heatmap(
        matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        annot=len(labels) <= 20,
        fmt=".2f",
        linewidths=0.3,
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def summarise_cross_corpus(
    dict_a: FeatureDictionary,
    dict_b: FeatureDictionary,
    proj_a: np.ndarray,
    proj_b: np.ndarray,
    thresholds: DifferentiationThresholds,
    layer: int,
) -> dict[str, float]:
    """
    Compute mean cosine, JSD, CKA and pct-ε-different between features in two
    FeatureDictionaries at a given layer.

    Within-corpus call (dict_a is dict_b): uses activation-profile cosine
    (same N samples available) and skips the diagonal.

    Cross-corpus call (dict_a is not dict_b): uses direction-vector cosine
    (the only cosine computable without matching samples).  CKA is computed
    after subsampling both projection columns to the minimum corpus size.
    """
    recs_a = dict_a.records_for_layer(layer)
    recs_b = dict_b.records_for_layer(layer)
    same_dict = dict_a is dict_b

    cosines, jsds, ckas, diffs = [], [], [], []
    for i, ra in enumerate(recs_a):
        for j, rb in enumerate(recs_b):
            if same_dict and i >= j:
                continue
            fa = int(ra.feature_id)
            fb = int(rb.feature_id)

            pi = np.array(ra.activation_histogram[1], dtype=float)
            pj = np.array(rb.activation_histogram[1], dtype=float)
            pi /= pi.sum() + 1e-10
            pj /= pj.sum() + 1e-10
            jsd = js_divergence(pi, pj)

            if same_dict:
                # Activation-profile cosine: meaningful when same sample set.
                ai = proj_a[:, fa].astype(float)
                aj = proj_b[:, fb].astype(float)
                ni, nj = np.linalg.norm(ai), np.linalg.norm(aj)
                cos = float(np.dot(ai, aj) / (ni * nj)) if ni > 1e-10 and nj > 1e-10 else 0.0
            else:
                # Direction-vector cosine: only option across different sample sets.
                vi = np.array(ra.vector, dtype=float)
                vj = np.array(rb.vector, dtype=float)
                cos = cosine_similarity(vi, vj)

            Xi = proj_a[:, fa:fa+1] if proj_a.shape[1] > fa else np.zeros((proj_a.shape[0], 1))
            Xj = proj_b[:, fb:fb+1] if proj_b.shape[1] > fb else np.zeros((proj_b.shape[0], 1))
            n = min(Xi.shape[0], Xj.shape[0])
            cka = linear_cka(Xi[:n], Xj[:n])

            is_diff = (abs(cos) < thresholds.eps_cos) and (jsd > thresholds.eps_jsd) and (cka < thresholds.eps_cka)
            cosines.append(abs(cos))
            jsds.append(jsd)
            ckas.append(cka)
            diffs.append(float(is_diff))

    if not cosines:
        return {"mean_cos": 0.0, "mean_jsd": 0.0, "mean_cka": 0.0, "pct_different": 0.0}
    return {
        "mean_cos": float(np.mean(cosines)),
        "mean_jsd": float(np.mean(jsds)),
        "mean_cka": float(np.mean(ckas)),
        "pct_different": float(np.mean(diffs)) * 100.0,
    }


def cross_projection_jsd(
    activations_b: np.ndarray,
    dict_a: FeatureDictionary,
    layer: int,
    n_hist_bins: int = 50,
) -> dict[int, float]:
    """
    Project corpus B activations onto corpus A's feature directions and measure
    how differently each feature activates on corpus B vs corpus A.

    For each feature i with direction v_i (from dict_a):
        proj_b_i = activations_b @ v_i          (B-samples projected onto A's direction)
        JSD(hist(proj_a_i) || hist(proj_b_i))   (distribution shift)

    High JSD → feature activates very differently on corpus B → corpus-specific to A.
    Low JSD  → feature generalises to corpus B → corpus-agnostic.

    This is a more principled cross-corpus test than comparing independently-learned
    feature sets: it holds the feature direction constant and measures transferability.

    Args:
        activations_b: R^(N_b, d_model) - raw last-token hidden states from corpus B.
        dict_a:        FeatureDictionary built from corpus A (provides vectors + histograms).
        layer:         Which layer's features to evaluate.
        n_hist_bins:   Histogram resolution for corpus B projections.

    Returns:
        {feature_id: jsd_score} for every feature in dict_a at the given layer.
    """
    records = dict_a.records_for_layer(layer)
    result: dict[int, float] = {}
    for rec in records:
        v = np.array(rec.vector, dtype=float)
        proj_b = activations_b @ v                               # [N_b]
        counts_b, _ = np.histogram(proj_b, bins=n_hist_bins)
        p_b = counts_b.astype(float) / (counts_b.sum() + 1e-10)

        p_a = np.array(rec.activation_histogram[1], dtype=float)
        p_a = p_a / (p_a.sum() + 1e-10)

        # Align histogram lengths (dict_a may have different n_hist_bins).
        if len(p_b) != len(p_a):
            p_b_interp = np.interp(
                np.linspace(0, 1, len(p_a)),
                np.linspace(0, 1, len(p_b)),
                p_b,
            )
            p_b_interp /= p_b_interp.sum() + 1e-10
            p_b = p_b_interp

        result[rec.feature_id] = js_divergence(p_a, p_b)
    return result


def rank_corpus_specific_features(
    cross_jsd_results: dict[str, dict[int, float]],
) -> list[dict]:
    """
    Rank features by corpus-specificity score = mean JSD across all other corpora.

    Args:
        cross_jsd_results: {corpus_name: {feature_id: jsd}} from cross_projection_jsd calls.

    Returns:
        List of dicts sorted descending by mean_jsd, each containing:
            feature_id, mean_jsd, per-corpus JSD values, specificity label.

    Specificity labels:
        "high"   mean_jsd > 0.5  - activates almost exclusively in the reference corpus
        "medium" mean_jsd > 0.25 - moderately corpus-specific
        "low"    otherwise       - corpus-agnostic / universal feature
    """
    if not cross_jsd_results:
        return []
    all_feature_ids = sorted({fid for jsds in cross_jsd_results.values() for fid in jsds})
    corpus_names = list(cross_jsd_results.keys())

    rows = []
    for fid in all_feature_ids:
        per_corpus = {name: cross_jsd_results[name].get(fid, float("nan")) for name in corpus_names}
        valid = [v for v in per_corpus.values() if not np.isnan(v)]
        mean_jsd = float(np.mean(valid)) if valid else float("nan")
        if mean_jsd > 0.5:
            label = "high"
        elif mean_jsd > 0.25:
            label = "medium"
        else:
            label = "low"
        rows.append({"feature_id": fid, "mean_jsd": mean_jsd, "specificity": label, **per_corpus})

    rows.sort(key=lambda r: r["mean_jsd"], reverse=True)
    return rows


def polysemanticity_score(
    projections: np.ndarray,
    feature_col: int,
    top_k: int = 10,
) -> float:
    """
    Estimate polysemanticity of feature `feature_col` from the activation context of its top-k examples.

    Strategy: take the top-k samples by activation, extract their full activation context
    (the whole activation vector row), compute all pairwise cosine similarities, and return
    1 - mean(off-diagonal similarities).  A monosemantic feature activates on contextually
    similar samples (mean cos ≈ 1 → score ≈ 0).  A polysemantic feature activates on diverse
    contexts (mean cos ≈ 0 → score ≈ 1).

    Returns float in [0, 1].  -1 signals too few valid samples.
    """
    if projections.shape[0] < 2 or projections.shape[1] <= feature_col:
        return -1.0

    col = projections[:, feature_col].astype(float)
    k = min(top_k, len(col))
    top_idx = np.argsort(col)[-k:]
    contexts = projections[top_idx].astype(float)

    norms = np.linalg.norm(contexts, axis=1, keepdims=True)
    valid = (norms.ravel() > 1e-10)
    if valid.sum() < 2:
        return -1.0
    contexts = contexts[valid]
    norms = norms[valid]
    unit = contexts / norms

    sim_matrix = unit @ unit.T
    n = sim_matrix.shape[0]
    off_diag = sim_matrix[~np.eye(n, dtype=bool)]
    mean_sim = float(np.mean(off_diag)) if off_diag.size else 0.0
    return float(1.0 - max(0.0, mean_sim))


def show_feature_pair(
    rec_i: FeatureRecord,
    rec_j: FeatureRecord,
    thresholds: DifferentiationThresholds,
    proj_i: np.ndarray | None = None,
    proj_j: np.ndarray | None = None,
) -> str:
    """
    Return a formatted side-by-side comparison string for a pair of features.

    Shows differentiation scores, pass/fail for each criterion, semantic labels,
    polysemanticity scores, and parallel top-k example columns.
    """
    is_diff, scores = epsilon_different(rec_i, rec_j, thresholds)
    status = "DIFFERENT" if is_diff else "similar"

    lines: list[str] = [
        f"Feature pair  f{rec_i.feature_id} (layer {rec_i.layer})  vs  f{rec_j.feature_id} (layer {rec_j.layer})",
        f"Status        {status}",
        f"  cosine      {scores['cosine']:+.4f}  {'PASS' if scores['cos_pass'] else 'fail'}  (eps={thresholds.eps_cos})",
        f"  JSD         {scores['jsd']:.4f}   {'PASS' if scores['jsd_pass'] else 'fail'}  (eps={thresholds.eps_jsd})",
        f"  CKA         {scores['cka']:.4f}   {'PASS' if scores['cka_pass'] else 'fail'}  (eps={thresholds.eps_cka})",
    ]

    label_i = rec_i.semantic_label or "unlabeled"
    label_j = rec_j.semantic_label or "unlabeled"
    poly_i = f"{rec_i.polysemanticity:.3f}" if rec_i.polysemanticity >= 0 else "n/a"
    poly_j = f"{rec_j.polysemanticity:.3f}" if rec_j.polysemanticity >= 0 else "n/a"
    lines += [
        f"  label       {label_i:<25}  {label_j}",
        f"  polysem.    {poly_i:<25}  {poly_j}",
        "",
        f"{'Top examples (f' + str(rec_i.feature_id) + ')':<42}  {'Top examples (f' + str(rec_j.feature_id) + ')'}",
        "-" * 90,
    ]

    examples_i = rec_i.top_k_examples
    examples_j = rec_j.top_k_examples
    n_rows = max(len(examples_i), len(examples_j))
    for row in range(n_rows):
        left = examples_i[row][:38] if row < len(examples_i) else ""
        right = examples_j[row][:44] if row < len(examples_j) else ""
        lines.append(f"  {left:<40}  {right}")

    return "\n".join(lines)


def find_most_different_pairs(
    records: list[FeatureRecord],
    projections: np.ndarray,
    thresholds: DifferentiationThresholds,
    n: int = 5,
) -> list[tuple[FeatureRecord, FeatureRecord, dict]]:
    """
    Return the top-n ε-different feature pairs, sorted by JSD descending.

    Uses activation-profile cosine (projections columns) for geometric criterion,
    consistent with differentiation_matrix.

    Returns list of (rec_i, rec_j, scores_dict) for the n most-different pairs.
    """
    F = len(records)
    candidates: list[tuple[float, FeatureRecord, FeatureRecord, dict]] = []

    for i in range(F):
        for j in range(i + 1, F):
            pi = np.array(records[i].activation_histogram[1], dtype=float)
            pj = np.array(records[j].activation_histogram[1], dtype=float)
            pi /= pi.sum() + 1e-10
            pj /= pj.sum() + 1e-10
            jsd = js_divergence(pi, pj)

            ai = projections[:, i].astype(float)
            aj = projections[:, j].astype(float)
            ni, nj = np.linalg.norm(ai), np.linalg.norm(aj)
            cos = float(np.dot(ai, aj) / (ni * nj)) if ni > 1e-10 and nj > 1e-10 else 0.0

            Xi = projections[:, i:i+1]
            Xj = projections[:, j:j+1]
            cka = linear_cka(Xi, Xj)

            is_diff = (abs(cos) < thresholds.eps_cos) and (jsd > thresholds.eps_jsd) and (cka < thresholds.eps_cka)
            if is_diff:
                scores = {
                    "cosine": cos, "jsd": jsd, "cka": cka,
                    "cos_pass": True, "jsd_pass": True, "cka_pass": True,
                }
                candidates.append((jsd, records[i], records[j], scores))

    candidates.sort(key=lambda t: t[0], reverse=True)
    return [(ri, rj, sc) for _, ri, rj, sc in candidates[:n]]


def plot_feature_histograms(
    records: list[FeatureRecord],
    output_path: str,
    max_features: int = 16,
    title: str = "Feature Activation Distributions",
) -> None:
    """Plot activation histogram for each feature as a grid of bar charts."""
    recs = records[:max_features]
    if not recs:
        return

    n_cols = 4
    n_rows = (len(recs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes_flat = np.array(axes).ravel()

    for i, rec in enumerate(recs):
        ax = axes_flat[i]
        edges = np.array(rec.activation_histogram[0])
        counts = np.array(rec.activation_histogram[1])
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge")
        label = rec.semantic_label or f"f{rec.feature_id}"
        subtitle = label + (f"  p={rec.polysemanticity:.2f}" if rec.polysemanticity >= 0 else "")
        ax.set_title(subtitle, fontsize=8)
        ax.set_xlabel("activation", fontsize=7)
        ax.set_ylabel("density", fontsize=7)
        ax.tick_params(labelsize=6)

    for j in range(len(recs), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title)
    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_token_feature_heatmap(
    token_activations: np.ndarray,
    components: np.ndarray,
    tokens: list[str],
    feature_labels: list[str],
    output_path: str,
    top_k_features: int = 16,
) -> None:
    """
    Project per-token activations onto feature directions and plot as heatmap.

    token_activations: (T, d_model); components: (K, d_model).
    len(tokens) must equal T.
    """
    k = min(top_k_features, components.shape[0])
    scores = (token_activations @ components[:k].T).astype(np.float32)
    labels = (feature_labels + [f"f{i}" for i in range(len(feature_labels), k)])[:k]

    fig, ax = plt.subplots(figsize=(max(6, k // 2), max(4, len(tokens) // 3 + 1)))
    sns.heatmap(scores, xticklabels=labels, yticklabels=tokens,
                cmap="RdBu_r", center=0.0, ax=ax)
    ax.set_xlabel("feature")
    ax.set_ylabel("token")
    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
