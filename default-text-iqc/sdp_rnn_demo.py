import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "pydeps"))

import cvxpy as cp
import numpy as np

from robustness_rnn_experiment import (
    attack_text,
    encode,
    forward,
    load_tsv,
    make_arrays,
)


def load_vocab(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def idx_map(seq_len, emb_dim, hidden_dim):
    idx = {}
    names = ["const"]
    cursor = 1
    for t in range(seq_len):
        for j in range(emb_dim):
            idx[("delta", t, j)] = cursor
            names.append(f"delta_{t}_{j}")
            cursor += 1
    for t in range(seq_len):
        for j in range(hidden_dim):
            idx[("pre", t, j)] = cursor
            names.append(f"pre_{t}_{j}")
            cursor += 1
    for t in range(seq_len):
        for j in range(hidden_dim):
            idx[("h", t, j)] = cursor
            names.append(f"h_{t}_{j}")
            cursor += 1
    return idx, names


def moment_expr(X, coeffs):
    expr = 0
    for i, ci in coeffs.items():
        for j, cj in coeffs.items():
            expr += ci * cj * X[i, j]
    return expr


def linear_expr(X, coeffs):
    expr = 0
    for i, ci in coeffs.items():
        expr += ci * X[0, i]
    return expr


def solve_sdp_bound(Wx, Wh, b, Wo, bo, clean_embeds, eps, label, solver):
    seq_len, emb_dim = clean_embeds.shape
    hidden_dim = Wh.shape[0]
    idx, names = idx_map(seq_len, emb_dim, hidden_dim)
    n = len(names)

    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0, X[0, 0] == 1]

    # Input uncertainty: each token embedding perturbation lies in an L2 ball.
    for t in range(seq_len):
        constraints.append(
            sum(X[idx[("delta", t, j)], idx[("delta", t, j)]] for j in range(emb_dim))
            <= float(eps[t] ** 2)
        )

    # Linear recurrent dynamics in the lifted moment matrix:
    # pre_t = Wx (e_t + delta_t) + Wh h_{t-1} + b.
    # For an exact rank-one matrix X = yy^T, an affine equality
    # y_pre = a^T y implies X[k, pre] = sum_j a_j X[k, j] for all k.
    # Enforcing the full lifted equality is much tighter than enforcing
    # only the first moment X[0, pre].
    for t in range(seq_len):
        for i in range(hidden_dim):
            rhs = float(b[i] + Wx[i] @ clean_embeds[t])
            coeffs = {0: rhs}
            for j in range(emb_dim):
                coeffs[idx[("delta", t, j)]] = float(Wx[i, j])
            if t > 0:
                for j in range(hidden_dim):
                    coeffs[idx[("h", t - 1, j)]] = float(Wh[i, j])
            pre_idx = idx[("pre", t, i)]
            for k in range(n):
                constraints.append(X[k, pre_idx] == sum(c * X[k, j] for j, c in coeffs.items()))

    # Tanh bounded-sector constraints. For tanh on [-rho, rho],
    # alpha * pre <= h <= pre in sector form, with
    # alpha = tanh(rho) / rho. In lifted form:
    # (h - alpha pre) * (h - pre) <= 0.
    for t in range(seq_len):
        for i in range(hidden_dim):
            h = idx[("h", t, i)]
            pre = idx[("pre", t, i)]
            pre_center = float(b[i] + Wx[i] @ clean_embeds[t])
            input_radius = float(np.sum(np.abs(Wx[i]) * eps[t]))
            recurrent_radius = float(np.sum(np.abs(Wh[i]))) if t > 0 else 0.0
            rho = abs(pre_center) + input_radius + recurrent_radius + 1e-6
            alpha = float(np.tanh(rho) / rho)
            constraints.append(X[h, h] - (1 + alpha) * X[h, pre] + alpha * X[pre, pre] <= 0)
            constraints.append(X[h, h] <= 1.0)
            constraints.append(X[0, h] <= 1.0)
            constraints.append(X[0, h] >= -1.0)

    # Output logit using mean hidden-state readout.
    logit = float(bo)
    for t in range(seq_len):
        for i in range(hidden_dim):
            logit += float(Wo[i] / seq_len) * X[0, idx[("h", t, i)]]

    maximize_problem = cp.Problem(cp.Maximize(logit), constraints)
    maximize_problem.solve(solver=solver, verbose=False, eps=1e-4, max_iters=20000)
    upper = maximize_problem.value
    upper_status = maximize_problem.status

    minimize_problem = cp.Problem(cp.Minimize(logit), constraints)
    minimize_problem.solve(solver=solver, verbose=False, eps=1e-4, max_iters=20000)
    lower = minimize_problem.value
    lower_status = minimize_problem.status

    if label == 1:
        certified = lower is not None and lower > 0
    else:
        certified = upper is not None and upper < 0

    return {
        "lower_logit_bound": None if lower is None else float(lower),
        "upper_logit_bound": None if upper is None else float(upper),
        "lower_status": lower_status,
        "upper_status": upper_status,
        "certified": bool(certified),
        "sdp_variables": n,
        "moment_matrix_shape": [n, n],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs_norm_cap/model.npz")
    parser.add_argument("--vocab", default="outputs_norm_cap/vocab.json")
    parser.add_argument("--data", default="SST-2/dev.tsv")
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--emb-dim", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=6)
    parser.add_argument("--example-index", type=int, default=None)
    parser.add_argument("--solver", default="SCS")
    parser.add_argument("--out", default="sdp_outputs/sdp_demo_result.json")
    args = parser.parse_args()

    model_full = dict(np.load(args.model))
    vocab = load_vocab(args.vocab)
    rows = load_tsv(Path(args.data))
    x_dev, len_dev, y_dev, texts = make_arrays(rows, vocab, max_len=32)
    eval_full = forward(model_full, x_dev, len_dev)
    logits_full = eval_full[0]
    preds_full = (logits_full >= 0).astype(np.float32)

    if args.example_index is None:
        candidates = [
            i
            for i, text in enumerate(texts)
            if preds_full[i] == y_dev[i] and len(encode(text, vocab, 32)[0][: args.seq_len]) >= args.seq_len
        ]
        example_index = candidates[0]
    else:
        example_index = args.example_index

    text = texts[example_index]
    label = int(y_dev[example_index])
    attacked_text, changed = attack_text(text)
    ids, _ = encode(text, vocab, 32)
    attacked_ids, _ = encode(attacked_text, vocab, 32)
    ids = ids[: args.seq_len]
    attacked_ids = attacked_ids[: args.seq_len]

    E = model_full["E"][:, : args.emb_dim]
    Wx = model_full["Wx"][: args.hidden_dim, : args.emb_dim]
    Wh = model_full["Wh"][: args.hidden_dim, : args.hidden_dim]
    b = model_full["b"][: args.hidden_dim]
    Wo = model_full["Wo"][: args.hidden_dim]
    bo = float(model_full["bo"])
    clean_embeds = E[ids]

    eps = np.linalg.norm(E[attacked_ids] - E[ids], axis=1)
    fallback = float(np.median(np.linalg.norm(E, axis=1)) * 0.1)
    eps = np.where(eps > 0, eps, fallback)

    # Clean logit for the reduced model slice.
    h = np.zeros(args.hidden_dim)
    hidden_states = []
    for t in range(args.seq_len):
        h = np.tanh(Wx @ clean_embeds[t] + Wh @ h + b)
        hidden_states.append(h.copy())
    clean_logit_reduced = float(Wo @ np.mean(hidden_states, axis=0) + bo)

    result = solve_sdp_bound(Wx, Wh, b, Wo, bo, clean_embeds, eps, label, args.solver)
    result.update(
        {
            "example_index": int(example_index),
            "text": text,
            "attacked_text": attacked_text,
            "attack_changed_text": bool(changed),
            "label": label,
            "full_model_clean_logit": float(logits_full[example_index]),
            "full_model_clean_prediction": int(preds_full[example_index]),
            "reduced_model_clean_logit": clean_logit_reduced,
            "seq_len": args.seq_len,
            "emb_dim": args.emb_dim,
            "hidden_dim": args.hidden_dim,
            "eps_per_token": [float(v) for v in eps],
            "solver": args.solver,
            "formulation": "Lifted moment-matrix SDP relaxation with input L2 balls, lifted affine RNN dynamics, hidden-state bounds, and bounded-sector tanh quadratic constraints.",
        }
    )

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
