import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

LOCAL_DEPS = Path(__file__).resolve().parent / "pydeps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))


TOKEN_RE = re.compile(r"[a-zA-Z]+(?:'[a-z]+)?|[0-9]+|[^\s]")


SYNONYMS = {
    "good": ["great", "nice", "fine"],
    "great": ["excellent", "good", "strong"],
    "excellent": ["great", "superb"],
    "bad": ["poor", "awful", "weak"],
    "awful": ["bad", "terrible"],
    "terrible": ["awful", "horrible"],
    "funny": ["amusing", "comic"],
    "boring": ["dull", "tedious"],
    "dull": ["boring", "flat"],
    "beautiful": ["lovely", "gorgeous"],
    "ugly": ["unattractive", "hideous"],
    "smart": ["clever", "intelligent"],
    "stupid": ["dumb", "foolish"],
    "moving": ["touching", "affecting"],
    "charming": ["delightful", "appealing"],
    "strong": ["powerful", "solid"],
    "weak": ["poor", "thin"],
    "love": ["like", "admire"],
    "loves": ["likes", "admires"],
    "like": ["enjoy", "appreciate"],
    "hated": ["disliked", "loathed"],
    "hate": ["dislike", "loathe"],
    "best": ["finest", "greatest"],
    "worst": ["poorest", "weakest"],
    "interesting": ["engaging", "intriguing"],
    "uninteresting": ["boring", "dull"],
    "enjoyable": ["pleasant", "fun"],
    "sad": ["unhappy", "bleak"],
    "bleak": ["grim", "dark"],
    "happy": ["glad", "cheerful"],
    "slow": ["sluggish", "leisurely"],
    "fast": ["quick", "rapid"],
    "predictable": ["obvious", "routine"],
    "original": ["fresh", "novel"],
    "fresh": ["new", "original"],
    "mess": ["disaster", "jumble"],
    "masterpiece": ["triumph", "classic"],
    "performances": ["acting"],
    "performance": ["acting"],
    "film": ["movie"],
    "movie": ["film"],
}


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def load_tsv(path, has_label=True):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = row["sentence"].strip()
            label = int(row["label"]) if has_label and row.get("label", "") != "" else None
            rows.append((text, label))
    return rows


def build_vocab(rows, max_vocab):
    counts = Counter()
    for text, _ in rows:
        counts.update(tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _ in counts.most_common(max_vocab - len(vocab)):
        vocab[token] = len(vocab)
    return vocab


def encode(text, vocab, max_len):
    ids = [vocab.get(tok, 1) for tok in tokenize(text)[:max_len]]
    if not ids:
        ids = [1]
    length = len(ids)
    ids = ids + [0] * (max_len - length)
    return np.array(ids, dtype=np.int64), length


def make_arrays(rows, vocab, max_len):
    x = np.zeros((len(rows), max_len), dtype=np.int64)
    lengths = np.zeros(len(rows), dtype=np.int64)
    y = np.zeros(len(rows), dtype=np.float32)
    texts = []
    for i, (text, label) in enumerate(rows):
        x[i], lengths[i] = encode(text, vocab, max_len)
        y[i] = label
        texts.append(text)
    return x, lengths, y, texts


def init_model(vocab_size, emb_dim, hidden_dim, rng):
    scale = 0.08
    return {
        "E": rng.normal(0, scale, size=(vocab_size, emb_dim)).astype(np.float32),
        "Wx": rng.normal(0, scale, size=(hidden_dim, emb_dim)).astype(np.float32),
        "Wh": rng.normal(0, scale, size=(hidden_dim, hidden_dim)).astype(np.float32),
        "b": np.zeros(hidden_dim, dtype=np.float32),
        "Wo": rng.normal(0, scale, size=(hidden_dim,)).astype(np.float32),
        "bo": np.array(0.0, dtype=np.float32),
    }


def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def forward(model, x, lengths, embedding_noise_std=0.0, rng=None, embedding_delta=None):
    batch, max_len = x.shape
    hidden_dim = model["Wh"].shape[0]
    h = np.zeros((batch, hidden_dim), dtype=np.float32)
    h_sum = np.zeros((batch, hidden_dim), dtype=np.float32)
    h_prev = []
    h_new = []
    masks = []
    embeds = []
    for t in range(max_len):
        mask = (t < lengths).astype(np.float32)[:, None]
        emb = model["E"][x[:, t]]
        if embedding_noise_std > 0.0 and rng is not None:
            emb = emb + rng.normal(0, embedding_noise_std, size=emb.shape).astype(np.float32) * mask
        if embedding_delta is not None:
            emb = emb + embedding_delta[:, t, :] * mask
        cand = np.tanh(emb @ model["Wx"].T + h @ model["Wh"].T + model["b"])
        h_prev.append(h)
        h_new.append(cand)
        masks.append(mask)
        embeds.append(emb)
        h_sum += mask * cand
        h = mask * cand + (1.0 - mask) * h
    h_readout = h_sum / lengths[:, None].astype(np.float32)
    logits = h_readout @ model["Wo"] + model["bo"]
    return logits, {
        "h_final": h_readout,
        "h_prev": h_prev,
        "h_new": h_new,
        "masks": masks,
        "embeds": embeds,
        "lengths": lengths.astype(np.float32),
    }


def batch_loss_grads(model, xb, lb, yb, embedding_noise_std=0.0, rng=None, embedding_delta=None):
    logits, cache = forward(
        model,
        xb,
        lb,
        embedding_noise_std=embedding_noise_std,
        rng=rng,
        embedding_delta=embedding_delta,
    )
    probs = sigmoid(logits)
    eps = 1e-7
    loss = -np.mean(yb * np.log(probs + eps) + (1 - yb) * np.log(1 - probs + eps))
    correct = int(np.sum((probs >= 0.5) == yb))

    dlogits = (probs - yb).astype(np.float32) / len(xb)
    grads = {k: np.zeros_like(v) for k, v in model.items()}
    pos_demb = np.zeros((xb.shape[0], xb.shape[1], model["E"].shape[1]), dtype=np.float32)
    grads["Wo"] += cache["h_final"].T @ dlogits
    grads["bo"] += np.sum(dlogits)
    dreadout = dlogits[:, None] * model["Wo"][None, :]
    dh_future = np.zeros_like(dreadout)

    for t in reversed(range(xb.shape[1])):
        mask = cache["masks"][t]
        cand = cache["h_new"][t]
        prev = cache["h_prev"][t]
        emb = cache["embeds"][t]
        dh_readout = dreadout / cache["lengths"][:, None]
        dh_cand = (dh_future + dh_readout) * mask
        da = dh_cand * (1.0 - cand * cand)

        grads["Wx"] += da.T @ emb
        grads["Wh"] += da.T @ prev
        grads["b"] += np.sum(da, axis=0)
        demb = da @ model["Wx"]
        pos_demb[:, t, :] = demb * mask
        np.add.at(grads["E"], xb[:, t], demb)
        dh_future = da @ model["Wh"]

    return float(loss), correct, grads, pos_demb


def train_epoch(
    model,
    x,
    lengths,
    y,
    lr,
    batch_size,
    rng,
    opt_state,
    clip=5.0,
    recurrent_norm_cap=None,
    embedding_noise_std=0.0,
    weight_decay=0.0,
    adversarial_embedding_eps=0.0,
):
    indices = np.arange(len(x))
    rng.shuffle(indices)
    total_loss = 0.0
    correct = 0
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        xb, lb, yb = x[idx], lengths[idx], y[idx]
        loss, batch_correct, grads, pos_demb = batch_loss_grads(
            model,
            xb,
            lb,
            yb,
            embedding_noise_std=embedding_noise_std,
            rng=rng,
        )
        if adversarial_embedding_eps > 0.0:
            denom = np.linalg.norm(pos_demb, axis=2, keepdims=True) + 1e-8
            embedding_delta = adversarial_embedding_eps * pos_demb / denom
            loss, batch_correct, grads, _ = batch_loss_grads(
                model,
                xb,
                lb,
                yb,
                embedding_delta=embedding_delta,
            )
        total_loss += float(loss) * len(idx)
        correct += batch_correct

        global_norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        if global_norm > clip:
            factor = clip / (global_norm + 1e-8)
            for g in grads.values():
                g *= factor
        if weight_decay > 0.0:
            for key in ("E", "Wx", "Wh", "Wo"):
                grads[key] += weight_decay * model[key]
        opt_state["step"] += 1
        beta1, beta2 = 0.9, 0.999
        for key in model:
            opt_state["m"][key] = beta1 * opt_state["m"][key] + (1 - beta1) * grads[key]
            opt_state["v"][key] = beta2 * opt_state["v"][key] + (1 - beta2) * (grads[key] * grads[key])
            m_hat = opt_state["m"][key] / (1 - beta1 ** opt_state["step"])
            v_hat = opt_state["v"][key] / (1 - beta2 ** opt_state["step"])
            model[key] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        if recurrent_norm_cap is not None:
            wh_norm = spectral_norm(model["Wh"], steps=10)
            if wh_norm > recurrent_norm_cap:
                model["Wh"] *= recurrent_norm_cap / (wh_norm + 1e-8)
        model["E"][0] = 0.0
    return total_loss / len(x), correct / len(x)


def evaluate(model, x, lengths, y):
    logits, _ = forward(model, x, lengths)
    probs = sigmoid(logits)
    preds = (probs >= 0.5).astype(np.float32)
    return {
        "accuracy": float(np.mean(preds == y)),
        "loss": float(-np.mean(y * np.log(probs + 1e-7) + (1 - y) * np.log(1 - probs + 1e-7))),
        "logits": logits,
        "probs": probs,
        "preds": preds,
    }


def typo_token(token):
    if len(token) <= 3 or not token.isalpha():
        return token
    chars = list(token)
    chars[1], chars[2] = chars[2], chars[1]
    return "".join(chars)


@lru_cache(maxsize=20000)
def raw_wordnet_synonyms(token):
    try:
        import nltk
        nltk.data.path.append(str(Path(__file__).resolve().parent / "nltk_data"))
        from nltk.corpus import wordnet as wn
    except Exception:
        return []

    candidates = []
    for synset in wn.synsets(token):
        for lemma in synset.lemmas():
            name = lemma.name().lower().replace("_", " ")
            if " " in name or name == token or not name.isalpha():
                continue
            if name not in candidates:
                candidates.append(name)
    return tuple(candidates)


def wordnet_synonyms(token, vocab=None, max_candidates=8):
    candidates = []
    for name in raw_wordnet_synonyms(token):
        if vocab is not None and name not in vocab:
            continue
        candidates.append(name)
        if len(candidates) >= max_candidates:
            break
    return candidates


def attack_text(text, mode="mixed", attack_source="manual", vocab=None):
    tokens = tokenize(text)
    changed = False
    out = []
    for tok in tokens:
        replacement = tok
        if mode in {"mixed", "synonym"} and not changed:
            synonyms = []
            if attack_source in {"manual", "combined"}:
                synonyms.extend(SYNONYMS.get(tok, []))
            if attack_source in {"wordnet", "combined"}:
                synonyms.extend(wordnet_synonyms(tok, vocab=vocab))
            synonyms = [s for s in dict.fromkeys(synonyms) if s != tok and (vocab is None or s in vocab)]
            if synonyms:
                replacement = synonyms[0]
                changed = replacement != tok
        elif mode in {"mixed", "typo"} and tok.isalpha() and len(tok) > 5 and not changed:
            replacement = typo_token(tok)
            changed = replacement != tok
        out.append(replacement)
    if not changed:
        for i, tok in enumerate(out):
            if tok.isalpha() and len(tok) > 3:
                out[i] = typo_token(tok)
                changed = out[i] != tok
                break
    return " ".join(out), changed


def attack_dataset(texts, y, vocab, max_len, attack_source="manual"):
    rows = []
    changed_flags = []
    for text, label in zip(texts, y):
        attacked, changed = attack_text(text, attack_source=attack_source, vocab=vocab)
        rows.append((attacked, int(label)))
        changed_flags.append(changed)
    x, lengths, labels, attacked_texts = make_arrays(rows, vocab, max_len)
    return x, lengths, labels, attacked_texts, np.array(changed_flags, dtype=bool)


def spectral_norm(matrix, steps=30):
    rng = np.random.default_rng(123)
    v = rng.normal(size=(matrix.shape[1],)).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(steps):
        u = matrix @ v
        u /= np.linalg.norm(u) + 1e-12
        v = matrix.T @ u
        v /= np.linalg.norm(v) + 1e-12
    return float(np.linalg.norm(matrix @ v))


def embedding_epsilons(text, attacked_text, model, vocab, max_len, fallback_eps):
    ids, length = encode(text, vocab, max_len)
    attack_ids, _ = encode(attacked_text, vocab, max_len)
    eps = np.zeros(max_len, dtype=np.float32)
    for i in range(length):
        delta = model["E"][attack_ids[i]] - model["E"][ids[i]]
        eps[i] = np.linalg.norm(delta)
    nonzero = eps[eps > 0]
    if len(nonzero) == 0:
        eps[:length] = fallback_eps
    return eps, length


def certify_examples(model, texts, attacked_texts, vocab, max_len, local_certificate=False):
    wx = spectral_norm(model["Wx"])
    wh = spectral_norm(model["Wh"])
    wo = float(np.linalg.norm(model["Wo"]))
    fallback_eps = float(np.median(np.linalg.norm(model["E"], axis=1))) * 0.1
    certs = []
    radii = []
    margins = []
    for text, attacked in zip(texts, attacked_texts):
        x, length = encode(text, vocab, max_len)
        logits, cache = forward(model, x[None, :], np.array([length]))
        margin = abs(float(logits[0]))
        eps, length = embedding_epsilons(text, attacked, model, vocab, max_len, fallback_eps)
        if local_certificate:
            readout_bound = 0.0
            prev_bound = 0.0
            for t in range(length):
                tanh_slope = float(np.max(1.0 - cache["h_new"][t][0] ** 2))
                hidden_bound_t = tanh_slope * (wx * float(eps[t]) + wh * prev_bound)
                readout_bound += hidden_bound_t / length
                prev_bound = hidden_bound_t
        else:
            readout_bound = 0.0
            for t in range(length):
                hidden_bound_t = 0.0
                for i in range(t + 1):
                    hidden_bound_t += wx * (wh ** (t - i)) * float(eps[i])
                readout_bound += hidden_bound_t / length
        logit_bound = wo * readout_bound
        certs.append(margin > logit_bound)
        radii.append(logit_bound)
        margins.append(margin)
    return np.array(certs, dtype=bool), np.array(margins), np.array(radii), {
        "wx_norm": wx,
        "wh_norm": wh,
        "wo_norm": wo,
        "local_certificate": local_certificate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="SST-2")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-vocab", type=int, default=8000)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--emb-dim", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--train-limit", type=int, default=20000)
    parser.add_argument("--recurrent-norm-cap", type=float, default=None)
    parser.add_argument("--embedding-noise-std", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adversarial-embedding-eps", type=float, default=0.0)
    parser.add_argument("--augment-attacks", action="store_true")
    parser.add_argument("--attack-source", choices=["manual", "wordnet", "combined"], default="manual")
    parser.add_argument("--local-certificate", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    train_rows = load_tsv(data_dir / "train.tsv")
    dev_rows = load_tsv(data_dir / "dev.tsv")
    rng.shuffle(train_rows)
    if args.train_limit:
        train_rows = train_rows[:args.train_limit]
    if args.augment_attacks:
        augmented = []
        for text, label in train_rows:
            attacked, changed = attack_text(text, attack_source=args.attack_source)
            if changed:
                augmented.append((attacked, label))
        train_rows = train_rows + augmented
        rng.shuffle(train_rows)

    vocab = build_vocab(train_rows, args.max_vocab)
    x_train, len_train, y_train, _ = make_arrays(train_rows, vocab, args.max_len)
    x_dev, len_dev, y_dev, dev_texts = make_arrays(dev_rows, vocab, args.max_len)

    model = init_model(len(vocab), args.emb_dim, args.hidden_dim, rng)
    opt_state = {
        "step": 0,
        "m": {k: np.zeros_like(v) for k, v in model.items()},
        "v": {k: np.zeros_like(v) for k, v in model.items()},
    }
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model,
            x_train,
            len_train,
            y_train,
            args.lr,
            args.batch_size,
            rng,
            opt_state,
            recurrent_norm_cap=args.recurrent_norm_cap,
            embedding_noise_std=args.embedding_noise_std,
            weight_decay=args.weight_decay,
            adversarial_embedding_eps=args.adversarial_embedding_eps,
        )
        dev_eval = evaluate(model, x_dev, len_dev, y_dev)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "dev_loss": dev_eval["loss"],
            "dev_accuracy": dev_eval["accuracy"],
        }
        history.append(row)
        print(json.dumps(row))

    clean = evaluate(model, x_dev, len_dev, y_dev)
    x_att, len_att, y_att, attacked_texts, changed = attack_dataset(
        dev_texts,
        y_dev,
        vocab,
        args.max_len,
        attack_source=args.attack_source,
    )
    attacked = evaluate(model, x_att, len_att, y_att)
    certs, margins, bounds, norm_info = certify_examples(
        model,
        dev_texts,
        attacked_texts,
        vocab,
        args.max_len,
        local_certificate=args.local_certificate,
    )

    clean_correct = clean["preds"] == y_dev
    attacked_correct = attacked["preds"] == y_dev
    attack_success = clean_correct & changed & (~attacked_correct)
    certified_correct = clean_correct & certs

    examples = []
    for i in np.where(changed)[0][:200]:
        if len(examples) >= 12:
            break
        examples.append({
            "text": dev_texts[int(i)],
            "attacked_text": attacked_texts[int(i)],
            "label": int(y_dev[i]),
            "clean_prob_positive": float(clean["probs"][i]),
            "attacked_prob_positive": float(attacked["probs"][i]),
            "clean_correct": bool(clean_correct[i]),
            "attacked_correct": bool(attacked_correct[i]),
            "certified": bool(certs[i]),
            "margin": float(margins[i]),
            "logit_perturbation_bound": float(bounds[i]),
        })

    metrics = {
        "config": vars(args),
        "vocab_size": len(vocab),
        "train_size": len(train_rows),
        "dev_size": len(dev_rows),
        "clean_accuracy": clean["accuracy"],
        "attacked_accuracy": attacked["accuracy"],
        "attack_coverage": float(np.mean(changed)),
        "attack_success_rate_over_changed": float(np.sum(attack_success) / max(1, np.sum(changed))),
        "certified_fraction_all": float(np.mean(certs)),
        "certified_robust_accuracy": float(np.mean(certified_correct)),
        "mean_clean_margin": float(np.mean(margins)),
        "mean_logit_perturbation_bound": float(np.mean(bounds)),
        "norm_info": norm_info,
        "history": history,
        "examples": examples,
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    np.savez(out_dir / "model.npz", **model)
    print("Saved outputs to", out_dir)


if __name__ == "__main__":
    main()
