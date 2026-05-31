# Задачі для перевірки здатності SSM запам'ятовувати інформацію.
# Перевіряємо, чи "виживає" сигнал у прихованому стані на довгих дистанціях.

import numpy as np
from typing import Tuple, Optional

def associative_recall_task(
    seq_len: int = 256,
    vocab_size: int = 8,
    n_pairs: int = 4,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, float]:
    # Пари ключ-значення розкидані по послідовності.
    # В самому кінці — ключ-запит. Треба витягнути відповідне значення.
    if rng is None:
        rng = np.random.default_rng(42)

    keys = rng.integers(0, vocab_size, size=n_pairs)
    values = rng.integers(0, vocab_size, size=n_pairs)

    # Обираємо випадковий ключ для запиту
    query_idx = rng.integers(0, n_pairs)
    target = float(values[query_idx]) / (vocab_size - 1) if vocab_size > 1 else 0.0

    u_seq = np.zeros(seq_len, dtype=np.float32)
    for i, (k, v) in enumerate(zip(keys, values)):
        if 2 * i < seq_len - 1:
            u_seq[2 * i] = (2 * k + 1) / (2 * vocab_size)     # Ключі
        if 2 * i + 1 < seq_len - 1:
            u_seq[2 * i + 1] = (2 * v + 2) / (2 * vocab_size) # Значення
    
    u_seq[-1] = (2 * keys[query_idx] + 1) / (2 * vocab_size)  # Запит

    return u_seq, target

def long_range_regression_task(
    seq_len: int = 256,
    rng: Optional[np.random.Generator] = None,
    signal_window: int = 16
) -> Tuple[np.ndarray, float]:
    # Задача на сумування сигналу з далекого минулого.
    if rng is None:
        rng = np.random.default_rng(42)

    u_seq = rng.standard_normal(seq_len).astype(np.float32)
    # Ціль — сума перших signal_window елементів
    target = float(np.sum(u_seq[:signal_window]))
    return u_seq, target

def evaluate_memory(
    ssm, task_fn, n_samples: int = 200, seq_len: int = 256, **kw
) -> dict:
    # Оцінюємо модель на пакеті прикладів.
    # Рахуємо MSE, SEM та кореляцію.
    rng = np.random.default_rng(0)
    preds, targets = [], []
    
    for _ in range(n_samples):
        u, tgt = task_fn(seq_len=seq_len, rng=rng, **kw)
        y = ssm.forward(u)
        preds.append(y[-1])
        targets.append(tgt)
    
    preds = np.array(preds)
    targets = np.array(targets)
    
    mse_vals = (preds - targets) ** 2
    mse = np.mean(mse_vals)
    mse_sem = np.std(mse_vals) / np.sqrt(n_samples)
    
    if np.std(preds) < 1e-9 or np.std(targets) < 1e-9:
        corr = 0.0
    else:
        corr = np.corrcoef(preds, targets)[0, 1]
    
    return {
        "mse": float(mse),
        "mse_sem": float(mse_sem),
        "correlation": float(corr)
    }
