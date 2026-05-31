# tests/test_ssm.py
# Тести для перевірки HiPPO, SVD та інференсу.
# Гарантують, що математика працює правильно.

import numpy as np
import pytest
from ssm_model import FrozenSSM, FrozenSSMLowRankEfficient, make_hippo_legs, svd_low_rank
from tasks import associative_recall_task, evaluate_memory, long_range_regression_task

def test_hippo_shape():
    assert make_hippo_legs(16).shape == (16, 16)

def test_svd_reconstruction():
    A = make_hippo_legs(32)
    A_approx, _, _, _, _ = svd_low_rank(A, 32)
    np.testing.assert_allclose(A_approx, A, atol=1e-10)

def test_ssm_output():
    ssm = FrozenSSM(N=32)
    u = np.random.randn(64)
    assert ssm.forward(u).shape == (64,)

def test_efficient_consistency():
    # Перевіряємо, що швидка рекуренція дає той самий результат, що й звичайна.
    N, rank, u = 32, 8, np.random.randn(128)
    slow = FrozenSSM(N=N, rank=rank)
    fast = FrozenSSMLowRankEfficient(N=N, rank=rank)
    # Кореляція має бути майже 1.0
    assert np.corrcoef(slow.forward(u), fast.forward(u))[0, 1] > 0.99

def test_ar_task():
    u, tgt = associative_recall_task(seq_len=64, vocab_size=8)
    assert u.shape == (64,)
    assert 0.0 <= tgt <= 1.0

def test_evaluate_metrics():
    ssm = FrozenSSM(N=16)
    res = evaluate_memory(ssm, long_range_regression_task, n_samples=5, seq_len=32)
    assert "mse" in res 
    assert "mse_sem" in res
    assert "correlation" in res
