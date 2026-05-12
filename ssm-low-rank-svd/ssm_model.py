import math
import time
import numpy as np

# HiPPO-LegS: будуємо матрицю A
def make_hippo_legs(N: int) -> np.ndarray:
    # Legendre projection — дозволяє SSM рівномірно "пам'ятати" довгий контекст.
    # Матриця нижньотрикутна: діагональ -(n+1), під нею -sqrt((2n+1)(2k+1)).
    A = np.zeros((N, N))
    for n in range(N):
        for k in range(n):
            A[n, k] = -math.sqrt((2 * n + 1) * (2 * k + 1))
        A[n, n] = -(n + 1)
    return A


# SVD: A ≈ U_r Σ_r V_r^T
def svd_low_rank(A: np.ndarray, rank: int):
    # Тінімо матрицю до рангу r через SVD.
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    U_r, S_r, Vt_r = U[:, :rank], S[:rank], Vt[:rank, :]
    A_approx = (U_r * S_r) @ Vt_r
    # Відсоток збереженої "енергії" σ²
    explained_variance = float(np.sum(S_r**2) / np.sum(S**2))
    return A_approx, U_r, S_r, Vt_r, explained_variance


# Дискретизація: неперервний час -> дискретний
def discretize_zoh(A: np.ndarray, B: np.ndarray, dt: float):
    # Білінійна (Тастіна) форма: стабільна й стандартна для S4/Mamba.
    # Ā = (I - dt/2·A)⁻¹(I + dt/2·A), B̄ = (I - dt/2·A)⁻¹·dt·B.
    N, I = A.shape[0], np.eye(A.shape[0])
    inv_term = np.linalg.inv(I - (dt / 2) * A)
    A_bar = inv_term @ (I + (dt / 2) * A)
    B_bar = (inv_term * dt) @ B
    return A_bar, B_bar


# Frozen SSM: повноранговий інференс
class FrozenSSM:
    # Заморожені ваги. Якщо є rank — юзаємо SVD-апроксимацію.
    def __init__(self, N: int = 64, dt: float = 0.01, rank: int | None = None):
        self.N, self.dt, self.rank = N, dt, rank
        A = make_hippo_legs(N)
        B, C, D = np.ones((N, 1)), np.ones((1, N)), np.zeros((1,))

        if rank and rank < N:
            A_used, _, _, _, self.explained_var = svd_low_rank(A, rank)
        else:
            A_used, self.explained_var = A, 1.0

        self.A_bar, self.B_bar, self.C, self.D = *discretize_zoh(A_used, B, dt), C, D

    def forward(self, u_seq: np.ndarray) -> np.ndarray:
        # x[t+1] = Āx[t] + B̄u[t]. Складність O(N²) на крок.
        x, y_seq = np.zeros((self.N, 1)), np.empty(len(u_seq))
        for t, u in enumerate(u_seq):
            y_seq[t] = (self.C @ x).item() + self.D[0] * u
            x = self.A_bar @ x + self.B_bar * u
        return y_seq

    def timed_forward(self, u_seq: np.ndarray, n_runs: int = 5):
        self.forward(u_seq)  # warm-up
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.forward(u_seq)
            times.append((time.perf_counter() - t0) * 1e3)
        return None, float(np.mean(times)), float(np.std(times))


# Low-Rank Efficient: O(Nr) замість O(N²)
class FrozenSSMLowRankEfficient:
    # Рекуренція через фактори SVD. Ā = I + UΣVᵀ.
    def __init__(self, N: int = 64, dt: float = 0.01, rank: int = 8):
        self.N, self.dt, self.rank = N, dt, rank
        A_approx, _, _, _, self.explained_var = svd_low_rank(make_hippo_legs(N), rank)
        self.A_bar, self.B_bar = discretize_zoh(A_approx, np.ones((N, 1)), dt)
        self.C, self.D = np.ones((1, N)), np.zeros((1,))

        # Факторизуємо Ā-I для схеми x = x + U(Σ(Vᵀx)).
        U, S, Vt = np.linalg.svd(self.A_bar - np.eye(N), full_matrices=False)
        self.U_d, self.S_d, self.Vt_d = U[:, :rank], S[:rank], Vt[:rank, :]

    def forward(self, u_seq: np.ndarray) -> np.ndarray:
        x, y_seq = np.zeros((self.N, 1)), np.empty(len(u_seq))
        for t, u in enumerate(u_seq):
            y_seq[t] = (self.C @ x).item() + self.D[0] * u
            # Ефективний крок: x + U·(S⊙(Vᵀx))
            x = x + self.U_d @ (self.S_d[:, None] * (self.Vt_d @ x)) + self.B_bar * u
        return y_seq

    def timed_forward(self, u_seq: np.ndarray, n_runs: int = 5):
        self.forward(u_seq)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.forward(u_seq)
            times.append((time.perf_counter() - t0) * 1e3)
        return None, float(np.mean(times)), float(np.std(times))
