import torch
import torch.nn as nn


class LinearTransformer(nn.Module):
    # Same as TheoreticalMamba but G ≡ 1 (no gating); tolerates outlier fraction < 0.5
    def __init__(self, input_dim: int):
        super().__init__()
        d = input_dim + 1
        self.d = d
        self.W_B = nn.Parameter(torch.zeros(d, d))
        self.W_C = nn.Parameter(torch.zeros(d, d))
        self._init_weights()

    def _init_weights(self, delta: float = 0.1):
        nn.init.zeros_(self.W_B)
        nn.init.zeros_(self.W_C)
        with torch.no_grad():
            for k in range(self.d - 1):
                self.W_B[k, k] = delta
                self.W_C[k, k] = delta

    def forward(self, P: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        p_query = P[:, :, -1]
        lin_q = (self.W_B.T @ self.W_C @ p_query.unsqueeze(-1)).squeeze(-1)
        scores = torch.einsum("bdk,bd->bk", P, lin_q)
        return (labels * scores).sum(dim=-1)  # no gate: uniform weight

    def predict(self, P: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.forward(P, labels).sign()
