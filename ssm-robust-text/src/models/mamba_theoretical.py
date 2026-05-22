import torch
import torch.nn as nn


class TheoreticalMamba(nn.Module):
    # One-layer Mamba: gated linear attention (Li et al. 2025, Eq. 3)
    def __init__(self, input_dim: int):
        super().__init__()
        d = input_dim + 1  # token dim = d+1 (x stacked with y)
        self.d = d
        self.W_B = nn.Parameter(torch.zeros(d, d))
        self.W_C = nn.Parameter(torch.zeros(d, d))
        self.w = nn.Parameter(torch.randn(d) / d**0.5)  # gating vector
        self._init_weights()

    def _init_weights(self, delta: float = 0.1):
        nn.init.zeros_(self.W_B)
        nn.init.zeros_(self.W_C)
        with torch.no_grad():
            for k in range(self.d - 1):
                self.W_B[k, k] = delta
                self.W_C[k, k] = delta
        nn.init.normal_(self.w, std=1.0 / self.d)

    @staticmethod
    def _compute_gates(w: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        # G_i = sigmoid(w·p_i) * prod_{j>i}(1 - sigmoid(w·p_j))
        logits = torch.einsum("d,bdk->bk", w, P)
        sig = torch.sigmoid(logits)
        incl_suffix = (1.0 - sig).flip(-1).cumprod(-1).flip(-1)
        ones = torch.ones(sig.shape[0], 1, device=sig.device, dtype=sig.dtype)
        excl_suffix = torch.cat([incl_suffix[:, 1:], ones], dim=-1)
        return sig * excl_suffix

    def forward(self, P: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        p_query = P[:, :, -1]
        gates = self._compute_gates(self.w, P)
        # linear attention key: W_B^T W_C p_query
        lin_q = (self.W_B.T @ self.W_C @ p_query.unsqueeze(-1)).squeeze(-1)
        scores = torch.einsum("bdk,bd->bk", P, lin_q)
        # F = sum_i G_i * y_i * score_i
        return (gates * labels * scores).sum(dim=-1)

    def predict(self, P: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.forward(P, labels).sign()
