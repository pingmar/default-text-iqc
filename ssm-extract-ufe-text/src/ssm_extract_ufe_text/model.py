from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import MambaConfig, MambaModel
from tqdm import tqdm

from ssm_extract_ufe_text.config import ExtractionConfig


class MambaProbe:
    """
    MambaModel wrapper that registers output hooks on backbone.layers[i].

    Hook target: each MambaBlock output after mixer + layer-norm (residual
    stream). Shape per forward call: [B, T, d_model]. We pool the last-token
    hidden state (index -1), which is correct given left-padding (final real
    token always sits at position -1 - see corpus.py).

    Usage:
        probe = MambaProbe.from_pretrained(config)
        probe.register_hooks([6, 12])
        activations = probe.collect_batch(batch)   # {6: Tensor[B,d], 12: Tensor[B,d]}
        probe.remove_hooks()
    """

    def __init__(self, model: MambaModel, config: ExtractionConfig) -> None:
        self.model = model
        self.config = config
        self._activations: dict[int, torch.Tensor] = {}
        self._hook_handles: list = []

        device = self._resolve_device(config.device)
        self.model.to(device)
        self.device = device

        layer_indices = config.layers if config.layers is not None else list(
            range(len(model.layers))
        )
        self.register_hooks(layer_indices)

    @classmethod
    def from_pretrained(cls, config: ExtractionConfig) -> "MambaProbe":
        model = MambaModel.from_pretrained(config.model_name)
        return cls(model, config)

    @classmethod
    def from_hf_config(
        cls, hf_config: MambaConfig, config: ExtractionConfig
    ) -> "MambaProbe":
        """Build from a MambaConfig without downloading weights - used in tests."""
        model = MambaModel(hf_config)
        return cls(model, config)

    def register_hooks(self, layer_indices: list[int]) -> None:
        self._layer_indices: list[int] = list(layer_indices)
        self.remove_hooks()
        for idx in layer_indices:
            handle = self.model.layers[idx].register_forward_hook(
                self._make_hook(idx)
            )
            self._hook_handles.append(handle)

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        self._activations.clear()

    def _make_hook(self, idx: int, pool_last: bool = True):
        def fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._activations[idx] = (
                hidden[:, -1, :].detach().cpu() if pool_last
                else hidden.detach().cpu()
            )
        return fn

    @torch.no_grad()
    def collect_batch(self, batch: dict) -> dict[int, torch.Tensor]:
        """
        Run a single batch dict (must contain 'input_ids') through the model.
        Returns {layer_idx: Tensor[B, d_model]} from the registered hooks.
        """
        self._activations.clear()
        input_ids = batch["input_ids"].to(self.device)
        self.model(input_ids=input_ids)
        return dict(self._activations)

    @torch.no_grad()
    def collect(self, loader: DataLoader) -> dict[int, torch.Tensor]:
        """
        Run all batches in loader; concatenate activations across batches.
        Returns {layer_idx: Tensor[N, d_model]}.
        """
        per_layer: dict[int, list[torch.Tensor]] = {}
        for batch in tqdm(loader, desc="collecting", leave=False):
            for layer_idx, act in self.collect_batch(batch).items():
                per_layer.setdefault(layer_idx, []).append(act)
        return {layer: torch.cat(chunks, dim=0) for layer, chunks in per_layer.items()}

    @torch.no_grad()
    def collect_batch_all_tokens(self, batch: dict) -> dict[int, torch.Tensor]:
        """
        Run a single batch; return {layer_idx: Tensor[B, T, d_model]} - all positions.
        Temporarily re-registers hooks without last-token pooling, then restores them.
        """
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        for idx in self._layer_indices:
            self._hook_handles.append(
                self.model.layers[idx].register_forward_hook(
                    self._make_hook(idx, pool_last=False)
                )
            )
        self._activations.clear()
        try:
            self.model(input_ids=batch["input_ids"].to(self.device))
            result = dict(self._activations)
        finally:
            for h in self._hook_handles:
                h.remove()
            self._hook_handles.clear()
            self._activations.clear()
            for idx in self._layer_indices:
                self._hook_handles.append(
                    self.model.layers[idx].register_forward_hook(
                        self._make_hook(idx, pool_last=True)
                    )
                )
        return result

    @contextmanager
    def ablate_feature(self, layer: int, direction: torch.Tensor):
        """
        Context manager: project out `direction` from the last-token position of
        `layer`'s output on every forward pass inside the `with` block.
        Implements causal intervention: h_ablated = h - (h·d̂)d̂.
        """
        d_hat = direction.to(self.device)
        d_hat = d_hat / d_hat.norm().clamp(min=1e-10)

        def ablation_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            last = hidden[:, -1:, :]
            proj = (last * d_hat).sum(dim=-1, keepdim=True)
            last_ablated = last - proj * d_hat
            ablated = torch.cat([hidden[:, :-1, :], last_ablated], dim=1)
            return (ablated,) + output[1:] if isinstance(output, tuple) else ablated

        handle = self.model.layers[layer].register_forward_hook(ablation_hook)
        try:
            yield
        finally:
            handle.remove()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @property
    def d_model(self) -> int:
        return self.model.config.hidden_size

    @property
    def n_layers(self) -> int:
        return len(self.model.layers)
