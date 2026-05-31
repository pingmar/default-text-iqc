"""Vision SSM probing helpers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class HookHandle:
    layer: int | str
    handle: Any


class MambaVisionProbe:
    """Collect spatial activations and feature objectives from a vision model."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model
        self.model.to(self.device)
        self.model.eval()
        self._captures: dict[int | str, torch.Tensor] = {}
        self._handles: list[HookHandle] = []

    def register_hooks(self, layers: list[int | str]) -> None:
        self.clear_hooks()
        modules = dict(self.model.named_modules())
        module_values = list(modules.items())
        for layer in layers:
            if isinstance(layer, str):
                if layer not in modules:
                    raise KeyError(f"unknown module name: {layer}")
                target = modules[layer]
            else:
                candidates = [(name, module) for name, module in module_values if _looks_like_block(name)]
                if not candidates:
                    candidates = module_values
                if layer >= len(candidates):
                    raise IndexError(f"layer index {layer} out of range for {len(candidates)} modules")
                _, target = candidates[layer]
            handle = target.register_forward_hook(self._make_hook(layer))
            self._handles.append(HookHandle(layer=layer, handle=handle))

    def clear_hooks(self) -> None:
        for item in self._handles:
            item.handle.remove()
        self._handles.clear()
        self._captures.clear()

    @contextmanager
    def hooks(self, layers: list[int | str]):
        self.register_hooks(layers)
        try:
            yield self
        finally:
            self.clear_hooks()

    @torch.no_grad()
    def collect_batch_spatial(self, batch: torch.Tensor) -> dict[int | str, torch.Tensor]:
        """Return captured activations as [B,H,W,D] tensors."""

        if not self._handles:
            raise RuntimeError("register_hooks must be called before collecting activations")
        self._captures.clear()
        self._forward(batch.to(self.device))
        return {layer: self._to_spatial(tensor).detach().cpu() for layer, tensor in self._captures.items()}

    @torch.no_grad()
    def collect_batch_pooled(self, batch: torch.Tensor) -> dict[int | str, torch.Tensor]:
        spatial = self.collect_batch_spatial(batch)
        return {layer: tensor.mean(dim=(1, 2)) for layer, tensor in spatial.items()}

    def feature_activation_map(
        self,
        images: torch.Tensor,
        layer: int | str,
        feature_dir: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Per-pixel projection onto a unit feature direction. Returns [B, H, W]."""

        if layer not in [handle.layer for handle in self._handles]:
            self.register_hooks([layer])
        self._captures.clear()
        self._forward(images.to(self.device))
        if layer not in self._captures:
            raise RuntimeError(f"layer {layer!r} did not produce a captured tensor")
        spatial = self._to_spatial(self._captures[layer])
        direction = torch.as_tensor(feature_dir, dtype=spatial.dtype, device=spatial.device)
        direction = direction / direction.norm().clamp_min(1e-12)
        return torch.einsum("bhwd,d->bhw", spatial, direction)

    def feature_activation(
        self,
        images: torch.Tensor,
        layer: int | str,
        feature_dir: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Mean projection over the batch and spatial dims. Returns a scalar."""

        return self.feature_activation_map(images, layer, feature_dir).mean()

    def _make_hook(self, layer: int | str):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(tensor):
                self._captures[layer] = tensor

        return hook

    def _forward(self, images: torch.Tensor):
        try:
            return self.model(images)
        except TypeError:
            return self.model(pixel_values=images)

    def _to_spatial(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 4:
            # Convert CNN-like [B,C,H,W] to the shared [B,H,W,D] shape.
            return tensor.permute(0, 2, 3, 1)
        if tensor.ndim != 3:
            raise ValueError(f"cannot interpret activation shape {tuple(tensor.shape)} as spatial")
        batch, tokens, channels = tensor.shape
        side = int(round(tokens**0.5))
        if side * side != tokens and tokens > 1:
            # Generic ViT-like models often expose a leading CLS token.
            side = int(round((tokens - 1) ** 0.5))
            if side * side == tokens - 1:
                tensor = tensor[:, 1:, :]
                tokens -= 1
        side = int(round(tokens**0.5))
        if side * side != tokens:
            raise ValueError(f"token count {tokens} is not square and cannot be spatialized")
        return tensor.reshape(batch, side, side, channels)

def _looks_like_block(name: str) -> bool:
    lowered = name.lower()
    return any(part in lowered for part in ("block", "layer", "mixer"))
