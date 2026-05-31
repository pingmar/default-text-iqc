"""Run the real Vision Mamba feature-differentiation pipeline.

Model: hustvl/vim-tiny-midclstok, 24 bidirectional SSM layers, d_model=192.
Checkpoint: vim_t_midclstok_76p1acc.pth (76.1% ImageNet top-1 accuracy).

The Vim architecture differs from ViT in that spatial tokens are processed
by a bidirectional selective state space model (SSM) rather than attention.
Each Mamba block scans the token sequence forward and backward in parallel,
accumulating hidden state that mixes spatial context recurrently.
"""

from __future__ import annotations

import argparse
import csv
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
import transformers.generation as transformers_generation
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from ssm_extract_ufe_viz.analysis import (
    differentiation_matrix,
    plot_heatmap,
    summarize_metric_confidence,
)
from ssm_extract_ufe_viz.config import (
    BootstrapConfig,
    DifferentiationThresholds,
    DistanceWeights,
    VisualizationConfig,
)
from ssm_extract_ufe_viz.controls import run_controls, signal_to_noise
from ssm_extract_ufe_viz.datasets import balanced_sample_indices, remap_topk_indices
from ssm_extract_ufe_viz.dictionary import FeatureDictionary, write_json
from ssm_extract_ufe_viz.features import (
    build_feature_records,
    decompose_spatial_features,
)
from ssm_extract_ufe_viz.model import MambaVisionProbe
from ssm_extract_ufe_viz.visualization import (
    optimize_feature_visualization,
    plot_feature_grid,
    save_image,
)

warnings.filterwarnings("ignore", category=FutureWarning)

CKPT_REPO = "hustvl/vim-tiny-midclstok"
CKPT_FILE = "vim_t_midclstok_76p1acc.pth"
CKPT_CACHE = "/tmp/vim_cache"
_SELECTIVE_SCAN_CUDA = None
_SELECTIVE_SCAN_REF = None


def _patch_transformers_generation() -> None:
    """Restore symbols old mamba-ssm imports expect from transformers."""

    for sym in ("GreedySearchDecoderOnlyOutput", "SampleDecoderOnlyOutput", "TextStreamer"):
        if not hasattr(transformers_generation, sym):
            setattr(transformers_generation, sym, type(sym, (), {}))


def _load_selective_scan_ops():
    global _SELECTIVE_SCAN_CUDA, _SELECTIVE_SCAN_REF
    if _SELECTIVE_SCAN_REF is not None:
        return _SELECTIVE_SCAN_CUDA, _SELECTIVE_SCAN_REF
    _patch_transformers_generation()
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as selective_scan_cuda
    except Exception:  # pragma: no cover - optional native extension may fail to load
        selective_scan_cuda = None
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

    _SELECTIVE_SCAN_CUDA = selective_scan_cuda
    _SELECTIVE_SCAN_REF = selective_scan_ref
    return _SELECTIVE_SCAN_CUDA, _SELECTIVE_SCAN_REF


def selective_scan_fn(*args: Any, **kwargs: Any):
    """Dispatch to the CUDA kernel when available, else the reference impl."""

    selective_scan_cuda, selective_scan_ref = _load_selective_scan_ops()
    if selective_scan_cuda is not None and any(
        torch.is_tensor(a) and a.is_cuda for a in args
    ):
        return selective_scan_cuda(*args, **kwargs)
    return selective_scan_ref(*args, **kwargs)

# ── Vim architecture ──────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no mean subtraction, no bias)."""

    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class VimMixer(nn.Module):
    """Bidirectional Mamba SSM mixer matching the hustvl/vim-tiny checkpoint."""

    def __init__(
        self,
        d_model: int = 192,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: int = 12,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand  # 384
        self.d_state = d_state
        self.dt_rank = dt_rank

        # Shared input / output projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Forward SSM components
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1
        )
        self.x_proj = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner)
        self.A_log = nn.Parameter(torch.zeros(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Backward SSM components
        self.conv1d_b = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1
        )
        self.x_proj_b = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj_b = nn.Linear(dt_rank, self.d_inner)
        self.A_b_log = nn.Parameter(torch.zeros(self.d_inner, d_state))
        self.D_b = nn.Parameter(torch.ones(self.d_inner))

    # ── internal SSM scan ──────────────────────────────────────────────────

    def _scan(
        self,
        x: torch.Tensor,         # [B, N, d_inner]
        conv1d: nn.Conv1d,
        x_proj: nn.Linear,
        dt_proj: nn.Linear,
        a_log: torch.Tensor,
        d_param: torch.Tensor,
    ) -> torch.Tensor:
        batch, n_tokens, _ = x.shape

        # Causal depthwise conv (trim causal padding)
        u = conv1d(x.transpose(1, 2))[:, :, :n_tokens]  # [B, d_inner, N]
        u = torch_functional.silu(u)

        # Project to delta/B/C
        dbc = x_proj(u.transpose(1, 2))  # [B, N, dt_rank + 2*d_state]
        dt_raw, b_mat, c_mat = dbc.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        # dt_proj: (dt_rank → d_inner) via weight matmul, bias is delta_bias
        dt = (dt_proj.weight @ dt_raw.reshape(-1, self.dt_rank).t()).t()
        dt = dt.reshape(batch, n_tokens, self.d_inner).transpose(1, 2)  # [B, d_inner, N]

        a_matrix = -torch.exp(a_log.float())  # [d_inner, d_state]

        y = selective_scan_fn(
            u,
            dt,
            a_matrix,
            b_mat.transpose(1, 2).float(),  # [B, d_state, N]
            c_mat.transpose(1, 2).float(),  # [B, d_state, N]
            d_param.float(),
            z=None,
            delta_bias=dt_proj.bias.float(),
            delta_softplus=True,
        )  # [B, d_inner, N]
        return y.transpose(1, 2)  # [B, N, d_inner]

    # ── forward ───────────────────────────────────────────────────────────

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        xz = self.in_proj(hidden_states)            # [B, N, 2*d_inner]
        x, z = xz.chunk(2, dim=-1)                  # each [B, N, d_inner]

        y_fwd = self._scan(x, self.conv1d, self.x_proj, self.dt_proj, self.A_log, self.D)
        y_bwd = self._scan(
            x.flip(1), self.conv1d_b, self.x_proj_b, self.dt_proj_b, self.A_b_log, self.D_b
        ).flip(1)

        y = (y_fwd + y_bwd) * torch_functional.silu(z)
        return self.out_proj(y)                     # [B, N, d_model]


class VimBlock(nn.Module):
    def __init__(self, d_model: int = 192) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = VimMixer(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int = 3, embed_dim: int = 192, patch_size: int = 16) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # [B, N, d]


class VimModel(nn.Module):
    """Vision Mamba tiny (midclstok variant, 24 layers, d_model=192)."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        d_model: int = 192,
        n_layers: int = 24,
        n_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        n_patches = (img_size // patch_size) ** 2
        self.n_patches = n_patches
        self.mid_cls_idx = n_patches // 2  # 98

        self.patch_embed = PatchEmbed(in_chans, d_model, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        self.layers = nn.ModuleList([VimBlock(d_model) for _ in range(n_layers)])
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    @classmethod
    def from_pretrained(cls, cache_dir: str = CKPT_CACHE) -> VimModel:
        print("  downloading checkpoint (hustvl/vim-tiny-midclstok)...")
        path = hf_hub_download(CKPT_REPO, CKPT_FILE, cache_dir=cache_dir)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt["model"]
        model = cls()
        missing, unexpected = model.load_state_dict(state, strict=False)
        loaded = len(state) - len(unexpected)
        print(f"  loaded {loaded}/{len(state)} weights  "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        return model

    def _interpolate_pos_embed(self, h: int, w: int) -> torch.Tensor:
        """Interpolate pos_embed when input resolution differs from training size."""
        pos = self.pos_embed  # [1, N_orig+1, d]
        n_orig = self.n_patches  # 196
        n_new = h * w
        if n_new == n_orig:
            return pos
        orig_side = int(n_orig ** 0.5)  # 14
        # Assume pos_embed layout: [cls, patch_0 ... patch_{N-1}]
        cls_pos = pos[:, :1, :]                        # [1, 1, d]
        patch_pos = pos[:, 1:, :]                       # [1, N_orig, d]
        d = patch_pos.shape[-1]
        patch_pos = patch_pos.reshape(1, orig_side, orig_side, d).permute(0, 3, 1, 2)
        patch_pos = torch_functional.interpolate(patch_pos, size=(h, w), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, n_new, d)
        return torch.cat([cls_pos, patch_pos], dim=1)  # [1, N_new+1, d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        tokens = self.patch_embed(x)          # [B, N, d]
        n_tokens = tokens.shape[1]
        h = w = int(n_tokens ** 0.5)

        # Insert CLS token at middle of the SEQUENCE (midclstok variant)
        mid = n_tokens // 2
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([tokens[:, :mid], cls, tokens[:, mid:]], dim=1)

        pos = self._interpolate_pos_embed(h, w)
        tokens = tokens + pos
        for block in self.layers:
            tokens = block(tokens)
        tokens = self.norm_f(tokens)
        return self.head(tokens[:, mid])      # classify via CLS token


# ── probe hook that strips mid-CLS token before _to_spatial ──────────────────


class VimProbe(MambaVisionProbe):
    """Extends MambaVisionProbe to handle the mid-CLS token in Vim."""

    def __init__(self, model: VimModel, device: str = "cpu") -> None:
        super().__init__(model=model, device=device)

    def _to_spatial(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor: [B, N+1, d_model] where CLS is at N//2 of the patch sequence
        if tensor.ndim == 3:
            _, token_count, _ = tensor.shape
            n_patches = token_count - 1           # e.g. 196 for 224-px input
            cls_pos = n_patches // 2    # CLS inserted at middle of patches
            tensor = torch.cat([tensor[:, :cls_pos], tensor[:, cls_pos + 1:]], dim=1)
        return super()._to_spatial(tensor)


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


# ── main pipeline ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract, visualize, and validate feature directions from Vim."
    )
    parser.add_argument("--layer", default="layers.22", help="Hooked Vim layer, for example layers.18 or layers.22.")
    parser.add_argument("--output-dir", default="results_imagenet_l22")
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--samples-per-class", type=int,
        help="Explicit balanced sample cap per class; overrides --max-samples.")
    parser.add_argument("--image-root", required=True,
        help="ImageFolder root for an ImageNet or ImageNet-style validation set.")
    parser.add_argument("--dataset-name",
        help="Name written to metadata and output filenames. Defaults from image root.")
    parser.add_argument("--decomposition", choices=("pca", "ica"), default="pca")
    parser.add_argument("--top-k", type=int, default=5,
        help="Number of top source images stored per feature.")
    parser.add_argument("--n-components", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--viz-steps", type=int, default=200)
    parser.add_argument("--viz-size", type=int, default=128)
    parser.add_argument("--n-facets", type=int, default=1,
        help="Distill diversity-loss facets per feature (1 = single image).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0,
        help="Random seed for torch + numpy + activation-max init.")
    parser.add_argument("--cpu-smoke", action="store_true",
        help="Use tiny defaults so the pipeline finishes on CPU for a smoke run.")
    args = parser.parse_args()
    if args.cpu_smoke:
        args.device = "cpu"
        args.max_samples = min(args.max_samples, 32)
        if args.samples_per_class is not None:
            args.samples_per_class = min(args.samples_per_class, 4)
        args.n_components = min(args.n_components, 8)
        args.viz_steps = min(args.viz_steps, 40)
        args.viz_size = min(args.viz_size, 64)
        args.batch_size = min(args.batch_size, 4)
        args.num_workers = 0

    out = Path(args.output_dir)
    (out / "features").mkdir(parents=True, exist_ok=True)
    (out / "viz").mkdir(parents=True, exist_ok=True)
    (out / "validation").mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Load pretrained Vision Mamba
    print("[0/4] loading Vision Mamba (vim-tiny-midclstok)...")
    model = VimModel.from_pretrained()
    model.eval()
    probe = VimProbe(model=model, device=args.device)
    probe.register_hooks([args.layer])

    # 2. ImageFolder corpus, typically an ImageNet validation subset.
    val_dir = Path(args.image_root)
    dataset_name = args.dataset_name or val_dir.name
    dataset_slug = _slug(dataset_name)
    corpus_note = "ImageFolder dataset, centre-cropped to 224px"
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    full_ds = ImageFolder(root=str(val_dir), transform=transform)
    indices = balanced_sample_indices(
        full_ds.targets,
        len(full_ds.classes),
        max_samples=args.max_samples,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    n_samples = len(indices)
    loader = DataLoader(
        Subset(full_ds, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"  {n_samples} images across {len(full_ds.classes)} classes: {full_ds.classes}")

    # 3. Collect spatial activations
    print(f"[1/4] collecting activations ({n_samples} samples on {args.device})...")
    t0 = time.time()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch, _ in loader:
            # VimProbe removes the mid-CLS token and returns [B,H,W,D].
            spatial = probe.collect_batch_spatial(batch)
            chunks.append(spatial[args.layer].numpy())
    activations = np.concatenate(chunks, axis=0)
    print(f"  activations shape: {activations.shape}  ({time.time() - t0:.1f}s)")

    # 4. Feature extraction
    print(f"[2/4] decomposing into feature directions ({args.decomposition.upper()})...")
    directions, scores = decompose_spatial_features(
        activations,
        args.n_components,
        method=args.decomposition,
        seed=args.seed,
    )
    records = build_feature_records(
        activations,
        directions,
        scores,
        layer=args.layer,
        decomposition=args.decomposition,
        top_k=args.top_k,
    )
    # build_feature_records sees the sampled activation matrix, so its top-k
    # rows are subset-relative. Store source ImageFolder indices for downstream
    # validation and hierarchy checks that receive the full image root.
    remap_topk_indices(records, indices)

    dictionary = FeatureDictionary(
        records=records,
        metadata={
            "model_name": "hustvl/vim-tiny-midclstok",
            "model_note": "Real Vision Mamba SSM, 24 bidirectional layers",
            "corpus": dataset_name,
            "corpus_note": corpus_note,
            "image_root": str(val_dir),
            "layer": args.layer,
            "n_components": args.n_components,
            "decomposition": args.decomposition,
            "top_k": args.top_k,
            "max_samples": n_samples,
            "requested_samples": args.max_samples,
            "samples_per_class": args.samples_per_class,
            "sample_seed": args.seed,
            "top_k_index_space": "image_root_sorted_sample_index",
            "device": args.device,
        },
    )
    layer_suffix = args.layer.replace(".", "_")
    method_suffix = "" if args.decomposition == "pca" else f"_{args.decomposition}"
    dict_path = out / "features" / f"vim_{dataset_slug}{method_suffix}_{layer_suffix}.json"

    # 5. Activation maximization on GPU — 200 steps at 128px gives a good
    # balance between visual quality and runtime on a single accelerator.
    print(f"[3/4] optimizing feature visualizations ({args.n_components} features on {args.device})...")
    cfg = VisualizationConfig(
        n_steps=args.viz_steps,
        lr=0.05,
        image_size=args.viz_size,
        n_facets=args.n_facets,
        color_decorrelation=True,
        fourier_init=True,
        jitter=8,
        rotate_degrees=5.0,
        scale_range=(0.95, 1.05),
        tv_weight=5e-5,
        l2_weight=5e-5,
        seed=args.seed,
    )
    image_paths: list[Path] = []
    labels: list[str] = []
    t_viz = time.time()
    for record in records:
        t0 = time.time()
        tensor = optimize_feature_visualization(
            probe, args.layer, np.asarray(record.vector, dtype=np.float32), cfg, device=args.device,
        )
        path = out / "viz" / f"feature_{record.feature_id:02d}.png"
        save_image(tensor, path)
        record.visualization_path = str(path)
        image_paths.append(path)
        labels.append(f"f{record.feature_id:02d}")
        done = record.feature_id + 1
        remaining = (time.time() - t_viz) / done * (args.n_components - done)
        print(
            f"  feature {record.feature_id:02d}: "
            f"{time.time() - t0:.1f}s  ~{remaining / 60:.1f} min left"
        )
    plot_feature_grid(image_paths, labels, out / "viz" / "grid.png", n_cols=4)
    dictionary.save(dict_path)

    # 6. Pairwise differentiation
    print("[4/4] computing pairwise differentiation...")
    thresholds = DifferentiationThresholds()
    weights = DistanceWeights.for_decomposition(args.decomposition)
    bool_matrix, dist_matrix, score_grid = differentiation_matrix(records, thresholds, weights)
    plot_heatmap(
        bool_matrix, out / "validation" / "heatmap.png",
        title="Vim feature differentiation (1 = different)",
        vmin=0.0, vmax=1.0,
    )
    plot_heatmap(
        dist_matrix, out / "validation" / "distance_heatmap.png",
        title="Vim composite pair distance",
        vmin=0.0, vmax=float(dist_matrix.max()) if dist_matrix.size else 1.0,
    )

    pair_scores: list[dict] = []
    with (out / "validation" / "pairwise_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = None
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                row = {"i": i, "j": j, **score_grid[i][j], "different": bool(bool_matrix[i, j])}
                pair_scores.append(score_grid[i][j])
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(row), lineterminator="\n")
                    writer.writeheader()
                writer.writerow(row)

    summary = summarize_metric_confidence(
        pair_scores, bootstrap=BootstrapConfig(n_bootstrap=500, seed=0),
    )
    write_json(out / "validation" / "confidence_intervals.json", summary)

    controls = run_controls(records, weights)
    write_json(
        out / "validation" / "controls.json",
        {key: value.to_dict() for key, value in controls.items()},
    )

    different_pairs = int(bool_matrix.sum() // 2)
    total_pairs = args.n_components * (args.n_components - 1) // 2
    distance_mean = float(dist_matrix[np.triu_indices_from(dist_matrix, k=1)].mean())
    snr = signal_to_noise(controls)
    write_json(
        out / "validation" / "summary.json",
        {
            "dictionary": str(dict_path),
            "dataset": dataset_name,
            "image_root": str(val_dir),
            "layer": args.layer,
            "decomposition": args.decomposition,
            "n_components": args.n_components,
            "seed": args.seed,
            "top_k": args.top_k,
            "n_samples": n_samples,
            "different_pairs": different_pairs,
            "total_pairs": total_pairs,
            "distance_mean": distance_mean,
            "snr": snr,
            "controls": {key: value.to_dict() for key, value in controls.items()},
        },
    )
    print()
    print(f"different pairs (gate): {different_pairs} / {total_pairs}")
    print(f"composite distance     mean={distance_mean:.3f}")
    print(f"controls: identical={controls['identical'].mean:.3f}  "
          f"perturbed={controls['perturbed'].mean:.3f}  "
          f"real={controls['real'].mean:.3f}  "
          f"SNR={snr:.2f}")
    print()
    print("metric           estimate  [95% CI]")
    for key, ci in summary.items():
        print(f"  {key:16s} {ci['estimate']:+.3f}  [{ci['lower']:+.3f}, {ci['upper']:+.3f}]")
    print()
    print("artifacts:")
    print(f"  dictionary:     {dict_path}")
    print(f"  feature grid:   {out / 'viz' / 'grid.png'}")
    print(f"  metrics CSV:    {out / 'validation' / 'pairwise_metrics.csv'}")
    print(f"  CI summary:     {out / 'validation' / 'confidence_intervals.json'}")
    print(f"  controls:       {out / 'validation' / 'controls.json'}")
    print(f"  heatmap:        {out / 'validation' / 'heatmap.png'}")
    print(f"  distance map:   {out / 'validation' / 'distance_heatmap.png'}")


if __name__ == "__main__":
    main()
