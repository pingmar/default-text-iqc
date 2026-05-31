# Visual Feature Differentiation for Vision SSMs

A calibrated **scalar distance** between features extracted from a
vision-focused state-space model (Vision Mamba / Vim), plus a derived
boolean gate (`epsilon_different`), calibration controls (identical /
perturbed / real), a hierarchy-aware visual dictionary, and a cross-layer
hierarchy probe.

The visualization recipe follows Distill's
["Feature Visualization"](https://distill.pub/2017/feature-visualization/) and
["Zoom In: An Introduction to Circuits"](https://distill.pub/2020/circuits/zoom-in/):
decorrelated color space, Fourier-filtered init, padded jitter, random
rotation and scale, optional diversity loss across facets.

## TL;DR

The project targets ImageNet or ImageNet-style `ImageFolder` datasets
directly. No dataset samples are committed to the repository.

The intended study is:

- Run Vim feature dictionaries over a class-balanced ImageNet validation
  subset.
- Sweep layers, seeds, feature counts, and PCA/ICA decompositions.
- Report separate claims: operational calibration, visual dictionary
  compression, and cross-layer hierarchy.

Do not interpret a high operational gate count as natural-language concept
distance. The core definition is operational and visual, not a language
alignment claim.

## Definition

A **feature** is a unit vector `v` in a chosen layer's activation space.
Vim block outputs are reshaped to `[batch, height, width, d_model]`; PCA
or ICA over the spatially-pooled activations yields directions
`v_1, ..., v_K`. Patch-level PCA (`pool=False` in
`features.spatial_to_matrix`) is also supported as an opt-in.

For a direction `v`, activation maximization optimizes an input image to
maximize `mean_spatial(activation(layer, image) . v)` under
Distill-style transformation robustness.

The **composite pair distance** is a non-negative scalar:

```text
d(i, j) = w_cos * (1 - cos(v_i, v_j))
        + w_jsd * JSD(hist_i, hist_j)
        + w_cka * (1 - CKA(activations_i, activations_j))      # if available
        + w_lpips * LPIPS(viz_i, viz_j)
        + w_ssim  * SSIM_distance(viz_i, viz_j)
        + w_iou   * (1 - IoU(map_i, map_j))
```

Axes whose underlying score is unavailable are **dropped**, not silently
masked. Default weights: `w_cos=1`, `w_jsd=2`, `w_cka=1`,
`w_lpips=1`, `w_ssim=1`, `w_iou=0.5`.
For PCA dictionaries, cosine distance is logged but not weighted because
PCA components are orthogonal by construction, so `1 - cos` is a constant
shift rather than a discriminator. ICA keeps the cosine weight because ICA
components are not necessarily orthogonal.
CKA is opt-in at the library level: generated dictionaries do not store
raw activation matrices, so CLI validation reports omit CKA unless a caller
passes raw activations directly to `compute_pair_scores`.

The derived boolean gate `epsilon_different` fires when every per-axis
condition holds:

```text
activation gate: cos < eps_cos and JSD > eps_jsd and (CKA absent or CKA < eps_cka)
visual gate:    LPIPS > eps_lpips and SSIM_distance > eps_ssim and IoU < eps_iou
```

`top_k_overlap` is reported separately as hierarchy evidence; it does
not gate.

## Validation Axes

`scripts/run_controls.py` and the built-in step in `run_demo_vim.py`
produce three reference distributions of the composite distance:

| Control | Meaning | Expected |
|---|---|---|
| `identical` | distance of each record to itself | near 0 |
| `perturbed` | distance to a noise-perturbed copy of the same record | small |
| `real` | distance over all real `(i, j)` pairs from the dictionary | larger |

`signal_to_noise = real.mean / perturbed.mean`. A large SNR means the
metric separates real pair distances above the small-perturbation noise
floor.

`scripts/validate_dictionary.py` also emits a hierarchy-aware table:
`d_content` is the content-like distance over direction and optimized
visualization axes, while `d_context` captures corpus and hierarchy
evidence such as top-k overlap, spatial IoU, and optional parent
Jaccard. The formal statement is in `docs/formal_definition.md`.
The empirical p-value can hit its finite-sample floor; the report also
writes an effect size that rescales the observed activation-axis distance
in perturbation-null standard deviations.

`scripts/cross_layer_hierarchy.py` reports how often deep-layer feature
pairs share shallow-layer parents by top-k source-image overlap. Use
`--top-k-limit` to run hierarchy sensitivity at 5/10/20 examples, then
pass its JSON to `validate_dictionary.py --parent-index-path` when
validating a deep-layer dictionary.

## Repository Layout

```text
scripts/
  run_demo_vim.py            # one Vim ImageFolder run
  run_imagenet_study.py      # layer/seed/K/method sweep runner
  aggregate_study.py         # CSV/JSON/Markdown study summary
  validate_dictionary.py     # metrics + controls for a saved dictionary
  run_controls.py            # standalone calibration
  cross_layer_hierarchy.py   # shallow -> deep parent overlap
docs/
  formal_definition.md       # hierarchy-aware definition and caveats
src/ssm_extract_ufe_viz/
  analysis.py                # pair scores, gates, matrices, hierarchy
  config.py                  # weights, thresholds, viz settings
  controls.py                # identical / perturbed / real controls
  datasets.py                # deterministic ImageFolder sampling helpers
  differentiation.py         # content/context split and visual dictionary
  dictionary.py              # JSON feature records
  features.py                # PCA/ICA and spatial maps
  image_metrics.py           # LPIPS / SSIM / IoU / top-k overlap
  metrics.py                 # cosine / JSD / CKA / bootstrap / distance
  model.py                   # spatial activation probe
  visualization.py           # Distill-style activation maximization
tests/
```

Generated artifacts (`results/`, `results_*/`) are gitignored.

## Setup

This project is managed with [uv](https://github.com/astral-sh/uv). The
checked-in `.python-version` pins new local environments to Python 3.12;
Python 3.13 is avoided for the model extras because some Torch/CUDA wheels
used by the Vim path are not available for `cp313`.

Lightweight tests:

```bash
uv venv
uv sync --extra dev
uv run pytest tests/ -q
```

For the real Vim pipeline, install PyTorch for the host CUDA/CPU setup
first. Do not let `mamba-ssm` pull an implicit Torch wheel: the local CUDA
toolkit used for building `mamba-ssm` must match `torch.version.cuda`.

CUDA 12.1 example:

```bash
uv venv --python 3.12
uv sync --extra dev --extra models
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

CPU-only smoke runs can instead use the CPU PyTorch index:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

For real Vim CUDA runs, then build/install Vim's selective-scan dependency
against that already installed Torch wheel:

```bash
uv pip install ninja einops wheel setuptools
MAX_JOBS=2 uv pip install --no-build-isolation --no-deps "mamba-ssm==2.2.4"
```

Optional extras:

```bash
uv pip install -e ".[lpips]"      # real LPIPS instead of L2 fallback
uv pip install -e ".[study]"      # ICA / study sweep dependencies
```

## Data Layout

Inputs use the standard `torchvision.datasets.ImageFolder` layout: each
subdirectory is one class, and the files inside can be ordinary `.jpg`,
`.jpeg`, or `.png` images.

```text
imagenet-val-subset/
  n01440764/
    ILSVRC2012_val_00000293.JPEG
  n01443537/
    ILSVRC2012_val_00000236.JPEG
```

The repository does not distribute ImageNet images for licensing reasons.
Use an official ImageNet validation set obtained through
https://image-net.org/ or another ImageNet-1k source you are licensed to
access, then arrange it in the `ImageFolder` structure above. You can pass
the full validation root to `--image-root`; `--samples-per-class` builds a
class-balanced subset inside the script, and `--max-samples` remains the
global upper bound.

Tiny CPU smoke fixture:

```bash
mkdir -p /tmp/tiny-imagefolder/class_a /tmp/tiny-imagefolder/class_b
python - <<'PY'
from pathlib import Path
from PIL import Image

root = Path("/tmp/tiny-imagefolder")
for cls, color in {"class_a": (220, 40, 40), "class_b": (40, 80, 220)}.items():
    for i in range(2):
        image = Image.new("RGB", (64, 64), tuple(min(255, c + i * 20) for c in color))
        image.save(root / cls / f"{i}.png")
PY
```

## Single Vim Run

Run one layer on an ImageNet validation subset laid out as `ImageFolder`:

```bash
uv run python -u scripts/run_demo_vim.py \
  --image-root /path/to/imagenet-val-subset \
  --dataset-name imagenet_subset \
  --layer layers.18 \
  --output-dir results_imagenet_subset/pca/k16/seed0/layers_18 \
  --max-samples 10000 \
  --n-components 16 \
  --decomposition pca \
  --top-k 20 \
  --viz-steps 200 \
  --viz-size 128
```

CPU smoke run with a tiny local `ImageFolder` fixture:

```bash
uv run python -u scripts/run_demo_vim.py \
  --image-root /path/to/tiny-imagefolder \
  --dataset-name smoke \
  --cpu-smoke \
  --output-dir results_smoke
```

Each run writes:

```text
results_*/features/*.json                       # feature dictionary
results_*/viz/feature_*.png                     # optimized inputs per feature
results_*/viz/grid.png                          # comparison grid
results_*/validation/pairwise_metrics.csv       # per-axis scores + distance + gate
results_*/validation/confidence_intervals.json  # bootstrap CIs
results_*/validation/controls.json              # identical / perturbed / real
results_*/validation/summary.json               # compact run summary
results_*/validation/heatmap.png                # gate matrix
results_*/validation/distance_heatmap.png       # scalar distance matrix
```

## Medium ImageNet-Subset Study

Run the default medium grid:

```bash
uv run python -u scripts/run_imagenet_study.py \
  --image-root /path/to/imagenet-val-subset \
  --output-root results_imagenet_subset \
  --dataset-name imagenet_subset \
  --max-samples 10000 \
  --layers layers.12,layers.16,layers.18,layers.20,layers.22 \
  --seeds 0,1,2 \
  --components 16,32 \
  --methods pca,ica \
  --top-k 20
```

This runs each Vim dictionary, validation metrics, and adjacent-layer
hierarchy probes at top-k limits 5/10/20. The runner is restartable with
`--skip-existing`.

Aggregate completed runs:

```bash
uv run python -u scripts/aggregate_study.py \
  --results-root results_imagenet_subset
```

The aggregator writes `study_summary.json`, `study_summary.csv`, and
`study_summary.md` with SNR, gate count, and top-k overlap per
layer/seed/K/method.
Visual-dictionary compression is meaningful mainly when `K` is intentionally
overcomplete; singleton classes such as `8/8` are reported as no compression
observed, not as evidence of compression.

## Standalone Validation

After running two depths, quantify hierarchy:

```bash
uv run python -u scripts/cross_layer_hierarchy.py \
  --shallow results_imagenet_subset/pca/k16/seed0/layers_18/features/vim_imagenet_subset_layers_18.json \
  --deep    results_imagenet_subset/pca/k16/seed0/layers_22/features/vim_imagenet_subset_layers_22.json \
  --threshold 0.2 \
  --top-k-limit 20
```

Validate a saved dictionary and emit the hierarchy-aware classification,
equivalence classes, report, and visual grids:

```bash
uv run python -u scripts/validate_dictionary.py \
  --dict-path results_imagenet_subset/pca/k16/seed0/layers_22/features/vim_imagenet_subset_layers_22.json \
  --output-dir results_imagenet_subset/pca/k16/seed0/layers_22/validation \
  --parent-index-path results_imagenet_subset/pca/k16/seed0/hierarchy/layers_18_to_layers_22/thr0.2_top20.json
```

## Limitations

- ImageNet subset composition controls what visual evidence can be observed. Use
  class-balanced sampling and report the exact `--image-root`, seed, and
  sample count from `summary.json`.
- LPIPS is optional: without `.[lpips]`, the implementation reports a
  normalized L2 proxy under the same column name.
- Activation maximization reveals what drives a direction; it does not
  prove that one natural-language concept fully explains it.
- Shared lower-level dependencies are surfaced via `top_k_overlap` and
  `cross_layer_overlap`, but hierarchy rates are sensitive to top-k.
