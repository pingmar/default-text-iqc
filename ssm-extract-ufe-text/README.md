# SSM Extract UFE Text

Feature interpretability and differentiation framework for Mamba state-space language models.

Establishes a formal, mathematically grounded framework for deciding when two text-based features extracted from a neural language model are meaningfully distinct. Built on Mamba-130M with mechanistic interpretability techniques inspired by circuit analysis.

---

## Framework Overview

```
Text Corpus
    │
    ▼
MambaProbe          ← forward hooks collect layer activations [N, d_model]
    │
    ▼
Decomposition       ← PCA or NMF → feature directions [K, d_model] + projections [N, K]
    │
    ▼
FeatureDictionary   ← per-feature: vector, activation histogram, top-k examples
    │
    ├─ SemanticLabeling   ← probe sentences × 8 linguistic categories → label per feature
    ├─ PolysemanticitySc. ← top-k context diversity → score ∈ [0, 1]
    └─ Orthogonalization  ← SVD on top-activating samples + QR → disentangled sub-features
    │
    ▼
Differentiation Metrics
    ├─ Cosine similarity   (direction vectors or activation profiles)
    ├─ Jensen-Shannon div. (activation histograms, base-2)
    └─ Linear CKA          (Kornblith et al. 2019)
    │
    ▼
ε-Differentiation Test  ← three-way AND gate: cos < ε_cos, JSD > ε_jsd, CKA < ε_cka
```

### What "different" means formally

Two features *i* and *j* are **ε-different** iff all three hold simultaneously:

- `|cos(v_i, v_j)| < ε_cos` — their direction vectors (or activation profiles) are not aligned
- `JSD(P_i ‖ P_j) > ε_jsd` — their activation distributions are informationally separated
- `CKA(A_i, A_j) < ε_cka` — their representational geometry is dissimilar

Default thresholds: `ε_cos = 0.3`, `ε_jsd = 0.2`, `ε_cka = 0.8`. JSD is the primary discriminator for PCA decompositions (where direction cosine is trivially ≈ 0 by orthogonality). All three carry signal for NMF.

---

## Installation

```bash
cd ssm-extract-ufe-text
pip install -e .

# Optional: GPU-accelerated Mamba kernels (~3× faster inference)
pip install -e ".[gpu]"
```

Requires Python ≥ 3.12. Model weights (~500 MB) are downloaded automatically from HuggingFace on first run.

---

## End-to-End Usage

### Step 1 — Extract features

Run for each corpus (`sentiment`, `syntactic`, `ner`). `--save-activations` is required for the cross-projection analysis in step 3.

```bash
for CORPUS in sentiment syntactic ner; do
  python3 scripts/extract_features.py \
    --corpus        $CORPUS \
    --layers        6 \
    --n-components  32 \
    --decomposition pca \
    --max-samples   1000 \
    --output-dir    results/features/ \
    --save-activations
done
```

**Outputs per corpus:** `{corpus}_l6.json` (FeatureDictionary), `{corpus}_l6_projections.npy`, `{corpus}_l6_activations.npy`.

Use `--decomposition nmf` for NMF decomposition (part-based, non-negative, encourages monosemanticity). Use `--layers 6 12` to extract from multiple layers simultaneously.

### Step 2 — Assign semantic labels and polysemanticity scores

```bash
for CORPUS in sentiment syntactic ner; do
  python3 scripts/label_features.py \
    --dict-path  results/features/${CORPUS}_l6.json \
    --layer      6
done
```

Runs 8-category probe sentence banks (120 sentences total) through the model. Each feature direction is scored by projecting per-category mean activations onto it; the winning category by z-score becomes the feature's `semantic_label`. Polysemanticity is estimated from the cosine diversity of the top-k activating samples' activation contexts.

**Probe categories:** `positive_sentiment`, `negative_sentiment`, `negation`, `named_entity_person`, `named_entity_location`, `passive_voice`, `comparative`, `question`.

### Step 3 — Cross-corpus validation

```bash
python3 scripts/validate_dictionary.py \
  --sentiment-dict  results/features/sentiment_l6.json \
  --syntactic-dict  results/features/syntactic_l6.json \
  --ner-dict        results/features/ner_l6.json \
  --layer           6 \
  --output-dir      results/validation/
```

**Console output:**
- Cross-corpus mean cosine / JSD / CKA / ε-different % table
- Per-feature semantic label and polysemanticity summary
- Top-3 most ε-different feature pairs with parallel top-k examples

**Saved files:**
- `results/validation/sentiment_l6_heatmap.png` — pairwise feature similarity matrix
- `results/validation/sentiment_l6_histograms.png` — activation distribution grid

### Optional — Orthogonalize polysemantic features

Apply SVD-based sub-feature decomposition to features above a polysemanticity threshold:

```python
import numpy as np
from ssm_extract_ufe_text.dictionary import FeatureDictionary
from ssm_extract_ufe_text.features import orthogonalize_polysemantic

fd    = FeatureDictionary.load("results/features/sentiment_l6.json")
projs = np.load("results/features/sentiment_l6_projections.npy")
acts  = np.load("results/features/sentiment_l6_activations.npy")

recs   = fd.records_for_layer(6)
comps  = np.array([r.vector for r in recs])
scores = [r.polysemanticity for r in recs]

new_comps, new_projs = orthogonalize_polysemantic(
    acts, comps, projs, scores, threshold=0.5, n_sub=2
)
print(f"Before: {comps.shape[0]} features  →  After: {new_comps.shape[0]} features")
```

For each polysemantic feature, the top-50 activating samples are collected, SVD is run on their activation sub-matrix, and the top `n_sub` right singular vectors replace the original direction. The full set is then re-orthogonalized via QR decomposition.

### Optional — Token-level analysis and causal ablation

```python
import torch
from ssm_extract_ufe_text.model import MambaProbe
from ssm_extract_ufe_text.config import ExtractionConfig

config = ExtractionConfig(layers=[6], device="cpu")
probe  = MambaProbe.from_pretrained(config)

# Per-token activations: {layer: Tensor[B, T, d_model]}
batch = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
all_tokens = probe.collect_batch_all_tokens(batch)

# Causal ablation: project out a feature direction during forward pass
direction = torch.tensor(recs[17].vector)  # e.g. the "negation" feature
with probe.ablate_feature(layer=6, direction=direction):
    ablated = probe.collect_batch(batch)
```

`ablate_feature` implements `h_ablated = h - (h · d̂) d̂` at the last token position of the target layer, causally removing a feature's contribution to downstream computation.

### Optional — Token-feature heatmap

```python
from ssm_extract_ufe_text.analysis import plot_token_feature_heatmap
import numpy as np

# token_acts: (T, d_model) — one row per token from collect_batch_all_tokens
token_acts = all_tokens[6][0].numpy()   # first sample, layer 6
components = np.array([r.vector for r in recs])
tokens     = ["The", "film", "was", "not", "good"]

plot_token_feature_heatmap(
    token_acts, components, tokens,
    feature_labels=[r.semantic_label or f"f{r.feature_id}" for r in recs],
    output_path="results/token_heatmap.png",
    top_k_features=16,
)
```

---

## Module Reference

| Module | Responsibility |
|---|---|
| `config.py` | `ExtractionConfig`, `DifferentiationThresholds` |
| `corpus.py` | SST-2 (sentiment), WikiText-2 (NER), PTB (syntactic) loaders |
| `model.py` | `MambaProbe` — hook-based activation collection, token-level collection, causal ablation |
| `features.py` | PCA/NMF decomposition, histograms, top-k examples, polysemantic orthogonalization |
| `dictionary.py` | `FeatureRecord`, `FeatureDictionary` — typed container with JSON I/O |
| `probing.py` | Probe sentence banks, semantic label assignment |
| `metrics.py` | `cosine_similarity`, `js_divergence`, `linear_cka` |
| `analysis.py` | ε-differentiation, differentiation matrix, cross-corpus statistics, visualization |

---

## Tests

```bash
python3 -m pytest tests/ -v
```

69 unit tests. No model download required — model tests use a randomly initialized tiny Mamba config.

```
tests/test_metrics.py        13 tests — cosine, JSD, CKA edge cases and invariances
tests/test_features.py       19 tests — PCA, NMF, histograms, top-k, orthogonalization
tests/test_probing.py        29 tests — probe sentences, label assignment, polysemanticity
tests/test_visualization.py   8 tests — histogram grid, token-feature heatmap
```

---

## Empirical Results (Mamba-130M, layer 6, PCA, 32 components)

| Corpus pair | mean JSD | ε-different % |
|---|---|---|
| sentiment × sentiment | 0.148 | 25.0% |
| sentiment × NER | 0.299 | 62.2% |
| syntactic × NER | 0.325 | 70.2% |
| NER × NER | 0.398 | 68.8% |

Cross-domain pairs score ~2× higher JSD than within-domain pairs, confirming that the ε-differentiation metric captures genuine distributional separation rather than noise.

Polysemanticity is pervasive under PCA: 81% of sentiment-corpus features scored > 0.5. Orthogonalization expands 32 features to 58 disentangled sub-features (26 polysemantic features split into 2 sub-directions each).

The most ε-different pair on the sentiment corpus: **f0 (question, poly=0.028) vs f17 (negation, poly=0.635)**, JSD = 0.748 — confirmed by orthogonal top-k example distributions (statement fragments vs. explicit negation tokens).
