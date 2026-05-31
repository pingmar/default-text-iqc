# Formal definition: when are two SSM features "different"?

This document gives the mathematical statement of feature differentiation
used by `ssm_extract_ufe_viz`. It accompanies the code in
`src/ssm_extract_ufe_viz/differentiation.py` and the validation pipeline
in `scripts/validate_dictionary.py`.

The goal, following the distill.pub *Circuits / Zoom-In* line of work,
is to specify — without human labels — what it means for a pair of
feature directions $(f_i, f_j)$ extracted from a vision SSM to be
*different*, while acknowledging that high-level features can share
lower-level dependencies (e.g. a "car" feature and a "bicycle" feature
both depend on a "wheel" feature).

## 1. Per-axis pair scores

For two `FeatureRecord`s the package computes a set of per-axis scores
via `analysis.compute_pair_scores`.
Each axis maps to a non-negative *distance contribution*. Most axes are
bounded in $[0, 1]$; $1 - \cos$ is direction-sensitive and can reach $2$,
and LPIPS is model-dependent rather than strictly bounded:

| Axis            | Raw quantity                          | Distance form |
|-----------------|---------------------------------------|----------------|
| `cos`           | cosine of feature directions          | $1 - \cos$     |
| `cka`           | linear CKA on raw activations         | $1 - \mathrm{CKA}$ |
| `lpips`         | LPIPS on optimized visualizations     | `LPIPS`        |
| `ssim_distance` | $1 - \mathrm{SSIM}$ on visualizations | itself         |
| `jsd`           | JS divergence of activation histograms| itself         |
| `topk_overlap`  | Jaccard of top-k example indices      | $1 - \mathrm{overlap}$ |
| `map_iou`       | IoU of high-activation spatial regions| $1 - \mathrm{IoU}$ |
| `parent_jaccard`| Jaccard of shallow-layer parent sets  | $1 - \mathrm{Jaccard}$ |

`parent_jaccard` is optional and only well-defined when a parent index
from `cross_layer_overlap` is supplied alongside the
deep-layer records. If neither feature has parent evidence, the axis is
dropped for that pair rather than treating "unknown" as shared context.
CKA is also opt-in: generated dictionaries do not store raw activation
matrices, so CLI validation reports omit CKA unless a library caller passes
`activations_i` and `activations_j` directly to `compute_pair_scores`.
For PCA dictionaries, cosine distance is logged but assigned zero weight:
PCA components are orthogonal by construction, so $1 - \cos$ is a constant
offset rather than a discriminator. ICA dictionaries keep the cosine weight
because ICA components are not necessarily orthogonal.

## 2. Content / context split

Let

$$
A_C = \{\mathrm{cos}, \mathrm{lpips}, \mathrm{ssim\_distance}\}
$$

be the content axes and

$$
A_X = \{\mathrm{jsd}, \mathrm{cka}, \mathrm{topk\_overlap},
\mathrm{map\_iou}, \mathrm{parent\_jaccard}\}
$$

be the context axes.
The choice is encoded by `ContentContextSplit` in `config.py`.

Define the *content distance* and the *context distance* of a pair as
weight-normalized means of the present axes:

$$
d_{\mathrm{content}}(i,j) =
\frac{\sum_{a \in A_C \cap P} w_a \, \mathrm{raw}_a(i,j)}
     {\sum_{a \in A_C \cap P} w_a}
$$

$$
d_{\mathrm{context}}(i,j) =
\frac{\sum_{a \in A_X \cap P} w_a \, \mathrm{raw}_a(i,j)}
     {\sum_{a \in A_X \cap P} w_a}
$$

where $P$ is the set of axes whose underlying score is actually
available for this pair, $\mathrm{raw}_a$ is the distance form from the
table above, and $w_a$ is the per-axis weight from `axis_weights`. Missing
axes are *dropped*, not zero-imputed, matching the convention in
`metrics.composite_distance`.

Both `d_content` and `d_context` are non-negative weighted means. They
usually fall near $[0, 1]$ for the current operating point, but they are
not mathematically capped at $1$.

The two quantities are intended as **separate diagnostic proxies**:

- `d_content(i, j)` — *do the feature direction and optimized
  visualization disagree?* This is a content-like proxy for what the
  feature responds to intrinsically. It is still not a proof of
  natural-language semantic difference.
- `d_context(i, j)` — *do they fire on different inputs / regions?* The
  histogram, CKA, top-k overlap, spatial IoU, and parent overlap all
  depend on the corpus and on lower-level structure shared between
  features.

## 3. The four-way classification

Fix thresholds $\varepsilon_C = \texttt{eps\_content}$ and
$\varepsilon_X = \texttt{eps\_context}$. Each pair is classified by the
quadrant of $(d_{\mathrm{content}}, d_{\mathrm{context}})$ it falls in:

| $d_{\mathrm{content}}$ | $d_{\mathrm{context}}$ | label        | meaning                                           |
|:------------|:------------|:-------------|:--------------------------------------------------|
| $> \varepsilon_C$     | $> \varepsilon_X$     | `distinct`   | content-different and different inputs            |
| $> \varepsilon_C$     | $\le \varepsilon_X$   | `siblings`   | content-different on shared inputs/parents        |
| $\le \varepsilon_C$   | $\le \varepsilon_X$   | `redundant`  | content-similar, same context (decomposition copy)|
| $\le \varepsilon_C$   | $> \varepsilon_X$     | `echo`       | content-similar on disjoint inputs (often noise)  |

**Definition ($i \ne j$).** Features $f_i$ and $f_j$ are *different* iff

$$
d_{\mathrm{content}}(i,j) > \varepsilon_C.
$$

Equivalently, the pair label is `distinct` or
`siblings`.

This is the hierarchy-aware predicate: shared lower-level dependencies
push `d_context` *down* while the content-like axes still decide whether
the pair is different. A "car" feature and a "bicycle" feature can
therefore count as different even when they share top-k images and
parent features (they would be labeled `siblings`). Conversely, nearly
duplicate directions with similar visualizations and activation
histograms land in `redundant` and are *not* called different. The
hierarchy is not ignored; it is exposed as the second coordinate of the
classification. The current metric is sign-sensitive: treating
opposite-sign directions as the same feature would require an explicit
$|\cos|$ or subspace-distance variant.

The legacy `epsilon_different` gate is the older
joint activation/visual predicate. It is retained for backwards
compatibility and reported alongside the new label rather than being
treated as the formal hierarchy-aware definition.

## 4. Per-pair empirical significance

The package already ships a perturbation control
(`controls.perturbed_pair_control`) that, for every
record, produces a noise-perturbed copy of its feature vector and
histogram. The distribution of `pair_distance(record, perturbed_copy)`
is a sample of distances under the null hypothesis

> $H_0(i,j)$: features $f_i$ and $f_j$ are the same feature, up to noise.

Because the perturbation only varies the feature vector and the
activation histogram (top-k indices, spatial maps, and visualization
path are left intact), the eligible activation axes are

$$
A_{\mathrm{act}} = (A_C \cup A_X) \cap \{\mathrm{cos}, \mathrm{jsd}\}.
$$

Restricting both the null and the observed distance to `A_act` keeps
the comparison matched on which axes are summed. For each pair compute

$$
d_{\mathrm{act}}(i,j) =
\frac{\sum_{a \in A_{\mathrm{act}}} w_a \, \mathrm{raw}_a(i,j)}
     {\sum_{a \in A_{\mathrm{act}}} w_a}.
$$

For a null sample of size $N$ drawn by perturbing every record
`n_resample` times, the right-tail empirical p-value with the standard
$+1/+1$ smoothing (so that no observed value gets $p = 0$) is

$$
p(i,j) =
\frac{1 + \#\{x \in \mathrm{null}: x \ge d_{\mathrm{act}}(i,j)\}}
     {1 + N}.
$$

A pair is *significantly different* at level $\alpha$ (default
$\alpha = 0.05$) iff

$$
p(i,j) < \alpha.
$$

See `differentiation.pairwise_pvalues`.

The implementation also reports

$$
z_{\mathrm{null}}(i,j) =
\frac{d_{\mathrm{act}}(i,j) - \mu_{\mathrm{null}}}
     {\sigma_{\mathrm{null}}}.
$$

This effect size is a standardized rescale of the observed activation-axis
distance against the shared perturbation null. It helps interpret p-values
that hit the empirical floor $1/(1+N)$, but it is not a separate statistical
test.

**Caveat.** The perturbation null does not re-optimize the activation
maximization visualization, so the LPIPS / SSIM / map-IoU / top-k axes
are excluded from the p-value. Folding them into significance would
require a separate, more expensive null. Visual-axis disagreement is
reported via the `d_content` / `d_context` split, not via `p(i, j)`.

## 5. The visual dictionary

Define the undirected graph $G = (V, E)$ on records with

$$
E = \{(i,j): d_{\mathrm{content}}(i,j) \le \varepsilon_C\}.
$$

That is, there is an edge whenever
two features are *not* declared different. Connected components of $G$
partition the features into *equivalence classes*. For a class `C` we
pick the *representative* (medoid) as

$$
\mathrm{rep}(C) =
\arg\min_{i \in C}
\sum_{\substack{j \in C \\ j \ne i}}
d_{\mathrm{content}}(i,j).
$$

with singleton classes trivially their own representative. The visual
dictionary is the ordered list of representatives together with the
within-class spread (`max` and 95th percentile of within-class
`d_content`). The dictionary's *compression ratio*,
$|\mathrm{classes}| / |\mathrm{features}|$, is a corpus-level summary
of how much redundancy the decomposition introduced; it should be $\le 1$
and is reported by
`scripts/validate_dictionary.py`.

## 6. Success criterion (no human in the loop)

Two complementary, label-free measurements:

1. **Signal-to-noise ratio**
   $\mathrm{SNR} = \mathrm{real.mean} / \mathrm{perturbed.mean}$ from
   `controls.signal_to_noise`. $\mathrm{SNR} \gg 1$
   indicates the operating point of the
   composite distance separates real features from their noise copies.
2. **Hierarchy-aware dictionary compression.** `validate_dictionary.py`
   reports the number of equivalence classes induced by
   $d_{\mathrm{content}}(i,j) \le \varepsilon_C$, the medoid representative
   for each class, and the within-class spread. A useful operating point
   should separate real pairs from perturbations while avoiding excessive
   duplicate classes. Compression is only a substantive claim when `K` is
   overcomplete relative to the effective feature rank; singleton classes
   are reported as no compression observed.

## 7. Limitations

- The thresholds $\varepsilon_C$, $\varepsilon_X$, and $\alpha$ are operating-point choices. They
  should be calibrated against controls for each layer/architecture;
  they are not derived from first principles.
- The perturbation null is anchored to the activation axes only. A
  visual-axis null would require re-running activation maximization on
  perturbed directions, which is `n_resample · K · viz_steps` Adam
  iterations — out of scope here.
- The `parent_jaccard` term assumes top-k overlap is a reasonable proxy
  for shared lower-level structure. A direct subspace residualization
  in activation space requires aligning feature directions across
  layers, which the current pipeline does not provide.
