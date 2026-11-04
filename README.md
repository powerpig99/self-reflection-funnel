# Determination Funnel: Replacing Convolutions and Self-Attention with `[0, 255]³` Coordinate Tuples, Gradual `ReLU` Cuts, and Closed-Loop Re-Observation

![Multi-Step Closed-Loop Occlusion Restoration on MNIST](assets/occlusion_refinement_grid.png)

**Determination Funnel** is a pure-NumPy reference implementation and benchmark suite demonstrating that a simple four-stage **`Linear + ReLU` compression funnel (`2,352 → 256 → 64 → 32 → 10`)**—with **zero convolutions, zero `O(N²)` self-attention, and zero causal masking**—matches or outperforms standard feedforward networks on MNIST (`97.93%` in `4.54s` on CPU), achieves **`30× to 60×` cleaner latent orthogonality**, discovers **`10 / 10` digit clusters with zero labels (`53.91%` unsupervised purity)**, generalizes zero-shot across resolutions (`14×14`, `28×28`, `56×56`), and restores severely occluded digits across iterative closed-loop passes (`+34.13%` to `+36.08%` over standard open-loop MLPs).

---

## 1. Core Architectural Thesis

Modern deep learning relies on two heavy structural crutches to compensate for two foundational confusions:
1. **Why Transformers Needed `O(N²)` Self-Attention (`X ↔ X`)**: A standard Transformer keeps `N` surface token vectors at every layer from input to output. Because it refuses to separate the **low-dimensional latent representation (`z*`, the collective identity)** from the **high-dimensional surface expression (`X`, the parts)**, every part must query every other part (`Q · Kᵀ`) horizontally to simulate a shared abstraction. Once representation (`z*`) is separated from expression (`X`)—compressing `X` upward into `z*`, re-expressing `z*` downward into `X̂ = G(z*)`, and feeding `X̂` back into the input encoder `E(X̂)` so that `E(X̂) = z*`—the self-referential relationship is established by the closed loop and compiled directly into the weights, making `O(N²)` self-attention redundant.
2. **Why CNNs Needed Rigid `k × k` Spatial Stencils**: Engineers stripped a pixel's coordinates `(x, y)` out of the datum and hid them in the 2D array index `I[x, y]`, forcing networks to slide a pre-frozen `3 × 3` stencil uniformly across empty background and occluded foreground alike. In biological vision, `~120M` photoreceptors compress into `~1M` optic nerve fibers right at Layer 1 because a single pixel has no independent meaning on its own.
3. **The `[0, 255]³` Coordinate-Value Universe & `ReLU` as Determination**:
   - Every pixel `i` already carries both its position and its value: `(xᵢ, yᵢ, Iᵢ)` for grayscale (`3D`), `(x, y, R, G, B)` for color (`5D/6D`), and `(x, y, t, R, G, B, …)` for video (`7D`).
   - Because a picture is the entire observational universe for that picture, scaling `x` and `y` to the full 8-bit range `[0, 255]` alongside the 8-bit intensity `I ∈ [0, 255]` (with empty background `I = 0` carrying zero spatial offset `(x · I/255, y · I/255, I) = (0, 0, 0)`) places every image of any resolution inside the same **`[0, 255]³` state space**.
   - Summing multi-dimensional tuples across the field (`u = Wx + b`) expands candidate linear combinations; applying **`ReLU` (`z = max(0, u)`)** slices off `71.6%–83.9%` of insignificant sub-threshold fluctuations to `0`. Because `z ≥ 0` lives in the non-negative orthant, **two representations are orthogonal (`z_a · z_b = 0`) if and only if they activate completely disjoint neurons**.

---

## 2. Key Experimental Results (`seed = 42`, Pure NumPy on CPU)

### A. Head-to-Head Comparison Against Traditional Standard Implementations

All three models share the exact same internal layer widths (`256 → 64 → 32 (z*) → 10`) and are evaluated on the official `10,000`-image MNIST test set:

| Regime / Metric | Determination Funnel (`(x, y, I)` + Latent Group Loss) | Standard Cross-Entropy MLP (`I`-Only, Output CE Loss) | Standard Pixel-Loss Autoencoder (Loss at Expression `‖X̂ − X‖²`) | Latent Orthogonality (`Off-Diag Cos` in `32d` `z*`: Funnel vs. Std MLP) | `32d` `z*` `ReLU` Cutoff (`%` Zeroed Out: Funnel vs. Std MLP) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1-Shot (10 samples total)** | **44.54%** (Pathway 1) / **43.39%** (Pathway 2) | 39.19% | 26.61% (`Cos: 0.382`) | **`0.014` vs. `0.634`** (`45×` more orthogonal) | **`78.8%` vs. `44.0%`** |
| **5-Shot (50 samples)** | **68.20%** | 65.23% | 23.02% (`Cos: 0.669`) | **`0.017` vs. `0.694`** (`41×` more orthogonal) | **`80.8%` vs. `23.5%`** |
| **10-Shot (100 samples)** | **74.92%** | 71.70% | 27.81% (`Cos: 0.877`) | **`0.017` vs. `0.736`** (`43×` more orthogonal) | **`76.2%` vs. `27.7%`** |
| **50-Shot (500 samples)** | **88.52%** | 88.84% | **20.31%** (`Cos: 0.956`) | **`0.011` vs. `0.683`** (`62×` more orthogonal) | **`81.9%` vs. `36.6%`** |
| **Full MNIST (60,000)** | **97.93%** (`4.54s` CPU) | 97.71% | — | **`0.019` vs. `0.551`** (`29×` more orthogonal) | **`83.9%` vs. `40.6%`** (`71.6%` at L1) |

- **Why Surface Expression Loss (`‖X̂ − X‖²`) Collapses**: As sample count grows from `1-shot` to `50-shot`, the Standard Pixel-Loss Autoencoder's `32d` bottleneck is forced to memorize surface stroke variations (slant, thickness, brightness), driving its 10 digit prototypes into a single direction (`Off-Diag Cos: 0.382 → 0.956`) and collapsing accuracy to **`20.31%`** (vs. **`88.52%`** when loss is measured at the abstraction `z*`).
- **Zero-Label Unsupervised 10-Group Discovery (`Pathway 2B`)**: With **zero labels**, training the forward pass solely to minimize conditional group entropy `H(group | x)`, maximize marginal group entropy `H(group)`, and align local `(x, y, I)` neighbors activates **all `10 / 10` groups** (`[837, 1143, 809, 951, 787, 1142, 1154, 962, 1081, 1134]` across the 10,000 test digits) and reaches **`53.91%` unsupervised cluster purity** (`5.4×` random chance) in `7.25s`.
- **Zero-Shot Multi-Resolution Generalization**: Evaluating the `28×28`-trained weights on test digits downsampled to `14×14` or upsampled to `56×56` and mapped onto the canonical `[0, 255]³` universe achieves **`96.26%` (`14×14`)**, **`97.77%` (`28×28`)**, and **`97.75%` (`56×56`)**.

### B. Multi-Step Closed-Loop Occlusion Restoration (`X_occ → z₀ → X̂₁ → E(X̂₁) = z₁ → …`)

Tested across all `10,000` unseen MNIST test digits under three severe corruptions:
1. **Picket Fence**: 3 horizontal bands erased to `0` + 2 bright vertical fence bars (`I = 220`) drawn across the digit.
2. **Center Blackout**: A `10 × 12` box wiping out the central intersection of the digit.
3. **Heavy Clutter**: `45%` random stroke dropout + `15%` bright static spikes across the `[0, 255]³` field.

| Corruption Mode | Standard Open-Loop MLP Acc | Loop Step 0 (`z₀ = E(X_occ)`) | Loop Step 1 (`E(G(z₀))`) | Loop Step 2 (`E(G(z₁))`) | Loop Step 3 (`E(G(z₂))`) | Loop Step 4 (`E(G(z₃))`) | Latent Cosine `cos(zₖ, z*_clean)` (`Step 0 → 4`) | Canvas Pixel MSE to Clean Digit (`Step 0 → 3`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Picket Fence** | 44.33% | 77.02% | 79.94% | 80.30% | **80.41%** | 80.37% | `0.8232 → 0.8478` | `0.0774 → 0.0294` (**`-62%` error**) |
| **Center Blackout** | 40.34% | 61.92% | 71.79% | 73.44% | 74.04% | **74.47%** | `0.6639 → 0.7722` | `0.0513 → 0.0284` (**`-45%` error**) |
| **Heavy Clutter** | 54.77% | 83.68% | 86.84% | 87.18% | 87.30% | **87.38%** | `0.8692 → 0.9055` | `0.1221 → 0.0309` (**`-75%` error**) |

---

## 3. Quick Start (Pure NumPy + Pillow, ~20 Seconds on CPU)

```bash
pip install numpy pillow
python3 run_experiment.py
python3 run_occlusion_loop.py
```

Both scripts automatically download the official MNIST IDX files into `./data/` if not already present, run all benchmarks with analytical gradients in pure NumPy, and write JSON metrics to `./results/` and the visual restoration grid to `./assets/occlusion_refinement_grid.png`.

## License

MIT License
