# Determination Funnel: Self-Reflection as the Dual Cut of Abstraction and Reverse-Diffusion Expression

![3-Stage Dual-Cut Pipeline on Hard MNIST Digits](assets/diffusion_reflection_grid.png)

![Multi-Step Closed-Loop Occlusion Restoration on MNIST](assets/occlusion_refinement_grid.png)

**Determination Funnel** is a pure-NumPy reference implementation and benchmark suite grounded in [The Dual Cut](https://not-a-toe.com/posts/the-dual-cut/). It demonstrates that **self-attention is a special case of self-reflection**—extracting a low-dimensional abstraction from an input and applying that abstraction back onto the input's own high-dimensional payload:
- **Stage 1 (Gradual Supervised Recognition Discovery `E`, Starting from `2,352 → 10`)**: Starting directly from `2,352 → 10` on `[0, 255]³` `(x, y, I)` coordinate tuples with **zero preset power-of-two widths and zero artificial width floors (`min_width = 1`)**, splitting only the remaining gap `d*_{l-1} → 10` via `D_init(l) = round(sqrt(d*_{l-1} / 10) * 10)` and elastic breathing discovers the exact 3-cut funnel **`2,352 → 105 → 32 → 11 → 10`** (`250,940` parameters—`60%` smaller than the legacy `256 → 64 → 32 → 10` network, reaching **`97.67%` accuracy** and **`0.0191` off-diagonal cosine**), where the **`11` (`10` digit cones + `1` shared bridge/uncertainty axis)** derives itself automatically and terminates layer discovery at the mathematical fixed point `round(sqrt(11 / 10) * 10) = 10`. `E` is then **frozen permanently**.
- **Stage 2 (Reverse-Diffusion Generator `G` Trained by Iterative Redraw Against Frozen `E`)**: Because expression (`G: Z → X̂`) requires expanding detail across the large gap (`105 → 784`), inserting the midpoint layer `287 = round(sqrt(784 / 105) * 105)` creates a uniform `~2.8×` reverse-diffusion cascade **`[10 + 11] → 32 → 105 → 287 → 784`**. `G` is trained on clean numbers by feeding **frozen `E`'s actual recognition vectors** `Z^{(k)} = E(X^{(k)})` into `G` to redraw `X^{(k+1)}`, re-recognizing `X^{(k+1)}` with frozen `E`, and measuring supervised loss against the desired label `y`.
- **Stage 3 (Test-Time Diffusion Gated by Internal Loss Until Recognition Vectors Converge)**: At test time (with both `E` and `G` frozen, zero retraining), gating by `E`'s intrinsic **Internal Loss (`L_internal`)** exits immediately at Step 0 on the low-loss `95%` of natural MNIST (**`99.13%` accuracy**, `mean = 0.03 steps`), while iterating the reverse-diffusion redraw loop on high-loss/uncertain digits and unseen occlusions in an **unbounded `while` loop** until `L_internal` stops dropping and the recognized vectors converge (`‖z^{(k+1)} - z^{(k)}‖_2 ≤ ε`), cutting internal loss by `36%–40%` and lifting accuracy across all three occlusion regimes (`+1.58%` on Picket Fence, `+3.29%` on Center Blackout, `+1.36%` on Heavy Clutter).

## 1. Documented 3-Stage Implementation Plan & Architectural Thesis

Full architectural specification: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

1. **Why Fixed-Task Perception Needs Zero `O(N²)` Token Polling**:
   - Under [The Dual Cut](https://not-a-toe.com/posts/the-dual-cut/), every token in an open-ended sequence carries both substantive payload (`V`) and operational instruction (`K`). In fixed-task visual recognition, however, the top-level command—*"recognize this picture as one of the 10 digits `0–9`"*—is **implied by the problem statement itself** (`d_out = 10`) and compiled into the network.
   - Running `O(N²)` self-attention across 784 raw pixels (`X ↔ X`) is an unconsolidated way to simulate a shared abstraction when the spatial coordinates `(x, y)` were stripped from pixel intensities `I`. Placing every pixel inside the canonical **`[0, 255]³` coordinate-value universe** `(xᵢ · Iᵢ/255, yᵢ · Iᵢ/255, Iᵢ)` allows a steep `Linear + ReLU` funnel to extract the low-dimensional abstraction directly.
2. **Why Abstraction (`E`) and Expression (`G`) Are Asymmetric**:
   - **Abstraction (`E: 2,352 → 105 → 32 → 11 → 10`)** is many-to-one lossy compression: `ReLU` cuts zero out `68.7%–84.3%` of sub-threshold fluctuations at each layer, isolating `10` orthogonal digit cones (`Off-Diag Cos = 0.0191`) plus a spontaneously derived `11th` bridge/indeterminacy axis ([eleventh_dimension_inspection.png](assets/eleventh_dimension_inspection.png)). Once learned, `E` stays invariant.
   - **Expression (`G: [10 + 11] → 32 → 105 → 287 → 784`)** is one-to-many detail synthesis: it bridges the `105 → 784` chasm via the midpoint layer `287 = round(sqrt(784 / 105) * 105)` and acts as an intermediate diffusion denoiser. Uncertain or occluded inputs enter as **middle steps `X_t` of a diffusion trajectory**—redrawn from their current recognition vectors `Z^{(k)} = E(X^{(k)})` and re-tested by frozen `E` until `L_internal` bottoms out and the recognized vectors converge (`E(G(z*)) = z*`).

## 2. Experimental Results (`seed = 42`, Pure NumPy on CPU)

### A. Stage 1 — Gradual Layer & Width Discovery Starting from `2,352 → 10` (`run_elastic_funnel.py`)

Starting directly from `2,352 → 10` (`L = 0`) with `min_width = 1` (zero artificial floor), each refinement step `l` splits the remaining gap between the **actual discovered width `d*_{l-1}`** and `10` at $D_{\text{init}}(l) = \text{round}(\sqrt{d^*_{l-1} / 10} \cdot 10)$ and breathes elastically via collinear cone folding (`cos ≥ 0.90`) and spent-scaffolding pruning (`< 5%` mean layer energy):

| Refinement Step (`L`) | Midpoint Init (from Discovered `d*_{l-1}`) | Discovered Architecture (`[2352, …, 10]`) | Parameters (`%` Shed vs. Init) | Test Accuracy (`Head / Proto`) | Mean Internal Loss (`L_internal`) | Latent Orthogonality (`Off-Diag Cos`) | Next Midpoint $\text{round}(\sqrt{d^*_l/10}\cdot 10)$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`L = 0` (Direct Cut)** | `[2352, 10]` | **`[2352, 10]`** | `23,530` (`0.0%`) | `92.59% / 92.59%` | `0.1492` | `0.2231` | `153` |
| **`L = 1` (1st Split)** | `[2352, 153, 10]` | **`[2352, 121, 10]`** | `285,933` (`-20.9%`) | `97.56% / 92.81%` | `0.1402` | `0.1433` | `35` |
| **`L = 2` (2nd Split)** | `[2352, 153, 35, 10]` | **`[2352, 121, 12, 10]`** | `286,307` (`-21.7%`) | `97.41% / 97.34%` | `0.0747` | `0.0253` | `11` |
| **`L = 3` (Fixed Point)** | `[2352, 153, 35, 11, 10]` | **`[2352, 105, 32, 11, 10]`** | **`250,940` (`-31.4%`)** | **`97.67% / 97.58%`** | **`0.0595`** | **`0.0191`** | **`10` (Converged!)** |

- **Why `11` Emerges Automatically and Terminates Depth Discovery**: At `L = 2`, the second hidden cut collapses from `35` to `12`, whose midpoint to `10` is `round(sqrt(12 / 10) * 10) = 11`. At `L = 3`, that `11`-neuron cut survives with zero pruning even under `min_width = 1` (`10` pure digit cones + `1` shared upper-loop bridge cone across digits `3, 9, 8`, visualized in [eleventh_dimension_inspection.png](assets/eleventh_dimension_inspection.png)), while its next midpoint to `10` is `round(sqrt(11 / 10) * 10) = 10`—reaching the exact fixed point of the refinement rule.

### B. Stage 2 & Stage 3 — Internal-Loss Gated Reverse-Diffusion Reflection (`run_diffusion_reflection.py`)

With `E` (`[2352, 105, 32, 11, 10]`) permanently frozen, `G` (`[10 + 11, 32, 105, 287, 784]`) is trained on clean numbers by iterative redraw-and-recognize (`X^{(k)} → Z^{(k)} = E(X^{(k)}) → X^{(k+1)} = Redraw(X^{(k)}, G(Z^{(k)})) → E(X^{(k+1)})`) and evaluated at test time in an **unbounded self-terminating loop** that halts for each image as soon as `L_internal` stops dropping or the recognized vectors converge (`‖z^{(k+1)} - z^{(k)}‖_2 ≤ 5×10⁻⁴`):

| Test Regime (`10,000` Unseen Test Set, Frozen `E` & `G`) | Step 0 Accuracy (`E(X^{(0)})`) | Converged Accuracy (`X^*`) | Loop Gain (`Δ Acc`) | Internal Loss (`Step 0 → Converged`) | Steps Taken (`Mean / Med / p90 / Max`) | Canvas Pixel MSE Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Natural MNIST — Low-Loss 95% (`L_int < 0.14`)** | `99.13%` | **`99.13%`** | `+0.00%` (Preserved) | `0.042 → 0.042` | **`0.03` / `0` / `0` / `2`** | Untouched (`O(1)` exit) |
| **Natural MNIST — High-Loss 5% (`L_int ≥ 0.14`)** | `63.80%` | **`64.60%`** | **`+0.80%`** (`14` rescued) | **`0.341 → 0.223`** (`-34.6%`) | **`10.3` / `8` / `24` / `89`** | Clarified strokes ([grid](assets/diffusion_reflection_grid.png)) |
| **Natural MNIST — Full `10,000` Test Set** | `97.36%` | **`97.40%`** | `+0.04%` | `0.057 → 0.051` | `0.54` / `0` / `1` / `89` | Selective redraw on hard tail |
| **Picket Fence (`Zero-Shot Unbounded Loop`)** | `74.11%` | **`75.69%`** | **`+1.58%`** | **`0.239 → 0.153`** (`-36.0%`) | **`3.7` / `0` / `11` / `225`** | **`-15.5%`** (`0.0774 → 0.0654`) |
| **Center Blackout (`Zero-Shot Unbounded Loop`)** | `55.91%` | **`59.20%`** | **`+3.29%`** | **`0.261 → 0.160`** (`-38.7%`) | **`4.8` / `0` / `16` / `174`** | **`-7.0%`** (`0.0513 → 0.0477`) |
| **Heavy Clutter (`Zero-Shot Unbounded Loop`)** | `83.90%` | **`85.26%`** | **`+1.36%`** | **`0.205 → 0.124`** (`-39.5%`) | **`3.4` / `0` / `10` / `243`** | **`-16.5%`** (`0.1221 → 0.1019`) |

### C. Reference Benchmark Comparisons (`run_experiment.py` & `run_occlusion_loop.py`)

Head-to-head comparison across sample regimes (`1-shot` to `60,000` full MNIST) and 4-step corruption-matched occlusion benchmarks:

| Regime / Metric | Determination Funnel (`(x, y, I)` + Latent Group Loss) | Standard Cross-Entropy MLP (`I`-Only, Output CE Loss) | Standard Pixel-Loss Autoencoder (Surface Loss `‖X̂ − X‖²`) | Latent Orthogonality (`Off-Diag Cos`: Funnel vs. Std MLP) | `ReLU` Cutoff (`%` Zeroed Out: Funnel vs. Std MLP) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1-Shot (10 samples total)** | **43.39%** (Fwd) / **42.08%** (Loop) | 39.19% | 30.05% (`Cos: 0.367`) | **`0.014` vs. `0.634`** (`45×` more orthogonal) | **`78.8%` vs. `44.0%`** |
| **5-Shot (50 samples)** | **68.20%** | 65.23% | 44.54% (`Cos: 0.870`) | **`0.017` vs. `0.694`** (`41×` more orthogonal) | **`80.8%` vs. `23.5%`** |
| **10-Shot (100 samples)** | **74.92%** | 71.70% | 60.21% (`Cos: 0.757`) | **`0.017` vs. `0.736`** (`43×` more orthogonal) | **`76.2%` vs. `27.7%`** |
| **50-Shot (500 samples)** | **88.52%** | 88.84% | 74.24% (`Cos: 0.894`) | **`0.011` vs. `0.683`** (`62×` more orthogonal) | **`81.9%` vs. `36.6%`** |
| **Full MNIST (60,000)** | **97.93%** (`256→64→32`) / **97.67%** (`105→32→11`) | 97.71% | **74.68%** (`Cos: 0.935`) | **`0.019` vs. `0.551`** (`29×` more orthogonal) | **`84.3%` vs. `40.6%`** |

- **Zero-Label Unsupervised 10-Group Discovery (`Pathway 2B`)**: Training the forward pass with zero labels to minimize conditional group entropy `H(group | x)`, maximize marginal group entropy `H(group)`, and align local `(x, y, I)` neighbors activates **all `10 / 10` groups** and reaches **`53.91%` unsupervised cluster purity** (`5.4×` random chance).
- **Zero-Shot Multi-Resolution Generalization**: Evaluating `28×28`-trained weights on test digits downsampled to `14×14` or upsampled to `56×56` in the `[0, 255]³` coordinate universe achieves **`96.26%` (`14×14`)**, **`97.77%` (`28×28`)**, and **`97.75%` (`56×56`)**.

## 3. Quick Start (Pure NumPy + Pillow on CPU)

```bash
pip install numpy pillow
python3 run_elastic_funnel.py         # Stage 1: Gradual layer & width discovery from 2352 -> 10
python3 run_diffusion_reflection.py   # Stages 1-3: Frozen E + Reverse-Diffusion G + Internal-Loss Convergence Loop
python3 run_experiment.py             # Reference few-shot, unsupervised, and multi-resolution benchmarks
python3 run_occlusion_loop.py         # Reference 4-step corruption-matched occlusion benchmark
python3 render_tables.py              # Render Retina PNG benchmark tables in ./assets/
```

## License

MIT License
