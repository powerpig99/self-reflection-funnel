# Determination Funnel: Self-Reflection as the Dual Cut of Abstraction and Reverse-Diffusion Expression

![Dual-Cut Pipeline on Hard MNIST Digits](assets/diffusion_reflection_grid.png)

![Multi-Step Closed-Loop Occlusion Restoration on MNIST](assets/occlusion_refinement_grid.png)

**Determination Funnel** is a pure-NumPy reference implementation and benchmark suite grounded in [The Dual Cut](https://not-a-toe.com/posts/the-dual-cut/). It proposes that **self-attention is a special case of self-reflection**—extracting a low-dimensional abstraction from an input and applying that abstraction back onto the input's own high-dimensional payload:
- **Stage 1 (Gradual Supervised Recognition Discovery `E`, Starting from `2,352 → 10`)**: Starting directly from `2,352 → 10` on `[0, 255]³` `(x, y, I)` coordinate tuples with **zero preset power-of-two widths and zero artificial width floors (`min_width = 1`)**, splitting only the remaining gap `d*_{l-1} → 10` via $D_{\text{init}}(l) = \text{round}(\sqrt{d^*_{l-1} / 10} \cdot 10)$ and elastic breathing discovers the 3-cut funnel **`2,352 → 105 → 32 → 11 → 10`** (`250,940` parameters—`59.6%` smaller than the `621,226`-parameter `2,352 → 256 → 64 → 32 → 10` funnel, reaching **`97.67%` head accuracy**, **`97.58%` prototype accuracy**, and **`0.0191` off-diagonal cosine**). At `L = 2`, the second cut contracts from `35 → 12`; at `L = 3`, the third cut is initialized by the midpoint rule at $\text{round}(\sqrt{12 / 10} \cdot 10) = 11$, expands to `13`, and returns to `11`, confirming the midpoint estimate and terminating layer discovery at the fixed point $\text{round}(\sqrt{11 / 10} \cdot 10) = 10$. `E` is then **frozen** for Stage 2.
- **Stage 2 (Reverse-Diffusion Generator `G` Trained by Iterative Redraw Against Frozen `E`)**: Because expression (`G: Z → X̂`) requires expanding detail across the large gap (`105 → 784`), inserting the midpoint layer $287 = \text{round}(\sqrt{784 / 105} \cdot 105)$ creates a uniform `~2.8×` reverse-diffusion cascade **`[10 + 11] → 32 → 105 → 287 → 784`**. `G` is trained on clean numbers by feeding **frozen `E`'s actual recognition vectors** $Z^{(k)} = E(X^{(k)})$ into `G` to redraw $X^{(k+1)}$, re-recognizing $X^{(k+1)}$ with frozen `E`, and measuring supervised loss against the desired label `y`.
- **Stage 3 (Reality-Anchored Joint Supervised Relaxation of `E` and `G`)**: Relaxing both `E` and `G` together across the closed-loop trajectory while anchoring `E` to reality (so `E` reinforces redrawn $X^{(1)}$ only when $E(X^{(0)})$ matched the true label $y$, preventing the `-8.86%` Picket Fence collapse seen in the unconstrained ablation) lifts Natural MNIST accuracy in our primary **Stage 3A** configuration (judged by stationary `E_frozen`) to **`98.00%` (`Step 0`) / `97.94%` (`Converged`)**, **Picket Fence** from `74.23% →` **`75.84%`** (`+1.61%` loop gain), **Center Blackout** from `59.22% →` **`61.75%`** (`+2.53%` loop gain; `62.40%` / `+3.18%` when self-judged in Stage 3B), and **Heavy Clutter** from `87.07% →` **`87.72%`** (`+0.65%` loop gain). Evaluated against a **Same-Width Standard MLP (`784 → 105 → 32 → 11 → 10`)** trained on the exact same corruption stream for the same 22 epochs (`97.51%` Natural, `64.60%` Fence, `65.42%` Blackout, `83.30%` Clutter), Stage 3A leads by **`+0.49%` on Natural MNIST**, **`+11.24%` on Picket Fence**, and **`+4.42%` on Heavy Clutter**, while matching a `2.5×` larger `256 → 64 → 32` network (`98.03%` Natural) with **`59.6%` fewer parameters**.
- **Stage 4 (Test-Time Diffusion Gated by Internal Loss & The Human in the Loop)**: At test time (with zero retraining), gating by `E`'s intrinsic **Internal Loss (`L_internal`)** means `95%` of natural digits take almost no loop steps (`0.00` mean steps in 3B, `0.16` in 3A), while uncertain inputs iterate the mask-free redraw loop until `L_internal` stops dropping and the recognized vectors converge (`‖z^{(k+1)} - z^{(k)}‖_2 ≤ ε`). Persistent residual `L_internal` on genuine cross-class hybrids provides the exact signal for when to defer the final judgment call to the human in the loop.

## 1. Documented 4-Stage Implementation Plan & Architectural Thesis

Full architectural specification: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

1. **Why Fixed-Task Perception Needs Zero `O(N²)` Token Polling**:
   - Under [The Dual Cut](https://not-a-toe.com/posts/the-dual-cut/), every token in an open-ended sequence carries both substantive payload (`V`) and operational instruction (`K`). In fixed-task visual recognition, however, the top-level command—*"recognize this picture as one of the 10 digits `0–9`"*—is **implied by the problem statement itself** (`d_out = 10`) and compiled into the network.
   - Running `O(N²)` self-attention across 784 raw pixels (`X ↔ X`) is an unconsolidated way to simulate a shared abstraction when the spatial coordinates `(x, y)` were stripped from pixel intensities `I`. Placing every pixel inside the canonical **`[0, 255]³` coordinate-value universe** `(xᵢ · Iᵢ/255, yᵢ · Iᵢ/255, Iᵢ)` allows a steep `Linear + ReLU` funnel to extract the low-dimensional abstraction directly.
2. **Why Abstraction (`E`) and Expression (`G`) Are Asymmetric—and Why Co-Adaptation Needs a Reality Anchor**:
   - **Abstraction (`E: 2,352 → 105 → 32 → 11 → 10`)** is many-to-one lossy compression: `ReLU` cuts zero out `68.7%`, `61.8%`, and `84.3%` of sub-threshold fluctuations across the three hidden cuts, isolating `10` orthogonal digit cones (`Off-Diag Cos = 0.0191`) plus an `11th` shared upper-loop feature across curved digits ([eleventh_dimension_inspection.png](assets/eleventh_dimension_inspection.png)).
   - **Expression (`G: [10 + 11] → 32 → 105 → 287 → 784`)** is one-to-many detail synthesis: it bridges the `105 → 784` chasm via the midpoint layer `287 = round(sqrt(784 / 105) * 105)` and acts as an intermediate diffusion denoiser.
   - **Judge Drift & The Human in the Loop**: In our committed **Stage 3 Unconstrained** ablation (`run_diffusion_reflection.py`), training `E` unconditionally to predict `y` on every redrawn canvas $X^{(1)}$ causes `E` to bend its boundaries around `G`'s wrong-class hallucinations—dropping Picket Fence accuracy from `75.69%` (Stage 2) down to `66.83%` (`-8.86%`) and burning `15.78` steps on hard natural digits. Anchoring `E` so it only reinforces valid redraws eliminates Judge Drift, while residual `L_internal` on borderline hybrids signals when the outer human observer must make the final judgment call.

## 2. Experimental Results (`seed = 42`, Pure NumPy on CPU)

### A. Stage 1 — Gradual Layer & Width Discovery Starting from `2,352 → 10` (`run_elastic_funnel.py`)

Starting directly from `2,352 → 10` (`L = 0`) with `min_width = 1` (zero artificial floor) on held-out `train[:4000]` for width selection, each refinement step `l` splits the remaining gap between the **actual discovered width `d*_{l-1}`** and `10` at $D_{\text{init}}(l) = \text{round}(\sqrt{d^*_{l-1} / 10} \cdot 10)$ and breathes elastically via collinear cone folding (`cos ≥ 0.90`) and spent-scaffolding pruning (`< 5%` mean layer energy):

| Refinement Step (`L`) | Midpoint Init (from Discovered `d*_{l-1}`) | Discovered Architecture (`[2352, …, 10]`) | Parameters (`%` Shed vs. Init) | Test Accuracy (`Head / Proto`) | Mean Internal Loss (`L_internal`) | Latent Orthogonality (`Off-Diag Cos`) | Next Midpoint $\text{round}(\sqrt{d^*_l/10}\cdot 10)$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`L = 0` (Direct Cut)** | `[2352, 10]` | **`[2352, 10]`** | `23,530` (`0.0%`) | `92.59% / 92.59%` | `0.1492` | `0.2231` | `153` |
| **`L = 1` (1st Split)** | `[2352, 153, 10]` | **`[2352, 121, 10]`** | `285,933` (`-20.9%`) | `97.56% / 92.81%` | `0.1402` | `0.1433` | `35` |
| **`L = 2` (2nd Split)** | `[2352, 153, 35, 10]` | **`[2352, 121, 12, 10]`** | `286,307` (`-21.7%`) | `97.41% / 97.34%` | `0.0747` | `0.0253` | `11` |
| **`L = 3` (Fixed Point)** | `[2352, 153, 35, 11, 10]` | **`[2352, 105, 32, 11, 10]`** | **`250,940` (`-31.4%`)** | **`97.67% / 97.58%`** | **`0.0595`** | **`0.0191`** | **`10` (Converged!)** |

### B. Stages 2–4 — Same-Stream MLP Baselines vs. Frozen `E+G`, Unconstrained, and Reality-Anchored Relaxation (`run_diffusion_reflection.py`)

All values below come directly from `results/diffusion_reflection_results.json`. Both Standard MLP baselines are trained on the **exact same corruption stream (`forward_diffusion_on_clean`) for the same 22 epochs of exposure** (`12` clean epochs + `10` clean & diffusion epochs on `train[4000:]`):

| Model / Pipeline Stage (`10,000` Unseen Test Set) | Parameters (`E` / Total) | Natural MNIST (`All 10k`) | Natural MNIST (`Hardest 5% Tail`) | Picket Fence (`Step 0 → Converged`) | Center Blackout (`Step 0 → Converged`) | Heavy Clutter (`Step 0 → Converged`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Same-Width Standard MLP (`784 → 105 → 32 → 11 → 10`, 22 ep)** | `86,300` | `97.51%` | — | `64.60%` | **`65.42%`** | `83.30%` |
| **2. 2.5× Larger Standard MLP (`784 → 256 → 64 → 32 → 10`, 22 ep)** | `219,818` | **`98.03%`** | — | `71.90%` | **`66.38%`** | `85.74%` |
| **3. Stage 2 (`Frozen E + Trained G`, `[10,11,32,105,287,784]`)** | `250,940` (`E`) | `97.36% → 97.40%` | `63.80% → 64.60%` (`10.3` steps) | `74.11% → 75.69%` (`+1.58%`, `3.7` steps) | `55.91% → 59.20%` (`+3.29%`, `4.8` steps) | `83.90% → 85.26%` (`+1.36%`, `3.4` steps) |
| **4. Stage 3 Unconstrained (`Relaxed E + G`, No Reality Anchor)** | `250,940` (`E`) | `98.05% → 97.99%` | `71.40% → 70.20%` (`15.8` steps) | `65.18% → 66.83%` (`-8.86%` Drift!) | `58.92% → 64.07%` (`+5.15%`, `7.8` steps) | `84.35% → 86.21%` (`+1.86%`, `8.1` steps) |
| **5. Stage 3B Reality-Anchored (`Relaxed E + G`, Self-Judged)** | `250,940` (`E`) | **`98.00%` → `97.92%`** | **`71.20%` → `69.60%`** (`8.8` steps) | `74.23% → 75.78%` (`+1.55%`, `3.5` steps) | `59.22% →` **`62.40%`** (`+3.18%`, `4.6` steps) | `87.07% → 87.67%` (`+0.60%`, `2.9` steps) |
| **6. Stage 3A Reality-Anchored (`Relaxed E + G`, Judged by `E_frozen`)** | `250,940` (`E`) | **`98.00%` → `97.94%`** | **`71.20%` → `70.00%`** (`6.3` steps) | `74.23% →` **`75.84%`** (`+1.61%`, `3.4` steps) | `59.22% → 61.75%` (`+2.53%`, `4.2` steps) | `87.07% →` **`87.72%`** (`+0.65%`, `2.9` steps) |

*Note on Center Blackout*: Because `forward_diffusion_on_clean` applies pixel dropout, horizontal band dimming, vertical bars, and static—never a contiguous `10×12` center box—wiping out the center zeroes out coupled `(x·I, y·I, I)` coordinate vectors at Step 0 (`59.22%` vs. `65.42%` on the intensity-only MLP). Once the self-reflection loop runs, however, Center Blackout achieves the **largest single-regime loop recovery** (`+2.53%` in 3A, `+3.18%` in 3B, `+3.29%` in Stage 2).

### C. Reference Benchmark Comparisons (`run_experiment.py` & `run_occlusion_loop.py` — Used in Companion Videos)

The companion films in `./video/` present our original `2,352 → 256 → 64 → 32 → 10` (`621,226` parameters with biases; `620,864` weights-only) reference benchmarks from `run_experiment.py` (`results/results.json`, 12 clean epochs) and `run_occlusion_loop.py` (`results/occlusion_results.json`, 12 corruption-matched epochs with original-canvas stroke anchoring):

| Regime / Metric (`run_experiment.py`, 12 Clean Epochs) | Determination Funnel (`256→64→32`, `(x, y, I)` + Latent Group Loss) | Standard Cross-Entropy MLP (`256→64→32`, `I`-Only, Output CE) | Standard Pixel-Loss Autoencoder (Surface Loss `‖X̂ − X‖²`) | Latent Orthogonality (`Off-Diag Cos`: Funnel vs. Std MLP) | `z*` `ReLU` Cutoff (`%` Zeroed Out: Funnel vs. Std MLP) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1-Shot (10 samples total)** | **43.39%** (Fwd) / **42.08%** (Loop) | 39.19% | 30.05% (`Cos: 0.367`) | **`0.014` vs. `0.634`** (`45×` more orthogonal) | **`78.8%` vs. `44.0%`** |
| **5-Shot (50 samples)** | **68.20%** | 65.23% | 44.54% (`Cos: 0.870`) | **`0.017` vs. `0.694`** (`41×` more orthogonal) | **`80.8%` vs. `23.5%`** |
| **10-Shot (100 samples)** | **74.92%** | 71.70% | 60.21% (`Cos: 0.757`) | **`0.017` vs. `0.736`** (`43×` more orthogonal) | **`76.2%` vs. `27.7%`** |
| **50-Shot (500 samples)** | **88.52%** | 88.84% | 74.24% (`Cos: 0.894`) | **`0.011` vs. `0.683`** (`62×` more orthogonal) | **`81.9%` vs. `36.6%`** |
| **Full MNIST (60,000, 12 Clean Ep)** | **97.93%** (Head) / **97.77%** (Proto) | 97.71% | **74.68%** (`Cos: 0.935`) | **`0.019` vs. `0.551`** (`29×` more orthogonal) | **`83.9%` vs. `40.6%`** (`71.6%` at L1) |

And on the 4-step corruption-matched mask-free loop in `run_occlusion_loop.py` (`results/occlusion_results.json`):
- **Picket Fence**: Clean MLP `42.95%` | Same-Corruption MLP `80.44%` | Funnel Loop `79.45%` (Step 0) → **`82.09%`** (Step 4, `+2.64%` loop gain, `-58%` pixel MSE).
- **Center Blackout**: Clean MLP `40.15%` | Same-Corruption MLP `68.53%` | Funnel Loop `69.47%` (Step 0) → **`76.53%`** (Step 4, `+7.06%` loop gain, `-40%` pixel MSE).
- **Heavy Clutter**: Clean MLP `52.24%` | Same-Corruption MLP `84.93%` | Funnel Loop `84.61%` (Step 0) → **`85.65%`** (Step 2) / **`85.49%`** (Step 4, `-74%` pixel MSE).

## 3. Quick Start (Pure NumPy + Pillow on CPU)

```bash
pip install numpy pillow
python3 run_elastic_funnel.py         # Stage 1: Gradual layer & width discovery from 2352 -> 10
python3 run_diffusion_reflection.py   # Stages 1-4: Frozen E + G + Unconstrained/Anchored Relaxation + Same-Stream Baselines
python3 run_experiment.py             # Reference 256->64->32 few-shot, unsupervised, and multi-resolution benchmarks
python3 run_occlusion_loop.py         # Reference 256->64->32 4-step corruption-matched occlusion benchmark
python3 render_tables.py              # Render Retina PNG benchmark tables directly from ./results/*.json
```

## License

MIT License
