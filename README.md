# Determination Funnel: Self-Reflection as the Dual Cut of Abstraction and Reverse-Diffusion Expression

![Dual-Cut Pipeline on Hard MNIST Digits](assets/diffusion_reflection_grid.png)

![Multi-Step Closed-Loop Occlusion Restoration on MNIST](assets/occlusion_refinement_grid.png)

**Determination Funnel** is a pure-NumPy reference implementation and benchmark suite grounded in [The Dual Cut](https://not-a-toe.com/posts/the-dual-cut/). It demonstrates that **self-attention is a special case of self-reflection**—extracting a low-dimensional abstraction from an input and applying that abstraction back onto the input's own high-dimensional payload:
- **Stage 1 (Gradual Supervised Recognition Discovery `E`, Starting from `2,352 → 10`)**: Starting directly from `2,352 → 10` on `[0, 255]³` `(x, y, I)` coordinate tuples with **zero preset power-of-two widths and zero artificial width floors (`min_width = 1`)**, splitting only the remaining gap `d*_{l-1} → 10` via $D_{\text{init}}(l) = \text{round}(\sqrt{d^*_{l-1} / 10} \cdot 10)$ and elastic breathing discovers the exact 3-cut funnel **`2,352 → 105 → 32 → 11 → 10`** (`250,940` parameters—`60%` smaller than the legacy `256 → 64 → 32 → 10` network, reaching **`97.67%` accuracy** and **`0.0191` off-diagonal cosine**), where the **`11` (`10` digit cones + `1` shared bridge/uncertainty axis)** derives itself automatically and terminates layer discovery at the mathematical fixed point $\text{round}(\sqrt{11 / 10} \cdot 10) = 10$. `E` is then **frozen** for Stage 2.
- **Stage 2 (Reverse-Diffusion Generator `G` Trained by Iterative Redraw Against Frozen `E`)**: Because expression (`G: Z → X̂`) requires expanding detail across the large gap (`105 → 784`), inserting the midpoint layer $287 = \text{round}(\sqrt{784 / 105} \cdot 105)$ creates a uniform `~2.8×` reverse-diffusion cascade **`[10 + 11] → 32 → 105 → 287 → 784`**. `G` is trained on clean numbers by feeding **frozen `E`'s actual recognition vectors** $Z^{(k)} = E(X^{(k)})$ into `G` to redraw $X^{(k+1)}$, re-recognizing $X^{(k+1)}$ with frozen `E`, and measuring supervised loss against the desired label `y`.
- **Stage 3 (Reality-Anchored Joint Supervised Relaxation of `E` and `G`)**: Relaxing both `E` and `G` together across the closed-loop trajectory while anchoring `E` to reality (so `E` reinforces redrawn $X^{(1)}$ only when $E(X^{(0)})$ matched the true label $y$, preventing self-confirming "Judge Drift") lifts Natural MNIST accuracy to **`98.07%` (`98.00%` converged)**—surpassing even the `620,864`-parameter `256 → 64 → 32 → 10` network (`97.93%`) with **`60%` fewer parameters**—while lifting the hardest `5%` tail by **`+9.20%`** (`63.80% → 73.00%`), **Heavy Clutter** to **`87.57%`**, **Picket Fence** to **`75.02%`**, and **Center Blackout** to **`64.40%`**.
- **Stage 4 (Test-Time Diffusion Gated by Internal Loss & The Human in the Loop)**: At test time (with zero retraining), gating by `E`'s intrinsic **Internal Loss (`L_internal`)** exits immediately at Step 0 on clear digits (`O(1)` compute), while iterating the reverse-diffusion redraw loop on high-loss inputs in an **unbounded `while` loop** until `L_internal` stops dropping and the recognized vectors converge (`‖z^{(k+1)} - z^{(k)}‖_2 ≤ ε`). Persistent residual `L_internal` on genuine cross-class hybrids provides the exact signal for when to defer the final judgment call to the human in the loop.

## 1. Documented 4-Stage Implementation Plan & Architectural Thesis

Full architectural specification: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

1. **Why Fixed-Task Perception Needs Zero `O(N²)` Token Polling**:
   - Under [The Dual Cut](https://not-a-toe.com/posts/the-dual-cut/), every token in an open-ended sequence carries both substantive payload (`V`) and operational instruction (`K`). In fixed-task visual recognition, however, the top-level command—*"recognize this picture as one of the 10 digits `0–9`"*—is **implied by the problem statement itself** (`d_out = 10`) and compiled into the network.
   - Running `O(N²)` self-attention across 784 raw pixels (`X ↔ X`) is an unconsolidated way to simulate a shared abstraction when the spatial coordinates `(x, y)` were stripped from pixel intensities `I`. Placing every pixel inside the canonical **`[0, 255]³` coordinate-value universe** `(xᵢ · Iᵢ/255, yᵢ · Iᵢ/255, Iᵢ)` allows a steep `Linear + ReLU` funnel to extract the low-dimensional abstraction directly.
2. **Why Abstraction (`E`) and Expression (`G`) Are Asymmetric—and Why Co-Adaptation Needs a Reality Anchor**:
   - **Abstraction (`E: 2,352 → 105 → 32 → 11 → 10`)** is many-to-one lossy compression: `ReLU` cuts zero out `68.7%–84.3%` of sub-threshold fluctuations at each layer, isolating `10` orthogonal digit cones (`Off-Diag Cos = 0.0191`) plus a spontaneously derived `11th` bridge/indeterminacy axis ([eleventh_dimension_inspection.png](assets/eleventh_dimension_inspection.png)).
   - **Expression (`G: [10 + 11] → 32 → 105 → 287 → 784`)** is one-to-many detail synthesis: it bridges the `105 → 784` chasm via the midpoint layer `287 = round(sqrt(784 / 105) * 105)` and acts as an intermediate diffusion denoiser.
   - **Judge Drift & The Human in the Loop**: If `E` and `G` are jointly relaxed without constraint, `E` learns to lower its internal loss on `G`'s wrong-class hallucinations. Anchoring `E` so it only reinforces valid redraws prevents echo-chamber drift inside the machine, while residual `L_internal` on borderline hybrids signals when the outer human observer must make the final judgment call.

## 2. Experimental Results (`seed = 42`, Pure NumPy on CPU)

### A. Stage 1 — Gradual Layer & Width Discovery Starting from `2,352 → 10` (`run_elastic_funnel.py`)

Starting directly from `2,352 → 10` (`L = 0`) with `min_width = 1` (zero artificial floor), each refinement step `l` splits the remaining gap between the **actual discovered width `d*_{l-1}`** and `10` at $D_{\text{init}}(l) = \text{round}(\sqrt{d^*_{l-1} / 10} \cdot 10)$ and breathes elastically via collinear cone folding (`cos ≥ 0.90`) and spent-scaffolding pruning (`< 5%` mean layer energy):

| Refinement Step (`L`) | Midpoint Init (from Discovered `d*_{l-1}`) | Discovered Architecture (`[2352, …, 10]`) | Parameters (`%` Shed vs. Init) | Test Accuracy (`Head / Proto`) | Mean Internal Loss (`L_internal`) | Latent Orthogonality (`Off-Diag Cos`) | Next Midpoint $\text{round}(\sqrt{d^*_l/10}\cdot 10)$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`L = 0` (Direct Cut)** | `[2352, 10]` | **`[2352, 10]`** | `23,530` (`0.0%`) | `92.59% / 92.59%` | `0.1492` | `0.2231` | `153` |
| **`L = 1` (1st Split)** | `[2352, 153, 10]` | **`[2352, 121, 10]`** | `285,933` (`-20.9%`) | `97.56% / 92.81%` | `0.1402` | `0.1433` | `35` |
| **`L = 2` (2nd Split)** | `[2352, 153, 35, 10]` | **`[2352, 121, 12, 10]`** | `286,307` (`-21.7%`) | `97.41% / 97.34%` | `0.0747` | `0.0253` | `11` |
| **`L = 3` (Fixed Point)** | `[2352, 153, 35, 11, 10]` | **`[2352, 105, 32, 11, 10]`** | **`250,940` (`-31.4%`)** | **`97.67% / 97.58%`** | **`0.0595`** | **`0.0191`** | **`10` (Converged!)** |

### B. Stages 2–4 — Frozen `E+G` vs. Unconstrained & Reality-Anchored Joint Relaxation (`run_diffusion_reflection.py`)

Comparing (1) **Stage 2 (`Frozen E + Trained G`)**, (2) **Stage 3 Unconstrained (`Relaxed E + G`)**, (3) **Stage 3B Reality-Anchored (`Self-Judged`)**, and (4) **Stage 3A Reality-Anchored (`Judged by Stage-1 E_frozen`)** under the unbounded self-terminating loop (`‖z^{(k+1)} - z^{(k)}‖_2 ≤ 5×10⁻⁴`):

| Test Regime (`10,000` Unseen Test Set) | 1. Stage 2 (`Frozen E + G`) | 2. Stage 3 Unconstrained (`Relaxed E + G`) | 3. Stage 3B Reality-Anchored (`Self-Judged`) | 4. Stage 3A Reality-Anchored (`Judged by E_frozen`) |
| :--- | :---: | :---: | :---: | :---: |
| **Natural MNIST — All `10,000`** | `97.36% → 97.40%` | `97.85% → 97.71%` | **`98.07%` → `97.99%`** | **`98.07%` → `98.00%`** |
| **Natural MNIST — Hardest 5% Tail** | `63.80% → 64.60%` (`10.3` steps) | `68.40% → 65.60%` (`15.9` steps) | **`73.00%` → `71.40%`** (`9.7` steps) | **`73.00%` → `71.60%`** (`8.0` steps) |
| **Picket Fence (`Step 0 → Converged`)** | `74.11% →` **`75.69%`** (`+1.58%`, `3.7` steps) | `70.69% → 72.03%` (`+1.34%`, `6.1` steps) | `74.37% →` **`75.02%`** (`+0.65%`, `3.7` steps) | `74.37% → 74.85%` (`+0.48%`, `3.6` steps) |
| **Center Blackout (`Step 0 → Converged`)** | `55.91% → 59.20%` (`+3.29%`, `4.8` steps) | **`62.10% → 65.74%`** (`+3.64%`, `6.5` steps) | `61.31% →` **`64.40%`** (`+3.09%`, `4.6` steps) | `61.31% → 63.70%` (`+2.39%`, `4.4` steps) |
| **Heavy Clutter (`Step 0 → Converged`)** | `83.90% → 85.26%` (`+1.36%`, `3.4` steps) | `86.14% → 86.84%` (`+0.70%`, `5.3` steps) | `86.75% → 87.35%` (`+0.60%`, `2.8` steps) | **`86.75% → 87.57%`** (`+0.82%`, `2.9` steps) |

### C. Reference Benchmark Comparisons (`run_experiment.py` & `run_occlusion_loop.py`)

Head-to-head comparison across sample regimes (`1-shot` to `60,000` full MNIST):

| Regime / Metric | Determination Funnel (`(x, y, I)` + Latent Group Loss) | Standard Cross-Entropy MLP (`I`-Only, Output CE Loss) | Standard Pixel-Loss Autoencoder (Surface Loss `‖X̂ − X‖²`) | Latent Orthogonality (`Off-Diag Cos`: Funnel vs. Std MLP) | `ReLU` Cutoff (`%` Zeroed Out: Funnel vs. Std MLP) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1-Shot (10 samples total)** | **43.39%** (Fwd) / **42.08%** (Loop) | 39.19% | 30.05% (`Cos: 0.367`) | **`0.014` vs. `0.634`** (`45×` more orthogonal) | **`78.8%` vs. `44.0%`** |
| **5-Shot (50 samples)** | **68.20%** | 65.23% | 44.54% (`Cos: 0.870`) | **`0.017` vs. `0.694`** (`41×` more orthogonal) | **`80.8%` vs. `23.5%`** |
| **10-Shot (100 samples)** | **74.92%** | 71.70% | 60.21% (`Cos: 0.757`) | **`0.017` vs. `0.736`** (`43×` more orthogonal) | **`76.2%` vs. `27.7%`** |
| **50-Shot (500 samples)** | **88.52%** | 88.84% | 74.24% (`Cos: 0.894`) | **`0.011` vs. `0.683`** (`62×` more orthogonal) | **`81.9%` vs. `36.6%`** |
| **Full MNIST (60,000)** | **`98.07%`** (`105→32→11`, Relaxed) / **97.93%** (`256→64→32`) | 97.71% | **74.68%** (`Cos: 0.935`) | **`0.019` vs. `0.551`** (`29×` more orthogonal) | **`84.3%` vs. `40.6%`** |

## 3. Quick Start (Pure NumPy + Pillow on CPU)

```bash
pip install numpy pillow
python3 run_elastic_funnel.py         # Stage 1: Gradual layer & width discovery from 2352 -> 10
python3 run_diffusion_reflection.py   # Stages 1-4: Frozen E + G + Reality-Anchored Joint Relaxation + Convergence Loop
python3 run_experiment.py             # Reference few-shot, unsupervised, and multi-resolution benchmarks
python3 run_occlusion_loop.py         # Reference 4-step corruption-matched occlusion benchmark
python3 render_tables.py              # Render Retina PNG benchmark tables in ./assets/
```

## License

MIT License
