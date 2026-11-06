# Attention Is Not at All What You Need—Self-Reflection Is: The 4-Stage Dual-Cut Funnel

*[Attach Header / Inline Image: `assets/diffusion_reflection_grid.png`]*

*[Attach Secondary Image: `assets/occlusion_refinement_grid.png`]*

In 2017, *"Attention Is All You Need"* ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) launched the modern AI era on a single premise: if every surface token attends to every other surface token (`O(N²)` self-attention), a network can learn any structure without recurrent loops or convolutions.

Yet why does self-attention work at all—and why did deep learning split into **`O(N²)` self-attention** in Transformers and **rigid `k × k` sliding stencils** in Convolutional Neural Networks (CNNs)?

Viewed through [The Dual Cut](https://not-a-toe.com/posts/the-dual-cut/), **what we call self-attention (`Attention(Q, K, V) = softmax(QKᵀ / √d_k)V`) is proposed here as a special case of self-reflection**: from the same input `X`, the layer extracts a low-dimensional abstraction (`Q, K` in `d_k ≪ d_model`) and applies that abstraction as an operator back onto the input's own high-dimensional payload (`V`). A standard Vision Transformer pays `O(N²)` at every layer only because it refuses to collapse spatial positions when extracting the abstraction—forcing every patch to poll every other patch (`X ↔ X`) horizontally to simulate a shared representation `z*`.

When the top-level command—*"recognize this picture as one of the 10 digits `0–9`"*—is already fixed by the problem statement (`d_out = 10`), we can drop **both** `3 × 3` convolutions and `O(N²)` token polling—replacing them with a **`Linear + ReLU` Determination Funnel** over 8-bit **`(x, y, I) ∈ [0, 255]³` coordinate-value tuples** and a **4-Stage Dual-Cut Self-Reflection Pipeline (`E(G(z*)) = z*`)**:

1. **Stage 1 — Problem-Driven Layer & Width Discovery Starting from `2,352 → 10` (`run_elastic_funnel.py`)**: Starting directly at `2,352 → 10` (`L = 0`) with **zero artificial width floor (`min_width = 1`)** on held-out `train[:4000]` and splitting only the remaining gap `d*_{l-1} → 10` via $D_{\text{init}}(l) = \text{round}(\sqrt{d^*_{l-1} / 10} \cdot 10)$, MNIST discovers the 3-cut funnel **`2,352 → 105 → 32 → 11 → 10`** (`250,940` parameters—`59.6%` smaller than the `621,226`-parameter `2,352 → 256 → 64 → 32 → 10` network, reaching **`97.67%` head accuracy**, **`97.58%` prototype accuracy**, and **`0.0191` off-diagonal cosine**). At `L = 2`, the second cut contracts from `35 → 12`; at `L = 3`, the third cut is initialized by the midpoint rule at $\text{round}(\sqrt{12 / 10} \cdot 10) = 11$, expands to `13`, and returns to `11`, confirming the midpoint estimate and terminating depth discovery at the fixed point $\text{round}(\sqrt{11 / 10} \cdot 10) = 10$. All weights of `E` are then **frozen** for Stage 2.
2. **Stage 2 — Reverse-Diffusion Expression Trained by Iterative Redraw Against Frozen `E` (`run_diffusion_reflection.py`)**: Inserting the midpoint layer $287 = \text{round}(\sqrt{784 / 105} \cdot 105)$ across the `105 → 784` gap creates a uniform `~2.8×` reverse-diffusion cascade **`[10 + 11] → 32 → 105 → 287 → 784`**, trained on clean numbers by taking **frozen `E`'s actual recognition vectors** $Z^{(k)} = E(X^{(k)})$ to redraw $X^{(k+1)}$ and measuring loss through frozen `E` against the desired digit `y`.
3. **Stage 3 — Reality-Anchored Joint Supervised Relaxation of `E` and `G` (`run_diffusion_reflection.py`)**: Once `E` and `G` are warmed up, relaxing both `E` and `G` together while anchoring `E` to reality (so `E` reinforces redrawn $X^{(1)}$ only when $E(X^{(0)})$ matched $y$, preventing the `-8.86%` Picket Fence Judge Drift seen in our unconstrained ablation) lifts our primary **Stage 3A** configuration (judged by stationary `E_frozen`) to **`98.00%` (`Step 0`) / `97.94%` (`Converged`)** on Natural MNIST, **`75.84%`** (`+1.61%` loop gain) on Picket Fence, **`61.75%`** (`+2.53%` loop gain; `62.40%` / `+3.18%` in Stage 3B) on Center Blackout, and **`87.72%`** (`+0.65%` loop gain) on Heavy Clutter. Against a **Same-Width Standard MLP (`784 → 105 → 32 → 11 → 10`)** trained on the exact same corruption stream for the same 22 epochs (`97.51%` Natural, `64.60%` Fence, `65.42%` Blackout, `83.30%` Clutter), Stage 3A leads by **`+0.49%` on Natural MNIST**, **`+11.24%` on Picket Fence**, and **`+4.42%` on Heavy Clutter**, while matching a `2.5×` larger `256 → 64 → 32` network (`98.03%` Natural) with **`59.6%` fewer parameters**.
4. **Stage 4 — Unbounded Test-Time Diffusion & Why We Still Need a Human in the Loop**: At test time (zero retraining), gating by `E`'s intrinsic **Internal Loss (`L_internal`)** means `95%` of clean digits take virtually no extra steps (`0.00` mean steps in 3B, `0.16` in 3A), while uncertain inputs iterate the reverse-diffusion redraw loop until `L_internal` stops dropping and the recognized vectors converge (`‖z^{(k+1)} - z^{(k)}‖_2 ≤ ε`). And when a genuine cross-class hybrid leaves residual `L_internal` high after convergence, the loop exposes why no closed machine loop can be its own ultimate sovereign: the human in the loop makes the final judgment call.

---

### 1. Three Structural Inferences & The Dimensionality Inversion

Start from the single premise that cannot be denied without being enacted: *self-distinguishing activity occurs—uncaused, unceasing*. Without assuming "parts" or "wholes" as separate axioms, three structural inferences follow immediately:

1. **Reciprocity of Abstraction and Parts**: Abstraction is the collective identity of the parts; conversely, the parts are the expression of the abstraction. Neither precedes the other: "abstraction" is the activity in its unity, and "parts" is the exact same activity in its articulation.
2. **Scale-Free Existence**: Every existence is an abstraction of its parts and a part of a higher abstraction. Looking downward is concrete expression; looking upward is abstract identity.
3. **Observation as Determination**: Every observation is a determination—the dissolving of lower-level parts and their independent degrees of freedom into a single collective identity.

Viewed through these three inferences, mainstream machine learning inverts dimensionality:
- Engineers call the latent space "high-dimensional" because they assign a wide vector (`d = 4096`) to each token, while the physical carrier looks low-dimensional (`1D` text, `2D` image grid, `3D` video volume).
- Measured by actual degrees of freedom, **the surface expression is vastly higher-dimensional than the latent abstraction**. A sequence of words or a grid of pixels spans an astronomical combinatorial space of independent surface degrees of freedom. Passing inward toward the abstraction dissolves those independent degrees of freedom into a unified collective identity. **Dissolving degrees of freedom is, by definition, dimensionality reduction.** The latent space (`z*`) is the **lowest-dimensional invariant bottleneck**—where an English sentence, a Chinese sentence, and a photograph of the same event collapse into the exact same determination before re-expanding into high-dimensional expressions.

---

### 2. Why Convolutions Were a Workaround—and How `[0, 255]³` Tuples + `ReLU` Replace Them

If Transformers stay flat (`N → N`) while letting everything talk to everything, CNNs do the opposite: they have a vertical compression funnel, but they **predetermine the type of abstraction** by locking part-grouping into rigid `3 × 3` spatial stencils.

Do we understand a picture through a uniform `3 × 3` sliding window? *Sometimes, but not always.* When a cat stands behind a picket fence, the head on the left and the tail on the right belong to **one collective identity** (`cat`), while the fence post sitting right between them inside the `3 × 3` window belongs to a **completely different identity** (`fence`).

Look at how biological vision and coordinate geometry solve this cleanly:

1. **100:1 Compression at Layer 1**: The human retina compresses `~120M` photoreceptors into `~1M` optic nerve fibers right at the entrance—because a single pixel has **zero independent meaning** on its own.
2. **Self-Contained `(Position, Value)` Tuples in a `[0, 255]³` Universe**: Engineers needed `3 × 3` convolutions only because they stripped a pixel's position `(x, y)` out of the datum and hid it in the 2D array index. When we keep position and value together inside the elementary datum:
   - **Grayscale pixel**: `(x, y, I)` — a `3D` vector.
   - **Color pixel**: `(x, y, R, G, B)` — a `5D/6D` vector.
   - **Video point**: `(x, y, t, R, G, B, …)` — a `7D` vector.
   Because a picture is the entire universe for that picture, scaling `x` and `y` to the full 8-bit span `[0, 255]` alongside the 8-bit pixel value `I ∈ [0, 255]` (with empty background `I = 0` carrying zero spatial offset `(x · I/255, y · I/255, I) = (0, 0, 0)`) places every image of any resolution inside the exact same **`[0, 255]³` coordinate universe**.
3. **`ReLU` as the Determination Cut**: Because every pixel tuple `(xᵢ, yᵢ, Iᵢ)` already carries its own coordinates, Layer 1 is a **single fully connected high-compression projection**, followed by gradual `ReLU` cuts (`max(0, Wx + b)`) that slice off `68.7%–84.3%` of sub-threshold fluctuations to `0`. Because `ReLU` vectors are non-negative (`z ≥ 0`), **two representations are orthogonal (`z_a · z_b = 0`) if and only if they activate completely disjoint neurons**.

---

### 3. Stage 1: Discovering the Layers and Widths from `2,352 → 10` (`run_elastic_funnel.py`)

*[Attach Table 3 Image: `assets/table3_dual_cut_3stage.png`]*

*[Attach 11th Dimension Image: `assets/eleventh_dimension_inspection.png`]*

In our initial prototype (`run_experiment.py`), we used legacy power-of-two layer widths (`2,352 → 256 → 64 → 32 → 10`). Under [The Dual Cut](https://not-a-toe.com/posts/the-dual-cut/), neither the number of layers (`L`) nor the width of each layer (`d*_l`) should be guessed in advance—only the input boundary (`2,352`) and the problem statement's target (`10` digits) are given.

Between any upstream representation `d_{l-1}` and the `10` output groups, the per-group compression factor is `d_{l-1} / 10`. Splitting that multiplicative factor in half gives a parameter-free midpoint initialization rule:

$$D_{\text{init}}(l) = \text{round}\!\left(\sqrt{\frac{d^*_{l-1}}{10}} \cdot 10\right)$$

Starting directly at `2,352 → 10` (`L = 0`) with **no artificial width floor (`min_width = 1`)** on held-out `train[:4000]`, training each cut against supervised label loss (`CE(u_{10}, y)` + internal label-group orthogonality), and folding collinear cones (`cos ≥ 0.90`) while pruning spent scaffolding (`< 5%` mean layer energy) produces an exact constructive sequence:

| Refinement Step (`L`) | Midpoint Init (from Discovered `d*_{l-1}`) | Discovered Architecture (`[2352, …, 10]`) | Parameters (`%` Shed vs. Init) | Test Accuracy (`Head / Proto`) | Mean Internal Loss (`L_internal`) | Latent Orthogonality (`Off-Diag Cos`) | Next Midpoint $\text{round}(\sqrt{d^*_l/10}\cdot 10)$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`L = 0` (Direct Cut)** | `[2352, 10]` | **`[2352, 10]`** | `23,530` (`0.0%`) | `92.59% / 92.59%` | `0.1492` | `0.2231` | `153` |
| **`L = 1` (1st Split)** | `[2352, 153, 10]` | **`[2352, 121, 10]`** | `285,933` (`-20.9%`) | `97.56% / 92.81%` | `0.1402` | `0.1433` | `35` |
| **`L = 2` (2nd Split)** | `[2352, 153, 35, 10]` | **`[2352, 121, 12, 10]`** | `286,307` (`-21.7%`) | `97.41% / 97.34%` | `0.0747` | `0.0253` | `11` |
| **`L = 3` (Fixed Point)** | `[2352, 153, 35, 11, 10]` | **`[2352, 105, 32, 11, 10]`** | **`250,940` (`-31.4%`)** | **`97.67% / 97.58%`** | **`0.0595`** | **`0.0191`** | **`10` (Converged!)** |

Look at what happens at `L = 2 → L = 3`:
- At `L = 2`, the second cut contracts from `35` down to `12`, whose midpoint to `10` is $\text{round}(\sqrt{12 / 10} \cdot 10) = 11$.
- At `L = 3`, that `11`-wide layer expands to `13` during elastic breathing and returns to `11` (`[2352, 153, 35, 11, 10] → [2352, 105, 32, 13, 10] → [2352, 105, 32, 11, 10]`), confirming the midpoint estimate (`10` digit cones + `1` shared upper-loop feature across curved digits `3, 9, 8`, shown in `assets/eleventh_dimension_inspection.png`).
- Because $\text{round}(\sqrt{11 / 10} \cdot 10) = 10$, the midpoint formula hits its own mathematical fixed point (`10 == 10`): no further layer gap exists, and **`E` (`[2352, 105, 32, 11, 10]`) is frozen** for Stage 2.

---

### 4. Stages 2–4: Reverse-Diffusion Expression (`G`), Reality-Anchored Relaxation, and the Same-Width Baseline (`run_diffusion_reflection.py`)

Why does human expression (drawing a person) take far more effort and detail memory than abstraction (recognizing a person in a single glance)?
- **Abstraction (`E: 2,352 → 105 → 32 → 11 → 10`)** is many-to-one compression: it only has to discard irrelevant variance, so it runs in a single fast pass.
- **Expression (`G: [10 + 11] → 32 → 105 → 287 → 784`)** is one-to-many detail synthesis: jumping directly from `105` back to `784` pixels (`2,352` tuples) is a `7.5×–22.4×` chasm. Applying the same midpoint formula between `105` and `784` inserts $d_{\text{mid}} = \text{round}(\sqrt{784 / 105} \cdot 105) = 287$, turning `G` into a uniform `~2.8×` reverse-diffusion cascade.

We train and test the closed loop in three progressive steps—and compare directly against a **Same-Width Standard MLP (`784 → 105 → 32 → 11 → 10`)** and a **2.5× Larger Standard MLP (`784 → 256 → 64 → 32 → 10`)** trained on the exact same corruption stream (`forward_diffusion_on_clean`) for the same 22 epochs:
1. **Stage 2 (Train `G` Against Frozen `E`)**: Keep `E` 100% frozen so representation basins cannot collapse while `G` learns to take `E`'s actual recognition vectors $Z^{(k)} = E(X^{(k)})$ and redraw $X^{(k+1)}$ to minimize supervised label loss through `E`.
2. **Stage 3 (Relax Both `E` and `G` With vs. Without a Reality Anchor)**: Unfreeze both `E` and `G` to co-adapt across the closed loop. In our **Stage 3 Unconstrained** ablation, training `E` unconditionally on every redrawn canvas $X^{(1)}$—even when $E(X^{(0)})$ guessed the wrong digit—causes `E` to flatter `G`'s wrong-class drawings ("Judge Drift"), dropping Picket Fence accuracy from `75.69%` down to `66.83%` (`-8.86%`). Anchoring `E` so it reinforces $X^{(1)}$ only when $E(X^{(0)})$ matched the true label $y$ eliminates that drift and lifts `E` in **Stage 3A** to **`98.00%` (`97.94%` converged)** on Natural MNIST, **`75.84%`** on Picket Fence, and **`87.72%`** on Heavy Clutter.
3. **Stage 4 (Unbounded Test-Time Convergence & The Human in the Loop)**: At test time (zero retraining), any hard or occluded digit with elevated $\mathcal{L}_{\text{internal}}$ is treated as the **middle step $X_t$ of a diffusion model**—iterating $Z^{(k)} = E(X^{(k)}) \to X^{(k+1)} = \text{Redraw}(X^{(k)}, G(Z^{(k)})) \to Z^{(k+1)} = E(X^{(k+1)})$ in an **unbounded `while` loop** until $\mathcal{L}_{\text{internal}}$ stops dropping and the recognized vectors converge ($\|z^{(k+1)} - z^{(k)}\|_2 \le \varepsilon$):

| Model / Pipeline Stage (`10,000` Unseen Test Set) | Parameters (`E` / Total) | Natural MNIST (`All 10k`) | Natural MNIST (`Hardest 5% Tail`) | Picket Fence (`Step 0 → Converged`) | Center Blackout (`Step 0 → Converged`) | Heavy Clutter (`Step 0 → Converged`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Same-Width Standard MLP (`784 → 105 → 32 → 11 → 10`, 22 ep)** | `86,300` | `97.51%` | — | `64.60%` | **`65.42%`** | `83.30%` |
| **2. 2.5× Larger Standard MLP (`784 → 256 → 64 → 32 → 10`, 22 ep)** | `219,818` | **`98.03%`** | — | `71.90%` | **`66.38%`** | `85.74%` |
| **3. Stage 2 (`Frozen E + Trained G`, `[10,11,32,105,287,784]`)** | `250,940` (`E`) | `97.36% → 97.40%` | `63.80% → 64.60%` (`10.3` steps) | `74.11% → 75.69%` (`+1.58%`, `3.7` steps) | `55.91% → 59.20%` (`+3.29%`, `4.8` steps) | `83.90% → 85.26%` (`+1.36%`, `3.4` steps) |
| **4. Stage 3 Unconstrained (`Relaxed E + G`, No Reality Anchor)** | `250,940` (`E`) | `98.05% → 97.99%` | `71.40% → 70.20%` (`15.8` steps) | `65.18% → 66.83%` (`-8.86%` Drift!) | `58.92% → 64.07%` (`+5.15%`, `7.8` steps) | `84.35% → 86.21%` (`+1.86%`, `8.1` steps) |
| **5. Stage 3B Reality-Anchored (`Relaxed E + G`, Self-Judged)** | `250,940` (`E`) | **`98.00%` → `97.92%`** | **`71.20%` → `69.60%`** (`8.8` steps) | `74.23% → 75.78%` (`+1.55%`, `3.5` steps) | `59.22% →` **`62.40%`** (`+3.18%`, `4.6` steps) | `87.07% → 87.67%` (`+0.60%`, `2.9` steps) |
| **6. Stage 3A Reality-Anchored (`Relaxed E + G`, Judged by `E_frozen`)** | `250,940` (`E`) | **`98.00%` → `97.94%`** | **`71.20%` → `70.00%`** (`6.3` steps) | `74.23% →` **`75.84%`** (`+1.61%`, `3.4` steps) | `59.22% → 61.75%` (`+2.53%`, `4.2` steps) | `87.07% →` **`87.72%`** (`+0.65%`, `2.9` steps) |

Look at why this also explains **why we still need a human in the loop**: on the `5%` hardest natural digits (`71.20%` at Step 0 vs. `70.00%` after `6.3` steps in Stage 3A), the remaining errors are genuine cross-class hybrids (a `9` written like an `a` or a `7` crossed like a `4`). An internal closed loop (`E ↔ G`) cannot manufacture external truth when the trace itself sits on the boundary between two concepts; instead, persistent residual $\mathcal{L}_{\text{internal}}$ after convergence is the exact signal to hand the final judgment call to the human observer at the outer loop.

---

### 5. Reference Benchmarks: Surface-Loss Entanglement vs. `60×` Orthogonality & 4-Step Corruption-Matched Loop

*[Attach Table 1 Image: `assets/table1_benchmark_comparison.png`]*

*[Attach Table 2 Image: `assets/table2_maskfree_occlusion_loop.png`]*

In our reference head-to-head benchmarks (`run_experiment.py` and `run_occlusion_loop.py`, shown in the companion videos) comparing the `2,352 → 256 → 64 → 32 → 10` (`621,226`-parameter) funnel against a **Standard Cross-Entropy MLP** (`I`-only input, output CE loss) and a **Standard Pixel-Loss Autoencoder** (minimizing surface pixel reconstruction error `‖X̂ − X‖²` across 12 clean epochs):

- **Why Surface Expression Loss (`‖X̂ − X‖²`) Fails to Separate the Abstraction**: Penalizing surface pixel error forces the positive `ReLU` bottleneck `z* ≥ 0` to memorize stroke thickness, slant, and shared pixel intensity—driving its 10 digit prototypes into a single shared cone (`Off-Diag Cos: 0.367, 0.870, 0.757, 0.894, 0.935` across `1-shot`, `5-shot`, `10-shot`, `50-shot`, and `Full MNIST`) and capping full-MNIST accuracy at **`74.68%`** (vs. **`97.93%`** head / **`97.77%`** prototype accuracy on the `256→64→32` funnel with `0.019` off-diagonal cosine and `83.9%` `z*` ReLU cutoff, vs. `40.6%` on the standard MLP).
- **Zero-Label Unsupervised Discovery & Multi-Resolution Generalization**: With **zero labels**, the forward funnel activates **all `10 / 10` groups** (`53.91%` unsupervised cluster purity, `5.4×` random chance), and evaluated zero-shot across `14×14`, `28×28`, and `56×56` resolutions mapped into `[0, 255]³`, the same weights score **`96.26%` / `97.77%` / `97.75%`**.
- **Corruption-Matched 4-Step Loop (`run_occlusion_loop.py`)**: When co-trained on the same corruption distribution as a corruption-trained Standard MLP (`80.44% / 68.53% / 84.93%`) with original-canvas stroke anchoring, the 4-step mask-free loop climbs monotonically to **`82.09%`** on Picket Fence (`-58%` canvas MSE), **`76.53%`** on Center Blackout (`+7.06%` loop gain, `-40%` MSE), and **`85.65%`** on Heavy Clutter (`-74%` MSE).

---

### Reproducible Code & Benchmark

- **Published X Article**: [https://x.com/powerpig/status/2117986860995621026](https://x.com/powerpig/status/2117986860995621026)
- **Standalone Code & Pure-NumPy Benchmark Suite**: [https://github.com/powerpig99/self-reflection-funnel](https://github.com/powerpig99/self-reflection-funnel)
