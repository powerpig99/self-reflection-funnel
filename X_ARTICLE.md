# Attention Is Not at All What You Need—Self-Reflection Is: The 4-Stage Dual-Cut Funnel

*[Attach Header / Inline Image: `assets/diffusion_reflection_grid.png`]*

*[Attach Secondary Image: `assets/occlusion_refinement_grid.png`]*

In 2017, *"Attention Is All You Need"* ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) launched the modern AI era on a single premise: if every surface token attends to every other surface token (`O(N²)` self-attention), a network can learn any structure without recurrent loops or convolutions.

Yet why does self-attention work at all—and why did deep learning split into **`O(N²)` self-attention** in Transformers and **rigid `k × k` sliding stencils** in Convolutional Neural Networks (CNNs)?

Viewed through [The Dual Cut](https://not-a-toe.com/posts/the-dual-cut/), **what we call self-attention (`Attention(Q, K, V) = softmax(QKᵀ / √d_k)V`) is proposed here as a special case of self-reflection**: from the same input `X`, the layer extracts a low-dimensional abstraction (`Q, K` in `d_k ≪ d_model`) and applies that abstraction as an operator back onto the input's own high-dimensional payload (`V`). A standard Vision Transformer pays `O(N²)` at every layer only because it refuses to collapse spatial positions when extracting the abstraction—forcing every patch to poll every other patch (`X ↔ X`) horizontally to simulate a shared representation `z*`.

When the top-level command—*"recognize this picture as one of the 10 digits `0–9`"*—is already fixed by the problem statement (`d_out = 10`), we can drop **both** `3 × 3` convolutions and `O(N²)` token polling—replacing them with a **`Linear + ReLU` Determination Funnel** and a **4-Stage Dual-Cut Self-Reflection Pipeline (`E(G(z*)) = z*`)**:

1. **Stage 1 — Problem-Driven Layer & Width Discovery (`run_elastic_funnel.py`)**: Starting directly at `d_in → 10` (`L = 0`) with **zero artificial width floor (`min_width = 1`)** on held-out `train[:4000]` and splitting only the remaining gap `d*_{l-1} → 10` via $D_{\text{init}}(l) = \text{round}(\sqrt{d^*_{l-1} / 10} \cdot 10)$:
   - **On `2,352` coupled `(x, y, I)` tuples (`2,352 → 10`)**: MNIST discovers **`2,352 → 105 → 32 → 11 → 10`** (`250,940` parameters—**`59.6%` smaller** than our earlier `621,226`-parameter `2,352 → 256 → 64 → 32 → 10` funnel, reaching **`97.67%` head accuracy**, **`97.58%` prototype accuracy**, and **`0.0191` off-diagonal cosine**). At `L = 2`, the second cut contracts from `35 → 12`; at `L = 3`, the third cut is initialized at $\text{round}(\sqrt{12 / 10} \cdot 10) = 11$, expands to `13`, and returns to `11`, terminating discovery at the fixed point $\text{round}(\sqrt{11 / 10} \cdot 10) = 10$.
   - **On `784` pixel-order intensities (`784 → 10`, where vector slot order already encodes `(x_i, y_i)`)**: MNIST discovers **`784 → 85 → 30 → 10 → 10`** (**`69,725` parameters**, `97.42%` head / `97.33%` prototype accuracy, `0.0171` off-diagonal cosine), where the third cut initializes at `14` and contracts by 4 units (`14 → 11 → 10`), landing directly on the exact class count `10`.
2. **Stage 2 — Reverse-Diffusion Expression Trained by Iterative Redraw Against Frozen `E` (`run_diffusion_reflection.py`)**: Inserting the midpoint layer $d_{\text{mid}} = \text{round}(\sqrt{784 / d_1} \cdot d_1)$ (`287` for `d_1 = 105`; `258` for `d_1 = 85`) across the `d_1 → 784` gap creates a uniform `~2.8×` reverse-diffusion cascade, trained on clean numbers by taking **frozen `E`'s actual recognition vectors** $Z^{(k)} = E(X^{(k)})$ to redraw $X^{(k+1)}$ and measuring loss through frozen `E` against the desired digit `y`.
3. **Stage 3 — Reality-Anchored vs. Single-Variable Unconstrained Relaxation (`run_diffusion_reflection.py`)**:
   - **Single-Variable Reality-Anchor Ablation**: Keeping every learning rate, clean update, corrupted update, and loss weight 100% identical to Stage 3 Anchored and removing *only* the `correct_0` gate on $X^{(1)}$ shows that without the anchor, the **hardest 5% tail drops `-2.60%` at Step 0** (`71.20% → 68.60%`) and **`-3.20%` (Self-Judged 3B: `69.60% → 66.40%`) / `-1.60%` (Frozen-Judged 3A: `70.00% → 68.40%`) converged**, **Picket Fence drops `-1.81%` (`75.78% → 73.97%` in 3B) / `-1.46%` (`75.84% → 74.38%` in 3A)**, and the self-judged loop takes **`11.63` steps instead of `8.76` steps** on the hardest 5% digits.
   - **True 1-to-1 Same-Size & Closest-Size Comparisons (22 Epochs Same Corruption Stream)**:
     - **At `86,300` parameters (`784 → 105 → 32 → 11 → 10`)**: The pure `784` pixel-order Funnel (`86,300` params) beats the exact same-size `86,300`-parameter Standard MLP (`97.51%` Natural, `64.60%` Fence, `65.42%` Blackout, `83.30%` Clutter) by **`+0.37%` on Natural MNIST** (`97.88%` Step 0 / `97.83%` converged), **`+8.64%` on Picket Fence** (`73.24%`), and **`+3.74%` on Heavy Clutter** (`87.04%`), while improving Center Blackout by **`+2.55%` at Step 0** (`61.77% → 62.98%`) over the `2,352`-tuple funnel.
     - **At `~220k–251k` parameters**: Headline **Stage 3A** (`250,940` params) ties the closest-in-size **`219,818`-parameter Standard MLP** (`784 → 256 → 64 → 32 → 10`, `98.03%` Natural) on Natural MNIST (`98.00%` Step 0 / `97.94%` converged), leads by **`+3.94%` on Picket Fence** (`75.84%` vs. `71.90%`) and **`+1.98%` on Heavy Clutter** (`87.72%` vs. `85.74%`), and trails on Center Blackout (`61.75%` vs. `66.38%`).
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

### 2. Why Convolutions Were a Workaround—and When Vector Order Already Encodes Position

If Transformers stay flat (`N → N`) while letting everything talk to everything, CNNs do the opposite: they have a vertical compression funnel, but they **predetermine the type of abstraction** by locking part-grouping into rigid `3 × 3` spatial stencils.

Do we understand a picture through a uniform `3 × 3` sliding window? *Sometimes, but not always.* When a cat stands behind a picket fence, the head on the left and the tail on the right belong to **one collective identity** (`cat`), while the fence post sitting right between them inside the `3 × 3` window belongs to a **completely different identity** (`fence`).

Look at how biological vision and linear algebra solve this cleanly:

1. **100:1 Compression at Layer 1**: The human retina compresses `~120M` photoreceptors into `~1M` optic nerve fibers right at the entrance—because a single pixel has **zero independent meaning** on its own.
2. **When Explicit `(x, y, I)` Tuples Help vs. When Vector Order Already Encodes `(x, y)`**:
   - When evaluating a single network across **multiple spatial resolutions** (`14×14`, `28×28`, and `56×56` in `run_experiment.py`) or unordered point sets, pairing each intensity with its normalized coordinates `(xᵢ · Iᵢ/255, yᵢ · Iᵢ/255, Iᵢ) ∈ [0, 255]³` places every resolution inside the same coordinate universe.
   - On a **fixed `28×28` grid**, however, a dense linear cut $u = \sum_{i=1}^{784} I_i w_i + b$ already assigns a dedicated weight row $w_i \in \mathbb{R}^{d_1}$ to each pixel slot $i \in \{0, \dots, 783\}$. Because $(x_i, y_i)$ are constants at slot $i$, $x_i I_i w_{i,x} + y_i I_i w_{i,y} + I_i w_{i,I} = I_i(x_i w_{i,x} + y_i w_{i,y} + w_{i,I})$ spans the exact same per-pixel linear subspace as $I_i$ alone! Passing the `784` ordered intensities directly eliminates the `3×` Layer-1 parameter expansion (`250,940 → 86,300` or `69,725` parameters) and removes the spatial norm skew when the center of a digit is blacked out.
3. **`ReLU` as the Determination Cut**: Layer 1 is a single fully connected high-compression projection, followed by gradual `ReLU` cuts (`max(0, Wx + b)`) that slice off `67.6%–85.7%` of sub-threshold fluctuations to `0`. Because `ReLU` vectors are non-negative (`z ≥ 0`), **two representations are orthogonal (`z_a · z_b = 0`) if and only if they activate completely disjoint neurons**.

---

### 3. Stage 1: Discovering the Layers and Widths (`run_elastic_funnel.py`)

*[Attach Table 3 Image: `assets/table3_dual_cut_3stage.png`]*

*[Attach 11th Dimension Image: `assets/eleventh_dimension_inspection.png`]*

In our initial prototype (`run_experiment.py`), we used legacy power-of-two layer widths (`2,352 → 256 → 64 → 32 → 10`, `621,226` parameters). Under [The Dual Cut](https://not-a-toe.com/posts/the-dual-cut/), neither the number of layers (`L`) nor the width of each layer (`d*_l`) should be guessed in advance—only the input boundary (`d_in`) and the problem statement's target (`10` digits) are given.

Between any upstream representation `d_{l-1}` and the `10` output groups, the per-group compression factor is `d_{l-1} / 10`. Splitting that multiplicative factor in half gives a parameter-free midpoint initialization rule:

$$D_{\text{init}}(l) = \text{round}\!\left(\sqrt{\frac{d^*_{l-1}}{10}} \cdot 10\right)$$

Starting directly at `d_in → 10` (`L = 0`) with **no artificial width floor (`min_width = 1`)** on held-out `train[:4000]`, training each cut against supervised label loss (`CE(u_{10}, y)` + internal label-group orthogonality), and folding collinear cones (`cos ≥ 0.90`) while pruning spent scaffolding (`< 5%` mean layer energy) produces an exact constructive sequence for both `2,352`-tuple and `784` pixel-order inputs:

| Input Representation & Step (`L`) | Midpoint Init (from Discovered `d*_{l-1}`) | Discovered Architecture (`[d_in, …, 10]`) | Parameters (`%` Shed vs. Init) | Test Accuracy (`Head / Proto`) | Mean Internal Loss (`L_internal`) | Latent Orthogonality (`Off-Diag Cos`) | Next Midpoint $\text{round}(\sqrt{d^*_l/10}\cdot 10)$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`2,352`-Tuple `L = 0`** | `[2352, 10]` | **`[2352, 10]`** | `23,530` (`0.0%`) | `92.59% / 92.59%` | `0.1492` | `0.2231` | `153` |
| **`2,352`-Tuple `L = 1`** | `[2352, 153, 10]` | **`[2352, 121, 10]`** | `285,933` (`-20.9%`) | `97.56% / 92.81%` | `0.1402` | `0.1433` | `35` |
| **`2,352`-Tuple `L = 2`** | `[2352, 153, 35, 10]` | **`[2352, 121, 12, 10]`** | `286,307` (`-21.7%`) | `97.41% / 97.34%` | `0.0747` | `0.0253` | `11` |
| **`2,352`-Tuple `L = 3` (Fixed Pt)** | `[2352, 153, 35, 11, 10]` | **`[2352, 105, 32, 11, 10]`** | **`250,940` (`-31.4%`)** | **`97.67% / 97.58%`** | **`0.0595`** | **`0.0191`** | **`10` (Converged!)** |
| **`784` Pixel-Order `L = 0`** | `[784, 10]` | **`[784, 10]`** | `7,850` (`0.0%`) | `92.54% / 92.54%` | `0.1559` | `0.1951` | `89` |
| **`784` Pixel-Order `L = 1`** | `[784, 89, 10]` | **`[784, 76, 10]`** | `60,430` (`-14.6%`) | `97.64% / 92.89%` | `0.1387` | `0.1407` | `28` |
| **`784` Pixel-Order `L = 2`** | `[784, 89, 28, 10]` | **`[784, 84, 21, 10]`** | `67,945` (`-6.5%`) | `97.50% / 97.36%` | `0.0786` | `0.0371` | `14` |
| **`784` Pixel-Order `L = 3` (Fixed Pt)** | `[784, 89, 28, 14, 10]` | **`[784, 85, 30, 10, 10]`** | **`69,725` (`-4.4%`)** | **`97.42% / 97.33%`** | **`0.0601`** | **`0.0171`** | **`10` (`14→10` Converged!)** |

Look at what happens at `L = 2 → L = 3` across both representations:
- On `2,352` tuples, `L = 2` contracts from `35 → 12`, whose midpoint to `10` is `11`; at `L = 3`, that layer expands to `13` and returns to `11` (`[2352, 105, 32, 11, 10]`, `250,940` params—`59.6%` smaller than the `621,226`-parameter legacy funnel).
- On `784` pixel-order inputs, `L = 2` contracts from `28 → 21`, whose midpoint to `10` is `14`; at `L = 3`, that third layer **genuinely contracts by 4 units (`14 → 11 → 10`)**, landing on **`[784, 85, 30, 10, 10]`** (`69,725` params) and terminating at the exact class count `10`.

---

### 4. Stages 2–4: Reverse-Diffusion Expression (`G`), Single-Variable Anchor Ablation, and Same-Size Baselines (`run_diffusion_reflection.py`)

Why does human expression (drawing a person) take far more effort and detail memory than abstraction (recognizing a person in a single glance)?
- **Abstraction (`E`)** is many-to-one compression: it only has to discard irrelevant variance, so it runs in a single fast pass.
- **Expression (`G`)** is one-to-many detail synthesis: jumping directly from `d_1` (`105` or `85`) back to `784` pixels is a `7.5×–9.2×` chasm. Applying the same midpoint formula between `d_1` and `784` inserts $d_{\text{mid}} = \text{round}(\sqrt{784 / d_1} \cdot d_1)$ (`287` or `258`), turning `G` into a uniform `~2.8×` reverse-diffusion cascade.

We evaluate the closed loop across our `784` and `2,352` funnels, a **strict single-variable unconstrained ablation** (removing *only* `* correct_0` while keeping all learning rates, clean updates, corrupted updates, and loss weights identical to Stage 3 Anchored), and three **Same-Stream Standard MLP Baselines** (`69,725`, `86,300`, and `219,818` parameters) trained on the exact same corruption stream for the same 22 epochs:

| Model / Pipeline Stage (`10,000` Unseen Test Set) | Parameters (`E`) | Natural MNIST (`All 10k`) | Natural MNIST (`Hardest 5% Tail`) | Picket Fence (`Step 0 → Converged`) | Center Blackout (`Step 0 → Converged`) | Heavy Clutter (`Step 0 → Converged`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Same-Size `69.7k` Standard MLP (`784 → 85 → 30 → 10 → 10`, 22 ep)** | `69,725` | `97.55%` | — | `74.27%` | `59.74%` | `82.92%` |
| **2. Discovered `69.7k` `784`-Funnel (`784 → 85 → 30 → 10 → 10`, Stage 3A)** | `69,725` | **`97.68%` → `97.64%`** | `65.80% → 65.00%` (`8.8` steps) | `73.05% → 73.38%` (`+0.33%`, `3.3` steps) | `60.23% →` **`61.22%`** (`+0.99%`, `4.7` steps) | `85.98% →` **`86.45%`** (`+0.47%`, `3.2` steps) |
| **3. Same-Size `86.3k` Standard MLP (`784 → 105 → 32 → 11 → 10`, 22 ep)** | `86,300` | `97.51%` | — | `64.60%` | **`65.42%`** | `83.30%` |
| **4. Same-Size `86.3k` `784`-Funnel (`784 → 105 → 32 → 11 → 10`, Stage 3A)** | `86,300` | **`97.88%` → `97.83%`** | `69.60% → 68.60%` (`7.9` steps) | `71.98% →` **`73.24%`** (`+1.26%`, `3.9` steps) | `61.77% →` **`62.98%`** (`+1.21%`, `4.5` steps) | `86.31% →` **`87.04%`** (`+0.73%`, `3.1` steps) |
| **5. Closest-to-`251k` Standard MLP (`784 → 256 → 64 → 32 → 10`, 22 ep)** | `219,818` | **`98.03%`** | — | `71.90%` | **`66.38%`** | `85.74%` |
| **6. Stage 2 (`Frozen E + Trained G`, `2352 → 105 → 32 → 11 → 10`)** | `250,940` | `97.36% → 97.40%` | `63.80% → 64.60%` (`10.3` steps) | `74.11% → 75.69%` (`+1.58%`, `3.7` steps) | `55.91% → 59.20%` (`+3.29%`, `4.8` steps) | `83.90% → 85.26%` (`+1.36%`, `3.4` steps) |
| **7. Stage 3B Unconstrained (1-Var No-Anchor, Self-Judged)** | `250,940` | `97.91% → 97.80%` | `68.60% → 66.40%` (`11.6` steps) | `73.22% → 73.97%` (`+0.75%`, `4.6` steps) | `60.16% → 63.61%` (`+3.45%`, `5.3` steps) | `86.99% → 87.63%` (`+0.64%`, `3.9` steps) |
| **8. Stage 3A Unconstrained (1-Var No-Anchor, Judged by `E_frozen`)** | `250,940` | `97.91% → 97.90%` | `68.60% → 68.40%` (`5.8` steps) | `73.22% → 74.38%` (`+1.16%`, `3.3` steps) | `60.16% → 62.87%` (`+2.71%`, `3.8` steps) | `86.99% → 87.76%` (`+0.77%`, `2.8` steps) |
| **9. Stage 3B Reality-Anchored (`Relaxed E + G`, Self-Judged)** | `250,940` | **`98.00%` → `97.92%`** | **`71.20%` → `69.60%`** (`8.8` steps) | `74.23% → 75.78%` (`+1.55%`, `3.5` steps) | `59.22% →` **`62.40%`** (`+3.18%`, `4.6` steps) | `87.07% → 87.67%` (`+0.60%`, `2.9` steps) |
| **10. Stage 3A Reality-Anchored (`Relaxed E + G`, Judged by `E_frozen`)** | `250,940` | **`98.00%` → `97.94%`** | **`71.20%` → `70.00%`** (`6.3` steps) | `74.23% →` **`75.84%`** (`+1.61%`, `3.4` steps) | `59.22% → 61.75%` (`+2.53%`, `4.2` steps) | `87.07% →` **`87.72%`** (`+0.65%`, `2.9` steps) |

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
