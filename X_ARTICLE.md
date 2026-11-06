# Attention Is Not at All What You Need—Self-Reflection Is: The 3-Stage Dual-Cut Funnel

*[Attach Header / Inline Image: `assets/diffusion_reflection_grid.png`]*

*[Attach Secondary Image: `assets/occlusion_refinement_grid.png`]*

In 2017, *"Attention Is All You Need"* ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) launched the modern AI era on a single premise: if every surface token attends to every other surface token (`O(N²)` self-attention), a network can learn any structure without recurrent loops or convolutions.

Yet why does self-attention work at all—and why did deep learning split into **`O(N²)` self-attention** in Transformers and **rigid `k × k` sliding stencils** in Convolutional Neural Networks (CNNs)?

Viewed through [The Dual Cut](https://not-a-toe.com/posts/the-dual-cut/), **what we call self-attention (`Attention(Q, K, V) = softmax(QKᵀ / √d_k)V`) is already a special case of self-reflection**: from the same input `X`, the layer extracts a low-dimensional abstraction (`Q, K` in `d_k ≪ d_model`) and applies that abstraction as an operator back onto the input's own high-dimensional payload (`V`). A standard Vision Transformer pays `O(N²)` at every layer only because it refuses to collapse spatial positions when extracting the abstraction—forcing every patch to poll every other patch (`X ↔ X`) horizontally to simulate a shared representation `z*`.

When the top-level command—*"recognize this picture as one of the 10 digits `0–9`"*—is already fixed by the problem statement (`d_out = 10`), we can drop **both** `3 × 3` convolutions and `O(N²)` token polling—replacing them with a **`Linear + ReLU` Determination Funnel** over 8-bit **`(x, y, I) ∈ [0, 255]³` coordinate-value tuples** and a **3-Stage Dual-Cut Self-Reflection Pipeline (`E(G(z*)) = z*`)**:

1. **Stage 1 — Problem-Driven Layer & Width Discovery Starting from `2,352 → 10` (`run_elastic_funnel.py`)**: Instead of inventing arbitrary power-of-two widths (`256 → 64 → 32`) or pre-building a multi-layer cascade, we start directly at `2,352 → 10` (`L = 0`) with **zero artificial width floor (`min_width = 1`)** and split only the remaining gap `d*_{l-1} → 10` via $D_{\text{init}}(l) = \text{round}(\sqrt{d^*_{l-1} / 10} \cdot 10)$. Under supervised label loss and elastic breathing (folding collinear cones and pruning spent scaffolding), MNIST discovers the exact 3-cut funnel **`2,352 → 105 → 32 → 11 → 10`** (`250,940` parameters—`60%` smaller than the `256 → 64 → 32 → 10` network, reaching **`97.67%` accuracy** and **`0.0191` off-diagonal cosine**), where **`11` (`10` pure digit cones + `1` shared bridge/indeterminacy cone) derives itself automatically** and terminates depth discovery at the mathematical fixed point $\text{round}(\sqrt{11 / 10} \cdot 10) = 10$. All weights of `E` are then **frozen permanently**.
2. **Stage 2 — Reverse-Diffusion Expression Trained by Iterative Redraw Against Frozen `E` (`run_diffusion_reflection.py`)**: Because abstraction (`E: X → Z`, throwing away detail) is fast and low-rank while expression (`G: Z → X̂`, supplying detail) requires bridging the large gap (`105 → 784`), inserting the midpoint layer $287 = \text{round}(\sqrt{784 / 105} \cdot 105)$ creates a uniform `~2.8×` reverse-diffusion cascade **`[10 + 11] → 32 → 105 → 287 → 784`**. Keeping `E` 100% frozen, `G` is trained on clean numbers by taking **frozen `E`'s actual recognition vectors** $Z^{(k)} = E(X^{(k)})$ to redraw $X^{(k+1)}$, re-recognizing $X^{(k+1)}$ with frozen `E`, and measuring loss against the desired digit `y`.
3. **Stage 3 — Unbounded Test-Time Diffusion Gated by Internal Loss (`run_diffusion_reflection.py`)**: At test time (with both `E` and `G` frozen, zero retraining), gating by `E`'s intrinsic **Internal Loss (`L_internal`)** exits immediately at Step 0 on the low-loss `95%` of natural MNIST (**`99.13%` accuracy**, `mean = 0.03 steps`), while treating high-loss/uncertain digits and unseen occlusions as **middle steps `X_t` of a diffusion trajectory**—iterating in an unbounded `while` loop until `L_internal` stops dropping and the recognized vectors converge (`‖z^{(k+1)} - z^{(k)}‖_2 ≤ ε`).

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

### 3. Stage 1: Discovering the Layers and Widths from `2,352 → 10` (Why `11` Emerges by Itself)

*[Attach Table 3 Image: `assets/table3_dual_cut_3stage.png`]*

*[Attach 11th Dimension Image: `assets/eleventh_dimension_inspection.png`]*

In our initial prototype (`run_experiment.py`), we used legacy power-of-two layer widths (`2,352 → 256 → 64 → 32 → 10`). Under [The Dual Cut](https://not-a-toe.com/posts/the-dual-cut/), neither the number of layers (`L`) nor the width of each layer (`d*_l`) should be guessed in advance—only the input boundary (`2,352`) and the problem statement's target (`10` digits) are given.

Between any upstream representation `d_{l-1}` and the `10` output groups, the per-group compression factor is `d_{l-1} / 10`. Splitting that multiplicative factor in half gives a parameter-free midpoint initialization rule:

$$D_{\text{init}}(l) = \text{round}\!\left(\sqrt{\frac{d^*_{l-1}}{10}} \cdot 10\right)$$

Starting directly at `2,352 → 10` (`L = 0`) with **no artificial width floor (`min_width = 1`)**, training each cut against supervised label loss (`CE(u_{10}, y)` + internal label-group orthogonality), and folding collinear cones (`cos ≥ 0.90`) while pruning spent scaffolding (`< 5%` mean layer energy) produces an exact constructive sequence:

| Refinement Step (`L`) | Midpoint Init (from Discovered `d*_{l-1}`) | Discovered Architecture (`[2352, …, 10]`) | Parameters (`%` Shed vs. Init) | Test Accuracy (`Head / Proto`) | Mean Internal Loss (`L_internal`) | Latent Orthogonality (`Off-Diag Cos`) | Next Midpoint $\text{round}(\sqrt{d^*_l/10}\cdot 10)$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`L = 0` (Direct Cut)** | `[2352, 10]` | **`[2352, 10]`** | `23,530` (`0.0%`) | `92.59% / 92.59%` | `0.1492` | `0.2231` | `153` |
| **`L = 1` (1st Split)** | `[2352, 153, 10]` | **`[2352, 121, 10]`** | `285,933` (`-20.9%`) | `97.56% / 92.81%` | `0.1402` | `0.1433` | `35` |
| **`L = 2` (2nd Split)** | `[2352, 153, 35, 10]` | **`[2352, 121, 12, 10]`** | `286,307` (`-21.7%`) | `97.41% / 97.34%` | `0.0747` | `0.0253` | `11` |
| **`L = 3` (Fixed Point)** | `[2352, 153, 35, 11, 10]` | **`[2352, 105, 32, 11, 10]`** | **`250,940` (`-31.4%`)** | **`97.67% / 97.58%`** | **`0.0595`** | **`0.0191`** | **`10` (Converged!)** |

Look at what happens at `L = 2 → L = 3`:
- At `L = 2`, the second cut collapses from `35` down to `12`, whose midpoint to `10` is $\text{round}(\sqrt{12 / 10} \cdot 10) = 11$.
- At `L = 3`, that `11`-neuron layer survives with zero pruning even when `min_width = 1`: `10` of its neurons map 1-to-1 onto the `10` pure digit cones (`0–9`), while the **`11th` neuron** captures the shared upper-loop bridge/indeterminacy across curved digits (`3, 9, 8`, shown in `assets/eleventh_dimension_inspection.png`).
- Because $\text{round}(\sqrt{11 / 10} \cdot 10) = 10$, the midpoint formula hits its own mathematical fixed point (`10 == 10`): no further layer gap exists, and **`E` (`[2352, 105, 32, 11, 10]`) is permanently frozen**.

---

### 4. Stages 2 & 3: Reverse-Diffusion Expression (`G`) & Unbounded Convergence on Uncertain Inputs

Why does human expression (drawing a person) take far more effort and detail memory than abstraction (recognizing a person in a single glance)?
- **Abstraction (`E: 2,352 → 105 → 32 → 11 → 10`)** is many-to-one compression: it only has to discard irrelevant variance, so it runs in a single fast pass and stays fixed.
- **Expression (`G: [10 + 11] → 32 → 105 → 287 → 784`)** is one-to-many detail synthesis: jumping directly from `105` back to `784` pixels (`2,352` tuples) is a `7.5×–22.4×` chasm. Applying the same midpoint formula between `105` and `784` inserts $d_{\text{mid}} = \text{round}(\sqrt{784 / 105} \cdot 105) = 287$, turning `G` into a uniform `~2.8×` reverse-diffusion cascade.

Instead of retraining `E` on corruptions, we keep `E` **100% frozen** and train `G` on clean numbers by **iterative redraw-and-recognize**:
1. Pass $X^{(k)}$ through frozen `E` to obtain the **actual recognition vectors** $Z^{(k)} = E(X^{(k)})$ (never the ground-truth one-hot label `y`) and the intrinsic **Internal Loss** $\mathcal{L}_{\text{internal}}(X^{(k)})$ (combining output cone indeterminacy and internal prototype cosine residuals across `E`'s cuts).
2. Redraw $X^{(k+1)} = \text{Redraw}(X^{(k)}, G(Z^{(k)}), \mathcal{L}_{\text{internal}}^{(k)})$, feed $X^{(k+1)}$ back into frozen `E` to get the new recognition vector $Z^{(k+1)} = E(X^{(k+1)})$, and backpropagate the supervised recognition loss of $E(X^{(k+1)})$ against the desired digit `y` through frozen `E` into `G`.
3. At test time (with both `E` and `G` frozen), any hard or occluded digit with elevated $\mathcal{L}_{\text{internal}}$ is treated as the **middle step $X_t$ of a diffusion model**—iterating $Z^{(k)} = E(X^{(k)}) \to X^{(k+1)} = \text{Redraw}(X^{(k)}, G(Z^{(k)})) \to Z^{(k+1)} = E(X^{(k+1)})$ in an **unbounded `while` loop** until $\mathcal{L}_{\text{internal}}$ stops dropping and the recognized vectors converge ($\|z^{(k+1)} - z^{(k)}\|_2 \le \varepsilon$):

| Test Regime (`10,000` Unseen Test Set, Frozen `E` & `G`) | Step 0 Accuracy (`E(X^{(0)})`) | Converged Accuracy (`X^*`) | Loop Gain (`Δ Acc`) | Internal Loss (`Step 0 → Converged`) | Steps Taken (`Mean / Med / p90 / Max`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Natural MNIST — Low-Loss 95% (`L_int < 0.14`)** | `99.13%` | **`99.13%`** | `+0.00%` (Preserved) | `0.042 → 0.042` | **`0.03` / `0` / `0` / `2`** |
| **Natural MNIST — High-Loss 5% (`L_int ≥ 0.14`)** | `63.80%` | **`64.60%`** | **`+0.80%`** (`14` rescued) | **`0.341 → 0.223`** (`-34.6%`) | **`10.3` / `8` / `24` / `89`** |
| **Picket Fence (`Zero-Shot Unbounded Loop`)** | `74.11%` | **`75.69%`** | **`+1.58%`** | **`0.239 → 0.153`** (`-36.0%`) | **`3.7` / `0` / `11` / `225`** |
| **Center Blackout (`Zero-Shot Unbounded Loop`)** | `55.91%` | **`59.20%`** | **`+3.29%`** | **`0.261 → 0.160`** (`-38.7%`) | **`4.8` / `0` / `16` / `174`** |
| **Heavy Clutter (`Zero-Shot Unbounded Loop`)** | `83.90%` | **`85.26%`** | **`+1.36%`** | **`0.205 → 0.124`** (`-39.5%`) | **`3.4` / `0` / `10` / `243`** |

---

### 5. Reference Benchmarks: Surface-Loss Entanglement vs. `60×` Orthogonality & 4-Step Corruption-Matched Loop

*[Attach Table 1 Image: `assets/table1_benchmark_comparison.png`]*

*[Attach Table 2 Image: `assets/table2_maskfree_occlusion_loop.png`]*

In our reference head-to-head benchmarks (`run_experiment.py` and `run_occlusion_loop.py`) comparing the funnel against a **Standard Cross-Entropy MLP** (`I`-only input, output CE loss) and a **Standard Pixel-Loss Autoencoder** (minimizing surface pixel reconstruction error `‖X̂ − X‖²`):

- **Why Surface Expression Loss (`‖X̂ − X‖²`) Fails to Separate the Abstraction**: Penalizing surface pixel error forces the positive `ReLU` bottleneck `z* ≥ 0` to memorize stroke thickness, slant, and shared pixel intensity—driving its 10 digit prototypes into a single shared cone (`Off-Diag Cos: 0.367, 0.870, 0.757, 0.894, 0.935` across `1-shot`, `5-shot`, `10-shot`, `50-shot`, and `Full MNIST`) and capping full-MNIST accuracy at **`74.68%`** (vs. **`97.93%`** on `256→64→32` and **`97.67%`** on the discovered `105→32→11` funnel with `0.019` off-diagonal cosine).
- **Zero-Label Unsupervised Discovery & Multi-Resolution Generalization**: With **zero labels**, the forward funnel activates **all `10 / 10` groups** (`53.91%` unsupervised cluster purity, `5.4×` random chance), and evaluated zero-shot across `14×14`, `28×28`, and `56×56` resolutions mapped into `[0, 255]³`, the same weights score **`96.26%` / `97.77%` / `97.75%`**.
- **Corruption-Matched 4-Step Loop**: When co-trained on the same corruption distribution as a corruption-trained Standard MLP (`80.44% / 68.53% / 84.93%`), the 4-step mask-free loop climbs monotonically to **`82.09%`** on Picket Fence (`-58%` canvas MSE), **`76.53%`** on Center Blackout (`+7.06%` loop gain, `-40%` MSE), and **`85.65%`** on Heavy Clutter (`-74%` MSE).

---

### Reproducible Code & Benchmark

- **Published X Article**: [https://x.com/powerpig/status/2117986860995621026](https://x.com/powerpig/status/2117986860995621026)
- **Standalone Code & Pure-NumPy Benchmark Suite**: [https://github.com/powerpig99/self-reflection-funnel](https://github.com/powerpig99/self-reflection-funnel)
