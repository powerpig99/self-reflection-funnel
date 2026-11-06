# 3-Stage Dual-Cut Implementation Plan (`Determination Funnel v2`)

This document specifies the three-stage architecture and training workflow for **Determination Funnel** under [The Dual Cut](https://not-a-toe.com/posts/the-dual-cut/), eliminating arbitrary power-of-two layer widths (`256 → 64 → 32`), pre-built layer cascades, and fixed test-time loop counts.

## Core Principles

1. **The Implied Command Is Fixed by the Problem Statement (`d_out = 10`)**:
   - A raw image `X ∈ ℝ²³⁵²` (`28 × 28` pixels × `3` coordinates `(x, y, I)` in `[0, 255]³`) carries only sensory payload (`V`). The instruction—*"recognize this picture as one of the 10 digits `0–9`"*—is decided by the problem statement itself (`d_out = 10`).
   - Because the command is fixed by the problem statement, `O(N²)` token-to-token self-attention across pixels is unnecessary; self-attention is a special case of **self-reflection** (extracting the low-dimensional abstraction of an input and applying it back onto the input's own high-dimensional payload).

2. **Asymmetry Between Abstraction (`E`) and Expression (`G`)**:
   - **Abstraction (`E: X → Z`)** is many-to-one lossy compression: it strips away irrelevant local variance via `Linear + ReLU` cuts. Once learned on clean data, `E` is fast (`O(1)`), low-rank, and **stays permanently frozen** as the invariant judge.
   - **Expression (`G: Z → X̂`)** is one-to-many detail expansion: it acts as a multi-stage reverse-diffusion denoiser that bridges large dimensional gaps with intermediate layers, iteratively redrawing ambiguous inputs until frozen `E` recognizes them with minimal internal loss.

## Stage 1: Gradual Supervised Recognition Discovery (`E`), Then Freeze

Instead of guessing hidden layer counts or widths (`256, 64, 32`) or pre-building a multi-layer cascade upfront (which introduces Image lag), `E` starts directly at `2,352 → 10` (`L = 0`) and refines one cut at a time from the **actual discovered width `d*_{l-1}`**:

1. **Midpoint Split Rule (`d_out = 10`)**:
   - At step `l ≥ 1`, split the remaining gap between the previously discovered bottom hidden width `d*_{l-1}` (with `d*_0 = 2,352`) and `d_out = 10` at the multiplicative midpoint:
     $$D_{\text{init}}(l) = \text{round}\!\left(\sqrt{\frac{d^*_{l-1}}{10}} \cdot 10\right)$$
2. **Supervised Label Loss Across All Cuts**:
   - Train `E` on clean training digits using output cross-entropy at the terminal `10` layer (`CE(softmax(u_{10}), y)`) plus supervised label-group orthogonality (`latent_group_orthogonality_grad(z_l, y)`) at every hidden `ReLU` cut `z_l`.
3. **Zero-Floor Elastic Breathing (`min_width = 1`)**:
   - **Collapse on Sufficiency**: Fold collinear cones (`cos(z_j, z_k) ≥ 0.90`) into a single neuron by summing their scaled outgoing weights (`W_{l+1}[j, :] += (rms(z_k)/rms(z_j)) W_{l+1}[k, :]`), and prune spent scaffolding neurons whose downstream contribution energy `E_j = E[z_j²] · ‖W_{l+1}[j, :]‖₂²` falls below `5%` of the layer mean.
   - **Expand on Insufficiency**: If a layer has zero redundant or low-energy slack left (`100%` saturation), allocate `+round(sqrt(d_l / 10))` orthogonal neurons in the nullspace of `W_l`.
4. **Mathematical Fixed-Point Termination (`11` Emerges Naturally)**:
   - **`L = 0`**: `[2352, 10]` → `Head Acc: 92.59%`, `Internal Loss: 0.1492`, `Off-Diag Cos: 0.2231`. Next midpoint: `round(sqrt(2352 / 10) * 10) = 153`.
   - **`L = 1`**: `[2352, 153, 10]` → collapses to **`[2352, 121, 10]`** (`Head Acc: 97.56%`, `Internal Loss: 0.1402`, `Off-Diag Cos: 0.1433`). Next midpoint from discovered `121`: `round(sqrt(121 / 10) * 10) = 35`.
   - **`L = 2`**: `[2352, 153, 35, 10]` → collapses to **`[2352, 121, 12, 10]`** (`Head Acc: 97.41%`, `Proto Acc: 97.34%`, `Internal Loss: 0.0747`, `Off-Diag Cos: 0.0253`). Next midpoint from discovered `12`: `round(sqrt(12 / 10) * 10) = 11`.
   - **`L = 3`**: `[2352, 153, 35, 11, 10]` → collapses to **`[2352, 105, 32, 11, 10]`** (`Head Acc: 97.67%`, `Proto Acc: 97.58%`, `Internal Loss: 0.0595`, `Off-Diag Cos: 0.0191`, `250,940` parameters—`60%` smaller than the legacy `620,864`-parameter network).
   - At `d*_3 = 11`, the midpoint formula gives `round(sqrt(11 / 10) * 10) = 10 == d_out`: no further gap exists. Notice that **`11` derived itself naturally** (`10` pure digit cones + `1` shared bridge/uncertainty axis) without ever being prescribed.
5. **Permanent Freeze**:
   - Once `[2352, 105, 32, 11, 10]` is discovered and anchored on supervised label loss, **all weights of `E` are frozen permanently** and never modified in Stage 2 or Stage 3.

## Stage 2: Reverse-Diffusion Generator (`G`) Trained by Iterative Redraw Against Frozen `E`

1. **Bridging the Large Gap (`105 → 287 → 784`)**:
   - In frozen `E` (`2352 → 105 → 32 → 11 → 10`), the adjacent hidden ratios are `~3×` (`10 → 11 → 32 → 105`), whereas jumping from `105` back to `784` pixels is a `7.5×` gap (`22.4×` in `2,352d` tuple space).
   - Applying the same midpoint rule between `105` and `784` inserts `d_mid = round(sqrt(784 / 105) * 105) = 287`, giving `G` a uniform `~2.7×–2.9×` reverse-diffusion expansion cascade:
     $$[10 + 11] \longrightarrow 32 \longrightarrow 105 \longrightarrow 287 \longrightarrow 784$$
2. **Intrinsic Internal Loss (`L_internal`) Inside Frozen `E`**:
   - For any input `X`, frozen `E` computes its **internal loss `L_internal(X)`** without needing an external label:
     $$\mathcal{L}_{\text{internal}}(X) = 0.45\,\underbrace{\bigl(1 - (p_{(1)} - p_{(2)})\bigr)}_{\text{output cone indeterminacy}} + 0.55 \sum_{l=1}^{3} w_l \underbrace{\bigl(1 - \cos(z_l(X), \text{proto}^{(l)}_{\hat{y}})\bigr)}_{\text{internal cone residual}}$$
3. **Unrolled Redraw-and-Recognize Training on Clean Numbers**:
   - Starting from clean training numbers `(X_0, y)` perturbed along a forward diffusion schedule `X^{(0)}`:
   - **Step `k = 0`**: Pass `X^{(0)}` through **frozen `E`** to obtain the **actual recognition vectors** `Z^{(0)} = (p_{10}^{(0)}, z_3^{(0)}, z_2^{(0)}, z_1^{(0)})` and `L_internal^{(0)}`. (Never feed the ground-truth one-hot label `y` into `G`.)
   - **Redraw `X^{(1)}`**: `G` uses `Z^{(0)}` (gating noisy bottom-up skip features `z_2, z_1` by `1 - 0.75 · L_internal^{(0)}`) to redraw the canvas `X^{(1)} = Redraw(X^{(0)}, G(Z^{(0)}), L_internal^{(0)})`.
   - **Re-Recognize & Backpropagate Supervised Label Loss Through Frozen `E`**: Pass `X^{(1)}` through **frozen `E`** to get the new recognition result `Z^{(1)} = E(X^{(1)})`, measure the supervised recognition loss of `E(X^{(1)})` against the desired label `y` (anchored by clean `X_0`), and backpropagate analytical gradients through frozen `E` into `G`.
   - **Iterate (`k = 1 → 2`)**: Feed the **new recognition vector `Z^{(1)}`** back into `G` to redraw `X^{(2)} = Redraw(X^{(1)}, G(Z^{(1)}), L_internal^{(1)})`, pass `X^{(2)}` through frozen `E`, and measure loss against `y` again—training `G` so repeated redrawing pulls the recognition vector progressively closer to the desired digit.

## Stage 3: Test-Time Diffusion Gated by Internal Loss Until Recognition Vectors Converge

At test time, both `E` and `G` remain **100% frozen** (zero retraining):

1. **Gate by Internal Loss (`L_internal`)**:
   - Pass test input `X^{(0)}` through frozen `E` to compute `Z^{(0)}` and `L_internal^{(0)}`.
   - If `L_internal^{(0)} ≤ τ_gate`, the recognition is already in a clean cone—exit immediately at Step 0 (`0` diffusion steps; `99.13%` accuracy on the low-loss `95%` of natural MNIST).
2. **Self-Terminating Redraw Loop Until Recognition Vectors Converge (`E(G(z*)) = z*`)**:
   - For uncertain inputs where `L_internal^{(0)} > τ_gate`, treat `X^{(0)}` as an intermediate diffusion state and iterate in an unbounded `while` loop (`k = 0, 1, 2, …`):
     $$Z^{(k)} = E\!\left(X^{(k)}\right) \;\longrightarrow\; X^{(k+1)} = \text{Redraw}\!\left(X^{(k)},\, G\!\left(Z^{(k)}\right)\right) \;\longrightarrow\; Z^{(k+1)} = E\!\left(X^{(k+1)}\right)$$
   - Accept step `k + 1` if and only if internal loss strictly decreases (`L_internal^{(k+1)} < L_internal^{(k)} - \varepsilon_{\text{loss}}`) and the recognized feature vector is still moving (`‖z_3^{(k+1)} - z_3^{(k)}‖_2 > \varepsilon_{\text{conv}}`). The moment `L_internal` stops dropping or the recognized vectors converge (`z^{(k+1)} ≈ z^{(k)}`), halt for that sample.
