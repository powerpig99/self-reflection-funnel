# 4-Stage Dual-Cut Implementation Plan (`Determination Funnel v2`)

This document specifies the four-stage architecture and training workflow for **Determination Funnel** under [The Dual Cut](https://not-a-toe.com/posts/the-dual-cut/), eliminating arbitrary power-of-two layer widths (`256 → 64 → 32`), pre-built layer cascades, and fixed test-time loop counts while demonstrating why a closed-loop system requires a reality anchor and a human in the loop at its outer boundary.

## Core Principles

1. **The Implied Command Is Fixed by the Problem Statement (`d_out = 10`)**:
   - A raw image `X ∈ ℝ²³⁵²` (`28 × 28` pixels × `3` coordinates `(x, y, I)` in `[0, 255]³`) carries only sensory payload (`V`). The instruction—*"recognize this picture as one of the 10 digits `0–9`"*—is decided by the problem statement itself (`d_out = 10`).
   - Because the command is fixed by the problem statement, `O(N²)` token-to-token self-attention across pixels is unnecessary; self-attention is a special case of **self-reflection** (extracting the low-dimensional abstraction of an input and applying it back onto the input's own high-dimensional payload).

2. **Asymmetry Between Abstraction (`E`) and Expression (`G`)**:
   - **Abstraction (`E: X → Z`)** is many-to-one lossy compression: it strips away irrelevant local variance via `Linear + ReLU` cuts.
   - **Expression (`G: Z → X̂`)** is one-to-many detail expansion: it acts as a multi-stage reverse-diffusion denoiser that bridges large dimensional gaps with intermediate layers, iteratively redrawing ambiguous inputs until `E` recognizes them with minimal internal loss.

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
5. **Freeze for Stage 2**:
   - Once `[2352, 105, 32, 11, 10]` is discovered and anchored on supervised label loss, **all weights of `E` are frozen (`E_frozen`)** so `G` can learn against a stationary reference in Stage 2.

## Stage 2: Reverse-Diffusion Generator (`G`) Trained by Iterative Redraw Against Frozen `E`

1. **Bridging the Large Gap (`105 → 287 → 784`)**:
   - In frozen `E` (`2352 → 105 → 32 → 11 → 10`), the adjacent hidden ratios are `~3×` (`10 → 11 → 32 → 105`), whereas jumping from `105` back to `784` pixels is a `7.5×` gap (`22.4×` in `2,352d` tuple space).
   - Applying the same midpoint rule between `105` and `784` inserts `d_mid = round(sqrt(784 / 105) * 105) = 287`, giving `G` a uniform `~2.7×–2.9×` reverse-diffusion expansion cascade:
     $$[10 + 11] \longrightarrow 32 \longrightarrow 105 \longrightarrow 287 \longrightarrow 784$$
2. **Intrinsic Internal Loss (`L_internal`) Inside `E`**:
   - For any input `X`, `E` computes its **internal loss `L_internal(X)`** without needing an external label:
     $$\mathcal{L}_{\text{internal}}(X) = 0.45\,\underbrace{\bigl(1 - (p_{(1)} - p_{(2)})\bigr)}_{\text{output cone indeterminacy}} + 0.55 \sum_{l=1}^{3} w_l \underbrace{\bigl(1 - \cos(z_l(X), \text{proto}^{(l)}_{\hat{y}})\bigr)}_{\text{internal cone residual}}$$
3. **Unrolled Redraw-and-Recognize Training on Clean Numbers**:
   - Starting from clean training numbers `(X_0, y)` perturbed along a forward diffusion schedule `X^{(0)}`:
   - Pass `X^{(0)}` through **frozen `E`** to obtain the **actual recognition vectors** `Z^{(0)} = (p_{10}^{(0)}, z_3^{(0)}, z_2^{(0)}, z_1^{(0)})` and `L_internal^{(0)}`. (Never feed the ground-truth one-hot label `y` into `G`.)
   - `G` uses `Z^{(0)}` to redraw `X^{(1)} = Redraw(X^{(0)}, G(Z^{(0)}), L_internal^{(0)})`.
   - Pass `X^{(1)}` through **frozen `E`** to get the new recognition result `Z^{(1)} = E(X^{(1)})`, measure the supervised recognition loss of `E(X^{(1)})` against the desired label `y`, and backpropagate analytical gradients through frozen `E` into `G` across unrolled steps (`k = 0 → 1 → 2`).

## Stage 3: Reality-Anchored Joint Supervised Relaxation of `E` and `G`

Once `E` has learned recognition (Stage 1) and `G` has learned reverse-diffusion redrawing against frozen `E` (Stage 2), we **relax (unfreeze) both `E` and `G`** and train the whole closed-loop network end-to-end supervised:

1. **Why Unconstrained Co-Adaptation Causes "Judge Drift"**:
   - If `E` is unconditionally trained to assign the true label `y` to every redrawn canvas `X^{(1)}`—even on samples where `E(X^(0))` guessed the wrong class and `G` redrew the wrong digit—`E` bends its decision boundary to validate `G`'s wrong-class hallucinations (`Picket Fence` drops from `75.69%` to `72.03%`, and `15.9` steps of self-confirming drift lower accuracy on borderline natural digits).
2. **The Reality Anchor (`Stage 3A / 3B`)**:
   - During joint relaxation, `E` updates on real inputs (`X_clean` and `X^{(0)}`) and reinforces the redrawn canvas `X^{(1)}` **only when the initial recognition `E(X^{(0)})` matched the true label `y`**, while `G` is always corrected toward `y`.
   - This prevents `E` from flattering `G`'s errors while allowing `E` and `G` to co-align their valid basins—lifting Natural MNIST to **`98.07%` (`Step 0`) / `98.00%` (`Converged`)** (`73.00%` on the hardest `5%` tail), **Heavy Clutter** to **`87.57%`**, **Picket Fence** to **`75.02%`**, and **Center Blackout** to **`64.40%`**.

## Stage 4: Test-Time Diffusion Gated by Internal Loss & The Human in the Loop

At test time, both `E` and `G` remain **100% frozen** (zero retraining):

1. **Gate by Internal Loss (`L_internal`)**:
   - Inputs with low internal loss (`L_internal ≤ τ_gate`) exit immediately at Step 0 (`0` diffusion steps).
2. **Self-Terminating Redraw Loop Until Recognition Vectors Converge (`E(G(z*)) = z*`)**:
   - Uncertain inputs (`L_internal > τ_gate`) iterate $Z^{(k)} = E(X^{(k)}) \to X^{(k+1)} = \text{Redraw}(X^{(k)}, G(Z^{(k)})) \to Z^{(k+1)} = E(X^{(k+1)})$ in an unbounded `while` loop, accepting steps only while `L_internal` strictly decreases and halting when the recognized vectors converge (`‖z^{(k+1)} - z^{(k)}‖_2 ≤ ε`).
3. **Why the Human in the Loop Remains Essential**:
   - When an input is a genuine cross-class hybrid (such as a `9` written like an `a` or a `7` crossed like a `4`), an internal loop (`E ↔ G`) cannot invent external ground truth out of thin air. Persistent residual `L_internal` after convergence is the exact signal to defer the final judgment call to the human observer in the outer loop.
