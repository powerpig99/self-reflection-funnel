# 4-Stage Dual-Cut Implementation Plan (`Determination Funnel v2`)

This document specifies the four-stage architecture and training workflow for **Determination Funnel** under [The Dual Cut](https://not-a-toe.com/posts/the-dual-cut/), eliminating arbitrary power-of-two layer widths (`256 → 64 → 32`), pre-built layer cascades, and fixed test-time loop counts while demonstrating why a closed-loop system requires a reality anchor and a human in the loop at its outer boundary.

## Core Principles

1. **The Implied Command Is Fixed by the Problem Statement (`d_out = 10`)**:
   - A raw image carries only sensory payload (`V`). The instruction—*"recognize this picture as one of the 10 digits `0–9`"*—is decided by the problem statement itself (`d_out = 10`).
   - Because the command is fixed by the problem statement, `O(N²)` token-to-token self-attention across pixels is unnecessary; we propose that self-attention is a special case of **self-reflection** (extracting the low-dimensional abstraction of an input and applying it back onto the input's own high-dimensional payload).
   - **When Vector Slot Order Already Encodes `(x, y)`**: In a dense linear cut $u = \sum_{i=1}^{784} I_i w_i + b$ on a fixed `28×28` grid, each pixel index $i \in \{0, \dots, 783\}$ has its own dedicated weight row $w_i \in \mathbb{R}^{d_1}$, so $x_i I_i w_{i,x} + y_i I_i w_{i,y} + I_i w_{i,I} = I_i(x_i w_{i,x} + y_i w_{i,y} + w_{i,I})$ spans the exact same per-pixel linear subspace as $I_i$ alone. Passing the `784` ordered intensities directly eliminates the `3×` Layer-1 parameter expansion (`250,940 → 86,300` or `69,725` parameters) and removes coordinate-norm distortion under Center Blackout, whereas `[0, 255]³` `(x, y, I)` tuples remain useful when evaluating a single model across multiple spatial resolutions (`14×14`, `28×28`, `56×56` in `run_experiment.py`).

2. **Asymmetry Between Abstraction (`E`) and Expression (`G`)**:
   - **Abstraction (`E: X → Z`)** is many-to-one lossy compression: it strips away irrelevant local variance via `Linear + ReLU` cuts.
   - **Expression (`G: Z → X̂`)** is one-to-many detail expansion: it acts as a multi-stage reverse-diffusion denoiser that bridges large dimensional gaps with intermediate layers, iteratively redrawing ambiguous inputs until `E` recognizes them with minimal internal loss.

## Stage 1: Gradual Supervised Recognition Discovery (`E`), Then Freeze

Instead of guessing hidden layer counts or widths (`256, 64, 32`) or pre-building a multi-layer cascade upfront (which introduces Image lag), `E` starts directly at `d_in → 10` (`L = 0`) and refines one cut at a time from the **actual discovered width `d*_{l-1}`** on held-out `train[:4000]`:

1. **Midpoint Split Rule (`d_out = 10`)**:
   - At step `l ≥ 1`, split the remaining gap between the previously discovered bottom hidden width `d*_{l-1}` (with `d*_0 = d_in`) and `d_out = 10` at the multiplicative midpoint:
     $$D_{\text{init}}(l) = \text{round}\!\left(\sqrt{\frac{d^*_{l-1}}{10}} \cdot 10\right)$$
2. **Supervised Label Loss Across All Cuts**:
   - Train `E` on clean training digits (`train[4000:]`) using output cross-entropy at the terminal `10` layer (`CE(softmax(u_{10}), y)`) plus supervised label-group orthogonality (`latent_group_orthogonality_grad(z_l, y)`) at every hidden `ReLU` cut `z_l`.
3. **Zero-Floor Elastic Breathing (`min_width = 1` on Held-Out `train[:4000]`)**:
   - **Collapse on Sufficiency**: Fold collinear cones (`cos(z_j, z_k) ≥ 0.90`) into a single neuron by summing their scaled outgoing weights (`W_{l+1}[j, :] += (rms(z_k)/rms(z_j)) W_{l+1}[k, :]`), and prune spent scaffolding neurons whose downstream contribution energy `E_j = E[z_j²] · ‖W_{l+1}[j, :]‖₂²` falls below `5%` of the layer mean.
   - **Expand on Insufficiency**: If a layer has zero redundant or low-energy slack left (`100%` saturation), allocate `+round(sqrt(d_l / 10))` orthogonal neurons in the nullspace of `W_l`.
4. **Mathematical Fixed-Point Termination (`run_elastic_funnel.py` → `results/elastic_funnel_results.json`)**:
   - **On `2,352` coupled `(x, y, I)` tuples (`2,352 → 10`)**:
     - **`L = 0`**: `[2352, 10]` → `Head Acc: 92.59%`, `Internal Loss: 0.1492`, `Off-Diag Cos: 0.2231`. Next midpoint: `153`.
     - **`L = 1`**: `[2352, 153, 10]` → contracts to **`[2352, 121, 10]`** (`Head Acc: 97.56%`, `Internal Loss: 0.1402`, `Off-Diag Cos: 0.1433`). Next midpoint: `35`.
     - **`L = 2`**: `[2352, 153, 35, 10]` → contracts to **`[2352, 121, 12, 10]`** (`Head Acc: 97.41%`, `Proto Acc: 97.34%`, `Internal Loss: 0.0747`, `Off-Diag Cos: 0.0253`). Next midpoint: `11`.
     - **`L = 3`**: `[2352, 153, 35, 11, 10]` → breathes through `[2352, 105, 32, 13, 10]` and settles at **`[2352, 105, 32, 11, 10]`** (`Head Acc: 97.67%`, `Proto Acc: 97.58%`, `Internal Loss: 0.0595`, `Off-Diag Cos: 0.0191`, `250,940` parameters—`59.6%` smaller than the `621,226`-parameter `2,352 → 256 → 64 → 32 → 10` funnel). Next midpoint: `round(sqrt(11 / 10) * 10) = 10 == d_out`.
   - **On `784` pixel-order intensities (`784 → 10`)**:
     - **`L = 0`**: `[784, 10]` (`7,850` params) → `Head Acc: 92.54%`. Next midpoint: `89`.
     - **`L = 1`**: `[784, 89, 10]` → contracts to **`[784, 76, 10]`** (`60,430` params, `Head Acc: 97.64%`). Next midpoint: `28`.
     - **`L = 2`**: `[784, 89, 28, 10]` → contracts to **`[784, 84, 21, 10]`** (`67,945` params, `Head Acc: 97.50%`, `Proto Acc: 97.36%`). Next midpoint: `14`.
     - **`L = 3`**: `[784, 89, 28, 14, 10]` → contracts to **`[784, 85, 30, 10, 10]`** (**`69,725` params**, `Head Acc: 97.42%`, `Proto Acc: 97.33%`, `Internal Loss: 0.0601`, `Off-Diag Cos: 0.0171`), where the third hidden layer contracts from `14 → 11 → 10`, landing directly on the exact class count `10`.
5. **Freeze for Stage 2**:
   - Once `E` is discovered and anchored on supervised label loss, **all weights of `E` are frozen (`E_frozen`)** so `G` can learn against a stationary reference in Stage 2.

## Stage 2: Reverse-Diffusion Generator (`G`) Trained by Iterative Redraw Against Frozen `E`

1. **Bridging the Large Gap (`d_1 → d_mid → 784`)**:
   - Jumping from `d_1` (`105` or `85`) back to `784` pixels is a `7.5×–9.2×` gap.
   - Applying the same midpoint rule between `d_1` and `784` inserts `d_mid = round(sqrt(784 / d_1) * d_1)` (`287` for `d_1 = 105`; `258` for `d_1 = 85`), giving `G` a uniform `~2.7×–2.9×` reverse-diffusion expansion cascade:
     $$[10 + d_3] \longrightarrow d_2 \longrightarrow d_1 \longrightarrow d_{\text{mid}} \longrightarrow 784$$
2. **Intrinsic Internal Loss (`L_internal`) Inside `E`**:
   - For any input `X`, `E` computes its **internal loss `L_internal(X)`** without needing an external label:
     $$\mathcal{L}_{\text{internal}}(X) = 0.45\,\underbrace{\bigl(1 - (p_{(1)} - p_{(2)})\bigr)}_{\text{output cone indeterminacy}} + 0.55 \sum_{l=1}^{3} w_l \underbrace{\bigl(1 - \cos(z_l(X), \text{proto}^{(l)}_{\hat{y}})\bigr)}_{\text{internal cone residual}}$$
3. **Unrolled Redraw-and-Recognize Training on Clean Numbers**:
   - Pass `X^{(0)}` through **frozen `E`** to obtain the **actual recognition vectors** `Z^{(0)} = (p_{10}^{(0)}, z_3^{(0)}, z_2^{(0)}, z_1^{(0)})` and `L_internal^{(0)}`.
   - `G` uses `Z^{(0)}` to redraw `X^{(1)} = Redraw(X^{(0)}, G(Z^{(0)}), L_internal^{(0)})`.
   - Pass `X^{(1)}` through **frozen `E`** to get the new recognition result `Z^{(1)} = E(X^{(1)})`, measure the supervised recognition loss of `E(X^{(1)})` against the desired label `y`, and backpropagate analytical gradients through frozen `E` into `G` across unrolled steps (`k = 0 → 1 → 2`).

## Stage 3: Reality-Anchored vs. Single-Variable Unconstrained Joint Relaxation of `E` and `G`

Once `E` has learned recognition (Stage 1) and `G` has learned reverse-diffusion redrawing against frozen `E` (Stage 2), we **relax (unfreeze) both `E` and `G`** and train the whole closed-loop network end-to-end supervised (`run_diffusion_reflection.py` → `results/diffusion_reflection_results.json`):

1. **Single-Variable Reality-Anchor Ablation (`stage3_unconstrained_relaxed_E_G` & `stage3a_unconstrained_frozen_judge`)**:
   - Keeping every learning rate (`E.lr = 4e-4`, `G.lr = 1e-3`), clean update (`0.7`), corrupted update (`0.6`), `G` loss weights (`1.6`, `0.85`), and `X^{(1)}` weight (`0.35`) 100% identical to Stage 3 Anchored and removing *only* the `correct_0` gate (so `E` reinforces every redrawn canvas `X^{(1)}` unconditionally) isolates Judge Drift:
   - Without `correct_0`, the **hardest 5% natural tail drops `-2.60%` at Step 0** (`71.20% → 68.60%`) and **`-3.20%` (Self-Judged 3B: `69.60% → 66.40%`) / `-1.60%` (Frozen-Judged 3A: `70.00% → 68.40%`) converged**, **Picket Fence drops `-1.81%` (`75.78% → 73.97%` in 3B) / `-1.46%` (`75.84% → 74.38%` in 3A)**, and the self-judged loop takes **`11.63` steps instead of `8.76` steps** on the hardest 5% digits.
2. **True 1-to-1 Same-Size & Closest-Size Comparisons (`same_stream_baselines_22ep` & `intensity_only_784_funnels`)**:
   - **At `86,300` parameters (`784 → 105 → 32 → 11 → 10`)**: The pure `784` pixel-order Funnel (`86,300` params) beats the exact same-size `86,300`-parameter Standard MLP (`97.51%` Natural, `64.60%` Fence, `65.42%` Blackout, `83.30%` Clutter) by **`+0.37%` on Natural MNIST** (`97.88%` Step 0 / `97.83%` converged), **`+8.64%` on Picket Fence** (`73.24%`), and **`+3.74%` on Heavy Clutter** (`87.04%`), while improving Center Blackout by **`+2.55%` at Step 0** (`61.77% → 62.98%`) over the `2,352`-tuple funnel.
   - **At `69,725` parameters (`784 → 85 → 30 → 10 → 10`)**: The discovered `784`-Funnel (`69,725` params) beats the exact same-size `69,725`-parameter Standard MLP (`97.55%` Natural, `74.27%` Fence, `59.74%` Blackout, `82.92%` Clutter) on Natural MNIST (`97.68%` Step 0 / `97.64%` converged), Center Blackout (`+1.48%`: `61.22%` vs. `59.74%`), and Heavy Clutter (`+3.53%`: `86.45%` vs. `82.92%`).
   - **At `~220k–251k` parameters**: Headline **Stage 3A** (`250,940` params) ties the closest-in-size **`219,818`-parameter Standard MLP** (`784 → 256 → 64 → 32 → 10`, `98.03%` Natural) on Natural MNIST (`98.00%` Step 0 / `97.94%` converged), leads by **`+3.94%` on Picket Fence** (`75.84%` vs. `71.90%`) and **`+1.98%` on Heavy Clutter** (`87.72%` vs. `85.74%`), and trails on Center Blackout (`61.75%` vs. `66.38%`).

## Stage 4: Test-Time Diffusion Gated by Internal Loss & The Human in the Loop

At test time, both `E` and `G` remain **100% frozen** (zero retraining):

1. **Gate by Internal Loss (`L_internal`)**:
   - Inputs with low internal loss (`L_internal ≤ τ_gate`) exit immediately at Step 0 (`95%` of natural digits take `0.00` mean steps in Stage 3B and `0.16` in Stage 3A).
2. **Self-Terminating Redraw Loop Until Recognition Vectors Converge (`E(G(z*)) = z*`)**:
   - Uncertain inputs (`L_internal > τ_gate`) iterate $Z^{(k)} = E(X^{(k)}) \to X^{(k+1)} = \text{Redraw}(X^{(k)}, G(Z^{(k)})) \to Z^{(k+1)} = E(X^{(k+1)})$ in an unbounded `while` loop, accepting steps only while `L_internal` strictly decreases and halting when the recognized vectors converge (`‖z^{(k+1)} - z^{(k)}‖_2 ≤ ε`).
3. **Why the Human in the Loop Remains Essential**:
   - When an input is a genuine cross-class hybrid (such as a `9` written like an `a` or a `7` crossed like a `4`), an internal loop (`E ↔ G`) cannot invent external ground truth out of thin air. Persistent residual `L_internal` after convergence is the exact signal to defer the final judgment call to the human observer in the outer loop.
