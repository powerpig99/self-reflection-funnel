#!/usr/bin/env python3
"""
Complete 3-Stage Pipeline Rethinking Determination Funnel Under 'The Dual Cut':

Stage 1 — Gradual Supervised Recognition Discovery (Starting from `2352 -> 10`, Zero Floor):
  - Starts at `2352 -> 10` (`L = 0`) and refines the remaining gap `d*_{l-1} -> 10` via
    `D_init(l) = round(sqrt(d*_{l-1} / 10) * 10)` under supervised label loss:
      L = 0: [2352, 10]
      L = 1: [2352, 153, 10] -> breathes to [2352, 121, 10]
      L = 2: [2352, 153, 35, 10] -> breathes to [2352, 121, 12, 10]
      L = 3: [2352, 153, 35, 11, 10] -> breathes to [2352, 105, 32, 11, 10]
    At `d*_3 = 11`, `round(sqrt(11 / 10) * 10) = 10` (the midpoint fixed point: no further
    gap exists!). The `11` emerges by itself with `min_width = 1`.
  - All weights of `E` (`[2352, 105, 32, 11, 10]`) are then FROZEN permanently.

Stage 2 — Iterative Redraw-and-Recognize Training of `G` on Clean Numbers Against Frozen `E`:
  - `G` mirrors `E` (`10 + 11 -> 32 -> 105 -> 287 -> 784`), inserting the midpoint layer
    `287 = round(sqrt(784 / 105) * 105)` across the big gap between `105` and `784`.
  - Starting from clean digits `X_0` (and forward-diffused states `X^(0)` of `X_0`), `G`
    uses the ACTUAL recognition vectors `Z^(k) = E(X^(k))` (never the ground-truth label `y`)
    to redraw `X^(k+1) = Redraw(X^(k), G(Z^(k)))`, feeds `X^(k+1)` back into FROZEN `E` to
    get `Z^(k+1) = E(X^(k+1))`, and measures the loss of `E(X^(k+1))` against the desired
    label `y` across a 2-step unrolled redraw loop (`k = 0 -> 1 -> 2`).

Stage 3 — Test-Time Diffusion Gated by Internal Loss Until Recognized Vectors Converge:
  - Both `E` and `G` are frozen (zero retraining).
  - Test inputs with elevated Internal Loss `L_internal(X)` enter the iterative redraw loop
    (`Z^(k) = E(X^(k)) -> X^(k+1) = Redraw(X^(k), G(Z^(k))) -> Z^(k+1) = E(X^(k+1))`) and
    continue until `L_internal` stops decreasing and the recognized vectors converge
    (`||z^(k+1) - z^(k)||_2 <= eps`).
"""

import json
import time

import numpy as np
from PIL import Image, ImageDraw

from render_tables import (
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_RED,
    BG_CARD,
    BG_OUTER,
    TEXT_MAIN,
    TEXT_MUTED,
    load_fonts,
)
from run_elastic_funnel import (
    ElasticFunnel,
    compute_internal_loss,
    compute_layer_prototypes,
    midpoint_width,
    run_gradual_refinement_from_2352_to_10,
)
from run_experiment import (
    EXP_DIR,
    RESULTS_DIR,
    build_8bit_tuple_representation,
    latent_group_orthogonality_grad,
    load_mnist,
    normalize_rows,
    softmax,
)
from run_occlusion_loop import apply_occlusions

ASSETS_DIR = EXP_DIR / "assets"


class ReverseDiffusionGenerator:
    """
    Reverse-diffusion generator `G` mirroring frozen `E` (`[2352, d1, d2, d3, 10]`):
        (10 + d3) -> d2 (32) -> d1 (105) -> d_mid (287) -> 784
    where `d_mid = round(sqrt(784 / d1) * d1)` bridges the big gap between `d1` and `784`.
    Uses the actual recognition vectors `(probs_10, z3_n, z2_n, z1_n)` and internal loss
    `L_internal` from frozen `E` to redraw the picture.
    """

    def __init__(self, d1=105, d2=32, d3=11, d_out=10, out_pixels=784, lr=2.5e-3, seed=42):
        self.rng = np.random.default_rng(seed)
        self.d1 = int(d1)
        self.d2 = int(d2)
        self.d3 = int(d3)
        self.d_out = int(d_out)
        self.d_mid = midpoint_width(self.d1, out_pixels)  # round(sqrt(784 / 105) * 105) = 287
        self.out_pixels = int(out_pixels)
        self.lr = lr

        top_dim = self.d_out + self.d3  # 10 + 11 = 21

        # Stage 1: (10 + d3) + clarity-gated upward z2 (d2) -> d2
        in1 = top_dim + self.d2
        self.W1 = self.rng.normal(0.0, np.sqrt(2.0 / in1), size=(in1, self.d2)).astype(np.float32)
        self.b1 = np.zeros((1, self.d2), dtype=np.float32)

        # Stage 2: (10 + d3) + z2_hat (d2) + clarity-gated upward z1 (d1) -> d1
        in2 = top_dim + self.d2 + self.d1
        self.W2 = self.rng.normal(0.0, np.sqrt(2.0 / in2), size=(in2, self.d1)).astype(np.float32)
        self.b2 = np.zeros((1, self.d1), dtype=np.float32)

        # Stage 3: (10 + d3) + z2_hat (d2) + z1_hat (d1) -> d_mid (287)
        in3 = top_dim + self.d2 + self.d1
        self.W3 = self.rng.normal(0.0, np.sqrt(2.0 / in3), size=(in3, self.d_mid)).astype(np.float32)
        self.b3 = np.zeros((1, self.d_mid), dtype=np.float32)

        # Stage 4: d_mid (287) -> 784 pixels
        self.W4 = self.rng.normal(0.0, np.sqrt(2.0 / self.d_mid), size=(self.d_mid, self.out_pixels)).astype(np.float32)
        self.b4 = np.zeros((1, self.out_pixels), dtype=np.float32)

        self.params = [
            (self.W1, self.b1),
            (self.W2, self.b2),
            (self.W3, self.b3),
            (self.W4, self.b4),
        ]
        self.moments = [
            [np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)]
            for W, b in self.params
        ]
        self.t = 0

    @property
    def cascade_dims(self):
        return [self.d_out, self.d3, self.d2, self.d1, self.d_mid, self.out_pixels]

    def forward(self, probs_10: np.ndarray, z3_n: np.ndarray, z2_n: np.ndarray, z1_n: np.ndarray, int_loss: np.ndarray):
        """
        Redraw canvas expectation `i_hat` from the actual recognition vectors `(probs_10, z3_n, z2_n, z1_n)`.
        `int_loss` (internal loss inside frozen E) gates how strongly G trusts the bottom-up
        stroke payload `(z2_n, z1_n)` vs. the top-down recognized abstraction `(probs_10, z3_n)`.
        """
        clarity = np.clip(1.0 - 0.75 * int_loss, 0.20, 1.0)
        z2_gated = z2_n * clarity
        z1_gated = z1_n * clarity

        top_vec = np.concatenate([probs_10, z3_n], axis=-1)
        top_dim = top_vec.shape[1]

        inp1 = np.concatenate([top_vec, z2_gated], axis=-1)
        u1 = inp1 @ self.W1 + self.b1
        z2_hat = np.maximum(0.0, u1)
        z2_hat_n, z2_norms = normalize_rows(z2_hat)

        inp2 = np.concatenate([top_vec, z2_hat_n, z1_gated], axis=-1)
        u2 = inp2 @ self.W2 + self.b2
        z1_hat = np.maximum(0.0, u2)
        z1_hat_n, z1_norms = normalize_rows(z1_hat)

        inp3 = np.concatenate([top_vec, z2_hat_n, z1_hat_n], axis=-1)
        u3 = inp3 @ self.W3 + self.b3
        h_mid = np.maximum(0.0, u3)

        u4 = h_mid @ self.W4 + self.b4
        i_hat = np.clip(u4, 0.0, 1.0)

        cache = (top_dim, inp1, u1, z2_hat_n, z2_norms, inp2, u2, z1_hat_n, z1_norms, inp3, u3, h_mid, u4, i_hat)
        return i_hat, z2_hat_n, z1_hat_n, cache

    def backward_and_step(self, cache, di_hat: np.ndarray, dz2_hat_n_extra: np.ndarray = None, dz1_hat_n_extra: np.ndarray = None, weight_decay: float = 1e-4):
        (top_dim, inp1, u1, z2_hat_n, z2_norms, inp2, u2, z1_hat_n, z1_norms, inp3, u3, h_mid, u4, i_hat) = cache

        du4 = di_hat * ((u4 > 0.0) & (u4 < 1.0)).astype(np.float32)
        dW4 = h_mid.T @ du4 + weight_decay * self.W4
        db4 = np.sum(du4, axis=0, keepdims=True)
        dh_mid = du4 @ self.W4.T

        du3 = dh_mid * (u3 > 0.0).astype(np.float32)
        dW3 = inp3.T @ du3 + weight_decay * self.W3
        db3 = np.sum(du3, axis=0, keepdims=True)
        dinp3 = du3 @ self.W3.T

        dz2_from_3 = dinp3[:, top_dim:top_dim + self.d2]
        dz1_hat_n = dinp3[:, top_dim + self.d2:]
        if dz1_hat_n_extra is not None:
            dz1_hat_n = dz1_hat_n + dz1_hat_n_extra

        proj1 = np.sum(dz1_hat_n * z1_hat_n, axis=-1, keepdims=True)
        dz1_hat = (dz1_hat_n - z1_hat_n * proj1) / z1_norms

        du2 = dz1_hat * (u2 > 0.0).astype(np.float32)
        dW2 = inp2.T @ du2 + weight_decay * self.W2
        db2 = np.sum(du2, axis=0, keepdims=True)
        dinp2 = du2 @ self.W2.T

        dz2_hat_n = dz2_from_3 + dinp2[:, top_dim:top_dim + self.d2]
        if dz2_hat_n_extra is not None:
            dz2_hat_n = dz2_hat_n + dz2_hat_n_extra

        proj2 = np.sum(dz2_hat_n * z2_hat_n, axis=-1, keepdims=True)
        dz2_hat = (dz2_hat_n - z2_hat_n * proj2) / z2_norms

        du1 = dz2_hat * (u1 > 0.0).astype(np.float32)
        dW1 = inp1.T @ du1 + weight_decay * self.W1
        db1 = np.sum(du1, axis=0, keepdims=True)

        grads = [(dW1, db1), (dW2, db2), (dW3, db3), (dW4, db4)]
        self.t += 1
        b1_c = 1.0 - 0.9 ** self.t
        b2_c = 1.0 - 0.999 ** self.t
        for (W, b), (dW, db), mom in zip(self.params, grads, self.moments):
            mW, vW, mb, vb = mom
            mW[:] = 0.9 * mW + 0.1 * dW
            vW[:] = 0.999 * vW + 0.001 * (dW ** 2)
            W -= self.lr * (mW / b1_c) / (np.sqrt(vW / b2_c) + 1e-8)
            mb[:] = 0.9 * mb + 0.1 * db
            vb[:] = 0.999 * vb + 0.001 * (db ** 2)
            b -= self.lr * (mb / b1_c) / (np.sqrt(vb / b2_c) + 1e-8)


def forward_diffusion_on_clean(ib_clean: np.ndarray, x_flat: np.ndarray, y_flat: np.ndarray, rng: np.random.Generator):
    """
    Create intermediate diffusion states `X^(0)` from clean training digits `ib_clean`
    across random diffusion levels `t in [0.15, 0.85]`.
    """
    bsz = len(ib_clean)
    t_level = rng.uniform(0.15, 0.85, size=(bsz, 1)).astype(np.float32)
    im = ib_clean.reshape(bsz, 28, 28).copy()

    drop_mask = (rng.random((bsz, 28, 28)) > (t_level[:, :, None] * 0.45)).astype(np.float32)
    im *= drop_mask

    r0 = rng.integers(5, 19)
    h_span = int(rng.integers(3, 7))
    im[:, r0:r0 + h_span, :] *= (1.0 - t_level[:, :, None])

    c0 = rng.integers(6, 20)
    half = bsz // 2
    im[:half, 4:24, c0:c0 + 2] = np.maximum(im[:half, 4:24, c0:c0 + 2], t_level[:half, :, None] * 0.90)
    static_noise = (rng.random((bsz - half, 28, 28)) < (t_level[half:, :, None] * 0.22)).astype(np.float32) * 0.85
    im[half:] = np.clip(im[half:] + static_noise, 0.0, 1.0)

    i_t = im.reshape(bsz, 784)
    x_t = np.stack([x_flat * i_t, y_flat * i_t, i_t], axis=-1).reshape(bsz, 2352)
    return x_t.astype(np.float32), i_t.astype(np.float32)


def recognize_with_frozen_E(encoder: ElasticFunnel, x: np.ndarray, hidden_protos: list):
    """
    Pass `x` through FROZEN `E` to obtain the actual recognition vectors `(probs_10, z3_n, z2_n, z1_n)`
    and the intrinsic Internal Loss `int_loss`.
    """
    int_loss, probs_10, (_, post_acts) = compute_internal_loss(encoder, x, hidden_protos)
    z1_n, _ = normalize_rows(post_acts[1])
    z2_n, _ = normalize_rows(post_acts[2])
    z3_n, _ = normalize_rows(post_acts[3])
    return probs_10, z3_n, z2_n, z1_n, int_loss


def redraw_canvas_step(cur_i: np.ndarray, i_hat: np.ndarray, int_loss: np.ndarray):
    """
    Redraw step `I^(k) -> I^(k+1)` gated by the Internal Loss `int_loss` of frozen E:
    Returns both `next_i` and the partial derivative `d(next_i)/d(i_hat)` so gradients
    from `E(X^(k+1))` flow cleanly back into `G`!
    """
    i_hat_clean = np.clip((i_hat - 0.05) / 0.95, 0.0, 1.0)
    prod_agree = np.clip((cur_i * i_hat_clean) / 0.15, 0.0, 1.0)
    no_spike = np.clip(1.0 - np.maximum(0.0, cur_i - i_hat_clean - 0.10) / 0.35, 0.0, 1.0)
    agree = prod_agree * no_spike

    loss_gate = np.clip(0.30 + 1.40 * int_loss, 0.30, 1.0)
    replace_w = loss_gate * (0.85 * (1.0 - agree) + 0.20 * agree)
    next_i = np.clip((1.0 - replace_w) * cur_i + replace_w * i_hat_clean, 0.0, 1.0)
    return next_i, replace_w


def main():
    print("Loading MNIST and building [0, 255]^3 tuple representation...")
    train_img, train_lbl, test_img, test_lbl = load_mnist()
    x_train = build_8bit_tuple_representation(train_img, couple_background=True)
    x_test = build_8bit_tuple_representation(test_img, couple_background=True)
    i_train = (train_img.astype(np.float32) / 255.0).reshape(len(train_img), 784)
    i_test = (test_img.astype(np.float32) / 255.0).reshape(len(test_img), 784)

    y_grid = np.linspace(0.0, 1.0, 28, dtype=np.float32)[:, None]
    x_flat = np.broadcast_to(np.linspace(0.0, 1.0, 28, dtype=np.float32)[None, :], (28, 28)).reshape(1, 784)
    y_flat = np.broadcast_to(y_grid, (28, 28)).reshape(1, 784)

    # =========================================================================
    # STAGE 1: Discover E from `2352 -> 10` by Supervised Label Loss, Then Freeze E
    # =========================================================================
    stage1_results, opt_depth, encoder = run_gradual_refinement_from_2352_to_10(x_train, train_lbl, x_test, test_lbl)

    # Fine-tune the discovered E architecture (`[2352, 105, 32, 11, 10]`) on clean + mildly diffused
    # training digits so frozen E has a smooth recognition basin around clean digits, then FREEZE E!
    rng = np.random.default_rng(42)
    bsz = 256
    tr_x, tr_i, tr_y = x_train[4000:], i_train[4000:], train_lbl[4000:]
    ortho_w = [0.15, 0.15, 1.50]

    print(f"\n[Stage 1 Finalize & Freeze] Anchoring discovered E {encoder.dims} on supervised label loss before freezing...")
    for ep in range(4):
        perm = rng.permutation(len(tr_x))
        for i in range(0, len(tr_x), bsz):
            idx = perm[i:i + bsz]
            xb_c, ib_c, yb = tr_x[idx], tr_i[idx], tr_y[idx]
            one_hot = np.eye(10, dtype=np.float32)[yb]

            _, u_c, cache_c = encoder.forward(xb_c)
            du_c = (softmax(u_c) - one_hot) / len(xb_c)
            encoder.backward_and_step(cache_c, du_c, yb=yb, ortho_weights=ortho_w)

            xb_t, _ = forward_diffusion_on_clean(ib_c, x_flat, y_flat, rng)
            _, u_t, cache_t = encoder.forward(xb_t)
            du_t = (softmax(u_t) - one_hot) / len(xb_c)
            encoder.backward_and_step(cache_t, du_t * 0.5, yb=yb, ortho_weights=ortho_w)

    # Compute frozen E's clean class prototypes across all hidden layers
    hidden_protos = compute_layer_prototypes(encoder, tr_x[:10000], tr_y[:10000], n_classes=10)
    _, u_clean_test, _ = encoder.forward(x_test)
    clean_step0_acc = float(np.mean(np.argmax(u_clean_test, axis=-1) == test_lbl) * 100.0)
    print(f"   FROZEN E {encoder.dims} ({encoder.param_count:,} params) -> Clean Test Step-0 Accuracy: {clean_step0_acc:.2f}%")

    # =========================================================================
    # STAGE 2: Train Reverse-Diffusion Generator G Against FROZEN E by Iterative Redraw
    # =========================================================================
    d1_star, d2_star, d3_star = encoder.hidden_dims
    decoder = ReverseDiffusionGenerator(d1=d1_star, d2=d2_star, d3=d3_star, d_out=10, out_pixels=784, lr=2.5e-3, seed=42)
    print(
        f"\n=== Stage 2: Training Reverse-Diffusion Generator G ({decoder.cascade_dims}) Against FROZEN E ===\n"
        f"  Loop: X^(k) -> Z^(k) = E(X^(k)) -> X^(k+1) = Redraw(X^(k), G(Z^(k))) -> Measure E(X^(k+1)) Loss vs. Label y"
    )
    t_g = time.time()

    for ep in range(10):
        perm = rng.permutation(len(tr_x))
        for i in range(0, len(tr_x), bsz):
            idx = perm[i:i + bsz]
            xb_c, ib_c, yb = tr_x[idx], tr_i[idx], tr_y[idx]

            # Clean reference features from frozen E(X_0)
            _, _, z2_clean_n, z1_clean_n, _ = recognize_with_frozen_E(encoder, xb_c, hidden_protos)

            # Start from intermediate diffusion state X^(0) of clean number X_0
            xb_0, ib_0 = forward_diffusion_on_clean(ib_c, x_flat, y_flat, rng)

            # --- Iteration k = 0: Recognize X^(0) -> Redraw X^(1) -> Recognize X^(1) & Measure Supervised Loss ---
            probs_0, z3_0, z2_0, z1_0, int_loss_0 = recognize_with_frozen_E(encoder, xb_0, hidden_protos)
            i_hat_0, z2_hat_0, z1_hat_0, cache_g_0 = decoder.forward(probs_0, z3_0, z2_0, z1_0, int_loss_0)
            ib_1, replace_w_0 = redraw_canvas_step(ib_0, i_hat_0, int_loss_0)
            xb_1 = np.stack([x_flat * ib_1, y_flat * ib_1, ib_1], axis=-1).reshape(len(xb_c), 2352)

            # Recognize redrawn X^(1) with FROZEN E and compute gradient of supervised label loss w.r.t. X^(1)
            dx_1, _, _ = encoder.input_gradient_from_supervised_loss(xb_1, yb, ortho_weights=ortho_w)
            dx_1_3d = dx_1.reshape(len(xb_c), 784, 3)
            di_from_E_1 = (dx_1_3d[:, :, 0] * x_flat + dx_1_3d[:, :, 1] * y_flat + dx_1_3d[:, :, 2]) * replace_w_0

            di_g0 = (i_hat_0 - ib_c) * (1.8 / len(xb_c)) + di_from_E_1 * 0.65
            dz2_g0 = (z2_hat_0 - z2_clean_n) * (0.5 / len(xb_c))
            dz1_g0 = (z1_hat_0 - z1_clean_n) * (0.5 / len(xb_c))
            decoder.backward_and_step(cache_g_0, di_g0, dz2_hat_n_extra=dz2_g0, dz1_hat_n_extra=dz1_g0)

            # --- Iteration k = 1: Use NEW Recognition Vector Z^(1) = E(X^(1)) to Redraw X^(2) -> Recognize X^(2) ---
            probs_1, z3_1, z2_1, z1_1, int_loss_1 = recognize_with_frozen_E(encoder, xb_1, hidden_protos)
            i_hat_1, z2_hat_1, z1_hat_1, cache_g_1 = decoder.forward(probs_1, z3_1, z2_1, z1_1, int_loss_1)
            ib_2, replace_w_1 = redraw_canvas_step(ib_1, i_hat_1, int_loss_1)
            xb_2 = np.stack([x_flat * ib_2, y_flat * ib_2, ib_2], axis=-1).reshape(len(xb_c), 2352)

            dx_2, _, _ = encoder.input_gradient_from_supervised_loss(xb_2, yb, ortho_weights=ortho_w)
            dx_2_3d = dx_2.reshape(len(xb_c), 784, 3)
            di_from_E_2 = (dx_2_3d[:, :, 0] * x_flat + dx_2_3d[:, :, 1] * y_flat + dx_2_3d[:, :, 2]) * replace_w_1

            di_g1 = (i_hat_1 - ib_c) * (1.8 / len(xb_c)) + di_from_E_2 * 0.65
            dz2_g1 = (z2_hat_1 - z2_clean_n) * (0.5 / len(xb_c))
            dz1_g1 = (z1_hat_1 - z1_clean_n) * (0.5 / len(xb_c))
            decoder.backward_and_step(cache_g_1, di_g1, dz2_hat_n_extra=dz2_g1, dz1_hat_n_extra=dz1_g1)

    print(f"   Trained G in {time.time() - t_g:.2f}s (E weights remained 100% frozen; Clean Test Acc = {clean_step0_acc:.2f}%).")

    # =========================================================================
    # STAGE 3: Test-Time Diffusion Gated by Internal Loss Until Recognized Vectors Converge
    # =========================================================================
    print("\n=== Stage 3: Test-Time Diffusion Reflection Gated by Internal Loss (Zero Retraining) ===")

    def run_convergence_loop(x_in: np.ndarray, i_in: np.ndarray, int_loss_gate: float = 0.14, conv_eps: float = 5e-4, record_snaps: bool = False):
        """
        Run test-time diffusion steps ONLY on inputs where `L_internal > int_loss_gate`,
        iterating `Z^(k) = E(X^(k)) -> X^(k+1) = Redraw(X^(k), G(Z^(k))) -> Z^(k+1) = E(X^(k+1))`
        until `L_internal` stops dropping or the recognized vectors converge (`||z3^(k+1) - z3^(k)||_2 <= conv_eps`).
        """
        cur_x = x_in.copy()
        cur_i = i_in.copy()
        probs_0, z3_0, z2_0, z1_0, int_loss_0 = recognize_with_frozen_E(encoder, cur_x, hidden_protos)

        best_i = cur_i.copy()
        best_probs = probs_0.copy()
        best_z3 = z3_0.copy()
        best_int_loss = int_loss_0[:, 0].copy()

        active = best_int_loss > int_loss_gate
        steps_taken = np.zeros(len(x_in), dtype=np.int32)

        snaps_i = [best_i.copy()] if record_snaps else None
        snaps_loss = [best_int_loss.copy()] if record_snaps else None
        snaps_preds = [np.argmax(best_probs, axis=-1)] if record_snaps else None

        step = 0
        while np.any(active):
            step += 1
            act_idx = np.where(active)[0]
            probs_a, z3_a, z2_a, z1_a, loss_a = recognize_with_frozen_E(encoder, cur_x[act_idx], hidden_protos)

            # Use the evolving recognized vector to redraw the picture
            i_hat_a, _, _, _ = decoder.forward(best_probs[act_idx], best_z3[act_idx], z2_a, z1_a, loss_a)
            next_i_a, _ = redraw_canvas_step(cur_i[act_idx], i_hat_a, loss_a)

            gamma_state = np.clip(0.55 * loss_a, 0.12, 0.45)
            cand_i = (1.0 - gamma_state) * cur_i[act_idx] + gamma_state * next_i_a
            cand_x = np.stack([x_flat * cand_i, y_flat * cand_i, cand_i], axis=-1).reshape(len(act_idx), 2352)

            # Recognize the redrawn picture with frozen E
            cand_probs, cand_z3, _, _, cand_loss_col = recognize_with_frozen_E(encoder, cand_x, hidden_protos)
            cand_loss = cand_loss_col[:, 0]

            beta_state = np.clip(0.55 * loss_a, 0.15, 0.48)
            blended_probs = (1.0 - beta_state) * best_probs[act_idx] + beta_state * cand_probs
            blended_probs = blended_probs / np.sum(blended_probs, axis=-1, keepdims=True)
            blended_z3, _ = normalize_rows((1.0 - beta_state) * best_z3[act_idx] + beta_state * cand_z3)

            # Measure recognition vector shift ||z^(k+1) - z^(k)||_2 and internal loss drop
            vec_delta = np.sqrt(np.sum((blended_z3 - best_z3[act_idx]) ** 2, axis=-1))
            loss_improved = cand_loss < (best_int_loss[act_idx] - 2e-4)
            still_moving = vec_delta > conv_eps
            accept = loss_improved & still_moving

            imp_global = act_idx[accept]
            halt_global = act_idx[~accept]

            best_i[imp_global] = cand_i[accept]
            cur_i[imp_global] = cand_i[accept]
            cur_x[imp_global] = cand_x[accept]
            best_probs[imp_global] = blended_probs[accept]
            best_z3[imp_global] = blended_z3[accept]
            best_int_loss[imp_global] = cand_loss[accept]
            steps_taken[imp_global] += 1

            active[halt_global] = False
            active[imp_global[best_int_loss[imp_global] <= int_loss_gate]] = False

            if record_snaps and step in (1, 2, 4):
                snaps_i.append(best_i.copy())
                snaps_loss.append(best_int_loss.copy())
                snaps_preds.append(np.argmax(best_probs, axis=-1))

        preds_final = np.argmax(best_probs, axis=-1)
        return preds_final, best_i, best_int_loss, steps_taken, (snaps_i, snaps_loss, snaps_preds)

    # 3A. Evaluate on Natural 10,000 MNIST Test Set
    probs_nat0, _, _, _, loss_nat0_col = recognize_with_frozen_E(encoder, x_test, hidden_protos)
    loss_nat0 = loss_nat0_col[:, 0]
    preds_nat0 = np.argmax(probs_nat0, axis=-1)
    hard_mask = loss_nat0 >= np.percentile(loss_nat0, 95)
    conf_mask = ~hard_mask

    preds_nat_f, _, loss_nat_f, steps_nat, (snaps_i_nat, snaps_loss_nat, snaps_preds_nat) = run_convergence_loop(
        x_test, i_test, int_loss_gate=0.14, record_snaps=True
    )

    acc_nat0_all = float(np.mean(preds_nat0 == test_lbl) * 100.0)
    acc_natf_all = float(np.mean(preds_nat_f == test_lbl) * 100.0)
    acc_nat0_conf = float(np.mean(preds_nat0[conf_mask] == test_lbl[conf_mask]) * 100.0)
    acc_natf_conf = float(np.mean(preds_nat_f[conf_mask] == test_lbl[conf_mask]) * 100.0)
    acc_nat0_hard = float(np.mean(preds_nat0[hard_mask] == test_lbl[hard_mask]) * 100.0)
    acc_natf_hard = float(np.mean(preds_nat_f[hard_mask] == test_lbl[hard_mask]) * 100.0)

    print(
        f"  [Natural MNIST 10,000 Test] All 10k: {acc_nat0_all:.2f}% -> {acc_natf_all:.2f}% | "
        f"Low-Loss 95%: {acc_nat0_conf:.2f}% -> {acc_natf_conf:.2f}% (Mean Steps: {np.mean(steps_nat[conf_mask]):.2f}) | "
        f"High-Loss 5%: {acc_nat0_hard:.2f}% -> {acc_natf_hard:.2f}% ({acc_natf_hard - acc_nat0_hard:+.2f}%, "
        f"Mean Steps: {np.mean(steps_nat[hard_mask]):.2f}, IntLoss: {np.mean(loss_nat0[hard_mask]):.3f} -> {np.mean(loss_nat_f[hard_mask]):.3f})"
    )

    # 3B. Evaluate on Occluded Digits
    occ_summary = {}
    for mode in ["picket_fence", "center_blackout", "heavy_clutter"]:
        occ_u8 = apply_occlusions(test_img, mode=mode, seed=42)
        x_occ = build_8bit_tuple_representation(occ_u8, couple_background=True)
        i_occ = (occ_u8.astype(np.float32) / 255.0).reshape(len(test_img), 784)

        probs_o0, _, _, _, loss_o0_col = recognize_with_frozen_E(encoder, x_occ, hidden_protos)
        loss_o0 = loss_o0_col[:, 0]
        acc_o0 = float(np.mean(np.argmax(probs_o0, axis=-1) == test_lbl) * 100.0)
        mse_o0 = float(np.mean((i_occ - i_test) ** 2))

        preds_of, best_i_o, loss_of, steps_o, _ = run_convergence_loop(x_occ, i_occ, int_loss_gate=0.12, record_snaps=False)
        acc_of = float(np.mean(preds_of == test_lbl) * 100.0)
        mse_of = float(np.mean((best_i_o - i_test) ** 2))
        mse_drop_pct = (1.0 - mse_of / max(1e-6, mse_o0)) * 100.0

        print(
            f"  [{mode.upper():15s}] Step 0: {acc_o0:6.2f}% -> Converged: {acc_of:6.2f}% ({acc_of - acc_o0:+5.2f}% gain) | "
            f"IntLoss: {np.mean(loss_o0):.3f} -> {np.mean(loss_of):.3f} | "
            f"Steps: mean={np.mean(steps_o):.1f}, med={int(np.percentile(steps_o, 50))}, p90={int(np.percentile(steps_o, 90))}, max={int(np.max(steps_o))} | "
            f"Canvas MSE: -{mse_drop_pct:.1f}%"
        )
        occ_summary[mode] = {
            "step0_acc_pct": round(acc_o0, 2),
            "converged_acc_pct": round(acc_of, 2),
            "loop_gain_pct": round(acc_of - acc_o0, 2),
            "initial_internal_loss": round(float(np.mean(loss_o0)), 4),
            "converged_internal_loss": round(float(np.mean(loss_of)), 4),
            "mean_steps": round(float(np.mean(steps_o)), 2),
            "median_steps": int(np.percentile(steps_o, 50)),
            "p90_steps": int(np.percentile(steps_o, 90)),
            "max_steps": int(np.max(steps_o)),
            "initial_mse": round(mse_o0, 4),
            "converged_mse": round(mse_of, 4),
            "mse_reduction_pct": round(mse_drop_pct, 1),
        }

    # Render updated visual progression grid on rescued natural digits
    rescued_mask = (preds_nat0 != test_lbl) & (preds_nat_f == test_lbl)
    rescued_indices = np.where(rescued_mask)[0]
    rescued_indices = rescued_indices[np.argsort(-loss_nat0[rescued_indices])]
    chosen_rescued = rescued_indices[:10]

    while len(snaps_i_nat) < 4:
        snaps_i_nat.append(snaps_i_nat[-1])
        snaps_loss_nat.append(snaps_loss_nat[-1])
        snaps_preds_nat.append(snaps_preds_nat[-1])

    fonts = load_fonts()
    cell_w, cell_h = 148, 168
    cols = len(chosen_rescued)
    margin_x, margin_y = 40, 36
    header_h = 92
    row_labels = [
        "Step 0 (Initial Input X^(0))",
        "Step 1 (Redrawn X^(1))",
        "Step 2 (Redrawn X^(2))",
        "Converged Fixed Point X*",
    ]

    img_w = margin_x * 2 + 240 + max(1, cols) * cell_w
    img_h = margin_y * 2 + header_h + 4 * cell_h + 20
    canvas = Image.new("RGB", (img_w, img_h), BG_OUTER)
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (margin_x, margin_y),
        f"3-Stage Dual-Cut Pipeline: Frozen E {encoder.dims} <-> Generator G {decoder.cascade_dims}",
        fill=TEXT_MAIN,
        font=fonts["title"],
    )
    draw.text(
        (margin_x, margin_y + 44),
        "Gated by Internal Loss inside Frozen E; redrawn from actual recognition vectors until recognized vectors converge",
        fill=TEXT_MUTED,
        font=fonts["subtitle"],
    )

    y_base = margin_y + header_h
    for r_idx in range(4):
        ry = y_base + r_idx * cell_h
        lbl_col = ACCENT_RED if r_idx == 0 else (ACCENT_GREEN if r_idx == 3 else ACCENT_BLUE)
        draw.text((margin_x, ry + 65), row_labels[r_idx], fill=lbl_col, font=fonts["footer"])
        for c_idx, sample_idx in enumerate(chosen_rescued):
            cx = margin_x + 240 + c_idx * cell_w
            pred_c = int(snaps_preds_nat[r_idx][sample_idx])
            true_c = int(test_lbl[sample_idx])
            l_val = float(snaps_loss_nat[r_idx][sample_idx])
            border_c = ACCENT_GREEN if pred_c == true_c else ACCENT_RED
            draw.rectangle([cx + 4, ry + 4, cx + cell_w - 4, ry + cell_h - 4], fill=BG_CARD, outline=border_c, width=2)
            patch_arr = (np.clip(snaps_i_nat[r_idx][sample_idx].reshape(28, 28), 0.0, 1.0) * 255.0).astype(np.uint8)
            patch_im = Image.fromarray(patch_arr, mode="L").resize((96, 96), Image.Resampling.NEAREST)
            canvas.paste(patch_im.convert("RGB"), (cx + (cell_w - 96) // 2, ry + 10))
            draw.text((cx + 12, ry + 112), f"True:{true_c} Pred:{pred_c}", fill=border_c, font=fonts["footer"])
            draw.text((cx + 12, ry + 136), f"IntLoss: {l_val:.3f}", fill=TEXT_MUTED, font=fonts["footer"])

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    grid_path = ASSETS_DIR / "diffusion_reflection_grid.png"
    canvas.save(grid_path)

    out_json = RESULTS_DIR / "diffusion_reflection_results.json"
    summary = {
        "stage1_frozen_encoder_dims": encoder.dims,
        "stage1_frozen_encoder_params": encoder.param_count,
        "stage1_gradual_discovery": stage1_results,
        "stage2_generator_cascade_dims": decoder.cascade_dims,
        "stage3_natural_mnist": {
            "all_10k_step0_acc_pct": round(acc_nat0_all, 2),
            "all_10k_converged_acc_pct": round(acc_natf_all, 2),
            "low_loss_95pct_step0_acc_pct": round(acc_nat0_conf, 2),
            "low_loss_95pct_converged_acc_pct": round(acc_natf_conf, 2),
            "low_loss_95pct_mean_steps": round(float(np.mean(steps_nat[conf_mask])), 2),
            "high_loss_5pct_step0_acc_pct": round(acc_nat0_hard, 2),
            "high_loss_5pct_converged_acc_pct": round(acc_natf_hard, 2),
            "high_loss_5pct_mean_steps": round(float(np.mean(steps_nat[hard_mask])), 2),
            "rescued_natural_digits_count": int(np.sum(rescued_mask)),
        },
        "stage3_occlusions_converged": occ_summary,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved visual grid to {grid_path} and complete 3-stage metrics to {out_json}")


if __name__ == "__main__":
    main()
