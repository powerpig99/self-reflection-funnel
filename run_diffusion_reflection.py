#!/usr/bin/env python3
"""
Complete 4-Stage Pipeline Rethinking Determination Funnel Under 'The Dual Cut':

Stage 1 — Gradual Supervised Recognition Discovery (Starting from `2352 -> 10`, Zero Floor):
  - Starts at `2352 -> 10` (`L = 0`) and refines the remaining gap `d*_{l-1} -> 10` via
    `D_init(l) = round(sqrt(d*_{l-1} / 10) * 10)` under supervised label loss:
      L = 0: [2352, 10]
      L = 1: [2352, 153, 10] -> breathes to [2352, 121, 10]
      L = 2: [2352, 153, 35, 10] -> breathes to [2352, 121, 12, 10]
      L = 3: [2352, 153, 35, 11, 10] -> breathes to [2352, 105, 32, 11, 10]
    At `d*_3 = 11` (initialized by `round(sqrt(12 / 10) * 10) = 11`, which expands to `13`
    and returns to `11`), `round(sqrt(11 / 10) * 10) = 10` (the midpoint fixed point).
  - All weights of `E` (`[2352, 105, 32, 11, 10]`) are then FROZEN for Stage 2.

Stage 2 — Iterative Redraw-and-Recognize Training of `G` on Clean Numbers Against Frozen `E`:
  - `G` mirrors `E` (`10 + 11 -> 32 -> 105 -> 287 -> 784`), inserting the midpoint layer
    `287 = round(sqrt(784 / 105) * 105)` across the big gap between `105` and `784`.
  - Keeping `E` frozen prevents representation collapse while `G` learns to redraw from `E`'s
    actual recognition vectors `Z^(k) = E(X^(k))` (`k = 0 -> 1 -> 2`).

Stage 3 — Reality-Anchored vs. Unconstrained Joint Supervised Relaxation of `E` and `G`:
  - Once `E` and `G` are warmed up, we relax both `E` and `G` weights to co-adapt across
    the closed-loop self-reflection trajectory.
  - Crucial Reality Anchor (preventing 'Judge Drift' / Echo-Chamber Delusion):
    If `E` unconditionally learns to assign the true label `y` to every redrawn canvas `X^(1)`—
    even when `E(X^(0))` guessed the wrong class and `G` redrew the wrong digit (`Stage 3 Unconstrained`)—
    `E` bends its boundaries to flatter `G`'s wrong-class hallucinations. By anchoring `E` to reality
    (`X_clean` and `X^(0)`) and updating `E` on `X^(1)` only when `E(X^(0))` matched `y` (while `G`
    is always corrected toward `y`), `E` reaches `98.00%` on Natural MNIST (`97.94%` converged in 3A),
    `75.84%` on Picket Fence, and `87.72%` on Heavy Clutter, outperforming a same-width Standard MLP
    (`784 -> 105 -> 32 -> 11 -> 10`) trained on the same corruption stream for 22 epochs (`97.70%`
    Natural, `66.90%` Fence, `83.46%` Clutter) and matching a 2.5x larger `256 -> 64 -> 32` network
    (`621,226` params) with `59.6%` fewer parameters (`250,940`).

Stage 4 — Test-Time Diffusion Gated by Internal Loss Until Recognized Vectors Converge:
  - Evaluates Stage 2 (`Frozen E + G`), Stage 3 Unconstrained (`Relaxed E + G`), Stage 3B
    (`Reality-Anchored E + G`, Self-Judged), and Stage 3A (`Reality-Anchored E + G`, Judged by `E_frozen`),
    plus Same-Stream Standard MLP baselines (`105->32->11` and `256->64->32`).
"""

import copy
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
    AdamMLP,
    build_8bit_tuple_representation,
    load_mnist,
    normalize_rows,
    softmax,
)
from run_occlusion_loop import apply_occlusions

ASSETS_DIR = EXP_DIR / "assets"


class ReverseDiffusionGenerator:
    """
    Reverse-diffusion generator `G` mirroring `E` (`[2352, d1, d2, d3, 10]`):
        (10 + d3) -> d2 (32) -> d1 (105) -> d_mid (287) -> 784
    where `d_mid = round(sqrt(784 / d1) * d1)` bridges the big gap between `d1` and `784`.
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

        in1 = top_dim + self.d2
        self.W1 = self.rng.normal(0.0, np.sqrt(2.0 / in1), size=(in1, self.d2)).astype(np.float32)
        self.b1 = np.zeros((1, self.d2), dtype=np.float32)

        in2 = top_dim + self.d2 + self.d1
        self.W2 = self.rng.normal(0.0, np.sqrt(2.0 / in2), size=(in2, self.d1)).astype(np.float32)
        self.b2 = np.zeros((1, self.d1), dtype=np.float32)

        in3 = top_dim + self.d2 + self.d1
        self.W3 = self.rng.normal(0.0, np.sqrt(2.0 / in3), size=(in3, self.d_mid)).astype(np.float32)
        self.b3 = np.zeros((1, self.d_mid), dtype=np.float32)

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


def recognize_with_E(encoder: ElasticFunnel, x: np.ndarray, hidden_protos: list):
    int_loss, probs_10, (pre_acts, post_acts) = compute_internal_loss(encoder, x, hidden_protos)
    z1_n, _ = normalize_rows(post_acts[1])
    z2_n, _ = normalize_rows(post_acts[2])
    z3_n, _ = normalize_rows(post_acts[3])
    return probs_10, z3_n, z2_n, z1_n, int_loss, (pre_acts, post_acts)


def redraw_canvas_step(cur_i: np.ndarray, i_hat: np.ndarray, int_loss: np.ndarray):
    i_hat_clean = np.clip((i_hat - 0.05) / 0.95, 0.0, 1.0)
    prod_agree = np.clip((cur_i * i_hat_clean) / 0.15, 0.0, 1.0)
    no_spike = np.clip(1.0 - np.maximum(0.0, cur_i - i_hat_clean - 0.10) / 0.35, 0.0, 1.0)
    agree = prod_agree * no_spike

    loss_gate = np.clip(0.30 + 1.40 * int_loss, 0.30, 1.0)
    replace_w = loss_gate * (0.85 * (1.0 - agree) + 0.20 * agree)
    next_i = np.clip((1.0 - replace_w) * cur_i + replace_w * i_hat_clean, 0.0, 1.0)
    return next_i, replace_w


def train_and_evaluate_same_stream_mlp(
    label: str,
    layer_dims: list,
    tr_i: np.ndarray,
    tr_y: np.ndarray,
    i_test: np.ndarray,
    test_img: np.ndarray,
    test_lbl: np.ndarray,
    x_flat: np.ndarray,
    y_flat: np.ndarray,
):
    """
    Train a Standard Cross-Entropy MLP (`AdamMLP` from `run_experiment.py`: `I`-only input,
    output cross-entropy only, zero coordinate tuples, zero orthogonality penalty, zero
    cone-column regularizer, zero generator/loop) on the same 22-epoch training exposure
    as `E` (12 clean epochs + 10 clean & `forward_diffusion_on_clean` epochs on `train[4000:]`).
    """
    t0 = time.time()
    mlp = AdamMLP(dims=tuple(layer_dims), lr=1.5e-3, seed=42)
    rng = np.random.default_rng(42)
    bsz = 256

    # Phase 1: 12 clean epochs (matching Stage 1 L=3 training)
    for _ in range(12):
        perm = rng.permutation(len(tr_i))
        for i in range(0, len(tr_i), bsz):
            idx = perm[i:i + bsz]
            ib_c, yb = tr_i[idx], tr_y[idx]
            one_hot = np.eye(10, dtype=np.float32)[yb]
            _, u_c, _, cache_c = mlp.forward(ib_c)
            du_c = (softmax(u_c) - one_hot) / len(ib_c)
            mlp.backward_and_step(cache_c, du_c)

    # Phase 2: 10 clean + diffusion epochs (matching Stage 1 Finalize [4] + Stage 3 [6] = 10 epochs)
    for _ in range(10):
        perm = rng.permutation(len(tr_i))
        for i in range(0, len(tr_i), bsz):
            idx = perm[i:i + bsz]
            ib_c, yb = tr_i[idx], tr_y[idx]
            one_hot = np.eye(10, dtype=np.float32)[yb]

            _, u_c, _, cache_c = mlp.forward(ib_c)
            du_c = (softmax(u_c) - one_hot) / len(ib_c)
            mlp.backward_and_step(cache_c, du_c)

            _, ib_t = forward_diffusion_on_clean(ib_c, x_flat, y_flat, rng)
            _, u_t, _, cache_t = mlp.forward(ib_t)
            du_t = (softmax(u_t) - one_hot) / len(ib_c)
            mlp.backward_and_step(cache_t, du_t * 0.5)

    _, u_nat, _, _ = mlp.forward(i_test)
    nat_acc = round(float(np.mean(np.argmax(u_nat, axis=-1) == test_lbl) * 100.0), 2)
    occ_accs = {}
    for mode in ["picket_fence", "center_blackout", "heavy_clutter"]:
        occ_u8 = apply_occlusions(test_img, mode=mode, seed=42)
        i_occ = (occ_u8.astype(np.float32) / 255.0).reshape(len(test_img), 784)
        _, u_occ, _, _ = mlp.forward(i_occ)
        occ_accs[mode] = round(float(np.mean(np.argmax(u_occ, axis=-1) == test_lbl) * 100.0), 2)

    param_count = int(sum(w.size + b.size for w, b in zip(mlp.W, mlp.b)))
    elapsed = round(time.time() - t0, 2)
    print(
        f"  [{label}] Dims: {list(mlp.dims)} ({param_count:,} params, {elapsed:.1f}s) -> "
        f"Natural: {nat_acc:.2f}% | Fence: {occ_accs['picket_fence']:.2f}% | "
        f"Blackout: {occ_accs['center_blackout']:.2f}% | Clutter: {occ_accs['heavy_clutter']:.2f}%"
    )
    return {
        "dims": list(mlp.dims),
        "params": param_count,
        "natural_mnist_acc_pct": nat_acc,
        "picket_fence_acc_pct": occ_accs["picket_fence"],
        "center_blackout_acc_pct": occ_accs["center_blackout"],
        "heavy_clutter_acc_pct": occ_accs["heavy_clutter"],
        "seconds": elapsed,
    }


def evaluate_pipeline_stage(
    stage_label: str,
    enc_step: ElasticFunnel,
    dec_step: ReverseDiffusionGenerator,
    protos_step: list,
    enc_judge: ElasticFunnel,
    protos_judge: list,
    x_test: np.ndarray,
    i_test: np.ndarray,
    test_img: np.ndarray,
    test_lbl: np.ndarray,
    x_flat: np.ndarray,
    y_flat: np.ndarray,
    record_nat_snaps: bool = False,
):
    print(f"\n--- Evaluating {stage_label} ---")

    def run_convergence_loop(x_in: np.ndarray, i_in: np.ndarray, int_loss_gate: float = 0.14, conv_eps: float = 5e-4, record_snaps: bool = False):
        cur_x = x_in.copy()
        cur_i = i_in.copy()
        probs_0, z3_0, _, _, _, _ = recognize_with_E(enc_step, cur_x, protos_step)
        _, _, _, _, j_loss_0, _ = recognize_with_E(enc_judge, cur_x, protos_judge)

        best_i = cur_i.copy()
        best_probs = probs_0.copy()
        best_z3 = z3_0.copy()
        best_int_loss = j_loss_0[:, 0].copy()

        active = best_int_loss > int_loss_gate
        steps_taken = np.zeros(len(x_in), dtype=np.int32)

        snaps_i = [best_i.copy()] if record_snaps else None
        snaps_loss = [best_int_loss.copy()] if record_snaps else None
        snaps_preds = [np.argmax(best_probs, axis=-1)] if record_snaps else None

        step = 0
        while np.any(active):
            step += 1
            act_idx = np.where(active)[0]
            probs_a, z3_a, z2_a, z1_a, s_loss_a, _ = recognize_with_E(enc_step, cur_x[act_idx], protos_step)

            i_hat_a, _, _, _ = dec_step.forward(best_probs[act_idx], best_z3[act_idx], z2_a, z1_a, s_loss_a)
            next_i_a, _ = redraw_canvas_step(cur_i[act_idx], i_hat_a, s_loss_a)

            gamma_state = np.clip(0.55 * s_loss_a, 0.12, 0.45)
            cand_i = (1.0 - gamma_state) * cur_i[act_idx] + gamma_state * next_i_a
            cand_x = np.stack([x_flat * cand_i, y_flat * cand_i, cand_i], axis=-1).reshape(len(act_idx), 2352)

            cand_probs, cand_z3, _, _, _, _ = recognize_with_E(enc_step, cand_x, protos_step)
            _, _, _, _, cand_j_loss_col, _ = recognize_with_E(enc_judge, cand_x, protos_judge)
            cand_loss = cand_j_loss_col[:, 0]

            beta_state = np.clip(0.55 * s_loss_a, 0.15, 0.48)
            blended_probs = (1.0 - beta_state) * best_probs[act_idx] + beta_state * cand_probs
            blended_probs = blended_probs / np.sum(blended_probs, axis=-1, keepdims=True)
            blended_z3, _ = normalize_rows((1.0 - beta_state) * best_z3[act_idx] + beta_state * cand_z3)

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

    probs_nat0, _, _, _, loss_nat0_col, _ = recognize_with_E(enc_step, x_test, protos_step)
    loss_nat0 = loss_nat0_col[:, 0]
    preds_nat0 = np.argmax(probs_nat0, axis=-1)
    hard_mask = loss_nat0 >= np.percentile(loss_nat0, 95)
    conf_mask = ~hard_mask

    preds_nat_f, _, loss_nat_f, steps_nat, snaps_bundle = run_convergence_loop(
        x_test, i_test, int_loss_gate=0.14, record_snaps=record_nat_snaps
    )

    acc_nat0_all = float(np.mean(preds_nat0 == test_lbl) * 100.0)
    acc_natf_all = float(np.mean(preds_nat_f == test_lbl) * 100.0)
    acc_nat0_conf = float(np.mean(preds_nat0[conf_mask] == test_lbl[conf_mask]) * 100.0)
    acc_natf_conf = float(np.mean(preds_nat_f[conf_mask] == test_lbl[conf_mask]) * 100.0)
    acc_nat0_hard = float(np.mean(preds_nat0[hard_mask] == test_lbl[hard_mask]) * 100.0)
    acc_natf_hard = float(np.mean(preds_nat_f[hard_mask] == test_lbl[hard_mask]) * 100.0)
    rescued_mask = (preds_nat0 != test_lbl) & (preds_nat_f == test_lbl)

    print(
        f"  [Natural MNIST 10,000 Test] All 10k: {acc_nat0_all:.2f}% -> {acc_natf_all:.2f}% | "
        f"Low-Loss 95%: {acc_nat0_conf:.2f}% -> {acc_natf_conf:.2f}% (Mean Steps: {np.mean(steps_nat[conf_mask]):.2f}) | "
        f"High-Loss 5%: {acc_nat0_hard:.2f}% -> {acc_natf_hard:.2f}% ({acc_natf_hard - acc_nat0_hard:+.2f}%, "
        f"Rescued: {int(np.sum(rescued_mask))}, Mean Steps: {np.mean(steps_nat[hard_mask]):.2f}, "
        f"IntLoss: {np.mean(loss_nat0[hard_mask]):.3f} -> {np.mean(loss_nat_f[hard_mask]):.3f})"
    )

    occ_summary = {}
    for mode in ["picket_fence", "center_blackout", "heavy_clutter"]:
        occ_u8 = apply_occlusions(test_img, mode=mode, seed=42)
        x_occ = build_8bit_tuple_representation(occ_u8, couple_background=True)
        i_occ = (occ_u8.astype(np.float32) / 255.0).reshape(len(test_img), 784)

        probs_o0, _, _, _, loss_o0_col, _ = recognize_with_E(enc_step, x_occ, protos_step)
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

    nat_summary = {
        "all_10k_step0_acc_pct": round(acc_nat0_all, 2),
        "all_10k_converged_acc_pct": round(acc_natf_all, 2),
        "low_loss_95pct_step0_acc_pct": round(acc_nat0_conf, 2),
        "low_loss_95pct_converged_acc_pct": round(acc_natf_conf, 2),
        "low_loss_95pct_mean_steps": round(float(np.mean(steps_nat[conf_mask])), 2),
        "high_loss_5pct_step0_acc_pct": round(acc_nat0_hard, 2),
        "high_loss_5pct_converged_acc_pct": round(acc_natf_hard, 2),
        "high_loss_5pct_mean_steps": round(float(np.mean(steps_nat[hard_mask])), 2),
        "rescued_natural_digits_count": int(np.sum(rescued_mask)),
    }
    return nat_summary, occ_summary, (loss_nat0, rescued_mask, snaps_bundle)


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

    E_frozen = copy.deepcopy(encoder)
    protos_frozen = compute_layer_prototypes(E_frozen, tr_x[:10000], tr_y[:10000], n_classes=10)
    _, u_clean_test, _ = E_frozen.forward(x_test)
    clean_step0_acc = float(np.mean(np.argmax(u_clean_test, axis=-1) == test_lbl) * 100.0)
    print(f"   FROZEN E {E_frozen.dims} ({E_frozen.param_count:,} params) -> Clean Test Step-0 Accuracy: {clean_step0_acc:.2f}%")

    # =========================================================================
    # STAGE 2: Train Reverse-Diffusion Generator G Against FROZEN E by Iterative Redraw
    # =========================================================================
    d1_star, d2_star, d3_star = E_frozen.hidden_dims
    G_s2 = ReverseDiffusionGenerator(d1=d1_star, d2=d2_star, d3=d3_star, d_out=10, out_pixels=784, lr=2.5e-3, seed=42)
    print(
        f"\n=== Stage 2: Training Reverse-Diffusion Generator G ({G_s2.cascade_dims}) Against FROZEN E ===\n"
        f"  Loop: X^(k) -> Z^(k) = E(X^(k)) -> X^(k+1) = Redraw(X^(k), G(Z^(k))) -> Measure E(X^(k+1)) Loss vs. Label y"
    )
    t_g = time.time()

    for ep in range(10):
        perm = rng.permutation(len(tr_x))
        for i in range(0, len(tr_x), bsz):
            idx = perm[i:i + bsz]
            xb_c, ib_c, yb = tr_x[idx], tr_i[idx], tr_y[idx]

            _, _, z2_clean_n, z1_clean_n, _, _ = recognize_with_E(E_frozen, xb_c, protos_frozen)
            xb_0, ib_0 = forward_diffusion_on_clean(ib_c, x_flat, y_flat, rng)

            # Iteration k = 0
            probs_0, z3_0, z2_0, z1_0, int_loss_0, _ = recognize_with_E(E_frozen, xb_0, protos_frozen)
            i_hat_0, z2_hat_0, z1_hat_0, cache_g_0 = G_s2.forward(probs_0, z3_0, z2_0, z1_0, int_loss_0)
            ib_1, replace_w_0 = redraw_canvas_step(ib_0, i_hat_0, int_loss_0)
            xb_1 = np.stack([x_flat * ib_1, y_flat * ib_1, ib_1], axis=-1).reshape(len(xb_c), 2352)

            dx_1, _, _ = E_frozen.input_gradient_from_supervised_loss(xb_1, yb, ortho_weights=ortho_w)
            dx_1_3d = dx_1.reshape(len(xb_c), 784, 3)
            di_from_E_1 = (dx_1_3d[:, :, 0] * x_flat + dx_1_3d[:, :, 1] * y_flat + dx_1_3d[:, :, 2]) * replace_w_0

            di_g0 = (i_hat_0 - ib_c) * (1.8 / len(xb_c)) + di_from_E_1 * 0.65
            dz2_g0 = (z2_hat_0 - z2_clean_n) * (0.5 / len(xb_c))
            dz1_g0 = (z1_hat_0 - z1_clean_n) * (0.5 / len(xb_c))
            G_s2.backward_and_step(cache_g_0, di_g0, dz2_hat_n_extra=dz2_g0, dz1_hat_n_extra=dz1_g0)

            # Iteration k = 1
            probs_1, z3_1, z2_1, z1_1, int_loss_1, _ = recognize_with_E(E_frozen, xb_1, protos_frozen)
            i_hat_1, z2_hat_1, z1_hat_1, cache_g_1 = G_s2.forward(probs_1, z3_1, z2_1, z1_1, int_loss_1)
            ib_2, replace_w_1 = redraw_canvas_step(ib_1, i_hat_1, int_loss_1)
            xb_2 = np.stack([x_flat * ib_2, y_flat * ib_2, ib_2], axis=-1).reshape(len(xb_c), 2352)

            dx_2, _, _ = E_frozen.input_gradient_from_supervised_loss(xb_2, yb, ortho_weights=ortho_w)
            dx_2_3d = dx_2.reshape(len(xb_c), 784, 3)
            di_from_E_2 = (dx_2_3d[:, :, 0] * x_flat + dx_2_3d[:, :, 1] * y_flat + dx_2_3d[:, :, 2]) * replace_w_1

            di_g1 = (i_hat_1 - ib_c) * (1.8 / len(xb_c)) + di_from_E_2 * 0.65
            dz2_g1 = (z2_hat_1 - z2_clean_n) * (0.5 / len(xb_c))
            dz1_g1 = (z1_hat_1 - z1_clean_n) * (0.5 / len(xb_c))
            G_s2.backward_and_step(cache_g_1, di_g1, dz2_hat_n_extra=dz2_g1, dz1_hat_n_extra=dz1_g1)

    print(f"   Trained G in {time.time() - t_g:.2f}s (E weights remained 100% frozen; Clean Test Acc = {clean_step0_acc:.2f}%).")

    rng_state_after_s2 = copy.deepcopy(rng.bit_generator.state)

    s2_nat, s2_occ, (loss_nat0_s2, rescued_mask_s2, (snaps_i_s2, snaps_loss_s2, snaps_preds_s2)) = evaluate_pipeline_stage(
        "Stage 2 Checkpoint (Frozen E + Trained G)",
        E_frozen,
        G_s2,
        protos_frozen,
        E_frozen,
        protos_frozen,
        x_test,
        i_test,
        test_img,
        test_lbl,
        x_flat,
        y_flat,
        record_nat_snaps=True,
    )

    # =========================================================================
    # STAGE 3: Reality-Anchored Joint Supervised Relaxation of Both E and G
    # =========================================================================
    print(
        f"\n=== Stage 3: Reality-Anchored Joint Supervised Relaxation of E {E_frozen.dims} and G {G_s2.cascade_dims} ===\n"
        f"  E updates on real inputs (X_clean, X^(0)) and reinforces redrawn X^(1) only when E(X^(0)) matched y,\n"
        f"  preventing E from flattering wrong-class redraws while G updates through E(X^(1)) toward label y."
    )
    t_joint = time.time()
    E_anc = copy.deepcopy(E_frozen)
    G_anc = copy.deepcopy(G_s2)
    E_anc.lr = 4.0e-4
    G_anc.lr = 1.0e-3
    protos_anc = copy.deepcopy(protos_frozen)

    for ep in range(6):
        perm = rng.permutation(len(tr_x))
        for i in range(0, len(tr_x), bsz):
            idx = perm[i:i + bsz]
            xb_c, ib_c, yb = tr_x[idx], tr_i[idx], tr_y[idx]
            one_hot = np.eye(10, dtype=np.float32)[yb]

            _, u_c, cache_e_c = E_anc.forward(xb_c)
            E_anc.backward_and_step(cache_e_c, (softmax(u_c) - one_hot) * (0.7 / len(xb_c)), yb=yb, ortho_weights=ortho_w)

            _, _, z2_clean_n, z1_clean_n, _, _ = recognize_with_E(E_anc, xb_c, protos_anc)
            xb_0, ib_0 = forward_diffusion_on_clean(ib_c, x_flat, y_flat, rng)
            _, u_0, cache_e_0 = E_anc.forward(xb_0)
            E_anc.backward_and_step(cache_e_0, (softmax(u_0) - one_hot) * (0.6 / len(xb_c)), yb=yb, ortho_weights=ortho_w)

            probs_0, z3_0, z2_0, z1_0, int_loss_0, _ = recognize_with_E(E_anc, xb_0, protos_anc)
            i_hat_0, z2_hat_0, z1_hat_0, cache_g_0 = G_anc.forward(probs_0, z3_0, z2_0, z1_0, int_loss_0)
            ib_1, replace_w_0 = redraw_canvas_step(ib_0, i_hat_0, int_loss_0)
            xb_1 = np.stack([x_flat * ib_1, y_flat * ib_1, ib_1], axis=-1).reshape(len(xb_c), 2352)

            dx_1, _, _ = E_anc.input_gradient_from_supervised_loss(xb_1, yb, ortho_weights=ortho_w)
            dx_1_3d = dx_1.reshape(len(xb_c), 784, 3)
            di_from_E_1 = (dx_1_3d[:, :, 0] * x_flat + dx_1_3d[:, :, 1] * y_flat + dx_1_3d[:, :, 2]) * replace_w_0
            G_anc.backward_and_step(
                cache_g_0,
                (i_hat_0 - ib_c) * (1.6 / len(xb_c)) + di_from_E_1 * 0.85,
                (z2_hat_0 - z2_clean_n) * (0.5 / len(xb_c)),
                (z1_hat_0 - z1_clean_n) * (0.5 / len(xb_c)),
            )

            correct_0 = (np.argmax(probs_0, axis=-1) == yb)[:, None].astype(np.float32)
            _, u_1, cache_e_1 = E_anc.forward(xb_1)
            du_1 = ((softmax(u_1) - one_hot) * correct_0) * (0.35 / len(xb_c))
            E_anc.backward_and_step(cache_e_1, du_1, yb=yb, ortho_weights=[0.05, 0.05, 0.40])

        protos_anc = compute_layer_prototypes(E_anc, tr_x[:10000], tr_y[:10000], n_classes=10)

    print(f"   Completed Reality-Anchored Joint Relaxation in {time.time() - t_joint:.2f}s.")

    s3b_nat, s3b_occ, _ = evaluate_pipeline_stage(
        "Stage 3B Checkpoint (Reality-Anchored Joint Relaxation E + G, Self-Judged)",
        E_anc,
        G_anc,
        protos_anc,
        E_anc,
        protos_anc,
        x_test,
        i_test,
        test_img,
        test_lbl,
        x_flat,
        y_flat,
        record_nat_snaps=False,
    )

    s3a_nat, s3a_occ, _ = evaluate_pipeline_stage(
        "Stage 3A Checkpoint (Reality-Anchored Joint Relaxation E + G, Judged by Stage-1 Frozen E)",
        E_anc,
        G_anc,
        protos_anc,
        E_frozen,
        protos_frozen,
        x_test,
        i_test,
        test_img,
        test_lbl,
        x_flat,
        y_flat,
        record_nat_snaps=False,
    )

    # =========================================================================
    # STAGE 3 UNCONSTRAINED ABLATION: Unconditional Joint Relaxation (Judge Drift)
    # =========================================================================
    print(
        f"\n=== Stage 3 Unconstrained Ablation: Joint Relaxation WITHOUT Reality Anchor ===\n"
        f"  E unconditionally reinforces every redrawn canvas X^(1) toward label y (even when E(X^(0)) guessed\n"
        f"  the wrong digit and G redrew a wrong-class hallucination) -> demonstrating Judge Drift."
    )
    t_unc = time.time()
    rng_unc = np.random.default_rng(42)
    rng_unc.bit_generator.state = rng_state_after_s2
    E_unc = copy.deepcopy(E_frozen)
    G_unc = copy.deepcopy(G_s2)
    E_unc.lr = 8.0e-4
    G_unc.lr = 1.5e-3
    protos_unc = copy.deepcopy(protos_frozen)

    for ep in range(6):
        perm = rng_unc.permutation(len(tr_x))
        for i in range(0, len(tr_x), bsz):
            idx = perm[i:i + bsz]
            xb_c, ib_c, yb = tr_x[idx], tr_i[idx], tr_y[idx]
            one_hot = np.eye(10, dtype=np.float32)[yb]

            _, u_c, cache_e_c = E_unc.forward(xb_c)
            E_unc.backward_and_step(cache_e_c, (softmax(u_c) - one_hot) * (0.25 / len(xb_c)), yb=yb, ortho_weights=ortho_w)

            _, _, z2_clean_n, z1_clean_n, _, _ = recognize_with_E(E_unc, xb_c, protos_unc)
            xb_0, ib_0 = forward_diffusion_on_clean(ib_c, x_flat, y_flat, rng_unc)

            probs_0, z3_0, z2_0, z1_0, int_loss_0, _ = recognize_with_E(E_unc, xb_0, protos_unc)
            i_hat_0, z2_hat_0, z1_hat_0, cache_g_0 = G_unc.forward(probs_0, z3_0, z2_0, z1_0, int_loss_0)
            ib_1, replace_w_0 = redraw_canvas_step(ib_0, i_hat_0, int_loss_0)
            xb_1 = np.stack([x_flat * ib_1, y_flat * ib_1, ib_1], axis=-1).reshape(len(xb_c), 2352)

            dx_1, _, _ = E_unc.input_gradient_from_supervised_loss(xb_1, yb, ortho_weights=ortho_w)
            dx_1_3d = dx_1.reshape(len(xb_c), 784, 3)
            di_from_E_1 = (dx_1_3d[:, :, 0] * x_flat + dx_1_3d[:, :, 1] * y_flat + dx_1_3d[:, :, 2]) * replace_w_0
            G_unc.backward_and_step(
                cache_g_0,
                (i_hat_0 - ib_c) * (1.2 / len(xb_c)) + di_from_E_1 * 0.90,
                (z2_hat_0 - z2_clean_n) * (0.5 / len(xb_c)),
                (z1_hat_0 - z1_clean_n) * (0.5 / len(xb_c)),
            )

            # Unconstrained update on X^(1) even when E(X^(0)) guessed the wrong digit!
            _, u_1, cache_e_1 = E_unc.forward(xb_1)
            du_1 = (softmax(u_1) - one_hot) * (0.95 / len(xb_c))
            E_unc.backward_and_step(cache_e_1, du_1, yb=yb, ortho_weights=ortho_w)

        protos_unc = compute_layer_prototypes(E_unc, tr_x[:10000], tr_y[:10000], n_classes=10)

    print(f"   Completed Unconstrained Joint Relaxation in {time.time() - t_unc:.2f}s.")

    s3u_nat, s3u_occ, _ = evaluate_pipeline_stage(
        "Stage 3 Unconstrained Checkpoint (Relaxed E + G Without Reality Anchor, Self-Judged)",
        E_unc,
        G_unc,
        protos_unc,
        E_unc,
        protos_unc,
        x_test,
        i_test,
        test_img,
        test_lbl,
        x_flat,
        y_flat,
        record_nat_snaps=False,
    )

    # =========================================================================
    # SAME-CORRUPTION-STREAM STANDARD MLP BASELINES (22 Epochs Exposure)
    # =========================================================================
    print("\n=== Training & Evaluating Same-Stream Standard MLP Baselines (22 Epochs Exposure) ===")
    same_width_mlp_summary = train_and_evaluate_same_stream_mlp(
        "Same-Width Standard MLP (784 -> 105 -> 32 -> 11 -> 10)",
        [784, d1_star, d2_star, d3_star, 10],
        tr_i,
        tr_y,
        i_test,
        test_img,
        test_lbl,
        x_flat,
        y_flat,
    )
    large_mlp_summary = train_and_evaluate_same_stream_mlp(
        "2.5x Larger Standard MLP (784 -> 256 -> 64 -> 32 -> 10)",
        [784, 256, 64, 32, 10],
        tr_i,
        tr_y,
        i_test,
        test_img,
        test_lbl,
        x_flat,
        y_flat,
    )

    # Render visual progression grid from the Stage-2 rescued digits
    rescued_indices = np.where(rescued_mask_s2)[0]
    rescued_indices = rescued_indices[np.argsort(-loss_nat0_s2[rescued_indices])]
    chosen_rescued = rescued_indices[:10]

    while len(snaps_i_s2) < 4:
        snaps_i_s2.append(snaps_i_s2[-1])
        snaps_loss_s2.append(snaps_loss_s2[-1])
        snaps_preds_s2.append(snaps_preds_s2[-1])

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
        f"Dual-Cut Pipeline: E {E_frozen.dims} <-> Reverse-Diffusion G {G_s2.cascade_dims}",
        fill=TEXT_MAIN,
        font=fonts["title"],
    )
    draw.text(
        (margin_x, margin_y + 44),
        "Gated by Internal Loss inside E; redrawn from actual recognition vectors until recognized vectors converge",
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
            pred_c = int(snaps_preds_s2[r_idx][sample_idx])
            true_c = int(test_lbl[sample_idx])
            l_val = float(snaps_loss_s2[r_idx][sample_idx])
            border_c = ACCENT_GREEN if pred_c == true_c else ACCENT_RED
            draw.rectangle([cx + 4, ry + 4, cx + cell_w - 4, ry + cell_h - 4], fill=BG_CARD, outline=border_c, width=2)
            patch_arr = (np.clip(snaps_i_s2[r_idx][sample_idx].reshape(28, 28), 0.0, 1.0) * 255.0).astype(np.uint8)
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
        "stage1_encoder_dims": E_frozen.dims,
        "stage1_encoder_params": E_frozen.param_count,
        "legacy_256_64_32_funnel_params_with_bias": 621226,
        "legacy_256_64_32_funnel_params_weights_only": 620864,
        "param_reduction_vs_legacy_pct": round((1.0 - E_frozen.param_count / 621226.0) * 100.0, 1),
        "stage1_gradual_discovery": stage1_results,
        "stage2_generator_cascade_dims": G_s2.cascade_dims,
        "same_stream_baselines_22ep": {
            "same_width_std_mlp_784_105_32_11_10": same_width_mlp_summary,
            "large_std_mlp_784_256_64_32_10": large_mlp_summary,
        },
        "stage2_frozen_E_trained_G": {
            "natural_mnist": s2_nat,
            "occlusions_converged": s2_occ,
        },
        "stage3_unconstrained_relaxed_E_G": {
            "natural_mnist": s3u_nat,
            "occlusions_converged": s3u_occ,
        },
        "stage3b_reality_anchored_self_judged": {
            "natural_mnist": s3b_nat,
            "occlusions_converged": s3b_occ,
        },
        "stage3a_reality_anchored_frozen_judge": {
            "natural_mnist": s3a_nat,
            "occlusions_converged": s3a_occ,
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved visual grid to {grid_path} and complete metrics to {out_json}")


if __name__ == "__main__":
    main()
