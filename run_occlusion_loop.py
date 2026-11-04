#!/usr/bin/env python3
"""
100% Mask-Free Multi-Step Closed-Loop Self-Reflection on Occluded & Corrupted MNIST Digits
(X_occluded -> z_0 -> X_hat_1 -> E(X_hat_1) = z_1 -> X_hat_2 -> ...)

Tests how the Ontological Determination Funnel (Upward E + Hierarchical Downward G)
resolves three severe occlusions on all 10,000 unseen test digits across iterative
self-reflection passes WITH ZERO EXTERNAL OCCLUSION MASK, and compares head-to-head
against BOTH:
  1. A Clean-Trained Standard Feedforward Cross-Entropy MLP, and
  2. A Same-Corruption-Trained Standard Feedforward Cross-Entropy MLP (trained on the
     exact same random_training_corruption distribution and step count as E).
Also generates `assets/occlusion_refinement_grid.png` and `results/occlusion_results.json`.
"""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from run_experiment import (
    EXP_DIR,
    RESULTS_DIR,
    AdamMLP,
    build_8bit_tuple_representation,
    compute_prototypes,
    load_mnist,
    normalize_rows,
    softmax,
    latent_group_orthogonality_grad,
)

ASSETS_DIR = EXP_DIR / "assets"


class HierarchicalReflectionDecoder:
    """
    Downward Self-Reflection Funnel G:
    Takes the upward funnel's multi-scale representations:
      - z*_norm (32d collective identity cone)
      - z2_norm (64d mid-level stroke abstraction)
      - z1_norm (256d first-layer global ReLU cut)
    Total input = 352d -> 256d (ReLU) -> 784d clean canvas expectation `i_hat`.
    """
    def __init__(self, d_in=352, d_hid=256, out_dim=784, lr=2.5e-3, seed=42):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / d_in), size=(d_in, d_hid)).astype(np.float32)
        self.b1 = np.zeros((1, d_hid), dtype=np.float32)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / d_hid), size=(d_hid, out_dim)).astype(np.float32)
        self.b2 = np.zeros((1, out_dim), dtype=np.float32)

        self.mW1, self.vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.mb1, self.vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.mW2, self.vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.mb2, self.vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.t = 0

        y_grid = np.linspace(0.0, 1.0, 28, dtype=np.float32)[:, None]
        x_grid = np.linspace(0.0, 1.0, 28, dtype=np.float32)[None, :]
        self.x_flat = np.broadcast_to(x_grid, (28, 28)).reshape(1, 784)
        self.y_flat = np.broadcast_to(y_grid, (28, 28)).reshape(1, 784)

    def forward(self, z_star_n: np.ndarray, z2_n: np.ndarray, z1_n: np.ndarray):
        zin = np.concatenate([z_star_n, z2_n, z1_n], axis=-1)
        u1 = zin @ self.W1 + self.b1
        h1 = np.maximum(0.0, u1)
        u2 = h1 @ self.W2 + self.b2
        i_hat = np.clip(u2, 0.0, 1.0)
        return i_hat, (zin, u1, h1, u2, i_hat)

    def backward_and_step(self, cache, di_hat: np.ndarray, weight_decay: float = 1e-4):
        zin, u1, h1, u2, i_hat = cache
        du2 = di_hat * ((u2 > 0.0) & (u2 < 1.0)).astype(np.float32)
        dW2 = h1.T @ du2 + weight_decay * self.W2
        db2 = np.sum(du2, axis=0, keepdims=True)
        dh1 = du2 @ self.W2.T
        du1 = dh1 * (u1 > 0.0).astype(np.float32)
        dW1 = zin.T @ du1 + weight_decay * self.W1
        db1 = np.sum(du1, axis=0, keepdims=True)

        self.t += 1
        b1_c = 1.0 - 0.9 ** self.t
        b2_c = 1.0 - 0.999 ** self.t
        for W, b, dW, db, mW, vW, mb, vb in [
            (self.W1, self.b1, dW1, db1, self.mW1, self.vW1, self.mb1, self.vb1),
            (self.W2, self.b2, dW2, db2, self.mW2, self.vW2, self.mb2, self.vb2),
        ]:
            mW[:] = 0.9 * mW + 0.1 * dW
            vW[:] = 0.999 * vW + 0.001 * (dW ** 2)
            W -= self.lr * (mW / b1_c) / (np.sqrt(vW / b2_c) + 1e-8)
            mb[:] = 0.9 * mb + 0.1 * db
            vb[:] = 0.999 * vb + 0.001 * (db ** 2)
            b -= self.lr * (mb / b1_c) / (np.sqrt(vb / b2_c) + 1e-8)


def compute_encoder_grads(encoder: AdamMLP, cache, du4: np.ndarray, dz_star_extra: np.ndarray, weight_decay: float = 1e-4):
    """Compute analytical gradients for AdamMLP so Step-0 and Step-1 passes combine into a single Adam update."""
    x, u1, z1, u2, z2, u3, z_star, u4, g_cone = cache
    dW3 = z_star.T @ du4 + weight_decay * encoder.W[3]
    db3 = np.sum(du4, axis=0, keepdims=True)
    dz_star = du4 @ encoder.W[3].T + dz_star_extra

    du3 = dz_star * (u3 > 0.0).astype(np.float32)
    dW2 = z2.T @ du3 + weight_decay * encoder.W[2]
    db2 = np.sum(du3, axis=0, keepdims=True)
    dz2 = du3 @ encoder.W[2].T

    du2 = dz2 * (u2 > 0.0).astype(np.float32)
    dW1 = z1.T @ du2 + weight_decay * encoder.W[1]
    db1 = np.sum(du2, axis=0, keepdims=True)
    dz1 = du2 @ encoder.W[1].T

    du1 = dz1 * (u1 > 0.0).astype(np.float32)
    dW0 = x.T @ du1 + weight_decay * encoder.W[0]
    db0 = np.sum(du1, axis=0, keepdims=True)
    return [dW0, dW1, dW2, dW3], [db0, db1, db2, db3]


def apply_occlusions(images_uint8: np.ndarray, mode: str, seed: int = 123) -> np.ndarray:
    """
    Create occluded/corrupted 28x28 uint8 images (returns ONLY the corrupted images—NO mask!).
    Modes:
      - 'picket_fence': 3 horizontal slats erased + 2 bright vertical fence bars drawn across the digit
      - 'center_blackout': a 10x12 rectangular blackout wiping out the core of the digit
      - 'heavy_clutter': 45% random stroke dropout + 15% bright random static across the canvas
    """
    rng = np.random.default_rng(seed)
    out = images_uint8.copy()
    n, h, w = out.shape

    if mode == "picket_fence":
        for r_start in [7, 13, 19]:
            out[:, r_start:r_start + 2, :] = 0
        for c_bar in [9, 18]:
            out[:, 4:24, c_bar:c_bar + 2] = 220
    elif mode == "center_blackout":
        out[:, 9:19, 8:20] = 0
    elif mode == "heavy_clutter":
        drop = rng.random((n, h, w)) < 0.45
        clutter = rng.random((n, h, w)) < 0.15
        out[drop] = 0
        out[clutter] = rng.integers(140, 255, size=int(np.sum(clutter)), dtype=np.uint8)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return out


def random_training_corruption(ib_clean: np.ndarray, x_flat: np.ndarray, y_flat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Generate training corruptions (missing bars, bright vertical bars, and clutter) applied
    identically to BOTH the Same-Corruption Standard MLP baseline and the Determination Funnel.
    """
    bsz = len(ib_clean)
    im = ib_clean.reshape(bsz, 28, 28).copy()
    im *= (rng.random((bsz, 28, 28)) > 0.30).astype(np.float32)
    r0 = rng.integers(5, 19)
    im[:, r0:r0 + 4, :] = 0.0
    c0 = rng.integers(6, 20)
    im[:bsz // 2, 4:24, c0:c0 + 2] = 0.85
    clutter = (rng.random((bsz - bsz // 2, 28, 28)) < 0.12).astype(np.float32) * 0.8
    im[bsz // 2:] = np.clip(im[bsz // 2:] + clutter, 0.0, 1.0)

    i_corr = im.reshape(bsz, 784)
    x_corr = np.stack([x_flat * i_corr, y_flat * i_corr, i_corr], axis=-1).reshape(bsz, 2352)
    return x_corr


def mask_free_target_canvas(i_occ: np.ndarray, i_hat: np.ndarray) -> np.ndarray:
    """
    100% MASK-FREE self-reflection operator Phi(i_occ, i_hat):
    - ReLU-cuts faint background noise in `i_hat` below 0.05.
    - Measures internal agreement between `i_occ` and `i_hat_clean`:
      `agree` is high iff both show a stroke (`i_occ * i_hat_clean`) AND `i_occ` does not have
      an unexplained bright spike/bar far above `i_hat_clean`.
    - Where they agree, preserves 80% of `i_occ`.
    - Where they conflict (missing stroke `i_occ=0` OR foreign bar/spike `i_occ >> i_hat_clean`),
      resolves 85% to `i_hat_clean`!
    """
    i_hat_clean = np.clip((i_hat - 0.05) / 0.95, 0.0, 1.0)
    prod_agree = np.clip((i_occ * i_hat_clean) / 0.15, 0.0, 1.0)
    no_excess_spike = np.clip(1.0 - np.maximum(0.0, i_occ - i_hat_clean - 0.10) / 0.35, 0.0, 1.0)
    agree = prod_agree * no_excess_spike

    keep_obs_w = 0.80 * agree + 0.15 * (1.0 - agree)
    return np.clip(keep_obs_w * i_occ + (1.0 - keep_obs_w) * i_hat_clean, 0.0, 1.0)


def main():
    print("Loading MNIST and building 8-bit (x, y, I) in [0, 255]^3 universe...")
    train_img, train_lbl, test_img, test_lbl = load_mnist()
    x_train = build_8bit_tuple_representation(train_img, couple_background=True)
    x_test = build_8bit_tuple_representation(test_img, couple_background=True)
    i_train = (train_img.astype(np.float32) / 255.0).reshape(len(train_img), 784)
    i_test = (test_img.astype(np.float32) / 255.0).reshape(len(test_img), 784)

    y_grid = np.linspace(0.0, 1.0, 28, dtype=np.float32)[:, None]
    x_grid = np.linspace(0.0, 1.0, 28, dtype=np.float32)[None, :]
    x_flat = np.broadcast_to(x_grid, (28, 28)).reshape(1, 784)
    y_flat = np.broadcast_to(y_grid, (28, 28)).reshape(1, 784)

    bsz = 256
    print("\n1. Training Standard Feedforward Cross-Entropy MLP Baselines (Clean-Trained & Same-Corruption-Trained)...")
    std_mlp_clean = AdamMLP(dims=(784, 256, 64, 32, 10), lr=1.5e-3, seed=42)
    std_mlp_corr = AdamMLP(dims=(784, 256, 64, 32, 10), lr=1.5e-3, seed=42)

    rng_m = np.random.default_rng(42)
    for ep in range(12):
        perm = rng_m.permutation(len(i_train))
        for i in range(0, len(i_train), bsz):
            b = perm[i:i + bsz]
            ib = i_train[b]
            yb = train_lbl[b]
            one_hot = np.eye(10, dtype=np.float32)[yb]

            _, u4_c, _, cache_c = std_mlp_clean.forward(ib)
            std_mlp_clean.backward_and_step(cache_c, (softmax(u4_c) - one_hot) / len(b))

            _, u4_a1, _, cache_a1 = std_mlp_corr.forward(ib)
            std_mlp_corr.backward_and_step(cache_a1, (softmax(u4_a1) - one_hot) / len(b))
            xb_corr = random_training_corruption(ib, x_flat, y_flat, rng_m)
            ib_corr = xb_corr[:, 2::3]
            _, u4_a2, _, cache_a2 = std_mlp_corr.forward(ib_corr)
            std_mlp_corr.backward_and_step(cache_a2, (softmax(u4_a2) - one_hot) / len(b))

    print("2. Training Closed-Loop Determination Funnel (E + Hierarchical G) on the EXACT SAME corruption stream...")
    encoder = AdamMLP(dims=(2352, 256, 64, 32, 10), lr=1.5e-3, seed=42)
    decoder = HierarchicalReflectionDecoder(d_in=352, d_hid=256, out_dim=784, lr=2.5e-3, seed=42)

    t0 = time.time()
    rng_f = np.random.default_rng(42)
    for ep in range(12):
        perm = rng_f.permutation(len(x_train))
        for i in range(0, len(x_train), bsz):
            b = perm[i:i + bsz]
            xb_clean = x_train[b]
            ib_clean = i_train[b]
            yb = train_lbl[b]
            one_hot = np.eye(10, dtype=np.float32)[yb]

            # Pass A: Clean upward pass
            z_clean, u4_clean, _, cache_clean = encoder.forward(xb_clean)
            z_clean_n, _ = normalize_rows(z_clean)

            # Pass B: Corrupted input (Step 0) -> E -> G reconstructs clean canvas
            xb_corr = random_training_corruption(ib_clean, x_flat, y_flat, rng_f)
            ib_corr = xb_corr[:, 2::3]
            z_0, u4_0, _, cache_0 = encoder.forward(xb_corr)
            z_0_n, norms_0 = normalize_rows(z_0)
            z2_0_n, _ = normalize_rows(cache_0[4])
            z1_0_n, _ = normalize_rows(cache_0[2])

            i_hat_0, dec_cache_0 = decoder.forward(z_0_n, z2_0_n, z1_0_n)
            decoder.backward_and_step(dec_cache_0, (i_hat_0 - ib_clean) * (2.0 / len(b)))

            # Step 1 mask-free reflection canvas -> E -> G
            ib_ref1 = mask_free_target_canvas(ib_corr, i_hat_0)
            xb_ref1 = np.stack([x_flat * ib_ref1, y_flat * ib_ref1, ib_ref1], axis=-1).reshape(len(b), 2352)
            z_1, u4_1, _, cache_1 = encoder.forward(xb_ref1)
            z_1_n, norms_1 = normalize_rows(z_1)
            z2_1_n, _ = normalize_rows(cache_1[4])
            z1_1_n, _ = normalize_rows(cache_1[2])

            i_hat_1, dec_cache_1 = decoder.forward(z_1_n, z2_1_n, z1_1_n)
            decoder.backward_and_step(dec_cache_1, (i_hat_1 - ib_clean) * (2.0 / len(b)))

            # Update 1 on E: Clean pass
            du4_c = (softmax(u4_clean) - one_hot) / len(b)
            dz_grp_c = latent_group_orthogonality_grad(z_clean, yb) * 1.5
            encoder.backward_and_step(cache_clean, du4_c, dz_star_extra=dz_grp_c)

            # Update 2 on E: Single combined Adam step on (0.45 * Step-0 + 0.55 * Step-1)
            dz_m0_n = (z_0_n - z_clean_n) * (2.0 / len(b))
            proj_0 = np.sum(dz_m0_n * z_0_n, axis=-1, keepdims=True)
            dz_m0 = (dz_m0_n - z_0_n * proj_0) / norms_0
            du4_0 = (softmax(u4_0) - one_hot) / len(b)
            dz_grp_0 = latent_group_orthogonality_grad(z_0, yb) * 1.0 + dz_m0 * 1.5
            gW0, gb0 = compute_encoder_grads(encoder, cache_0, du4_0, dz_grp_0)

            dz_m1_n = (z_1_n - z_clean_n) * (2.0 / len(b))
            proj_1 = np.sum(dz_m1_n * z_1_n, axis=-1, keepdims=True)
            dz_m1 = (dz_m1_n - z_1_n * proj_1) / norms_1
            du4_1 = (softmax(u4_1) - one_hot) / len(b)
            dz_grp_1 = latent_group_orthogonality_grad(z_1, yb) * 1.0 + dz_m1 * 2.0
            gW1, gb1 = compute_encoder_grads(encoder, cache_1, du4_1, dz_grp_1)

            gW_comb = [0.45 * w0 + 0.55 * w1 for w0, w1 in zip(gW0, gW1)]
            gb_comb = [0.45 * b0 + 0.55 * b1 for b0, b1 in zip(gb0, gb1)]
            encoder._adam_update(gW_comb, gb_comb)

    print(f"   Trained E + G in {time.time() - t0:.2f}s.")
    protos = compute_prototypes(encoder, x_train[:10000], train_lbl[:10000])

    z_test_clean, _, _, _ = encoder.forward(x_test)
    z_test_clean_n, _ = normalize_rows(z_test_clean)

    print("\n3. Running 100% Mask-Free Multi-Step Closed-Loop Inference (Step 0 -> Step 1 -> Step 2 -> Step 3 -> Step 4)...")
    all_metrics = {}
    grid_samples = None

    for mode in ["picket_fence", "center_blackout", "heavy_clutter"]:
        occ_uint8 = apply_occlusions(test_img, mode=mode, seed=42)  # Zero mask returned or used!
        x_occ = build_8bit_tuple_representation(occ_uint8, couple_background=True)
        i_occ = (occ_uint8.astype(np.float32) / 255.0).reshape(len(test_img), 784)

        _, u4_std_clean, _, _ = std_mlp_clean.forward(i_occ)
        std_clean_acc = float(np.mean(np.argmax(u4_std_clean, axis=-1) == test_lbl) * 100.0)

        _, u4_std_corr, _, _ = std_mlp_corr.forward(i_occ)
        std_corr_acc = float(np.mean(np.argmax(u4_std_corr, axis=-1) == test_lbl) * 100.0)

        cur_i = i_occ.copy()
        cur_x = x_occ.copy()
        z_0, _, _, _ = encoder.forward(x_occ)
        z_0_n, _ = normalize_rows(z_0)
        z_run_n = z_0_n.copy()

        step_stats = []
        snapshots = [i_occ[:200].copy()]

        for step in range(5):
            z_k, u4_k, _, cache_k = encoder.forward(cur_x)
            z_k_n, _ = normalize_rows(z_k)
            z2_k_n, _ = normalize_rows(cache_k[4])
            z1_k_n, _ = normalize_rows(cache_k[2])

            if step > 0:
                beta_k = 0.60 * (0.75 ** (step - 1))
                z_run_n, _ = normalize_rows((1.0 - beta_k) * z_run_n + beta_k * z_k_n)
            else:
                z_run_n = z_k_n

            sims = z_run_n @ protos.T
            preds = np.argmax(sims, axis=-1)
            acc_k = float(np.mean(preds == test_lbl) * 100.0)
            cos_to_clean = float(np.mean(np.sum(z_run_n * z_test_clean_n, axis=-1)))
            pixel_mse = float(np.mean((cur_i - i_test) ** 2))

            step_stats.append({
                "step": step,
                "accuracy_pct": round(acc_k, 2),
                "latent_cosine_to_clean": round(cos_to_clean, 4),
                "pixel_mse_to_clean": round(pixel_mse, 4),
            })

            # Downward reflection from current multi-scale state (z_run_n, z2_k_n, z1_k_n)
            i_hat_k, _ = decoder.forward(z_run_n, z2_k_n, z1_k_n)
            target_i = mask_free_target_canvas(i_occ, i_hat_k)
            # Damped Banach fixed-point canvas step (100% mask-free):
            gamma_k = 0.55 * (0.70 ** step)
            cur_i = (1.0 - gamma_k) * cur_i + gamma_k * target_i
            cur_x = np.stack(
                [decoder.x_flat * cur_i, decoder.y_flat * cur_i, cur_i], axis=-1
            ).reshape(len(test_img), 2352)

            if step < 4:
                snapshots.append(cur_i[:200].copy())

        if mode == "picket_fence":
            grid_samples = (i_test[:200], snapshots, test_lbl[:200])

        print(
            f"\n  [{mode.upper()}] Std MLP (Clean-Trained): {std_clean_acc:.2f}% | "
            f"Std MLP (Same-Corruption-Trained): {std_corr_acc:.2f}%"
        )
        for st in step_stats:
            print(
                f"    Step {st['step']} (Mask-Free) -> Acc: {st['accuracy_pct']:6.2f}% | "
                f"Latent Cos to Clean z*: {st['latent_cosine_to_clean']:.4f} | "
                f"Canvas MSE to Clean Digit: {st['pixel_mse_to_clean']:.4f}"
            )
        all_metrics[mode] = {
            "standard_mlp_clean_trained_acc": round(std_clean_acc, 2),
            "standard_mlp_same_corruption_trained_acc": round(std_corr_acc, 2),
            "mask_free_closed_loop_steps": step_stats,
        }

    clean_200, snaps, lbls_200 = grid_samples
    chosen_indices = []
    for d in range(10):
        idx_d = np.where(lbls_200 == d)[0][0]
        chosen_indices.append(idx_d)

    row_labels = [
        "1. Original Clean",
        "2. Fence Occluded",
        "3. Loop Step 1",
        "4. Loop Step 2",
        "5. Loop Step 4",
    ]
    rows_data = [clean_200, snaps[0], snaps[1], snaps[2], snaps[4]]

    cell_size = 56
    pad = 8
    left_margin = 145
    top_margin = 36
    img_w = left_margin + 10 * (cell_size + pad) + pad
    img_h = top_margin + len(rows_data) * (cell_size + pad) + pad

    canvas = Image.new("RGB", (img_w, img_h), (13, 17, 23))
    draw = ImageDraw.Draw(canvas)

    for col_idx, d in enumerate(range(10)):
        x0 = left_margin + pad + col_idx * (cell_size + pad)
        draw.text((x0 + 18, 10), f"Digit {d}", fill=(88, 166, 255))

    for row_idx, (r_label, r_arr) in enumerate(zip(row_labels, rows_data)):
        y0 = top_margin + pad + row_idx * (cell_size + pad)
        color = (248, 81, 73) if row_idx == 1 else ((63, 185, 80) if row_idx == 4 else (240, 246, 252))
        draw.text((12, y0 + 20), r_label, fill=color)
        for col_idx, sample_idx in enumerate(chosen_indices):
            x0 = left_margin + pad + col_idx * (cell_size + pad)
            patch = (np.clip(r_arr[sample_idx].reshape(28, 28), 0.0, 1.0) * 255.0).astype(np.uint8)
            patch_im = Image.fromarray(patch, mode="L").resize((cell_size, cell_size), Image.Resampling.NEAREST)
            canvas.paste(patch_im.convert("RGB"), (x0, y0))
            draw.rectangle([x0 - 1, y0 - 1, x0 + cell_size, y0 + cell_size], outline=(48, 54, 61))

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    grid_path = ASSETS_DIR / "occlusion_refinement_grid.png"
    canvas.save(grid_path)
    out_json = RESULTS_DIR / "occlusion_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved visual refinement grid to {grid_path} and metrics to {out_json}")


if __name__ == "__main__":
    main()
