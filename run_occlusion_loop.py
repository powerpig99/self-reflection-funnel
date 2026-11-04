#!/usr/bin/env python3
"""
Multi-Step Closed-Loop Inference on Occluded & Corrupted MNIST Digits
(X_occluded -> z_0 -> X_hat_1 -> E(X_hat_1) = z_1 -> X_hat_2 -> ...)

Tests how the Ontological Determination Funnel (Upward E + Downward G coupled on the
32d unit sphere z*) cleans up three severe occlusions on unseen test digits across
iterative closed-loop passes, and compares head-to-head against a Standard Feedforward
Cross-Entropy MLP. Also generates `assets/occlusion_refinement_grid.png`.
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


class TwoWayDecoder:
    """
    2-Layer ReLU + Clipped-Linear Downward Articulation Funnel G operating on normalized z* (32d unit sphere):
    z*_norm (32) -> h1 (128, ReLU) -> I_hat (784, clipped to [0, 1]) -> (x, y, I_hat) (2352).
    """
    def __init__(self, d_latent=32, d_hid=128, out_dim=784, lr=3e-3, seed=42):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / d_latent), size=(d_latent, d_hid)).astype(np.float32)
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

    def forward(self, z_star_norm: np.ndarray):
        u1 = z_star_norm @ self.W1 + self.b1
        h1 = np.maximum(0.0, u1)
        u2 = h1 @ self.W2 + self.b2
        i_hat = np.clip(u2, 0.0, 1.0)
        x_hat = np.stack([self.x_flat * i_hat, self.y_flat * i_hat, i_hat], axis=-1).reshape(len(z_star_norm), 2352)
        return i_hat, x_hat, (z_star_norm, u1, h1, u2, i_hat)

    def backward_and_step(self, cache, di_hat: np.ndarray, weight_decay: float = 1e-4):
        z_star_norm, u1, h1, u2, i_hat = cache
        du2 = di_hat * ((u2 > 0.0) & (u2 < 1.0)).astype(np.float32)
        dW2 = h1.T @ du2 + weight_decay * self.W2
        db2 = np.sum(du2, axis=0, keepdims=True)
        dh1 = du2 @ self.W2.T
        du1 = dh1 * (u1 > 0.0).astype(np.float32)
        dW1 = z_star_norm.T @ du1 + weight_decay * self.W1
        db1 = np.sum(du1, axis=0, keepdims=True)

        self.t += 1
        b1_c = 1.0 - 0.9 ** self.t
        b2_c = 1.0 - 0.999 ** self.t
        for W, b, dW, db, mW, vW, mb, vb in [
            (self.W1, self.b1, dW1, db1, self.mW1, self.vW1, self.mb1, self.mb1),
            (self.W2, self.b2, dW2, db2, self.mW2, self.vW2, self.mb2, self.vb2),
        ]:
            mW[:] = 0.9 * mW + 0.1 * dW
            vW[:] = 0.999 * vW + 0.001 * (dW ** 2)
            W -= self.lr * (mW / b1_c) / (np.sqrt(vW / b2_c) + 1e-8)
            mb[:] = 0.9 * mb + 0.1 * db
            vb[:] = 0.999 * vb + 0.001 * (db ** 2)
            b -= self.lr * (mb / b1_c) / (np.sqrt(vb / b2_c) + 1e-8)


def apply_occlusions(images_uint8: np.ndarray, mode: str, seed: int = 123):
    """
    Create occluded/corrupted 28x28 uint8 images and a binary mask M (1 = corrupted/occluded region, 0 = known).
    Modes:
      - 'picket_fence': 3 horizontal slats erased + 2 bright vertical fence bars drawn across the digit
      - 'center_blackout': a 10x12 rectangular blackout wiping out the core of the digit
      - 'heavy_clutter': 45% random stroke dropout + 15% bright random static across the canvas
    """
    rng = np.random.default_rng(seed)
    out = images_uint8.copy()
    n, h, w = out.shape
    occluded_mask = np.zeros((n, h, w), dtype=np.float32)

    if mode == "picket_fence":
        for r_start in [7, 13, 19]:
            out[:, r_start:r_start + 2, :] = 0
            occluded_mask[:, r_start:r_start + 2, :] = 1.0
        for c_bar in [9, 18]:
            out[:, 4:24, c_bar:c_bar + 2] = 220
            occluded_mask[:, 4:24, c_bar:c_bar + 2] = 1.0
    elif mode == "center_blackout":
        out[:, 9:19, 8:20] = 0
        occluded_mask[:, 9:19, 8:20] = 1.0
    elif mode == "heavy_clutter":
        drop = rng.random((n, h, w)) < 0.45
        clutter = rng.random((n, h, w)) < 0.15
        out[drop] = 0
        out[clutter] = rng.integers(140, 255, size=int(np.sum(clutter)), dtype=np.uint8)
        occluded_mask = (drop | clutter).astype(np.float32)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return out, occluded_mask.reshape(n, 784)


def random_training_corruption(ib_clean: np.ndarray, x_flat: np.ndarray, y_flat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Generate diverse occlusions (missing bars, bright fence bars, box cutouts, and clutter) during training
    so E learns to slice off both missing-stroke gaps and bright occluder clutter via ReLU.
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


def main():
    print("Loading MNIST and building 8-bit (x, y, I) in [0, 255]^3 universe...")
    train_img, train_lbl, test_img, test_lbl = load_mnist()
    x_train = build_8bit_tuple_representation(train_img, couple_background=True)
    x_test = build_8bit_tuple_representation(test_img, couple_background=True)
    i_train = (train_img.astype(np.float32) / 255.0).reshape(len(train_img), 784)
    i_test = (test_img.astype(np.float32) / 255.0).reshape(len(test_img), 784)

    print("\n1. Training Standard Feedforward Cross-Entropy MLP Baseline...")
    rng = np.random.default_rng(42)
    std_mlp = AdamMLP(dims=(784, 256, 64, 32, 10), lr=1.5e-3, seed=42)
    bsz = 256
    for ep in range(10):
        perm = rng.permutation(len(i_train))
        for i in range(0, len(i_train), bsz):
            b = perm[i:i + bsz]
            _, u4, _, cache = std_mlp.forward(i_train[b])
            du4 = (softmax(u4) - np.eye(10, dtype=np.float32)[train_lbl[b]]) / len(b)
            std_mlp.backward_and_step(cache, du4)

    print("2. Training Closed-Loop Ontological Determination Funnel (E + G coupled on 32d unit sphere z*)...")
    encoder = AdamMLP(dims=(2352, 256, 64, 32, 10), lr=1.5e-3, seed=42)
    decoder = TwoWayDecoder(d_latent=32, d_hid=128, out_dim=784, lr=3e-3, seed=42)

    t0 = time.time()
    for ep in range(12):
        perm = rng.permutation(len(x_train))
        for i in range(0, len(x_train), bsz):
            b = perm[i:i + bsz]
            xb_clean = x_train[b]
            yb = train_lbl[b]
            ib_clean = i_train[b]

            z_clean, u4_clean, _, cache_clean = encoder.forward(xb_clean)
            z_clean_n, _ = normalize_rows(z_clean)

            i_hat, x_hat, dec_cache = decoder.forward(z_clean_n)
            di_hat = (i_hat - ib_clean) * (2.0 / len(b))
            decoder.backward_and_step(dec_cache, di_hat)

            xb_corr = random_training_corruption(ib_clean, decoder.x_flat, decoder.y_flat, rng)
            mix_input = 0.75 * xb_corr + 0.25 * x_hat
            z_corr, u4_corr, _, cache_corr = encoder.forward(mix_input)
            z_corr_n, corr_norms = normalize_rows(z_corr)

            dz_match_n = (z_corr_n - z_clean_n) * (2.0 / len(b))
            proj = np.sum(dz_match_n * z_corr_n, axis=-1, keepdims=True)
            dz_match = (dz_match_n - z_corr_n * proj) / corr_norms

            one_hot = np.eye(10, dtype=np.float32)[yb]
            du4_c = (softmax(u4_clean) - one_hot) / len(b)
            dz_grp_c = latent_group_orthogonality_grad(z_clean, yb) * 1.5
            encoder.backward_and_step(cache_clean, du4_c, dz_star_extra=dz_grp_c)

            du4_r = (softmax(u4_corr) - one_hot) / len(b)
            dz_grp_r = latent_group_orthogonality_grad(z_corr, yb) * 1.0 + dz_match * 2.0
            encoder.backward_and_step(cache_corr, du4_r, dz_star_extra=dz_grp_r)

    print(f"   Trained E + G in {time.time() - t0:.2f}s.")
    protos = compute_prototypes(encoder, x_train[:10000], train_lbl[:10000])

    z_test_clean, _, _, _ = encoder.forward(x_test)
    z_test_clean_n, _ = normalize_rows(z_test_clean)

    print("\n3. Running Multi-Step Closed-Loop Inference (Step 0 -> Step 1 -> Step 2 -> Step 3 -> Step 4)...")
    all_metrics = {}
    grid_samples = None

    for mode in ["picket_fence", "center_blackout", "heavy_clutter"]:
        occ_uint8, occ_mask = apply_occlusions(test_img, mode=mode, seed=42)
        x_occ = build_8bit_tuple_representation(occ_uint8, couple_background=True)
        i_occ = (occ_uint8.astype(np.float32) / 255.0).reshape(len(test_img), 784)

        _, u4_std, _, _ = std_mlp.forward(i_occ)
        std_acc = float(np.mean(np.argmax(u4_std, axis=-1) == test_lbl) * 100.0)

        cur_i = i_occ.copy()
        cur_x = x_occ.copy()
        z_0, _, _, _ = encoder.forward(x_occ)
        z_0_n, _ = normalize_rows(z_0)
        z_running_n = z_0_n.copy()

        step_stats = []
        snapshots = [i_occ[:200].copy()]

        for step in range(5):
            z_k, u4_k, _, _ = encoder.forward(cur_x)
            z_k_n, _ = normalize_rows(z_k)
            if step > 0:
                z_running_n, _ = normalize_rows(0.45 * z_0_n + 0.55 * z_k_n)
            else:
                z_running_n = z_k_n

            sims = z_running_n @ protos.T
            preds = np.argmax(sims, axis=-1)
            acc_k = float(np.mean(preds == test_lbl) * 100.0)
            cos_to_clean = float(np.mean(np.sum(z_running_n * z_test_clean_n, axis=-1)))
            pixel_mse = float(np.mean((cur_i - i_test) ** 2))

            step_stats.append({
                "step": step,
                "accuracy_pct": round(acc_k, 2),
                "latent_cosine_to_clean": round(cos_to_clean, 4),
                "pixel_mse_to_clean": round(pixel_mse, 4),
            })

            temp = 10.0 + 5.0 * step
            weights_10 = softmax(sims * temp)
            z_proto_n, _ = normalize_rows(weights_10 @ protos)
            z_step_n, _ = normalize_rows((0.60 - 0.08 * step) * z_running_n + (0.40 + 0.08 * step) * z_proto_n)

            i_hat_k, _, _ = decoder.forward(z_step_n)

            occ_fill = min(0.75 + 0.06 * step, 0.95)
            unocc_denoise = 0.25 + 0.05 * step
            cur_i = (
                occ_mask * ((1.0 - occ_fill) * i_occ + occ_fill * i_hat_k)
                + (1.0 - occ_mask) * ((1.0 - unocc_denoise) * i_occ + unocc_denoise * i_hat_k)
            )
            cur_x = np.stack(
                [decoder.x_flat * cur_i, decoder.y_flat * cur_i, cur_i], axis=-1
            ).reshape(len(test_img), 2352)

            if step < 4:
                snapshots.append(cur_i[:200].copy())

        if mode == "picket_fence":
            grid_samples = (i_test[:200], snapshots, test_lbl[:200])

        print(f"\n  [{mode.upper()}] Standard Open-Loop MLP Acc: {std_acc:.2f}%")
        for st in step_stats:
            print(
                f"    Step {st['step']} -> Acc: {st['accuracy_pct']:6.2f}% | "
                f"Latent Cos to Clean z*: {st['latent_cosine_to_clean']:.4f} | "
                f"Canvas MSE to Clean Digit: {st['pixel_mse_to_clean']:.4f}"
            )
        all_metrics[mode] = {
            "standard_mlp_open_loop_acc": round(std_acc, 2),
            "closed_loop_steps": step_stats,
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
