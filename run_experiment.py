#!/usr/bin/env python3
"""
Ontological Determination Funnel on MNIST (Pure NumPy + Apple Accelerate / OpenBLAS)

Tests the architectural principles derived from Ontological Clarity (Not a ToE):
1. Full 8-Bit Universe Representation:
   - Every image is its own [0, 255]^3 universe where x in [0, 255], y in [0, 255], and I in [0, 255].
   - Each pixel is a 3D tuple (x_i, y_i, I_i), forming a 784 x 3 = 2,352 dimensional input field.
   - Any other resolution (e.g. 14x14 or 56x56) is resized onto the canonical [0, 255]^3 canvas.
2. Gradual Lossy Compression via Linear + ReLU Funnel (Zero Convolutions, Zero Self-Attention):
   - Layer 1 performs high-ratio global compression (2352 -> 256) with ReLU cutting off insignificant bits.
   - Subsequent layers gradually compress into the lowest-dimensional representation z* (256 -> 64 -> 32 -> 10).
3. Two Training Pathways:
   - Pathway 2 (Forward-Only 10-Group Determination): Trains only the forward pass by measuring how cleanly
     representations dissolve into 10 mutually orthogonal collective identity groups (tested in both
     few-shot/full-set prototype grouping and purely unsupervised 10-way entropy/orthogonality grouping).
   - Pathway 1 (Diffusion-Like Per-Sample Closed-Loop Convergence): Trains on each few-shot sample until
     convergence by unfolding z* -> X_hat and feeding X_hat back into the input encoder E(X_hat) so
     E(G(z*)) = z* in latent space while anchoring previously converged basins.
"""

import gzip
import json
import struct
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
RESULTS_DIR = EXP_DIR / "results"

MNIST_MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def ensure_mnist_downloaded():
    """Download raw MNIST IDX files into ./data/ if not already present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for fname in MNIST_FILES:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            url = MNIST_MIRROR + fname
            print(f"Downloading {url} -> {fpath}...")
            urllib.request.urlretrieve(url, fpath)


def load_mnist():
    """Load raw 8-bit [0, 255] MNIST images and labels from IDX files."""
    ensure_mnist_downloaded()

    def read_images(path):
        with gzip.open(path, "rb") as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
        return data

    def read_labels(path):
        with gzip.open(path, "rb") as f:
            magic, num = struct.unpack(">II", f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data

    train_img = read_images(DATA_DIR / "train-images-idx3-ubyte.gz")
    train_lbl = read_labels(DATA_DIR / "train-labels-idx1-ubyte.gz")
    test_img = read_images(DATA_DIR / "t10k-images-idx3-ubyte.gz")
    test_lbl = read_labels(DATA_DIR / "t10k-labels-idx1-ubyte.gz")
    return train_img, train_lbl, test_img, test_lbl


def build_8bit_tuple_representation(images_uint8: np.ndarray, couple_background: bool = True) -> np.ndarray:
    """
    Convert (N, H, W) uint8 images in [0, 255] into (N, 784 * 3) 8-bit (x, y, I) tuples in [0, 255]^3.
    - x and y are scaled to the full 8-bit range [0, 255].
    - I preserves the full 8-bit pixel value [0, 255].
    - When couple_background=True, (x_i, y_i) are weighted by I_i/255 so empty background pixels (I_i = 0)
      carry zero spatial energy into the first ReLU cut, while active pixels carry their full 8-bit (x, y, I)
      ratios scaled to [0, 1] for numerical stability.
    """
    n, h, w = images_uint8.shape
    y_grid_8bit = np.linspace(0.0, 255.0, h, dtype=np.float32)[:, None]
    x_grid_8bit = np.linspace(0.0, 255.0, w, dtype=np.float32)[None, :]
    x_coord = np.broadcast_to(x_grid_8bit, (h, w))  # [0, 255]
    y_coord = np.broadcast_to(y_grid_8bit, (h, w))  # [0, 255]

    intensity_8bit = images_uint8.astype(np.float32)  # [0, 255]
    if couple_background:
        weight = intensity_8bit / 255.0
        px = (x_coord[None, :, :] * weight) / 255.0
        py = (y_coord[None, :, :] * weight) / 255.0
        pi = intensity_8bit / 255.0
    else:
        px = np.broadcast_to(x_coord[None, :, :], (n, h, w)) / 255.0
        py = np.broadcast_to(y_coord[None, :, :], (n, h, w)) / 255.0
        pi = intensity_8bit / 255.0

    tuples = np.stack([px, py, pi], axis=-1).reshape(n, h * w * 3)
    return tuples.astype(np.float32)


def resize_to_canonical_universe(images_uint8: np.ndarray, intermediate_res: int) -> np.ndarray:
    """
    Simulate observing the digits at a different resolution (e.g. 14x14 or 56x56) and resizing
    back onto the canonical 28x28 [0, 255]^3 universe.
    """
    out = np.empty_like(images_uint8)
    for idx in range(len(images_uint8)):
        im = Image.fromarray(images_uint8[idx], mode="L")
        im_res = im.resize((intermediate_res, intermediate_res), Image.Resampling.BILINEAR)
        im_canon = im_res.resize((28, 28), Image.Resampling.BILINEAR)
        out[idx] = np.asarray(im_canon, dtype=np.uint8)
    return out


class AdamMLP:
    """
    Pure NumPy Gradual Linear + ReLU Funnel with analytical gradients and Adam optimizer.
    Architecture: 2352 (784 x 3 tuples) -> 256 (ReLU) -> 64 (ReLU) -> 32 (ReLU, z*) -> 10 (Group scores).
    """
    def __init__(self, dims=(2352, 256, 64, 32, 10), lr=1.5e-3, seed=42):
        rng = np.random.default_rng(seed)
        self.dims = dims
        self.lr = lr
        self.W = []
        self.b = []
        self.mW, self.vW = [], []
        self.mb, self.vb = [], []
        self.t = 0
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            w = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out)).astype(np.float32)
            bias = np.zeros((1, fan_out), dtype=np.float32)
            self.W.append(w)
            self.b.append(bias)
            self.mW.append(np.zeros_like(w))
            self.vW.append(np.zeros_like(w))
            self.mb.append(np.zeros_like(bias))
            self.vb.append(np.zeros_like(bias))

    def forward(self, x: np.ndarray):
        u1 = x @ self.W[0] + self.b[0]
        z1 = np.maximum(0.0, u1)
        u2 = z1 @ self.W[1] + self.b[1]
        z2 = np.maximum(0.0, u2)
        u3 = z2 @ self.W[2] + self.b[2]
        z_star = np.maximum(0.0, u3)
        u4 = z_star @ self.W[3] + self.b[3]
        g_cone = np.maximum(0.0, u4)
        cache = (x, u1, z1, u2, z2, u3, z_star, u4, g_cone)
        return z_star, u4, g_cone, cache

    def backward_and_step(self, cache, du4: np.ndarray, dz_star_extra: np.ndarray = None, weight_decay: float = 1e-4):
        x, u1, z1, u2, z2, u3, z_star, u4, g_cone = cache
        dW3 = z_star.T @ du4 + weight_decay * self.W[3]
        db3 = np.sum(du4, axis=0, keepdims=True)
        dz_star = du4 @ self.W[3].T
        if dz_star_extra is not None:
            dz_star = dz_star + dz_star_extra

        du3 = dz_star * (u3 > 0.0).astype(np.float32)
        dW2 = z2.T @ du3 + weight_decay * self.W[2]
        db2 = np.sum(du3, axis=0, keepdims=True)
        dz2 = du3 @ self.W[2].T

        du2 = dz2 * (u2 > 0.0).astype(np.float32)
        dW1 = z1.T @ du2 + weight_decay * self.W[1]
        db1 = np.sum(du2, axis=0, keepdims=True)
        dz1 = du2 @ self.W[1].T

        du1 = dz1 * (u1 > 0.0).astype(np.float32)
        dW0 = x.T @ du1 + weight_decay * self.W[0]
        db0 = np.sum(du1, axis=0, keepdims=True)

        grads_W = [dW0, dW1, dW2, dW3]
        grads_b = [db0, db1, db2, db3]
        self._adam_update(grads_W, grads_b)

    def _adam_update(self, grads_W, grads_b, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        b1_corr = 1.0 - beta1 ** self.t
        b2_corr = 1.0 - beta2 ** self.t
        for i in range(len(self.W)):
            self.mW[i] = beta1 * self.mW[i] + (1.0 - beta1) * grads_W[i]
            self.vW[i] = beta2 * self.vW[i] + (1.0 - beta2) * (grads_W[i] ** 2)
            m_hat = self.mW[i] / b1_corr
            v_hat = self.vW[i] / b2_corr
            self.W[i] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

            self.mb[i] = beta1 * self.mb[i] + (1.0 - beta1) * grads_b[i]
            self.vb[i] = beta2 * self.vb[i] + (1.0 - beta2) * (grads_b[i] ** 2)
            mb_hat = self.mb[i] / b1_corr
            vb_hat = self.vb[i] / b2_corr
            self.b[i] -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)


def normalize_rows(v: np.ndarray, eps: float = 1e-6):
    norms = np.sqrt(np.sum(v * v, axis=-1, keepdims=True) + eps)
    return v / norms, norms


def softmax(u: np.ndarray) -> np.ndarray:
    u_shift = u - np.max(u, axis=-1, keepdims=True)
    exp_u = np.exp(u_shift)
    return exp_u / np.sum(exp_u, axis=-1, keepdims=True)


def latent_group_orthogonality_grad(z_star: np.ndarray, y: np.ndarray):
    bsz = len(z_star)
    z_norm, norms = normalize_rows(z_star)
    sim = z_norm @ z_norm.T
    target = (y[:, None] == y[None, :]).astype(np.float32)
    diff = (sim - target) * (2.0 / (bsz * bsz))
    dz_norm = (diff + diff.T) @ z_norm
    proj = np.sum(dz_norm * z_norm, axis=-1, keepdims=True)
    dz_star = (dz_norm - z_norm * proj) / norms
    return dz_star


def evaluate_model(model: AdamMLP, x_test: np.ndarray, y_test: np.ndarray, prototypes: np.ndarray = None) -> float:
    z_star, u4, _, _ = model.forward(x_test)
    if prototypes is not None:
        z_norm, _ = normalize_rows(z_star)
        preds = np.argmax(z_norm @ prototypes.T, axis=-1)
    else:
        preds = np.argmax(u4, axis=-1)
    return float(np.mean(preds == y_test) * 100.0)


def compute_prototypes(model: AdamMLP, x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
    z_star, _, _, _ = model.forward(x_data)
    z_norm, _ = normalize_rows(z_star)
    protos = np.zeros((10, z_star.shape[1]), dtype=np.float32)
    for c in range(10):
        mask = (y_data == c)
        if np.any(mask):
            mean_c = np.mean(z_norm[mask], axis=0, keepdims=True)
            protos[c] = normalize_rows(mean_c)[0][0]
    return protos


def run_pathway2_forward_grouping(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
    print("\n=== Pathway 2A: Forward-Only 10-Group Determination (Few-Shot & Full Dataset) ===")
    results = {}
    full_model = None
    full_protos = None

    for shots_per_class in [1, 5, 10, 50, 6000]:
        rng = np.random.default_rng(42)
        if shots_per_class < 6000:
            indices = []
            for c in range(10):
                c_idx = np.where(y_train == c)[0]
                chosen = rng.choice(c_idx, size=shots_per_class, replace=False)
                indices.append(chosen)
            sub_idx = np.concatenate(indices)
            x_sub, y_sub = x_train[sub_idx], y_train[sub_idx]
            epochs = 100 if shots_per_class <= 10 else 60
            batch_size = min(len(x_sub), 64)
            lr = 2e-3
        else:
            x_sub, y_sub = x_train, y_train
            epochs = 12
            batch_size = 256
            lr = 1.5e-3

        model = AdamMLP(dims=(2352, 256, 64, 32, 10), lr=lr, seed=42)
        t0 = time.time()
        n_samples = len(x_sub)

        for ep in range(epochs):
            perm = rng.permutation(n_samples)
            for i in range(0, n_samples, batch_size):
                idx = perm[i:i + batch_size]
                xb = x_sub[idx]
                yb = y_sub[idx]
                bsz = len(xb)

                z_star, u4, _, cache = model.forward(xb)
                probs = softmax(u4)
                one_hot = np.eye(10, dtype=np.float32)[yb]
                du4 = (probs - one_hot) / bsz
                dz_group = latent_group_orthogonality_grad(z_star, yb) * 1.5
                model.backward_and_step(cache, du4, dz_star_extra=dz_group)

        elapsed = time.time() - t0
        protos = compute_prototypes(model, x_sub[:5000], y_sub[:5000])
        acc_head = evaluate_model(model, x_test, y_test, prototypes=None)
        acc_proto = evaluate_model(model, x_test, y_test, prototypes=protos)

        _, _, _, cache_test = model.forward(x_test[:1000])
        z1_test, z_star_test = cache_test[2], cache_test[6]
        l1_cut_pct = float(np.mean(z1_test == 0.0) * 100.0)
        z_cut_pct = float(np.mean(z_star_test == 0.0) * 100.0)

        sim_protos = protos @ protos.T
        off_diag = sim_protos[~np.eye(10, dtype=bool)]
        mean_off_diag_cos = float(np.mean(np.abs(off_diag)))

        label_str = f"{shots_per_class}-shot ({n_samples} samples)" if shots_per_class < 6000 else "Full MNIST (60,000 samples)"
        print(
            f"  [{label_str:28s}] Group Acc: {acc_head:6.2f}% | Proto Acc: {acc_proto:6.2f}% | "
            f"Off-Diag Cos: {mean_off_diag_cos:.3f} | L1 ReLU Cut: {l1_cut_pct:4.1f}% | z* Cut: {z_cut_pct:4.1f}% | {elapsed:.2f}s"
        )
        results[f"shots_{shots_per_class}"] = {
            "samples": int(n_samples),
            "group_head_accuracy": round(acc_head, 2),
            "prototype_accuracy": round(acc_proto, 2),
            "inter_group_off_diagonal_cosine": round(mean_off_diag_cos, 4),
            "l1_relu_cutoff_pct": round(l1_cut_pct, 1),
            "z_star_relu_cutoff_pct": round(z_cut_pct, 1),
            "seconds": round(elapsed, 2),
        }
        if shots_per_class == 6000:
            full_model = model
            full_protos = protos

    return results, full_model, full_protos


def run_pathway1_diffusion_fewshot(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
    print("\n=== Pathway 1: Diffusion-Like Per-Sample Closed-Loop Convergence (E(G(z*)) = z*) ===")
    results = {}

    y_grid = np.linspace(0.0, 1.0, 28, dtype=np.float32)[:, None]
    x_grid = np.linspace(0.0, 1.0, 28, dtype=np.float32)[None, :]
    x_flat = np.broadcast_to(x_grid, (28, 28)).reshape(1, 784)
    y_flat = np.broadcast_to(y_grid, (28, 28)).reshape(1, 784)

    for shots_per_class in [1, 5, 10, 50]:
        rng = np.random.default_rng(42)
        indices = []
        for c in range(10):
            c_idx = np.where(y_train == c)[0]
            chosen = rng.choice(c_idx, size=shots_per_class, replace=False)
            indices.append(chosen)
        sub_idx = np.column_stack(indices).reshape(-1)
        x_sub, y_sub = x_train[sub_idx], y_train[sub_idx]
        i_sub = x_sub[:, 2::3]

        encoder = AdamMLP(dims=(2352, 256, 64, 32, 10), lr=2e-3, seed=42)
        W_dec = rng.normal(0.0, np.sqrt(2.0 / 32), size=(32, 784)).astype(np.float32)
        b_dec = np.zeros((1, 784), dtype=np.float32)
        mW_dec, vW_dec = np.zeros_like(W_dec), np.zeros_like(W_dec)
        mb_dec, vb_dec = np.zeros_like(b_dec), np.zeros_like(b_dec)

        t0 = time.time()
        chunk = 10
        step_stride = 10 if shots_per_class <= 10 else 50
        stages = list(range(chunk, len(x_sub) + 1, step_stride))
        if stages[-1] != len(x_sub):
            stages.append(len(x_sub))

        total_steps = 0
        for end_k in stages:
            xb_all = x_sub[:end_k]
            yb_all = y_sub[:end_k]
            ib_all = i_sub[:end_k]
            n_cur = len(xb_all)
            bsz = min(n_cur, 64)

            n_iters = 35 if shots_per_class <= 10 else 25
            for it in range(n_iters):
                total_steps += 1
                idx = rng.choice(n_cur, size=bsz, replace=False) if n_cur > bsz else np.arange(n_cur)
                xb = xb_all[idx]
                yb = yb_all[idx]
                ib = ib_all[idx]

                z_clean, u4_clean, _, cache_clean = encoder.forward(xb)
                z_clean_norm, _ = normalize_rows(z_clean)

                u_dec = z_clean @ W_dec + b_dec
                i_hat = np.clip(u_dec, 0.0, 1.0)
                x_hat = np.stack([x_flat * i_hat, y_flat * i_hat, i_hat], axis=-1).reshape(len(xb), 2352)

                di_hat = (i_hat - ib) * (2.0 / len(xb)) * ((u_dec > 0.0) & (u_dec < 1.0)).astype(np.float32)
                dW_dec = z_clean.T @ di_hat + 1e-4 * W_dec
                db_dec = np.sum(di_hat, axis=0, keepdims=True)
                mW_dec = 0.9 * mW_dec + 0.1 * dW_dec
                vW_dec = 0.999 * vW_dec + 0.001 * (dW_dec ** 2)
                mb_dec = 0.9 * mb_dec + 0.1 * db_dec
                vb_dec = 0.999 * vb_dec + 0.001 * (db_dec ** 2)
                W_dec -= 3e-3 * (mW_dec / (1 - 0.9 ** total_steps)) / (np.sqrt(vW_dec / (1 - 0.999 ** total_steps)) + 1e-8)
                b_dec -= 3e-3 * (mb_dec / (1 - 0.9 ** total_steps)) / (np.sqrt(vb_dec / (1 - 0.999 ** total_steps)) + 1e-8)

                sigma = 0.25 * (1.0 - it / n_iters) + 0.05
                noise = rng.normal(0.0, sigma, size=xb.shape).astype(np.float32) * (xb > 0).astype(np.float32)
                xb_reobs = np.clip(0.7 * xb + 0.3 * x_hat + noise, 0.0, 1.0)

                z_reobs, u4_reobs, _, cache_reobs = encoder.forward(xb_reobs)
                z_reobs_norm, reobs_norms = normalize_rows(z_reobs)

                dz_match_norm = (z_reobs_norm - z_clean_norm) * (2.0 / len(xb))
                proj = np.sum(dz_match_norm * z_reobs_norm, axis=-1, keepdims=True)
                dz_match = (dz_match_norm - z_reobs_norm * proj) / reobs_norms

                one_hot = np.eye(10, dtype=np.float32)[yb]
                du4_clean = (softmax(u4_clean) - one_hot) / len(xb)
                dz_group_clean = latent_group_orthogonality_grad(z_clean, yb) * 1.5
                encoder.backward_and_step(cache_clean, du4_clean, dz_star_extra=dz_group_clean)

                du4_reobs = (softmax(u4_reobs) - one_hot) / len(xb)
                dz_group_reobs = latent_group_orthogonality_grad(z_reobs, yb) * 1.0 + dz_match * 1.5
                encoder.backward_and_step(cache_reobs, du4_reobs, dz_star_extra=dz_group_reobs)

        elapsed = time.time() - t0
        protos = compute_prototypes(encoder, x_sub, y_sub)
        acc_test = evaluate_model(encoder, x_test, y_test, prototypes=protos)

        z_t, _, _, _ = encoder.forward(x_test)
        i_hat_t = np.clip(z_t @ W_dec + b_dec, 0.0, 1.0)
        x_hat_t = np.stack([x_flat * i_hat_t, y_flat * i_hat_t, i_hat_t], axis=-1).reshape(len(x_test), 2352)
        z_reobs_t, _, _, _ = encoder.forward(x_hat_t)
        zt_n, _ = normalize_rows(z_t)
        zr_n, _ = normalize_rows(z_reobs_t)
        cycle_cos = float(np.mean(np.sum(zt_n * zr_n, axis=-1)))

        z_refined, _ = normalize_rows(0.7 * zt_n + 0.3 * zr_n)
        acc_refined = float(np.mean(np.argmax(z_refined @ protos.T, axis=-1) == y_test) * 100.0)

        print(
            f"  [{shots_per_class:2d}-shot ({len(x_sub):3d} samples)] Direct Acc: {acc_test:6.2f}% | "
            f"Re-Observed Acc: {acc_refined:6.2f}% | Cycle cos(z*, E(G(z*))): {cycle_cos:.4f} | {elapsed:.2f}s"
        )
        results[f"shots_{shots_per_class}"] = {
            "samples": int(len(x_sub)),
            "direct_test_accuracy": round(acc_test, 2),
            "reobserved_test_accuracy": round(acc_refined, 2),
            "latent_cycle_cosine": round(cycle_cos, 4),
            "seconds": round(elapsed, 2),
        }

    return results


def run_unsupervised_10group_test(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
    print("\n=== Pathway 2B: Unsupervised Forward-Only 10-Group Determination (Zero Labels) ===")
    rng = np.random.default_rng(42)

    n_sub = 15000
    x_sub = x_train[:n_sub]

    model = AdamMLP(dims=(2352, 256, 64, 32, 10), lr=2e-3, seed=42)
    t0 = time.time()
    batch_size = 512

    for ep in range(18):
        perm = rng.permutation(n_sub)
        for i in range(0, n_sub, batch_size):
            idx = perm[i:i + batch_size]
            xb = x_sub[idx]
            bsz = len(xb)

            noise1 = rng.normal(0.0, 0.12, size=xb.shape).astype(np.float32) * (xb > 0).astype(np.float32)
            xb_aug = np.clip(xb + noise1, 0.0, 1.0)

            z_a, u4_a, _, cache_a = model.forward(xb)
            z_b, u4_b, _, _ = model.forward(xb_aug)

            xb_n, _ = normalize_rows(xb)
            sim_in = xb_n @ xb_n.T
            pos_thresh = np.percentile(sim_in, 90)
            neg_thresh = np.percentile(sim_in, 50)
            pos_mask = (sim_in >= pos_thresh).astype(np.float32)
            neg_mask = (sim_in <= neg_thresh).astype(np.float32)

            p_a = softmax(u4_a)
            p_b = softmax(u4_b)
            p_mean = np.mean(p_a, axis=0, keepdims=True)

            prob_sim = p_a @ p_a.T
            d_prob_sim = (
                pos_mask * (prob_sim - 1.0) * 2.0
                + neg_mask * (prob_sim - 0.0) * 1.0
            ) / (bsz * bsz)
            grad_pair = 2.0 * (d_prob_sim + d_prob_sim.T) @ p_a

            eps = 1e-6
            grad_ent = (
                -0.15 * (np.log(p_a + eps) + 1.0)
                + 1.50 * (np.log(p_mean + eps) + 1.0)
                + 1.00 * (p_a - p_b)
            ) / bsz

            grad_p_a = grad_pair + grad_ent
            dot_a = np.sum(grad_p_a * p_a, axis=-1, keepdims=True)
            du4_a = p_a * (grad_p_a - dot_a)

            dz_unsup = latent_group_orthogonality_grad(z_a, np.argmax(p_a, axis=-1)) * 0.5
            model.backward_and_step(cache_a, du4_a, dz_star_extra=dz_unsup)

    elapsed = time.time() - t0
    _, u4_test, _, _ = model.forward(x_test)
    pred_groups = np.argmax(u4_test, axis=-1)
    correct = 0
    active_groups = 0
    group_counts = []
    for g in range(10):
        mask = (pred_groups == g)
        cnt = int(np.sum(mask))
        group_counts.append(cnt)
        if cnt > 0:
            active_groups += 1
            best_digit = int(np.argmax(np.bincount(y_test[mask], minlength=10)))
            correct += int(np.sum(y_test[mask] == best_digit))

    purity = 100.0 * correct / len(y_test)
    print(
        f"  [Unsupervised 10-Group Determination] Cluster Purity: {purity:.2f}% | "
        f"Active Groups: {active_groups}/10 | Group Sizes: {group_counts} | {elapsed:.2f}s"
    )
    return {
        "unsupervised_purity_pct": round(purity, 2),
        "active_groups": active_groups,
        "group_sizes_on_10k_test": group_counts,
        "seconds": round(elapsed, 2),
    }


def main():
    print("Loading MNIST and building 8-bit (x, y, I) in [0, 255]^3 tuple representation...")
    train_img, train_lbl, test_img, test_lbl = load_mnist()
    x_train = build_8bit_tuple_representation(train_img, couple_background=True)
    x_test = build_8bit_tuple_representation(test_img, couple_background=True)
    print(f"  Train shape: {x_train.shape} (784 pixels x 3 coordinates [x, y, I]), Test shape: {x_test.shape}")

    p2_results, full_model, full_protos = run_pathway2_forward_grouping(x_train, train_lbl, x_test, test_lbl)
    p1_results = run_pathway1_diffusion_fewshot(x_train, train_lbl, x_test, test_lbl)
    unsup_results = run_unsupervised_10group_test(x_train, train_lbl, x_test, test_lbl)

    print("\n=== Multi-Resolution Universe Test (14x14 and 56x56 Resized to Canonical [0, 255]^3) ===")
    res_results = {}
    for res in [14, 28, 56]:
        if res == 28:
            x_res = x_test
        else:
            resized_imgs = resize_to_canonical_universe(test_img, intermediate_res=res)
            x_res = build_8bit_tuple_representation(resized_imgs, couple_background=True)
        acc = evaluate_model(full_model, x_res, test_lbl, prototypes=full_protos)
        print(f"  Resolution {res:2d}x{res:2d} -> Canonical [0, 255]^3 Universe Accuracy: {acc:.2f}%")
        res_results[f"res_{res}x{res}"] = round(acc, 2)

    summary = {
        "representation": "784 x 3 (x, y, I) in [0, 255]^3 (2,352 dims -> 256 -> 64 -> 32 -> 10 Linear+ReLU Funnel)",
        "pathway2_forward_grouping": p2_results,
        "pathway1_diffusion_fewshot_closed_loop": p1_results,
        "pathway2b_unsupervised_10group": unsup_results,
        "multi_resolution_generalization": res_results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved full experimental results to {results_path}")


if __name__ == "__main__":
    main()
