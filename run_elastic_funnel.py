#!/usr/bin/env python3
"""
Stage 1 of Rethinking Determination Funnel Under 'The Dual Cut':
Gradual Supervised Recognition Discovery (Starting from d_in -> 10, Zero Pre-Built Cascade).

Key Principles:
1. Target 10 (The Actual Problem Statement):
   - Keep the terminal `-> 10` layer (`d_out = 10`). Never prescribe `11` or set an artificial
     `min_width = 10` floor (`min_width = 1`):
     * Starting from `2,352` coupled `(x, y, I)` tuples (`2352 -> 10`), `L = 3` initializes at
       `round(sqrt(12 / 10) * 10) = 11`, expands to `13`, and returns to `11` (`[2352, 105, 32, 11, 10]`,
       `250,940` params), where `round(sqrt(11 / 10) * 10) = 10` terminates discovery.
     * Starting directly from `784` pixel intensities (`784 -> 10`), where vector slot order already
       encodes spatial position `(x_i, y_i)`, `L = 3` initializes at `round(sqrt(21 / 10) * 10) = 14`
       and contracts by 4 units (`14 -> 10`) to `[784, 85, 30, 10, 10]` (`69,725` params), landing
       directly on the exact class count `10`.
"""

import copy
import json
import time

import numpy as np

from run_experiment import (
    RESULTS_DIR,
    build_8bit_tuple_representation,
    latent_group_orthogonality_grad,
    load_mnist,
    normalize_rows,
    softmax,
)


def midpoint_width(d_in: int, d_out: int = 10) -> int:
    """Calculate the multiplicative midpoint width: round(sqrt(d_in / d_out) * d_out)."""
    return int(np.round(np.sqrt(float(d_in) / float(d_out)) * float(d_out)))


class ElasticFunnel:
    """
    Variable-depth, variable-width Linear + ReLU Funnel (`d_in -> d_1 -> ... -> d_L -> 10`)
    trained by supervised label loss at the terminal `10` layer and hidden ReLU cuts,
    with energy-preserving collinear cone folding and spent-scaffolding pruning.
    """

    def __init__(self, layer_dims, lr=1.6e-3, seed=42):
        self.rng = np.random.default_rng(seed)
        self.lr = lr
        self.W = []
        self.b = []
        self.mW, self.vW = [], []
        self.mb, self.vb = [], []
        self.t = 0

        for i in range(len(layer_dims) - 1):
            fan_in, fan_out = int(layer_dims[i]), int(layer_dims[i + 1])
            w = self.rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out)).astype(np.float32)
            bias = np.zeros((1, fan_out), dtype=np.float32)
            self.W.append(w)
            self.b.append(bias)
            self.mW.append(np.zeros_like(w))
            self.vW.append(np.zeros_like(w))
            self.mb.append(np.zeros_like(bias))
            self.vb.append(np.zeros_like(bias))

    @property
    def dims(self):
        return [int(self.W[0].shape[0])] + [int(w.shape[1]) for w in self.W]

    @property
    def hidden_dims(self):
        return [int(w.shape[1]) for w in self.W[:-1]]

    @property
    def param_count(self):
        return int(sum(w.size + b.size for w, b in zip(self.W, self.b)))

    def forward(self, x: np.ndarray):
        pre_acts = []
        post_acts = [x]
        h = x
        for i in range(len(self.W) - 1):
            u = h @ self.W[i] + self.b[i]
            h = np.maximum(0.0, u)
            pre_acts.append(u)
            post_acts.append(h)

        u_out = h @ self.W[-1] + self.b[-1]
        pre_acts.append(u_out)
        z_top = post_acts[-1]
        return z_top, u_out, (pre_acts, post_acts)

    def backward_and_step(
        self,
        cache,
        du_out: np.ndarray,
        yb: np.ndarray = None,
        ortho_weights=None,
        weight_decay: float = 1e-4,
    ):
        pre_acts, post_acts = cache
        n_layers = len(self.W)
        grads_W = [None] * n_layers
        grads_b = [None] * n_layers

        z_top = post_acts[-1]
        grads_W[-1] = z_top.T @ du_out + weight_decay * self.W[-1]
        grads_b[-1] = np.sum(du_out, axis=0, keepdims=True)

        dh = du_out @ self.W[-1].T

        for i in range(n_layers - 2, -1, -1):
            if yb is not None:
                w_ortho = 1.5 if i == n_layers - 2 else 0.15
                if ortho_weights is not None and i < len(ortho_weights):
                    w_ortho = ortho_weights[i]
                if w_ortho > 0.0:
                    dh = dh + latent_group_orthogonality_grad(post_acts[i + 1], yb) * w_ortho

            u_i = pre_acts[i]
            h_prev = post_acts[i]
            du_i = dh * (u_i > 0.0).astype(np.float32)
            col_norms = np.sqrt(np.sum(self.W[i] ** 2, axis=0, keepdims=True) + 1e-8)
            grads_W[i] = h_prev.T @ du_i + weight_decay * self.W[i] + 3e-5 * (self.W[i] / col_norms)
            grads_b[i] = np.sum(du_i, axis=0, keepdims=True)
            if i > 0:
                dh = du_i @ self.W[i].T

        self._adam_update(grads_W, grads_b)
        return grads_W

    def input_gradient_from_supervised_loss(self, x: np.ndarray, yb: np.ndarray, ortho_weights=None):
        """
        Pass `x` through FROZEN E, measure the supervised recognition loss against the
        desired label `yb` (terminal CE + internal label-group orthogonality), and return
        the analytical gradient `dx` with respect to `x` WITHOUT modifying E's weights.
        """
        _, u_out, (pre_acts, post_acts) = self.forward(x)
        n_layers = len(self.W)
        bsz = len(x)
        probs = softmax(u_out)
        one_hot = np.eye(10, dtype=np.float32)[yb]
        du_out = (probs - one_hot) / bsz

        dh = du_out @ self.W[-1].T
        for i in range(n_layers - 2, -1, -1):
            w_ortho = 1.5 if i == n_layers - 2 else 0.15
            if ortho_weights is not None and i < len(ortho_weights):
                w_ortho = ortho_weights[i]
            if w_ortho > 0.0:
                dh = dh + latent_group_orthogonality_grad(post_acts[i + 1], yb) * w_ortho
            du_i = dh * (pre_acts[i] > 0.0).astype(np.float32)
            dh = du_i @ self.W[i].T
        return dh, probs, post_acts

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

    def breathe_layer(
        self,
        layer_idx: int,
        x_val: np.ndarray,
        d_out: int = 10,
        min_width: int = 1,
        cos_thresh: float = 0.90,
        min_rel_energy: float = 0.05,
        allow_expand: bool = True,
    ):
        _, _, (_, post_acts) = self.forward(x_val)
        z = post_acts[layer_idx + 1]
        d_cur = z.shape[1]

        W_in = self.W[layer_idx].copy()
        b_in = self.b[layer_idx].copy()
        mW_in, vW_in = self.mW[layer_idx].copy(), self.vW[layer_idx].copy()
        mb_in, vb_in = self.mb[layer_idx].copy(), self.vb[layer_idx].copy()

        W_out = self.W[layer_idx + 1].copy()
        mW_out, vW_out = self.mW[layer_idx + 1].copy(), self.vW[layer_idx + 1].copy()

        z_rms = np.sqrt(np.mean(z ** 2, axis=0) + 1e-12)
        out_norms = np.sqrt(np.sum(W_out ** 2, axis=1) + 1e-12)
        energy = (z_rms ** 2) * (out_norms ** 2)

        order = np.argsort(-energy)
        z_unit = z / (np.sqrt(np.sum(z ** 2, axis=0, keepdims=True)) + 1e-12)
        sim_mat = z_unit.T @ z_unit

        alive = np.ones(d_cur, dtype=bool)
        folded_count = 0
        if d_cur > min_width:
            for idx_pos, j in enumerate(order):
                if not alive[j]:
                    continue
                for k in order[idx_pos + 1:]:
                    if not alive[k]:
                        continue
                    if np.sum(alive) <= min_width:
                        break
                    if sim_mat[j, k] >= cos_thresh and z_rms[j] > 1e-6:
                        scale = float(z_rms[k] / z_rms[j])
                        W_out[j, :] += scale * W_out[k, :]
                        alive[k] = False
                        folded_count += 1

        keep_idx = np.where(alive)[0]
        W_in, b_in = W_in[:, keep_idx], b_in[:, keep_idx]
        mW_in, vW_in = mW_in[:, keep_idx], vW_in[:, keep_idx]
        mb_in, vb_in = mb_in[:, keep_idx], vb_in[:, keep_idx]
        W_out = W_out[keep_idx, :]
        mW_out, vW_out = mW_out[keep_idx, :], vW_out[keep_idx, :]
        z_surv = z[:, keep_idx]

        z_rms_s = np.sqrt(np.mean(z_surv ** 2, axis=0))
        active_rate = np.mean(z_surv > 0.0, axis=0)
        out_norms_s = np.sqrt(np.sum(W_out ** 2, axis=1))
        energy_s = (z_rms_s ** 2) * (out_norms_s ** 2)
        mean_e = float(np.mean(energy_s)) + 1e-12

        e_order = np.argsort(-energy_s)
        valid_mask = (energy_s >= min_rel_energy * mean_e) & (active_rate >= 0.005)
        n_valid = int(np.sum(valid_mask))
        n_keep = max(min_width, n_valid)
        final_local_idx = np.sort(e_order[:n_keep])
        pruned_energy_count = len(keep_idx) - len(final_local_idx)

        self.W[layer_idx] = W_in[:, final_local_idx]
        self.b[layer_idx] = b_in[:, final_local_idx]
        self.mW[layer_idx] = mW_in[:, final_local_idx]
        self.vW[layer_idx] = vW_in[:, final_local_idx]
        self.mb[layer_idx] = mb_in[:, final_local_idx]
        self.vb[layer_idx] = vb_in[:, final_local_idx]

        self.W[layer_idx + 1] = W_out[final_local_idx, :]
        self.mW[layer_idx + 1] = mW_out[final_local_idx, :]
        self.vW[layer_idx + 1] = vW_out[final_local_idx, :]

        expanded_count = 0
        if allow_expand and folded_count == 0 and pruned_energy_count == 0:
            min_e_ratio = float(np.min(energy_s) / mean_e)
            if min_e_ratio > min_rel_energy * 2.5:
                add_d = max(2, int(np.round(np.sqrt(float(d_cur) / float(d_out)))))
                expanded_count = self.expand_layer_orthogonal(layer_idx, add_d)

        return {
            "folded": int(folded_count),
            "pruned_energy": int(pruned_energy_count),
            "expanded": int(expanded_count),
            "width_before": int(d_cur),
            "width_after": int(self.W[layer_idx].shape[1]),
        }

    def expand_layer_orthogonal(self, layer_idx: int, add_dim: int):
        if add_dim <= 0:
            return 0
        W_in = self.W[layer_idx]
        fan_in, d_cur = W_in.shape

        raw_new = self.rng.normal(0.0, 1.0, size=(fan_in, add_dim)).astype(np.float32)
        if fan_in > d_cur:
            q_exist, _ = np.linalg.qr(W_in, mode="reduced")
            raw_new = raw_new - q_exist @ (q_exist.T @ raw_new)
        q_new, _ = np.linalg.qr(raw_new, mode="reduced")
        target_col_norm = float(np.median(np.sqrt(np.sum(W_in ** 2, axis=0))))
        if target_col_norm < 1e-4:
            target_col_norm = np.sqrt(2.0)
        W_add_in = (q_new[:, :add_dim] * target_col_norm).astype(np.float32)
        b_add_in = np.zeros((1, add_dim), dtype=np.float32)

        fan_out_next = self.W[layer_idx + 1].shape[1]
        W_add_out = self.rng.normal(0.0, 0.02 / np.sqrt(d_cur + add_dim), size=(add_dim, fan_out_next)).astype(np.float32)

        self.W[layer_idx] = np.concatenate([W_in, W_add_in], axis=1)
        self.b[layer_idx] = np.concatenate([self.b[layer_idx], b_add_in], axis=1)
        self.mW[layer_idx] = np.concatenate([self.mW[layer_idx], np.zeros_like(W_add_in)], axis=1)
        self.vW[layer_idx] = np.concatenate([self.vW[layer_idx], np.zeros_like(W_add_in)], axis=1)
        self.mb[layer_idx] = np.concatenate([self.mb[layer_idx], np.zeros_like(b_add_in)], axis=1)
        self.vb[layer_idx] = np.concatenate([self.vb[layer_idx], np.zeros_like(b_add_in)], axis=1)

        self.W[layer_idx + 1] = np.concatenate([self.W[layer_idx + 1], W_add_out], axis=0)
        self.mW[layer_idx + 1] = np.concatenate([self.mW[layer_idx + 1], np.zeros_like(W_add_out)], axis=0)
        self.vW[layer_idx + 1] = np.concatenate([self.vW[layer_idx + 1], np.zeros_like(W_add_out)], axis=0)
        return int(add_dim)


def compute_layer_prototypes(model: ElasticFunnel, x_ref: np.ndarray, y_ref: np.ndarray, n_classes: int = 10):
    """Compute normalized class prototypes at every hidden layer `z_1, ..., z_L`."""
    _, _, (_, post_acts) = model.forward(x_ref)
    hidden_protos = []
    for h in post_acts[1:]:
        h_n, _ = normalize_rows(h)
        protos_l = np.zeros((n_classes, h.shape[1]), dtype=np.float32)
        for c in range(n_classes):
            mask = (y_ref == c)
            if np.any(mask):
                mean_c = np.mean(h_n[mask], axis=0, keepdims=True)
                protos_l[c] = normalize_rows(mean_c)[0][0]
        hidden_protos.append(protos_l)
    return hidden_protos


def compute_internal_loss(model: ElasticFunnel, x: np.ndarray, hidden_protos: list):
    """
    Compute the per-sample intrinsic Internal Loss `L_internal(X)` inside E without needing an external label:
      1. Output cone indeterminacy: `1 - (p_top1 - p_top2)` at the terminal `10` layer, and
      2. Internal cone prototype residual: `1 - cos(z_l(X), proto^{(l)}_{y_hat})` across the hidden ReLU cuts.
    """
    _, u_out, (pre_acts, post_acts) = model.forward(x)
    probs = softmax(u_out)
    y_hat = np.argmax(probs, axis=-1)

    top2 = np.sort(probs, axis=-1)[:, -2:]
    out_indeterminacy = 1.0 - (top2[:, 1] - top2[:, 0])

    if len(hidden_protos) == 0:
        return out_indeterminacy[:, None].astype(np.float32), probs, (pre_acts, post_acts)

    weights = [0.20, 0.30, 0.50] if len(hidden_protos) == 3 else ([0.35, 0.65] if len(hidden_protos) == 2 else [1.0] * len(hidden_protos))
    w_sum = sum(weights[:len(hidden_protos)])
    total_cone_res = np.zeros(len(x), dtype=np.float32)

    for l_idx, (h, protos_l) in enumerate(zip(post_acts[1:], hidden_protos)):
        h_n, _ = normalize_rows(h)
        target_p = protos_l[y_hat]
        cos_sim = np.sum(h_n * target_p, axis=-1)
        res_l = np.clip(1.0 - cos_sim, 0.0, 1.0)
        w_l = weights[l_idx] / w_sum
        total_cone_res += w_l * res_l

    internal_loss = (0.45 * out_indeterminacy + 0.55 * total_cone_res)[:, None].astype(np.float32)
    return internal_loss, probs, (pre_acts, post_acts)


def evaluate_funnel_metrics(model: ElasticFunnel, x_eval: np.ndarray, y_eval: np.ndarray, x_proto_ref: np.ndarray, y_proto_ref: np.ndarray, n_classes: int = 10):
    z_top, u_out, (_, post_acts) = model.forward(x_eval)
    probs = softmax(u_out)
    eps = 1e-8
    ce_loss = float(-np.mean(np.log(probs[np.arange(len(y_eval)), y_eval] + eps)))
    head_acc = float(np.mean(np.argmax(u_out, axis=-1) == y_eval) * 100.0)

    hidden_protos = compute_layer_prototypes(model, x_proto_ref, y_proto_ref, n_classes=n_classes)
    int_loss, _, _ = compute_internal_loss(model, x_eval, hidden_protos)
    mean_int_loss = float(np.mean(int_loss))

    if len(hidden_protos) > 0:
        protos_top = hidden_protos[-1]
        z_top_n, _ = normalize_rows(z_top)
        proto_preds = np.argmax(z_top_n @ protos_top.T, axis=-1)
        proto_acc = float(np.mean(proto_preds == y_eval) * 100.0)
        sim_p = protos_top @ protos_top.T
    else:
        proto_acc = head_acc
        u_n, _ = normalize_rows(np.maximum(0.0, u_out))
        protos_u = np.zeros((n_classes, n_classes), dtype=np.float32)
        for c in range(n_classes):
            protos_u[c] = normalize_rows(np.mean(u_n[y_eval == c], axis=0, keepdims=True))[0][0]
        sim_p = protos_u @ protos_u.T

    off_diag = sim_p[~np.eye(n_classes, dtype=bool)]
    off_diag_cos = float(np.mean(np.abs(off_diag)))

    layer_cutoffs = []
    for h in post_acts[1:]:
        layer_cutoffs.append(round(float(np.mean(h == 0.0) * 100.0), 1))

    return {
        "ce_loss": round(ce_loss, 4),
        "mean_internal_loss": round(mean_int_loss, 4),
        "head_acc_pct": round(head_acc, 2),
        "proto_acc_pct": round(proto_acc, 2),
        "off_diag_cos": round(off_diag_cos, 4),
        "relu_cutoff_pct_per_layer": layer_cutoffs,
        "dims": model.dims,
        "param_count": model.param_count,
    }


def _run_gradual_refinement_loop(x_train, y_train, x_test, y_test, lr: float, banner: str):
    print(f"\n=== {banner} ===")
    d_in = x_train.shape[1]
    d_out = 10
    val_x = x_train[:4000]
    tr_x, tr_y = x_train[4000:], y_train[4000:]

    depth_results = {}
    best_model = None
    last_discovered_bottom_width = d_in
    upstream_init_chain = []
    optimal_depth = 0

    for depth in [0, 1, 2, 3]:
        t0 = time.time()
        if depth == 0:
            init_dims = [d_in, d_out]
        else:
            new_cut_init = midpoint_width(last_discovered_bottom_width, d_out)
            upstream_init_chain.append(new_cut_init)
            init_dims = [d_in] + list(upstream_init_chain) + [d_out]

        model = ElasticFunnel(init_dims, lr=lr, seed=42)
        init_params = model.param_count
        rng = np.random.default_rng(42)
        bsz = 256
        epochs = 12
        width_trajectory = [model.dims]
        ortho_w = [0.15] * (depth - 1) + [1.50] if depth > 0 else []

        for ep in range(epochs):
            perm = rng.permutation(len(tr_x))
            for i in range(0, len(tr_x), bsz):
                idx = perm[i:i + bsz]
                xb, yb = tr_x[idx], tr_y[idx]
                _, u_out, cache = model.forward(xb)
                probs = softmax(u_out)
                one_hot = np.eye(d_out, dtype=np.float32)[yb]
                du_out = (probs - one_hot) / len(xb)
                model.backward_and_step(cache, du_out, yb=yb if depth > 0 else None, ortho_weights=ortho_w)

            if depth > 0 and ep in (4, 7, 9):
                for l_idx in range(depth):
                    model.breathe_layer(
                        l_idx,
                        val_x,
                        d_out=d_out,
                        min_width=1,
                        cos_thresh=0.90,
                        min_rel_energy=0.05,
                        allow_expand=(ep < 9),
                    )
                width_trajectory.append(model.dims)

        elapsed = time.time() - t0
        metrics = evaluate_funnel_metrics(model, x_test, y_test, tr_x[:5000], tr_y[:5000], n_classes=d_out)
        if depth > 0:
            last_discovered_bottom_width = model.hidden_dims[-1]
        next_midpoint = midpoint_width(last_discovered_bottom_width, d_out)

        param_saving_pct = round((1.0 - metrics["param_count"] / max(1, init_params)) * 100.0, 1)
        print(
            f"  [Step L={depth}] Init: {str(init_dims):26s} -> Discovered: {str(model.dims):24s} "
            f"({param_saving_pct:+5.1f}% params) | Head: {metrics['head_acc_pct']:6.2f}% | "
            f"Proto: {metrics['proto_acc_pct']:6.2f}% | IntLoss: {metrics['mean_internal_loss']:.4f} | "
            f"Off-Diag Cos: {metrics['off_diag_cos']:.4f} | Next Midpoint: {next_midpoint}"
        )
        depth_results[f"depth_{depth}"] = {
            "initial_dims": init_dims,
            "final_discovered_dims": model.dims,
            "width_trajectory": width_trajectory,
            "initial_params": init_params,
            "final_params": metrics["param_count"],
            "head_accuracy_pct": metrics["head_acc_pct"],
            "prototype_accuracy_pct": metrics["proto_acc_pct"],
            "ce_loss": metrics["ce_loss"],
            "mean_internal_loss": metrics["mean_internal_loss"],
            "off_diag_cos": metrics["off_diag_cos"],
            "relu_cutoff_pct_per_layer": metrics["relu_cutoff_pct_per_layer"],
            "next_midpoint_to_10": next_midpoint,
            "seconds": round(elapsed, 2),
        }
        best_model = copy.deepcopy(model)
        optimal_depth = depth
        if next_midpoint <= d_out:
            print(
                f"  => Midpoint Fixed Point Reached at L = {depth}: discovered d*_{depth} = {last_discovered_bottom_width}, "
                f"so round(sqrt({last_discovered_bottom_width} / 10) * 10) = {next_midpoint} == 10 (No further layer gap exists!)."
            )
            break

    return depth_results, optimal_depth, best_model


def run_gradual_refinement_from_2352_to_10(x_train, y_train, x_test, y_test):
    return _run_gradual_refinement_loop(
        x_train,
        y_train,
        x_test,
        y_test,
        lr=1.6e-3,
        banner="Stage 1A: Gradual Refinement Starting from 2352 -> 10 (Coupled Tuple Input, Zero Floor)",
    )


def run_gradual_refinement_from_784_to_10(i_train, y_train, i_test, y_test):
    return _run_gradual_refinement_loop(
        i_train,
        y_train,
        i_test,
        y_test,
        lr=1.8e-3,
        banner="Stage 1B: Gradual Refinement Starting from 784 -> 10 (Pure Pixel-Order Input, Zero Floor)",
    )


def main():
    print("Loading MNIST and building both 2,352-tuple and 784 pixel-order representations...")
    train_img, train_lbl, test_img, test_lbl = load_mnist()
    x_train = build_8bit_tuple_representation(train_img, couple_background=True)
    x_test = build_8bit_tuple_representation(test_img, couple_background=True)
    i_train = (train_img.astype(np.float32) / 255.0).reshape(len(train_img), 784)
    i_test = (test_img.astype(np.float32) / 255.0).reshape(len(test_img), 784)

    depth_results, opt_depth, _ = run_gradual_refinement_from_2352_to_10(x_train, train_lbl, x_test, test_lbl)
    depth_results_784, opt_depth_784, _ = run_gradual_refinement_from_784_to_10(i_train, train_lbl, i_test, test_lbl)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "elastic_funnel_results.json"
    summary = {
        "refinement_rule": "Start at d_in -> 10; split each remaining gap d*_{l-1} -> 10 at D_init(l) = round(sqrt(d*_{l-1} / 10) * 10) until round(sqrt(d*_L / 10) * 10) == 10",
        "optimal_discovered_depth": opt_depth,
        "gradual_depth_discovery": depth_results,
        "intensity_only_784_optimal_depth": opt_depth_784,
        "intensity_only_784_gradual_depth_discovery": depth_results_784,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved Stage 1 Gradual Refinement results to {out_path}")


if __name__ == "__main__":
    main()
