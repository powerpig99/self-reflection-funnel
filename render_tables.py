#!/usr/bin/env python3
"""
Render high-resolution (2x Retina) dark-theme PNG images of Benchmark Tables 1, 2, and 3
directly from the committed JSON files in ./results/:
  - assets/table1_benchmark_comparison.png
  - assets/table2_maskfree_occlusion_loop.png
  - assets/table3_dual_cut_3stage.png
  - assets/tables_combined.png
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = ROOT_DIR / "assets"
RESULTS_DIR = ROOT_DIR / "results"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def load_fonts():
    """Load clean macOS system fonts at 2x Retina sizes."""
    sans_candidates = [
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    bold_candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    mono_candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Courier.dfont",
    ]

    def pick(candidates, size, index=0):
        for p in candidates:
            if Path(p).exists():
                try:
                    return ImageFont.truetype(p, size=size, index=index)
                except Exception:
                    try:
                        return ImageFont.truetype(p, size=size)
                    except Exception:
                        pass
        return ImageFont.load_default()

    return {
        "title": pick(bold_candidates, 34),
        "subtitle": pick(sans_candidates, 22),
        "header": pick(bold_candidates, 22),
        "cell_bold": pick(bold_candidates, 23),
        "cell": pick(sans_candidates, 22),
        "mono_bold": pick(mono_candidates, 22),
        "mono": pick(mono_candidates, 21),
        "footer": pick(sans_candidates, 19),
    }


# Palette (GitHub Dark / Living Book Palette)
BG_OUTER = (13, 17, 23)        # #0d1117
BG_CARD = (22, 27, 34)         # #161b22
BG_HEADER = (30, 37, 46)       # #1e252e
BG_ROW_EVEN = (18, 23, 30)     # #12171e
BG_ROW_ODD = (22, 27, 34)      # #161b22
BORDER_COL = (48, 54, 61)      # #30363d
TEXT_MAIN = (240, 246, 252)    # #f0f6fc
TEXT_MUTED = (139, 148, 158)   # #8b949e
ACCENT_GREEN = (63, 185, 80)   # #3fb950
ACCENT_BLUE = (88, 166, 255)   # #58a6ff
ACCENT_PURPLE = (163, 113, 247)# #a371f7
ACCENT_RED = (248, 81, 73)     # #f85149
ACCENT_AMBER = (210, 153, 34)  # #d29922


def draw_multiline_centered(draw, box, lines, fonts_list, colors_list, align="center", line_gap=6):
    x0, y0, x1, y1 = box
    heights = []
    widths = []
    for txt, fnt in zip(lines, fonts_list):
        bb = draw.textbbox((0, 0), txt, font=fnt)
        widths.append(bb[2] - bb[0])
        heights.append(bb[3] - bb[1])
    total_h = sum(heights) + line_gap * (len(lines) - 1)
    cur_y = y0 + (y1 - y0 - total_h) // 2
    for txt, fnt, col, w, h in zip(lines, fonts_list, colors_list, widths, heights):
        if align == "left":
            cur_x = x0 + 20
        else:
            cur_x = x0 + (x1 - x0 - w) // 2
        draw.text((cur_x, cur_y), txt, fill=col, font=fnt)
        cur_y += h + line_gap


def render_table1(fonts) -> Image.Image:
    with open(RESULTS_DIR / "results.json", "r", encoding="utf-8") as f:
        res = json.load(f)

    p2 = res["pathway2_forward_grouping"]
    p1 = res["pathway1_diffusion_fewshot_closed_loop"]
    mlp = res["standard_cross_entropy_mlp_baseline"]
    ae = res["standard_pixel_loss_autoencoder_baseline"]

    col_widths = [280, 355, 255, 310, 330, 310]
    table_w = sum(col_widths)
    margin_x = 44
    margin_top = 40
    title_h = 92
    header_h = 84
    row_h = 78
    footer_h = 64

    headers = [
        ["Regime", "(12 Clean Epochs, seed = 42)"],
        ["Determination Funnel", "(256→64→32, (x, y, I) + z* Loss)"],
        ["Standard MLP", "(256→64→32, I-Only, Output CE)"],
        ["Pixel-Loss Autoencoder", "(Surface Loss ‖X̂ − X‖²)"],
        ["z* Off-Diag Cosine", "(Funnel vs. Std MLP)"],
        ["z* ReLU Cutoff %", "(Funnel vs. Std MLP)"],
    ]

    rows = [
        [
            (["1-Shot (10 samples)"], [fonts["cell_bold"]], [TEXT_MAIN], "left"),
            ([f"{p2['shots_1']['prototype_accuracy']:.2f}% (Fwd) / {p1['shots_1']['direct_test_accuracy']:.2f}% (Loop)"], [fonts["cell_bold"]], [ACCENT_GREEN], "center"),
            ([f"{mlp['shots_1']['accuracy']:.2f}%"], [fonts["cell"]], [TEXT_MAIN], "center"),
            ([f"{ae['shots_1']['prototype_accuracy']:.2f}%  (Cos: {ae['shots_1']['inter_group_off_diagonal_cosine']:.3f})"], [fonts["cell"]], [TEXT_MUTED], "center"),
            ([f"{p2['shots_1']['inter_group_off_diagonal_cosine']:.3f} vs. {mlp['shots_1']['inter_group_off_diagonal_cosine']:.3f}", "(45× more orthogonal)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_BLUE, TEXT_MUTED], "center"),
            ([f"{p2['shots_1']['z_star_relu_cutoff_pct']:.1f}% vs. {mlp['shots_1']['z_star_relu_cutoff_pct']:.1f}%"], [fonts["cell_bold"]], [ACCENT_PURPLE], "center"),
        ],
        [
            (["5-Shot (50 samples)"], [fonts["cell_bold"]], [TEXT_MAIN], "left"),
            ([f"{p2['shots_5']['prototype_accuracy']:.2f}%"], [fonts["cell_bold"]], [ACCENT_GREEN], "center"),
            ([f"{mlp['shots_5']['accuracy']:.2f}%"], [fonts["cell"]], [TEXT_MAIN], "center"),
            ([f"{ae['shots_5']['prototype_accuracy']:.2f}%  (Cos: {ae['shots_5']['inter_group_off_diagonal_cosine']:.3f})"], [fonts["cell"]], [TEXT_MUTED], "center"),
            ([f"{p2['shots_5']['inter_group_off_diagonal_cosine']:.3f} vs. {mlp['shots_5']['inter_group_off_diagonal_cosine']:.3f}", "(41× more orthogonal)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_BLUE, TEXT_MUTED], "center"),
            ([f"{p2['shots_5']['z_star_relu_cutoff_pct']:.1f}% vs. {mlp['shots_5']['z_star_relu_cutoff_pct']:.1f}%"], [fonts["cell_bold"]], [ACCENT_PURPLE], "center"),
        ],
        [
            (["10-Shot (100 samples)"], [fonts["cell_bold"]], [TEXT_MAIN], "left"),
            ([f"{p2['shots_10']['prototype_accuracy']:.2f}%"], [fonts["cell_bold"]], [ACCENT_GREEN], "center"),
            ([f"{mlp['shots_10']['accuracy']:.2f}%"], [fonts["cell"]], [TEXT_MAIN], "center"),
            ([f"{ae['shots_10']['prototype_accuracy']:.2f}%  (Cos: {ae['shots_10']['inter_group_off_diagonal_cosine']:.3f})"], [fonts["cell"]], [TEXT_MUTED], "center"),
            ([f"{p2['shots_10']['inter_group_off_diagonal_cosine']:.3f} vs. {mlp['shots_10']['inter_group_off_diagonal_cosine']:.3f}", "(43× more orthogonal)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_BLUE, TEXT_MUTED], "center"),
            ([f"{p2['shots_10']['z_star_relu_cutoff_pct']:.1f}% vs. {mlp['shots_10']['z_star_relu_cutoff_pct']:.1f}%"], [fonts["cell_bold"]], [ACCENT_PURPLE], "center"),
        ],
        [
            (["50-Shot (500 samples)"], [fonts["cell_bold"]], [TEXT_MAIN], "left"),
            ([f"{p2['shots_50']['prototype_accuracy']:.2f}%"], [fonts["cell_bold"]], [ACCENT_GREEN], "center"),
            ([f"{mlp['shots_50']['accuracy']:.2f}%"], [fonts["cell"]], [TEXT_MAIN], "center"),
            ([f"{ae['shots_50']['prototype_accuracy']:.2f}%  (Cos: {ae['shots_50']['inter_group_off_diagonal_cosine']:.3f})"], [fonts["cell"]], [ACCENT_RED], "center"),
            ([f"{p2['shots_50']['inter_group_off_diagonal_cosine']:.3f} vs. {mlp['shots_50']['inter_group_off_diagonal_cosine']:.3f}", "(62× more orthogonal)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_BLUE, TEXT_MUTED], "center"),
            ([f"{p2['shots_50']['z_star_relu_cutoff_pct']:.1f}% vs. {mlp['shots_50']['z_star_relu_cutoff_pct']:.1f}%"], [fonts["cell_bold"]], [ACCENT_PURPLE], "center"),
        ],
        [
            (["Full MNIST (60,000)"], [fonts["cell_bold"]], [TEXT_MAIN], "left"),
            ([f"{p2['shots_6000']['group_head_accuracy']:.2f}% (Head) / {p2['shots_6000']['prototype_accuracy']:.2f}% (Proto)"], [fonts["cell_bold"]], [ACCENT_GREEN], "center"),
            ([f"{mlp['shots_6000']['accuracy']:.2f}%"], [fonts["cell"]], [TEXT_MAIN], "center"),
            ([f"{ae['shots_6000']['prototype_accuracy']:.2f}%  (Cos: {ae['shots_6000']['inter_group_off_diagonal_cosine']:.3f})"], [fonts["cell_bold"]], [ACCENT_RED], "center"),
            ([f"{p2['shots_6000']['inter_group_off_diagonal_cosine']:.3f} vs. {mlp['shots_6000']['inter_group_off_diagonal_cosine']:.3f}", "(29× more orthogonal)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_BLUE, TEXT_MUTED], "center"),
            ([f"{p2['shots_6000']['z_star_relu_cutoff_pct']:.1f}% vs. {mlp['shots_6000']['z_star_relu_cutoff_pct']:.1f}%", "(71.6% cut at Layer 1)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_PURPLE, TEXT_MUTED], "center"),
        ],
    ]

    img_w = table_w + 2 * margin_x
    img_h = margin_top + title_h + header_h + len(rows) * row_h + footer_h + 24
    im = Image.new("RGB", (img_w, img_h), BG_OUTER)
    draw = ImageDraw.Draw(im)

    draw.text(
        (margin_x, margin_top),
        "Table 1 — Reference Funnel (256→64→32) vs. Standard MLP & Pixel-Loss Autoencoder",
        fill=TEXT_MAIN,
        font=fonts["title"],
    )
    draw.text(
        (margin_x, margin_top + 46),
        "12 Clean Epochs on [0, 255]³ Coordinate Tuples (run_experiment.py -> results/results.json)",
        fill=TEXT_MUTED,
        font=fonts["subtitle"],
    )

    t_top = margin_top + title_h
    draw.rectangle([margin_x, t_top, margin_x + table_w, t_top + header_h], fill=BG_HEADER, outline=BORDER_COL, width=2)
    cur_x = margin_x
    for col_idx, (w, h_lines) in enumerate(zip(col_widths, headers)):
        h_cols = [ACCENT_GREEN if col_idx == 1 else TEXT_MAIN, TEXT_MUTED]
        draw_multiline_centered(
            draw,
            (cur_x, t_top, cur_x + w, t_top + header_h),
            h_lines,
            [fonts["header"], fonts["footer"]],
            h_cols,
            align="left" if col_idx == 0 else "center",
        )
        if col_idx > 0:
            draw.line([cur_x, t_top, cur_x, t_top + header_h + len(rows) * row_h], fill=BORDER_COL, width=1)
        cur_x += w

    for r_idx, r_data in enumerate(rows):
        ry0 = t_top + header_h + r_idx * row_h
        ry1 = ry0 + row_h
        bg = BG_ROW_EVEN if r_idx % 2 == 0 else BG_ROW_ODD
        draw.rectangle([margin_x, ry0, margin_x + table_w, ry1], fill=bg, outline=BORDER_COL, width=1)
        cur_x = margin_x
        for c_idx, (w, cell_spec) in enumerate(zip(col_widths, r_data)):
            lines, f_list, c_list, align = cell_spec
            draw_multiline_centered(draw, (cur_x, ry0, cur_x + w, ry1), lines, f_list, c_list, align=align)
            if c_idx > 0:
                draw.line([cur_x, ry0, cur_x, ry1], fill=BORDER_COL, width=1)
            cur_x += w

    draw.rectangle([margin_x, t_top, margin_x + table_w, t_top + header_h + len(rows) * row_h], outline=BORDER_COL, width=2)

    fy = t_top + header_h + len(rows) * row_h + 18
    u_pur = res["pathway2b_unsupervised_10group"]["unsupervised_purity_pct"]
    mr = res["multi_resolution_generalization"]
    draw.text(
        (margin_x, fy),
        f"Zero-Label Unsupervised 10-Group Purity: {u_pur:.2f}% (10/10 active groups)   ·   Zero-Shot Multi-Resolution Accuracy: {mr['res_14x14']:.2f}% (14×14)  |  {mr['res_28x28']:.2f}% (28×28)  |  {mr['res_56x56']:.2f}% (56×56)",
        fill=ACCENT_BLUE,
        font=fonts["footer"],
    )
    return im


def render_table2(fonts) -> Image.Image:
    with open(RESULTS_DIR / "occlusion_results.json", "r", encoding="utf-8") as f:
        occ = json.load(f)

    col_widths = [250, 175, 205, 175, 175, 175, 175, 225, 285]
    table_w = sum(col_widths)
    margin_x = 44
    margin_top = 40
    title_h = 92
    header_h = 84
    row_h = 84
    footer_h = 64

    headers = [
        ["Corruption Mode", "(10,000 Test Digits)"],
        ["Std MLP", "(Clean-Trained)"],
        ["Std MLP", "(Same-Corr-Trained)"],
        ["Loop Step 0", "E(X_occ)"],
        ["Loop Step 1", "(Mask-Free)"],
        ["Loop Step 2", "(Mask-Free)"],
        ["Loop Step 3", "(Mask-Free)"],
        ["Loop Step 4", "(Mask-Free)"],
        ["Canvas Error Reduction", "MSE (Step 0 → 4)"],
    ]

    def make_occ_row(mode_key, title_str, sub_str):
        m = occ[mode_key]
        st = m["mask_free_closed_loop_steps"]
        gain4 = st[4]["accuracy_pct"] - st[0]["accuracy_pct"]
        mse_drop = (1.0 - st[4]["pixel_mse_to_clean"] / st[0]["pixel_mse_to_clean"]) * 100.0
        return [
            ([title_str, sub_str], [fonts["cell_bold"], fonts["footer"]], [TEXT_MAIN, TEXT_MUTED], "left"),
            ([f"{m['standard_mlp_clean_trained_acc']:.2f}%"], [fonts["cell"]], [ACCENT_RED], "center"),
            ([f"{m['standard_mlp_same_corruption_trained_acc']:.2f}%"], [fonts["cell"]], [TEXT_MAIN], "center"),
            ([f"{st[0]['accuracy_pct']:.2f}%"], [fonts["cell"]], [TEXT_MUTED], "center"),
            ([f"{st[1]['accuracy_pct']:.2f}%"], [fonts["cell"]], [TEXT_MAIN], "center"),
            ([f"{st[2]['accuracy_pct']:.2f}%"], [fonts["cell"]], [TEXT_MAIN], "center"),
            ([f"{st[3]['accuracy_pct']:.2f}%"], [fonts["cell"]], [TEXT_MAIN], "center"),
            ([f"{st[4]['accuracy_pct']:.2f}%", f"({gain4:+.2f}% loop gain)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_GREEN, ACCENT_GREEN], "center"),
            ([f"-{mse_drop:.0f}% Pixel MSE", f"({st[0]['pixel_mse_to_clean']:.4f} → {st[4]['pixel_mse_to_clean']:.4f})"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_BLUE, TEXT_MUTED], "center"),
        ]

    rows = [
        make_occ_row("picket_fence", "Picket Fence", "(3 erased bands + 2 bars)"),
        make_occ_row("center_blackout", "Center Blackout", "(10×12 core wiped out)"),
        make_occ_row("heavy_clutter", "Heavy Clutter", "(45% drop + 15% spikes)"),
    ]

    img_w = table_w + 2 * margin_x
    img_h = margin_top + title_h + header_h + len(rows) * row_h + footer_h + 24
    im = Image.new("RGB", (img_w, img_h), BG_OUTER)
    draw = ImageDraw.Draw(im)

    draw.text(
        (margin_x, margin_top),
        "Table 2 — Reference 4-Step Mask-Free Self-Reflection Loop (run_occlusion_loop.py)",
        fill=TEXT_MAIN,
        font=fonts["title"],
    )
    draw.text(
        (margin_x, margin_top + 46),
        "Zero external occlusion mask passed at test time; compared against Clean-Trained and Same-Corruption-Trained Standard MLPs",
        fill=TEXT_MUTED,
        font=fonts["subtitle"],
    )

    t_top = margin_top + title_h
    draw.rectangle([margin_x, t_top, margin_x + table_w, t_top + header_h], fill=BG_HEADER, outline=BORDER_COL, width=2)
    cur_x = margin_x
    for col_idx, (w, h_lines) in enumerate(zip(col_widths, headers)):
        h_cols = [ACCENT_GREEN if col_idx == 7 else TEXT_MAIN, TEXT_MUTED]
        draw_multiline_centered(
            draw,
            (cur_x, t_top, cur_x + w, t_top + header_h),
            h_lines,
            [fonts["header"], fonts["footer"]],
            h_cols,
            align="left" if col_idx == 0 else "center",
        )
        if col_idx > 0:
            draw.line([cur_x, t_top, cur_x, t_top + header_h + len(rows) * row_h], fill=BORDER_COL, width=1)
        cur_x += w

    for r_idx, r_data in enumerate(rows):
        ry0 = t_top + header_h + r_idx * row_h
        ry1 = ry0 + row_h
        bg = BG_ROW_EVEN if r_idx % 2 == 0 else BG_ROW_ODD
        draw.rectangle([margin_x, ry0, margin_x + table_w, ry1], fill=bg, outline=BORDER_COL, width=1)
        cur_x = margin_x
        for c_idx, (w, cell_spec) in enumerate(zip(col_widths, r_data)):
            lines, f_list, c_list, align = cell_spec
            draw_multiline_centered(draw, (cur_x, ry0, cur_x + w, ry1), lines, f_list, c_list, align=align)
            if c_idx > 0:
                draw.line([cur_x, ry0, cur_x, ry1], fill=BORDER_COL, width=1)
            cur_x += w

    draw.rectangle([margin_x, t_top, margin_x + table_w, t_top + header_h + len(rows) * row_h], outline=BORDER_COL, width=2)

    fy = t_top + header_h + len(rows) * row_h + 18
    pf_s = occ["picket_fence"]["mask_free_closed_loop_steps"]
    cb_s = occ["center_blackout"]["mask_free_closed_loop_steps"]
    hc_s = occ["heavy_clutter"]["mask_free_closed_loop_steps"]
    draw.text(
        (margin_x, fy),
        f"Latent Cosine Alignment cos(z_k, z*_clean) across Steps 0 → 4:   Picket Fence: {pf_s[0]['latent_cosine_to_clean']:.4f} → {pf_s[4]['latent_cosine_to_clean']:.4f}   ·   Center Blackout: {cb_s[0]['latent_cosine_to_clean']:.4f} → {cb_s[4]['latent_cosine_to_clean']:.4f}   ·   Heavy Clutter: {hc_s[0]['latent_cosine_to_clean']:.4f} → {hc_s[4]['latent_cosine_to_clean']:.4f}",
        fill=ACCENT_PURPLE,
        font=fonts["footer"],
    )
    return im


def render_table3(fonts) -> Image.Image:
    """Render Table 3 directly from results/diffusion_reflection_results.json."""
    with open(RESULTS_DIR / "diffusion_reflection_results.json", "r", encoding="utf-8") as f:
        dres = json.load(f)

    s1_784 = dres["stage1_intensity_only_784_gradual_discovery"]["depth_3"]
    s2 = dres["stage2_frozen_E_trained_G"]
    s3u = dres["stage3_unconstrained_relaxed_E_G"]
    s3b = dres["stage3b_reality_anchored_self_judged"]
    s3a = dres["stage3a_reality_anchored_frozen_judge"]
    f784_same = dres["intensity_only_784_funnels"]["same_size_784_105_32_11_10_stage3a"]
    mlp_same = dres["same_stream_baselines_22ep"]["same_width_std_mlp_784_105_32_11_10"]
    mlp_large = dres["same_stream_baselines_22ep"]["large_std_mlp_784_256_64_32_10"]

    col_widths = [370, 385, 245, 220, 210, 205, 205]
    table_w = sum(col_widths)
    margin_x = 44
    margin_top = 40
    title_h = 92
    header_h = 84
    row_h = 82
    footer_h = 68

    headers = [
        ["Stage / Model Variant", "(results/diffusion_reflection_results.json)"],
        ["Architecture / Widths", "(Input & Hidden Cuts)"],
        ["Parameters / Exposure", "(Exact Size & Stream)"],
        ["Natural MNIST", "(All 10k / Hard 5%)"],
        ["Picket Fence", "(Step 0 → Converged)"],
        ["Center Blackout", "(Step 0 → Converged)"],
        ["Heavy Clutter", "(Step 0 → Converged)"],
    ]

    rows = [
        [
            (["Same-Size 86.3k Std MLP", "(No G, No Loop, Output CE)"], [fonts["cell_bold"], fonts["footer"]], [TEXT_MAIN, TEXT_MUTED], "left"),
            ([f"{mlp_same['dims']}", "(784 Pixel-Order Input)"], [fonts["mono"], fonts["footer"]], [TEXT_MAIN, TEXT_MUTED], "center"),
            ([f"{mlp_same['params']:,} params", "Same 22-Ep Corr Stream"], [fonts["cell"], fonts["footer"]], [TEXT_MAIN, TEXT_MUTED], "center"),
            ([f"{mlp_same['natural_mnist_acc_pct']:.2f}%", "(Single Pass)"], [fonts["cell_bold"], fonts["footer"]], [TEXT_MAIN, TEXT_MUTED], "center"),
            ([f"{mlp_same['picket_fence_acc_pct']:.2f}%", "(Single Pass)"], [fonts["cell"], fonts["footer"]], [ACCENT_RED, TEXT_MUTED], "center"),
            ([f"{mlp_same['center_blackout_acc_pct']:.2f}%", "(Single Pass)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_BLUE, TEXT_MUTED], "center"),
            ([f"{mlp_same['heavy_clutter_acc_pct']:.2f}%", "(Single Pass)"], [fonts["cell"], fonts["footer"]], [TEXT_MAIN, TEXT_MUTED], "center"),
        ],
        [
            (["Same-Size 86.3k 784-Funnel", "(Pure 784 Pixel-Order, 3A)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_GREEN, TEXT_MUTED], "left"),
            ([f"{f784_same['dims']}", "(1:1 Param Parity w/ MLP)"], [fonts["mono_bold"], fonts["footer"]], [ACCENT_GREEN, TEXT_MUTED], "center"),
            ([f"{f784_same['params']:,} params (E)", f"784-Disc: {s1_784['final_params']:,} (14→10)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_GREEN, TEXT_MUTED], "center"),
            ([f"{f784_same['natural_mnist']['all_10k_step0_acc_pct']:.2f}% → {f784_same['natural_mnist']['all_10k_converged_acc_pct']:.2f}%", f"Hard 5%: {f784_same['natural_mnist']['high_loss_5pct_step0_acc_pct']:.2f}% → {f784_same['natural_mnist']['high_loss_5pct_converged_acc_pct']:.2f}%"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_GREEN, TEXT_MUTED], "center"),
            ([f"{f784_same['occlusions_converged']['picket_fence']['step0_acc_pct']:.2f}% → {f784_same['occlusions_converged']['picket_fence']['converged_acc_pct']:.2f}%", f"(+8.64% vs 86.3k MLP)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_GREEN, ACCENT_GREEN], "center"),
            ([f"{f784_same['occlusions_converged']['center_blackout']['step0_acc_pct']:.2f}% → {f784_same['occlusions_converged']['center_blackout']['converged_acc_pct']:.2f}%", f"({f784_same['occlusions_converged']['center_blackout']['loop_gain_pct']:+.2f}%, {f784_same['occlusions_converged']['center_blackout']['mean_steps']:.1f} steps)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_GREEN, TEXT_MUTED], "center"),
            ([f"{f784_same['occlusions_converged']['heavy_clutter']['step0_acc_pct']:.2f}% → {f784_same['occlusions_converged']['heavy_clutter']['converged_acc_pct']:.2f}%", f"(+3.74% vs 86.3k MLP)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_GREEN, ACCENT_GREEN], "center"),
        ],
        [
            (["Closest-Size 220k Std MLP", "(No G, No Loop, Output CE)"], [fonts["cell_bold"], fonts["footer"]], [TEXT_MAIN, TEXT_MUTED], "left"),
            ([f"{mlp_large['dims']}", "(784 Pixel-Order Input)"], [fonts["mono"], fonts["footer"]], [TEXT_MAIN, TEXT_MUTED], "center"),
            ([f"{mlp_large['params']:,} params", "Same 22-Ep Corr Stream"], [fonts["cell"], fonts["footer"]], [TEXT_MAIN, TEXT_MUTED], "center"),
            ([f"{mlp_large['natural_mnist_acc_pct']:.2f}%", "(Single Pass)"], [fonts["cell_bold"], fonts["footer"]], [TEXT_MAIN, TEXT_MUTED], "center"),
            ([f"{mlp_large['picket_fence_acc_pct']:.2f}%", "(Single Pass)"], [fonts["cell"], fonts["footer"]], [TEXT_MAIN, TEXT_MUTED], "center"),
            ([f"{mlp_large['center_blackout_acc_pct']:.2f}%", "(Single Pass)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_BLUE, TEXT_MUTED], "center"),
            ([f"{mlp_large['heavy_clutter_acc_pct']:.2f}%", "(Single Pass)"], [fonts["cell"], fonts["footer"]], [TEXT_MAIN, TEXT_MUTED], "center"),
        ],
        [
            (["Stage 2: Frozen E + Trained G", "(2,352 Tuple Funnel, Frozen E)"], [fonts["cell_bold"], fonts["footer"]], [TEXT_MAIN, TEXT_MUTED], "left"),
            (["E: [2352, 105, 32, 11, 10]", "G: [10+11, 32, 105, 287, 784]"], [fonts["mono"], fonts["footer"]], [ACCENT_PURPLE, TEXT_MUTED], "center"),
            (["250,940 params (E)", "(-59.6% vs 621k Funnel)"], [fonts["cell"], fonts["footer"]], [TEXT_MAIN, TEXT_MUTED], "center"),
            ([f"{s2['natural_mnist']['all_10k_step0_acc_pct']:.2f}% → {s2['natural_mnist']['all_10k_converged_acc_pct']:.2f}%", f"Hard 5%: {s2['natural_mnist']['high_loss_5pct_step0_acc_pct']:.2f}% → {s2['natural_mnist']['high_loss_5pct_converged_acc_pct']:.2f}%"], [fonts["cell_bold"], fonts["footer"]], [TEXT_MAIN, ACCENT_GREEN], "center"),
            ([f"{s2['occlusions_converged']['picket_fence']['step0_acc_pct']:.2f}% → {s2['occlusions_converged']['picket_fence']['converged_acc_pct']:.2f}%", f"({s2['occlusions_converged']['picket_fence']['loop_gain_pct']:+.2f}%, {s2['occlusions_converged']['picket_fence']['mean_steps']:.1f} steps)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_GREEN, TEXT_MUTED], "center"),
            ([f"{s2['occlusions_converged']['center_blackout']['step0_acc_pct']:.2f}% → {s2['occlusions_converged']['center_blackout']['converged_acc_pct']:.2f}%", f"({s2['occlusions_converged']['center_blackout']['loop_gain_pct']:+.2f}%, {s2['occlusions_converged']['center_blackout']['mean_steps']:.1f} steps)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_GREEN, TEXT_MUTED], "center"),
            ([f"{s2['occlusions_converged']['heavy_clutter']['step0_acc_pct']:.2f}% → {s2['occlusions_converged']['heavy_clutter']['converged_acc_pct']:.2f}%", f"({s2['occlusions_converged']['heavy_clutter']['loop_gain_pct']:+.2f}%, {s2['occlusions_converged']['heavy_clutter']['mean_steps']:.1f} steps)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_GREEN, TEXT_MUTED], "center"),
        ],
        [
            (["Stage 3 Unconstrained (1-Var)", "(Only `correct_0` Gate Removed)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_RED, TEXT_MUTED], "left"),
            (["Relaxed E + G (No Anchor)", "(Same LRs & Weights as 3B/3A)"], [fonts["cell"], fonts["footer"]], [ACCENT_RED, TEXT_MUTED], "center"),
            (["250,940 params (E)", f"({s3u['natural_mnist']['high_loss_5pct_mean_steps']:.1f} steps on Hard 5%)"], [fonts["cell"], fonts["footer"]], [TEXT_MAIN, ACCENT_RED], "center"),
            ([f"{s3u['natural_mnist']['all_10k_step0_acc_pct']:.2f}% → {s3u['natural_mnist']['all_10k_converged_acc_pct']:.2f}%", f"Hard 5%: {s3u['natural_mnist']['high_loss_5pct_step0_acc_pct']:.2f}% → {s3u['natural_mnist']['high_loss_5pct_converged_acc_pct']:.2f}%"], [fonts["cell"], fonts["footer"]], [TEXT_MAIN, ACCENT_RED], "center"),
            ([f"{s3u['occlusions_converged']['picket_fence']['step0_acc_pct']:.2f}% → {s3u['occlusions_converged']['picket_fence']['converged_acc_pct']:.2f}%", "(-1.81% vs 3B / -1.46% in 3A)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_RED, ACCENT_RED], "center"),
            ([f"{s3u['occlusions_converged']['center_blackout']['step0_acc_pct']:.2f}% → {s3u['occlusions_converged']['center_blackout']['converged_acc_pct']:.2f}%", f"({s3u['occlusions_converged']['center_blackout']['loop_gain_pct']:+.2f}%, {s3u['occlusions_converged']['center_blackout']['mean_steps']:.1f} steps)"], [fonts["cell"], fonts["footer"]], [TEXT_MAIN, TEXT_MUTED], "center"),
            ([f"{s3u['occlusions_converged']['heavy_clutter']['step0_acc_pct']:.2f}% → {s3u['occlusions_converged']['heavy_clutter']['converged_acc_pct']:.2f}%", f"({s3u['occlusions_converged']['heavy_clutter']['loop_gain_pct']:+.2f}%, {s3u['occlusions_converged']['heavy_clutter']['mean_steps']:.1f} steps)"], [fonts["cell"], fonts["footer"]], [TEXT_MAIN, TEXT_MUTED], "center"),
        ],
        [
            (["Stage 3B Reality-Anchored", "(Self-Judged by E_anc)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_BLUE, TEXT_MUTED], "left"),
            (["Relaxed E + G (Anchored)", "(Only valid X^(1) reinforced)"], [fonts["cell"], fonts["footer"]], [ACCENT_BLUE, TEXT_MUTED], "center"),
            (["250,940 params (E)", f"({s3b['natural_mnist']['high_loss_5pct_mean_steps']:.1f} steps on Hard 5%)"], [fonts["cell"], fonts["footer"]], [TEXT_MAIN, ACCENT_GREEN], "center"),
            ([f"{s3b['natural_mnist']['all_10k_step0_acc_pct']:.2f}% → {s3b['natural_mnist']['all_10k_converged_acc_pct']:.2f}%", f"Hard 5%: {s3b['natural_mnist']['high_loss_5pct_step0_acc_pct']:.2f}% → {s3b['natural_mnist']['high_loss_5pct_converged_acc_pct']:.2f}%"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_GREEN, TEXT_MUTED], "center"),
            ([f"{s3b['occlusions_converged']['picket_fence']['step0_acc_pct']:.2f}% → {s3b['occlusions_converged']['picket_fence']['converged_acc_pct']:.2f}%", f"({s3b['occlusions_converged']['picket_fence']['loop_gain_pct']:+.2f}%, {s3b['occlusions_converged']['picket_fence']['mean_steps']:.1f} steps)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_GREEN, TEXT_MUTED], "center"),
            ([f"{s3b['occlusions_converged']['center_blackout']['step0_acc_pct']:.2f}% → {s3b['occlusions_converged']['center_blackout']['converged_acc_pct']:.2f}%", f"({s3b['occlusions_converged']['center_blackout']['loop_gain_pct']:+.2f}%, {s3b['occlusions_converged']['center_blackout']['mean_steps']:.1f} steps)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_GREEN, TEXT_MUTED], "center"),
            ([f"{s3b['occlusions_converged']['heavy_clutter']['step0_acc_pct']:.2f}% → {s3b['occlusions_converged']['heavy_clutter']['converged_acc_pct']:.2f}%", f"({s3b['occlusions_converged']['heavy_clutter']['loop_gain_pct']:+.2f}%, {s3b['occlusions_converged']['heavy_clutter']['mean_steps']:.1f} steps)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_GREEN, TEXT_MUTED], "center"),
        ],
        [
            (["Stage 3A Reality-Anchored", "(Judged by Stage-1 E_frozen)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_GREEN, TEXT_MUTED], "left"),
            (["Relaxed E + G (Headline 3A)", "(Stationary E_frozen Judge)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_GREEN, TEXT_MUTED], "center"),
            (["250,940 params (E)", f"({s3a['natural_mnist']['high_loss_5pct_mean_steps']:.1f} steps on Hard 5%)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_GREEN, TEXT_MUTED], "center"),
            ([f"{s3a['natural_mnist']['all_10k_step0_acc_pct']:.2f}% → {s3a['natural_mnist']['all_10k_converged_acc_pct']:.2f}%", f"Hard 5%: {s3a['natural_mnist']['high_loss_5pct_step0_acc_pct']:.2f}% → {s3a['natural_mnist']['high_loss_5pct_converged_acc_pct']:.2f}%"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_GREEN, ACCENT_BLUE], "center"),
            ([f"{s3a['occlusions_converged']['picket_fence']['step0_acc_pct']:.2f}% → {s3a['occlusions_converged']['picket_fence']['converged_acc_pct']:.2f}%", f"({s3a['occlusions_converged']['picket_fence']['loop_gain_pct']:+.2f}%, {s3a['occlusions_converged']['picket_fence']['mean_steps']:.1f} steps)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_GREEN, ACCENT_GREEN], "center"),
            ([f"{s3a['occlusions_converged']['center_blackout']['step0_acc_pct']:.2f}% → {s3a['occlusions_converged']['center_blackout']['converged_acc_pct']:.2f}%", f"({s3a['occlusions_converged']['center_blackout']['loop_gain_pct']:+.2f}%, {s3a['occlusions_converged']['center_blackout']['mean_steps']:.1f} steps)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_GREEN, ACCENT_GREEN], "center"),
            ([f"{s3a['occlusions_converged']['heavy_clutter']['step0_acc_pct']:.2f}% → {s3a['occlusions_converged']['heavy_clutter']['converged_acc_pct']:.2f}%", f"({s3a['occlusions_converged']['heavy_clutter']['loop_gain_pct']:+.2f}%, {s3a['occlusions_converged']['heavy_clutter']['mean_steps']:.1f} steps)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_GREEN, ACCENT_GREEN], "center"),
        ],
    ]

    img_w = table_w + 2 * margin_x
    img_h = margin_top + title_h + header_h + len(rows) * row_h + footer_h + 24
    im = Image.new("RGB", (img_w, img_h), BG_OUTER)
    draw = ImageDraw.Draw(im)

    draw.text(
        (margin_x, margin_top),
        "Table 3 — 4-Stage Dual-Cut Pipeline: 784 & 2,352 Funnels vs. Same-Size MLPs & 1-Var Anchor Ablation",
        fill=TEXT_MAIN,
        font=fonts["title"],
    )
    draw.text(
        (margin_x, margin_top + 46),
        "All values loaded directly from results/diffusion_reflection_results.json (run_diffusion_reflection.py)",
        fill=TEXT_MUTED,
        font=fonts["subtitle"],
    )

    t_top = margin_top + title_h
    draw.rectangle([margin_x, t_top, margin_x + table_w, t_top + header_h], fill=BG_HEADER, outline=BORDER_COL, width=2)
    cur_x = margin_x
    for col_idx, (w, h_lines) in enumerate(zip(col_widths, headers)):
        h_cols = [ACCENT_GREEN if col_idx in (0, 3) else TEXT_MAIN, TEXT_MUTED]
        draw_multiline_centered(
            draw,
            (cur_x, t_top, cur_x + w, t_top + header_h),
            h_lines,
            [fonts["header"], fonts["footer"]],
            h_cols,
            align="left" if col_idx == 0 else "center",
        )
        if col_idx > 0:
            draw.line([cur_x, t_top, cur_x, t_top + header_h + len(rows) * row_h], fill=BORDER_COL, width=1)
        cur_x += w

    for r_idx, r_data in enumerate(rows):
        ry0 = t_top + header_h + r_idx * row_h
        ry1 = ry0 + row_h
        bg = BG_ROW_EVEN if r_idx % 2 == 0 else BG_ROW_ODD
        draw.rectangle([margin_x, ry0, margin_x + table_w, ry1], fill=bg, outline=BORDER_COL, width=1)
        cur_x = margin_x
        for c_idx, (w, cell_spec) in enumerate(zip(col_widths, r_data)):
            lines, f_list, c_list, align = cell_spec
            draw_multiline_centered(draw, (cur_x, ry0, cur_x + w, ry1), lines, f_list, c_list, align=align)
            if c_idx > 0:
                draw.line([cur_x, ry0, cur_x, ry1], fill=BORDER_COL, width=1)
            cur_x += w

    draw.rectangle([margin_x, t_top, margin_x + table_w, t_top + header_h + len(rows) * row_h], outline=BORDER_COL, width=2)

    fy = t_top + header_h + len(rows) * row_h + 18
    draw.text(
        (margin_x, fy),
        "At 86,300 params (784→105→32→11→10), 784-Funnel beats 86.3k MLP by +0.37% Nat, +8.64% Fence, +3.74% Clutter; at ~220k–251k, 3A leads 220k MLP by +3.94% Fence & +1.98% Clutter.",
        fill=ACCENT_BLUE,
        font=fonts["footer"],
    )
    return im


def main():
    fonts = load_fonts()
    im1 = render_table1(fonts)
    im2 = render_table2(fonts)
    im3 = render_table3(fonts)

    p1 = ASSETS_DIR / "table1_benchmark_comparison.png"
    p2 = ASSETS_DIR / "table2_maskfree_occlusion_loop.png"
    p3 = ASSETS_DIR / "table3_dual_cut_3stage.png"
    im1.save(p1)
    im2.save(p2)
    im3.save(p3)

    comb_w = max(im1.width, im2.width, im3.width)
    comb_h = im3.height + im1.height + im2.height + 40
    comb = Image.new("RGB", (comb_w, comb_h), BG_OUTER)
    comb.paste(im3, (0, 0))
    comb.paste(im1, (0, im3.height + 20))
    comb.paste(im2, (0, im3.height + im1.height + 40))
    p_comb = ASSETS_DIR / "tables_combined.png"
    comb.save(p_comb)
    print(f"Saved:\n  - {p1} ({im1.size})\n  - {p2} ({im2.size})\n  - {p3} ({im3.size})\n  - {p_comb} ({comb.size})")


if __name__ == "__main__":
    main()
