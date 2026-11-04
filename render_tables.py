#!/usr/bin/env python3
"""
Render high-resolution (2x Retina) dark-theme PNG images of Benchmark Table 1 and Table 2
for embedding in X Articles (which do not support native Markdown tables).
Outputs:
  - assets/table1_benchmark_comparison.png
  - assets/table2_maskfree_occlusion_loop.png
  - assets/tables_combined.png
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
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
    col_widths = [280, 340, 270, 310, 330, 310]
    table_w = sum(col_widths)
    margin_x = 44
    margin_top = 40
    title_h = 92
    header_h = 84
    row_h = 78
    footer_h = 64

    headers = [
        ["Regime", "(seed = 42)"],
        ["Determination Funnel", "(x, y, I) + Latent z* Loss"],
        ["Standard MLP", "(I-Only, Output CE)"],
        ["Pixel-Loss Autoencoder", "(Surface Loss ‖X̂ − X‖²)"],
        ["z* Off-Diag Cosine", "(Funnel vs. Std MLP)"],
        ["z* ReLU Cutoff %", "(Funnel vs. Std MLP)"],
    ]

    rows = [
        [
            (["1-Shot (10 samples)"], [fonts["cell_bold"]], [TEXT_MAIN], "left"),
            (["43.39% (Fwd) / 42.08% (Loop)"], [fonts["cell_bold"]], [ACCENT_GREEN], "center"),
            (["39.19%"], [fonts["cell"]], [TEXT_MAIN], "center"),
            (["30.05%  (Cos: 0.367)"], [fonts["cell"]], [TEXT_MUTED], "center"),
            (["0.014 vs. 0.634", "(45× more orthogonal)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_BLUE, TEXT_MUTED], "center"),
            (["78.8% vs. 44.0%"], [fonts["cell_bold"]], [ACCENT_PURPLE], "center"),
        ],
        [
            (["5-Shot (50 samples)"], [fonts["cell_bold"]], [TEXT_MAIN], "left"),
            (["68.20%"], [fonts["cell_bold"]], [ACCENT_GREEN], "center"),
            (["65.23%"], [fonts["cell"]], [TEXT_MAIN], "center"),
            (["44.54%  (Cos: 0.870)"], [fonts["cell"]], [TEXT_MUTED], "center"),
            (["0.017 vs. 0.694", "(41× more orthogonal)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_BLUE, TEXT_MUTED], "center"),
            (["80.8% vs. 23.5%"], [fonts["cell_bold"]], [ACCENT_PURPLE], "center"),
        ],
        [
            (["10-Shot (100 samples)"], [fonts["cell_bold"]], [TEXT_MAIN], "left"),
            (["74.92%"], [fonts["cell_bold"]], [ACCENT_GREEN], "center"),
            (["71.70%"], [fonts["cell"]], [TEXT_MAIN], "center"),
            (["60.21%  (Cos: 0.757)"], [fonts["cell"]], [TEXT_MUTED], "center"),
            (["0.017 vs. 0.736", "(43× more orthogonal)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_BLUE, TEXT_MUTED], "center"),
            (["76.2% vs. 27.7%"], [fonts["cell_bold"]], [ACCENT_PURPLE], "center"),
        ],
        [
            (["50-Shot (500 samples)"], [fonts["cell_bold"]], [TEXT_MAIN], "left"),
            (["88.52%"], [fonts["cell_bold"]], [ACCENT_GREEN], "center"),
            (["88.84%"], [fonts["cell"]], [TEXT_MAIN], "center"),
            (["74.24%  (Cos: 0.894)"], [fonts["cell"]], [ACCENT_RED], "center"),
            (["0.011 vs. 0.683", "(62× more orthogonal)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_BLUE, TEXT_MUTED], "center"),
            (["81.9% vs. 36.6%"], [fonts["cell_bold"]], [ACCENT_PURPLE], "center"),
        ],
        [
            (["Full MNIST (60,000)"], [fonts["cell_bold"]], [TEXT_MAIN], "left"),
            (["97.93%  (4.79s CPU)"], [fonts["cell_bold"]], [ACCENT_GREEN], "center"),
            (["97.71%"], [fonts["cell"]], [TEXT_MAIN], "center"),
            (["74.68%  (Cos: 0.935)"], [fonts["cell_bold"]], [ACCENT_RED], "center"),
            (["0.019 vs. 0.551", "(29× more orthogonal)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_BLUE, TEXT_MUTED], "center"),
            (["83.9% vs. 40.6%", "(71.6% cut at Layer 1)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_PURPLE, TEXT_MUTED], "center"),
        ],
    ]

    img_w = table_w + 2 * margin_x
    img_h = margin_top + title_h + header_h + len(rows) * row_h + footer_h + 24
    im = Image.new("RGB", (img_w, img_h), BG_OUTER)
    draw = ImageDraw.Draw(im)

    # Title & Subtitle
    draw.text(
        (margin_x, margin_top),
        "Table 1 — Determination Funnel vs. Standard MLP & Pixel-Loss Autoencoder",
        fill=TEXT_MAIN,
        font=fonts["title"],
    )
    draw.text(
        (margin_x, margin_top + 46),
        "4-Layer Linear + ReLU Funnel (2,352 → 256 → 64 → 32 → 10) on [0, 255]³ Coordinate Tuples (Zero Convolutions, Zero Attention)",
        fill=TEXT_MUTED,
        font=fonts["subtitle"],
    )

    t_top = margin_top + title_h
    # Header row background
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

    # Data rows
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

    # Outer table border
    draw.rectangle([margin_x, t_top, margin_x + table_w, t_top + header_h + len(rows) * row_h], outline=BORDER_COL, width=2)

    # Footer note
    fy = t_top + header_h + len(rows) * row_h + 18
    draw.text(
        (margin_x, fy),
        "Zero-Label Unsupervised 10-Group Purity: 53.91% (10/10 active groups)   ·   Zero-Shot Multi-Resolution Accuracy: 96.26% (14×14)  |  97.77% (28×28)  |  97.75% (56×56)",
        fill=ACCENT_BLUE,
        font=fonts["footer"],
    )
    return im


def render_table2(fonts) -> Image.Image:
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

    rows = [
        [
            (["Picket Fence", "(3 erased bands + 2 bars)"], [fonts["cell_bold"], fonts["footer"]], [TEXT_MAIN, TEXT_MUTED], "left"),
            (["42.95%"], [fonts["cell"]], [ACCENT_RED], "center"),
            (["80.44%"], [fonts["cell"]], [TEXT_MAIN], "center"),
            (["79.45%"], [fonts["cell"]], [TEXT_MUTED], "center"),
            (["81.05%"], [fonts["cell"]], [TEXT_MAIN], "center"),
            (["81.76%"], [fonts["cell"]], [TEXT_MAIN], "center"),
            (["82.04%"], [fonts["cell"]], [TEXT_MAIN], "center"),
            (["82.09%", "(+2.64% loop gain)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_GREEN, ACCENT_GREEN], "center"),
            (["-58% Pixel MSE", "(0.0774 → 0.0324)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_BLUE, TEXT_MUTED], "center"),
        ],
        [
            (["Center Blackout", "(10×12 core wiped out)"], [fonts["cell_bold"], fonts["footer"]], [TEXT_MAIN, TEXT_MUTED], "left"),
            (["40.15%"], [fonts["cell"]], [ACCENT_RED], "center"),
            (["68.53%"], [fonts["cell"]], [TEXT_MAIN], "center"),
            (["69.47%"], [fonts["cell"]], [TEXT_MUTED], "center"),
            (["73.88%"], [fonts["cell"]], [TEXT_MAIN], "center"),
            (["75.69%"], [fonts["cell"]], [TEXT_MAIN], "center"),
            (["76.18%"], [fonts["cell"]], [TEXT_MAIN], "center"),
            (["76.53%", "(+7.06% loop gain)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_GREEN, ACCENT_GREEN], "center"),
            (["-40% Pixel MSE", "(0.0513 → 0.0307)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_BLUE, TEXT_MUTED], "center"),
        ],
        [
            (["Heavy Clutter", "(45% drop + 15% spikes)"], [fonts["cell_bold"], fonts["footer"]], [TEXT_MAIN, TEXT_MUTED], "left"),
            (["52.24%"], [fonts["cell"]], [ACCENT_RED], "center"),
            (["84.93%"], [fonts["cell"]], [TEXT_MAIN], "center"),
            (["84.61%"], [fonts["cell"]], [TEXT_MUTED], "center"),
            (["85.58%"], [fonts["cell"]], [TEXT_MAIN], "center"),
            (["85.65%", "(+1.04% peak)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_GREEN, ACCENT_GREEN], "center"),
            (["85.56%"], [fonts["cell"]], [TEXT_MAIN], "center"),
            (["85.49%", "(+0.88% loop)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_GREEN, TEXT_MUTED], "center"),
            (["-74% Pixel MSE", "(0.1221 → 0.0318)"], [fonts["cell_bold"], fonts["footer"]], [ACCENT_BLUE, TEXT_MUTED], "center"),
        ],
    ]

    img_w = table_w + 2 * margin_x
    img_h = margin_top + title_h + header_h + len(rows) * row_h + footer_h + 24
    im = Image.new("RGB", (img_w, img_h), BG_OUTER)
    draw = ImageDraw.Draw(im)

    # Title & Subtitle
    draw.text(
        (margin_x, margin_top),
        "Table 2 — 100% Mask-Free Multi-Step Self-Reflection Loop (X_occ → z_k → X̂_k → E(X̂_k))",
        fill=TEXT_MAIN,
        font=fonts["title"],
    )
    draw.text(
        (margin_x, margin_top + 46),
        "Zero external occlusion mask passed at test time; compared against both Clean-Trained and Same-Corruption-Trained Standard MLPs",
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
    draw.text(
        (margin_x, fy),
        "Latent Cosine Alignment cos(z_k, z*_clean) across Steps 0 → 4:   Picket Fence: 0.8613 → 0.8746   ·   Center Blackout: 0.7365 → 0.8129   ·   Heavy Clutter: 0.8850 → 0.9037",
        fill=ACCENT_PURPLE,
        font=fonts["footer"],
    )
    return im


def main():
    fonts = load_fonts()
    im1 = render_table1(fonts)
    im2 = render_table2(fonts)

    p1 = ASSETS_DIR / "table1_benchmark_comparison.png"
    p2 = ASSETS_DIR / "table2_maskfree_occlusion_loop.png"
    im1.save(p1)
    im2.save(p2)

    # Also save a combined stacked image
    comb_w = max(im1.width, im2.width)
    comb_h = im1.height + im2.height + 20
    comb = Image.new("RGB", (comb_w, comb_h), BG_OUTER)
    comb.paste(im1, (0, 0))
    comb.paste(im2, (0, im1.height + 20))
    p_comb = ASSETS_DIR / "tables_combined.png"
    comb.save(p_comb)
    print(f"Saved:\n  - {p1} ({im1.size})\n  - {p2} ({im2.size})\n  - {p_comb} ({comb.size})")


if __name__ == "__main__":
    main()
