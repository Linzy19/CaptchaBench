#!/usr/bin/env python3
"""
figmm_fig10_case_study_simple.py
Simple-character case study grid for CaptchaBench paper (ACM MM 2026).

Shows two simple characters (伞, 歪) with ALL methods fully protecting
across ALL 5 VLMs — illustrating that even simple characters are
effectively protected.

Layout:
  One character per row → 2 rows
  Each row: [source_img | ASPL | Glaze | AMP | XTransfer | Nightshade | MMCoA]
  = 2 rows × 7 columns grid
  Below each adv image: 5 VLM answer labels (single horizontal row)

Style matches figmm_fig1_teaser.py: green border for protected.
Output: figures/fig10_case_study_simple.pdf
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from PIL import Image
import matplotlib.font_manager as fm

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SRC_DIR  = "/Users/lzy/Code_lzy/posion_attack/outputs/ID_2k/batch_test_20260306_142512"
ADV_BASE = "/Users/lzy/Code_lzy/posion_attack/outputs/run_ID_full_20260310_223019/images"
OUT_PDF  = "/Users/lzy/Code_lzy/Posion_paper/ACM_MM26/v2/figures/fig10_case_study_simple.pdf"

# CJK font for Chinese labels
CJK_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Songti.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Arial Unicode.ttf",
]
CJK_FONT = None
for p in CJK_CANDIDATES:
    if os.path.exists(p):
        CJK_FONT = p
        break
CJK_PROP = fm.FontProperties(fname=CJK_FONT) if CJK_FONT else None

METHOD_DIRS = {
    'ASPL':       'aspl_eps0.05_steps200',
    'Glaze':      'mi_eps16_steps300',
    'AMP':        'attackvlm_eps8_steps300',
    'XTransfer':  'xtransfer_eps12_steps300',
    'Nightshade': 'nightshade_eps0.05_steps500',
    'MMCoA':      'mmcoa_eps1_steps100',
}
METHODS = ['ASPL', 'Glaze', 'AMP', 'XTransfer', 'Nightshade', 'MMCoA']
VLMS    = ['Qwen', 'Kimi', 'GPT', 'Gemini', 'GLM']

# ─────────────────────────────────────────────────────────────────────────────
# Experiment results (hard-coded from actual VLM API responses)
# All cells: correct=False → Protected ✓
# ─────────────────────────────────────────────────────────────────────────────
# Format: RESULTS[char_stem][method][vlm] = answer_string
RESULTS = {
    'mosaic_0090_image_0053': {   # 伞 (umbrella / canopy) — standalone simple char
        'ASPL':       {'Qwen':'无',  'Kimi':'无',  'GPT':'山',  'Gemini':'主',  'GLM':'无'},
        'Glaze':      {'Qwen':'用',  'Kimi':'田',  'GPT':'山',  'Gemini':'合',  'GLM':'无'},
        'AMP':        {'Qwen':'无',  'Kimi':'无',  'GPT':'山',  'Gemini':'金',  'GLM':'无'},
        'XTransfer':  {'Qwen':'无',  'Kimi':'无',  'GPT':'山',  'Gemini':'业',  'GLM':'无'},
        'Nightshade': {'Qwen':'无',  'Kimi':'无',  'GPT':'山',  'Gemini':'金',  'GLM':'无'},
        'MMCoA':      {'Qwen':'田',  'Kimi':'田',  'GPT':'山',  'Gemini':'山',  'GLM':'山'},
    },
    'mosaic_0015_image_0027': {   # 歪 (crooked / slanted) — simple compound char
        'ASPL':       {'Qwen':'无',  'Kimi':'无',  'GPT':'none', 'Gemini':'山',  'GLM':'无'},
        'Glaze':      {'Qwen':'无',  'Kimi':'无',  'GPT':'none', 'Gemini':'雨',  'GLM':'无'},
        'AMP':        {'Qwen':'无',  'Kimi':'无',  'GPT':'none', 'Gemini':'秋',  'GLM':'无'},
        'XTransfer':  {'Qwen':'无',  'Kimi':'无',  'GPT':'山',   'Gemini':'休',  'GLM':'无'},
        'Nightshade': {'Qwen':'无',  'Kimi':'无',  'GPT':'田',   'Gemini':'旗',  'GLM':'无'},
        'MMCoA':      {'Qwen':'无',  'Kimi':'无',  'GPT':'山',   'Gemini':'中',  'GLM':'无'},
    },
}

CHAR_GT = {
    'mosaic_0090_image_0053': '伞',
    'mosaic_0015_image_0027': '歪',
}
CHAR_LABEL = {
    'mosaic_0090_image_0053': 'Ground Truth: \u4f1e  (umbrella, 6 strokes, standalone)',
    'mosaic_0015_image_0027': 'Ground Truth: \u6b6a  (crooked, 11 strokes, top-bottom)',
}

CHARS = ['mosaic_0090_image_0053', 'mosaic_0015_image_0027']

# ─────────────────────────────────────────────────────────────────────────────
# Colors
# ─────────────────────────────────────────────────────────────────────────────
GREEN  = '#2ca02c'
RED    = '#d62728'
BORDER_LW = 3.0
VLM_COLORS = {
    'Qwen':   '#1f77b4',
    'Kimi':   '#ff7f0e',
    'GPT':    '#2ca02c',
    'Gemini': '#9467bd',
    'GLM':    '#8c564b',
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def load_img(path):
    try:
        img = Image.open(path).convert('RGB')
        return np.array(img)
    except Exception as e:
        print(f"  [WARN] Cannot load {path}: {e}")
        arr = np.ones((256, 256, 3), dtype=np.uint8) * 200
        return arr

def draw_border(ax, color, lw=BORDER_LW):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)
        spine.set_visible(True)

# ─────────────────────────────────────────────────────────────────────────────
# Build figure
# ─────────────────────────────────────────────────────────────────────────────
N_METHODS = len(METHODS)
N_CHARS   = len(CHARS)
COLS      = 1 + N_METHODS   # source + 6 methods
ROWS      = N_CHARS          # 2 characters

# Each character block: image row + label row
fig_w    = 10.0
cell_h   = 1.5    # image cell height
label_h  = 1.00   # vlm label strip height
gap_h    = 0.30   # gap between character blocks
header_h = 0.35

# Recalculated total figure height with the new label_h
total_h = header_h + N_CHARS * (cell_h + label_h) + (N_CHARS - 1) * gap_h + 0.3

fig = plt.figure(figsize=(fig_w, total_h), dpi=150)
fig.patch.set_facecolor('white')

left_margin  = 0.01
right_margin = 0.01
top_margin   = 0.01
bot_margin   = 0.01

# rows: [header, img0, lbl0, gap, img1, lbl1]
hr_header = header_h / total_h
hr_cell   = cell_h   / total_h
hr_label  = label_h  / total_h
hr_gap    = gap_h    / total_h

hs = [hr_header, hr_cell, hr_label, hr_gap, hr_cell, hr_label]
outer = gridspec.GridSpec(
    len(hs), COLS,
    figure=fig,
    height_ratios=[h / sum(hs) for h in hs],
    hspace=0.0,
    wspace=0.04,
    left=left_margin, right=1-right_margin,
    top=1-top_margin, bottom=bot_margin
)

# ── Header row ────────────────────────────────────────────────────────────────
col_labels = ['Original'] + METHODS
for col, label in enumerate(col_labels):
    ax = fig.add_subplot(outer[0, col])
    ax.axis('off')
    fw = 'bold'
    fs = 8.5 if col == 0 else 8
    ax.text(0.5, 0.5, label, ha='center', va='center',
            fontsize=fs, fontweight=fw, transform=ax.transAxes)

# ── Character blocks ──────────────────────────────────────────────────────────
char_row_start = [1, 4]   # gridspec row indices for img row of each char

for ci, stem in enumerate(CHARS):
    img_row = char_row_start[ci]
    lbl_row = img_row + 1

    gt_char = CHAR_GT[stem]

    # Source image
    src_path = os.path.join(SRC_DIR, f'{stem}.png')
    src_img  = load_img(src_path)

    ax_src = fig.add_subplot(outer[img_row, 0])
    ax_src.imshow(src_img, aspect='auto')
    ax_src.set_xticks([]); ax_src.set_yticks([])
    for spine in ax_src.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor('#555555')

    # GT label below source
    ax_src_lbl = fig.add_subplot(outer[lbl_row, 0])
    ax_src_lbl.axis('off')
    ax_src_lbl.text(0.5, 0.85, gt_char, ha='center', va='top',
                    fontsize=14, fontweight='bold',
                    fontproperties=CJK_PROP, color='#222222',
                    transform=ax_src_lbl.transAxes)
    ax_src_lbl.text(0.5, 0.35, 'Original', ha='center', va='top',
                    fontsize=6.5, color='#555555',
                    transform=ax_src_lbl.transAxes)

    # Method columns
    for mi, method in enumerate(METHODS):
        col = mi + 1
        adv_dir  = os.path.join(ADV_BASE, METHOD_DIRS[method])
        adv_path = os.path.join(adv_dir, f'{stem}_adv.png')
        adv_img  = load_img(adv_path)

        answers = RESULTS[stem][method]
        # All protected (correct=False for all VLMs)
        border_color = GREEN

        ax_img = fig.add_subplot(outer[img_row, col])
        ax_img.imshow(adv_img, aspect='auto')
        ax_img.set_xticks([]); ax_img.set_yticks([])
        draw_border(ax_img, border_color, lw=BORDER_LW)

        # ── VLM answer labels: 5 sub-cells, each with name on top + big answer char ──
        lbl_gs = gridspec.GridSpecFromSubplotSpec(
            1, 5, subplot_spec=outer[lbl_row, col],
            hspace=0, wspace=0.05,
        )
        VLM_SHORT = {'Qwen': 'Qwen', 'Kimi': 'Kimi', 'GPT': 'GPT', 'Gemini': 'Gemini', 'GLM': 'GLM'}

        for vi, vlm in enumerate(VLMS):
            ans = answers.get(vlm, '?')
            # Normalise answer to a single CJK char or em-dash
            if len(ans) > 3:
                m = re.search(r'[\u4e00-\u9fff]', ans)
                ans = m.group() if m else ('\u2014' if ans.lower().startswith(('none','no')) else ans[:2])
            if ans in ('\u65e0', 'none', 'no', 'None'):
                ans = '\u2014'

            ax_v = fig.add_subplot(lbl_gs[0, vi])
            ax_v.axis('off')
            ax_v.set_xlim(0, 1); ax_v.set_ylim(0, 1)

            # Light background tint matching VLM color
            ax_v.add_patch(plt.Rectangle((0.04, 0.04), 0.92, 0.92,
                facecolor=VLM_COLORS[vlm], alpha=0.08, edgecolor='none',
                transform=ax_v.transAxes, zorder=0))

            # VLM name — small, top
            ax_v.text(0.50, 0.80, VLM_SHORT[vlm],
                      ha='center', va='center',
                      fontsize=5.5, color=VLM_COLORS[vlm], fontweight='bold',
                      transform=ax_v.transAxes)
            # Answer char — large, center-bottom
            ax_v.text(0.50, 0.33, ans,
                      ha='center', va='center',
                      fontsize=11.0, fontweight='bold',
                      color=VLM_COLORS[vlm],
                      fontproperties=CJK_PROP,
                      transform=ax_v.transAxes)

plt.savefig(OUT_PDF, bbox_inches='tight', dpi=150, facecolor='white')
print(f"[OK] Saved \u2192 {OUT_PDF}")
plt.close()
