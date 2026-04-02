#!/usr/bin/env python3
"""
figmm_fig11_case_study_stroke.py  (renamed from figmm_case_study_stroke.py) — ACM MM 2026
Output: figures/fig11_case_study_stroke.pdf
============================================================================
Stroke-complexity case study: 5 characters spanning all stroke bins
(1–5 / 6–10 / 11–15 / 16–20 / 21+), each shown with all 6 methods.

Design echoes figmm_fig6_stroke.py color scheme:
  Image-only (ASPL/Glaze/AMP/XTransfer): blue shades
  Text-only  (Nightshade):               deep orange
  Image+Text (MMCoA):                    dark green

Layout (5 rows × 7 cols):
  ┌──────────┬──────┬──────┬──────┬──────┬──────┬──────┐
  │  Bin     │ ASPL │ Glaze│  AMP │  XT  │  NS  │MMCoA │
  ├──────────┼──────┴──────┴──────┴──────┴──────┴──────┤
  │1–5 (白)  │  original + 6 adv images (green border)  │
  │6–10(伞)  │  ...                                     │
  │11–15(歪) │  ...                                     │
  │16–20(壁) │  ...                                     │
  │21+ (囊)  │  ...                                     │
  └──────────┴──────────────────────────────────────────┘

Border: green = Protected  (all cells are fully protected here)
Per-image label: 5 VLM answers shown in two minirows
Output: figures/fig11_case_study_stroke.pdf
"""

import os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from PIL import Image
import matplotlib.font_manager as fm

# ── Fonts ──────────────────────────────────────────────────────────────────
CJK_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/AssetsV2/com_apple_MobileAsset_Font8/"
    "86ba2c91f017a3749571a82f2c6d890ac7ffb2fb.asset/AssetData/PingFang.ttc",
    "/Library/Fonts/Songti.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Arial Unicode.ttf",
]
CJK_FONT = next((p for p in CJK_CANDIDATES if os.path.exists(p)), None)
CJK_PROP = fm.FontProperties(fname=CJK_FONT) if CJK_FONT else None

plt.rcParams.update({
    "font.family":  "serif",
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})

# ── Paths ──────────────────────────────────────────────────────────────────
SRC_DIR  = "/Users/lzy/Code_lzy/posion_attack/outputs/ID_2k/batch_test_20260306_142512"
ADV_BASE = "/Users/lzy/Code_lzy/posion_attack/outputs/run_ID_full_20260310_223019/images"
OUT_PDF  = os.path.join(os.path.dirname(__file__), "../figures/fig11_case_study_stroke.pdf")

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

# ── Fig6 colour scheme ─────────────────────────────────────────────────────
METHOD_COLOR = {
    'ASPL':       '#1976D2',   # mid blue      (Image-only)
    'Glaze':      '#90CAF9',   # pale blue     (Image-only)
    'AMP':        '#0D47A1',   # deep blue     (Image-only)
    'XTransfer':  '#42A5F5',   # light blue    (Image-only)
    'Nightshade': '#E64A19',   # deep orange   (Text-only)
    'MMCoA':      '#2E7D32',   # dark green    (Image+Text)
}
MODALITY_LABEL = {
    'ASPL':       'Image', 'Glaze': 'Image',
    'AMP':        'Image', 'XTransfer': 'Image',
    'Nightshade': 'Text',  'MMCoA': 'Img+Txt',
}
VLM_COLORS = {
    'Qwen':   '#1f77b4',
    'Kimi':   '#ff7f0e',
    'GPT':    '#2ca02c',
    'Gemini': '#9467bd',
    'GLM':    '#8c564b',
}
GREEN = '#2ca02c'
RED   = '#d62728'

# ── Characters (one per stroke bin) ───────────────────────────────────────
# (stem, char, strokes, bin_label, struct_label)
CHARS = [
    ('mosaic_0193_image_0010', '白', 5,  '1–5',   'standalone'),
    ('mosaic_0090_image_0106', '伞', 6,  '6–10',  'standalone'),
    ('mosaic_0015_image_0027', '歪', 11, '11–15', 'top-bottom'),
    ('mosaic_0367_image_0076', '壁', 16, '16–20', 'left-right'),
    ('mosaic_0820_image_0098', '囊', 22, '21+',   'enclosure'),
]

# ── All-VLM Q3 answers (hard-coded from API) ───────────────────────────────
# Format: RESULTS[stem][method] = {vlm: answer}  (all correct=False)
RESULTS = {
    # 白 (5 strokes) ─ mosaic_0193_image_0010
    'mosaic_0193_image_0010': {
        'ASPL':       {'Qwen':'目',  'Kimi':'目',  'GPT':'日',  'Gemini':'目', 'GLM':'—'},
        'Glaze':      {'Qwen':'—',   'Kimi':'—',   'GPT':'三',  'Gemini':'目', 'GLM':'—'},
        'AMP':        {'Qwen':'—',   'Kimi':'无',  'GPT':'三',  'Gemini':'目', 'GLM':'—'},
        'XTransfer':  {'Qwen':'—',   'Kimi':'山',  'GPT':'白',  'Gemini':'目', 'GLM':'—'},  # GPT correct→red
        'Nightshade': {'Qwen':'山',  'Kimi':'无',  'GPT':'日',  'Gemini':'目', 'GLM':'—'},
        'MMCoA':      {'Qwen':'—',   'Kimi':'—',   'GPT':'目',  'Gemini':'目', 'GLM':'—'},
    },
    # 伞 (6 strokes) ─ mosaic_0090_image_0106
    'mosaic_0090_image_0106': {
        'ASPL':       {'Qwen':'—',  'Kimi':'—',  'GPT':'山',  'Gemini':'山', 'GLM':'—'},
        'Glaze':      {'Qwen':'山', 'Kimi':'山', 'GPT':'山',  'Gemini':'本', 'GLM':'—'},
        'AMP':        {'Qwen':'—',  'Kimi':'山', 'GPT':'山',  'Gemini':'山', 'GLM':'—'},
        'XTransfer':  {'Qwen':'—',  'Kimi':'山', 'GPT':'—',   'Gemini':'山', 'GLM':'—'},
        'Nightshade': {'Qwen':'—',  'Kimi':'山', 'GPT':'山',  'Gemini':'山', 'GLM':'—'},
        'MMCoA':      {'Qwen':'—',  'Kimi':'山', 'GPT':'—',   'Gemini':'米', 'GLM':'—'},
    },
    # 歪 (11 strokes) ─ mosaic_0015_image_0027
    'mosaic_0015_image_0027': {
        'ASPL':       {'Qwen':'—',  'Kimi':'无', 'GPT':'—',  'Gemini':'山', 'GLM':'—'},
        'Glaze':      {'Qwen':'—',  'Kimi':'—',  'GPT':'—',  'Gemini':'雨', 'GLM':'—'},
        'AMP':        {'Qwen':'—',  'Kimi':'—',  'GPT':'—',  'Gemini':'秋', 'GLM':'—'},
        'XTransfer':  {'Qwen':'—',  'Kimi':'无', 'GPT':'山', 'Gemini':'休', 'GLM':'—'},
        'Nightshade': {'Qwen':'—',  'Kimi':'无', 'GPT':'田', 'Gemini':'旗', 'GLM':'—'},
        'MMCoA':      {'Qwen':'—',  'Kimi':'—',  'GPT':'山', 'Gemini':'中', 'GLM':'—'},
    },
    # 壁 (16 strokes) ─ mosaic_0367_image_0076
    'mosaic_0367_image_0076': {
        'ASPL':       {'Qwen':'—',  'Kimi':'无', 'GPT':'—',  'Gemini':'多', 'GLM':'—'},
        'Glaze':      {'Qwen':'鼎', 'Kimi':'川', 'GPT':'山', 'Gemini':'春', 'GLM':'—'},
        'AMP':        {'Qwen':'—',  'Kimi':'山', 'GPT':'山', 'Gemini':'多', 'GLM':'—'},
        'XTransfer':  {'Qwen':'無', 'Kimi':'无', 'GPT':'—',  'Gemini':'得', 'GLM':'—'},
        'Nightshade': {'Qwen':'山', 'Kimi':'无', 'GPT':'田', 'Gemini':'雷', 'GLM':'—'},
        'MMCoA':      {'Qwen':'山', 'Kimi':'川', 'GPT':'—',  'Gemini':'美', 'GLM':'—'},
    },
    # 囊 (22 strokes) ─ mosaic_0820_image_0098
    'mosaic_0820_image_0098': {
        'ASPL':       {'Qwen':'—',  'Kimi':'无', 'GPT':'田', 'Gemini':'美', 'GLM':'梯'},
        'Glaze':      {'Qwen':'大', 'Kimi':'山', 'GPT':'—',  'Gemini':'善', 'GLM':'梯'},
        'AMP':        {'Qwen':'—',  'Kimi':'人', 'GPT':'田', 'Gemini':'目', 'GLM':'—'},
        'XTransfer':  {'Qwen':'—',  'Kimi':'九', 'GPT':'—',  'Gemini':'寿', 'GLM':'梯'},
        'Nightshade': {'Qwen':'—',  'Kimi':'人', 'GPT':'—',  'Gemini':'事', 'GLM':'梯'},
        'MMCoA':      {'Qwen':'人', 'Kimi':'人', 'GPT':'—',  'Gemini':'目', 'GLM':'梯'},
    },
}

# 白×XTransfer GPT='白' is correct → mark red
CORRECT = {
    ('mosaic_0193_image_0010', 'XTransfer', 'GPT'): True,   # GPT correctly said 白
}

# ── Image loader ──────────────────────────────────────────────────────────
def load_img(path):
    try:
        return np.array(Image.open(path).convert('RGB'))
    except:
        return np.ones((256, 256, 3), dtype=np.uint8) * 200

def draw_border(ax, color, lw=3.0):
    for sp in ax.spines.values():
        sp.set_edgecolor(color); sp.set_linewidth(lw); sp.set_visible(True)

# ── Layout constants ───────────────────────────────────────────────────────
N_CHARS   = len(CHARS)
N_METHODS = len(METHODS)
COLS      = 1 + N_METHODS          # bin-label col + 6 method cols

IMG_H  = 1.35   # image cell height (inches)
LBL_H  = 1.00   # VLM-label strip height
GAP_H  = 0.18   # gap between character rows
HDR_H  = 0.40   # header row height

fig_w = 11.0
total_h = HDR_H + N_CHARS * (IMG_H + LBL_H) + (N_CHARS - 1) * GAP_H + 0.25

fig = plt.figure(figsize=(fig_w, total_h), dpi=150)
fig.patch.set_facecolor('white')

# height_ratios for gridspec
hr_list = [HDR_H]
for i in range(N_CHARS):
    hr_list.append(IMG_H)
    hr_list.append(LBL_H)
    if i < N_CHARS - 1:
        hr_list.append(GAP_H)

N_ROWS = len(hr_list)
gs = gridspec.GridSpec(
    N_ROWS, COLS,
    figure=fig,
    height_ratios=[h / sum(hr_list) for h in hr_list],
    hspace=0.0, wspace=0.05,
    left=0.01, right=0.99, top=0.97, bottom=0.01
)

# ── Header row (row 0) ─────────────────────────────────────────────────────
# col 0: blank / "Stroke Bin"
ax_h0 = fig.add_subplot(gs[0, 0])
ax_h0.axis('off')
ax_h0.text(0.5, 0.5, 'Stroke\nBin', ha='center', va='center',
           fontsize=8, fontfamily='serif', fontweight='bold',
           transform=ax_h0.transAxes)

for mi, method in enumerate(METHODS):
    ax_hm = fig.add_subplot(gs[0, mi+1])
    ax_hm.axis('off')
    col_c = METHOD_COLOR[method]
    mod   = MODALITY_LABEL[method]
    ax_hm.text(0.5, 0.65, method, ha='center', va='center',
               fontsize=8.5, fontfamily='serif', fontweight='bold',
               color=col_c, transform=ax_hm.transAxes)
    ax_hm.text(0.5, 0.15, f'[{mod}]', ha='center', va='center',
               fontsize=6.5, fontfamily='serif', color=col_c,
               style='italic', transform=ax_hm.transAxes)

# ── Character rows ─────────────────────────────────────────────────────────
def row_start(char_idx):
    """gridspec row index for the image row of character char_idx"""
    # header=0, then each char takes 2 rows (img+lbl) + optional gap row
    r = 1
    for i in range(char_idx):
        r += 2  # img + lbl
        r += 1  # gap
    return r

for ci, (stem, char, strokes, bin_lbl, struct) in enumerate(CHARS):
    img_row = row_start(ci)
    lbl_row = img_row + 1

    src_img = load_img(os.path.join(SRC_DIR, f'{stem}.png'))

    # ── col 0: bin label + source image ──────────────────────────────────
    ax_src = fig.add_subplot(gs[img_row, 0])
    ax_src.imshow(src_img, aspect='auto')
    ax_src.set_xticks([]); ax_src.set_yticks([])
    for sp in ax_src.spines.values():
        sp.set_linewidth(1.5); sp.set_edgecolor('#888')

    ax_src_lbl = fig.add_subplot(gs[lbl_row, 0])
    ax_src_lbl.axis('off')
    # Bin badge (matches Fig6 x-axis labels)
    ax_src_lbl.text(0.5, 0.80, bin_lbl, ha='center', va='top',
                    fontsize=9, fontfamily='serif', fontweight='bold',
                    color='#1A1A2E', transform=ax_src_lbl.transAxes)
    ax_src_lbl.text(0.5, 0.48, char, ha='center', va='top',
                    fontsize=13, fontweight='bold',
                    fontproperties=CJK_PROP, color='#333',
                    transform=ax_src_lbl.transAxes)
    ax_src_lbl.text(0.5, 0.10, f'{strokes}str · {struct}', ha='center', va='top',
                    fontsize=5.8, fontfamily='serif', color='#666',
                    transform=ax_src_lbl.transAxes)

    # ── method columns ────────────────────────────────────────────────────
    for mi, method in enumerate(METHODS):
        col = mi + 1
        adv_path = os.path.join(ADV_BASE, METHOD_DIRS[method], f'{stem}_adv.png')
        adv_img  = load_img(adv_path)
        answers  = RESULTS[stem][method]

        # Border color: check if any VLM is correct → red
        any_fail = any(CORRECT.get((stem, method, v), False) for v in VLMS)
        border_c = RED if any_fail else GREEN

        ax_img = fig.add_subplot(gs[img_row, col])
        ax_img.imshow(adv_img, aspect='auto')
        ax_img.set_xticks([]); ax_img.set_yticks([])
        draw_border(ax_img, border_c, lw=2.8)

        # VLM answer labels — 5 sub-cells, each with name on top + big answer char
        lbl_gs = gridspec.GridSpecFromSubplotSpec(
            1, 5, subplot_spec=gs[lbl_row, col],
            hspace=0, wspace=0.05,
        )
        VLM_FULL = {'Qwen': 'Qwen', 'Kimi': 'Kimi', 'GPT': 'GPT', 'Gemini': 'Gemini', 'GLM': 'GLM'}
        for vi, vlm in enumerate(VLMS):
            ans = answers.get(vlm, '?')
            is_correct = CORRECT.get((stem, method, vlm), False)
            txt_color  = RED if is_correct else VLM_COLORS[vlm]

            ax_v = fig.add_subplot(lbl_gs[0, vi])
            ax_v.axis('off')
            ax_v.set_xlim(0, 1); ax_v.set_ylim(0, 1)

            # Light background tint
            ax_v.add_patch(plt.Rectangle((0.04, 0.04), 0.92, 0.92,
                facecolor=txt_color, alpha=0.08, edgecolor='none',
                transform=ax_v.transAxes, zorder=0))

            # VLM name — small, top
            ax_v.text(0.50, 0.80, VLM_FULL[vlm],
                      ha='center', va='center',
                      fontsize=5.5, color=txt_color, fontweight='bold',
                      fontfamily='serif', transform=ax_v.transAxes)
            # Answer char — large, center-bottom
            ax_v.text(0.50, 0.33, ans,
                      ha='center', va='center',
                      fontsize=11.0, fontweight='bold', color=txt_color,
                      fontproperties=CJK_PROP,
                      transform=ax_v.transAxes)

# ── Supra-title ────────────────────────────────────────────────────────────
fig.text(0.50, 0.998,
         "Stroke-Complexity Case Study: Protection Across All Stroke Bins",
         ha='center', va='top',
         fontsize=9.0, fontfamily='serif', fontweight='bold', color='#1A1A2E')

out = os.path.abspath(OUT_PDF)
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, bbox_inches='tight', dpi=150, facecolor='white')
print(f"[OK] Saved → {out}")
plt.close()
