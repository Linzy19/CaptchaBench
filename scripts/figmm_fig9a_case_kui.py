#!/usr/bin/env python3
"""
figmm_fig9a_case_kui.py — ACM MM 2026
============================================================================
Case-study grid for character 「魁」(mosaic_0083_image_0002).
5 VLMs × 6 attack methods = 30 cells, ALL fully protected (green borders).

Layout:
  ┌────────────────────────────────────────────────────────┐
  │  Character header: 「魁」— All 30 Cells Protected      │
  ├────────────────────────────────────────────────────────┤
  │  [method header: ASPL  Glaze  AMP  XTransfer  NS  MM]  │
  ├──────────┬────────────────────────────────────────────┤
  │ Qwen-VL  │  6 images (green border, →answer overlay)  │
  │ Kimi 2.5 │  ...                                       │
  │ GPT-5.2  │  ...                                       │
  │Gemini-3.0│  ...                                       │
  │ GLM-4V   │  ...                                       │
  └──────────┴────────────────────────────────────────────┘
  [legend strip]

Output: figures/fig9a_case_kui.pdf
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import numpy as np
from PIL import Image
from matplotlib import font_manager as fm

# ── font setup ────────────────────────────────────────────────────────────────
CJK_CANDIDATES = [
    "/System/Library/AssetsV2/com_apple_MobileAsset_Font8/"
    "86ba2c91f017a3749571a82f2c6d890ac7ffb2fb.asset/AssetData/PingFang.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Songti.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Arial Unicode.ttf",
]
CJK_FONT_PATH = next((p for p in CJK_CANDIDATES if os.path.exists(p)), None)
HAS_CJK = CJK_FONT_PATH is not None
if HAS_CJK:
    fm.fontManager.addfont(CJK_FONT_PATH)

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"]  = 42
rcParams["font.family"]  = "sans-serif"

def cjk_prop(size, bold=False):
    w = "bold" if bold else "normal"
    if HAS_CJK:
        return fm.FontProperties(fname=CJK_FONT_PATH, size=size, weight=w)
    return fm.FontProperties(size=size, weight=w)

# ── paths ─────────────────────────────────────────────────────────────────────
HERE     = Path(__file__).parent
IMG_BASE = Path("/Users/lzy/Code_lzy/posion_attack/outputs/run_ID_full_20260310_223019/images")
OUT_PDF  = HERE.parent / "figures" / "fig9a_case_kui.pdf"

# ── data ──────────────────────────────────────────────────────────────────────
CHAR    = "魁"
STEM    = "mosaic_0083_image_0002"
TITLE   = "Case Study: \u9b41  \u2014  All 30 Cells Protected"

METHODS = [
    {"key": "ASPL",       "label": "ASPL",       "color": "#0D47A1"},
    {"key": "Glaze",      "label": "Glaze",       "color": "#1976D2"},
    {"key": "AMP",        "label": "AMP",         "color": "#42A5F5"},
    {"key": "XTransfer",  "label": "XTransfer",   "color": "#90CAF9"},
    {"key": "Nightshade", "label": "Nightshade",  "color": "#E64A19"},
    {"key": "MMCoA",      "label": "MMCoA",       "color": "#2E7D32"},
]

VLM_ORDER = [
    {"key": "qwen",   "label": "Qwen-VL"},
    {"key": "kimi",   "label": "Kimi 2.5"},
    {"key": "gpt52",  "label": "GPT-5.2"},
    {"key": "gemini", "label": "Gemini-3.0"},
    {"key": "glm4",   "label": "GLM-4V"},
]

METHOD_DIRS = {
    "ASPL":       "aspl_eps0.05_steps200",
    "Glaze":      "mi_eps16_steps300",
    "AMP":        "attackvlm_eps8_steps300",
    "XTransfer":  "xtransfer_eps12_steps300",
    "Nightshade": "nightshade_eps0.05_steps500",
    "MMCoA":      "mmcoa_eps1_steps100",
}

# ALL correct=False → green border
RESULTS = {
    "ASPL":       {"qwen": (False,"无"),  "kimi": (False,"无"),  "gpt52": (False,"山"),  "gemini": (False,"出"),  "glm4": (False,"无")},
    "Glaze":      {"qwen": (False,"无"),  "kimi": (False,"无"),  "gpt52": (False,"—"),   "gemini": (False,"春"), "glm4": (False,"无")},
    "AMP":        {"qwen": (False,"山"),  "kimi": (False,"山"),  "gpt52": (False,"山"),  "gemini": (False,"爱"), "glm4": (False,"无")},
    "XTransfer":  {"qwen": (False,"山"),  "kimi": (False,"山"),  "gpt52": (False,"—"),   "gemini": (False,"精"), "glm4": (False,"无")},
    "Nightshade": {"qwen": (False,"无"),  "kimi": (False,"无"),  "gpt52": (False,"山"),  "gemini": (False,"山"), "glm4": (False,"无")},
    "MMCoA":      {"qwen": (False,"无"),  "kimi": (False,"无"),  "gpt52": (False,"—"),   "gemini": (False,"佛"), "glm4": (False,"海")},
}

C_PROTECTED = "#2E7D32"   # green
C_FAILED    = "#C62828"   # red (not used here)
BORDER_LW   = 2.6

N_COLS = len(METHODS)   # 6
N_ROWS = len(VLM_ORDER) # 5

FONT_TITLE  = 11.0
FONT_METHOD = 10.5
FONT_VLM    = 9.5
FONT_ANS    = 9.0
FONT_CHAR   = 14.0

# ── helpers ───────────────────────────────────────────────────────────────────
def load_img(path):
    try:
        return np.array(Image.open(path).convert("RGB"))
    except Exception:
        return np.ones((256, 256, 3), dtype=np.uint8) * 200

def set_spines(ax, color, lw):
    for sp in ax.spines.values():
        sp.set_edgecolor(color); sp.set_linewidth(lw)
        sp.set_visible(True)

def extract_cjk(s):
    s = str(s).strip()
    if not s or s.lower() in ("none", "n/a", "?", "—", ""):
        return "—"
    for ch in s:
        if "\u4e00" <= ch <= "\u9fff":
            return ch
    return s[:2] if s else "—"

# ── layout ────────────────────────────────────────────────────────────────────
# Outer GridSpec: 4 rows × 2 cols
#   rows: [title | method-header | image-block | legend]
#   cols: [vlm-labels | images]

VLM_W       = 0.13   # wider left column for VLM labels (was 0.095)
FIG_W       = 9.0
CELL_W_IN   = FIG_W * (1.0 - VLM_W) / N_COLS
BLOCK_H_IN  = CELL_W_IN * N_ROWS
TITLE_H_IN  = 0.32
HDR_H_IN    = 0.36
LEGEND_H_IN = 0.34
FIG_H = TITLE_H_IN + HDR_H_IN + BLOCK_H_IN + LEGEND_H_IN

_T = FIG_H
H_TITLE = TITLE_H_IN  / _T
H_HDR   = HDR_H_IN    / _T
H_BLK   = BLOCK_H_IN  / _T
H_LEG   = LEGEND_H_IN / _T

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

outer = gridspec.GridSpec(
    4, 2, figure=fig,
    left=0.005, right=0.998,
    bottom=0.005, top=0.996,
    hspace=0.0, wspace=0.0,
    height_ratios=[H_TITLE, H_HDR, H_BLK, H_LEG],
    width_ratios=[VLM_W, 1.0 - VLM_W],
)

# ── Row 0: title spanning both cols ──────────────────────────────────────────
ax_title = fig.add_subplot(outer[0, :])
ax_title.set_axis_off()
ax_title.text(0.5, 0.5, TITLE,
              transform=ax_title.transAxes, ha="center", va="center",
              fontsize=FONT_TITLE, fontweight="bold", color="#1A1A2E",
              fontproperties=cjk_prop(FONT_TITLE, bold=True))

# ── Row 1, Col 0: blank corner ────────────────────────────────────────────────
ax_tl = fig.add_subplot(outer[1, 0])
ax_tl.set_axis_off()

# ── Row 1, Col 1: method name headers ────────────────────────────────────────
hdr_gs = gridspec.GridSpecFromSubplotSpec(
    1, N_COLS, subplot_spec=outer[1, 1],
    hspace=0, wspace=0,
)
for ci, m in enumerate(METHODS):
    ax = fig.add_subplot(hdr_gs[0, ci])
    ax.set_axis_off()
    ax.text(0.5, 0.48, m["label"],
            transform=ax.transAxes, ha="center", va="center",
            fontsize=FONT_METHOD, fontweight="bold", color=m["color"])

# ── Row 2, Col 0: blank (VLM labels will be on image axes) ───────────────────
ax_vl = fig.add_subplot(outer[2, 0])
ax_vl.set_axis_off()
# VLM labels drawn in left col as text, spaced evenly
for ri, vlm in enumerate(VLM_ORDER):
    ypos = 1.0 - (ri + 0.5) / N_ROWS
    ax_vl.text(0.88, ypos, vlm["label"],
               transform=ax_vl.transAxes, ha="right", va="center",
               fontsize=FONT_VLM, fontweight="bold", color="#212121")
# right-edge separator line
ax_vl.axvline(x=0.92, ymin=0.01, ymax=0.99,
              color="#BDBDBD", linewidth=0.8, zorder=1)

# ── Row 2, Col 1: 5×6 image grid ─────────────────────────────────────────────
img_gs = gridspec.GridSpecFromSubplotSpec(
    N_ROWS, N_COLS, subplot_spec=outer[2, 1],
    hspace=0.025, wspace=0.025,
)

for ri, vlm in enumerate(VLM_ORDER):
    for ci, m in enumerate(METHODS):
        ax = fig.add_subplot(img_gs[ri, ci])
        ax.set_xticks([]); ax.set_yticks([])
        ax.tick_params(bottom=False, left=False, top=False, right=False)

        img_path = IMG_BASE / METHOD_DIRS[m["key"]] / f"{STEM}_adv.png"
        try:
            img = np.array(Image.open(img_path).convert("RGB"))
            ax.imshow(img, aspect="auto", interpolation="lanczos")
        except Exception:
            ax.set_facecolor("#BDBDBD")
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=6, color="#555")

        correct, ans_raw = RESULTS[m["key"]][vlm["key"]]
        disp = extract_cjk(ans_raw)

        border_col = C_FAILED if correct else C_PROTECTED
        set_spines(ax, border_col, BORDER_LW)

        # Answer tag overlay
        ax.text(0.50, 0.04, "→" + disp,
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=FONT_ANS, fontweight="bold",
                color="#FFFFFF",
                fontproperties=cjk_prop(FONT_ANS, bold=True),
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="#000000", edgecolor="none", alpha=0.65),
                zorder=5)

# ── Row 3: legend ─────────────────────────────────────────────────────────────
ax_leg = fig.add_subplot(outer[3, :])
ax_leg.set_axis_off()
ax_leg.text(0.50, 0.55, "Border: green = Protected (VLM confused)",
            transform=ax_leg.transAxes, ha="center", va="center",
            fontsize=9.5, color="#333333")

# ── save ──────────────────────────────────────────────────────────────────────
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT_PDF}")
plt.close(fig)
print("Done.")
