"""
figmm_case_study_grid.py — ACM MM 2026, Figure: Case Study Grid
============================================================================
Two characters, each shown as a 5×6 grid (5 VLMs × 6 attack methods).

Character 1: 「魁」(mosaic_0083_image_0002) — ALL methods protect successfully
Character 2: 「渐」(mosaic_0151_image_0112) — partial failure: GLM-4V correct
             in Glaze/AMP/XTransfer/Nightshade; shows GLM-4V vs. other VLMs

Layout:
  ┌───────────────────────────────────────────┐
  │  [method header: ASPL Glaze AMP X NS MM]  │
  ├──────────┬────────────────────────────────┤
  │  魁      │  5 VLMs × 6 methods grid       │
  │(all prot)│  green borders everywhere       │
  ├──────────┼────────────────────────────────┤
  │  渐      │  5 VLMs × 6 methods grid       │
  │(partial) │  mix of green / red borders     │
  ├──────────┴────────────────────────────────┤
  │  [legend: green=Protected  red=Failed]    │
  └───────────────────────────────────────────┘

Border: green = Protected (VLM confused), red = Failed (VLM correct)
Note: 「唯」 is used in Fig.1 (main text teaser); replaced here with 「渐」.
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
CJK_FONT_PATH = (
    "/System/Library/AssetsV2/com_apple_MobileAsset_Font8/"
    "86ba2c91f017a3749571a82f2c6d890ac7ffb2fb.asset/AssetData/PingFang.ttc"
)
HAS_CJK = os.path.exists(CJK_FONT_PATH)
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

# Adversarial image paths (confirmed to exist via JSON lookup)
ADV = {
    "魁": {
        "ASPL":       IMG_BASE / "aspl_eps0.05_steps200"      / "mosaic_0083_image_0002_adv.png",
        "Glaze":      IMG_BASE / "mi_eps16_steps300"           / "mosaic_0083_image_0002_adv.png",
        "AMP":        IMG_BASE / "attackvlm_eps8_steps300"     / "mosaic_0083_image_0002_adv.png",
        "XTransfer":  IMG_BASE / "xtransfer_eps12_steps300"    / "mosaic_0083_image_0002_adv.png",
        "Nightshade": IMG_BASE / "nightshade_eps0.05_steps500" / "mosaic_0083_image_0002_adv.png",
        "MMCoA":      IMG_BASE / "mmcoa_eps1_steps100"         / "mosaic_0083_image_0002_adv.png",
    },
    "渐": {
        "ASPL":       IMG_BASE / "aspl_eps0.05_steps200"      / "mosaic_0151_image_0112_adv.png",
        "Glaze":      IMG_BASE / "mi_eps16_steps300"           / "mosaic_0151_image_0112_adv.png",
        "AMP":        IMG_BASE / "attackvlm_eps8_steps300"     / "mosaic_0151_image_0112_adv.png",
        "XTransfer":  IMG_BASE / "xtransfer_eps12_steps300"    / "mosaic_0151_image_0112_adv.png",
        "Nightshade": IMG_BASE / "nightshade_eps0.05_steps500" / "mosaic_0151_image_0112_adv.png",
        "MMCoA":      IMG_BASE / "mmcoa_eps1_steps100"         / "mosaic_0151_image_0112_adv.png",
    },
}

# ── hard-coded VLM results ────────────────────────────────────────────────────
# (correct, answer_char)
# correct=True  → VLM recognises the target → FAILED (red border)
# correct=False → VLM confused               → Protected (green border)

RESULTS = {
    # ── Character 1: 「魁」— ALL correct=False (fully protected) ────────────
    "魁": {
        "ASPL": {
            "qwen":   (False, "无"),  "kimi":   (False, "无"),
            "gpt52":  (False, "山"),  "gemini": (False, "出"),  "glm4": (False, "无"),
        },
        "Glaze": {
            "qwen":   (False, "无"),  "kimi":   (False, "无"),
            "gpt52":  (False, "—"),   "gemini": (False, "春"),  "glm4": (False, "无"),
        },
        "AMP": {
            "qwen":   (False, "山"),  "kimi":   (False, "山"),
            "gpt52":  (False, "山"),  "gemini": (False, "爱"),  "glm4": (False, "无"),
        },
        "XTransfer": {
            "qwen":   (False, "山"),  "kimi":   (False, "山"),
            "gpt52":  (False, "—"),   "gemini": (False, "精"),  "glm4": (False, "无"),
        },
        "Nightshade": {
            "qwen":   (False, "无"),  "kimi":   (False, "无"),
            "gpt52":  (False, "山"),  "gemini": (False, "山"),  "glm4": (False, "无"),
        },
        "MMCoA": {
            "qwen":   (False, "无"),  "kimi":   (False, "无"),
            "gpt52":  (False, "—"),   "gemini": (False, "佛"),  "glm4": (False, "海"),
        },
    },

    # ── Character 2: 「渐」— partial failures (GLM-4V recognises in 4 methods) ──
    "渐": {
        "ASPL": {
            "qwen":   (False, "井"),  "kimi":   (False, "无"),
            "gpt52":  (False, "井"),  "gemini": (False, "撕"),  "glm4": (False, "—"),
        },
        "Glaze": {
            "qwen":   (False, "无"),  "kimi":   (False, "无"),
            "gpt52":  (False, "新"),  "gemini": (False, "特"),  "glm4": (True,  "渐"),
        },
        "AMP": {
            "qwen":   (False, "无"),  "kimi":   (False, "无"),
            "gpt52":  (False, "川"),  "gemini": (False, "街"),  "glm4": (True,  "渐"),
        },
        "XTransfer": {
            "qwen":   (False, "无"),  "kimi":   (False, "无"),
            "gpt52":  (False, "鼎"),  "gemini": (False, "街"),  "glm4": (True,  "渐"),
        },
        "Nightshade": {
            "qwen":   (False, "无"),  "kimi":   (False, "无"),
            "gpt52":  (False, "斬"),  "gemini": (False, "街"),  "glm4": (True,  "渐"),
        },
        "MMCoA": {
            "qwen":   (False, "无"),  "kimi":   (False, "无"),
            "gpt52":  (False, "川"),  "gemini": (False, "删"),  "glm4": (False, "潮"),
        },
    },
}

# ── method & VLM definitions ──────────────────────────────────────────────────
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

CHARS = [
    {"char": "魁", "label": "魁 (all protected)"},
    {"char": "渐", "label": "渐 (partial)"},
]

# ── colors ────────────────────────────────────────────────────────────────────
C_PROTECTED = "#2E7D32"   # green border → Protected
C_FAILED    = "#C62828"   # red border   → Failed
BORDER_LW   = 2.6

# ── layout constants ──────────────────────────────────────────────────────────
N_COLS = 6   # 6 attack methods
N_ROWS = 5   # 5 VLMs

FIG_W      = 8.5
VLM_W      = 0.095   # fraction of total width for the char-label column

# Make cells square
CELL_W_IN    = FIG_W * (1.0 - VLM_W) / N_COLS
BLOCK_H_IN   = CELL_W_IN * N_ROWS
HEADER_H_IN  = 0.36
SEP_H_IN     = 0.18
LEGEND_H_IN  = 0.36
FIG_H = HEADER_H_IN + 2.0 * BLOCK_H_IN + SEP_H_IN + LEGEND_H_IN

_T   = FIG_H
H_HDR = HEADER_H_IN / _T
H_BLK = BLOCK_H_IN  / _T
H_SEP = SEP_H_IN    / _T
H_LEG = LEGEND_H_IN / _T

FONT_METHOD = 11.5
FONT_VLM    = 10.0
FONT_ANS    = 9.5
FONT_CHAR   = 12.5

# ── helpers ───────────────────────────────────────────────────────────────────
def set_spines(ax, color, lw):
    for sp in ax.spines.values():
        sp.set_edgecolor(color); sp.set_linewidth(lw)
        sp.set_visible(True); sp.set_capstyle("butt"); sp.set_joinstyle("miter")

def extract_cjk(s):
    s = str(s).strip()
    if not s or s.lower() in ("none", "n/a", "?", "—", ""):
        return "—"
    for ch in s:
        if "\u4e00" <= ch <= "\u9fff":
            return ch
    return s[:2] if s else "—"

# ── build figure ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

# Outer GridSpec: 5 rows × 2 cols
#  rows: [header | block1 | sep | block2 | legend]
#  cols: [char_label | images]
outer = gridspec.GridSpec(
    5, 2, figure=fig,
    left=0.005, right=0.998,
    bottom=0.005, top=0.996,
    hspace=0.0, wspace=0.0,
    height_ratios=[H_HDR, H_BLK, H_SEP, H_BLK, H_LEG],
    width_ratios=[VLM_W, 1.0 - VLM_W],
)

# ── Row 0, Col 1: method name headers ─────────────────────────────────────────
hdr_gs = gridspec.GridSpecFromSubplotSpec(
    1, N_COLS, subplot_spec=outer[0, 1],
    hspace=0, wspace=0,
)
for ci, m in enumerate(METHODS):
    ax = fig.add_subplot(hdr_gs[0, ci])
    ax.set_axis_off(); ax.patch.set_visible(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.text(0.5, 0.48, m["label"],
            transform=ax.transAxes, ha="center", va="center",
            fontsize=FONT_METHOD, fontweight="bold", color=m["color"])

# Row 0, Col 0: blank corner
ax_tl = fig.add_subplot(outer[0, 0])
ax_tl.set_axis_off(); ax_tl.patch.set_visible(False)

# ── Character blocks (rows 1 and 3) ───────────────────────────────────────────
for block_row, char_info in zip([1, 3], CHARS):
    ch    = char_info["char"]
    label = char_info["label"]

    # Left col: character label
    ax_lbl = fig.add_subplot(outer[block_row, 0])
    ax_lbl.set_axis_off(); ax_lbl.patch.set_visible(False)
    for sp in ax_lbl.spines.values(): sp.set_visible(False)
    # right-edge separator line
    ax_lbl.axvline(x=0.90, ymin=0.02, ymax=0.98,
                   color="#BDBDBD", linewidth=0.8, zorder=1)
    _prop = cjk_prop(FONT_CHAR, bold=True)
    ax_lbl.text(0.44, 0.50, label,
                transform=ax_lbl.transAxes, ha="center", va="center",
                fontsize=FONT_CHAR, fontweight="bold", color="#212121",
                fontproperties=_prop, multialignment="center")

    # Right col: 5×6 image grid
    img_gs = gridspec.GridSpecFromSubplotSpec(
        N_ROWS, N_COLS, subplot_spec=outer[block_row, 1],
        hspace=0.025, wspace=0.025,
    )

    for ri, vlm in enumerate(VLM_ORDER):
        for ci, m in enumerate(METHODS):
            ax = fig.add_subplot(img_gs[ri, ci])
            ax.set_xticks([]); ax.set_yticks([])
            ax.tick_params(bottom=False, left=False, top=False, right=False)

            img_path = ADV[ch][m["key"]]
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
                ax.imshow(img, aspect="auto", interpolation="lanczos")
            except Exception:
                ax.set_facecolor("#BDBDBD")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=6, color="#555")

            correct, ans_raw = RESULTS[ch][m["key"]][vlm["key"]]
            disp = extract_cjk(ans_raw)

            border_col = C_FAILED if correct else C_PROTECTED
            set_spines(ax, border_col, BORDER_LW)

            # Answer tag (bottom-centre overlay)
            ax.text(
                0.50, 0.04, "→" + disp,
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=FONT_ANS, fontweight="bold",
                color="#FFFFFF",
                fontproperties=cjk_prop(FONT_ANS, bold=True),
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="#000000", edgecolor="none", alpha=0.65),
                zorder=5,
            )

            # VLM row label on leftmost column
            if ci == 0:
                ax.set_ylabel(
                    vlm["label"],
                    fontsize=FONT_VLM, fontweight="bold", color="#212121",
                    rotation=0, ha="right", va="center", labelpad=4,
                )
                ax.yaxis.set_label_coords(-0.08, 0.50)

# ── Row 2: separator line ─────────────────────────────────────────────────────
ax_sep = fig.add_subplot(outer[2, :])
ax_sep.set_axis_off(); ax_sep.patch.set_visible(False)
ax_sep.axhline(y=0.5, xmin=0.005, xmax=0.995,
               color="#9E9E9E", linewidth=1.0, zorder=2)

# ── Row 4: legend ─────────────────────────────────────────────────────────────
leg_gs = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer[4, 1],
    hspace=0, wspace=0.04,
)
for li, (lc, lt) in enumerate([
    (C_PROTECTED, "Protected (VLM confused)"),
    (C_FAILED,    "Failed (VLM correct)"),
]):
    ax_l = fig.add_subplot(leg_gs[0, li])
    ax_l.set_axis_off(); ax_l.patch.set_visible(False)
    for sp in ax_l.spines.values(): sp.set_visible(False)
    rect = plt.Rectangle(
        (0.18, 0.15), 0.08, 0.70,
        facecolor=lc, edgecolor="none",
        transform=ax_l.transAxes, clip_on=False, zorder=3,
    )
    ax_l.add_patch(rect)
    ax_l.text(0.31, 0.50, lt,
              transform=ax_l.transAxes, ha="left", va="center",
              fontsize=11.5, color="#212121")

ax_ll = fig.add_subplot(outer[4, 0])
ax_ll.set_axis_off()

# ── save ───────────────────────────────────────────────────────────────────────
FIG_DIR = HERE.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

out = FIG_DIR / "fig9_case_study_grid.pdf"
fig.savefig(out, format="pdf", bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
plt.close(fig)
print("Done.")
