"""
figmm_fig1_teaser.py — ACM MM 2026, Figure 1: Teaser Matrix  [v3]
============================================================================
5 VLMs: Qwen-VL / Kimi 2.5 / GPT-5.2 / Gemini-3.0 / GLM-4V
6 columns: 6 attack methods (Original column hidden)
Layout: rows = VLMs (5), columns = methods (6)
Case: 唯 (mosaic_1319_image_0109) — high variance + moderate glyph
  Protection rates: Glaze=40%, XTransfer=60%, MMCoA=60%, Nightshade=80%, ASPL=100%, AMP=100%
Key: green border = Protected (VLM confused), red = Failed (VLM correct)
"""

import os, json
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
HERE        = Path(__file__).parent
RESULTS_DIR = Path("/Users/lzy/Code_lzy/AttackVLM/eval_results_v2/run_20260316_201833")
RUN1_DIR    = Path("/Users/lzy/Code_lzy/AttackVLM/eval_results_v2/run_20260324_174525")
TARGET_STEM = "mosaic_1319_image_0109"
TARGET_CHAR = "唯"
SOURCE_IMAGE = Path(
    "/Users/lzy/Code_lzy/posion_attack/outputs/ID_2k/"
    "batch_test_20260306_142512/mosaic_1319_image_0109.png"
)

# ── hard-coded results ────────────────────────────────────────────────────────
# (correct, answer_char): correct=True → VLM recognizes → FAILED (red)
#                         correct=False → VLM confused   → Protected (green)
# Case: 唯 (mosaic_1319_image_0109) — high variance + moderate glyph
# Protection rates: Glaze=40%, XTransfer=60%, MMCoA=60%, Nightshade=80%, ASPL=100%, AMP=100%
# Qwen & Kimi from Run1; GPT-5.2/Gemini/GLM-4V from Run2
RESULTS = {
    # ASPL: 100% — all 5 VLMs confused (5 green, 0 red)
    "ASPL":       {
        "qwen":   (False,"雁"),  "kimi":  (False,"雅"),
        "gpt52":  (False,"鸣"),  "gemini": (False,"雁"),  "glm4": (False,"图"),
    },
    # Glaze: 40% — kimi+gpt52 confused (2 green, 3 red)
    "Glaze":      {
        "qwen":   (True, "唯"),  "kimi":  (False,"唱"),
        "gpt52":  (False,"囬"),  "gemini": (True, "唯"),  "glm4": (True, "唯"),
    },
    # AMP: 100% — all 5 VLMs confused (5 green, 0 red)
    "AMP":        {
        "qwen":   (False,"能"),  "kimi":  (False,"富"),
        "gpt52":  (False,"国"),  "gemini": (False,"谁"),  "glm4": (False,"图"),
    },
    # XTransfer: 60% — qwen+kimi+gpt52 confused (3 green, 2 red)
    "XTransfer":  {
        "qwen":   (False,"鹿"),  "kimi":  (False,"但"),
        "gpt52":  (False,"晶"),  "gemini": (True, "唯"),  "glm4": (True, "唯"),
    },
    # Nightshade: 80% — qwen+kimi+gpt52+glm4 confused (4 green, 1 red)
    "Nightshade": {
        "qwen":   (False,"会"),  "kimi":  (False,"鱼"),
        "gpt52":  (False,"囍"),  "gemini": (True, "唯"),  "glm4": (False,"图"),
    },
    # MMCoA: 60% — qwen+gpt52+gemini confused (3 green, 2 red)
    "MMCoA":      {
        "qwen":   (False,"非"),  "kimi":  (True, "唯"),
        "gpt52":  (False,"馬"),  "gemini": (False,"埋"),  "glm4": (True, "唯"),
    },
}

# ── load adv image paths from JSON ────────────────────────────────────────────
METHOD_FILES = {
    "ASPL":       (RESULTS_DIR, "results_SD-ID_aspl_eps0.05_steps200_20260316_201833.json"),
    "Glaze":      (RESULTS_DIR, "results_SD-ID_mi_eps16_steps300_20260316_201833.json"),
    "AMP":        (RESULTS_DIR, "results_SD-ID_attackvlm_eps8_steps300_20260316_201833.json"),
    "MMCoA":      (RESULTS_DIR, "results_SD-ID_mmcoa_eps1_steps100_20260316_201833.json"),
    "Nightshade": (RESULTS_DIR, "results_SD-ID_nightshade_eps0.05_steps500_20260316_201833.json"),
    "XTransfer":  (RESULTS_DIR, "results_SD-ID_xtransfer_eps12_steps300_20260316_201833.json"),
}
ADV_IMAGES = {}
for mkey, (rdir, fname) in METHOD_FILES.items():
    jpath = rdir / fname
    if not jpath.exists():
        print(f"WARNING: JSON not found: {jpath}")
        continue
    with open(jpath) as f:
        data = json.load(f)
    for item in data.get("adv_results", []):
        if TARGET_STEM in item.get("image_path", ""):
            ADV_IMAGES[mkey] = Path(item["image_path"])
            break

IMG_PATHS = {"source": SOURCE_IMAGE, **ADV_IMAGES}

# ── method & VLM definitions ──────────────────────────────────────────────────
# Method colors: per modality group — blue family (Image-only), orange (Text-only), green (Image+Text)
# Original column is hidden; only 6 attack methods shown
METHODS = [
    {"key": "ASPL",       "label": "ASPL",         "color": "#0D47A1"},   # deep blue (Image-only)
    {"key": "Glaze",      "label": "Glaze",        "color": "#1976D2"},   # mid blue (Image-only)
    {"key": "AMP",        "label": "AMP",           "color": "#42A5F5"},   # light blue (Image-only)
    {"key": "XTransfer",  "label": "XTransfer",    "color": "#90CAF9"},   # pale blue (Image-only)
    {"key": "Nightshade", "label": "Nightshade",   "color": "#E64A19"},   # deep orange (Text-only)
    {"key": "MMCoA",      "label": "MMCoA",         "color": "#2E7D32"},   # dark green (Image+Text)
]

# 5 VLMs: Qwen-VL / Kimi 2.5 / GPT-5.2 / Gemini-3.0 / GLM-4V
VLM_ORDER = [
    {"key": "qwen",   "label": "Qwen-VL"},
    {"key": "kimi",   "label": "Kimi 2.5"},
    {"key": "gpt52",  "label": "GPT-5.2"},
    {"key": "gemini", "label": "Gemini-3.0"},
    {"key": "glm4",   "label": "GLM-4V"},
]

# ── colors ────────────────────────────────────────────────────────────────────
C_PROTECTED = "#2E7D32"   # green border
C_FAILED    = "#C62828"   # red border
C_ORIG      = "#78909C"   # gray border for original col

# ── layout ────────────────────────────────────────────────────────────────────
N_COLS = 6;  N_ROWS = 5   # 6 methods (no Original), 5 VLMs
FIG_W = 8.5
# Formula: FIG_H = FIG_W * (1-VLM_W) * N_ROWS / ((1-HEADER_H-LEGEND_H) * N_COLS)
# = 8.5 * 0.905 * 5 / (0.875 * 6) ≈ 7.33  → each cell is a perfect square
FIG_H = FIG_W * 0.905 * 5 / (0.875 * 6)  # ≈ 7.33
HEADER_H    = 0.070
VLM_W       = 0.095
COL_PAD     = 0.0
ROW_PAD     = 0.0
LEGEND_H    = 0.055
BORDER_LW   = 2.6

FONT_METHOD = 12.0
FONT_VLM    = 12
FONT_ANS    = 10

def set_spines(ax, color, lw):
    for sp in ax.spines.values():
        sp.set_edgecolor(color); sp.set_linewidth(lw)
        sp.set_visible(True);    sp.set_capstyle("butt"); sp.set_joinstyle("miter")

def extract_cjk(s):
    s = str(s).strip()
    if not s or s.lower() in ("none", "n/a", "?", ""):
        return "—"
    for ch in s:
        if '\u4e00' <= ch <= '\u9fff':
            return ch
    return s[:2] if s else "—"

# ── build figure ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(FIG_W, FIG_H))

outer = gridspec.GridSpec(
    3, 2, figure=fig,
    left=0.005, right=0.998,
    bottom=0.005, top=0.988,
    hspace=0.0, wspace=0.0,
    height_ratios=[HEADER_H, 1 - HEADER_H - LEGEND_H, LEGEND_H],
    width_ratios=[VLM_W, 1 - VLM_W],
)

# ── header row: method names ───────────────────────────────────────────────────
hdr_gs = gridspec.GridSpecFromSubplotSpec(
    1, N_COLS, subplot_spec=outer[0, 1],
    hspace=0, wspace=0,
)
for ci, m in enumerate(METHODS):
    ax = fig.add_subplot(hdr_gs[0, ci])
    ax.set_axis_off(); ax.patch.set_visible(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.text(0.5, 0.5, m["label"],
            transform=ax.transAxes, ha="center", va="center",
            fontsize=FONT_METHOD, fontweight="bold", color=m["color"])

# ── top-left: "Target 能" ──────────────────────────────────────────────────────
ax_tl = fig.add_subplot(outer[0, 0])
ax_tl.set_axis_off(); ax_tl.patch.set_visible(False)
for sp in ax_tl.spines.values(): sp.set_visible(False)
_tl_prop = cjk_prop(FONT_METHOD + 2.0, bold=True)
ax_tl.text(0.92, 0.50, f"Target {TARGET_CHAR}",
           transform=ax_tl.transAxes, ha="right", va="center",
           fontsize=FONT_METHOD + 2.0, fontweight="bold", color="#212121",
           fontproperties=_tl_prop)

# ── VLM labels ─────────────────────────────────────────────────────────────────
vlm_gs = gridspec.GridSpecFromSubplotSpec(
    N_ROWS, 1, subplot_spec=outer[1, 0],
    hspace=0, wspace=0,
)
for ri, vlm in enumerate(VLM_ORDER):
    ax = fig.add_subplot(vlm_gs[ri, 0])
    ax.set_axis_off(); ax.patch.set_visible(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.text(0.97, 0.50, vlm["label"],
            transform=ax.transAxes, ha="right", va="center",
            fontsize=FONT_VLM, fontweight="bold", color="#212121")

# ── image grid ─────────────────────────────────────────────────────────────────
img_gs = gridspec.GridSpecFromSubplotSpec(
    N_ROWS, N_COLS, subplot_spec=outer[1, 1],
    hspace=0.025, wspace=0.025,
)

for ri, vlm in enumerate(VLM_ORDER):
    for ci, m in enumerate(METHODS):
        ax = fig.add_subplot(img_gs[ri, ci])
        ax.set_xticks([]); ax.set_yticks([])
        ax.tick_params(bottom=False, left=False, top=False, right=False)

        # draw image
        try:
            img = np.array(Image.open(IMG_PATHS[m["key"]]).convert("RGB"))
            ax.imshow(img, aspect="auto", interpolation="lanczos")
        except Exception:
            ax.set_facecolor("#BDBDBD")
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=6)

        correct, ans_raw = RESULTS[m["key"]][vlm["key"]]
        disp = extract_cjk(ans_raw)

        # border
        if correct:
            border_col = C_FAILED
        else:
            border_col = C_PROTECTED
        set_spines(ax, border_col, BORDER_LW)

        # answer tag: unified white text for visibility
        tag_fg = "#FFFFFF"

        tag_text = f"\u2192{disp}"   # →字
        ax.text(
            0.50, 0.04, tag_text,
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=FONT_ANS, fontweight="bold",
            color=tag_fg,
            fontproperties=cjk_prop(FONT_ANS, bold=True),
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="#000000",
                edgecolor="none",
                alpha=0.65,
            ),
            zorder=5,
        )

# ── legend row ─────────────────────────────────────────────────────────────────
leg_gs = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer[2, 1],
    hspace=0, wspace=0.04,
)
for li, (lc, lt) in enumerate([
    (C_PROTECTED, "Protected (VLM confused)"),
    (C_FAILED,    "Failed (VLM correct)"),
]):
    ax_l = fig.add_subplot(leg_gs[0, li])
    ax_l.set_axis_off(); ax_l.patch.set_visible(False)
    for sp in ax_l.spines.values(): sp.set_visible(False)
    rect = plt.Rectangle((0.18, 0.12), 0.08, 0.76,
                          facecolor=lc, edgecolor="none",
                          transform=ax_l.transAxes, clip_on=False, zorder=3)
    ax_l.add_patch(rect)
    ax_l.text(0.31, 0.50, lt,
              transform=ax_l.transAxes, ha="left", va="center",
              fontsize=12, color="#212121")

ax_ll = fig.add_subplot(outer[2, 0])
ax_ll.set_axis_off()

# ── save ───────────────────────────────────────────────────────────────────────
FIG_DIR = HERE.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

out = FIG_DIR / "fig1_teaser.pdf"
fig.savefig(out, format="pdf", bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")

plt.close(fig)
print("Done.")
