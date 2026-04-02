"""
figmm_fig4_pareto.py  (renamed from figmm_fig3_pareto.py)
--------------------
ACM MM 2026 — Figure 4: Quality–Effectiveness Pareto Frontier
Output: figures/fig4_pareto_efficiency.pdf
X-axis: SSIM (perturbation imperceptibility), higher=better.
Y-axis: ASR (%).
Each method appears twice (filled = ID model, hollow = SDXL model), connected
by a thin dashed vertical line.  Labels placed ABOVE the midpoint of each pair.
MMCoA uses star marker but same dual-point ID+SDXL logic as all other methods.
"Ideal Region" fancy box in upper-right.  Diagonal efficiency trend line.
Per-method color legend + Generator section at lower-right.
"""

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"]       = "serif"
matplotlib.rcParams["pdf.fonttype"]      = 42
matplotlib.rcParams["ps.fonttype"]       = 42
matplotlib.rcParams["axes.spines.top"]   = False
matplotlib.rcParams["axes.spines.right"] = False

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines   as mlines
import matplotlib.ticker  as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent                          # …/main/scripts/
RESULTS_DIR = SCRIPT_DIR.parent.parent / "results"           # ACM_MM26/results/
FIGURES_DIR = SCRIPT_DIR.parent / "figures"                  # ACM_MM26/main/figures/
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Font sizes
# ---------------------------------------------------------------------------
FONT_LABEL = 8.5
FONT_TICK  = 9
FONT_AXIS  = 10
FONT_TITLE = 11

# ---------------------------------------------------------------------------
# Palette & markers
# ---------------------------------------------------------------------------
MODALITY_COLOR = {
    "Image-only": "#1976D2",   # mid blue (legend representative)
    "Text-only":  "#E64A19",   # deep orange
    "Image+Text": "#2E7D32",   # dark green
}

METHOD_COLOR = {
    "AMP":        "#0D47A1",   # deep blue      (Image-only)
    "ASPL":       "#1976D2",   # mid blue       (Image-only)
    "XTransfer":  "#42A5F5",   # light blue     (Image-only)
    "Glaze":      "#90CAF9",   # pale blue      (Image-only)
    "Nightshade": "#E64A19",   # deep orange    (Text-only)
    "MMCoA":      "#2E7D32",   # dark green     (Image+Text)
}

METHOD_MARKER = {
    "AMP":        "^",   # triangle up
    "ASPL":       "s",   # square
    "XTransfer":  "X",   # x filled
    "Glaze":      "D",   # diamond
    "Nightshade": "P",   # plus filled
    "MMCoA":      "*",   # star
}

MARKER_SIZE = 160   # uniform size for all markers

# ---------------------------------------------------------------------------
# Data  (name, time_s, asr_id, asr_sdxl, modality)
# Throughput = 60 / time_s  (img/min)
# ---------------------------------------------------------------------------
DATA = [
    ("MMCoA",       2.6,  99.13, 98.73, "Image+Text"),
    ("XTransfer",   5.0,  98.73, 98.90, "Image-only"),
    ("ASPL",        9.0,  99.08, 98.77, "Image-only"),
    ("AMP",        10.4,  99.15, 98.90, "Image-only"),
    ("Glaze",      20.3,  98.42, 98.90, "Image-only"),
    ("Nightshade", 96.8,  98.98, 98.77, "Text-only"),
]

# Label offsets in data units (dx, dy, ha, va)
# Change 1: Nightshade label placed to the right of its point (ha="left")
LABEL_OFFSETS = {
    "MMCoA":      (0, +0.07, "center", "bottom"),
    "XTransfer":  (0, +0.07, "center", "bottom"),
    "ASPL":       (0, +0.07, "center", "bottom"),
    "AMP":        (0, +0.07, "center", "bottom"),
    "Glaze":      (0, +0.07, "center", "bottom"),
    "Nightshade": (0, +0.07, "left",   "bottom"),
}

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5.4))
plt.subplots_adjust(left=0.11, right=0.97, bottom=0.14, top=0.90)

# Grid
ax.set_axisbelow(True)
ax.grid(True, which="major", linestyle="--", color="#CCCCCC",
        linewidth=0.6, alpha=0.8)

# Linear x-axis (throughput)
ax.set_xscale("linear")

# ---------------------------------------------------------------------------
# Diagonal efficiency trend line  (gray dashed, behind everything)
# ---------------------------------------------------------------------------
trend_x = np.array([0.0, 25.5])
trend_y = np.array([98.0, 99.5])
ax.plot(trend_x, trend_y,
        linestyle="--", color="#AAAAAA", linewidth=1.2,
        alpha=0.7, zorder=1, label="_nolegend_")

# ---------------------------------------------------------------------------
# "Ideal Region" fancy box  (upper-right, 参考 ICANN 版本)
# ---------------------------------------------------------------------------
# Open interval: X > 20, Y > 99.0  (extends beyond axes = "infinite")
# Use a large rectangle with clip_on=False so it visually extends past the axes
ideal_rect = mpatches.FancyBboxPatch(
    (20, 99.0), 10.0, 2.0,
    boxstyle="round,pad=0,rounding_size=0.12",
    linewidth=1.0, linestyle="--",
    edgecolor="#A5D6A7", facecolor="#E8F5E9",
    alpha=0.55, zorder=2,
    clip_on=True,          # clip to axes area so it doesn't bleed into margins
)
ax.add_patch(ideal_rect)
ax.text(
    23.0, 99.30,
    "Ideal Region\n(Fast & Effective)",
    ha="center", va="center",
    fontsize=7.5, fontfamily="serif",
    color="#2E7D32", fontstyle="italic", fontweight="bold", zorder=3,
)

# ---------------------------------------------------------------------------
# Plot each method  (all methods use dual ID + SDXL points with connector)
# MMCoA keeps its star marker but follows the same logic as other methods.
# ---------------------------------------------------------------------------
for (name, time_s, asr_id, asr_sdxl, modality) in DATA:
    color  = METHOD_COLOR[name]
    marker = METHOD_MARKER[name]
    x      = 60.0 / time_s       # throughput in img/min
    y_id   = asr_id
    y_sdxl = asr_sdxl

    # Connecting dashed vertical line (behind markers)
    ax.plot(
        [x, x], [min(y_id, y_sdxl), max(y_id, y_sdxl)],
        linestyle="--", color=color, alpha=0.45, linewidth=1.0, zorder=3,
    )

    # Glow ring beneath filled marker (ID)
    ax.scatter(
        x, y_id,
        marker=marker, s=MARKER_SIZE * 3.5,
        color=color, alpha=0.18, edgecolors="none",
        zorder=3,
    )

    # Glow ring beneath hollow marker (SDXL)
    ax.scatter(
        x, y_sdxl,
        marker=marker, s=MARKER_SIZE * 3.5,
        color=color, alpha=0.18, edgecolors="none",
        zorder=3,
    )

    # Filled marker → ID model
    ax.scatter(
        x, y_id,
        marker=marker, s=MARKER_SIZE,
        color=color, edgecolors="white", linewidths=1.2,
        zorder=5,
    )

    # Hollow marker → SDXL model
    ax.scatter(
        x, y_sdxl,
        marker=marker, s=MARKER_SIZE,
        facecolors="none", edgecolors=color, linewidths=1.8,
        zorder=5,
    )

    # Label: above the top of the pair + dy offset
    top_y = max(y_id, y_sdxl)
    dx, dy, ha, va = LABEL_OFFSETS[name]
    ax.annotate(
        name,
        xy=(x + dx, top_y + dy),
        xytext=(0, 0),
        textcoords="offset points",
        ha=ha, va=va,
        fontsize=FONT_LABEL,
        fontfamily="serif",
        color=color,
        zorder=6,
    )

# ---------------------------------------------------------------------------
# Axis formatting
# ---------------------------------------------------------------------------
ax.set_xlabel(r"Throughput (img/min) $\rightarrow$ Higher is Better",
              fontsize=FONT_AXIS, fontfamily="serif", labelpad=6)
ax.set_ylabel(r"ASR (%) $\rightarrow$ Higher is Better",
              fontsize=FONT_AXIS, fontfamily="serif", labelpad=6)
ax.set_title(
    "Throughput  vs.  ASR",
    fontsize=FONT_TITLE, fontfamily="serif",
    fontweight="bold", pad=8,
)

ax.tick_params(axis="both", labelsize=FONT_TICK)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontfamily("serif")

ax.set_ylim(98.0, 99.5)
ax.set_xlim(-0.5, 26.0)
ax.set_xticks([0, 5, 10, 15, 20, 25])
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_yticks(np.arange(98.0, 99.55, 0.5))

# Minor ticks
import matplotlib.ticker as mticker
ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(5))
ax.grid(True, which="minor", linestyle=":", linewidth=0.25, alpha=0.2, color="gray")

# Spine
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)
ax.spines["left"].set_color("#555555")
ax.spines["bottom"].set_color("#555555")

# ---------------------------------------------------------------------------
# Legend: 两列并排
#   左列: 列标题"Method"(不可见占位) + 6个方法色块
#   右列: 列标题"Generator"(不可见占位) + 实心"ID" + 空心"SDXL" + 空行×4
#
# 技巧: 用 color="none" 的 Line2D 做加粗标题行,
#       通过 get_texts() 事后把标题行字体改成 bold
# ---------------------------------------------------------------------------
# 左列 header + 6方法
m_title  = mlines.Line2D([], [], color="none", label="Method")
method_handles = [m_title] + [
    mpatches.Patch(facecolor=METHOD_COLOR[name], label=name)
    for name, *_ in DATA
]

# 右列 header + ID + SDXL + 空行×4
g_title  = mlines.Line2D([], [], color="none", label="Generator")
filled_h = mlines.Line2D(
    [], [], marker="o", linestyle="None",
    markersize=7, markerfacecolor="#555555", markeredgecolor="white",
    markeredgewidth=0.9, label="ID",
)
hollow_h = mlines.Line2D(
    [], [], marker="o", linestyle="None",
    markersize=7, markerfacecolor="none", markeredgecolor="#555555",
    markeredgewidth=1.5, label="SD",
)
spacer = mlines.Line2D([], [], color="none", label=" ")
gen_handles = [g_title, filled_h, hollow_h, spacer, spacer, spacer, spacer]

# ncol=2 时 matplotlib 按【列优先】填充：先填满左列再填右列
# 所以直接把左列7个 + 右列7个 顺序拼接即可
col_ordered = method_handles + gen_handles   # 左列7 + 右列7 = 14

leg = ax.legend(
    handles=col_ordered,
    ncol=2,
    loc="lower right",
    bbox_to_anchor=(1.0, 0.0),
    fontsize=FONT_LABEL,
    frameon=True, framealpha=0.92, edgecolor="#AAAAAA",
    prop={"family": "serif", "size": FONT_LABEL},
    handlelength=1.4,
    handletextpad=0.5,
    labelspacing=0.38,
    columnspacing=1.0,
    borderpad=0.7,
)

# 把 "Method" 和 "Generator" 两个标题行设为 bold
for text in leg.get_texts():
    if text.get_text() in ("Method", "Generator"):
        text.set_fontweight("bold")
        text.set_color("#222222")

# ---------------------------------------------------------------------------
# Change 4: Save PDF only (no PNG)
# ---------------------------------------------------------------------------
stem      = "figmm_fig3_pareto"
copy_stem = "fig3_pareto_efficiency"

out_pdf = RESULTS_DIR / f"{stem}.pdf"
fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
print(f"Saved → {out_pdf}")

copy_pdf = FIGURES_DIR / f"{copy_stem}.pdf"
shutil.copy2(out_pdf, copy_pdf)
print(f"Copied → {copy_pdf}")

plt.close(fig)
