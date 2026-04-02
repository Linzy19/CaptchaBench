"""
figmm_fig5_quality.py — ACM MM 2026  (v5 2026-03-27)
Single-panel scatter plot: X=SSIM(↑), Y=ASR(%)
Each method has 2 points (ID-based filled, SDXL hollow) connected by dashed line.
Color logic: modality-unified (same as Fig3).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family":       "serif",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Colors (modality-logic, unified for both generators)
METHOD_COLOR = {
    "AMP":        "#0D47A1",
    "ASPL":       "#1976D2",
    "XTransfer":  "#42A5F5",
    "Glaze":      "#90CAF9",
    "Nightshade": "#E64A19",
    "MMCoA":      "#2E7D32",
}

MARKERS = {
    "AMP":        "o",
    "ASPL":       "s",
    "XTransfer":  "^",
    "Glaze":      "D",
    "Nightshade": "P",
    "MMCoA":      "*",
}

id_data = {
    "AMP":        (0.759, 99.40),
    "ASPL":       (0.789, 99.40),
    "XTransfer":  (0.663, 98.60),
    "Glaze":      (0.527, 97.90),
    "Nightshade": (0.657, 99.20),
    "MMCoA":      (0.770, 99.50),
}
sdxl_data = {
    "AMP":        (0.739, 98.90),
    "ASPL":       (0.775, 98.72),
    "XTransfer":  (0.641, 98.62),
    "Glaze":      (0.508, 98.45),
    "Nightshade": (0.638, 98.95),
    "MMCoA":      (0.753, 98.42),
}

METHODS = ["AMP", "ASPL", "XTransfer", "Glaze", "Nightshade", "MMCoA"]

# ── Label offsets: (dx, dy, ha, position)
# position: "top" = above the higher point, "bottom" = below the lower point, "left" = left of the leftmost point
LABEL_RULES = {
    "AMP":        (-0.008, -0.08, "right",  "bottom"),
    "ASPL":       (+0.008, -0.08, "left",   "bottom"),
    "XTransfer":  ( 0.000, -0.08, "center", "mid_bottom"),
    "Glaze":      (-0.012, -0.06, "right",  "upper_left_below"),
    "Nightshade": (-0.005, +0.06, "right",  "left"),
    "MMCoA":      ( 0.000, -0.08, "center", "bottom"),
}

# ── Figure
fig, ax = plt.subplots(figsize=(7.5, 5.5))
plt.subplots_adjust(left=0.11, right=0.97, bottom=0.12, top=0.91)

for m in METHODS:
    c = METHOD_COLOR[m]
    mk = MARKERS[m]
    sx_id,  sy_id  = id_data[m]
    sx_sdxl, sy_sdxl = sdxl_data[m]
    s_base = 220 if m == "MMCoA" else 160

    # ── Dashed connector line
    ax.plot([sx_id, sx_sdxl], [sy_id, sy_sdxl],
            color=c, lw=1.0, linestyle="--", alpha=0.45, zorder=3)

    # ── Glow ring (MMCoA only)
    glow_s = 450 if m == "MMCoA" else s_base + 290
    for sx, sy in [(sx_id, sy_id), (sx_sdxl, sy_sdxl)]:
        ax.scatter(sx, sy, s=glow_s, marker=mk,
                   facecolors=c, edgecolors="none", alpha=0.12, zorder=4)

    # ── ID-based: filled
    ax.scatter(sx_id, sy_id, s=s_base, marker=mk,
               facecolor=c, edgecolors="white", linewidths=1.2,
               zorder=5)

    # ── SDXL: hollow
    ax.scatter(sx_sdxl, sy_sdxl, s=s_base, marker=mk,
               facecolors="none", edgecolors=c, linewidths=1.8,
               zorder=5)

    # ── Label positioning
    dx, dy, ha, pos = LABEL_RULES[m]
    if pos == "top":
        anchor_y = max(sy_id, sy_sdxl)
        anchor_x = (sx_id + sx_sdxl) / 2
        va = "bottom"
    elif pos == "bottom":
        anchor_y = min(sy_id, sy_sdxl)
        anchor_x = (sx_id + sx_sdxl) / 2
        va = "top"
    elif pos == "mid_bottom":
        anchor_x = (sx_id + sx_sdxl) / 2
        anchor_y = (sy_id + sy_sdxl) / 2
        va = "top"
    elif pos == "upper_left_below":
        # Left-below the upper point (higher Y = upper in scatter)
        if sy_id >= sy_sdxl:
            anchor_x = sx_id
            anchor_y = sy_id
        else:
            anchor_x = sx_sdxl
            anchor_y = sy_sdxl
        va = "top"
    else:  # "left"
        anchor_x = min(sx_id, sx_sdxl)
        anchor_y = (sy_id + sy_sdxl) / 2
        va = "center"
    ax.annotate(m,
                xy=(anchor_x + dx, anchor_y + dy),
                xytext=(0, 0),
                textcoords="offset points",
                ha=ha, va=va,
                fontsize=9.5, fontfamily="serif", color=c,
                annotation_clip=False)

# ── Diagonal efficiency trend line (gray dashed, behind everything)
trend_x = np.array([0.42, 0.85])
trend_y = np.array([97.5, 99.8])
ax.plot(trend_x, trend_y,
        linestyle="--", color="#AAAAAA", linewidth=1.2,
        alpha=0.7, zorder=1, label="_nolegend_")

# ── "Ideal Region" fancy box (upper-right, same style as Fig3)
# X >= 0.75, Y >= 99.0, extends to axis limits (0.85, 99.8)
ideal_rect = mpatches.FancyBboxPatch(
    (0.75, 99.0), 0.10, 0.80,          # (x0, y0), width, height — extends beyond ylim
    boxstyle="round,pad=0,rounding_size=0.008",
    linewidth=1.0, linestyle="--",
    edgecolor="#A5D6A7", facecolor="#E8F5E9",
    alpha=0.55, zorder=2,
    clip_on=True,
)
ax.add_patch(ideal_rect)
ax.text(
    0.80, 99.72,
    "Ideal Region\n(High Quality & Effective)",
    ha="center", va="top",
    fontsize=7.5, fontfamily="serif",
    color="#2E7D32", fontstyle="italic", fontweight="bold", zorder=3,
)

# ── Axes formatting
ax.set_xlim(0.42, 0.85)
ax.set_ylim(97.5, 99.8)
ax.set_xlabel(r"SSIM $\rightarrow$ Higher is Better",
              fontsize=11, fontfamily="serif", labelpad=6)
ax.set_ylabel(r"ASR (%) $\rightarrow$ Higher is Better",
              fontsize=11, fontfamily="serif", labelpad=6)
ax.tick_params(axis="both", labelsize=10)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontfamily("serif")
ax.set_title("SSIM vs. ASR",
             fontsize=11, fontfamily="serif", fontweight="bold")
ax.grid(True, which="major", linestyle="--", linewidth=0.48, alpha=0.42, color="gray")

import matplotlib.lines as mlines
import matplotlib.ticker as mticker

ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(5))
ax.grid(True, which="minor", linestyle=":", linewidth=0.25, alpha=0.2, color="gray")
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)
ax.spines["left"].set_color("#555555")
ax.spines["bottom"].set_color("#555555")

# ── Legend: 两列并排（列优先），与 Fig3 完全一致
#   左列: "Method"标题 + 6方法色块
#   右列: "Generator"标题 + ID + SDXL + 空行×4
m_title = mlines.Line2D([], [], color="none", label="Method")
method_handles = [m_title] + [
    mpatches.Patch(facecolor=METHOD_COLOR[m], label=m)
    for m in METHODS
]

g_title  = mlines.Line2D([], [], color="none", label="Generator")
filled_h = mlines.Line2D([], [], marker="o", linestyle="None",
    markersize=7, markerfacecolor="#555555", markeredgecolor="white",
    markeredgewidth=0.9, label="ID")
hollow_h = mlines.Line2D([], [], marker="o", linestyle="None",
    markersize=7, markerfacecolor="none", markeredgecolor="#555555",
    markeredgewidth=1.5, label="SD")
spacer = mlines.Line2D([], [], color="none", label=" ")
gen_handles = [g_title, filled_h, hollow_h, spacer, spacer, spacer, spacer]

# 列优先：左列7 + 右列7 = 14个 handle，ncol=2 自动填满两列
col_ordered = method_handles + gen_handles

leg = ax.legend(
    handles=col_ordered,
    ncol=2,
    loc="upper left",
    bbox_to_anchor=(0.01, 0.99),
    fontsize=8.5, framealpha=0.92, edgecolor="#AAAAAA",
    prop={"family": "serif", "size": 8.5},
    handlelength=1.4, handletextpad=0.5,
    labelspacing=0.38, columnspacing=1.0, borderpad=0.7,
)
for text in leg.get_texts():
    if text.get_text() in ("Method", "Generator"):
        text.set_fontweight("bold")
        text.set_color("#222222")

# ── Save
HERE    = Path(__file__).parent
RES_DIR = HERE.parent.parent / "results"
FIG_DIR = HERE.parent / "figures"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

for d, name in [(RES_DIR, "figmm_fig5_quality"), (FIG_DIR, "fig4_pareto_efficiency")]:
    fig.savefig(d / f"{name}.pdf", dpi=300, bbox_inches="tight")
    print(f"Saved -> {d / name}.pdf")

plt.close(fig)
