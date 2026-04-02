"""
figmm_fig6a_stroke_line.py — ACM MM 2026
Panel A standalone: ASR vs Stroke Complexity (line chart)
Extracted from figmm_fig6_stroke.py (v3 2026-03-27)
Output: figures/fig6a_stroke_line.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family":       "serif",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Method styling (identical to combined script) ─────────────────────────────
METHOD_COLOR = {
    "AMP":        "#0D47A1",   # deep blue   (Image-only)
    "ASPL":       "#1976D2",   # mid blue    (Image-only)
    "XTransfer":  "#42A5F5",   # light blue  (Image-only)
    "Glaze":      "#90CAF9",   # pale blue   (Image-only)
    "Nightshade": "#E64A19",   # deep orange (Text-only)
    "MMCoA":      "#2E7D32",   # dark green  (Image+Text)
}
MODALITY = {
    "ASPL":       "Image-only",
    "Glaze":      "Image-only",
    "AMP":        "Image-only",
    "XTransfer":  "Image-only",
    "Nightshade": "Text-only",
    "MMCoA":      "Image+Text",
}
LINESTYLES = {
    "ASPL":       "-",
    "Glaze":      "-",
    "AMP":        "-",
    "XTransfer":  "-",
    "Nightshade": "-",
    "MMCoA":      "-",
}
MARKERS = {
    "ASPL":       "o",
    "Glaze":      "s",
    "AMP":        "^",
    "XTransfer":  "X",
    "Nightshade": "P",
    "MMCoA":      "D",
}

# ── Data (SD-ID, identical to combined script) ────────────────────────────────
bins = ["1–5", "6–10", "11–15", "16–20", "21+"]
stroke_data = {
    "ASPL":       [98.2, 99.4, 99.4, 100.0,  95.2],
    "Glaze":      [95.3, 97.9, 97.9,  99.3,  95.2],
    "AMP":        [99.3, 99.4, 99.3, 100.0, 100.0],
    "XTransfer":  [98.6, 98.6, 98.4,  98.6,  95.2],
    "Nightshade": [98.2, 99.1, 98.9,  98.6, 100.0],
    "MMCoA":      [99.3, 99.7, 99.3, 100.0, 100.0],
}
METHODS = ["ASPL", "Glaze", "AMP", "XTransfer", "Nightshade", "MMCoA"]

# ── Figure — Panel A standalone ───────────────────────────────────────────────
fig, ax_a = plt.subplots(figsize=(7.5, 5.5))
fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.13)

x = np.arange(len(bins))

# Image-only fill band (same as combined)
img_methods = [m for m in METHODS if MODALITY[m] == "Image-only"]
mod_min = [min(stroke_data[m][i] for m in img_methods) for i in range(len(bins))]
mod_max = [max(stroke_data[m][i] for m in img_methods) for i in range(len(bins))]
ax_a.fill_between(x, mod_min, mod_max, alpha=0.08, color="#1976D2", zorder=1,
                  label="_nolegend_")

# Line plots
for m in METHODS:
    ax_a.plot(x, stroke_data[m],
              marker=MARKERS[m], color=METHOD_COLOR[m],
              linestyle=LINESTYLES[m],
              linewidth=2.0, markersize=7, label=m, zorder=4)

ax_a.set_xticks(x)
ax_a.set_xticklabels(bins, fontsize=10.5, fontfamily="serif")
ax_a.set_xlim(-0.4, len(bins) - 0.6)
ax_a.set_ylim(93.5, 101.8)
ax_a.set_yticks(np.arange(94, 102, 1))
ax_a.tick_params(axis="y", labelsize=10.5)
for lbl in ax_a.get_xticklabels() + ax_a.get_yticklabels():
    lbl.set_fontfamily("serif")
ax_a.set_xlabel("Stroke-Count Bin", fontsize=11, fontfamily="serif")
ax_a.set_ylabel("ASR (%)", fontsize=11, fontfamily="serif")
ax_a.set_title("(A) ASR vs. Stroke Complexity",
               fontsize=11.5, fontfamily="serif", fontweight="bold", pad=9)
ax_a.grid(True, linestyle="--", linewidth=0.48, alpha=0.42, color="gray")

# Legend
method_lines = [
    plt.Line2D([0], [0], color=METHOD_COLOR[m], linestyle=LINESTYLES[m],
               marker=MARKERS[m], markersize=4.5, linewidth=1.4, label=m)
    for m in METHODS
]
ax_a.legend(handles=method_lines,
            title="Method", title_fontsize=7.5,
            fontsize=7.2, loc="lower right",
            framealpha=0.92, edgecolor="#CCCCCC",
            prop={"family": "serif"}, ncol=2,
            handlelength=1.4, handletextpad=0.4,
            columnspacing=0.8, labelspacing=0.3,
            borderpad=0.4)
ax_a.get_legend().get_title().set_fontfamily("serif")
ax_a.get_legend().get_title().set_fontweight("bold")

# Modality annotation (identical position as combined)
# ax_a.annotate("── Image-only  ── Text-only  ── Img+Txt",
#               xy=(0.99, 0.01), xycoords="axes fraction",
#               ha="right", va="bottom",
#               fontsize=7.5, fontfamily="serif", color="#555555",
#               style="italic")

# ── Save ──────────────────────────────────────────────────────────────────────
FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

out = FIG_DIR / "fig6a_stroke_line.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved -> {out}")

plt.close(fig)
