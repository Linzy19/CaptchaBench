"""
figmm_fig6b_stroke_heatmap.py — ACM MM 2026
Panel B standalone: ASR Heatmap by Stroke Complexity
Extracted from figmm_fig6_stroke.py (v3 2026-03-27)
Output: figures/fig6b_stroke_heatmap.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
# Heatmap row order: Image-only(4) → Text-only(1) → Image+Text(1)
METHOD_ORDER = ["AMP", "ASPL", "XTransfer", "Glaze", "Nightshade", "MMCoA"]

# ── Figure — Panel B standalone ───────────────────────────────────────────────
fig, ax_b = plt.subplots(figsize=(7.0, 5.5))
fig.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.13)

# Restore top/right spines for the heatmap (same as combined)
ax_b.spines["top"].set_visible(True)
ax_b.spines["right"].set_visible(True)

heatmap_data = np.array([stroke_data[m] for m in METHOD_ORDER])

# Custom colormap (identical to combined script)
cmap_custom = LinearSegmentedColormap.from_list(
    "asr_cmap",
    [
        (0.00, "#F0C080"),   # 95.0 → warm orange (low)
        (0.25, "#FAF3E0"),   # 97.4 → pale cream  (transition)
        (0.50, "#DDEEFF"),   # 98.8 → pale blue
        (0.75, "#7BAFD4"),   # 99.6 → mid blue
        (1.00, "#2C5F8A"),   # 100.5 → deep blue  (high)
    ],
    N=256
)
VMIN, VMAX = 95.0, 100.5

im = ax_b.imshow(heatmap_data, aspect="auto", cmap=cmap_custom,
                 vmin=VMIN, vmax=VMAX, origin="upper", alpha=0.92)

# White grid lines
for r in range(len(METHOD_ORDER) + 1):
    ax_b.axhline(r - 0.5, color="white", linewidth=2.0, zorder=3)
for c in range(len(bins) + 1):
    ax_b.axvline(c - 0.5, color="white", linewidth=2.0, zorder=3)

# In-cell value annotations
for r in range(len(METHOD_ORDER)):
    for c_idx in range(len(bins)):
        val = heatmap_data[r, c_idx]
        norm_val = (val - VMIN) / (VMAX - VMIN)
        txt_color = "white" if norm_val > 0.62 else "#1A1A2E"
        weight = "bold" if val < 96.5 else "normal"
        ax_b.text(c_idx, r, f"{val:.1f}",
                  ha="center", va="center",
                  fontsize=9.5, fontfamily="serif",
                  color=txt_color, fontweight=weight,
                  zorder=5)

ax_b.set_xlim(-0.5, len(bins) - 0.5)
ax_b.set_ylim(len(METHOD_ORDER) - 0.5, -0.5)

ax_b.set_xticks(np.arange(len(bins)))
ax_b.set_xticklabels(bins, fontsize=10.5, fontfamily="serif")
ax_b.set_yticks(np.arange(len(METHOD_ORDER)))
ax_b.set_yticklabels(METHOD_ORDER, fontsize=10.5, fontfamily="serif")
for lbl, m in zip(ax_b.get_yticklabels(), METHOD_ORDER):
    lbl.set_color(METHOD_COLOR[m])
    lbl.set_fontweight("bold")
ax_b.set_xlabel("Stroke-Count Bin", fontsize=11, fontfamily="serif")
ax_b.set_title("(B) ASR Heatmap by Stroke Complexity",
               fontsize=11.5, fontfamily="serif", fontweight="bold", pad=9)

# Colorbar (right side, identical settings)
divider = make_axes_locatable(ax_b)
cax = divider.append_axes("right", size="4%", pad=0.08)
cb = plt.colorbar(im, cax=cax)
cb.set_label("ASR (%)", fontsize=9, fontfamily="serif")
cb.ax.tick_params(labelsize=8)
cb.set_ticks([95, 96, 97, 98, 99, 100])
for tick_lbl in cb.ax.get_yticklabels():
    tick_lbl.set_fontfamily("serif")

# ── Save ──────────────────────────────────────────────────────────────────────
FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

out = FIG_DIR / "fig6b_stroke_heatmap.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved -> {out}")

plt.close(fig)
