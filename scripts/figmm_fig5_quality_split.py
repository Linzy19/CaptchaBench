"""
figmm_fig5_quality_split.py — ACM MM 2026  (v1 2026-03-30)
Generates two separate scatter plots split from figmm_fig5_quality.py:
  • fig5_quality_ID.pdf   — ID-based targets only  (filled markers)
  • fig5_quality_SDXL.pdf — SDXL targets only      (hollow markers)
No connector lines. Each plot is self-contained and ready for single-column use.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# ── Global rcParams (serif font, PDF Type-42 embedding)
plt.rcParams.update({
    "font.family":       "serif",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Shared color / marker dictionaries (identical to original script)
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

METHODS = ["AMP", "ASPL", "XTransfer", "Glaze", "Nightshade", "MMCoA"]

# ── Data
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

# ── Per-method annotation offsets for each plot
#    Tuple: (dx, dy, ha, va)
ID_LABEL_OFFSETS = {
    "AMP":        (-0.010, -0.08, "right",  "top"),
    "ASPL":       (+0.010, -0.08, "left",   "top"),
    "XTransfer":  ( 0.000, -0.09, "center", "top"),
    "Glaze":      (-0.012, -0.07, "right",  "top"),
    "Nightshade": (-0.012, +0.06, "right",  "bottom"),
    "MMCoA":      ( 0.000, -0.08, "center", "top"),
}

SDXL_LABEL_OFFSETS = {
    "AMP":        (-0.010, -0.08, "right",  "top"),
    "ASPL":       (+0.010, -0.08, "left",   "top"),
    "XTransfer":  ( 0.000, -0.09, "center", "top"),
    "Glaze":      (-0.012, -0.07, "right",  "top"),
    "Nightshade": (-0.010, +0.06, "right",  "bottom"),
    "MMCoA":      ( 0.000, +0.07, "center", "bottom"),
}

# ── Output directories
HERE    = Path(__file__).parent
FIG_DIR = HERE.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper: shared axis decorations
# ═══════════════════════════════════════════════════════════════════════════════
def _decorate_axes(ax, title: str) -> None:
    """Apply axis limits, labels, grid, ticks, and title."""
    ax.set_xlim(0.42, 0.85)
    ax.set_ylim(97.5, 99.8)

    ax.set_xlabel(r"SSIM $\rightarrow$ Higher is Better",
                  fontsize=11, fontfamily="serif", labelpad=6)
    ax.set_ylabel(r"ASR (%) $\rightarrow$ Higher is Better",
                  fontsize=11, fontfamily="serif", labelpad=6)
    ax.set_title(title, fontsize=11, fontfamily="serif", fontweight="bold")

    ax.tick_params(axis="both", labelsize=10)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("serif")

    # Major grid
    ax.grid(True, which="major", linestyle="--",
            linewidth=0.48, alpha=0.42, color="gray")
    # Minor ticks + grid
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax.grid(True, which="minor", linestyle=":",
            linewidth=0.25, alpha=0.20, color="gray")

    # Spine style
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color("#555555")


def _add_ideal_region(ax) -> None:
    """Draw the 'Ideal Region' FancyBboxPatch and its label."""
    ideal_rect = mpatches.FancyBboxPatch(
        (0.75, 99.0), 0.10, 0.80,
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


def _add_trend_line(ax) -> None:
    """Draw the diagonal efficiency trend line (gray dashed)."""
    trend_x = np.array([0.42, 0.85])
    trend_y = np.array([97.5, 99.8])
    ax.plot(trend_x, trend_y,
            linestyle="--", color="#AAAAAA", linewidth=1.2,
            alpha=0.7, zorder=1, label="_nolegend_")


def _add_legend(ax) -> None:
    """
    Single-column legend showing only method colour patches.
    No 'Generator' section needed since each plot has one generator type.
    """
    m_title = mlines.Line2D([], [], color="none", label="Method")
    method_handles = [m_title] + [
        mpatches.Patch(facecolor=METHOD_COLOR[m], label=m)
        for m in METHODS
    ]
    leg = ax.legend(
        handles=method_handles,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        fontsize=8.5, framealpha=0.92, edgecolor="#AAAAAA",
        prop={"family": "serif", "size": 8.5},
        handlelength=1.4, handletextpad=0.5,
        labelspacing=0.38, columnspacing=1.0, borderpad=0.7,
    )
    for text in leg.get_texts():
        if text.get_text() == "Method":
            text.set_fontweight("bold")
            text.set_color("#222222")


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 1 — ID-based targets (filled markers, white edge)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_id() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    plt.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.91)

    _add_trend_line(ax)
    _add_ideal_region(ax)

    for m in METHODS:
        c   = METHOD_COLOR[m]
        mk  = MARKERS[m]
        sx, sy = id_data[m]
        s_base = 220 if m == "MMCoA" else 160

        # Glow ring
        glow_s = 450 if m == "MMCoA" else s_base + 290
        ax.scatter(sx, sy, s=glow_s, marker=mk,
                   facecolors=c, edgecolors="none", alpha=0.12, zorder=4)

        # Filled marker — ID style
        ax.scatter(sx, sy, s=s_base, marker=mk,
                   facecolor=c, edgecolors="white", linewidths=1.2,
                   zorder=5)

        # Annotation
        dx, dy, ha, va = ID_LABEL_OFFSETS[m]
        ax.annotate(
            m,
            xy=(sx + dx, sy + dy),
            xytext=(0, 0),
            textcoords="offset points",
            ha=ha, va=va,
            fontsize=9.5, fontfamily="serif", color=c,
            annotation_clip=False,
        )

    _decorate_axes(ax, "SSIM vs. ASR (ID-based Targets)")
    _add_legend(ax)

    out = FIG_DIR / "fig5_quality_ID.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 2 — SDXL targets (hollow markers, colored edge)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_sdxl() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    plt.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.91)

    _add_trend_line(ax)
    _add_ideal_region(ax)

    for m in METHODS:
        c   = METHOD_COLOR[m]
        mk  = MARKERS[m]
        sx, sy = sdxl_data[m]
        s_base = 220 if m == "MMCoA" else 160

        # Glow ring
        glow_s = 450 if m == "MMCoA" else s_base + 290
        ax.scatter(sx, sy, s=glow_s, marker=mk,
                   facecolors=c, edgecolors="none", alpha=0.12, zorder=4)

        # Hollow marker — SDXL style
        ax.scatter(sx, sy, s=s_base, marker=mk,
                   facecolors="none", edgecolors=c, linewidths=1.8,
                   zorder=5)

        # Annotation
        dx, dy, ha, va = SDXL_LABEL_OFFSETS[m]
        ax.annotate(
            m,
            xy=(sx + dx, sy + dy),
            xytext=(0, 0),
            textcoords="offset points",
            ha=ha, va=va,
            fontsize=9.5, fontfamily="serif", color=c,
            annotation_clip=False,
        )

    _decorate_axes(ax, "SSIM vs. ASR (SDXL Targets)")
    _add_legend(ax)

    out = FIG_DIR / "fig5_quality_SDXL.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    plot_id()
    plot_sdxl()
    print("Done — both PDFs generated successfully.")
