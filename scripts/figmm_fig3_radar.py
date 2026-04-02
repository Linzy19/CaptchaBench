"""
figmm_fig3_radar.py  (renamed from figmm_fig7_radar_cb_vision.py)
==============================
ACM MM 2026 — CaptchaShield Modality Radar Charts
Outputs: fig3_modality_radar.pdf (main text), fig7_radar_appendix.pdf (appendix)

Colorblind-friendly & visually enhanced version of the radar chart.

Key improvements over v1:
  - Colorblind-safe palette (Wong / Tableau-CB10 inspired)
  - Distinct markers for maximum visual separation
  - Refined grid: dotted style, softer color, repositioned level labels
  - All methods treated equally (no single method is "ours";
    this paper proposes a dataset/benchmark, not a method)
  - Tighter legend with marker-line handles
  - Gemini anomaly: dedicated annotation box instead of subtitle
  - Consistent serif font, PDF Type-42 embedding
  - Output files suffixed with _cb_vision
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines   as mlines
import numpy as np
from pathlib import Path

HERE    = Path(__file__).parent
OUT_DIR = HERE.parent.parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = HERE.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":  "serif",
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})

# ── Colorblind-safe palette (Wong 2011 + Tableau CB10 hybrid) ─────────────────
# Each color is distinguishable under protanopia, deuteranopia, tritanopia
METHOD_COLOR = {
    "AMP":        "#0072B2",   # blue
    "ASPL":       "#56B4E9",   # sky blue
    "XTransfer":  "#E69F00",   # orange
    "Glaze":      "#CC79A7",   # reddish purple / pink
    "Nightshade": "#D55E00",   # vermillion
    "MMCoA":      "#009E73",   # bluish green
}

# Distinct markers — chosen for maximum visual separation
MARKERS = {
    "AMP":        "^",   # triangle up
    "ASPL":       "s",   # square
    "XTransfer":  "D",   # diamond
    "Glaze":      "o",   # circle
    "Nightshade": "P",   # plus (filled)
    "MMCoA":      "*",   # star
}

LINEWIDTHS   = {m: 1.2 for m in METHOD_COLOR}
FILL_ALPHA   = {m: 0.10 for m in METHOD_COLOR}
MARKER_SIZE  = {m: 6 for m in METHOD_COLOR}
ZORDER_LINE  = {m: 4 for m in METHOD_COLOR}
ZORDER_FILL  = {m: 2.5 for m in METHOD_COLOR}

METHODS = ["AMP", "ASPL", "XTransfer", "Glaze", "Nightshade", "MMCoA"]

# ── Data (SD-ID, adv_metrics) ────────────────────────────────────────────────
PER_VLM_DATA = {
    "Qwen-VL": {
        "AMP":        {"CR":  0.4, "1-TVR": 94.9, "ASR":  99.7},
        "ASPL":       {"CR":  1.4, "1-TVR": 92.4, "ASR":  99.0},
        "XTransfer":  {"CR":  4.1, "1-TVR": 84.9, "ASR":  95.9},
        "Glaze":      {"CR":  5.5, "1-TVR": 84.9, "ASR":  94.5},
        "Nightshade": {"CR":  1.0, "1-TVR": 94.1, "ASR":  99.2},
        "MMCoA":      {"CR":  2.2, "1-TVR": 90.8, "ASR":  98.0},
    },
    "Kimi 2.5": {
        "AMP":        {"CR":  0.8, "1-TVR": 94.1, "ASR":  99.7},
        "ASPL":       {"CR":  1.5, "1-TVR": 91.5, "ASR":  99.0},
        "XTransfer":  {"CR":  2.3, "1-TVR": 80.3, "ASR":  98.7},
        "Glaze":      {"CR":  2.1, "1-TVR": 79.2, "ASR":  98.0},
        "Nightshade": {"CR":  1.0, "1-TVR": 91.7, "ASR":  99.7},
        "MMCoA":      {"CR":  2.1, "1-TVR": 86.5, "ASR":  98.6},
    },
    "GPT-5.2": {
        "AMP":        {"CR":  0.2, "1-TVR": 96.3, "ASR":  99.9},
        "ASPL":       {"CR":  0.3, "1-TVR": 92.2, "ASR":  99.8},
        "XTransfer":  {"CR":  0.2, "1-TVR": 89.3, "ASR":  99.7},
        "Glaze":      {"CR":  0.6, "1-TVR": 87.1, "ASR":  99.7},
        "Nightshade": {"CR":  0.4, "1-TVR": 89.6, "ASR": 100.0},
        "MMCoA":      {"CR":  0.1, "1-TVR": 96.8, "ASR":  99.9},
    },
    "Gemini-3.0": {
        "AMP":        {"CR":  1.1, "1-TVR":  3.0, "ASR":  98.8},
        "ASPL":       {"CR":  1.4, "1-TVR":  4.2, "ASR":  98.8},
        "XTransfer":  {"CR":  2.2, "1-TVR":  4.6, "ASR":  98.1},
        "Glaze":      {"CR":  4.5, "1-TVR":  5.7, "ASR":  95.5},
        "Nightshade": {"CR":  2.2, "1-TVR":  4.3, "ASR":  98.2},
        "MMCoA":      {"CR":  0.6, "1-TVR":  2.7, "ASR":  99.4},
    },
    "GLM-4V": {
        "AMP":        {"CR":  0.5, "1-TVR": 100.0, "ASR":  99.5},
        "ASPL":       {"CR":  0.4, "1-TVR": 100.0, "ASR":  99.6},
        "XTransfer":  {"CR":  2.1, "1-TVR":  99.0, "ASR":  97.9},
        "Glaze":      {"CR":  1.7, "1-TVR":  99.4, "ASR":  98.6},
        "Nightshade": {"CR":  0.6, "1-TVR": 100.0, "ASR":  99.4},
        "MMCoA":      {"CR":  0.8, "1-TVR":  99.0, "ASR":  99.3},
    },
}

# ── Normalization range (unified across all VLMs) ────────────────────────────
AXES_LABELS = ["CR (Q1) ↑", "1−TVR (Q2) ↑", "ASR (Q3) ↑"]
N = 3
FLOOR = [94.0,   0.0,  93.0]
CEIL  = [100.0, 100.0, 100.5]

def normalise(val, floor, ceil):
    """Normalize value to [0, 1] range."""
    return max(0.0, min(1.0, (val - floor) / (ceil - floor)))

def denormalise(norm_val, floor, ceil):
    """Convert normalized value back to original scale."""
    return floor + norm_val * (ceil - floor)

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]


# ═══════════════════════════════════════════════════════════════════════════════
# Drawing functions (cb_vision enhanced)
# ═══════════════════════════════════════════════════════════════════════════════
def draw_radar(ax, vlm, show_labels=True):
    """Draw a single VLM radar chart on a polar ax (colorblind-friendly)."""
    vlm_data = PER_VLM_DATA[vlm]

    # Background fill — very light warm gray
    ax.fill(angles, [1.0] * (N + 1), color="#FAFAFA", alpha=0.6, zorder=0)

    # Grid circles — dotted, softer
    grid_levels = [0.25, 0.50, 0.75, 1.00]
    for level in grid_levels:
        ax.plot(angles, [level] * (N + 1),
                color="#999999", linewidth=0.7, linestyle=":",
                zorder=1, alpha=0.9)

    # Radial separator lines — clear, visible at 0°/120°/240° visual positions
    for ang in angles[:-1]:
        ax.plot([ang, ang], [0, 1.08],
                color="#888888", linewidth=1.0, linestyle="-",
                zorder=1.5, alpha=0.9)

    # Compute all raw values for annotation ranking
    all_raw = {}
    all_norm = {}
    for m in METHODS:
        d = vlm_data[m]
        crr     = 100.0 - d["CR"]
        tvr_inv = d["1-TVR"]
        asr     = d["ASR"]
        raw_vals = [crr, tvr_inv, asr]
        vals_n  = [normalise(v, FLOOR[i], CEIL[i])
                   for i, v in enumerate(raw_vals)]
        vals_n_closed = vals_n + vals_n[:1]
        all_raw[m] = raw_vals
        all_norm[m] = vals_n_closed

    # Method polygons with markers
    for m in METHODS:
        vals_n = all_norm[m]
        c  = METHOD_COLOR[m]
        lw = LINEWIDTHS[m]
        ms = MARKER_SIZE[m]
        fa = FILL_ALPHA[m]

        # Polygon line + markers
        ax.plot(angles, vals_n,
                color=c, linewidth=lw, linestyle="-",
                marker=MARKERS[m], markersize=ms,
                markerfacecolor=c, markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=ZORDER_LINE[m],
                solid_capstyle="round")
        ax.fill(angles, vals_n, color=c, alpha=fa, zorder=ZORDER_FILL[m])

    # No per-method value annotations — this figure benchmarks
    # existing methods on our dataset; no single method is "ours".

    # Axis labels — per-label pad via manual text placement
    # Label order: index 0 → CRR (Q1), 1 → 1−TVR (Q2), 2 → ASR (Q3)
    # r_offsets: larger = farther from circle edge (1.03 is the ylim)
    # ← modify these values to tune each label's distance independently
    LABEL_R_OFFSETS = [1.30, 1.15, 1.30]

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])  # hide default labels
    ax.set_ylim(0, 1.03)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_linewidth(1.0)
    ax.spines["polar"].set_color("#888888")

    # Rotation: Q1(CRR) lower-left, Q2(1-TVR) top, Q3(ASR) lower-right
    ax.set_theta_offset(7 * np.pi / 6)
    ax.set_theta_direction(-1)

    # Place labels manually after rotation is set
    label_fs = 13.5 if show_labels else 12.5
    for ang, lbl, r_off in zip(angles[:-1], AXES_LABELS, LABEL_R_OFFSETS):
        ax.text(ang, r_off, lbl,
                ha="center", va="center",
                fontsize=label_fs, fontfamily="serif", fontweight="bold",
                clip_on=False)

    # Subplot title
    ax.set_title(vlm,
                 fontsize=18, fontfamily="serif", fontweight="bold",
                 color="#333333", pad=50, loc="center")


def draw_legend_bottom(fig, ncol=6):
    """
    Shared legend at the bottom center of the figure.
    Colorblind-friendly: marker shape + color dual encoding.
    """
    method_handles = []
    for m in METHODS:
        handle = mlines.Line2D(
            [], [],
            color=METHOD_COLOR[m],
            linewidth=2.5,
            linestyle="-",
            marker=MARKERS[m],
            markersize=12,
            markerfacecolor=METHOD_COLOR[m],
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=m,
        )
        method_handles.append(handle)

    leg = fig.legend(
        handles=method_handles,
        ncol=ncol,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        fontsize=15,
        frameon=True, framealpha=0.95, edgecolor="#BBBBBB",
        fancybox=True,
        shadow=False,
        prop={"family": "serif", "size": 15},
        handlelength=3.5,
        handletextpad=1.0,
        columnspacing=2.5,
        borderpad=0.6,
    )
    leg.get_frame().set_linewidth(0.6)
    return leg


# ═══════════════════════════════════════════════════════════════════════════════
# Main figure: Qwen-VL / Gemini-3.0 / Kimi 2.5  (1×3 grid + bottom legend)
# ═══════════════════════════════════════════════════════════════════════════════
MAIN_VLMS = ["Qwen-VL", "Gemini-3.0", "GPT-5.2"]

fig_main, axes_main = plt.subplots(
    1, 3, figsize=(17, 7.0),
    subplot_kw=dict(projection="polar"),
    gridspec_kw=dict(wspace=0.48)
)
fig_main.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=-0.02)

for ax, vlm in zip(axes_main, MAIN_VLMS):
    draw_radar(ax, vlm, show_labels=True)

draw_legend_bottom(fig_main, ncol=6)

fig_main.text(0.50, 0.95,
              "Per-VLM Protection Quality  (CR ↑ / 1−TVR ↑ / ASR ↑)",
              ha="center", va="top",
              fontsize=20, fontfamily="serif", fontweight="bold",
              color="#222222")

for d, name in [(OUT_DIR, "figmm_fig7_radar_cb_vision"),
                (FIG_DIR, "fig3_modality_radar")]:
    fig_main.savefig(d / f"{name}.pdf", dpi=300, bbox_inches="tight")
    print(f"Saved → {d / name}.pdf")

plt.close(fig_main)


# ═══════════════════════════════════════════════════════════════════════════════
# Appendix figure: GPT-5.2 / GLM-4V  (1×2 grid + bottom legend, same style)
# ═══════════════════════════════════════════════════════════════════════════════
APPENDIX_VLMS = ["Kimi 2.5", "GLM-4V"]

fig_app, axes_app = plt.subplots(
    1, 2, figsize=(12, 7.0),
    subplot_kw=dict(projection="polar"),
    gridspec_kw=dict(wspace=0.48)
)
fig_app.subplots_adjust(left=0.05, right=0.95, top=0.86, bottom=0.10)

for ax, vlm in zip(axes_app, APPENDIX_VLMS):
    draw_radar(ax, vlm, show_labels=True)

draw_legend_bottom(fig_app, ncol=6)

fig_app.text(0.50, 0.975,
             "Per-VLM Protection Quality — Appendix  (Kimi 2.5 / GLM-4V)",
             ha="center", va="top",
             fontsize=20, fontfamily="serif", fontweight="bold",
             color="#222222")

for d, name in [(OUT_DIR, "figmm_fig7_radar_appendix_cb_vision"),
                (FIG_DIR, "fig7_radar_appendix")]:
    fig_app.savefig(d / f"{name}.pdf", dpi=300, bbox_inches="tight")
    print(f"Saved → {d / name}.pdf")

plt.close(fig_app)

print("\n✅ All cb_vision radar charts generated successfully.")
