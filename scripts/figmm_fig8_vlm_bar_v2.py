"""
figmm_fig8_vlm_bar_v2.py  (renamed from figmm_fig4_vlm_bar_v2.py) — ACM MM 2026
Outputs: figures/fig8_vlm_bar_ID.pdf, figures/fig8_vlm_bar_SDXL.pdf
Single-panel bar chart for ID-based / SDXL Targets
5 VLM: Qwen-VL / Kimi 2.5 / GPT-5.2 / Gemini-3.0 / GLM-4V
Legend placed below the main title for better readability.
Generates two PDFs: one for ID-based, one for SDXL.
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

# ── 5VLM color palette (Tableau10) ───────────────────────────────────────────
vlm_labels = ["Qwen-VL", "Kimi 2.5", "GPT-5.2", "Gemini-3.0", "GLM-4V"]
vlm_colors = ["#4E79A7",   "#59A14F",  "#F28E2B",  "#E15759",   "#76B7B2"]

methods_display = ["No Attack", "AMP",  "ASPL",  "Glaze",    "MMCoA", "Nightshade", "XTransfer"]
methods_key     = ["Baseline",  "AMP",  "ASPL",  "Glaze(MI)", "MMCoA", "Nightshade", "XTransfer"]

# ── Real data (ID-based ASR) ─────────────────────────────────────────────────
# Order: [Qwen, Kimi, GPT-5.2, Gemini, GLM-4V]
idbase_asr = {
    "Baseline":   [100.0, 99.9,  100.0, 99.4,  100.0],
    "AMP":        [99.7,  99.7,  99.9,  98.8,  99.5 ],
    "ASPL":       [99.0,  99.0,  99.8,  98.8,  99.6 ],
    "Glaze(MI)":  [94.5,  98.0,  99.7,  95.5,  98.6 ],
    "MMCoA":      [98.0,  98.6,  99.9,  99.4,  99.3 ],
    "Nightshade": [99.2,  99.7,  100.0, 98.2,  99.4 ],
    "XTransfer":  [95.9,  98.7,  99.7,  98.1,  97.9 ],
}

# ── Real data (SDXL ASR) ─────────────────────────────────────────────────────
# Order: [Qwen, Kimi, GPT-5.2, Gemini, GLM-4V]
sdxl_asr = {
    "Baseline":   [100.0, 99.9,  100.0, 99.4,  100.0],
    "AMP":        [99.1,  99.4,  99.7,  97.6,  99.4 ],
    "ASPL":       [98.8,  99.8,  99.7,  97.0,  99.6 ],
    "Glaze(MI)":  [97.5,  99.6,  99.8,  97.3,  99.6 ],
    "MMCoA":      [97.7,  98.8,  99.6,  97.6,  99.0 ],
    "Nightshade": [98.6,  99.9,  99.7,  97.0,  99.6 ],
    "XTransfer":  [97.2,  99.3,  99.8,  97.8,  99.1 ],
}

def wilson_ci(p_pct, n=1000):
    """Wilson score interval, returns (upper_err, lower_err)"""
    p = p_pct / 100.0
    z = 1.96
    denom = 1 + z**2 / n
    ctr   = (p + z**2 / (2 * n)) / denom
    half  = z * (p * (1 - p) / n + z**2 / (4 * n**2))**0.5 / denom
    upper = min(100.0, (ctr + half) * 100) - p_pct
    lower = p_pct - max(0.0, (ctr - half) * 100)
    return upper, lower


def draw_chart(data_dict, main_title, res_name, fig_name):
    """Draw a single-panel bar chart and save to PDF."""
    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    plt.subplots_adjust(left=0.08, right=0.97,
                        bottom=0.10, top=0.90)

    n_methods   = len(methods_key)
    n_vlms      = len(vlm_labels)          # 5
    total_width = 0.92
    bar_w       = total_width / n_vlms
    x_base      = np.arange(n_methods)
    Y_LO, Y_HI = 93.0, 101.0

    for vi, (vlabel, vc) in enumerate(zip(vlm_labels, vlm_colors)):
        offset = (vi - (n_vlms - 1) / 2) * bar_w
        for xi, mk in enumerate(methods_key):
            v  = data_dict[mk][vi]
            bx = x_base[xi] + offset

            if mk == "Baseline":
                ax.bar(bx, 0.35, bottom=Y_LO, width=bar_w * 0.82,
                       color=vc, alpha=0.40, zorder=3)
            else:
                eu, ed = wilson_ci(v)
                ax.bar(bx, max(0.0, v - Y_LO), bottom=Y_LO,
                       width=bar_w * 0.82, color=vc, alpha=0.85, zorder=3,
                       yerr=[[ed], [eu]],
                       error_kw=dict(elinewidth=1.2, capsize=1.5,
                                     ecolor="#333333", alpha=0.65))
                # Value annotation (centered on bar, no tilt)
                ax.text(bx, v + eu + 0.08, f"{v:.1f}",
                        ha="center", va="bottom",
                        fontsize=7.5, fontfamily="serif", color="#333333",
                        rotation=0, clip_on=False)

    # ── Baseline annotation ──────────────────────────────────────────────────
    bi = methods_key.index("Baseline")
    ax.annotate("≈0 %",
        xy=(x_base[bi], Y_LO + 0.12),
        xytext=(x_base[bi], Y_LO + 1.40),
        ha="center", va="bottom",
        fontsize=9.0, fontfamily="serif", color="#555555",
        arrowprops=dict(arrowstyle="-|>", color="#777777",
                        lw=0.9, mutation_scale=9), zorder=8)

    # ── Separator line after Baseline ────────────────────────────────────────
    ax.axvline(x_base[0] + 0.52, color="gray", linestyle="--",
               linewidth=1.0, alpha=0.55, zorder=2)

    # ── Axes formatting ──────────────────────────────────────────────────────
    ax.set_xlim(x_base[0] - 0.58, x_base[-1] + 0.58)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_xticks(x_base)
    ax.set_xticklabels(methods_display, fontsize=11,
                       fontfamily="serif", rotation=0, ha="center")
    ax.set_yticks(np.arange(94, 102, 1))
    for child in ax.get_children():
        child.set_clip_on(False)
    ax.tick_params(axis="y", labelsize=10.5)
    ax.set_ylabel("ASR (%)", fontsize=12, fontfamily="serif")
    ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.38, color="gray")
    ax.set_axisbelow(True)

    xlim = ax.get_xlim()
    ax.text(xlim[0] - 0.04, Y_LO + 0.15, "axis\nbreak↑",
            ha="right", va="bottom", fontsize=6.0, color="gray",
            fontfamily="serif")

    # ── Main title ───────────────────────────────────────────────────────────
    fig.text(0.52, 0.96, main_title,
             ha="center", va="top",
             fontsize=14, fontfamily="serif", fontweight="bold")

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [mpatches.Patch(facecolor=vc, alpha=0.85, label=vl)
               for vl, vc in zip(vlm_labels, vlm_colors)]
    fig.legend(handles=handles,
               fontsize=18,
               loc="center",
               bbox_to_anchor=(0.52, 0.880),
               ncol=5,
               framealpha=0.92,
               edgecolor="#CCCCCC",
               prop={"family": "serif"})

    # ── Save ─────────────────────────────────────────────────────────────────
    HERE    = Path(__file__).parent
    RES_DIR = HERE.parent.parent / "results"
    FIG_DIR = HERE.parent / "figures"
    RES_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for d, name in [(RES_DIR, res_name), (FIG_DIR, fig_name)]:
        fig.savefig(d / f"{name}.pdf", dpi=300, bbox_inches="tight")
        print(f"Saved -> {d / name}.pdf")

    plt.close(fig)


# ── Generate both charts ─────────────────────────────────────────────────────
draw_chart(
    data_dict  = idbase_asr,
    main_title = "Attack Success Rate (ASR) Across VLMs and Attack Methods — ID",
    res_name   = "figmm_fig4_vlm_bar_v2_ID",
    fig_name   = "fig8_vlm_bar_ID",
)

draw_chart(
    data_dict  = sdxl_asr,
    main_title = "Attack Success Rate (ASR) Across VLMs and Attack Methods — SD",
    res_name   = "figmm_fig4_vlm_bar_v2_SDXL",
    fig_name   = "fig8_vlm_bar_SDXL",
)
