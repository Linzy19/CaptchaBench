"""
figmm_fig5_vlm_bar.py  (renamed from figmm_fig4_vlm_bar.py) — ACM MM 2026
Output: figures/fig5_vlm_bar.pdf
双panel（ID-based 左 | SDXL 右），5VLM：Qwen-VL / Kimi 2.5 / GPT-5.2 / Gemini-3.0 / GLM-4V
figsize=(14, 5.5)，Y轴93-101%，字体放大，数值标注清晰
色盘：Tableau10推荐（蓝/绿/橙/红/青），避免遮挡
数据来源：真实实验数据，直接硬编码
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

# ── 5VLM配色：Tableau10推荐款式（蓝/绿/橙/红/青） ──────────────────────────
vlm_labels = ["Qwen-VL", "Kimi 2.5", "GPT-5.2", "Gemini-3.0", "GLM-4V"]
vlm_colors = ["#4E79A7",   "#59A14F",  "#F28E2B",  "#E15759",   "#76B7B2"]

methods_display = ["No Attack", "AMP",  "ASPL",  "Glaze",    "MMCoA", "Nightshade", "XTransfer"]
methods_key     = ["Baseline",  "AMP",  "ASPL",  "Glaze(MI)", "MMCoA", "Nightshade", "XTransfer"]

# ── 真实数据（Q3/ASR，SD-ID adv_metrics）────────────────────────────────────
# 顺序：[Qwen, Kimi, GPT-5.2, Gemini, GLM-4V]
idbase_asr = {
    "Baseline":   [100.0, 99.9,  100.0, 99.4,  100.0],
    "AMP":        [99.7,  99.7,  99.9,  98.8,  99.5 ],
    "ASPL":       [99.0,  99.0,  99.8,  98.8,  99.6 ],
    "Glaze(MI)":  [94.5,  98.0,  99.7,  95.5,  98.6 ],
    "MMCoA":      [98.0,  98.6,  99.9,  99.4,  99.3 ],
    "Nightshade": [99.2,  99.7,  100.0, 98.2,  99.4 ],
    "XTransfer":  [95.9,  98.7,  99.7,  98.1,  97.9 ],
}

# ── 真实数据（SDXL adv_metrics）───────────────────────────────────────────────
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
    """Wilson score interval，返回 (upper_err, lower_err)"""
    p = p_pct / 100.0
    z = 1.96
    denom = 1 + z**2 / n
    ctr   = (p + z**2 / (2 * n)) / denom
    half  = z * (p * (1 - p) / n + z**2 / (4 * n**2))**0.5 / denom
    upper = min(100.0, (ctr + half) * 100) - p_pct
    lower = p_pct - max(0.0, (ctr - half) * 100)
    return upper, lower

# ── 布局 ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18.0, 8.5),
                         sharey=False, constrained_layout=False)
# 布局：子图区域 top=0.78，图例紧贴其上 bbox=0.86，大标题 y=0.97
plt.subplots_adjust(left=0.06, right=0.98,
                    bottom=0.11, top=0.82, wspace=0.10)

n_methods   = len(methods_key)
n_vlms      = len(vlm_labels)          # 5
total_width = 0.92                     # 扩大bar组宽度
bar_w       = total_width / n_vlms
x_base      = np.arange(n_methods)
Y_LO, Y_HI = 93.0, 102.5              # Y轴上限留空间给横排数字

def draw_panel(ax, data_dict, title):
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
                # 所有方法所有VLM都标注数值，横放
                ax.text(bx, v + eu + 0.08, f"{v:.1f}",
                        ha="center", va="bottom",
                        fontsize=4.8, fontfamily="serif", color="#333333",
                        rotation=0)

    bi = methods_key.index("Baseline")
    ax.annotate("≈0 %",
        xy=(x_base[bi], Y_LO + 0.12),
        xytext=(x_base[bi], Y_LO + 1.40),
        ha="center", va="bottom",
        fontsize=8.0, fontfamily="serif", color="#555555",
        arrowprops=dict(arrowstyle="-|>", color="#777777",
                        lw=0.9, mutation_scale=9), zorder=8)

    ax.axvline(x_base[0] + 0.52, color="gray", linestyle="--",
               linewidth=1.0, alpha=0.55, zorder=2)

    ax.set_xlim(x_base[0] - 0.58, x_base[-1] + 0.58)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_xticks(x_base)
    ax.set_xticklabels(methods_display, fontsize=9.5,
                       fontfamily="serif", rotation=0, ha="center")
    ax.set_yticks(np.arange(94, 102, 1))
    ax.tick_params(axis="y", labelsize=9.5)
    ax.set_ylabel("ASR (%)", fontsize=11, fontfamily="serif")
    ax.set_title(title, fontsize=12, fontfamily="serif",
                 fontweight="bold", pad=4)
    ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.38, color="gray")
    ax.set_axisbelow(True)

    xlim = ax.get_xlim()
    # ax.text(xlim[0] - 0.04, Y_LO + 0.15, "axis\nbreak↑",
    #         ha="right", va="bottom", fontsize=5.5, color="gray",
    #         fontfamily="serif")

draw_panel(axes[0], idbase_asr, "(A) ID Targets")
draw_panel(axes[1], sdxl_asr,   "(B) SD Targets")

# ── 共享图例：在 subplots 上方居中，ncol=5 一行，紧贴图区顶部 ──────────────
handles = [mpatches.Patch(facecolor=vc, alpha=0.85, label=vl)
           for vl, vc in zip(vlm_labels, vlm_colors)]
fig.legend(handles=handles,
           title="VLM Evaluator", title_fontsize=9.5,
           fontsize=9.5,
           loc="upper center",
           bbox_to_anchor=(0.50, 0.89),
           ncol=5,
           framealpha=0.92,
           edgecolor="#CCCCCC",
           prop={"family": "serif"})

fig.suptitle("Attack Success Rate (ASR) Across VLMs and Attack Methods",
             fontsize=12, fontfamily="serif", fontweight="bold", y=0.92)

# ── 保存 ──────────────────────────────────────────────────────────────────────
HERE    = Path(__file__).parent
RES_DIR = HERE.parent.parent / "results"
FIG_DIR = HERE.parent / "figures"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

for d, name in [(RES_DIR, "figmm_fig4_vlm_bar"), (FIG_DIR, "fig5_vlm_bar")]:
    fig.savefig(d / f"{name}.pdf", dpi=300, bbox_inches="tight")
    print(f"Saved -> {d / name}.pdf")

plt.close(fig)
