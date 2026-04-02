"""
figmm_fig6_stroke.py — ACM MM 2026  (v3 2026-03-27)
双panel（折线图 左 | 热力图 右），仅SD-ID数据

Panel A色系：
  Image-only 4方法：4种明显可区分的颜色（不再全蓝）
    AMP:       #0D47A1 (深蓝)
    ASPL:      #1976D2 (中蓝)
    XTransfer: #42A5F5 (浅蓝)
    Glaze:     #90CAF9 (淡蓝)
  Text-only:   #E64A19 (深橙)
  Image+Text:  #2E7D32 (深绿)
  → 色系与Fig3完全一致

Panel B热力图：
  改用 YlGn（黄→绿）单色渐变，vmin=92 扩大范围让色差明显
  → 避免 RdYlGn 全绿问题，YlGn 在92-100.5范围内色差丰富

图例：Panel A 单个图例（方法线条），Panel B 右侧colorbar
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

# ── 每方法独立色（与Fig3/Fig5-A一致的模态逻辑）──────────────────────────────
METHOD_COLOR = {
    "AMP":        "#0D47A1",   # deep blue   (Image-only)
    "ASPL":       "#1976D2",   # mid blue    (Image-only)
    "XTransfer":  "#42A5F5",   # light blue  (Image-only)
    "Glaze":      "#90CAF9",   # pale blue   (Image-only)
    "Nightshade": "#E64A19",   # deep orange (Text-only)
    "MMCoA":      "#2E7D32",   # dark green  (Image+Text)
}

MODALITY_COLOR = {
    "Image-only":  "#1976D2",
    "Text-only":   "#E64A19",
    "Image+Text":  "#2E7D32",
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

# ── 真实笔画数据（SD-ID）─────────────────────────────────────────────────────
bins = ["1–5", "6–10", "11–15", "16–20", "21+"]
stroke_data = {
    "ASPL":       [98.2, 99.4, 99.4, 100.0,  95.2],
    "Glaze":      [95.3, 97.9, 97.9,  99.3,  95.2],
    "AMP":        [99.3, 99.4, 99.3, 100.0, 100.0],
    "XTransfer":  [98.6, 98.6, 98.4,  98.6,  95.2],
    "Nightshade": [98.2, 99.1, 98.9,  98.6, 100.0],
    "MMCoA":      [99.3, 99.7, 99.3, 100.0, 100.0],
}

# 热力图方法顺序：Image-only(4) → Text-only(1) → Image+Text(1)
METHOD_ORDER = ["AMP", "ASPL", "XTransfer", "Glaze", "Nightshade", "MMCoA"]
METHODS      = ["ASPL", "Glaze", "AMP", "XTransfer", "Nightshade", "MMCoA"]

# ── 图形布局 ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15.0, 6.5))
gs  = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1.0],
                        left=0.07, right=0.97,
                        top=0.86, bottom=0.12, wspace=0.36)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — 折线图
# ══════════════════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[0])
x = np.arange(len(bins))

# Image-only 方法填充带（浅蓝背景，标注分组范围）
img_methods = [m for m in METHODS if MODALITY[m] == "Image-only"]
mod_min = [min(stroke_data[m][i] for m in img_methods) for i in range(len(bins))]
mod_max = [max(stroke_data[m][i] for m in img_methods) for i in range(len(bins))]
ax_a.fill_between(x, mod_min, mod_max, alpha=0.08, color="#1976D2", zorder=1,
                  label="_nolegend_")

# 折线（每方法独立颜色）
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

# Panel A 图例：方法颜色+线型+标记，单一图例，ncol=2，右上角
method_lines = [
    plt.Line2D([0], [0], color=METHOD_COLOR[m], linestyle=LINESTYLES[m],
               marker=MARKERS[m], markersize=6, linewidth=1.8, label=m)
    for m in METHODS
]
ax_a.legend(handles=method_lines,
            title="Method", title_fontsize=9.0,
            fontsize=8.8, loc="lower right",
            framealpha=0.92, edgecolor="#CCCCCC",
            prop={"family": "serif"}, ncol=2,
            handlelength=1.8)
ax_a.get_legend().get_title().set_fontfamily("serif")
ax_a.get_legend().get_title().set_fontweight("bold")

# 在图例外侧加模态颜色标注（右下角图例上方）
# ax_a.annotate("── Image-only  ── Text-only  ── Img+Txt",
#               xy=(0.99, 0.01), xycoords="axes fraction",
#               ha="right", va="bottom",
#               fontsize=7.5, fontfamily="serif", color="#555555",
#               style="italic")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — 热力图：YlOrRd 反向（高值浅蓝/白，低值深橙）→ 改用 YlGnBu
# 目标：凸显每列（笔画区间）内各方法的ASR分布差异
# 色系：BuPu（浅→深紫蓝），美观学术，低值用暖橙区分
# ══════════════════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[1])
ax_b.spines["top"].set_visible(True)
ax_b.spines["right"].set_visible(True)

heatmap_data = np.array([stroke_data[m] for m in METHOD_ORDER])

import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 自定义颜色：低值(95)→暖橙，中值(97.5)→浅米，高值(100.5)→深靛蓝
# 用 LinearSegmentedColormap 创建美观渐变
from matplotlib.colors import LinearSegmentedColormap
cmap_custom = LinearSegmentedColormap.from_list(
    "asr_cmap",
    [
        (0.00, "#F0C080"),   # 95.0 → 浅暖橙（低值）
        (0.25, "#FAF3E0"),   # 97.4 → 极浅米白（过渡）
        (0.50, "#DDEEFF"),   # 98.8 → 极浅蓝
        (0.75, "#7BAFD4"),   # 99.6 → 中蓝
        (1.00, "#2C5F8A"),   # 100.5 → 深蓝（高值）
    ],
    N=256
)
VMIN, VMAX = 95.0, 100.5

im = ax_b.imshow(heatmap_data, aspect="auto", cmap=cmap_custom,
                 vmin=VMIN, vmax=VMAX, origin="upper", alpha=0.92)

# 白色格线
for r in range(len(METHOD_ORDER) + 1):
    ax_b.axhline(r - 0.5, color="white", linewidth=2.0, zorder=3)
for c in range(len(bins) + 1):
    ax_b.axvline(c - 0.5, color="white", linewidth=2.0, zorder=3)

# 格内数值标注
for r in range(len(METHOD_ORDER)):
    for c_idx in range(len(bins)):
        val = heatmap_data[r, c_idx]
        # 深色背景（高值）用白字，浅色背景（低/中值）用深字
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

# Colorbar（右侧）
divider = make_axes_locatable(ax_b)
cax = divider.append_axes("right", size="4%", pad=0.08)
cb = plt.colorbar(im, cax=cax)
cb.set_label("ASR (%)", fontsize=9, fontfamily="serif")
cb.ax.tick_params(labelsize=8)
cb.set_ticks([95, 96, 97, 98, 99, 100])
# 标注语义区间
# cb.ax.axhline(97.0, color="#C62828", linewidth=1.0, linestyle="--", alpha=0.7)
# cb.ax.axhline(99.5, color="#2E7D32", linewidth=1.0, linestyle="--", alpha=0.7)
# cb.ax.text(1.15, 97.0, "97%", transform=cb.ax.get_yaxis_transform(),
#            fontsize=7, color="#C62828", va="center")
# cb.ax.text(1.15, 99.5, "99.5%", transform=cb.ax.get_yaxis_transform(),
#            fontsize=7, color="#2E7D32", va="center")
for tick_lbl in cb.ax.get_yticklabels():
    tick_lbl.set_fontfamily("serif")

# 大标题放在两图之间正上方
# Panel A 右边缘 ≈ 0.54，两图中点 x ≈ 0.545
fig.text(0.545, 0.97,
         "Stroke Complexity Analysis by Input Modality",
         ha="center", va="top",
         fontsize=13, fontfamily="serif", fontweight="bold",
         color="#1A1A2E")

# ── 保存 ──────────────────────────────────────────────────────────────────────
HERE    = Path(__file__).parent
RES_DIR = HERE.parent.parent / "results"
FIG_DIR = HERE.parent / "figures"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

for d, name in [(RES_DIR, "figmm_fig6_stroke"), (FIG_DIR, "fig6_stroke_analysis")]:
    fig.savefig(d / f"{name}.pdf", dpi=300, bbox_inches="tight")
    print(f"Saved -> {d / name}.pdf")

plt.close(fig)
