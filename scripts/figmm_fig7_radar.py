"""
figmm_fig7_radar.py
===================
ACM MM 2026 — CaptchaShield Fig.7  (v7 2026-03-30)

主图：Qwen-VL / Gemini-3.0 / Kimi 2.5  (3列)
附录：GPT-5.2 / GLM-4V                  (2列)

修复清单（v7）：
  1. 总标题与子图标题不重叠 → top=0.80, suptitle y=1.01
  2. 旋转公式修正 → 1-TVR (index=1) 朝正上方
     set_theta_offset = π/2 + 2π/N，方向逆时针
  3. 轴标签不碰撞 → pad 减小至 8，wspace 加大，figsize 加宽
  4. 图例居中底部 → 用 fig.add_axes 单独放图例行
  5. Gemini 1-TVR 独立归一化 → FLOOR_GEMINI[1]=0, CEIL_GEMINI[1]=10
  6. 去掉 MMCoA 数值标注（Gemini 图圆心区域太乱）
  7. 去掉网格百分比标签（避免遮挡）
  8. 填充 alpha 降低至 0.06，仅 MMCoA 保留 0.12
  9. 轴端标注实际数值范围（r=1.10 处）
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

# ── 颜色（与全论文一致）──────────────────────────────────────────────────────
METHOD_COLOR = {
    "AMP":        "#0D47A1",
    "ASPL":       "#1976D2",
    "XTransfer":  "#42A5F5",
    "Glaze":      "#90CAF9",
    "Nightshade": "#E64A19",
    "MMCoA":      "#2E7D32",
}
LINEWIDTHS = {m: (2.2 if m == "MMCoA" else 1.4) for m in METHOD_COLOR}
FILL_ALPHA  = {m: (0.12 if m == "MMCoA" else 0.06) for m in METHOD_COLOR}
METHODS = ["AMP", "ASPL", "XTransfer", "Glaze", "Nightshade", "MMCoA"]

# ── 数据（SD-ID, adv_metrics）────────────────────────────────────────────────
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

# ── 归一化范围 ────────────────────────────────────────────────────────────────
# 轴顺序：[CRR, 1-TVR, ASR]
# CRR = 100 - CR
AXES_LABELS = ["CRR (Q1) ↑", "1−TVR (Q2) ↑", "ASR (Q3) ↑"]
N = 3

# 通用范围（非Gemini VLM）
FLOOR_STD = [94.0,   0.0,  93.0]
CEIL_STD  = [100.0, 100.0, 100.5]

# Gemini 专用范围：1-TVR 轴缩放至 0–10%，显示异常
FLOOR_GEM = [94.0,  0.0,  93.0]
CEIL_GEM  = [100.0, 10.0, 100.5]

def normalise(val, floor, ceil):
    return max(0.0, min(1.0, (val - floor) / (ceil - floor)))

# 角度：逆时针，1-TVR(index=1)在顶部
# set_theta_offset = π/2 + 2π/N，set_theta_direction = +1（逆时针）
angles_base = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles_base += angles_base[:1]

THETA_OFFSET = np.pi / 2 + 2 * np.pi / N   # 让 index=1 朝上


# ═══════════════════════════════════════════════════════════════════════════════
def draw_radar(ax, vlm):
    """Draw single VLM radar on a polar ax."""
    is_gemini = (vlm == "Gemini-3.0")
    FLOOR = FLOOR_GEM if is_gemini else FLOOR_STD
    CEIL  = CEIL_GEM  if is_gemini else CEIL_STD
    vlm_data = PER_VLM_DATA[vlm]

    # 旋转设置（先设置，之后绘制才生效）
    ax.set_theta_offset(THETA_OFFSET)
    ax.set_theta_direction(1)   # 逆时针

    # 背景
    ax.fill(angles_base, [1.0] * (N + 1),
            color="#F5F5F5", alpha=0.55, zorder=0)

    # 网格圆圈（虚线，浅色）
    for level in [0.25, 0.50, 0.75, 1.00]:
        ax.plot(angles_base, [level] * (N + 1),
                color="#CCCCCC", linewidth=0.7, linestyle="--",
                zorder=1, alpha=0.80)

    # 径向分隔线
    for ang in angles_base[:-1]:
        ax.plot([ang, ang], [0, 1],
                color="#DDDDDD", linewidth=0.6,
                zorder=1, alpha=0.7)

    # 方法多边形（纯线条，无marker）
    for m in METHODS:
        d = vlm_data[m]
        crr     = 100.0 - d["CR"]
        tvr_inv = d["1-TVR"]
        asr     = d["ASR"]
        vals_n  = [normalise(v, FLOOR[i], CEIL[i])
                   for i, v in enumerate([crr, tvr_inv, asr])]
        vals_n += vals_n[:1]
        c  = METHOD_COLOR[m]
        lw = LINEWIDTHS[m]
        fa = FILL_ALPHA[m]

        ax.plot(angles_base, vals_n,
                color=c, linewidth=lw, linestyle="-", zorder=4)
        ax.fill(angles_base, vals_n, color=c, alpha=fa, zorder=3)

    # 轴设置
    ax.set_xticks(angles_base[:-1])
    ax.set_xticklabels(AXES_LABELS,
                       fontsize=9.0, fontfamily="serif", fontweight="bold")
    ax.tick_params(pad=8)
    ax.set_ylim(0, 1.10)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_linewidth(1.0)
    ax.spines["polar"].set_color("#BBBBBB")

    # 轴端实际数值标注（r=1.12处，每条轴）
    axis_ranges = list(zip(FLOOR, CEIL))
    axis_raw_labels = [
        f"{FLOOR[0]:.0f}–{CEIL[0]:.0f}%",
        f"{FLOOR[1]:.0f}–{CEIL[1]:.0f}%",
        f"{FLOOR[2]:.0f}–{CEIL[2]:.0f}%",
    ]
    for i, ang in enumerate(angles_base[:-1]):
        ax.text(ang, 1.22, axis_raw_labels[i],
                fontsize=6.0, fontfamily="serif",
                color="#888888", ha="center", va="center",
                zorder=6)

    # 标题
    title_color = "#C62828" if is_gemini else "#1A1A2E"
    gem_note    = "\n(1−TVR axis: 0–10%)" if is_gemini else ""
    ax.set_title(f"{vlm}{gem_note}",
                 fontsize=10.0, fontfamily="serif", fontweight="bold",
                 color=title_color, pad=20, loc="center")


def make_legend_handles():
    """6方法纯线条 handles（无marker）"""
    handles = []
    for m in METHODS:
        lbl = m + " (Ours)" if m == "MMCoA" else m
        h = mlines.Line2D([], [],
                          color=METHOD_COLOR[m],
                          linewidth=2.2 if m == "MMCoA" else 1.6,
                          linestyle="-",
                          label=lbl)
        handles.append(h)
    return handles


# ═══════════════════════════════════════════════════════════════════════════════
# 通用绘图函数：给定 VLM 列表，生成整张 figure
# ═══════════════════════════════════════════════════════════════════════════════
def make_figure(vlm_list, title_str, out_names):
    n = len(vlm_list)
    fig_w = 5.2 * n + 2.5   # 每图约5.2"，加图例行空间

    fig = plt.figure(figsize=(fig_w, 5.8))

    # 雷达图区域：上方 75%
    # 图例行：下方 12%
    radar_bottom = 0.18
    radar_top    = 0.84
    radar_height = radar_top - radar_bottom
    radar_width  = 0.86 / n   # 各子图宽度，留左右各 0.07

    axes = []
    for i, vlm in enumerate(vlm_list):
        left = 0.07 + i * (0.86 / n)
        ax = fig.add_axes(
            [left, radar_bottom, radar_width * 0.88, radar_height],
            polar=True
        )
        draw_radar(ax, vlm)
        axes.append(ax)

    # 图例：单行居中，放在底部独立区域
    leg_ax = fig.add_axes([0.10, 0.01, 0.80, 0.12])
    leg_ax.set_axis_off()
    handles = make_legend_handles()
    leg = leg_ax.legend(
        handles=handles,
        ncol=len(METHODS),
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        fontsize=9.0,
        frameon=True, framealpha=0.93, edgecolor="#AAAAAA",
        prop={"family": "serif", "size": 9.0},
        handlelength=2.0,
        handletextpad=0.5,
        columnspacing=1.2,
        borderpad=0.7,
        title="Method",
        title_fontsize=9.5,
    )
    leg.get_title().set_fontweight("bold")

    # 总标题
    fig.text(0.50, 0.97,
             title_str,
             ha="center", va="top",
             fontsize=12.0, fontfamily="serif", fontweight="bold",
             color="#1A1A2E")

    for d, name in out_names:
        fig.savefig(d / f"{name}.pdf", dpi=300, bbox_inches="tight")
        print(f"Saved → {d / name}.pdf")

    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 主图：Qwen-VL / Gemini-3.0 / Kimi 2.5
# ═══════════════════════════════════════════════════════════════════════════════
make_figure(
    vlm_list=["Qwen-VL", "Gemini-3.0", "Kimi 2.5"],
    title_str="Per-VLM Protection Quality  (CRR ↑ / 1−TVR ↑ / ASR ↑)",
    out_names=[
        (OUT_DIR, "figmm_fig7_radar"),
        (FIG_DIR, "fig7_radar"),
    ]
)

# ═══════════════════════════════════════════════════════════════════════════════
# 附录：GPT-5.2 / GLM-4V
# ═══════════════════════════════════════════════════════════════════════════════
make_figure(
    vlm_list=["GPT-5.2", "GLM-4V"],
    title_str="Per-VLM Protection Quality — Appendix  (GPT-5.2 / GLM-4V)",
    out_names=[
        (OUT_DIR, "figmm_fig7_radar_appendix"),
        (FIG_DIR, "fig7_radar_appendix"),
    ]
)
